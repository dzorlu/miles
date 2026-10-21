"""Muse Spark through the generic native-LoRA tables: attention q/gate/k/v + o, routed experts: no GPU.

The adapted forward of every adapted linear must equal the frozen linear run with the merged weight
(``merge_lora_into_weights``), and the q/gate/k/v adapters must sit in the checkpoint's frame, so the
merged fused weight converts to ``base + scale * B @ A`` for each HF projection.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from miles.backends.megatron_utils.lora.native.config import LoRAConfig
from miles.backends.megatron_utils.lora.native.hf_adapter import export_lora_hf_named
from miles.backends.megatron_utils.lora.native.lora import _attach_layer
from miles.backends.megatron_utils.lora.native.merge import merge_lora_into_weights
from miles.backends.megatron_utils.lora.native.modules.moe import _grouped_linear
from miles.backends.megatron_utils.lora.native.registry import MODEL_SPECS, resolve_config_spec
from miles.backends.megatron_utils.lora.native.spec.attention import build_muse_spark_qgkv_permutation
from miles.backends.megatron_utils.lora.native.spec.base import AttachContext
from miles_plugins.models.muse_spark.weight_mapping import MuseSpark1xParamMapping

HIDDEN, LATENT, RANK, ALPHA = 8, 6, 2, 6
HEADS, GROUPS, HEAD_DIM = 4, 2, 4
NUM_EXPERTS, MOE_INTERMEDIATE = 3, 5
MUSE = MODEL_SPECS["muse_spark"]


class _Linear(nn.Module):
    """A TE column/row-parallel linear at TP 1: forward returns (output, bias)."""

    def __init__(self, out_features, in_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        return x @ self.weight.t(), None


class _GroupedLinear(nn.Module):
    """A TE grouped expert linear with weight0..weight{E-1}."""

    def __init__(self, num_experts, out_features, in_features):
        super().__init__()
        for index in range(num_experts):
            self.register_parameter(f"weight{index}", nn.Parameter(torch.randn(out_features, in_features)))

    def forward(self, x, tokens_per_expert):
        weights = torch.stack([getattr(self, f"weight{i}") for i in range(len(tokens_per_expert))])
        return _grouped_linear(x, weights, tokens_per_expert), None


def _attention(output_gate):
    attention = nn.Module()
    q_rows = HEADS * HEAD_DIM
    attention.linear_qkv = _Linear(q_rows * (2 if output_gate else 1) + 2 * GROUPS * HEAD_DIM, HIDDEN)
    attention.linear_proj = _Linear(HIDDEN, q_rows)
    attention.num_attention_heads_per_partition = HEADS
    attention.num_query_groups_per_partition = GROUPS
    attention.hidden_size_per_attention_head = HEAD_DIM
    return attention


def _moe():
    """Muse Spark's MoE block: latent experts (fc1 reads the latent, fc2 writes it), no shared experts."""
    moe = nn.Module()
    moe.config = SimpleNamespace(moe_ffn_hidden_size=MOE_INTERMEDIATE, moe_latent_size=LATENT)
    moe.experts = nn.Module()
    moe.experts.num_local_experts = NUM_EXPERTS
    moe.experts.linear_fc1 = _GroupedLinear(NUM_EXPERTS, 2 * MOE_INTERMEDIATE, LATENT)
    moe.experts.linear_fc2 = _GroupedLinear(NUM_EXPERTS, LATENT, MOE_INTERMEDIATE)
    moe.fc1_latent_proj = _Linear(LATENT, HIDDEN)
    moe.fc2_latent_proj = _Linear(HIDDEN, LATENT)
    moe.shared_experts = None
    return moe


def _layer(output_gate, layer_number=4):
    layer = nn.Module()
    layer.layer_number = layer_number
    layer.self_attention = _attention(output_gate)
    layer.mlp = _moe()
    return layer


def _context(output_gate):
    return AttachContext(
        lora=LoRAConfig(rank=RANK, alpha=ALPHA, dropout=0.0, target_modules=None),
        transformer_config=SimpleNamespace(
            hidden_size=HIDDEN,
            sequence_parallel=False,
            attention_output_gate=output_gate,
            gated_linear_unit=True,
            num_query_groups=GROUPS,
        ),
        tp_size=1,
        tp_rank=0,
        naming=MUSE.declared_naming,
    )


@pytest.fixture(autouse=True)
def _expert_parallel_rank(monkeypatch):
    from megatron.core import parallel_state

    monkeypatch.setattr(parallel_state, "get_expert_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(parallel_state, "get_expert_model_parallel_world_size", lambda: 2)


def _adapted_layer(output_gate):
    layer = _layer(output_gate)
    context = _context(output_gate)
    MUSE.validate(context)
    wrapped = _attach_layer(layer, MUSE, context)
    assert wrapped == 4  # qkv, o, experts fc1, experts fc2
    torch.manual_seed(1)
    with torch.no_grad():
        for name, parameter in layer.named_parameters():
            if "lora" in name:
                parameter.normal_()
    return layer


def _base_weights(module):
    return {name: parameter for name, parameter in module.named_parameters() if "lora" not in name}


def test_muse_spark_resolves_from_its_checkpoint_config():
    model_type, spec = resolve_config_spec(
        {"model_type": "muse_spark", "text_config": {"model_type": "muse_spark_text"}}
    )
    assert model_type == "muse_spark" and spec is MUSE
    assert spec.fixed_targets is not None and spec.fixed_targets.select_all


@pytest.mark.parametrize("output_gate", [True, False], ids=["output-gate", "no-gate"])
def test_adapted_forward_equals_the_merged_weight(output_gate):
    layer = _adapted_layer(output_gate)
    merged = merge_lora_into_weights([layer], _base_weights(layer))
    attention = layer.self_attention
    for name, host in (
        ("self_attention.linear_qkv.weight", attention.linear_qkv),
        ("self_attention.linear_proj.weight", attention.linear_proj),
    ):
        x = torch.randn(5, host.weight.shape[1])
        torch.testing.assert_close(host(x)[0], x @ merged[name].t(), rtol=1e-4, atol=1e-4, msg=name)

    tokens = [2, 0, 3]
    experts = layer.mlp.experts
    for fc in ("linear_fc1", "linear_fc2"):
        host = getattr(experts, fc)
        x = torch.randn(sum(tokens), host.weight0.shape[1])
        merged_weights = torch.stack([merged[f"mlp.experts.{fc}.weight{i}"] for i in range(NUM_EXPERTS)])
        torch.testing.assert_close(
            host(x, tokens)[0], _grouped_linear(x, merged_weights, tokens), rtol=1e-4, atol=1e-4, msg=fc
        )


@pytest.mark.parametrize("output_gate", [True, False], ids=["output-gate", "no-gate"])
def test_frozen_linears_get_no_adapter(output_gate):
    layer = _adapted_layer(output_gate)
    adapted = {name.rsplit(".", 1)[0] for name, _ in layer.named_parameters() if "lora" in name}
    assert adapted == {
        "self_attention.lora_qkv_adapter",
        "self_attention.lora_o_adapter",
        "mlp.experts.lora_fc1_adapter",
        "mlp.experts.lora_fc2_adapter",
    }


@pytest.mark.parametrize("output_gate", [True, False], ids=["output-gate", "no-gate"])
@pytest.mark.parametrize("partition_size", [1, 2], ids=["tp1", "tp2-local"])
def test_qgkv_adapters_are_in_the_checkpoint_frame(output_gate, partition_size):
    """Converting the merged fused delta with the checkpoint mapping yields scale * B @ A per HF projection."""
    num_q, num_kv = HEADS // partition_size, GROUPS // partition_size
    spark = SimpleNamespace(
        num_key_value_heads=GROUPS,
        num_attention_heads=HEADS,
        head_dim=HEAD_DIM,
        hidden_size=HIDDEN,
        use_attn_o_gate=output_gate,
    )
    keys = ["q", "g", "k", "v"] if output_gate else ["q", "k", "v"]
    mapping = MuseSpark1xParamMapping(
        "decoder.layers.0.self_attention.linear_qkv.weight",
        {key: key for key in keys},
        spark,
        hf_specs={},
        kind="qgkv",
    )
    mapping.partition_size = partition_size
    torch.manual_seed(0)
    rows = {"q": num_q * HEAD_DIM, "g": num_q * HEAD_DIM, "k": num_kv * HEAD_DIM, "v": num_kv * HEAD_DIM}
    per_projection = {key: torch.randn(rows[key], HIDDEN) for key in keys}
    plain = torch.cat([per_projection[key] for key in keys])
    perm = build_muse_spark_qgkv_permutation(num_q, num_kv, HEAD_DIM, output_gate)
    fused = plain.index_select(0, perm)

    converted = mapping.megatron_to_hf(fused)
    for key in keys:
        assert torch.equal(converted[key], per_projection[key]), key
    assert torch.equal(mapping.hf_slices_to_megatron(per_projection), fused)


def test_adapter_export_names_follow_the_checkpoint():
    layer = _adapted_layer(True)
    names = {name for name, _ in export_lora_hf_named([layer])}
    prefix = "model.language_model.layers.3."
    expected = {
        f"{prefix}{module}.lora_{ab}.weight"
        for module in ("self_attn.q_proj", "self_attn.gate_proj", "self_attn.k_proj", "self_attn.v_proj")
        + ("self_attn.o_proj",)
        for ab in "AB"
    } | {f"{prefix}mlp.experts.{leaf}.lora_{ab}.weight" for leaf in ("w1", "w3", "w2") for ab in "AB"}
    assert names == expected
