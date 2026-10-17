"""Muse Spark native LoRA with merged weight sync (--lora-sync-merged).

The adapted forward of every linear equals the frozen linear run with the merged weight that the
engines receive (``MergedLoRAWeights``), so rollouts sample from the trained policy; the merge is local
per shard and maps global expert indices to local ones."""

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from miles.backends.megatron_utils.lora.merge import MergedLoRAWeights
from miles.utils.lora import arguments as lora_arguments
from miles.utils.lora.utils import lora_adapter_memory_layout, lora_rollout_enabled
from miles_plugins.models.muse_spark.lora_targets import ADAPTER_TARGETS

RANK, SCALE = 4, 2.0


class _Linear(nn.Module):
    """A frozen TE-style linear: forward returns (output, bias)."""

    def __init__(self, out_features, in_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)

    def forward(self, x):
        return x @ self.weight.T, None


class _GroupedLinear(nn.Module):
    """A frozen TE-style grouped expert linear with weight0..weight{E-1}."""

    def __init__(self, num_experts, out_features, in_features):
        super().__init__()
        for e in range(num_experts):
            self.register_parameter(
                f"weight{e}", nn.Parameter(torch.randn(out_features, in_features), requires_grad=False)
            )
        self.num_experts = num_experts

    def forward(self, x, tokens_per_expert):
        pieces = torch.split(x, list(tokens_per_expert))
        return torch.cat([p @ getattr(self, f"weight{e}").T for e, p in enumerate(pieces)]), None


@pytest.fixture
def muse_lora(monkeypatch):
    """The LoRA module (it imports Megatron) at TP 1 without sequence parallelism: gather and reduce are identities."""
    module = pytest.importorskip("miles_plugins.models.muse_spark.lora")
    monkeypatch.setattr(module, "_gather_sequence_parallel", lambda x, sequence_parallel: x)
    monkeypatch.setattr(module, "_reduce_row_parallel", lambda x, sequence_parallel: x)
    return module


SINGLE_RANK = SimpleNamespace(sequence_parallel=False)


def _randomize_adapter(linear):
    with torch.no_grad():
        for param in linear.lora_adapter.parameters():
            param.normal_()


def _weights(prefix, linear):
    return {f"{prefix}.{name}": tensor.detach() for name, tensor in linear.named_parameters()}


@pytest.mark.parametrize("kind", ["column", "row"])
def test_the_adapted_forward_equals_the_merged_weight(muse_lora, kind):
    torch.manual_seed(0)
    linear = _Linear(12, 10)
    apply = muse_lora._apply_column_parallel if kind == "column" else muse_lora._apply_row_parallel
    apply(linear, SINGLE_RANK, RANK, scale=SCALE, dropout=0.0, a_init="xavier")
    _randomize_adapter(linear)
    x = torch.randn(5, 10)

    adapted, _ = linear(x)
    merged = MergedLoRAWeights(_weights("m.self_attention.linear_qkv", linear), scale=SCALE, expert_offset=0)
    expected = x @ merged["m.self_attention.linear_qkv.weight"].T

    torch.testing.assert_close(adapted, expected, rtol=1e-5, atol=1e-5)


def test_grouped_experts_merge_per_expert_with_the_global_offset(muse_lora):
    torch.manual_seed(0)
    linear = _GroupedLinear(3, 8, 6)
    muse_lora._apply_grouped_experts(linear, 3, False, RANK, scale=SCALE, dropout=0.0, a_init="xavier")
    _randomize_adapter(linear)
    tokens = [2, 0, 3]
    x = torch.randn(sum(tokens), 6)

    adapted, _ = linear(x, tokens)
    # this rank holds global experts 6, 7, 8
    module = "module.module.decoder.layers.0.mlp.experts.linear_fc1"
    weights = {
        f"{module}.lora_adapter.lora_A": linear.lora_adapter.lora_A.detach(),
        f"{module}.lora_adapter.lora_B": linear.lora_adapter.lora_B.detach(),
        **{f"{module}.weight{6 + e}": getattr(linear, f"weight{e}").detach() for e in range(3)},
    }
    merged = MergedLoRAWeights(weights, scale=SCALE, expert_offset=6)
    pieces = torch.split(x, tokens)
    expected = torch.cat([p @ merged[f"{module}.weight{6 + e}"].T for e, p in enumerate(pieces)])

    torch.testing.assert_close(adapted, expected, rtol=1e-5, atol=1e-5)


def test_zero_initialized_b_merges_to_the_base_and_unadapted_weights_pass_through(muse_lora):
    linear = _Linear(4, 3)
    muse_lora._apply_column_parallel(linear, SINGLE_RANK, RANK, scale=SCALE, dropout=0.0, a_init="xavier")
    weights = {**_weights("a.linear_qkv", linear), "a.norm.weight": torch.ones(3)}
    merged = MergedLoRAWeights(weights, scale=SCALE, expert_offset=0)

    assert torch.equal(merged["a.linear_qkv.weight"], linear.weight.detach())
    assert merged["a.norm.weight"] is weights["a.norm.weight"]
    assert set(merged) == set(weights)


def test_the_merge_keeps_the_base_dtype_and_never_modifies_the_base():
    base = torch.randn(4, 3).to(torch.bfloat16)
    weights = {
        "x.weight": base,
        "x.lora_adapter.lora_A": torch.randn(RANK, 3).to(torch.bfloat16),
        "x.lora_adapter.lora_B": torch.randn(4, RANK).to(torch.bfloat16),
    }
    before = base.clone()
    merged = MergedLoRAWeights(weights, scale=SCALE, expert_offset=0)["x.weight"]

    assert merged.dtype == torch.bfloat16 and torch.equal(base, before)
    expected = (
        base.float() + SCALE * weights["x.lora_adapter.lora_B"].float() @ weights["x.lora_adapter.lora_A"].float()
    )
    torch.testing.assert_close(merged.float(), expected.to(torch.bfloat16).float())


def test_an_adapter_without_a_synced_base_weight_is_refused():
    weights = {"x.lora_adapter.lora_A": torch.zeros(RANK, 3), "x.lora_adapter.lora_B": torch.zeros(4, RANK)}
    with pytest.raises(AssertionError, match="base weight is not synced"):
        MergedLoRAWeights(weights, scale=SCALE, expert_offset=0)


def test_a_non_local_expert_is_refused():
    weights = {
        "m.mlp.experts.linear_fc2.lora_adapter.lora_A": torch.zeros(2, RANK, 3),
        "m.mlp.experts.linear_fc2.lora_adapter.lora_B": torch.zeros(2, 4, RANK),
        "m.mlp.experts.linear_fc2.weight2": torch.zeros(4, 3),
    }
    merged = MergedLoRAWeights(weights, scale=SCALE, expert_offset=0)
    with pytest.raises(AssertionError, match="not local"):
        merged["m.mlp.experts.linear_fc2.weight2"]


def _lora_args(**over):
    base = dict(
        lora_rank=RANK,
        lora_adapter_path=None,
        lora_train_only=False,
        lora_sync_merged=True,
        megatron_to_hf_mode="raw",
        target_modules=None,
        exclude_modules=None,
        lora_type="lora",
        experts_shared_outer_loras=False,
        multi_lora=False,
        multi_lora_n_adapters=0,
        train_backend="megatron",
        hf_checkpoint="unused",
    )
    base.update(over)
    return Namespace(**base)


@pytest.fixture
def muse_config(monkeypatch):
    monkeypatch.setattr(lora_arguments, "load_hf_config", lambda path: SimpleNamespace(model_type="muse_spark"))
    monkeypatch.setattr(lora_arguments, "validate_multi_lora_args", lambda args: None)


def test_merged_lora_validates_to_the_fixed_muse_spark_targets(muse_config):
    args = _lora_args()
    lora_arguments.validate_lora_args(args)
    assert args.lora_adapter_targets == list(ADAPTER_TARGETS) == args.hf_lora_targets


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"lora_train_only": True}, "keeps them frozen"),
        ({"target_modules": "attn"}, "fixed set"),
        ({"megatron_to_hf_mode": "bridge"}, "raw Megatron"),
        ({"experts_shared_outer_loras": True}, "per-expert"),
    ],
)
def test_merged_lora_refuses_conflicting_options(muse_config, override, message):
    with pytest.raises(AssertionError, match=message):
        lora_arguments.validate_lora_args(_lora_args(**override))


def test_merged_lora_is_muse_spark_only(monkeypatch):
    monkeypatch.setattr(lora_arguments, "load_hf_config", lambda path: SimpleNamespace(model_type="kimi_k3"))
    monkeypatch.setattr(lora_arguments, "validate_multi_lora_args", lambda args: None)
    with pytest.raises(AssertionError, match="Muse Spark native LoRA only"):
        lora_arguments.validate_lora_args(_lora_args())


def test_merged_lora_serves_no_adapter_but_keeps_the_lora_memory_layout():
    merged = _lora_args()
    assert not lora_rollout_enabled(merged) and lora_adapter_memory_layout(merged)
    served = _lora_args(lora_sync_merged=False)
    assert lora_rollout_enabled(served) and lora_adapter_memory_layout(served)
    train_only = _lora_args(lora_sync_merged=False, lora_train_only=True)
    assert not lora_rollout_enabled(train_only) and not lora_adapter_memory_layout(train_only)
    assert not lora_adapter_memory_layout(_lora_args(lora_rank=0))
