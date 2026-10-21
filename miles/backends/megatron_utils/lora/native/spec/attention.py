"""Native-LoRA attention specs for fused GQA, gated GQA, MLA, and Inkling."""

from __future__ import annotations

import torch
import torch.nn as nn

from miles.backends.megatron_utils.lora.native.modules.linear import (
    LoRASplitAdapter,
    LoRASplitQKV,
    LoRAUnreducedRowLinear,
)
from miles.backends.megatron_utils.lora.native.spec import layout as L
from miles.backends.megatron_utils.lora.native.spec.base import (
    AttachContext,
    AttentionFamily,
    ProjectionSpec,
    ShardLayout,
)
from miles.backends.megatron_utils.lora.native.spec.layout import (
    AttentionSpecBase,
    FusedAttach,
    ModuleLayout,
    ProjectionBinding,
    attach_layout,
)


def _build_split_qkv(
    attention: nn.Module,
    hf_prefix: str,
    context: AttachContext,
    active: tuple[ProjectionSpec, ...],
) -> LoRASplitQKV:
    return LoRASplitQKV(
        hf_prefix=hf_prefix,
        reference=attention.linear_qkv.weight,
        context=context,
        projections=active,
        num_q=attention.num_attention_heads_per_partition,
        num_kv=attention.num_query_groups_per_partition,
        head_dim=attention.hidden_size_per_attention_head,
    )


def _replicated_guard(host: nn.Module, _context: AttachContext, projection: ProjectionSpec, full_out: int) -> None:
    assert _is_replicated_linear(host, full_out), (
        f"native MLA LoRA expects a replicated {projection.hf_leaf} (TELinear parallel_mode='duplicated'); "
        f"this build shards it ({tuple(host.weight.shape)} vs full out {full_out}). "
        "Use --megatron-to-hf-mode bridge for this variant."
    )


class GQAAttentionSpec(AttentionSpecBase):
    """Fused MCore QKV, including the gated-query layout used by Qwen hybrids."""

    name = "gqa"
    family = AttentionFamily.GQA
    layout = ModuleLayout(
        name="gqa",
        present_when_attr="linear_qkv",
        fused=(
            FusedAttach(
                module_attr="linear_qkv",
                projections=(
                    ProjectionSpec("q_proj", "q", ShardLayout.COLUMN),
                    ProjectionSpec("k_proj", "k", ShardLayout.COLUMN),
                    ProjectionSpec("v_proj", "v", ShardLayout.COLUMN),
                ),
                adapter_attr="lora_qkv_adapter",
                build=_build_split_qkv,
            ),
        ),
        singles=(
            ProjectionBinding(
                projection=ProjectionSpec("o_proj", "o", ShardLayout.ROW),
                module_attr="linear_proj",
                in_dim=L.gqa_o_in_local,
                out_dim=L.hidden_size,
                adapter_attr="lora_o_adapter",
            ),
        ),
    )

    def validate(self, config, *, tp_size: int) -> None:
        num_query_groups = getattr(config, "num_query_groups", None)
        assert num_query_groups is None or num_query_groups >= tp_size, (
            "native LoRA (--megatron-to-hf-mode raw) does not support this architecture: "
            f"num_query_groups ({num_query_groups}) < tensor parallel size ({tp_size}), so mcore splits a "
            "single query group across ranks and the local qkv rows are not a per-group slice. "
            "Use --megatron-to-hf-mode bridge."
        )


class MLAAttentionSpec(AttentionSpecBase):
    """Compressed query and key/value projection layout used by DeepSeek/GLM/Kimi."""

    name = "mla"
    family = AttentionFamily.MLA

    layout = ModuleLayout(
        name="mla",
        singles=(
            ProjectionBinding(
                projection=ProjectionSpec("q_a_proj", "a", ShardLayout.REPLICATED),
                module_attr="linear_q_down_proj",
                in_dim=L.hidden_size,
                out_dim=L.config_dim("q_lora_rank"),
                adapter_attr="lora_mla_q_a_adapter",
                guard=_replicated_guard,
            ),
            ProjectionBinding(
                projection=ProjectionSpec("q_b_proj", "b", ShardLayout.COLUMN),
                module_attr="linear_q_up_proj",
                in_dim=L.config_dim("q_lora_rank"),
                out_dim=L.mla_q_up_out_local,
                adapter_attr="lora_mla_q_b_adapter",
            ),
            ProjectionBinding(
                projection=ProjectionSpec("kv_a_proj_with_mqa", "a", ShardLayout.REPLICATED),
                module_attr="linear_kv_down_proj",
                in_dim=L.hidden_size,
                out_dim=L.mla_kv_down_out,
                adapter_attr="lora_mla_kv_a_adapter",
                guard=_replicated_guard,
            ),
            ProjectionBinding(
                projection=ProjectionSpec("kv_b_proj", "b", ShardLayout.COLUMN),
                module_attr="linear_kv_up_proj",
                in_dim=L.config_dim("kv_lora_rank"),
                out_dim=L.mla_kv_up_out_local,
                adapter_attr="lora_mla_kv_b_adapter",
            ),
            ProjectionBinding(
                projection=ProjectionSpec("o_proj", "o", ShardLayout.ROW),
                module_attr="linear_proj",
                in_dim=L.mla_o_in_local,
                out_dim=L.hidden_size,
                adapter_attr="lora_o_adapter",
            ),
        ),
    )

    def validate(self, config, *, tp_size: int) -> None:
        del tp_size
        assert getattr(config, "q_lora_rank", None), (
            "native LoRA does not support multi-latent attention without q_lora_rank "
            "(DeepSeek-V2-Lite, Moonlight): the query path is uncompressed, so the adapter exports "
            "an unfused q_proj alongside kv_a_proj_with_mqa, and SGLang's loader expects the fused "
            "qkv_a layout. Use --megatron-to-hf-mode bridge."
        )


class KimiK3AttentionSpec(AttentionSpecBase):
    """Kimi K3's HF-named attention: ``o_proj`` on every layer, the duplicated MLA down projections on MLA layers.

    The block gathers the sequence-parallel input once, so its linears run without SP. KDA
    layers have no ``q_a_proj``/``kv_a_proj_with_mqa``; their ``o_proj`` is the head-sharded
    linear-attention layer's ``out_proj``, whose partial sums the layer reduces itself.
    """

    name = "kimi_k3"
    family = AttentionFamily.MLA
    layout = ModuleLayout(
        name="kimi_k3_attention",
        full_sequence=True,
        singles=(
            ProjectionBinding(
                projection=ProjectionSpec("q_a_proj", "q_a", ShardLayout.REPLICATED),
                module_attr="q_a_proj",
                in_dim=L.hidden_size,
                out_dim=L.config_dim("q_lora_rank"),
                adapter_attr="lora_mla_q_a_adapter",
                guard=_replicated_guard,
            ),
            ProjectionBinding(
                projection=ProjectionSpec("kv_a_proj_with_mqa", "kv_a", ShardLayout.REPLICATED),
                module_attr="kv_a_proj_with_mqa",
                in_dim=L.hidden_size,
                out_dim=L.mla_kv_down_out,
                adapter_attr="lora_mla_kv_a_adapter",
                guard=_replicated_guard,
            ),
            ProjectionBinding(
                projection=ProjectionSpec("o_proj", "o", ShardLayout.ROW),
                module_attr="o_proj",
                in_dim=L.host_in_local("o_proj"),
                out_dim=L.hidden_size,
                adapter_attr="lora_o_adapter",
            ),
        ),
    )
    kda_layout = ModuleLayout(
        name="kimi_k3_kda_attention",
        full_sequence=True,
        singles=(
            ProjectionBinding(
                projection=ProjectionSpec("o_proj", "o", ShardLayout.ROW),
                module_attr="out_proj",
                in_dim=L.host_in_local("out_proj"),
                out_dim=L.hidden_size,
                adapter_attr="lora_o_adapter",
                adapter_class=LoRAUnreducedRowLinear,
            ),
        ),
    )

    def applies_to(self, block: nn.Module) -> bool:
        return True

    def attach(self, block: nn.Module, layer: nn.Module, context: AttachContext) -> int:
        if block.is_kda:
            return attach_layout(block.linear_attn, self.kda_layout, layer, context)
        return attach_layout(block, self.layout, layer, context)


class HybridGQAGDNAttentionSpec(GQAAttentionSpec):
    """Qwen hybrids: GQA layers carry adapters; GDN mixer layers lack ``linear_qkv`` and are skipped."""

    name = "gqa_gdn"


class _InklingSplitQKVR(LoRASplitAdapter):
    """Four independent adapters over Inkling's plain-concat fused [q;k;v;r]."""


def _build_split_qkvr(
    attention: nn.Module,
    hf_prefix: str,
    context: AttachContext,
    active: tuple[ProjectionSpec, ...],
) -> _InklingSplitQKVR:
    return _InklingSplitQKVR(
        hf_prefix=hf_prefix,
        reference=attention.linear_qkv.weight,
        context=context,
        projections=active,
        rows_by_param_prefix={
            "q": attention.nh_l * attention.hd,
            "k": attention.nkv_l * attention.hd,
            "v": attention.nkv_l * attention.hd,
            "r": attention.nh_l * attention.d_rel,
        },
    )


class InklingAttentionSpec(AttentionSpecBase):
    """GQA-with-relative-projection: fused [q;k;v;r], plain concat (no group permute).

    Exports carry Inkling's TML names (``attn.wq_du/wk_dv/wv_dv/wr_du/wo_ud``),
    which SGLang auto-detects, so serving passes ``lora_target_modules=["all"]``.
    """

    name = "inkling"
    family = AttentionFamily.GQA
    layout = ModuleLayout(
        name="inkling_attention",
        present_when_attr="linear_qkv",
        hf_block_prefix="attn.",
        fused=(
            FusedAttach(
                module_attr="linear_qkv",
                projections=(
                    ProjectionSpec("wq_du", "q", ShardLayout.COLUMN),
                    ProjectionSpec("wk_dv", "k", ShardLayout.COLUMN),
                    ProjectionSpec("wv_dv", "v", ShardLayout.COLUMN),
                    ProjectionSpec("wr_du", "r", ShardLayout.COLUMN),
                ),
                adapter_attr="lora_qkv_adapter",
                build=_build_split_qkvr,
            ),
        ),
        singles=(
            ProjectionBinding(
                projection=ProjectionSpec("wo_ud", "o", ShardLayout.ROW),
                module_attr="linear_proj",
                in_dim=L.inkling_o_in_local,
                out_dim=L.hidden_size,
                adapter_attr="lora_o_adapter",
            ),
        ),
    )


class _MuseSparkSplitQGKV(LoRASplitAdapter):
    """Independent q/gate/k/v adapters over Muse Spark's fused ``linear_qkv``.

    The adapters live in the checkpoint's frame: each ``B`` has the rows of the HF ``q_proj`` /
    ``gate_proj`` / ``k_proj`` / ``v_proj`` weight (q and k rows with the two rotary halves of a head
    consecutive). ``out_perm`` maps that plain concat onto the fused rows MCore keeps per query group,
    ``[q heads; gate heads; k; v]`` with the rotary pairs interleaved, the inverse of
    ``MuseSpark1xParamMapping``'s ``qgkv`` conversion.
    """

    def __init__(self, *, rows_by_param_prefix: dict[str, int], num_q: int, num_kv: int, head_dim: int, **kwargs):
        super().__init__(rows_by_param_prefix=rows_by_param_prefix, **kwargs)
        permutation = build_muse_spark_qgkv_permutation(num_q, num_kv, head_dim, "gate" in rows_by_param_prefix)
        self.register_buffer("out_perm", permutation.to(kwargs["reference"].device), persistent=False)
        # The merged weight sync runs while a colocated trainer is offloaded, when ``out_perm`` (allocated with
        # the model) is paused; the merge packs rows with this host copy instead.
        self._host_out_perm = permutation

    def _pack(self, delta: torch.Tensor) -> torch.Tensor:
        return delta.index_select(-1, self.out_perm)

    def _pack_rows(self, delta: torch.Tensor) -> torch.Tensor:
        return delta.index_select(0, self._host_out_perm.to(delta.device))


def build_muse_spark_qgkv_permutation(
    num_q: int, num_kv: int, head_dim: int, output_gate: bool, device=None
) -> torch.Tensor:
    """Index of the plain ``[q; gate; k; v]`` row (checkpoint frame) that feeds each fused MCore row."""
    assert head_dim % 2 == 0, f"Muse Spark rotary pairs need an even head_dim, got {head_dim}"
    q_rows, kv_rows = num_q * head_dim, num_kv * head_dim

    def to_kernel_order(rows: torch.Tensor) -> torch.Tensor:
        # per head: (first halves, second halves) -> interleaved rotary pairs (weight_mapping.rope_pairs_hf_to_kernel)
        return rows.reshape(-1, 2, head_dim // 2).transpose(1, 2).reshape(-1)

    offset = 0
    pieces = []
    for name, rows in (("q", q_rows), ("gate", q_rows if output_gate else 0), ("k", kv_rows), ("v", kv_rows)):
        if not rows:
            continue
        index = torch.arange(offset, offset + rows)
        offset += rows
        if name in ("q", "k"):
            index = to_kernel_order(index)
        pieces.append(index.reshape(num_kv, -1))
    return torch.cat(pieces, dim=1).reshape(-1).to(device)


def _build_muse_spark_split_qgkv(
    attention: nn.Module,
    hf_prefix: str,
    context: AttachContext,
    active: tuple[ProjectionSpec, ...],
) -> _MuseSparkSplitQGKV:
    num_q = attention.num_attention_heads_per_partition
    num_kv = attention.num_query_groups_per_partition
    head_dim = attention.hidden_size_per_attention_head
    rows = {"q": num_q * head_dim}
    if context.output_gate:
        rows["gate"] = num_q * head_dim
    rows.update(k=num_kv * head_dim, v=num_kv * head_dim)
    assert sum(rows.values()) == attention.linear_qkv.weight.shape[0], (
        f"Muse Spark linear_qkv has {attention.linear_qkv.weight.shape[0]} local rows, the q/gate/k/v layout "
        f"{rows} sums to {sum(rows.values())}"
    )
    return _MuseSparkSplitQGKV(
        hf_prefix=hf_prefix,
        reference=attention.linear_qkv.weight,
        context=context,
        projections=tuple(projection for projection in active if projection.param_prefix in rows),
        rows_by_param_prefix=rows,
        num_q=num_q,
        num_kv=num_kv,
        head_dim=head_dim,
    )


class MuseSparkAttentionSpec(AttentionSpecBase):
    """Muse Spark's GQA: fused ``linear_qkv`` holding q, the output gate (``gate_proj``), k and v per
    query group, plus the row-parallel ``o_proj``. Without the output gate the gate slot is absent."""

    name = "muse_spark"
    family = AttentionFamily.GQA
    layout = ModuleLayout(
        name="muse_spark_attention",
        present_when_attr="linear_qkv",
        fused=(
            FusedAttach(
                module_attr="linear_qkv",
                projections=(
                    ProjectionSpec("q_proj", "q", ShardLayout.COLUMN),
                    ProjectionSpec("gate_proj", "gate", ShardLayout.COLUMN),
                    ProjectionSpec("k_proj", "k", ShardLayout.COLUMN),
                    ProjectionSpec("v_proj", "v", ShardLayout.COLUMN),
                ),
                adapter_attr="lora_qkv_adapter",
                build=_build_muse_spark_split_qgkv,
            ),
        ),
        singles=(
            ProjectionBinding(
                projection=ProjectionSpec("o_proj", "o", ShardLayout.ROW),
                module_attr="linear_proj",
                in_dim=L.gqa_o_in_local,
                out_dim=L.hidden_size,
                adapter_attr="lora_o_adapter",
            ),
        ),
    )

    def validate(self, config, *, tp_size: int) -> None:
        GQAAttentionSpec().validate(config, tp_size=tp_size)


def _is_replicated_linear(module: nn.Module, full_out: int) -> bool:
    if getattr(module, "parallel_mode", None) == "duplicated":
        return True
    return module.weight.shape[0] == full_out
