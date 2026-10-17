"""Native LoRA for Muse Spark's Megatron model.

SGLang's Muse Spark model serves no LoRA, so this LoRA trains in Megatron and reaches the engines merged
into the base weights (``--lora-sync-merged``, miles.backends.megatron_utils.lora.merge). Each adapted
linear gets one ``lora_adapter`` module holding ``lora_A`` and ``lora_B`` in Megatron's own layout for that
linear, so the merged weight is ``W + scale * B @ A`` on every rank's local shard with no communication:

- column-parallel (``linear_qkv``): ``A`` [r, in] replicated, ``B`` [out / tp, r]
- row-parallel (``linear_proj``): ``A`` [r, in / tp], ``B`` [out, r] replicated
- grouped experts (``linear_fc1`` / ``linear_fc2``, expert-TP 1): ``A`` [local experts, r, in],
  ``B`` [local experts, out, r]

The fused ``linear_qkv`` (queries, keys, values and the output gate) and the fused expert gate/up matrix
are adapted as one matrix each, so the merge needs no knowledge of their row layout. Router, embeddings,
output layer, norms and the expert latent projections stay frozen.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core import parallel_state

from miles.backends.megatron_utils.lora.utils import patch_param_grad_buffer_for_colocate_mode_lora
from miles_plugins.models.inkling.lora import (
    _dropout,
    _gather_sequence_parallel,
    _grouped_linear,
    _new_param,
    _reduce_row_parallel,
)
from miles_plugins.models.muse_spark.ops.attention import MuseSpark1xSelfAttention
from miles_plugins.models.muse_spark.ops.moe import MuseSpark1xMoELayer

logger = logging.getLogger(__name__)

ADAPTER_ATTR = "lora_adapter"


class MuseSparkLoRAAdapter(nn.Module):
    """``lora_A`` / ``lora_B`` of one linear; invisible to Megatron dist-checkpointing (saved as adapter shards)."""

    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        del prefix, sharded_offsets, metadata
        return {}


def _attach(linear: nn.Module, kind: str, a: nn.Parameter, b: nn.Parameter) -> MuseSparkLoRAAdapter:
    adapter = MuseSparkLoRAAdapter(kind)
    adapter.register_parameter("lora_A", a)
    adapter.register_parameter("lora_B", b)
    setattr(linear, ADAPTER_ATTR, adapter)
    return adapter


def _apply_column_parallel(linear, config, rank: int, *, scale: float, dropout: float, a_init: str) -> None:
    out_local, in_features = linear.weight.shape
    sequence_parallel = bool(config.sequence_parallel)
    adapter = _attach(
        linear,
        "column",
        _new_param(linear.weight, (rank, in_features), init=a_init, grad_sum_group="tp"),
        _new_param(linear.weight, (out_local, rank), init="zero"),
    )
    original = linear.forward

    def forward(inputs, *forward_args, **forward_kwargs):
        output, bias = original(inputs, *forward_args, **forward_kwargs)
        full = _dropout(_gather_sequence_parallel(inputs, sequence_parallel), dropout, linear.training)
        delta = F.linear(F.linear(full, adapter.lora_A), adapter.lora_B)
        return torch.add(output, delta, alpha=scale), bias

    linear.forward = forward


def _apply_row_parallel(linear, config, rank: int, *, scale: float, dropout: float, a_init: str) -> None:
    out_features, in_local = linear.weight.shape
    sequence_parallel = bool(config.sequence_parallel)
    adapter = _attach(
        linear,
        "row",
        _new_param(linear.weight, (rank, in_local), init=a_init),
        _new_param(
            linear.weight, (out_features, rank), init="zero", grad_sum_group="tp" if sequence_parallel else None
        ),
    )
    original = linear.forward

    def forward(inputs, *forward_args, **forward_kwargs):
        output, bias = original(inputs, *forward_args, **forward_kwargs)
        local = F.linear(_dropout(inputs, dropout, linear.training), adapter.lora_A)
        delta = F.linear(_reduce_row_parallel(local, sequence_parallel), adapter.lora_B)
        return torch.add(output, delta, alpha=scale), bias

    linear.forward = forward


def _apply_grouped_experts(linear, num_local_experts: int, is_ep: bool, rank: int, *, scale, dropout, a_init) -> None:
    out_features, in_features = linear.weight0.shape
    adapter = _attach(
        linear,
        "experts",
        _new_param(linear.weight0, (num_local_experts, rank, in_features), init=a_init, expert=is_ep),
        _new_param(linear.weight0, (num_local_experts, out_features, rank), init="zero", expert=is_ep),
    )
    original = linear.forward

    def forward(inputs, tokens_per_expert, *forward_args, **forward_kwargs):
        output, bias = original(inputs, tokens_per_expert, *forward_args, **forward_kwargs)
        inner = _grouped_linear(_dropout(inputs, dropout, linear.training), adapter.lora_A, tokens_per_expert)
        delta = _grouped_linear(inner, adapter.lora_B, tokens_per_expert)
        return torch.add(output, delta, alpha=scale), bias

    linear.forward = forward


def apply_muse_spark_lora(model, args):
    """Attach LoRA to ONE built Muse Spark model chunk (before Float16Module / DDP wrapping)."""
    if args.offload_train:
        # keep adapter param/grad buffers out of the pausable memory-saver region
        patch_param_grad_buffer_for_colocate_mode_lora()

    rank = int(args.lora_rank)
    assert rank > 0, "apply_muse_spark_lora requires --lora-rank > 0"
    kwargs = dict(
        scale=float(args.lora_alpha) / float(rank),
        dropout=float(getattr(args, "lora_dropout", 0.0) or 0.0),
        a_init=getattr(args, "lora_A_init_method", "xavier") or "xavier",
    )
    is_ep = parallel_state.get_expert_model_parallel_world_size() > 1

    for param in model.parameters():
        param.requires_grad = False

    for layer in model.decoder.layers:
        attention = layer.self_attention
        assert isinstance(attention, MuseSpark1xSelfAttention), f"unexpected attention {type(attention)}"
        _apply_column_parallel(attention.linear_qkv, attention.config, rank, **kwargs)
        _apply_row_parallel(attention.linear_proj, attention.config, rank, **kwargs)

        moe = layer.mlp
        assert isinstance(moe, MuseSpark1xMoELayer), f"unexpected mlp {type(moe)}: every Muse Spark layer is MoE"
        assert (getattr(moe.config, "expert_tensor_parallel_size", 1) or 1) == 1, "Muse Spark LoRA assumes ETP=1"
        experts = moe.experts
        for linear in (experts.linear_fc1, experts.linear_fc2):
            _apply_grouped_experts(linear, experts.num_local_experts, is_ep, rank, **kwargs)

    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    logger.info(
        "[muse-spark-lora] rank=%d alpha=%s | trainable %s / %s (%.4f%%) on this rank",
        rank,
        args.lora_alpha,
        f"{trainable:,}",
        f"{total:,}",
        100.0 * trainable / max(total, 1),
    )
    return model


def wrap_model_provider_with_muse_spark_lora(provider_func, args):
    """Wrap a miles model provider so every built chunk gets LoRA before DDP wrap."""

    def wrapped(*provider_args, **provider_kwargs):
        return apply_muse_spark_lora(provider_func(*provider_args, **provider_kwargs), args)

    return wrapped
