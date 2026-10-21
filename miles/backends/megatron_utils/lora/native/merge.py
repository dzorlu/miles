"""Merged-weight view of a native-LoRA model: full-model HF export, and the weight sync to engines
that serve no LoRA (``--lora-sync-merged``)."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence

import torch
import torch.nn as nn

from miles.backends.megatron_utils.lora.native.modules.linear import iter_adapters


class MergedWeights(Mapping[str, torch.Tensor]):
    """``weights`` with every adapted host weight read as ``weight + delta``.

    Deltas are computed on access in the host's local layout, so the base-weight
    converters gather and rename merged tensors exactly as they would the base.
    A merged tensor is on its delta's device (the adapter's), whatever device the weight is on.
    """

    def __init__(self, weights: Mapping[str, torch.Tensor], deltas: dict[int, Callable[[], torch.Tensor]]):
        self._weights = weights
        self._deltas = deltas

    @torch.no_grad()
    def __getitem__(self, name: str) -> torch.Tensor:
        weight = self._weights[name]
        delta = self._deltas.get(id(weight))
        if delta is None:
            return weight
        update = delta()
        # Move first, widen on the delta's device: a host weight (e.g. a pinned memory-saver backup) widened
        # with .float() or a combined .to(cuda, float32) converts on the CPU. bf16 -> fp32 is exact either way.
        return (weight.to(update.device).float() + update).to(weight.dtype)

    def __iter__(self) -> Iterator[str]:
        return iter(self._weights)

    def __len__(self) -> int:
        return len(self._weights)


def merge_lora_into_weights(model_chunks: Sequence[nn.Module], weights: Mapping[str, torch.Tensor]) -> MergedWeights:
    exported = {id(weight) for weight in weights.values()}
    deltas: dict[int, Callable[[], torch.Tensor]] = {}
    for adapter in iter_adapters(model_chunks):
        for host_weight, delta in adapter.weight_deltas():
            assert id(host_weight) in exported, f"the host weight of adapter {adapter.hf_prefix!r} is not exported"
            assert id(host_weight) not in deltas, f"two adapters merge into one host weight of {adapter.hf_prefix!r}"
            deltas[id(host_weight)] = delta
    return MergedWeights(weights, deltas)


def merge_lora_into_named_weights(
    model_chunks: Sequence[nn.Module],
    weights: Mapping[str, torch.Tensor],
    names: Mapping[int, str],
    *,
    device: torch.device | None = None,
) -> MergedWeights:
    """``weights`` (name -> any copy of the model's local tensors, adapters included) with every adapted
    host weight read as ``weight + delta``, for ``--lora-sync-merged``.

    ``names`` maps ``id`` of each live model tensor to its name in ``weights``, so the deltas read the
    adapter tensors from ``weights`` too: a colocated trainer syncs from host backups while its live
    parameters are paused. ``device`` is where the adapter tensors are moved before the delta is computed.
    Nothing is written: every sync merges from the frozen base afresh.
    """

    def read(parameter: nn.Parameter) -> torch.Tensor:
        tensor = weights[names[id(parameter)]]
        return tensor if device is None else tensor.to(device)

    deltas: dict[int, Callable[[], torch.Tensor]] = {}
    for adapter in iter_adapters(model_chunks):
        for host_weight, delta in adapter.weight_deltas(read):
            name = names.get(id(host_weight))
            assert name in weights, f"the host weight of adapter {adapter.hf_prefix!r} is not exported"
            key = id(weights[name])
            assert key not in deltas, f"two adapters merge into one host weight of {adapter.hf_prefix!r}"
            deltas[key] = delta
    return MergedWeights(weights, deltas)
