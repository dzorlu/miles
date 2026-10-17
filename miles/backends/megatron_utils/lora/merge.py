"""Base weights with native LoRA merged in, for engines that serve no LoRA (``--lora-sync-merged``).

The adapters follow one layout (miles_plugins.models.muse_spark.lora): an adapted linear's module holds
``lora_adapter.lora_A`` and ``lora_adapter.lora_B`` in that linear's own local layout, so a rank's merged
shard is ``W + scale * B @ A`` with no communication. Grouped experts hold one ``A`` / ``B`` per local
expert; the global name ``...experts.linear_fc1.weight{g}`` maps to local expert ``g - expert_offset``.

The base weights are frozen and never modified: every sync merges from them afresh, so bf16 rounding of
the merged weight does not accumulate across steps.
"""

import re
from argparse import Namespace
from collections.abc import Iterator, Mapping

import torch

from miles.backends.training_utils.parallel import get_parallel_state

_A_SUFFIX = ".lora_adapter.lora_A"
_B_SUFFIX = ".lora_adapter.lora_B"
_EXPERT_WEIGHT = re.compile(r"^(?P<module>.+\.mlp\.experts\.linear_fc[12])\.weight(?P<expert>\d+)$")


def merged_lora_weights(args: Namespace, weights: Mapping[str, torch.Tensor]) -> "MergedLoRAWeights":
    """``weights`` (global names, this rank's shards) with the native adapters merged in, for
    ``--lora-sync-merged``; the scale and this rank's first global expert index come from ``args``."""
    parallel = get_parallel_state()
    expert_offset = parallel.ep.rank * args.num_experts // parallel.ep.size if args.num_experts else 0
    return MergedLoRAWeights(weights, scale=args.lora_alpha / args.lora_rank, expert_offset=expert_offset)


class MergedLoRAWeights(Mapping):
    """A read-only view of ``weights`` (global name -> local tensor, adapters included) in which every
    adapted base weight reads as its merged value; computed per access, on the current CUDA device."""

    def __init__(self, weights: Mapping[str, torch.Tensor], *, scale: float, expert_offset: int) -> None:
        self._weights = weights
        self._scale = scale
        self._expert_offset = expert_offset
        self._adapted = frozenset(name.removesuffix(_A_SUFFIX) for name in weights if name.endswith(_A_SUFFIX))
        merged_modules = {module for module, _ in filter(None, map(self._adapter_of, weights))}
        orphans = self._adapted - merged_modules
        assert not orphans, f"LoRA adapters whose base weight is not synced: {sorted(orphans)[:4]}"

    def _adapter_of(self, name: str) -> tuple[str, int | None] | None:
        if name.endswith(".weight") and name.removesuffix(".weight") in self._adapted:
            return name.removesuffix(".weight"), None
        match = _EXPERT_WEIGHT.match(name)
        if match and match["module"] in self._adapted:
            return match["module"], int(match["expert"]) - self._expert_offset
        return None

    def __getitem__(self, name: str) -> torch.Tensor:
        base = self._weights[name]
        found = self._adapter_of(name)
        if found is None:
            return base
        module, expert = found
        lora_a, lora_b = self._weights[module + _A_SUFFIX], self._weights[module + _B_SUFFIX]
        if expert is not None:
            assert 0 <= expert < lora_a.shape[0], f"{name}: expert {expert} is not local ({lora_a.shape[0]} local)"
            lora_a, lora_b = lora_a[expert], lora_b[expert]
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else base.device
        delta = lora_b.to(device, torch.float32) @ lora_a.to(device, torch.float32)
        assert delta.shape == base.shape, f"{name}: LoRA delta {tuple(delta.shape)} vs base {tuple(base.shape)}"
        return (base.to(device, torch.float32) + self._scale * delta).to(base.dtype)

    def __iter__(self) -> Iterator[str]:
        return iter(self._weights)

    def __len__(self) -> int:
        return len(self._weights)
