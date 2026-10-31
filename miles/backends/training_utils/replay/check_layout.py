"""Where the R3 checker's mismatched tokens sit, for --ci-test runs.

The checker sees one rank's local slice of a packed microbatch. This module keeps, per
microbatch, the map from local rows to packed positions and each sample's layout (offset,
image placeholder spans, media references, a hash of each image's trainer pixels), so a
mismatch can be reported against the sample and image it belongs to: a shifted offset breaks
every token after a point, a wrong image breaks one span and leaves the text after it clean.
"""

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

_AFTER_SPAN_WINDOW = 512
_MIN_SAMPLE_MISMATCHES = 4


@dataclass(frozen=True)
class SampleLayout:
    offset: int
    length: int
    sample_index: int | None
    rollout_id: int | None
    # (start, end) of each image placeholder run, sample-relative, end exclusive
    image_spans: tuple[tuple[int, int], ...]
    image_refs: tuple[str, ...]
    pixel_patches: tuple[int, ...]
    pixel_hashes: tuple[str, ...]


@dataclass(frozen=True)
class MicrobatchLayout:
    # local row -> packed position, -1 for padding
    local_to_packed: torch.Tensor
    samples: tuple[SampleLayout, ...]


def _image_spans(tokens: torch.Tensor, image_token_id: int) -> tuple[tuple[int, int], ...]:
    is_image = (tokens == image_token_id).to(torch.int8)
    edges = torch.diff(is_image, prepend=is_image.new_zeros(1), append=is_image.new_zeros(1))
    starts = (edges == 1).nonzero().flatten().tolist()
    ends = (edges == -1).nonzero().flatten().tolist()
    return tuple(zip(starts, ends, strict=True))


def _pixel_digests(train_inputs: dict | None) -> tuple[tuple[int, ...], tuple[str, ...]]:
    if not train_inputs or "pixel_values" not in train_inputs or "image_grid_thw" not in train_inputs:
        return (), ()
    patches = train_inputs["image_grid_thw"].prod(dim=-1).tolist()
    hashes = []
    for chunk in torch.split(train_inputs["pixel_values"], patches, dim=0):
        raw = chunk.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        hashes.append(hashlib.sha256(raw).hexdigest()[:12])
    return tuple(patches), tuple(hashes)


def _sample_layout(offset, tokens, image_token_id, sample_index, rollout_id, refs, train_inputs) -> SampleLayout:
    patches, hashes = _pixel_digests(train_inputs)
    return SampleLayout(
        offset=offset,
        length=int(tokens.shape[0]),
        sample_index=sample_index,
        rollout_id=rollout_id,
        image_spans=_image_spans(tokens.cpu(), image_token_id),
        image_refs=tuple((refs or {}).get("images") or ()),
        pixel_patches=patches,
        pixel_hashes=hashes,
    )


def build_layouts(
    *,
    data_iterator,
    num_microbatches: list[int],
    entry_lengths: list[int],
    image_token_id: int,
    align: Callable[[torch.Tensor, int], torch.Tensor] | None,
) -> list[MicrobatchLayout]:
    """One layout per microbatch, in the order the replay queues hold them."""
    keys = ["tokens", "sample_indices", "rollout_ids", "multimodal_media_refs", "multimodal_train_inputs"]
    data_iterator.reset()
    layouts = []
    for entry_length in entry_lengths[: sum(num_microbatches)]:
        batch = data_iterator.get_next(keys)
        n = len(batch["tokens"])
        column = {key: batch[key] if batch[key] is not None else [None] * n for key in keys}
        samples, offset = [], 0
        for i, tokens in enumerate(column["tokens"]):
            samples.append(
                _sample_layout(
                    offset,
                    tokens,
                    image_token_id,
                    column["sample_indices"][i],
                    column["rollout_ids"][i],
                    column["multimodal_media_refs"][i],
                    column["multimodal_train_inputs"][i],
                )
            )
            offset += int(tokens.shape[0])
        positions = torch.arange(entry_length, dtype=torch.long)
        local_to_packed = align(positions, -1).cpu() if align is not None else positions
        layouts.append(MicrobatchLayout(local_to_packed=local_to_packed, samples=tuple(samples)))
    data_iterator.reset()
    return layouts


def _span_report(span_index, span, positions, sample_length) -> str:
    start, end = span
    inside = sum(start <= p < end for p in positions)
    after = sum(end <= p < end + _AFTER_SPAN_WINDOW for p in positions)
    rest = sum(p >= end + _AFTER_SPAN_WINDOW for p in positions)
    return f"img{span_index}[{start},{end}) in={inside} next{_AFTER_SPAN_WINDOW}={after} rest={rest}"


def describe_mismatches(layout: MicrobatchLayout, local_rows: list[int]) -> list[str]:
    """One line per sample that holds at least a few mismatched tokens."""
    packed = [int(layout.local_to_packed[row]) for row in local_rows if row < layout.local_to_packed.shape[0]]
    packed = [p for p in packed if p >= 0]
    lines = []
    for sample in layout.samples:
        positions = sorted(p - sample.offset for p in packed if sample.offset <= p < sample.offset + sample.length)
        if len(positions) < _MIN_SAMPLE_MISMATCHES:
            continue
        in_images = sum(any(s <= p < e for s, e in sample.image_spans) for p in positions)
        hit = [i for i, (s, e) in enumerate(sample.image_spans) if any(s <= p < e for p in positions)]
        spans = "; ".join(_span_report(i, sample.image_spans[i], positions, sample.length) for i in hit)
        refs = ", ".join(
            f"img{i}={sample.image_refs[i].rsplit('/', 1)[-1] if i < len(sample.image_refs) else '?'}"
            f"/px{sample.pixel_hashes[i] if i < len(sample.pixel_hashes) else '?'}"
            f"/patches{sample.pixel_patches[i] if i < len(sample.pixel_patches) else '?'}"
            for i in hit
        )
        lines.append(
            f"sample_index={sample.sample_index} rollout_id={sample.rollout_id} len={sample.length} "
            f"mismatch={len(positions)} first={positions[0]} last={positions[-1]} in_images={in_images} "
            f"text={len(positions) - in_images} spans={len(sample.image_spans)} refs={len(sample.image_refs)} "
            f"pixel_images={len(sample.pixel_hashes)} span_lengths={[e - s for s, e in sample.image_spans]} "
            f"patches={list(sample.pixel_patches)} hit=[{spans}] media=[{refs}]"
        )
    return lines
