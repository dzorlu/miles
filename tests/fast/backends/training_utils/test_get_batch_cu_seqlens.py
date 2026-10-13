"""`get_batch`'s host copy of `cu_seqlens` must describe the same packed stream as the device tensor."""

from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.data import context_parallel
from miles.backends.training_utils.data import rollout as data_utils


@pytest.mark.parametrize("allgather_cp", [False, True])
@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_the_host_cu_seqlens_match_the_device_tensor(
    monkeypatch: pytest.MonkeyPatch, allgather_cp: bool, cp_size: int
):
    """The DSv4 row balancer plans from the host copy to avoid a device sync, so a drift would misroute rows."""
    lengths = [13, 11, 5, 1]
    rollout = {
        "tokens": [torch.arange(1, n + 1) for n in lengths],
        "loss_masks": [torch.ones(n // 2, dtype=torch.int) for n in lengths],
        "total_lengths": lengths,
        "response_lengths": [n // 2 for n in lengths],
    }
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    for cp_rank in range(cp_size):
        state = SimpleNamespace(cp=SimpleNamespace(rank=cp_rank, size=cp_size), tp=SimpleNamespace(rank=0, size=1))
        monkeypatch.setattr(data_utils, "get_parallel_state", lambda state=state: state)
        monkeypatch.setattr(context_parallel, "get_parallel_state", lambda state=state: state)
        batch = data_utils.get_batch(
            data_utils.DataIterator(rollout, micro_batch_size=len(lengths)),
            list(rollout),
            pad_multiplier=7,
            qkv_format="thd",
            allgather_cp=allgather_cp,
        )
        assert batch["cu_seqlens_host"] == tuple(batch["cu_seqlens"].tolist())


def test_get_batch_rebuilds_pngs_only_when_the_batch_moves_to_the_device(monkeypatch: pytest.MonkeyPatch):
    """A model that keeps multimodal inputs on the host (torchtitan's image
    encoder) rebuilds the PNG bytes itself; the others get pixel_values here."""
    from miles.backends.training_utils.data import png_images

    calls = []
    monkeypatch.setattr(png_images, "rebuild_png_images", lambda mm: (calls.append(sorted(mm)), {**mm, "pixel_values": torch.zeros(1, 3)})[1])
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    state = SimpleNamespace(cp=SimpleNamespace(rank=0, size=1), tp=SimpleNamespace(rank=0, size=1))
    monkeypatch.setattr(data_utils, "get_parallel_state", lambda: state)
    monkeypatch.setattr(context_parallel, "get_parallel_state", lambda: state)
    rollout = {
        "tokens": [torch.arange(1, 4)],
        "loss_masks": [torch.ones(1, dtype=torch.int)],
        "total_lengths": [3],
        "response_lengths": [1],
        "multimodal_train_inputs": [{"image_png": torch.zeros(4, dtype=torch.uint8), "image_png_lengths": torch.tensor([4]), "image_grid_thw": torch.tensor([[1, 1, 1]])}],
    }
    keys = ["tokens", "loss_masks", "total_lengths", "response_lengths", "multimodal_train_inputs"]
    on_host = data_utils.get_batch(data_utils.DataIterator(rollout, micro_batch_size=1), keys, multimodal_on_host=True)
    assert calls == [] and "image_png" in on_host["multimodal_train_inputs"]
    moved = data_utils.get_batch(data_utils.DataIterator(rollout, micro_batch_size=1), keys, multimodal_on_host=False)
    assert calls == [["image_grid_thw", "image_png", "image_png_lengths"]] and "pixel_values" in moved["multimodal_train_inputs"]
