"""The Muse Spark rollout check (``_validate_muse_spark_rollout_data``) on
session samples: PNG bytes instead of patch rows, and later screenshots
inside the response as untrained observation tokens."""

import pytest
import torch

from miles.backends.training_utils.data.multimodal import _validate_muse_spark_rollout_data


def _batch(positions, loss_mask, *, png=True):
    mm = {"mm_vision_grids": torch.tensor([[4, 4]]), "mm_vision_positions": torch.tensor(positions)}
    if png:
        mm["image_png"] = torch.zeros(3, dtype=torch.uint8)
    else:
        mm["mm_vision_patches"] = torch.zeros(16, 8)
    return {"tokens": [torch.arange(12)], "loss_masks": [loss_mask], "multimodal_train_inputs": [mm]}


def test_png_samples_with_images_in_untrained_response_rows_pass():
    # prompt 4 tokens, response 8: rows 6..9 hold an observation screenshot (loss_mask 0)
    _validate_muse_spark_rollout_data(_batch([6, 7, 8, 9], [1, 1, 0, 0, 0, 0, 1, 1]))
    _validate_muse_spark_rollout_data(_batch([0, 1, 2, 3], [1] * 8, png=False))


def test_an_image_token_that_is_trained_fails():
    with pytest.raises(AssertionError, match="loss_mask 1"):
        _validate_muse_spark_rollout_data(_batch([4, 5, 6, 7], [1, 1, 0, 0, 0, 0, 1, 1]))
