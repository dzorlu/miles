import torch
from transformers.models.glm_ocr.configuration_glm_ocr import GlmOcrVisionConfig

from miles_plugins.models.glm5_next.vision import Glm5NextVisionModel


def _tiny_tower() -> Glm5NextVisionModel:
    config = GlmOcrVisionConfig(
        depth=2,
        hidden_size=32,
        out_hidden_size=24,
        intermediate_size=48,
        num_heads=4,
        patch_size=14,
        temporal_patch_size=2,
        spatial_merge_size=2,
        in_channels=3,
    )
    config.swiglu_limit = 7.0
    config.projection_intermediate_size = 40
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return Glm5NextVisionModel(config).eval()


def test_encoding_a_subset_of_images_gives_their_rows_of_encoding_all():
    """Each CP rank encodes only its own screenshots; that is exact only if images do not mix in the tower."""
    tower = _tiny_tower()
    grid = torch.tensor([[1, 4, 6], [1, 2, 4], [2, 4, 4]])
    patches = grid.prod(dim=-1)
    pixel_values = torch.randn(int(patches.sum()), 3 * 2 * 14 * 14)
    with torch.no_grad():
        everything = tower(pixel_values, grid_thw=grid).pooler_output
        subset = tower(
            torch.cat([pixel_values[: patches[0]], pixel_values[patches[0] + patches[1] :]]), grid_thw=grid[[0, 2]]
        ).pooler_output
    rows = patches // 4
    assert everything.shape == (int(rows.sum()), 24)
    expected = torch.cat([everything[: rows[0]], everything[rows[0] + rows[1] :]])
    torch.testing.assert_close(subset, expected)
