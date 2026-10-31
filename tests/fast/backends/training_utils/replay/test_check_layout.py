import torch

from miles.backends.training_utils.replay.check_layout import build_layouts, describe_mismatches

IMAGE = 9


class _Iterator:
    def __init__(self, batches):
        self.batches = batches
        self.offset = 0

    def reset(self):
        self.offset = 0

    def get_next(self, keys):
        batch = self.batches[self.offset]
        self.offset += 1
        return {key: batch.get(key) for key in keys}


def _layout():
    first = torch.tensor([1, 2, IMAGE, IMAGE, IMAGE, 3, 4, 5])
    second = torch.tensor([6, IMAGE, IMAGE, 7, 8, 8, 8, 8, 8, 8])
    pixels = {"pixel_values": torch.arange(24.0).reshape(6, 4), "image_grid_thw": torch.tensor([[1, 2, 3]])}
    batch = {
        "tokens": [first, second],
        "sample_indices": [11, 12],
        "rollout_ids": [3, 4],
        "multimodal_media_refs": [{"images": ["/m/aaa.png"]}, {"images": ["/m/bbb.png"]}],
        "multimodal_train_inputs": [pixels, pixels],
    }
    # the packed microbatch is 18 tokens, padded to 20; reversed stands in for a CP shard's row order
    return build_layouts(
        data_iterator=_Iterator([batch]),
        num_microbatches=[1],
        entry_lengths=[20],
        image_token_id=IMAGE,
        align=lambda positions, pad: positions.flip(0),
    )[0]


def test_layout_maps_rows_and_finds_image_spans():
    layout = _layout()
    assert layout.local_to_packed.tolist() == list(range(19, -1, -1))
    assert [s.offset for s in layout.samples] == [0, 8]
    assert layout.samples[0].image_spans == ((2, 5),)
    assert layout.samples[1].image_spans == ((1, 3),)
    assert layout.samples[0].pixel_patches == (6,)


def test_mismatches_are_reported_against_their_sample_and_image():
    layout = _layout()
    # packed positions 9, 10 (second sample's image) and 12, 13 (its text after the image)
    rows = [19 - p for p in (9, 10, 12, 13)]
    (line,) = describe_mismatches(layout, rows)
    assert "sample_index=12" in line
    assert "mismatch=4 first=1 last=5 in_images=2 text=2" in line
    assert "img0[1,3) in=2 next512=2 rest=0" in line
    assert "img0=bbb.png" in line


def test_samples_with_few_mismatches_are_not_reported():
    assert describe_mismatches(_layout(), [19 - 9]) == []


def test_ref_labels_cover_paths_data_urls_and_image_objects():
    from miles.backends.training_utils.replay.check_layout import _ref_label

    class Image:
        pass

    assert _ref_label("/m/aaa.png") == "aaa.png"
    assert _ref_label("data:image/png;base64,AAAA").startswith("data:")
    assert len(_ref_label("data:image/png;base64,AAAA")) == len("data:") + 12
    assert _ref_label(Image()) == "Image"
