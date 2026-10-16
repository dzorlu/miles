"""Screenshots of a Muse Spark session through the v2 session server, with the
real Muse Spark tokenizer and train processor and a stub engine.

Muse Spark's fixed template writes ``<image>`` (three text tokens) per image;
the session renders it as ``<|image_start|><|patch|><|image_end|>``, sends the
engine the expanded ids (the engine keeps an expanded block as it is and would
read a single ``<|patch|>`` as a 1-token image), and the sample holds exactly
those ids. The reference for what the model must see is the Muse Spark train
processor (``MuseSparkTrainProcessor``) on the same messages.

``MILES_TEST_MUSE_SPARK_DIR`` names a directory with the checkpoint's
config.json and tokenizer files (no weights needed).
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import uuid

import pytest
import torch
from PIL import Image
from tests.fast.fixtures.session_fixtures import make_session_server_config
from tests.fast.rollout.session.test_media import _data_url

from miles.rollout.session.samples.codec import COMPUTED_FIELDS_V2, decode_samples_and_merge_input_sample
from miles.rollout.session.v2.core import SessionCoreV2
from miles.rollout.session.v2.session_state import SessionRegistryV2
from miles.utils.chat_template_utils import get_tito_tokenizer
from miles.utils.types import Sample

MUSE_DIR = os.environ.get("MILES_TEST_MUSE_SPARK_DIR")


@pytest.fixture(scope="module")
def muse():
    """The Muse Spark tokenizer with the fixed template installed (as ``--chat-template-path`` does)
    and the train processor built on it."""
    if not MUSE_DIR or not os.path.isfile(os.path.join(MUSE_DIR, "config.json")):
        pytest.skip("MILES_TEST_MUSE_SPARK_DIR (config.json + tokenizer files) not set")
    from transformers import AutoTokenizer

    from miles.utils.chat_template_utils.tito_tokenizer import TEMPLATE_DIR, MuseSparkTITOTokenizer
    from miles_plugins.models.muse_spark.vision.processor import MuseSparkTrainProcessor

    tokenizer = AutoTokenizer.from_pretrained(MUSE_DIR)
    tokenizer.chat_template = (TEMPLATE_DIR / MuseSparkTITOTokenizer.FIXED_TEMPLATE.template).read_text(
        encoding="utf-8"
    )
    return tokenizer, MuseSparkTrainProcessor(MUSE_DIR, tokenizer=tokenizer)


class MuseEngine:
    """A chat backend that runs the request ids as sent, as the Muse Spark
    engine does with expanded images: each image must arrive as
    ``<|image_start|>`` + ``<|patch|>`` * count + ``<|image_end|>``. When the
    sglang Muse Spark processor is importable, its own id normalization must
    leave the ids unchanged."""

    def __init__(self, tokenizer, counts: dict[str, int], reply: str):
        self.tokenizer = tokenizer
        self.counts = counts  # data URL -> token count
        self.reply = reply
        self.start, self.patch, self.end = (
            tokenizer.convert_tokens_to_ids(t) for t in ("<|image_start|>", "<|patch|>", "<|image_end|>")
        )
        self.bodies: list[dict] = []

    def _check_images(self, input_ids: list[int], urls: list[str]) -> None:
        starts = [i for i, t in enumerate(input_ids) if t == self.start]
        assert len(starts) == len(urls), f"{len(starts)} image blocks for {len(urls)} images"
        for position, url in zip(starts, urls, strict=True):
            count = self.counts[url]
            assert input_ids[position + 1 : position + 1 + count] == [self.patch] * count
            assert input_ids[position + 1 + count] == self.end, "an image block must hold exactly its token count"
        try:
            from sglang.srt.multimodal.processors.muse_spark_omni import _replace_unexpanded_hf_markers
        except Exception:  # noqa: BLE001 - sglang without the Muse Spark processor
            return
        # the engine's own normalization of client ids: a lone <|patch|> would become "<image>" again
        normalized = _replace_unexpanded_hf_markers(input_ids, {self.patch: [-1]}, {self.start: self.end})
        assert normalized == input_ids, "the engine would rewrite the ids"

    async def do_proxy(self, request, path, *, body, headers):
        payload = json.loads(body)
        self.bodies.append(payload)
        input_ids = payload["input_ids"]
        urls = [
            part["image_url"]["url"]
            for message in payload["messages"]
            if isinstance(message.get("content"), list)
            for part in message["content"]
            if part.get("type") == "image_url"
        ]
        assert all(url.startswith("data:") for url in urls), "the engine must receive data URLs"
        self._check_images(input_ids, urls)
        output_ids = self.tokenizer.encode(self.reply, add_special_tokens=False) + [
            self.tokenizer.convert_tokens_to_ids("<|eot|>")
        ]
        response = {
            "id": f"stub-{len(self.bodies)}",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": self.reply},
                    "finish_reason": "stop",
                    "meta_info": {
                        "prompt_tokens": len(input_ids),
                        "completion_tokens": len(output_ids),
                        "output_token_logprobs": [[-0.1, tid] for tid in output_ids],
                    },
                }
            ],
            "usage": {"prompt_tokens": len(input_ids), "completion_tokens": len(output_ids)},
        }
        return {
            "status_code": 200,
            "response_body": json.dumps(response),
            "headers": {"content-type": "application/json"},
        }


def _chat(core, session_id, messages):
    body = json.dumps({"model": "default", "messages": messages, "max_tokens": 8}).encode()
    return asyncio.run(core.chat_completions(session_id, method="POST", query="", headers={}, body=body))


def _screenshot(width: int, height: int, color) -> bytes:
    """A screenshot-sized PNG with vertical lines, so the patches are not constant."""
    image = Image.new("RGB", (width, height), color)
    for x in range(0, width, 37):
        for y in range(height):
            image.putpixel((x, y), (255, 255, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_text_rendering_is_unchanged(muse):
    """Without images the family renders exactly as the HF tokenizer does; an
    "<image>" typed as text stays text (the template escapes it)."""
    tokenizer, _ = muse
    tito = get_tito_tokenizer(tokenizer, tokenizer_type="muse_spark")
    messages = [{"role": "user", "content": "write <image> as text"}, {"role": "assistant", "content": "ok"}]
    expected = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=False)
    assert tito.apply_chat_template(messages, add_generation_prompt=False, tokenize=True) == list(expected)
    assert tokenizer.convert_tokens_to_ids("<|patch|>") not in expected


def test_screenshot_ids_round_trip_through_a_muse_spark_session(muse):
    tokenizer, processor = muse
    tito = get_tito_tokenizer(tokenizer, tokenizer_type="muse_spark")
    ids = tito.image_token_ids()
    start_id, patch_id, end_id = ids["begin"], ids["placeholder"], ids["end"]
    assert patch_id == tokenizer.convert_tokens_to_ids("<|patch|>")

    first_png, second_png = _screenshot(1024, 576, (10, 20, 30)), _screenshot(1280, 720, (40, 50, 60))
    first_url, second_url = _data_url(first_png), _data_url(second_png)
    images = {
        url: Image.open(io.BytesIO(png)).convert("RGB")
        for url, png in ((first_url, first_png), (second_url, second_png))
    }
    counts = {url: processor.image_patches(image)[2] for url, image in images.items()}
    engine = MuseEngine(tokenizer, counts, reply="ok")
    config = make_session_server_config(
        use_session_server="v2",
        hf_checkpoint=MUSE_DIR,
        tito_model="muse_spark",
        instance_id=uuid.uuid4().hex,
        session_sample_picker_path="miles.rollout.session.v2.picker_hub.drop_rolled_back_leaves",
        session_sample_postprocessor_path="miles.rollout.session.v2.postprocessor_hub.default_postprocess",
    )
    registry = SessionRegistryV2(tokenizer, tito_tokenizer=tito, processor=processor)
    core = SessionCoreV2(engine, registry, config, config.instance_id)
    session_id = registry.create_session()

    turn1 = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": first_url}}],
        }
    ]
    response = _chat(core, session_id, turn1)
    assert response.status_code == 200, response.body
    assistant = json.loads(response.body)["choices"][0]["message"]
    turn2 = turn1 + [
        assistant,
        {
            "role": "user",
            "content": [{"type": "text", "text": "next"}, {"type": "image_url", "image_url": {"url": second_url}}],
        },
    ]
    assert _chat(core, session_id, turn2).status_code == 200

    # Rendered (stored) ids: one start/patch/end per image.
    session = registry.get_session(session_id)
    stored = session.latest().token_ids
    assert stored.count(start_id) == 2 and stored.count(patch_id) == 2 and stored.count(end_id) == 2
    for position in [i for i, t in enumerate(stored) if t == patch_id]:
        assert stored[position - 1] == start_id and stored[position + 1] == end_id

    # Wire ids, turn 1: what the Muse Spark train processor makes of the same messages.
    rendered = tokenizer.apply_chat_template(turn1, tokenize=False, add_generation_prompt=True)
    reference = list(processor(text=rendered, images=[images[first_url]])["input_ids"][0])
    wires = [body["input_ids"] for body in engine.bodies]
    assert wires[0] == reference
    assert [wire.count(patch_id) for wire in wires] == [counts[first_url], counts[first_url] + counts[second_url]]

    # Sample ids: the last wire prompt plus the reply, unchanged; turn 1's prompt is a prefix.
    reply = asyncio.run(core.collect_samples(session_id, max_seq_len=None))
    assert reply.status_code == 200, reply.body
    decoded = decode_samples_and_merge_input_sample(reply.body, Sample(), fields=COMPUTED_FIELDS_V2)
    (sample,) = decoded.samples
    reply_len = len(tokenizer.encode("ok", add_special_tokens=False)) + 1  # plus <|eot|>
    assert sample.tokens[: len(wires[1])] == wires[1]
    assert len(sample.tokens) == len(wires[1]) + reply_len
    assert sample.tokens[: len(wires[0])] == wires[0]

    # Trainer inputs: grids and positions of the expanded ids; the PNG rebuild gives the processor's patches.
    from miles.backends.training_utils.data.png_images import vision_patches_from_pngs

    mm = sample.multimodal_train_inputs
    grids = [list(processor.image_patches(images[url])[1]) for url in (first_url, second_url)]
    assert mm["mm_vision_grids"].tolist() == grids
    assert mm["mm_vision_positions"].tolist() == [i for i, t in enumerate(sample.tokens) if t == patch_id]
    assert "image_grid_thw" not in mm
    rebuilt = vision_patches_from_pngs(processor, mm["image_png"], mm["image_png_lengths"], mm["mm_vision_grids"])
    expected = torch.cat([processor.image_patches(images[url])[0] for url in (first_url, second_url)])
    assert torch.equal(rebuilt, expected)
    assert sample.loss_mask.count(1) == 2 * reply_len, "only the two sampled replies are trained"
