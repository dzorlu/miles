"""Images through the v2 session server end to end, with a stub engine:
the request's data URLs are stored once and referenced as ``mm://`` in the
session's records, the engine receives the client's data URLs with the
pre-tokenized ids (one placeholder per image), and the collected samples
carry the expanded prompt plus the images in the trainer's PNG format."""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest
import torch
from tests.fast.fixtures.session_fixtures import make_session_server_config
from tests.fast.rollout.session.test_media import FakeProcessor, _data_url, _png

from miles.rollout.session.media import MEDIA_URL_PREFIX
from miles.rollout.session.samples.codec import COMPUTED_FIELDS_V2, decode_samples_and_merge_input_sample
from miles.rollout.session.v2.core import SessionCoreV2
from miles.rollout.session.v2.session_state import SessionRegistryV2
from miles.utils.chat_template_utils import get_tito_tokenizer
from miles.utils.types import Sample

MODEL = "Qwen/Qwen3.8-27B"


@pytest.fixture(scope="module")
def tokenizer():
    """The Qwen3.8 tokenizer with the qwen38small fixed template installed, as
    the session server does through ``--chat-template-path``."""
    from transformers import AutoTokenizer

    from miles.utils.chat_template_utils.tito_tokenizer import TEMPLATE_DIR, Qwen38SmallTITOTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(MODEL)
    except Exception as e:  # noqa: BLE001 - no network or no cache
        pytest.skip(f"{MODEL} tokenizer unavailable: {e}")
    tok.chat_template = (TEMPLATE_DIR / Qwen38SmallTITOTokenizer.FIXED_TEMPLATE.template).read_text(encoding="utf-8")
    return tok


class ExactTokenEngine:
    """A chat backend that behaves like SGLang's exact-token path: it runs the
    request ids as sent, expands each image placeholder to the processor's
    count, and reports that prompt length."""

    def __init__(self, tokenizer, pad_id: int, counts: dict[str, int], reply: str):
        self.tokenizer = tokenizer
        self.pad_id = pad_id
        self.counts = counts  # data URL -> token count
        self.reply = reply
        self.bodies: list[dict] = []

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
        prompt_tokens = len(input_ids) - len(urls) + sum(self.counts[url] for url in urls)
        output_ids = self.tokenizer.encode(self.reply, add_special_tokens=False) + [
            self.tokenizer.convert_tokens_to_ids("<|im_end|>")
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
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": len(output_ids),
                        "output_token_logprobs": [[-0.1, tid] for tid in output_ids],
                    },
                }
            ],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(output_ids)},
        }
        return {
            "status_code": 200,
            "response_body": json.dumps(response),
            "headers": {"content-type": "application/json"},
        }


def _chat(core, session_id, messages):
    body = json.dumps({"model": "default", "messages": messages, "max_tokens": 8}).encode()
    return asyncio.run(core.chat_completions(session_id, method="POST", query="", headers={}, body=body))


def test_images_flow_through_the_v2_session_server(tokenizer):
    tito = get_tito_tokenizer(tokenizer, tokenizer_type="qwen38small")
    pad_id = tito.image_pad_id()
    assert pad_id == tokenizer.convert_tokens_to_ids("<|image_pad|>")
    first_url, second_url = _data_url(_png()), _data_url(_png(width=128, color=(5, 5, 5)))
    engine = ExactTokenEngine(tokenizer, pad_id, {first_url: 4, second_url: 8}, reply="ok")
    config = make_session_server_config(
        use_session_server="v2",
        hf_checkpoint=MODEL,
        tito_model="qwen38small",
        instance_id=uuid.uuid4().hex,
        session_sample_picker_path="miles.rollout.session.v2.picker_hub.drop_rolled_back_leaves",
        session_sample_postprocessor_path="miles.rollout.session.v2.postprocessor_hub.default_postprocess",
    )
    registry = SessionRegistryV2(tokenizer, tito_tokenizer=tito, processor=FakeProcessor())
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
    # The Qwen3.8 template requires text in a user turn, so a screenshot rides with its note.
    turn2 = turn1 + [
        assistant,
        {
            "role": "user",
            "content": [{"type": "text", "text": "next"}, {"type": "image_url", "image_url": {"url": second_url}}],
        },
    ]
    assert _chat(core, session_id, turn2).status_code == 200

    # The engine got one placeholder per image and the client's bytes; the
    # session kept each image once as a reference.
    assert [body["input_ids"].count(pad_id) for body in engine.bodies] == [1, 2]
    session = registry.get_session(session_id)
    assert len(session.media.entries) == 2
    stored = session.latest().path_messages()
    refs = [
        p["image_url"]["url"]
        for m in stored
        if isinstance(m.get("content"), list)
        for p in m["content"]
        if p.get("type") == "image_url"
    ]
    assert len(refs) == 2 and all(ref.startswith(MEDIA_URL_PREFIX) for ref in refs)
    assert all(
        ref.startswith(MEDIA_URL_PREFIX)
        for node in session.tree.nodes
        for ref in [
            p["image_url"]["url"]
            for m in node.record.request["messages"]
            if isinstance(m.get("content"), list)
            for p in m["content"]
            if p.get("type") == "image_url"
        ]
    )

    # The samples carry the expanded prompt and the images.
    reply = asyncio.run(core.collect_samples(session_id, max_seq_len=None))
    assert reply.status_code == 200, reply.body
    decoded = decode_samples_and_merge_input_sample(reply.body, Sample(), fields=COMPUTED_FIELDS_V2)
    (sample,) = decoded.samples
    wire = engine.bodies[1]["input_ids"]
    reply_len = len(tokenizer.encode("ok", add_special_tokens=False)) + 1  # plus <|im_end|>
    # Turn 2's wire prompt holds 2 placeholders; expanded to 4 + 8 tokens, plus the second reply.
    assert sample.tokens.count(pad_id) == 12 and len(sample.tokens) == len(wire) - 2 + 12 + reply_len
    mm = sample.multimodal_train_inputs
    assert isinstance(mm["image_png"], torch.Tensor)
    assert mm["image_grid_thw"].tolist() == [[1, 4, 4], [1, 4, 8]]
    assert mm["image_png_lengths"].sum().item() == mm["image_png"].numel()
    assert sample.loss_mask.count(1) == 2 * reply_len, "only the two sampled replies are trained"


def test_an_engine_that_retokenizes_is_refused_at_sample_time(tokenizer):
    tito = get_tito_tokenizer(tokenizer, tokenizer_type="qwen38small")
    url = _data_url(_png())
    engine = ExactTokenEngine(tokenizer, tito.image_pad_id(), {url: 3}, reply="ok")  # processor says 4
    config = make_session_server_config(
        use_session_server="v2",
        hf_checkpoint=MODEL,
        tito_model="qwen38small",
        instance_id=uuid.uuid4().hex,
        session_sample_picker_path="miles.rollout.session.v2.picker_hub.drop_rolled_back_leaves",
        session_sample_postprocessor_path="miles.rollout.session.v2.postprocessor_hub.default_postprocess",
    )
    registry = SessionRegistryV2(tokenizer, tito_tokenizer=tito, processor=FakeProcessor())
    core = SessionCoreV2(engine, registry, config, config.instance_id)
    session_id = registry.create_session()
    message = {
        "role": "user",
        "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": url}}],
    }
    assert _chat(core, session_id, [message]).status_code == 200
    reply = asyncio.run(core.collect_samples(session_id, max_seq_len=None))
    assert reply.status_code == 422 and b"did not run the request ids as sent" in reply.body


def test_a_text_only_server_refuses_images(tokenizer):
    tito = get_tito_tokenizer(tokenizer, tokenizer_type="qwen38small")
    config = make_session_server_config(
        use_session_server="v2",
        hf_checkpoint=MODEL,
        tito_model="qwen38small",
        instance_id=uuid.uuid4().hex,
        session_sample_picker_path="miles.rollout.session.v2.picker_hub.drop_rolled_back_leaves",
        session_sample_postprocessor_path="miles.rollout.session.v2.postprocessor_hub.default_postprocess",
    )
    registry = SessionRegistryV2(tokenizer, tito_tokenizer=tito, processor=None)
    engine = ExactTokenEngine(tokenizer, tito.image_pad_id(), {}, reply="ok")
    core = SessionCoreV2(engine, registry, config, config.instance_id)
    session_id = registry.create_session()
    from miles.rollout.session.errors import MessageValidationError

    with pytest.raises(MessageValidationError, match="cannot carry images"):
        _chat(
            core,
            session_id,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {"type": "image_url", "image_url": {"url": _data_url(_png())}},
                    ],
                }
            ],
        )
