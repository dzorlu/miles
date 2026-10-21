"""--lora-sync-merged: argument validation and the rollout-facing LoRA predicates: no GPU."""

from argparse import Namespace

import pytest

from miles.backends.megatron_utils.lora.native.registry import MUSE_SPARK_LORA_TARGETS
from miles.utils.lora import arguments as lora_arguments
from miles.utils.lora.utils import lora_adapter_memory_layout, lora_rollout_enabled

RANK = 4


class _HfConfig:
    def __init__(self, model_type):
        self.model_type = model_type

    def to_dict(self):
        return {"model_type": self.model_type}


def _lora_args(**overrides):
    values = dict(
        lora_rank=RANK,
        lora_adapter_path=None,
        lora_train_only=False,
        lora_sync_merged=True,
        megatron_to_hf_mode="raw",
        target_modules=None,
        exclude_modules=None,
        lora_type="lora",
        experts_shared_outer_loras=False,
        multi_lora_n_adapters=0,
        train_backend="megatron",
        hf_checkpoint="unused",
    )
    values.update(overrides)
    return Namespace(**values)


@pytest.fixture
def hf_model_type(monkeypatch):
    def use(model_type):
        monkeypatch.setattr(lora_arguments, "load_hf_config", lambda path: _HfConfig(model_type))

    use("muse_spark")
    return use


def test_merged_sync_validates_to_the_fixed_muse_spark_targets(hf_model_type):
    args = _lora_args()
    lora_arguments.validate_lora_args(args)
    assert args.hf_lora_targets == args.lora_adapter_targets == list(MUSE_SPARK_LORA_TARGETS)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"lora_train_only": True}, "keeps them frozen"),
        ({"target_modules": "attn"}, "fixed set"),
        ({"exclude_modules": "o_proj"}, "fixed set"),
        ({"megatron_to_hf_mode": "bridge"}, "raw Megatron"),
        ({"lora_type": "canonical_lora"}, "canonical_lora"),
        ({"lora_sync_merged": False}, "serves no LoRA"),
    ],
)
def test_merged_sync_refuses_conflicting_options(hf_model_type, override, message):
    with pytest.raises(AssertionError, match=message):
        lora_arguments.validate_lora_args(_lora_args(**override))


def test_merged_sync_is_muse_spark_only(hf_model_type):
    hf_model_type("kimi_k3")
    with pytest.raises(AssertionError, match="implemented for"):
        lora_arguments.validate_lora_args(_lora_args())


def test_merged_sync_serves_no_adapter_but_keeps_the_lora_memory_layout():
    merged = _lora_args()
    assert not lora_rollout_enabled(merged) and lora_adapter_memory_layout(merged)
    served = _lora_args(lora_sync_merged=False)
    assert lora_rollout_enabled(served) and lora_adapter_memory_layout(served)
    train_only = _lora_args(lora_sync_merged=False, lora_train_only=True)
    assert not lora_rollout_enabled(train_only) and not lora_adapter_memory_layout(train_only)
    assert not lora_adapter_memory_layout(_lora_args(lora_rank=0))


def test_the_flag_parses_off_by_default():
    import argparse

    parser = lora_arguments.add_lora_arguments(argparse.ArgumentParser())
    assert parser.parse_args([]).lora_sync_merged is False
    assert parser.parse_args(["--lora-sync-merged"]).lora_sync_merged is True
