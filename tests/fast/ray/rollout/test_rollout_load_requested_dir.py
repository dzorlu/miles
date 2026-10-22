"""The rollout state resumes from the directory the user asked to load.

resolve_args_checkpoint_load points --load at the reference weights when the run directory holds no
Megatron checkpoint, which is the case for a native LoRA run (it saves adapter shards only). The rollout
state of that run still lives under the requested directory, kept in args.requested_load.
"""

import asyncio
from argparse import Namespace
from pathlib import Path

import pytest

from miles.ray.rollout.output_snapshotter import _RolloutExecutorOutputSnapshotter
from miles.ray.rollout.rollout_executor import RolloutExecutor
from miles.utils.simple_checkpointer import load_simple_checkpoint, save_simple_checkpoint


class _DataSource:
    def __init__(self, position: int) -> None:
        self.position = position

    def save(self, directory: Path) -> None:
        save_simple_checkpoint(directory=directory, data=self.position)

    def load(self, directory: Path) -> None:
        self.position = load_simple_checkpoint(directory=directory)


class _Executor(RolloutExecutor):
    def __init__(self, *, save: str, load: str, requested_load: str | None, position: int) -> None:
        self.args = Namespace(
            save=save,
            load=load,
            requested_load=requested_load,
            save_debug_event_data=None,
            ci_test=False,
            ci_inject_missing_prefetched_batch_bug=False,
        )
        self.use_legacy_rollout_v1 = True
        self.data_source = _DataSource(position)
        self._output_snapshotter = _RolloutExecutorOutputSnapshotter(args=self.args)


def test_the_rollout_state_loads_from_the_requested_directory(tmp_path):
    run, ref = tmp_path / "run", tmp_path / "ref"
    ref.mkdir()
    asyncio.run(_Executor(save=str(run), load=str(run), requested_load=str(run), position=24).save(2))

    resumed = _Executor(save=str(tmp_path / "next"), load=str(ref), requested_load=str(run), position=0)
    asyncio.run(resumed.load(2))

    assert resumed.data_source.position == 24


def test_without_a_requested_directory_the_load_reads_load(tmp_path):
    run, ref = tmp_path / "run", tmp_path / "ref"
    ref.mkdir()
    asyncio.run(_Executor(save=str(run), load=str(run), requested_load=None, position=24).save(2))

    resumed = _Executor(save=str(tmp_path / "next"), load=str(ref), requested_load=None, position=0)
    with pytest.raises(AssertionError, match="does not exist"):
        asyncio.run(resumed.load(2))
