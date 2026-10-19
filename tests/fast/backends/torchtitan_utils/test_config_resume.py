"""A ``--load`` without a torchtitan save is announced, not silently ignored."""

import logging

import pytest

pytest.importorskip("torchtitan")

from miles.backends.torchtitan_utils.config import resume_step_dir


def test_the_latest_actor_step_is_the_resume_point(tmp_path, caplog):
    for step in (1, 2, 10):
        (tmp_path / "torchtitan" / "actor" / "checkpoint" / f"step-{step}").mkdir(parents=True)
    with caplog.at_level(logging.WARNING):
        found = resume_step_dir(str(tmp_path), "actor", "/hf")
    assert found == str(tmp_path / "torchtitan" / "actor" / "checkpoint" / "step-10")
    assert caplog.text == ""


def test_a_load_without_an_actor_save_warns_and_starts_from_the_hf_weights(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        assert resume_step_dir(str(tmp_path), "actor", "/hf") is None
    assert "holds no torchtitan checkpoint" in caplog.text and str(tmp_path) in caplog.text and "/hf" in caplog.text


def test_the_reference_trainer_never_resumes_and_stays_quiet(tmp_path, caplog):
    (tmp_path / "torchtitan" / "actor" / "checkpoint" / "step-2").mkdir(parents=True)
    with caplog.at_level(logging.WARNING):
        assert resume_step_dir(str(tmp_path), "ref", "/hf") is None
        assert resume_step_dir(None, "actor", "/hf") is None
    assert caplog.text == ""
