from types import SimpleNamespace

import pytest

from miles.ray.rollout.rollout_executor import _check_fully_async_rollout_fn
from miles.rollout.fully_async_rollout import FullyAsyncRolloutFn


class _Subclass(FullyAsyncRolloutFn):
    def __init__(self):  # the check reads only the type; skip the producer setup
        pass


def test_a_fully_async_subclass_is_accepted():
    args = SimpleNamespace(fully_async=True, rollout_function_path="pkg.Subclass")

    _check_fully_async_rollout_fn(args, _Subclass())


def test_fully_async_refuses_a_class_that_is_not_a_subclass():
    """The async driver relies on FullyAsyncRolloutFn's producer, so another class must fail at load."""
    args = SimpleNamespace(fully_async=True, rollout_function_path="pkg.Other")

    with pytest.raises(TypeError, match="FullyAsyncRolloutFn"):
        _check_fully_async_rollout_fn(args, object())


def test_a_run_that_is_not_fully_async_is_not_checked():
    args = SimpleNamespace(fully_async=False, rollout_function_path="pkg.Other")

    _check_fully_async_rollout_fn(args, object())
