"""Pick hooks for the session samples op — this is the customization point.

Point ``--session-sample-picker-path`` at any ``fn(leaf_samples,
session_metadata) -> list[Sample]``: a pure selection over the per-leaf raw
samples (drop or reorder, never rewrite). Public inputs are each sample's
``metadata["leaf"]`` descriptor and the structural tree tables in
``session_metadata``. Runs synchronously inside the session server process.
"""

from miles.rollout.session.v2.picker_hub.drop_retries import drop_retries
from miles.rollout.session.v2.picker_hub.drop_rolled_back_leaves import drop_rolled_back_leaves
from miles.rollout.session.v2.picker_hub.drop_same_prompt_retries import drop_same_prompt_retries

__all__ = ["drop_retries", "drop_rolled_back_leaves", "drop_same_prompt_retries"]
