"""Reusable extensions that plug into mjlab without patching mjlab itself."""

from mjlab_kid_rl.plugins.step_reward_logging import (
  StepRewardLogger,
  enable_step_reward_logging,
)

__all__ = ["StepRewardLogger", "enable_step_reward_logging"]
