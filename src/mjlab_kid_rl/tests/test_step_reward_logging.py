"""Tests for rollout-wide per-transition reward logging."""

import types
import unittest

import torch

from mjlab_kid_rl.plugins.step_reward_logging import StepRewardLogger
from mjlab_kid_rl.tasks.kid_RL_env_cfg import make_kid_rl_velocity_env_cfg


class TestStepRewardLogger(unittest.TestCase):
  def _env(self) -> types.SimpleNamespace:
    reward_manager = types.SimpleNamespace(
      active_terms=("positive", "penalty"),
      # Unscaled reward rates: reward function * configured weight.
      _step_reward=torch.tensor([[1.0, -2.0], [3.0, -4.0]]),
    )
    return types.SimpleNamespace(
      num_envs=2,
      step_dt=0.02,
      cfg=types.SimpleNamespace(scale_rewards_by_dt=True),
      reward_manager=reward_manager,
      reward_buf=torch.tensor([-0.02, -0.02]),
      extras={"log": {"Existing/value": torch.tensor(7.0)}},
    )

  def test_constructor_does_not_require_reward_manager_yet(self) -> None:
    # mjlab builds EventManager before it creates RewardManager.
    construction_env = types.SimpleNamespace()
    logger = StepRewardLogger(types.SimpleNamespace(), construction_env)

    runtime_env = self._env()
    logger(runtime_env)
    self.assertIn("Rollout_Reward/total", runtime_env.extras["log"])

  def test_logs_actual_dt_scaled_transition_means(self) -> None:
    env = self._env()
    logger = StepRewardLogger(types.SimpleNamespace(), env)

    logger(env)

    self.assertEqual(env.extras["log"]["Existing/value"].item(), 7.0)
    self.assertAlmostEqual(env.extras["log"]["Rollout_Reward/positive"].item(), 0.04)
    self.assertAlmostEqual(env.extras["log"]["Rollout_Reward/penalty"].item(), -0.06)
    self.assertAlmostEqual(env.extras["log"]["Rollout_Reward/total"].item(), -0.02)

  def test_each_step_gets_an_independent_log_snapshot(self) -> None:
    env = self._env()
    logger = StepRewardLogger(types.SimpleNamespace(), env)

    logger(env)
    first_step_log = env.extras["log"]
    env.reward_manager._step_reward[:] = 10.0
    env.reward_buf[:] = 0.4
    logger(env)

    self.assertIsNot(first_step_log, env.extras["log"])
    self.assertAlmostEqual(first_step_log["Rollout_Reward/positive"].item(), 0.04)
    self.assertAlmostEqual(env.extras["log"]["Rollout_Reward/positive"].item(), 0.2)

  def test_velocity_training_config_enables_plugin_but_play_does_not(self) -> None:
    train_cfg = make_kid_rl_velocity_env_cfg()
    play_cfg = make_kid_rl_velocity_env_cfg(play=True)

    self.assertIn("log_step_reward_means", train_cfg.events)
    self.assertNotIn("log_step_reward_means", play_cfg.events)


if __name__ == "__main__":
  unittest.main()
