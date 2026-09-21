"""Per-transition reward logging for mjlab/RSL-RL training.

``Episode_Reward/*`` is emitted only when an environment resets and contains a
full-episode accumulation divided by the maximum episode duration.  That is
useful for judging completed episodes, but it is not the reward batch used by a
PPO update.

This plugin adds ``Rollout_Reward/*`` values instead.  On every control step it
logs the mean over all parallel environments of each reward term's actual
contribution (reward function * weight * dt).  RSL-RL's logger then averages
those step means over the rollout.  With a constant number of environments this
is exactly::

  mean over [num_steps_per_env, num_envs]

For the current training setup that means all 24 * 1024 transitions, including
transitions from episodes that did not finish during the iteration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnvCfg
  from mjlab.managers.event_manager import EventTermCfg


class StepRewardLogger:
  """Publish reward means for every environment step to ``env.extras``.

  The plugin intentionally reads ``RewardManager._step_reward`` because mjlab
  currently has no public batched accessor for per-term step rewards.  The
  buffer is documented by ``RewardManager`` as ``raw_value * weight`` and is
  populated immediately before step-mode events run.  A compatibility check
  fails loudly if a future mjlab release changes that contract.
  """

  def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv) -> None:
    del cfg, env
    # mjlab constructs EventManager before RewardManager. Resolve the reward
    # term list lazily on the first step, by which time every manager exists.
    self._term_names: tuple[str, ...] | None = None

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None = None,
    *,
    namespace: str = "Rollout_Reward",
  ) -> None:
    del env_ids  # Step events always operate on every parallel environment.

    if self._term_names is None:
      self._term_names = tuple(env.reward_manager.active_terms)

    step_reward = getattr(env.reward_manager, "_step_reward", None)
    expected_shape = (env.num_envs, len(self._term_names))
    if (
      not isinstance(step_reward, torch.Tensor)
      or step_reward.shape != expected_shape
    ):
      actual_shape = getattr(step_reward, "shape", None)
      raise RuntimeError(
        "StepRewardLogger is incompatible with this mjlab RewardManager: "
        f"expected _step_reward shape {expected_shape}, got {actual_shape}."
      )

    # RewardManager._step_reward is always the unscaled reward rate
    # (reward_function * weight). Reapply the same scale used for reward_buf so
    # these values exactly match the per-transition contributions seen by PPO.
    scale = env.step_dt if env.cfg.scale_rewards_by_dt else 1.0
    contribution_means = step_reward.mean(dim=0) * scale

    # RSL-RL keeps references to each step's log dictionary until the iteration
    # ends. Copy it here so later steps cannot overwrite earlier step samples.
    step_log = dict(env.extras.get("log", {}))
    for term_name, value in zip(self._term_names, contribution_means, strict=True):
      step_log[f"{namespace}/{term_name}"] = value.detach().clone()

    # This is the exact environment reward handed to PPO for the current step,
    # before PPO's special time-limit bootstrap adjustment, averaged over envs.
    # Its rollout mean should equal the sum of the per-term means above.
    step_log[f"{namespace}/total"] = env.reward_buf.mean().detach().clone()
    env.extras["log"] = step_log


def enable_step_reward_logging(
  cfg: ManagerBasedRlEnvCfg,
  *,
  namespace: str = "Rollout_Reward",
  event_name: str = "log_step_reward_means",
) -> None:
  """Register :class:`StepRewardLogger` on an mjlab environment config."""

  # Import lazily so this reusable plugin can itself be imported while mjlab is
  # discovering the project's task entry point, without creating a cycle.
  from mjlab.managers.event_manager import EventTermCfg

  cfg.events[event_name] = EventTermCfg(
    func=StepRewardLogger,
    mode="step",
    params={"namespace": namespace},
  )
