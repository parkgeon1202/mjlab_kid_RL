"""BAM actuator whose command delay stays in range and never reorders commands.

Two fixes on top of mjlab's ``DelayBuffer``, both local to this project (mjlab is
reinstalled from PyPI and bam is shared with the microban project):

1. **In range after a reset.** ``DelayBuffer.reset()`` sets the lag of the reset
   environments to 0 and only draws a real one on the next "update turn". With
   ``delay_per_env_phase = True`` that turn can be up to ``delay_update_period - 1``
   steps away, and with ``delay_hold_prob > 0`` the 0 is kept with that probability
   even on a turn. So every episode started with a stretch of *zero* command delay,
   below ``delay_min_lag`` (about 2% of physics steps with min 2 / max 5, period 4,
   hold 0.8, 4096 envs). Here reset (and construction) draws a lag in range at once.

2. **No command reversal.** The command a motor receives at physics step ``s`` is the
   one issued at ``s - lag``. Between two steps that index goes *backwards* whenever
   the lag grows by 2 or more (e.g. 2 -> 5 gives ``s-3`` then ``s-5``), so a motor
   that already got the new policy command briefly receives the previous one again.
   A real servo bus never does that: a slower bus delivers the next command later,
   it does not resend an older one. Here the sampled lag is kept as a *target* and
   the lag actually applied rises by at most 1 per physics step towards it (falling
   is immediate). A +1 step only repeats the current command, so the index never goes
   back. Keeping the target separate -- instead of clipping the draw -- keeps the lag
   distribution from drifting low: a draw of 5 still reaches 5, three steps later.

Sampling itself -- per-env lags, update period, per-env phase, hold probability --
is mjlab's, unchanged: the parent class samples and holds the target exactly as it
would sample and hold the applied lag.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from bam.mjlab import BamActuator, BamActuatorCfg
from mjlab.utils.buffers import DelayBuffer


class CommandDelayBuffer(DelayBuffer):
  """``DelayBuffer`` for motor commands: in-range after reset, never reorders."""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    fresh = self._draw()
    self._current_lags[:] = fresh
    self._target_lags = fresh.clone()

  def _draw(self) -> torch.Tensor:
    return torch.randint(
      self.min_lag,
      self.max_lag + 1,
      (self.batch_size,),
      dtype=torch.long,
      device=self.device,
      generator=self.generator,
    )

  def reset(self, batch_ids=None) -> None:
    super().reset(batch_ids)
    fresh = self._draw()
    if batch_ids is None:
      self._current_lags[:] = fresh
      self._target_lags[:] = fresh
    else:
      self._current_lags[batch_ids] = fresh[batch_ids]
      self._target_lags[batch_ids] = fresh[batch_ids]

  def _update_lags(self) -> None:
    # Let the parent sample/hold the *target* (its logic reads and writes
    # self._current_lags), then derive the lag actually applied from it.
    applied = self._current_lags
    self._current_lags = self._target_lags
    super()._update_lags()
    self._target_lags = self._current_lags
    self._current_lags = torch.minimum(self._target_lags, applied + 1)


class KidBamActuator(BamActuator):
  """``BamActuator`` using :class:`CommandDelayBuffer` for its command delay."""

  def _init_delay_buffer(self, num_envs: int, device: str) -> None:
    super()._init_delay_buffer(num_envs, device)
    if self._delay_buffer is None:
      return
    self._delay_buffer = CommandDelayBuffer(
      min_lag=self.cfg.delay_min_lag,
      max_lag=self.cfg.delay_max_lag,
      batch_size=num_envs,
      device=device,
      hold_prob=self.cfg.delay_hold_prob,
      update_period=self.cfg.delay_update_period,
      per_env_phase=self.cfg.delay_per_env_phase,
    )


@dataclass(kw_only=True)
class KidBamActuatorCfg(BamActuatorCfg):
  """``BamActuatorCfg`` that builds a :class:`KidBamActuator`."""

  def build(self, entity, target_ids, target_names) -> KidBamActuator:
    return KidBamActuator(self, entity, target_ids, target_names)
