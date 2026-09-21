"""Single switch between the baseline domain-randomization set and the widened one.

Every randomization that has two candidate ranges reads ``DR_WIDE`` and picks one
of them inline, so both values stay visible side by side in the config and cannot
drift out of sync with a stale comment. The two call sites are
``tasks/kid_RL_env_cfg.py`` (push, base COM, observation delay, body/torso mass,
armature, joint friction) and ``robot/kid_rl_dance/kid_rl_dance_constants.py``
(BAM motor command delay).

Flip it by editing ``_DEFAULT`` below, or without touching the file at all by
setting the environment variable -- useful when the training host is reached over
rsync and you would rather not re-sync the source to change one bool::

    KID_RL_DR_WIDE=1 python -m mjlab.scripts.train ...

Note that the motor-delay constant is shared by the velocity task and the
tracking task, so this switch moves both.
"""

import os

# Baseline (False) or widened (True) randomization ranges.
_DEFAULT = False

_TRUTHY = {"1", "true", "yes", "on"}

DR_WIDE: bool = (
  os.environ["KID_RL_DR_WIDE"].strip().lower() in _TRUTHY
  if os.environ.get("KID_RL_DR_WIDE")
  else _DEFAULT
)
