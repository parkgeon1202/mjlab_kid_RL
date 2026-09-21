"""Kid_RL_v3 humanoid, 4-axis-arm revision ("dance").

Same legs/pelvis linkage as `kid_rl`, but the upper body is re-authored: each arm
gains a shoulder yaw and a wrist pitch, and a 2-DoF neck/head is added (19 -> 25
actuated joints).
"""

from mjlab_kid_rl.robot.kid_rl_dance.kid_rl_dance_constants import (
  KID_RL_DANCE_ARTICULATION as KID_RL_DANCE_ARTICULATION,
)
from mjlab_kid_rl.robot.kid_rl_dance.kid_rl_dance_constants import (
  get_kid_rl_dance_robot_cfg as get_kid_rl_dance_robot_cfg,
)
