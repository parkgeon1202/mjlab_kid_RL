from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from mjlab_kid_rl.tasks.kid_RL_env_cfg import (
  KID_RL_IRC_HURDLE_RL_CFG,
  KID_RL_IRC_VELOCITY_RL_CFG,
  make_kid_rl_hurdle_env_cfg,
  make_kid_rl_velocity_env_cfg,
)

# IRC robot (kid_RL_irc.xml): kid_RL legs with the IRC feet and the IRC man's upper
# body, velocity tracking on a plane. The upper body has no joints, so only the 12
# leg joints are observed and actuated. Logs go to logs/rsl_rl/mjlab_kid_rl_irc_velocity.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-KidRL-IRC",
  env_cfg=make_kid_rl_velocity_env_cfg(flat=True),
  play_env_cfg=make_kid_rl_velocity_env_cfg(play=True, flat=True),
  rl_cfg=KID_RL_IRC_VELOCITY_RL_CFG,
  runner_cls=VelocityOnPolicyRunner,
)

# Same robot on the plane, 70% of the envs walking over a mocap hurdle bar seen through
# a model of the robot camera (tasks/hurdle.py). Logs go
# to logs/rsl_rl/mjlab_kid_rl_irc_hurdle.
register_mjlab_task(
  task_id="Mjlab-Velocity-Hurdle-KidRL-IRC",
  env_cfg=make_kid_rl_hurdle_env_cfg(),
  play_env_cfg=make_kid_rl_hurdle_env_cfg(play=True),
  rl_cfg=KID_RL_IRC_HURDLE_RL_CFG,
  runner_cls=VelocityOnPolicyRunner,
)
