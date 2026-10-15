from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from mjlab_kid_rl.tasks.kid_RL_env_cfg import (
  KID_RL_VELOCITY_RL_CFG,
  make_kid_rl_velocity_env_cfg,
)

# Back to the stock runner: common_step_counter now restores on resume again
# (KidRLOnPolicyRunner in rl.py forced it to 0 to dodge the staged-curriculum
# fast-forward-on-resume bug -- see its docstring -- at the cost of re-
# climbing the curriculum ramp on every resume). rl.py is left in place,
# unused, in case this needs to be swapped back.
register_mjlab_task(
  task_id="Mjlab-Velocity-KidRL",
  env_cfg=make_kid_rl_velocity_env_cfg(),
  play_env_cfg=make_kid_rl_velocity_env_cfg(play=True),
  rl_cfg=KID_RL_VELOCITY_RL_CFG,
  runner_cls=VelocityOnPolicyRunner,
)

# Same task with the generator terrain replaced by a plane -- for isolating gait
# work from terrain difficulty. See make_kid_rl_velocity_env_cfg's Terrain section
# for what the swap entails; obs/action shapes are unchanged, so checkpoints move
# between this and Mjlab-Velocity-KidRL in either direction.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-KidRL",
  env_cfg=make_kid_rl_velocity_env_cfg(flat=True),
  play_env_cfg=make_kid_rl_velocity_env_cfg(play=True, flat=True),
  rl_cfg=KID_RL_VELOCITY_RL_CFG,
  runner_cls=VelocityOnPolicyRunner,
)

# kid_RL with the IRC feet (kid_RL_irc.xml): same task on a plane. Observation and
# action shapes are unchanged, so kid_RL checkpoints load here, but the feet sit
# 16 mm higher in the foot frame and weigh 0.123 kg less each.
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-KidRL-IRC",
  env_cfg=make_kid_rl_velocity_env_cfg(flat=True, robot="irc"),
  play_env_cfg=make_kid_rl_velocity_env_cfg(play=True, flat=True, robot="irc"),
  rl_cfg=KID_RL_VELOCITY_RL_CFG,
  runner_cls=VelocityOnPolicyRunner,
)
