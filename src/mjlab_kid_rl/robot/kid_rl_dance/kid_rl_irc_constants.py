"""Kid_RL legs with the IRC feet and the IRC man's upper body.

`kid_RL_irc.xml` is kid_RL_dance.xml with the IRC feet swapped in (see its header
comment for how the IRC STLs were placed on the ankle-roll motor, the foot body
inertials and the five cylinder studs) and the kid upper body replaced by the IRC
man's: one rigid body, no joints, elbows folded forward. Only the 12 leg joints are
actuated; they keep kid_RL_dance's BAM actuators, collision setup and leg pose.
"""

from pathlib import Path

import mujoco

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from mjlab_kid_rl.robot.kid_rl_dance.kid_rl_dance_constants import (
  FULL_COLLISION,
  HOME_KEYFRAME,
  MX64V2_ACTUATOR,
  MX106V2_ACTUATOR,
  XH540_ACTUATOR,
)

KID_RL_IRC_XML: Path = Path(__file__).parent / "kid_RL_irc.xml"
assert KID_RL_IRC_XML.exists()


def get_irc_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(KID_RL_IRC_XML))


# Leg actuators only: the upper body has no joints, so MX28_ACTUATOR (arms, neck,
# head) has nothing to drive. MX64V2's torso_yaw alternative simply finds no match.
KID_RL_IRC_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(MX64V2_ACTUATOR, MX106V2_ACTUATOR, XH540_ACTUATOR),
  soft_joint_pos_limit_factor=0.9,
)

# The IRC stud tips sit 16.097 mm higher in the foot frame than kid_RL_dance's
# (-0.039403 vs -0.0555), so the same pose stands that much lower. Leg joints only:
# the elbow entries of HOME_KEYFRAME name joints this model does not have.
HOME_KEYFRAME_IRC = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, HOME_KEYFRAME.pos[2] - 0.016097),
  joint_pos={k: v for k, v in HOME_KEYFRAME.joint_pos.items() if "elbow" not in k},
)


def get_kid_rl_irc_robot_cfg() -> EntityCfg:
  """Get a fresh Kid_RL robot configuration with the IRC feet."""
  return EntityCfg(
    init_state=HOME_KEYFRAME_IRC,
    collisions=(FULL_COLLISION,),
    spec_fn=get_irc_spec,
    articulation=KID_RL_IRC_ARTICULATION,
  )
