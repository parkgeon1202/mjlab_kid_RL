"""Kid_RL "dance" robot with the IRC feet.

Everything except the two foot bodies is kid_RL_dance: the same joints, actuators
(BAM), collision setup and home pose. `kid_RL_irc.xml` is kid_RL_dance.xml with
the IRC feet swapped in -- see its header comment for how the IRC STLs were
placed on the ankle-roll motor, how each foot body's inertial was built (foot
0.126 kg + MX-106 0.153 kg) and the five cylinder studs.
"""

from pathlib import Path

import mujoco

from mjlab.entity import EntityCfg

from mjlab_kid_rl.robot.kid_rl_dance.kid_rl_dance_constants import (
  FULL_COLLISION,
  HOME_KEYFRAME,
  KID_RL_DANCE_ARTICULATION,
)

KID_RL_IRC_XML: Path = Path(__file__).parent / "kid_RL_irc.xml"
assert KID_RL_IRC_XML.exists()


def get_irc_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(KID_RL_IRC_XML))


# The IRC stud tips sit 16.097 mm higher in the foot frame than kid_RL_dance's
# (-0.039403 vs -0.0555), so the same pose stands that much lower.
HOME_KEYFRAME_IRC = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, HOME_KEYFRAME.pos[2] - 0.016097),
  joint_pos=dict(HOME_KEYFRAME.joint_pos),
)


def get_kid_rl_irc_robot_cfg() -> EntityCfg:
  """Get a fresh Kid_RL robot configuration with the IRC feet."""
  return EntityCfg(
    init_state=HOME_KEYFRAME_IRC,
    collisions=(FULL_COLLISION,),
    spec_fn=get_irc_spec,
    articulation=KID_RL_DANCE_ARTICULATION,
  )
