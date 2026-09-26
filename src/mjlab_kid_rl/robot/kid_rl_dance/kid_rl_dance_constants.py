"""Kid_RL_v3 humanoid constants -- 4-axis-arm revision ("dance").

Diverges from `kid_rl/kid_rl_constants.py` only where the new MJCF differs:

  * Each arm gained a shoulder *yaw* and a *wrist* pitch, and a neck_yaw/
    head_pitch pair was added: 19 -> 25 actuated joints (nq 40, njnt 34).
  * shoulder_pitch dropped from an MX64V2 to an MX28 on the real robot, so it
    moved between the two BamActuatorCfg groups below.
  * The torso mounts 98 mm higher (torso_1 at z=0.0962 vs -0.0018) and the whole
    upper body is ~0.9 kg lighter (7.216 -> 6.319 kg total).

The legs, pelvis/ankle 4-bar linkages, tendons, equalities, contact excludes and
IMU sensors are byte-identical to the old model, so everything keyed off those
carries over unchanged.

`kid_RL_dance.xml` is the CAD export with the conventions the exporter drops
re-applied by hand (see the header comment in that file): left_/right_ joint
naming, the two foot sites, group="1" on every collision geom, names on the
self-collision boxes, and armature=1e-3 on the 8 passive linkage joints.
"""

import math
from pathlib import Path

import mujoco

from mjlab_kid_rl.dr_switch import DR_WIDE

from bam.mjlab import BamActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

KID_RL_DANCE_XML: Path = Path(__file__).parent / "kid_RL_dance.xml"
assert KID_RL_DANCE_XML.exists()


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(KID_RL_DANCE_XML))


##
# Actuator config.
##

# One BamActuatorCfg per Dynamixel model used in the MJCF's mx28/mx64v2/mx106v2/xh540
# default classes. target_names_expr selects which *joints* each group drives (a
# tuple of regexes, matched against joint names since transmission_type defaults to
# JOINT) -- BamActuator then converts each matched joint's XML <position> actuator
# into a raw motor actuator and drives it with BAM's physical motor model instead.
# kp_fw is pulled straight from that class's <position kp="..."> in the MJCF so the
# firmware P-gain matches what the model already declares. vin_range/vin_min reflect
# the pack's voltage under load (4S ~15.8-16.8V, floor 15.6V), and
# vin_drop_resistance_range models the battery+wiring resistance that causes vin to
# sag further under load. The hip/ankle roll "cap" and "actual" joints have no XML
# actuator (they're coupled to the "crank" joint via a fixed tendon equality) and
# are silently skipped.
_VIN_RANGE = (15.8, 16.8)
_VIN_MIN = 15.6
_VIN_DROP_RESISTANCE_RANGE = (0.0, 0.01)
# Command delay, in simulation steps (0.005 s each), modeling policy->motor
# latency. Note this constant is shared by both tasks -- the velocity task and
# the tracking task build the same robot cfg -- so DR_WIDE moves both.
#
# 2026-09-25: 2-5 (10-25 ms) -> 1-2 (5-10 ms). 명령 전달만 담당한다.
# 실기 루프 지연 = 관측 나이 + 명령 전달이고, 관측 나이(실측 2~11 ms, CSV joint_age_*)는
# 관측 지연(tasks/kid_RL_env_cfg.py, lag 0~1 = 0 또는 20 ms, 평균 10 ms)이 이미 덮는다.
# 예전 2-5 는 관측 나이까지 여기에 합쳐 잡은 값이라 같은 지연을 두 번 셌다
# (학습 루프 지연 평균 약 27 ms vs 실기 약 5~15 ms).
# 명령 전달 = SyncWrite 호출 0.6 ms(실측, CSV prev_write_ms) + 선 위 전송 1.4 ms
# (계산: 139 바이트 @ 1 Mbps, 서보는 패킷 끝까지 받고 CRC 확인 후 적용)
# + 서보가 새 목표를 쓰기까지 1~2 ms(가정, 측정 안 함) = 약 3~4 ms.
# 하한 1(5 ms)은 이 값을 덮고, 상한 2(10 ms)는 측정 안 한 서보 반영 시간에 대한 여유다.
# 관측 지연과 합친 학습 루프 지연은 5~30 ms(평균 약 17.5 ms)로, 실기보다 짧아지는
# 경우는 없다 -- 학습 지연이 실기보다 짧으면 실기에서 진동하므로 그쪽만은 피한다.
_DELAY_MIN_LAG = 1
_DELAY_MAX_LAG = 3 if DR_WIDE else 2
# Use BAM's stock DelayBuffer sampling. Without the removed local subclass,
# lag can briefly be zero after reset and a rising lag can revisit an older command.
_DELAY_UPDATE_PERIOD = 4
_DELAY_HOLD_PROB = 0.6
_DELAY_PER_ENV_PHASE = True

# Passive follower joints of the hip/ankle roll 4-bar linkage (coupled to the
# actuated "_roll_crank" joint via a fixed tendon equality, not driven by their own
# motor). Currently not matched by any target_names_expr below, so this has no
# effect yet -- see the note after MX106V2_ACTUATOR.
_PRESERVE_ROLL_LINKAGE_FRICTION = (
  "left_hip_roll_cap",
  "right_hip_roll_cap",
  "left_ankle_roll_cap",
  "right_ankle_roll_cap",
  "left_hip_roll_actual",
  "right_hip_roll_actual",
  "left_ankle_roll_actual",
  "right_ankle_roll_actual",
)

# Every upper-body joint is an MX28 on this revision: the 5 arm axes per side
# (shoulder pitch/roll/yaw, elbow, wrist) plus the 2 neck/head axes. shoulder_pitch
# in particular moved here from MX64V2 -- the new arm uses the smaller motor, so its
# torque ceiling drops from +/-3.0 Nm to +/-1.4 Nm.
MX28_ACTUATOR = BamActuatorCfg(
  motor_name="mx28",
  model="m5",
  target_names_expr=(
    r"^(left|right)_(shoulder_pitch|shoulder_roll|shoulder_yaw|elbow_pitch|wrist_pitch)$",
    r"^(neck_yaw|head_pitch)$",
  ),
  kp_fw=35,
  vin_range=_VIN_RANGE,
  vin_min=_VIN_MIN,
  vin_drop_resistance_range=_VIN_DROP_RESISTANCE_RANGE,
  max_current=1.84,
  delay_min_lag=_DELAY_MIN_LAG,
  delay_max_lag=_DELAY_MAX_LAG,
  delay_update_period=_DELAY_UPDATE_PERIOD,
  delay_hold_prob=_DELAY_HOLD_PROB,
  delay_per_env_phase=_DELAY_PER_ENV_PHASE,
)
# Only torso_yaw and the two hip yaws are left on the MX64V2 now that
# shoulder_pitch moved to MX28.
MX64V2_ACTUATOR = BamActuatorCfg(
  motor_name="mx64v2",
  model="m5",
  target_names_expr=(r"^(torso_yaw|(left|right)_hip_yaw)$",),
  kp_fw=60,
  vin_range=_VIN_RANGE,
  vin_min=_VIN_MIN,
  vin_drop_resistance_range=_VIN_DROP_RESISTANCE_RANGE,
  max_current=6.52,
  delay_min_lag=_DELAY_MIN_LAG,
  delay_max_lag=_DELAY_MAX_LAG,
  delay_update_period=_DELAY_UPDATE_PERIOD,
  delay_hold_prob=_DELAY_HOLD_PROB,
  delay_per_env_phase=_DELAY_PER_ENV_PHASE,
)
MX106V2_ACTUATOR = BamActuatorCfg(
  motor_name="mx106v2",
  model="m5",
  target_names_expr=(r"^(left|right)_(hip|ankle)_roll_crank$",),
  kp_fw=134,
  vin_range=_VIN_RANGE,
  vin_min=_VIN_MIN,
  vin_drop_resistance_range=_VIN_DROP_RESISTANCE_RANGE,
  max_current=6.8,
  delay_min_lag=_DELAY_MIN_LAG,
  delay_max_lag=_DELAY_MAX_LAG,
  delay_update_period=_DELAY_UPDATE_PERIOD,
  delay_hold_prob=_DELAY_HOLD_PROB,
  delay_per_env_phase=_DELAY_PER_ENV_PHASE,
)
XH540_ACTUATOR = BamActuatorCfg(
  motor_name="xh540",
  model="m5",
  target_names_expr=(r"^(left|right)_(hip_pitch|knee_pitch|ankle_pitch)$",),
  kp_fw=165,
  vin_range=_VIN_RANGE,
  vin_min=_VIN_MIN,
  vin_drop_resistance_range=_VIN_DROP_RESISTANCE_RANGE,
  max_current=5.5,
  delay_min_lag=_DELAY_MIN_LAG,
  delay_max_lag=_DELAY_MAX_LAG,
  delay_update_period=_DELAY_UPDATE_PERIOD,
  delay_hold_prob=_DELAY_HOLD_PROB,
  delay_per_env_phase=_DELAY_PER_ENV_PHASE,
)

KID_RL_DANCE_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(MX28_ACTUATOR, MX64V2_ACTUATOR, MX106V2_ACTUATOR, XH540_ACTUATOR),
  soft_joint_pos_limit_factor=0.9,
)

##
# Keyframe config.
##

# All 33 joints in the "home" keyframe baked into kid_RL_dance.xml are 0.0, so the
# dict below is the only thing that shapes the stance -- the 27 new/unlisted joints
# (arms, neck, head) fall back to resolve_expr's default_val=0.0. We use {} rather
# than None even when empty to avoid a mjlab bug: the joint_pos=None path
# (mjlab/entity/entity.py) reads the keyframe's qpos straight from MuJoCo (numpy
# float64) without casting, producing a float64 default_joint_pos that then fails to
# write into the sim's float32 qpos buffer on reset ("Index put requires the source
# and destination dtypes match, got Float for the destination and Double for the
# source").
#
# pos MUST be set explicitly to the keyframe's root height (0.476329, from the
# "home" keyframe's qpos[2]). InitialStateCfg.pos defaults to (0,0,0), and only
# joint_pos is overridden below, so leaving it out spawns the robot with its root at
# z ~= 0 + reset noise + terrain height -- i.e. face-down in the ground.
#
# The legs are unchanged from `kid_rl`, so this pose reproduces the same stance:
# lowest collision extent sits at z=0.00566 with the root at 0.476329.
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.476329),
  joint_pos={
    "left_knee_pitch": math.radians(30),
    "right_knee_pitch": -math.radians(30),
    "left_hip_pitch": -math.radians(15),
    "left_ankle_pitch": -math.radians(15),
    "right_hip_pitch": math.radians(15),
    "right_ankle_pitch": math.radians(15),
  },
)

##
# Collision config.
##

# Ground-contact geoms: real sliding friction (condim=3) + custom friction, and
# priority=1 so this friction always wins over whatever the floor/terrain geom
# specifies (no mixing).
_FOOT_PATTERN = r".*_foot_collision_.*"

# Invisible 15 mm extensions on only the inward face of each foot. Collision
# bit 32 is private to this pair, so they warn the policy about foot-to-foot
# clearance without touching the terrain or any other robot geometry.
_FOOT_INNER_SAFETY_PATTERN = r"^(left|right)_foot_inner_safety$"

# Hip/ankle parallel-linkage self-contact geoms (groove walls, roll caps). These
# only ever touch each other (contype=conaffinity=16, isolated from the floor and
# the rest of the robot). Also condim=3 (real sliding friction), since these
# contacts model an actual slider/crank mechanism; their existing MJCF
# friction/solref values are left untouched.
_LINKAGE_PATTERN = r"^(left|right)_(hip|ankle)_(cap|groove_wall\d)$"

# Self-collision proxies. Two flavours on this revision, both ending in
# "_collision" so one pattern covers them:
#   * legs + base_link: per-body boxes named "<body>_self_collision_<n>", same as
#     the old model.
#   * torso/neck/head/arms/hands: the CAD collision *mesh* itself, named
#     "<body>_collision". Kept as meshes on purpose -- no primitive proxies for the
#     upper body on this robot.
# condim=1 (frictionless point contact) is enough to stop limbs interpenetrating;
# no need to pay for a full friction cone. Also matches ".*_collision" (the foot
# pattern's substring), so it must come *after* _FOOT_PATTERN in every dict below --
# dict order is match order, first pattern wins.
_SELF_COLLISION_PATTERN = r".*_collision"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(
    _FOOT_PATTERN,
    _FOOT_INNER_SAFETY_PATTERN,
    _LINKAGE_PATTERN,
    _SELF_COLLISION_PATTERN,
  ),
  # contype/conaffinity: feet and self-collision geoms fall back to the CollisionCfg
  # default (1). The linkage geoms must keep 16 explicitly or they'd be reset to 1
  # and start colliding with the floor/rest of the robot instead of just themselves.
  contype={_FOOT_INNER_SAFETY_PATTERN: 32, _LINKAGE_PATTERN: 16},
  conaffinity={_FOOT_INNER_SAFETY_PATTERN: 32, _LINKAGE_PATTERN: 16},
  condim={
    _FOOT_PATTERN: 3,
    _FOOT_INNER_SAFETY_PATTERN: 1,
    _LINKAGE_PATTERN: 3,
    _SELF_COLLISION_PATTERN: 1,
  },
  priority={_FOOT_PATTERN: 1},
  friction={_FOOT_PATTERN: (1.0,)},
  disable_other_geoms=False,
)

##
# Final config.
##


def get_kid_rl_dance_robot_cfg() -> EntityCfg:
  """Get a fresh Kid_RL "dance" robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=KID_RL_DANCE_ARTICULATION,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer
  import torch

  from mjlab.scene import Scene, SceneCfg
  from mjlab.terrains import TerrainEntityCfg

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  scene_cfg = SceneCfg(
    terrain=TerrainEntityCfg(terrain_type="plane"),
    entities={"robot": get_kid_rl_dance_robot_cfg()},
  )
  scene = Scene(scene_cfg, device=device)
  model = scene.compile()
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, model.key("init_state").id)
  viewer.launch(model, data=data)
