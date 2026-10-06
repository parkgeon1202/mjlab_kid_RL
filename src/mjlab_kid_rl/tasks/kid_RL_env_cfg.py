"""Kid_RL_v3 velocity-tracking locomotion task."""

from copy import deepcopy
from dataclasses import dataclass, fields

import mujoco
import numpy as np

import mjlab.terrains as terrain_gen
from bam.mjlab import bam_init
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.terminations import root_height_below_minimum
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils import spec_config as spec_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from mjlab_kid_rl.dr_switch import DR_WIDE
from mjlab_kid_rl.plugins import enable_step_reward_logging
from mjlab_kid_rl.robot.kid_rl_dance.kid_rl_dance_constants import (
  get_kid_rl_dance_robot_cfg,
)
from mjlab_kid_rl.tasks.mdp import (
  UniformVelocityCommandWithRotationCfg,
  default_joint_pose_exp,
  set_soft_joint_pos_limit_margin,
  balance_assist,
  airborne_foot_arm_swing_reward,
  feet_distance_penalty,
  base_height_penalty,
  feet_crossing_reward,
  foot_flatness_penalty,
  foot_non_stud_contact_penalty,
  foot_landing_alignment_reward,
  flatness_weighted_foot_slip_penalty,
  feet_air_time_continuous_reward,
  feet_air_time_touchdown_reward,
  feet_single_contact_reward,
  both_feet_airborne_penalty,
  forward_step_reward,
  lateral_swing_progress_reward,
  gait_phase,
  gait_phase_contact_reward,
  imu_gyro_lowpass,
  joint_vel_window_avg,
  randomize_bam_kp,
  gait_phase_swing_clearance_reward,
  swing_progress_reward,
  gait_symmetry_reward,
  velocity_shortfall_penalty,
  relative_angular_velocity_error_penalty,
  relative_linear_velocity_error_penalty,
  foot_base_heading_error_penalty,
  log_solver_buffer_usage,
  no_stepping_penalty,
  not_stepping_penalty,
  not_stepping_each_foot_penalty,
  overlong_swing_penalty,
  same_foot_repeat_penalty,
  self_collision_cost_excluding_linkage,
  set_push_parameters,
  make_interpolated_push_stages,
  reward_based_staged_curriculum,
  selected_action_excess_l2,
  upper_body_excursion_penalty,
  settled_standing_penalty,
)
from mjlab_kid_rl.tasks.mdp import upright as local_upright
from mjlab_kid_rl.tasks.symmetry import compute_symmetric_states

##
# Terrain.
##

# Step/noise magnitudes below are scaled down from typical human-sized-humanoid
# values (e.g. G1, ~1.3m tall): kid_RL_v3 stands ~0.476m tall at home with a
# ~0.14m thigh and ~0.14m shin, so a 0.15m step (common for G1) would be taller
# than the whole shin -- unclimbable. Scaled by robot height (~0.476/1.3 ~= 0.37x).
KID_RL_TERRAIN_GENERATOR = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  num_rows=10,
  # In curriculum mode, TerrainGenerator always uses len(sub_terrains) columns
  # internally regardless of num_cols (see its own docstring) -- but mjlab's
  # `out_of_terrain_bounds` termination computes its y-limit straight from this
  # cfg field, not from the generator's actual internal column count. Left at
  # the default of 1, that made every env with |y| > ~3.7m (i.e. any of the 4
  # non-center terrain-type columns) truncate on literally the first step after
  # reset, before it could ever do anything. Must track len(sub_terrains) below
  # (5) for that termination to compute the right bound.
  num_cols=5,
  border_width=20.0,
  curriculum=True,
  sub_terrains={
    "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.35),
    "nested_rings": terrain_gen.BoxNestedRingsTerrainCfg(
      proportion=0.15,
      num_rings=5,
      ring_width_range=(0.15, 0.3),
      gap_range=(0.0, 0.05),
      height_range=(0.02, 0.06),
      platform_width=0.6,
    ),
    "random_stairs": terrain_gen.BoxRandomStairsTerrainCfg(
      proportion=0.15,
      step_height_range=(0.0, 0.055),
      step_width=0.2,
      platform_width=0.6,
    ),
    "rough": terrain_gen.HfRandomUniformTerrainCfg(
      proportion=0.2,
      noise_range=(0.005, 0.035),
      noise_step=0.005,
    ),
    "random_grid": terrain_gen.BoxRandomGridTerrainCfg(
      proportion=0.15,
      grid_width=0.25,
      grid_height_range=(0.0, 0.04),
      platform_width=0.6,
      # grid_width=0.15 alone produces ~2800 boxes/patch (~28k across 10 rows,
      # ~9s to compile); merge similar-height neighbors to keep geom count sane.
      merge_similar_heights=True,
      height_merge_threshold=0.01,
      max_merge_distance=3,
    ),
  },
)

##
# Sensors.
##

# Per-body (P=2), NOT per-geom. The velocity task's foot rewards (air_time,
# foot_slip, foot_clearance, foot_swing_height) index this sensor in parallel with
# the 2 foot sites, so P must equal the number of feet. Matching per-geom instead
# would give P=4 (2 collision boxes per foot) and silently misalign against them.
FEET_GROUND_SENSOR_CFG = ContactSensorCfg(
  name="feet_ground_contact",
  primary=ContactMatch(
    mode="body",
    pattern=r"^(left|right)_foot_1$",
    entity="robot",
  ),
  secondary=ContactMatch(mode="body", pattern="terrain"),
  fields=("found", "force"),
  reduce="netforce",
  num_slots=1,
  track_air_time=True,
)

# base_link is the free-joint root body, so its subtree is the whole robot -- this
# flags any part of the robot touching any other part of itself.
SELF_COLLISION_SENSOR_CFG = ContactSensorCfg(
  name="self_collision",
  primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
  secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
  fields=("found",),
  reduce="none",
  num_slots=1,
)

# One sensor per hip/ankle 4-bar linkage's cap<->groove-wall pair (contype/
# conaffinity=16 in kid_RL_dance.xml -- a real, permanent contact during
# normal articulation, redundant with the tendon equality for kinematics).
# SELF_COLLISION_SENSOR_CFG can't exclude just these pairs itself (it's one
# symmetric "anything in base_link touching anything in base_link" sensor,
# so removing either side from the match still leaves it visible from the
# other side); these track just the 4 known pairs so their count can be
# subtracted out in the reward instead (see
# mdp.self_collision_cost_excluding_linkage).
_LINKAGE_PAIRS = (
  ("left_hip_linkage_contact", "left_hip_cap_1", "left_hip_slider_1"),
  ("right_hip_linkage_contact", "right_hip_cap_1", "right_hip_slider_1"),
  ("left_ankle_linkage_contact", "left_ankle_cap_1", "left_ankle_1"),
  ("right_ankle_linkage_contact", "right_ankle_cap_1", "right_ankle_1"),
)
LINKAGE_CONTACT_SENSOR_CFGS = tuple(
  ContactSensorCfg(
    name=name,
    primary=ContactMatch(mode="body", pattern=cap_body, entity="robot"),
    secondary=ContactMatch(mode="body", pattern=partner_body, entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  for name, cap_body, partner_body in _LINKAGE_PAIRS
)

# Any robot body other than the feet touching the terrain -- catches face-
# plants/collapses (torso, head, arms, knees down) that root_height_below_
# minimum can miss if the fallen pose still keeps the root above 0.2m (e.g.
# lying on an outstretched arm/torso rather than flat). exclude keeps the
# feet themselves out, since foot-terrain contact is the normal walking case.
NON_FOOT_TERRAIN_CONTACT_SENSOR_CFG = ContactSensorCfg(
  name="non_foot_terrain_contact",
  primary=ContactMatch(
    mode="body",
    pattern=r".*",
    entity="robot",
    exclude=(r"^(left|right)_foot_1$",),
  ),
  secondary=ContactMatch(mode="body", pattern="terrain"),
  fields=("found",),
  reduce="none",
  num_slots=1,
)

# Foot geoms that are not studs -- the raised sole plate (collision_1) and the
# ankle block (collision_2) -- against the terrain. The foot should stand on its
# four studs; these only touch when it tips onto the toe edge or rolls over.
FOOT_NON_STUD_CONTACT_SENSOR_CFG = ContactSensorCfg(
  name="foot_non_stud_contact",
  primary=ContactMatch(
    mode="geom",
    pattern=r"^(left|right)_foot_collision_[12]$",
    entity="robot",
  ),
  secondary=ContactMatch(mode="body", pattern="terrain"),
  fields=("found",),
  reduce="none",
  num_slots=1,
)

# One frame per foot (site placed at the sole's bottom face, see kid_RL_dance.xml).
# include_geom_groups=(0,) means "terrain only": the 52 foot/self-collision boxes
# are tagged group=1 in kid_RL_dance.xml specifically so this doesn't also hit the
# robot's own geoms (see FEET_GROUND_SENSOR_CFG comment above).
_FOOT_SITE_NAMES = ("left_foot", "right_foot")

FOOT_HEIGHT_SCAN_CFG = TerrainHeightSensorCfg(
  name="foot_height_scan",
  frame=tuple(
    ObjRef(type="site", name=s, entity="robot") for s in _FOOT_SITE_NAMES
  ),
  pattern=RingPatternCfg.single_ring(radius=0.04, num_samples=2),
  ray_alignment="yaw",
  max_distance=1.0,
  exclude_parent_body=True,
  include_geom_groups=(0,),
  debug_vis=False,
)

##
# Scene.
##

# TerrainEntityCfg's default groundplane texture already matches kid_RL_dance.xml's
# scene.xml exactly (same checker rgb1/rgb2/markrgb), except texrepeat (mjlab
# default is 4x4, scene.xml uses 5x5) -- override just that field. Only matters for
# terrain_type="plane" (geom_names_expr targets the "terrain" plane geom, which
# doesn't exist for terrain_type="generator" -- harmless no-op there).
_GROUNDPLANE_MATERIAL = spec_cfg.MaterialCfg(
  name="groundplane",
  texuniform=True,
  texrepeat=(5.0, 5.0),
  reflectance=0.2,
  texture="groundplane",
  geom_names_expr=("terrain$",),
)


def _apply_scene_xml_visuals(spec: mujoco.MjSpec) -> None:
  """Match kid_RL_dance.xml's scene.xml sky + default camera framing.

  Scene always starts from mjlab's own bundled base scene.xml (headlight/haze
  already match ours by coincidence), then adds terrain/entities, then calls this
  hook. The skybox texture and the free-camera azimuth/elevation aren't exposed by
  TerrainEntityCfg's texture/material/light fields, so they're added/overridden
  here instead of touching mjlab's bundled scene.xml.
  """
  spec_cfg.TextureCfg(
    name="sky",
    type="skybox",
    builtin="gradient",
    rgb1=(0.3, 0.5, 0.7),
    rgb2=(0.0, 0.0, 0.0),
    width=512,
    height=3072,
  ).edit_spec(spec)
  spec.visual.global_.azimuth = 160
  spec.visual.global_.elevation = -20


SCENE_CFG = SceneCfg(
  terrain=TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=KID_RL_TERRAIN_GENERATOR,
    max_init_terrain_level=5,
    materials=(_GROUNDPLANE_MATERIAL,),
  ),
  num_envs=1,
  extent=2.0,
  entities={"robot": get_kid_rl_dance_robot_cfg()},
  sensors=(FEET_GROUND_SENSOR_CFG, SELF_COLLISION_SENSOR_CFG, FOOT_HEIGHT_SCAN_CFG),
  spec_fn=_apply_scene_xml_visuals,
)

##
# Simulation.
##

# timestep/integrator originally matched kid_RL_dance.xml's own <option>
# (0.001/implicitfast) -- Scene.compile() silently keeps mjlab's scene-level
# defaults instead (see the "Attach conflict" warning), so this is what actually
# makes them take effect. The hip/ankle 4-bar linkage's tendon equality +
# groove/cap contacts were the reason 0.001 was picked over the faster 0.005 used
# elsewhere -- coarsening back to 0.005 risks reintroducing instability there
# (constraint divergence / NaN, or nconmax/njmax overflow from self-collision).
# Bumped to 0.005 (decimation dropped to match, see cfg.decimation below) to speed
# up iteration time for a resumed run; watch the console for overflow warnings and
# NaNs, especially around the linkage joints, and revert to 0.001 if they appear.
#
# nconmax/njmax left at mujoco_warp's own heuristic (None) for now. For our model
# (nv=33, has a heightfield sub-terrain) that heuristic works out to ~192/~1536 --
# left commented as a starting point to bump if the console ever prints an
# "nconmax overflow" / "njmax overflow" warning during an actual run.
SIM_CFG = SimulationCfg(
  mujoco=MujocoCfg(
    timestep=0.005,
    integrator="implicitfast",
    iterations=50,
    ls_iterations=30,
    ccd_iterations=100,
    # Route box-box through the analytic primitive path instead of GJK/EPA.
    # In mujoco_warp this flag *only* rewrites the (BOX, BOX) table entry; our
    # BOX-MESH (cap/groove), MESH-MESH (cap/cap) and HFIELD-BOX (rough terrain)
    # pairs still use CCD, so ccd_iterations above stays in effect for them.
    # Boxes are flat-faced polyhedra, so the analytic manifold is exact rather
    # than iterated to a tolerance -- no precision given up here. Covers ~79% of
    # observed contacts (foot/self-collision boxes vs box terrain).
    disableflags=("nativeccd",),
  ),
  # nconmax/njmax are *per-world* (mjlab multiplies by num_envs internally), so
  # they were the dominant cost of the 6.8GB/512-env GPU footprint that capped us
  # at ~512-600 envs on an 8GB card -- not geom count, which barely moved it.
  #
  # The original 512/4096 was sized against a historical "njmax~2315" adversarial
  # spike (see old comment below), but re-measuring under 500 steps x 512 envs of
  # mixed uniform-random + bang-bang actions (~256k env-steps) on the *current*
  # model -- with its contact excludes and contype/conaffinity grouping already
  # tuned -- never exceeded ncon=47 / nefc=219. That historical 2315 could not be
  # reproduced and is presumably stale (pre-dates some of that tuning).
  #
  # Went as low as 150/620 (mujoco_warp's hard floor for this model+keyframe:
  # 146/596, independent of num_envs) to squeeze num_envs on an 8GB laptop GPU,
  # then to 200/768 (~4x/~3.5x headroom over a 47/219 runtime peak). Both were
  # measured under *random-action* stress tests, not a real trained policy
  # running the final (hardest) curriculum stage at full scale -- and 150/620
  # training at that stage produced near-universal early fell_over terminations
  # (mean episode length ~19 steps) within ~13 iterations of resuming an
  # otherwise-good checkpoint, almost certainly via the exact failure mode
  # described below (silent contact/constraint drops -> energy gain -> falls).
  # The random-action measurements underestimated the real worst case.
  #
  # 300/1536 (~7x/~7x headroom over the 47/219 random-action peak, roughly
  # double 200/768's relative margin) is the current compromise -- more room
  # for num_envs than the historically-proven 512/4096, but this specific
  # combination has NOT yet been validated under real trained-policy rollouts
  # at the final curriculum stage the way 512/4096 effectively has (by not
  # failing). Treat any run on this as provisional until it's been watched for
  # a couple hundred iterations without mean episode length craters or
  # fell_over dominating -- if the console ever prints an nconmax/njmax
  # overflow warning, or that happens, these need to go back up (or num_envs
  # down) before suspecting anything else.
  #
  # 2026-08-24: tried 300/1536 on a from-scratch resume, but that run turned
  # out to still be running the pre-edit code (old 500/1500/3000 curriculum
  # thresholds, old upright weight) -- not a clean comparison. Its episode
  # length/fell_over were actually *worse* than a 150/620 run at a similar
  # iteration, but that's confounded and not trustworthy either way.
  #
  # Reverted to 150/620 here so the next run is a clean baseline, now paired
  # with the log_solver_buffer_usage step-event (see mdp.py) that reads
  # d.nefc/d.nacon directly against njmax/naconmax every step and logs
  # Metrics/njmax_overflow_frac + Metrics/nconmax_overflow -- a direct signal
  # instead of inferring overflow from episode length / fell_over, which has
  # now proven unreliable twice. If njmax_overflow_frac stays ~0 through a
  # fell_over-dominated run, that rules out this theory entirely and points
  # elsewhere (actuator/torque, reward shaping, policy capacity). If it's
  # meaningfully >0, bump nconmax/njmax back up (or drop num_envs) with actual
  # evidence this time.
  #
  # 2026-08-24 (later): confirmed clean at num_envs<=1300 (nacon_raw ~14-17k,
  # healthy) -- nacon_raw drops to exactly 0 (no contacts detected at all,
  # system-wide) somewhere between 1300 and 1536, unrelated to nconmax/njmax
  # capacity (njmax_overflow_count_total/nconmax_overflow_count_total stayed 0
  # in every run, including the broken ones -- so that num_envs cliff is a
  # separate, still-unexplained bug, not a buffer-size issue). Briefly tried
  # njmax=256 (~2.6x margin over the ~97 peak nefc seen so far) to cut the
  # per-step solver cost njmax drives, then reverted back to 620 -- keep the
  # bigger margin until the num_envs cliff itself is understood, revisit the
  # njmax tightening once that's resolved.
  #
  # Original reasoning (kept for context): random/early-training actions can send
  # the robot into violent, highly self-collided poses that overflow the
  # heuristic's estimate ("nefc overflow" / "contact match overflow" in the
  # console), silently dropping contacts/constraints and letting the sim gain
  # energy until it diverges to NaN. Observed overflow requests reached
  # njmax~2315 and contact_sensor_maxmatch~194 under adversarial random actions;
  # sized with headroom above that.
  # 160/640, not the old 150/620: mujoco_warp's put_data() builds its initial
  # MjData from qpos0 (root at z=0, i.e. the robot buried in the terrain), and
  # requires nconmax/njmax >= that configuration's counts. The 4-axis-arm model's
  # extra upper-body collision geometry pushes that hard floor from 146/596 to
  # 151/616, so 150/620 now fails to construct the env at all
  # ("nconmax overflow (nconmax must be >= 151)"). These keep the same ~1.03x/1.04x
  # margin over the floor that 150/620 had -- NOT a re-sizing against runtime peaks.
  nconmax=160,
  njmax=640,
  contact_sensor_maxmatch=256,
)

##
# Viewer.
##

VIEWER_CONFIG = ViewerConfig(
  origin_type=ViewerConfig.OriginType.ASSET_BODY,
  entity_name="robot",
  body_name="torso_1",
  distance=3.0,
  elevation=-15.0,
  azimuth=90.0,
)

##
# Environment.
##

# Joints the policy sees / drives. Excludes the 8 passive follower joints of the
# hip/ankle roll 4-bar linkages (*_roll_cap, *_roll_actual): they carry no encoder
# on the real robot and are mechanically determined by the *_roll_crank joint via
# the tendon equality, so feeding them to the policy would be both unobservable
# in hardware and redundant. Leaves the 25 actuated joints.
DOFS_FILTER = r"^(?!.*_(?:cap|actual)$).*$"
POSE_DOFS_FILTER = r"^(?!torso_yaw$)(?!.*_(?:hip|ankle)_roll)(?!.*_(?:cap|actual)$).*$"
"""Actuated joints scored by the pose reward: DOFS_FILTER minus the leg (hip/ankle)
roll joints and torso_yaw (its target is clipped to 0, so the policy cannot move it)."""
LEG_DOFS_FILTER = r"^(left|right)_(hip_yaw|hip_roll_crank|hip_pitch|knee_pitch|ankle_pitch|ankle_roll_crank)$"
"""The 12 actuated leg joints (no arms, torso, head or passive linkage joints)."""
UPPER_BODY_DOFS_FILTER = r"^(torso_yaw|neck_yaw|head_pitch|(left|right)_(shoulder_pitch|shoulder_roll|shoulder_yaw|elbow_pitch|wrist_pitch))$"
"""The 13 actuated upper-body joints: torso, neck/head and both arms."""

BALANCE_ASSIST_SCALE: float | None = 0.0
"""Strength of the balance-assist torque during training: 0 = off, 1 = full.
None removes the event entirely (then the scale file is ignored too)."""

BALANCE_ASSIST_SCALE_FILE = "balance_assist_scale.txt"
"""If this file exists (path relative to where training is launched), the number
in it overrides BALANCE_ASSIST_SCALE and is re-read while training runs."""

USE_GAIT_PHASE = False
"""보행 위상(gait_phase) 관측과 위상 보상(gait_phase_contact, gait_phase_swing_clearance) 사용 여부.
2026-09-30 재학습부터 끈다. 실기 컨트롤러는 모델 메타데이터로 84/86 차원을 자동 판별한다."""

IMU_GYRO_FILTER_HZ = 6.0
"""actor gyro 관측 저역통과 컷오프. deploy_ws kid_rl controllers.yaml imu_gyro_filter_hz 와 같아야 한다."""

_DEFAULT_FOOT_SEPARATION = 0.13425
"""Horizontal distance between the two foot sites in HOME_KEYFRAME [m].

Measured from the compiled model after forward kinematics. The site-local y
coordinates cannot be subtracted directly because the sites belong to different
foot body frames.
"""

_FOOT_SEPARATION_MIN = 0.13
"""Minimum allowed distance between the two foot sites [m].

The compiled HOME_KEYFRAME separation is about 0.13425 m. This lower bound only
prevents the feet from crossing too closely and does not fight the nominal
stance."""

_FOOT_SEPARATION_MAX = 0.5
"""Maximum allowed distance between the two foot sites [m].

Hip roll joints sit ~0.11 m apart (y = +/-0.054988, see kid_RL_dance.xml), so
0.2 gives real room for a wider dynamic stance (turning, the +/-0.25 m/s
lateral command) without letting the policy permanently splay the legs out
sideways."""


@dataclass(kw_only=True)
class KidRLVelocityEnvCfg(ManagerBasedRlEnvCfg):
  """kid_RL velocity config with CLI-switchable domain randomization."""

  enable_randomization: bool = True

  def __post_init__(self) -> None:
    if not self.enable_randomization:
      self._disable_randomization()

  def _disable_randomization(self) -> None:
    """Disable DR while preserving required runtime and curriculum events.

    Use from the training CLI with ``--env.enable-randomization False``.
    ``bam_init`` is required by the actuator model and the solver-buffer event
    is diagnostic only, so neither is removed.
    """
    print("[INFO] Domain randomization disabled.")

    for event_name in (
      "push_robot",
      "foot_friction",
      "left_hip_contact_friction",
      "right_hip_contact_friction",
      "left_ankle_contact_friction",
      "right_ankle_contact_friction",
      "encoder_bias",
      "base_com",
      "body_mass_randomization",
      "torso_mass_randomization",
      "dof_armature_randomization",
      "dof_cap_armature_randomization",
      "dof_actual_armature_randomization",
      "dof_cap_friction_randomization",
      "dof_actual_friction_randomization",
      "bam_kp_randomization",
    ):
      self.events.pop(event_name, None)

    self.events["reset_base"].params = {
      "pose_range": {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "yaw": (0.0, 0.0),
      },
      "velocity_range": {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
      },
    }
    self.events["reset_robot_joints"].params.update(
      {"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)}
    )

    # In mjlab, group corruption disables observation noise but not delay.
    actor_observations = self.observations["actor"]
    actor_observations.enable_corruption = False
    for term in actor_observations.terms.values():
      term.noise = None
      term.delay_min_lag = 0
      term.delay_max_lag = 0

    # BAM has its own voltage, voltage-drop, and command-delay randomization.
    robot_cfg = self.scene.entities["robot"]
    assert robot_cfg.articulation is not None
    for actuator in robot_cfg.articulation.actuators:
      if hasattr(actuator, "vin_range"):
        actuator.vin = 16.3
        actuator.vin_range = None
        actuator.vin_drop_resistance_range = None
        actuator.delay_min_lag = 0
        actuator.delay_max_lag = 0


def make_kid_rl_velocity_env_cfg(
  play: bool = False,
  flat: bool = False,
  enable_randomization: bool = True,
) -> KidRLVelocityEnvCfg:
  """Velocity-tracking locomotion task for kid_RL_v3.

  Starts from mjlab's stock velocity task and patches in the robot-specific
  scene, sensors, joint filters, and reward tuning.

  Args:
    flat: Replace the generator terrain with a single infinite plane. See the
      Terrain section below for what else has to change with it -- a plane is
      not a drop-in swap for the generator.
    enable_randomization: Disable physics, reset, observation, and actuator
      randomization when false.
  """
  cfg = make_velocity_env_cfg()

  cfg.viewer = deepcopy(VIEWER_CONFIG)
  cfg.sim = deepcopy(SIM_CFG)
  cfg.scene = deepcopy(SCENE_CFG)

  # 50 Hz policy rate. Matches the template's timestep=0.005/decimation=4 now that
  # SIM_CFG is back to 0.005 -- see the timestep comment above for why this is a
  # reversion from the finer 0.001/decimation=20 the linkage originally needed.
  cfg.decimation = 4

  foot_site_names = list(_FOOT_SITE_NAMES)

  # ---------------------------- Sensors ---------------------------
  # Defined at module scope; already namespaced to this robot's geom/body names.
  cfg.scene.sensors = (
    FEET_GROUND_SENSOR_CFG,
    FOOT_HEIGHT_SCAN_CFG,
    SELF_COLLISION_SENSOR_CFG,
    *LINKAGE_CONTACT_SENSOR_CFGS,
    NON_FOOT_TERRAIN_CONTACT_SENSOR_CFG,
    FOOT_NON_STUD_CONTACT_SENSOR_CFG,
  )

  # ---------------------------- Terminations ----------------------
  # Do not terminate solely because the base tilt exceeds the inherited
  # bad-orientation threshold. A fall is still caught when a non-foot body
  # touches the terrain.
  cfg.terminations.pop("fell_over", None)

  # End the episode when anything that is not a foot touches the ground.
  # NON_FOOT_TERRAIN_CONTACT_SENSOR_CFG is registered in cfg.scene.sensors above
  # and matches all 32 non-foot bodies against the terrain body; without a term
  # reading it the sensor still runs every step and nothing acts on the result.
  #
  # Applies on both terrain types. The scene exposes a body named "terrain" for
  # terrain_type="plane" as well as for the generator (verified by inspecting the
  # compiled model), so the sensor resolves its 32 slots either way -- this is
  # not a rough-terrain-only condition.
  #
  # The sensor declares fields=("found",) and no force history, so illegal_contact
  # takes its any(found) branch: any contact at all terminates, with no force
  # threshold to pass. That is intentionally strict -- a dragging elbow is as much
  # a failed gait as a fall -- but it does mean this fires on light brushes too.
  cfg.terminations["non_foot_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": NON_FOOT_TERRAIN_CONTACT_SENSOR_CFG.name},
  )

  # ---------------------------- Terrain ---------------------------
  # Default: the generator terrain from SCENE_CFG (flat/rings/stairs/rough/grid).
  # This is what keeps the template's `terrain_levels` curriculum and
  # `out_of_terrain_bounds` termination meaningful -- both are no-ops on a plane.
  #
  # flat=True swaps in a single infinite plane. Three things have to move with it,
  # all verified by running the task:
  #
  #   1. `terrain_levels` must go. TerrainEntity only grows a `terrain_levels`
  #      attribute under terrain_type="generator", so the stock curriculum raises
  #      AttributeError on its first call otherwise.
  #   2. `out_of_terrain_bounds` must go. It derives its bound from the terrain
  #      generator cfg, which is still attached but no longer describes anything
  #      the robot is standing on.
  #   3. nconmax has to come up. mujoco_warp builds its initial MjData from qpos0
  #      -- the robot buried in the ground -- and refuses any nconmax below that
  #      configuration's contact count. On the generator terrain that is under the
  #      160 set in SIM_CFG; on a plane it is 179. 192/768 matches what
  #      mujoco_warp's own heuristic allocates for this robot on a plane, which is
  #      what the tracking task (plane terrain, no hand-sizing) already runs on.
  if flat:
    cfg.scene.terrain = TerrainEntityCfg(
      terrain_type="plane",
      materials=(_GROUNDPLANE_MATERIAL,),
    )
    # 2026-10-05: 192/768 -> 224/896. The feet now stand on four stud boxes each
    # (kid_RL_dance.xml), which adds 32 contacts to the sunk-into-the-plane
    # configuration mujoco_warp sizes against (179 -> 211).
    cfg.sim.nconmax = 224
    cfg.sim.njmax = 896

  # ---------------------------- Actions ---------------------------
  # actuator_names (not joint names): the MJCF defines exactly 25 <position>
  # actuators, one per actuated joint, so ".*" already excludes the passive ones.
  # 25, not the old model's 19: this revision adds a shoulder yaw and a wrist pitch
  # per arm plus neck_yaw/head_pitch, and all of them are policy-driven.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = 0.5
  joint_pos_action.actuator_names = (r".*",)
  # Hold torso_yaw at 0 rad whatever the policy outputs: the clip acts on the
  # processed joint target, so the action stays 25-dim with one scalar scale
  # (checkpoints and the ONNX metadata are unchanged) but the waist cannot be
  # used to twist the trunk -- a suspected source of left/right gait asymmetry.
  joint_pos_action.clip = {"torso_yaw": (0.0, 0.0)}

  # ---------------------------- Observations ----------------------
  # base_lin_vel: no sensor for it on hardware (integrating IMU accel drifts), so
  # actor-only removal -- the critic keeps it as privileged information.
  del cfg.observations["actor"].terms["base_lin_vel"]
  # height_scan: dropped from BOTH groups. The actor is blind by design (no terrain
  # sensor planned on hardware), and the term reads the template's body-mounted
  # "terrain_scan" raycast grid, which our scene.sensors override does not provide
  # -- leaving it on the critic would fault at runtime on a missing sensor. The
  # critic still gets per-foot terrain clearance via `foot_height`
  # (foot_height_scan), just without the look-ahead a body-mounted grid would give.
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Noise is additive and in each term's own units (rad, rad/s, unitless for the
  # normalized gravity vector). 0.002 rad also clears one Dynamixel encoder tick
  # (4096 counts/rev -> 0.00153 rad), which the previous 0.001 sat below.
  #
  # delay_*_lag counts *control* steps (50 Hz -> 20 ms each), unlike BAM's motor
  # delay, which counts simulation steps.
  #
  # 2026-09-23: 관절 관측 지연 1-3 -> 0-1.
  # 실기에서는 매 틱 SyncRead 로 관절값을 새로 읽고 바로 정책을 돌리므로, 정책이
  # 받는 관절값은 1~6 ms 된 값이지 이전 스텝(20 ms 전) 값이 아니다. deploy_ws CSV
  # 6회분(POLICY_RUNNING 17,625 틱)에서 99.13% 가 이번 틱에 읽은 값이었고, 읽기가
  # 실패해 직전 값을 재사용한 틱이 0.84%, 2스텝 이상은 0.04% 였다.
  # 실제 루프 지연(읽기 -> 정책 -> 쓰기 -> 서보 반응)은 '명령' 쪽에 있고, 그것은
  # BAM 명령 지연(kid_rl_dance_constants.py _DELAY_*_LAG, 시뮬 스텝 5 ms 단위)이
  # 모사한다. 예전처럼 여기에도 min=1 을 두면 같은 지연을 두 번 세게 된다.
  # max=1 은 가끔 읽기가 실패해 한 스텝 늦은 값이 들어오는 경우에 대한 견고성이다
  # (균등 분포라 학습에서는 실기 빈도보다 훨씬 자주 나온다. 좁게 학습해 실기에서
  # 진동하는 쪽보다 낫다).
  cfg.observations["actor"].terms["joint_pos"] = ObservationTermCfg(
    func=mdp.joint_pos_rel,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(DOFS_FILTER,))},
    noise=Unoise(n_min=-0.002, n_max=0.002),
    delay_min_lag=0,
    delay_max_lag=2 if DR_WIDE else 1,
    # 매 스텝 지연을 새로 뽑을 차례지만 80% 는 지금 지연을 유지한다(2026-09-23).
    delay_hold_prob=0.8,
  )
  # 2026-09-28: 실기 Dynamixel 속도 레지스터는 순간 속도가 아니라 약 60 ms 구간 평균
  # 속도다(위치 차분 대비 전 관절 30 ms 지연, 2/5/12.5 Hz 에서 0.95/0.82/0.30 배 --
  # 60 ms 평균의 이론값 0.98/0.86/0.30 과 일치. deploy_ws kid_rl/docs/
  # SIM2REAL_OSCILLATION_2026-09-28.md). 학습도 같은 신호를 보도록 순간 속도 대신
  # (q(now) - q(now - 60 ms)) / 60 ms 를 넣는다. 읽기(버스) 지연은 아래 delay 로 따로 둔다.
  # critic 은 참값(joint_vel_rel)을 그대로 쓴다.
  cfg.observations["actor"].terms["joint_vel"] = ObservationTermCfg(
    func=joint_vel_window_avg,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(DOFS_FILTER,)),
      "window_s": 0.06,
    },
    noise=Unoise(n_min=-0.125, n_max=0.125),
    delay_min_lag=0,
    delay_max_lag=2 if DR_WIDE else 1,
    # 매 스텝 지연을 새로 뽑을 차례지만 80% 는 지금 지연을 유지한다(2026-09-23).
    delay_hold_prob=0.8,
  )

  # IMU noise/delay, actor only. What actually keeps the critic clean is its
  # group-level enable_corruption=False, not this deepcopy -- the critic's own
  # terms carry (unused) noise settings of their own. The deepcopy matters
  # because the template may hand the same term object to both groups, so
  # mutating in place would reach into the critic's config too.
  for term_name, noise in (
    ("projected_gravity", Unoise(n_min=-0.06, n_max=0.06)),
    ("base_ang_vel", Unoise(n_min=-0.06, n_max=0.06)),
  ):
    term = deepcopy(cfg.observations["actor"].terms[term_name])
    term.noise = noise
    # 2026-09-23: 1-8 -> 1-3 -> 0-1. 관절 관측과 같은 논리다(위 joint_pos 주석).
    # 실기 EBIMU 샘플은 정책이 쓰는 시점에 중앙값 5.0 ms, p99 10.1 ms, 최대 19.7 ms
    # 묵어 있었고(POLICY_RUNNING 17,625 틱), 20 ms(1 스텝)를 넘은 틱은 0개였다.
    # 루프 지연은 BAM 명령 지연이 모사하므로 여기서 min=1 을 두면 중복이다.
    # max=1 은 IMU 내부 자세 필터 지연(밖에서 측정 불가)에 대한 여유다.
    term.delay_min_lag = 0
    term.delay_max_lag = 2 if DR_WIDE else 1
    # 64 -> 0 (2026-09-23): 매 정책 스텝마다 새로 뽑는다. 64 면 한 번 뽑은 지연을
    # 1.28 초 동안 유지해서, 20 ms 늦은 IMU 가 1.28 초 내내 이어지는 구간이 생겼다.
    # 실기 EBIMU 샘플 나이는 샘플마다 0~10 ms 로 흔들리지 오래 고정되지 않는다.
    term.delay_update_period = 0
    # 관절 관측과 같게, 차례마다 80% 는 지금 지연을 유지한다(2026-09-23).
    term.delay_hold_prob = 0.8
    cfg.observations["actor"].terms[term_name] = term

  # 2026-09-28: actor gyro 를 실기 컨트롤러와 같은 1차 저역통과(6 Hz)로 거른다
  # (deploy_ws kid_rl imu_gyro_filter_hz, include/kid_rl/lowpass.hpp). 실기에서 10-15 Hz
  # 몸 떨림이 gyro -> 정책 -> 다리로 되먹임되어 발진했고, 컨트롤러에서만 거르자 학습에 없던
  # 약 27 ms 지연 때문에 6-8 Hz 발진으로 옮겨갔다. 학습도 같은 필터를 봐야 한다.
  # 노이즈·지연은 위에서 정한 그대로 필터 뒤에 붙는다. critic 은 참값.
  _gyro = cfg.observations["actor"].terms["base_ang_vel"]
  assert _gyro.params.get("sensor_name") == "robot/imu_ang_vel", _gyro.params
  _gyro.func = imu_gyro_lowpass
  # 2026-09-30: 센서 잡음은 필터 앞에서 넣는다(실기: 센서 잡음 -> 6 Hz 필터 -> 정책). 관측
  # 매니저 noise 는 func 뒤에 붙어 필터를 안 거친 백색잡음이 되므로 끈다. 크기는 그대로 ±0.06.
  _gyro.params = {
    "sensor_name": "robot/imu_ang_vel",
    "cutoff_hz": IMU_GYRO_FILTER_HZ,
    "noise": 0.06,
    "corruption_group": "actor",
  }
  _gyro.noise = None

  # Gait clock. See gait_phase's docstring for why a limit cycle needs an
  # external rhythm rather than more penalties on standing. Fixed, not
  # randomised: see the docstring for why.
  #
  # Stated as swing time -- how long one foot is off the ground, which is the
  # number that is actually being specified -- and the cycle is derived, since
  # mixing the two up is easy: one cycle is *two* steps, so cycle_time is a bit
  # over twice the swing.
  #
  # With the reward's double-support band b, a leg is in stance for
  # 0.5 + arcsin(b)/pi of the cycle (exact, verified against a numerical sweep),
  # so swing is the complement. At b=0.1 that is 0.4681, giving a 0.854 s cycle
  # for a 0.4 s swing and a 0.427 s step period.
  _GAIT_DOUBLE_SUPPORT_BAND = 0.1
  _GAIT_SWING_TIME = 0.4
  _GAIT_SWING_FRACTION = 0.5 - np.arcsin(_GAIT_DOUBLE_SUPPORT_BAND) / np.pi
  _GAIT_CYCLE_TIME = float(_GAIT_SWING_TIME / _GAIT_SWING_FRACTION)
  if USE_GAIT_PHASE:
    for _group in ("actor", "critic"):
      cfg.observations[_group].terms["gait_phase"] = ObservationTermCfg(
        func=gait_phase,
        params={"cycle_time": _GAIT_CYCLE_TIME},
      )

  # ---------------------------- Rewards ---------------------------
  # Reuse one velocity-command cutoff for upright target selection, pose,
  # foot clearance, air time, and foot slip.
  walking_threshold = 0.01
  max_swing_time = 0.55

  # Track commanded velocity continuously with mjlab's built-in rewards.
  cfg.rewards["track_linear_velocity"].func = mdp.track_linear_velocity
  cfg.rewards["track_linear_velocity"].params = {
    "command_name": "twist",
    "std": np.sqrt(0.1),
  }
  cfg.rewards["track_linear_velocity"].weight = 3.0
  cfg.rewards["track_angular_velocity"].func = mdp.track_angular_velocity
  cfg.rewards["track_angular_velocity"].params = {
    "command_name": "twist",
    "std": np.sqrt(0.2),
  }
  cfg.rewards["track_angular_velocity"].weight = 2.0

  # Keyed by joint-name regex. ".*hip_roll.*" etc. also match the passive
  # *_cap/*_actual joints, so the asset_cfg below narrows the term to the
  # 25 actuated joints first.
  #
  # Every one of those 25 needs an entry: mjlab's resolve_matching_names_values
  # silently drops joints no key fullmatches, which would leave the std tensor
  # shorter than asset_cfg.joint_ids and blow up on the first reward step. The
  # shoulder_yaw / wrist_pitch / neck_yaw / head_pitch keys below exist for that
  # reason -- they are new on this robot revision.
  #
  # neck/head stay tight in every band: the head carries no gait function here, so
  # the pose term's job is just to stop the policy from flailing it around.
  std_standing = {
    r".*shoulder_pitch.*": 0.1,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist_pitch.*": 0.1,
    r"neck_yaw": 0.1,
    r"head_pitch": 0.1,
    r".*hip_pitch.*": 0.15,
    r".*hip_yaw.*": 0.1,
    # Left/right knee sit asymmetrically bent (+/-30deg, see HOME_KEYFRAME).
    r".*knee.*": 0.15,
    r".*ankle_pitch.*": 0.1,
  }
  std_walking = {
    r".*shoulder_pitch.*": 0.4,
    r".*shoulder_roll.*": 0.2,
    r".*shoulder_yaw.*": 0.2,
    r".*elbow.*": 0.2,
    r".*wrist_pitch.*": 0.2,
    r"neck_yaw": 0.1,
    r"head_pitch": 0.1,
    r".*hip_pitch.*": 0.4,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.4,
    r".*ankle_pitch.*": 0.3,
  }
  # Active from the start: with lin_vel_x widened to +/-0.8 and running_threshold
  # at 0.7, a forward command alone crosses into this band (~48% of normal envs,
  # and nearly all rotation envs). Loosened ~1.5x over
  # std_walking, with the sagittal (pitch) joints given the most slack since they
  # do the swinging; roll/yaw stay comparatively tight to discourage the legs
  # splaying sideways at speed.
  std_running = {
    r".*shoulder_pitch.*": 0.6,
    r".*shoulder_roll.*": 0.3,
    r".*shoulder_yaw.*": 0.3,
    r".*elbow.*": 0.3,
    r".*wrist_pitch.*": 0.3,
    r"neck_yaw": 0.15,
    r"head_pitch": 0.15,
    r".*hip_pitch.*": 0.6,
    r".*hip_yaw.*": 0.3,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.45,
  }
  # The template's 1.5 is sized for a ~1.3 m humanoid and is unreachable here on
  # linear velocity alone: 1.5 m/s is Froude ~0.9 for this robot's 0.28 m legs,
  # i.e. a sprint. At 1.5 the "running" band was effectively a rotation-only
  # regime. 0.7 sits inside the widened lin_vel_x range (Froude ~0.18), so a pure
  # forward command crosses into it well before the curriculum stage fires.
  running_threshold = 0.7

  # Leg roll joints (hip/ankle) are left out of the pose prior. Their actuated
  # joint is the 4-bar crank, which turns about twice as far as the leg itself,
  # so a 0.2 rad std there only allowed ~6deg of real hip roll -- far short of
  # the ~16deg a lateral step needs.
  cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=(POSE_DOFS_FILTER,)
  )
  cfg.rewards["pose"].params["std_standing"] = std_standing
  cfg.rewards["pose"].params["std_walking"] = std_walking
  cfg.rewards["pose"].params["std_running"] = std_running
  cfg.rewards["pose"].params["walking_threshold"] = walking_threshold
  cfg.rewards["pose"].params["running_threshold"] = running_threshold
  cfg.rewards["pose"].weight = 0.5

  # Reward proximity to HOME_KEYFRAME during stand commands.
  # The target comes from robot.data.default_joint_pos; walking is unaffected.
  cfg.rewards["default_joint_pose"] = RewardTermCfg(
    func=default_joint_pose_exp,
    weight=1.0,
    params={
      "std": 0.15,
      "command_name": "twist",
      "walking_threshold": walking_threshold,
      "asset_cfg": SceneEntityCfg("robot", joint_names=(DOFS_FILTER,)),
    },
  )

  cfg.rewards["upright"].func = local_upright
  cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["upright"].params["pitch"] = np.deg2rad(0.0)
  cfg.rewards["upright"].params["standing_pitch"] = 0.0
  cfg.rewards["upright"].params["std"] = 0.1
  cfg.rewards["upright"].params["command_threshold"] = walking_threshold
  # Bumped 1.0 -> 2.0: the fell_over-dominated run showed upright contributing
  # almost nothing (Episode_Reward/upright ~0.03) next to termination (~-0.99)
  # and self_collisions (~-0.62) -- doubling it to push staying-upright harder
  # before the policy ever gets punished for falling.
  cfg.rewards["upright"].weight = 1.0

  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  # Same quantity as cyclo_lab's ang_vel_xy_l2 (base roll/pitch rate squared),
  # at that task's weight.
  cfg.rewards["body_ang_vel"].weight = -0.4
  cfg.rewards["angular_momentum"].weight = -0.1

  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = tuple(foot_site_names)

  # Target swing height scaled to this robot: ~0.476 m tall with ~0.14 m shins,
  # vs the template's 0.1 m default sized for a ~1.3 m humanoid.
  cfg.rewards["foot_clearance"].params["command_threshold"] = walking_threshold
  cfg.rewards["foot_clearance"].params["target_height"] = 0.03
  # This is not a positive "lift the foot" reward. Its cost is
  # |height - target| * foot_xy_speed, so it teaches an already-moving swing
  # foot to travel near 3 cm clearance. The one-shot feet_crossing term below
  # supplies the positive lift signal immediately at the 3 cm threshold.
  cfg.rewards["foot_clearance"].weight = -0.1
  # The stock foot_swing_height term evaluates the peak only when the foot
  # lands.  Remove it: reaching 3 cm is rewarded immediately by feet_crossing.
  del cfg.rewards["foot_swing_height"]

  cfg.rewards["air_time"].func = feet_air_time_continuous_reward
  cfg.rewards["air_time"].params = {
    "sensor_name": FEET_GROUND_SENSOR_CFG.name,
    "threshold_min": 0.3,
    "threshold_max": max_swing_time,
    "command_name": "twist",
    "command_threshold": walking_threshold,
    "ramp_from": 0.0,
  }
  # Pay every step while exactly one foot has been airborne for 0.3--0.7 s.
  # Before 0.3 s the per-step pay ramps linearly from 0 (just lifted) to 1, so
  # a short swing still earns more the longer the foot stays up.
  # Landing and overlong swings receive no air-time reward.
  cfg.rewards["air_time"].weight = 1.5
  # Two stepping terms from ROBOTIS cyclo_lab's K1 velocity task (which walks
  # without a gait clock), with that task's weights. The stand test is this
  # task's own (planar speed + |yaw rate| <= walking_threshold).
  #
  # feet_single_contact: 1 on every step that is single stance or within 0.2 s
  # of a touchdown. Standing on both feet longer than that pays 0, so only
  # continued stepping keeps it. It needs no memory of which foot stepped last.
  cfg.rewards["feet_single_contact"] = RewardTermCfg(
    func=feet_single_contact_reward,
    weight=1.0,
    params={
      "sensor_name": FEET_GROUND_SENSOR_CFG.name,
      "command_name": "twist",
      "command_threshold": walking_threshold,
      "window_s": 0.2,
    },
  )
  # feet_air_time_touchdown: at each landing pays (air time - max_swing_time),
  # so the shorter the swing, the more the landing costs.
  cfg.rewards["feet_air_time_touchdown"] = RewardTermCfg(
    func=feet_air_time_touchdown_reward,
    weight=0.0,
    params={
      "sensor_name": FEET_GROUND_SENSOR_CFG.name,
      "command_name": "twist",
      "command_threshold": walking_threshold,
      "target_air_time": max_swing_time,
    },
  )
  cfg.rewards["both_feet_airborne"] = RewardTermCfg(
    func=both_feet_airborne_penalty,
    weight=-0.0,
    params={
      "sensor_name": FEET_GROUND_SENSOR_CFG.name,
      "min_air_time_s": 0.06,
      "command_name": "twist",
      "command_threshold": walking_threshold,
    },
  )

  cfg.rewards["no_stepping"] = RewardTermCfg(
    func=no_stepping_penalty,
    weight=0.0,
    params={
      "sensor_name": FEET_GROUND_SENSOR_CFG.name,
      "command_name": "twist",
      "command_threshold": walking_threshold,
    },
  )
  # Penalize missing foot contact or base motion during a stand command.
  cfg.rewards["settled_standing"] = RewardTermCfg(
    func=settled_standing_penalty,
    weight=-5.0,
    params={
      "sensor_name": FEET_GROUND_SENSOR_CFG.name,
      "command_name": "twist",
      "command_threshold": walking_threshold,
      "ang_vel_std": 0.3,
      "lin_vel_std": 0.15,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.rewards["not_stepping"] = RewardTermCfg(
    func=not_stepping_penalty,
    # Charge only after one uninterrupted second of double support under
    # a walking command. Landing between steps resets the timer.
    weight=-0.5,
    params={
      "sensor_name": FEET_GROUND_SENSOR_CFG.name,
      "command_name": "twist",
      "command_threshold": walking_threshold,
      "min_duration_s": 1.0,
    },
  )
  # Independently charge each foot that has not lost ground contact for 2 s
  # under a walking command. RewardManager multiplies by step_dt=0.02, so
  # weight=-500 gives an actual one-step cost of -10 per overdue foot.
  cfg.rewards["not_stepping_each_foot"] = RewardTermCfg(
    func=not_stepping_each_foot_penalty,
    weight=-0.0,
    params={
      "sensor_name": FEET_GROUND_SENSOR_CFG.name,
      "command_name": "twist",
      "command_threshold": walking_threshold,
      "max_time_without_lift_s": 2.0,
    },
  )
  cfg.rewards["overlong_swing"] = RewardTermCfg(
    func=overlong_swing_penalty,
    # Penalize air time after the 0.7 s reward window closes.
    weight=-2.0,
    params={
      "sensor_name": FEET_GROUND_SENSOR_CFG.name,
      "max_air_time": max_swing_time,
      "command_name": "twist",
      "command_threshold": walking_threshold,
    },
  )
  del cfg.rewards["soft_landing"]

  cfg.rewards["foot_slip"].func = flatness_weighted_foot_slip_penalty
  cfg.rewards["foot_slip"].params = {
    "sensor_name": FEET_GROUND_SENSOR_CFG.name,
    "command_name": "twist",
    "command_threshold": walking_threshold,
    "flat_slip_scale": 10.0,
    "tilted_slip_scale": 1.0,
    "flat_alignment_threshold": float(np.cos(np.deg2rad(15.0))),
    "asset_cfg": SceneEntityCfg(
      "robot",
      site_names=tuple(foot_site_names),
      body_names=("left_foot_1", "right_foot_1"),
    ),
  }
  # The function carries the 1x--10x state-dependent scale. Keeping the outer
  # weight at -1 makes a flat-foot slip equivalent to the previous -10 weight,
  # while a foot tilted by 15 degrees or more receives the smaller -1 penalty.
  cfg.rewards["foot_slip"].weight = -1.0
  cfg.rewards["action_rate_l2"].func = envs_mdp.action_rate_l2
  cfg.rewards["action_rate_l2"].weight = -0.05
  cfg.rewards["action_rate_l2"].params = {}

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=self_collision_cost_excluding_linkage,
    weight=-50.0,
    params={
      "sensor_name": SELF_COLLISION_SENSOR_CFG.name,
      "linkage_sensor_names": tuple(c.name for c in LINKAGE_CONTACT_SENSOR_CFGS),
    },
  )

  # Continuous low-base-height penalty. 0.35m gives real headroom below the
  # ~0.476-0.478m home/standing height (bent-knee default included); a
  # crouch/settle deep enough to start eating into that gap gets a growing penalty
  # while still allowing the policy to attempt recovery.
  cfg.rewards["base_height"] = RewardTermCfg(
    func=base_height_penalty,
    weight=-200.0,
    params={"minimum_height": 0.35},
  )

  # A hard cap on how far the upper body may swing off HOME_KEYFRAME, covering
  # torso_yaw, both arms and neck/head (elbows default to +/-130deg folded), and
  # their physical ranges are far too wide for dof_pos_limits (90% of physical
  # range) to ever engage: torso_yaw and shoulder_yaw are +/-180deg, elbow_pitch
  # is -143..120deg. Without this term nothing stops the policy from throwing an
  # arm behind its back or cranking the head sideways to game a reward.
  #
  # Each limit equals that joint's std_running above, so this is a backstop and
  # not a second shaping term: `pose` does the shaping inside the band, and this
  # only fires once a joint leaves the range `pose` would ever ask for. It also
  # stays clear of arm_swing, whose shoulder-pitch target is only 0.25 rad.
  #
  # wrist_pitch's range (-1.34..0.119) is asymmetric, so a symmetric cap really
  # only constrains the flexion side; the extension side is already bounded by
  # the joint itself.
  #
  # weight matches base_height's (same "0 up to the line, then linear and steep"
  # pattern) since a flailing upper body is just as much a "something's wrong"
  # signal as sinking below the height floor. The term sums excess across the
  # 13 selected joints, but in normal operation every one contributes exactly 0.
  cfg.rewards["upper_body_excursion"] = RewardTermCfg(
    func=upper_body_excursion_penalty,
    weight=-1.0,
    params={
      # Tight on purpose: the upper body should barely move off HOME. 0.05 rad
      # is ~3 deg. Three joints sag under gravity with zero action (measured:
      # elbow 0.09-0.10 rad folded at 130 deg, head_pitch 0.07, shoulder_pitch
      # 0.05), so their caps sit just above that -- a tighter one would charge
      # the resting pose on every step.
      "max_excursion": {
        r".*torso_yaw.*": 0.05,
        r".*shoulder_pitch.*": 0.1,
        r".*shoulder_roll.*": 0.05,
        r".*shoulder_yaw.*": 0.05,
        r".*elbow.*": 0.15,
        r".*wrist_pitch.*": 0.05,
        r"neck_yaw": 0.05,
        r"head_pitch": 0.1,
      },
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=(
          r".*torso_yaw.*",
          r".*shoulder_pitch.*",
          r".*shoulder_roll.*",
          r".*shoulder_yaw.*",
          r".*elbow.*",
          r".*wrist_pitch.*",
          r"neck_yaw",
          r"head_pitch",
        ),
      ),
    },
  )

  # During straight forward motion, score both shoulder pitches against the
  # airborne foot and still penalize the squared raw actions of the other
  # eight arm joints, so the elbows stay folded while the shoulders swing.
  # Otherwise penalize the squared raw actions of all ten arm joints.
  cfg.rewards["arm_swing"] = RewardTermCfg(
    func=airborne_foot_arm_swing_reward,
    weight=0.5,
    params={
      "sensor_name": FEET_GROUND_SENSOR_CFG.name,
      "target_angle": 0.15,
      "std": 0.25,
      "command_name": "twist",
      "min_forward_command": 0.1,
      "max_lateral_command": 0.1,
      "max_yaw_command": 0.05,
      "action_name": "joint_pos",
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  # Fires once, on the step an episode ends by failure. `is_terminated` reads
  # termination_manager.terminated, which excludes the time_out term, so surviving
  # the full 20 s is not penalised -- only non_foot_contact and, when enabled,
  # out_of_terrain_bounds are.
  # scale_rewards_by_dt is on, so the weight is multiplied by step_dt (0.02):
  # the effective one-off penalty is -20, against a per-step budget where the
  # largest positive term (track_linear_velocity, weight 2.0) contributes 0.04.
  cfg.rewards["termination"] = RewardTermCfg(
    func=envs_mdp.is_terminated,
    weight=-400.0,
    params={},
  )
  # air_time pays each foot out independently, so hopping twice on one leg
  # scores the same as two alternating steps. This is the term that notices,
  # firing once per landing that repeats the foot that landed last.
  #
  # Weight is deliberately small to start. This is a gait-shaping nudge, not a
  # hard constraint: the policy currently cannot hold a gait at all, and a large
  # penalty here would suppress stepping entirely rather than teach alternation
  # (no_stepping already pushes the other way, and the two would fight). Raise it
  # once Episode_Reward/same_foot_repeat is nonzero and episodes survive long
  # enough for a gait cycle to exist.
  # Humanoid-Gym's feet_contact_number, unchanged: score each foot against the
  # stance mask the clock implies. Standing through a walk command scores 0.35
  # (one foot always agrees with an alternating mask), correct alternation
  # scores 1.0 -- so the weight has to be big enough that the 0.65 gap beats
  # what standing already collects elsewhere. This is the term that is meant to
  # do the work the -30 penalties could not. Note same_foot_repeat (-30) now
  # says the same thing far more bluntly and is the weight that turned
  # standing into falling; consider dropping it back toward -1 so the clock,
  # not the penalty, is what shapes alternation.
  # 2026-09-30: 위상 보상 두 개는 gait_phase 관측이 만드는 시계를 읽으므로 USE_GAIT_PHASE 일 때만
  # 정의한다(시계가 없으면 assert 로 멈추고, 위상을 못 보는 정책은 위상 맞춤 보상을 배울 수 없다).
  if USE_GAIT_PHASE:
    cfg.rewards["gait_phase_contact"] = RewardTermCfg(
      func=gait_phase_contact_reward,
      weight=0.5,
      params={
        "sensor_name": FEET_GROUND_SENSOR_CFG.name,
        "command_name": "twist",
        "command_threshold": walking_threshold,
        "double_support_band": _GAIT_DOUBLE_SUPPORT_BAND,
      },
    )

    # Give immediate progress toward the clock-selected swing, before the
    # one-off 3 cm crossing reward and 0.2 s air-time reward can activate.
    cfg.rewards["gait_phase_swing_clearance"] = RewardTermCfg(
      func=gait_phase_swing_clearance_reward,
      weight=0.0,
      params={
        "sensor_name": FEET_GROUND_SENSOR_CFG.name,
        "height_sensor_name": FOOT_HEIGHT_SCAN_CFG.name,
        "command_name": "twist",
        "command_threshold": walking_threshold,
        "double_support_band": _GAIT_DOUBLE_SUPPORT_BAND,
        "min_height": 0.005,
        "target_height": 0.03,
      },
    )

  # Pay only for new lift height, then once for a matched alternating pair
  # of landed swings. Holding one foot up does not pay each step.
  cfg.rewards["swing_progress"] = RewardTermCfg(
    func=swing_progress_reward,
    weight=1.0,
    params={
      "sensor_name": FEET_GROUND_SENSOR_CFG.name,
      "height_sensor_name": FOOT_HEIGHT_SCAN_CFG.name,
      "min_height": 0.005,
      "target_height": 0.03,
      "command_name": "twist",
      "command_threshold": walking_threshold,
      "height_std": 0.01,
      "pair_bonus_scale": 5.0,
      # No reward while the base is tilted more than 30deg in roll or pitch:
      # a foot lifted by a toppling trunk is not a step.
      "max_tilt": float(np.deg2rad(30.0)),
    },
  )

  cfg.rewards["same_foot_repeat"] = RewardTermCfg(
    func=same_foot_repeat_penalty,
    weight=-30.0,
    params={
      "sensor_name": FEET_GROUND_SENSOR_CFG.name,
      "command_name": "twist",
      "command_threshold": walking_threshold,
    },
  )

  # same_foot_repeat above covers the *order* of a gait (left, right, left).
  # This covers its symmetry: that the two legs do the same amount of work.
  # See gait_symmetry_penalty's docstring for why a literal mirror penalty on
  # joint angles is the wrong tool here -- it would reward hopping, since a
  # walking pose is *supposed* to be half a cycle asymmetric. Left/right
  # mirroring of the actual trajectories is handled at the policy level
  # instead, by the symmetry_cfg on the agent config.
  #
  # Weights kept in the same small "gait-shaping nudge" range as
  # same_foot_repeat: the policy has to hold a gait at all before an
  # asymmetry penalty has anything meaningful to act on. Raise once
  # Episode_Reward/gait_symmetry is nonzero and episodes last a few cycles.
  cfg.rewards["gait_symmetry"] = RewardTermCfg(
    func=gait_symmetry_reward,
    weight=10.0,
    params={
      "sensor_name": FEET_GROUND_SENSOR_CFG.name,
      "asset_cfg": SceneEntityCfg("robot", site_names=tuple(foot_site_names)),
      "command_name": "twist",
      "command_threshold": walking_threshold,
      # Per-gap Gaussian widths; the two gaps are in different units, so they
      # get their own std rather than a shared linear scale. 0.1 s of swing-time
      # mismatch and 0.05 m of step-length mismatch (matching forward_step's own
      # std) each read as "clearly limping".
      "duration_std": 0.1,
      "length_std": 0.05,
      "min_double_air_time_s": 0.06,
    },
  )

  cfg.rewards["feet_distance"] = RewardTermCfg(
    func=feet_distance_penalty,
    weight=-1.0,
    params={
      "min_dist": _FOOT_SEPARATION_MIN,
      "max_dist": _FOOT_SEPARATION_MAX,
      "asset_cfg": SceneEntityCfg("robot", site_names=tuple(foot_site_names)),
    },
  )

  # Session testing confirmed each foot body's local +z axis points straight up
  # in world frame when that foot sits flat -- dotting it with world-up gives a
  # direct sole-parallel-to-ground measure (see foot_flatness_penalty docstring).
  # Gated to stance only (sensor_name) -- a swing foot is *supposed* to tilt
  # during toe-off/mid-air/heel-strike, and charging it for that fights every
  # other stepping-related reward in this file that's trying to get the
  # policy to actually lift its feet.
  # Charge every step for each non-stud foot geom on the ground (the sole
  # plate's toe edge, the ankle block): walk on the studs, not on the toe.
  cfg.rewards["foot_non_stud_contact"] = RewardTermCfg(
    func=foot_non_stud_contact_penalty,
    weight=0.0,
    params={"sensor_name": FOOT_NON_STUD_CONTACT_SENSOR_CFG.name},
  )
  cfg.rewards["foot_flatness"] = RewardTermCfg(
    func=foot_flatness_penalty,
    weight=-3.0,
    params={
      "sensor_name": FEET_GROUND_SENSOR_CFG.name,
      "asset_cfg": SceneEntityCfg("robot", body_names=("left_foot_1", "right_foot_1")),
    },
  )

  # Reward sole alignment with the actual terrain normal only on landing.
  cfg.rewards["foot_landing_alignment"] = RewardTermCfg(
    func=foot_landing_alignment_reward,
    weight=2.0,
    params={
      "sensor_name": FEET_GROUND_SENSOR_CFG.name,
      "height_sensor_name": FOOT_HEIGHT_SCAN_CFG.name,
      "asset_cfg": SceneEntityCfg("robot", body_names=("left_foot_1", "right_foot_1")),
      "min_double_air_time_s": 0.06,
    },
  )

  # Pay once per swing when a foot first clears 3 cm; rearm on landing.
  # Keep the one-off bonus when the next qualifying foot is the opposite one.
  cfg.rewards["feet_crossing"] = RewardTermCfg(
    func=feet_crossing_reward,
    weight=2.0,
    params={
      "sensor_name": FEET_GROUND_SENSOR_CFG.name,
      "height_sensor_name": FOOT_HEIGHT_SCAN_CFG.name,
      "min_swing_height": 0.03,
      "alternation_bonus": 1.0,
      "command_name": "twist",
      "command_threshold": walking_threshold,
      "min_double_air_time_s": 0.06,
    },
  )

  # One-off reward when a foot lands with an x/y displacement matching the
  # body-frame planar velocity command. This covers forward, backward, lateral,
  # and diagonal steps; see forward_step_reward for the frame conversion.
  cfg.rewards["forward_step"] = RewardTermCfg(
    func=forward_step_reward,
    weight=0.0,
    params={
      "sensor_name": FEET_GROUND_SENSOR_CFG.name,
      "height_sensor_name": FOOT_HEIGHT_SCAN_CFG.name,
      "asset_cfg": SceneEntityCfg("robot", site_names=tuple(foot_site_names)),
      "command_name": "twist",
      "command_threshold": walking_threshold,
      # Gaussian width on the 2-D step-vector error. 0.05 m is ~16% of
      # the 0.31 m stride the retargeted human walk shows for this robot at
      # 0.32 m/s, so a step has to land within a few centimetres of what the
      # command implies to score most of the term. With the 5 mm
      # min_step_distance below, 0.1 paid ~57% for a foot that barely moved
      # against an 8 cm target; 0.05 pays ~10% there and 53% at 4 cm.
      "std": 0.05,
      "min_step_distance": 0.005,
      "min_swing_height": 0.03,
    },
  )

  # The in-progress counterpart of forward_step for sideways commands. Pays
  # each step's *new* ground the swing foot gains in the commanded lateral
  # direction, past its own furthest previous landing, normalised by a target
  # step of |vy| * 0.7 s (clamped to 2..12 cm). Only the foot opposite the last
  # landing earns it, and only while it is at least 3 cm up with the other foot
  # planted -- reaching a leg out and back, or dragging a foot, pays nothing.
  # A full target step pays weight * dt = 1.0. The base-link share of the term
  # is switched off (base_scale 0); base motion is left to velocity tracking.
  cfg.rewards["lateral_swing_progress"] = RewardTermCfg(
    func=lateral_swing_progress_reward,
    weight=30.0,
    params={
      "sensor_name": FEET_GROUND_SENSOR_CFG.name,
      "height_sensor_name": FOOT_HEIGHT_SCAN_CFG.name,
      "asset_cfg": SceneEntityCfg("robot", site_names=tuple(foot_site_names)),
      "command_name": "twist",
      "command_threshold": walking_threshold,
      "nominal_cycle_time": 0.7,
      "min_target": 0.02,
      "max_target": 0.12,
      "base_scale": 0.0,
      "min_swing_height": 0.03,
    },
  )

  cfg.rewards["velocity_shortfall"] = RewardTermCfg(
    func=velocity_shortfall_penalty,
    weight=-0.0,
    params={"command_name": "twist"},
  )
  cfg.rewards["relative_linear_velocity_error"] = RewardTermCfg(
    func=relative_linear_velocity_error_penalty,
    weight=-0.5,
    params={
      "command_name": "twist",
      "command_threshold": walking_threshold,
      "max_relative_error": 2.0,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.rewards["relative_angular_velocity_error"] = RewardTermCfg(
    func=relative_angular_velocity_error_penalty,
    weight=-0.5,
    params={
      "command_name": "twist",
      "command_threshold": 0.05,
      "max_relative_error": 2.0,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.rewards["foot_base_heading_error"] = RewardTermCfg(
    func=foot_base_heading_error_penalty,
    weight=-1.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=tuple(foot_site_names)),
    },
  )
  cfg.rewards["joint_torques_l2"] = RewardTermCfg(
    func=envs_mdp.joint_torques_l2,
    weight=-1.0e-4,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )
  cfg.rewards["action_acc_l2"] = RewardTermCfg(
    func=envs_mdp.action_acc_l2,
    weight=-0.0,
    params={},
  )
  cfg.rewards["roll_action_excess_l2"] = RewardTermCfg(
    func=selected_action_excess_l2,
    weight=-0.1,
    params={
      "action_name": "joint_pos",
      "target_names": (
        "left_hip_roll_crank",
        "right_hip_roll_crank",
        "left_ankle_roll_crank",
        "right_ankle_roll_crank",
      ),
      "max_abs_action": 2.0,
    },
  )

  # ---------------------------- Commands --------------------------
  # Swap in the real Cfg subclass instead of attaching fields and a build
  # override to the template's UniformVelocityCommandCfg: train.py re-creates
  # the env cfg through tyro, which keeps only declared dataclass fields, so ad
  # hoc attributes (rel_lateral_only_envs, build, ...) silently vanished and
  # the stock mixed vx/vy/wz sampler ran instead.
  base_command = cfg.commands["twist"]
  command = UniformVelocityCommandWithRotationCfg(
    **{f.name: getattr(base_command, f.name) for f in fields(base_command)}
  )
  cfg.commands["twist"] = command
  command.viz.z_offset = 0.5
  # Match the 20 s episode horizon: reset samples one fresh command and the
  # timeout resets the environment before an in-episode resample can occur.
  # Both bounds are required; ``(20.0)`` would be a float, not a one-item tuple.
  command.resampling_time_range = (20.0, 20.0)
  command.rel_standing_envs = 0.1
  command.rel_heading_envs = 0.0
  command.rel_rotation_envs = 0.1
  # Exclusive fractions of all environments: standing/rotation/forward/backward
  # each take 10%, lateral-only and diagonal planar translation each take 30%.
  # These sum to 100%, leaving no mixed vx/vy/wz commands during training.
  command.rel_forward_only_envs = 0.1
  command.rel_backward_only_envs = 0.1
  command.rel_lateral_only_envs = 0.3
  command.rel_planar_only_envs = 0.3
  command.directional_min_lin_vel = 0.05
  # Use this task's explicit forward-only sampler instead of the template's
  # separate forward-mode distribution.
  command.rel_forward_envs = 0.0
  # Fixed command envelope. Curriculum difficulty comes from push magnitude and
  # tighter tracking tolerances, not from changing the command distribution.
  command.ranges.lin_vel_x = (-0.4, 0.4)
  command.ranges.lin_vel_y = (-0.3, 0.3)
  command.ranges.ang_vel_z = (-0.7, 0.7)
  command.rotation_env_ang_vel_range = (-0.7, 0.7)
  command.rotation_min_ang_vel = 0.3

  # ---------------------------- Events ----------------------------
  cfg.events["reset_base"].params["pose_range"]["z"] = (0.0, 0.01)
  # Sample initial base roll and pitch in either direction.
  init_tilt = float(np.deg2rad(0.0))
  cfg.events["reset_base"].params["pose_range"]["roll"] = (-init_tilt, init_tilt)
  cfg.events["reset_base"].params["pose_range"]["pitch"] = (-init_tilt, init_tilt)
  # Randomize only the actuated leg joints. The passive roll-linkage followers
  # are constrained by the mechanism and must not be offset independently, and
  # the arms/torso/head start at HOME: with the elbows folded to 130deg a
  # +/-0.15 rad offset on the arm joints put a forearm into the torso on ~30%
  # of resets.
  cfg.events["reset_robot_joints"].params.update(
    {
      "position_range": (-0.0, 0.0),
      "velocity_range": (-0.0, 0.0),
      "asset_cfg": SceneEntityCfg("robot", joint_names=(LEG_DOFS_FILTER,)),
    }
  )
  # reset_joints_by_offset is also what puts joints at HOME, so the upper body
  # needs its own zero-offset reset or it would come back at qpos 0 (elbows
  # straight) instead of the folded default.
  cfg.events["reset_upper_body_joints"] = EventTermCfg(
    func=envs_mdp.reset_joints_by_offset,
    mode="reset",
    params={
      "position_range": (0.0, 0.0),
      "velocity_range": (0.0, 0.0),
      "asset_cfg": SceneEntityCfg("robot", joint_names=(UPPER_BODY_DOFS_FILTER,)),
    },
  )
  # Push velocity is zero initially and at every curriculum stage below.
  # DR_WIDE still controls the other DR ranges.
  cfg.events["push_robot"].params["velocity_range"] = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
  }
  cfg.events["push_robot"].interval_range_s = (15.0, 20.0)
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
    r".*left_foot_collision.*",
    r".*right_foot_collision.*",
  )
  # Raised the low end from mjlab's default 0.3 -- that's wet-tile-slippery and
  # was a plausible contributor to fell_over alongside the other issues found
  # this session (double curriculum, action-std runaway). 0.6 still gives real
  # low-friction variety without handing every env a near-ice floor.
  cfg.events["foot_friction"].params["ranges"] = (0.4, 1.2)
  for side in ("left", "right"):
    for joint in ("hip", "ankle"):
      linkage = f"{side}_{joint}"
      cfg.events[f"{linkage}_contact_friction"] = EventTermCfg(
        mode="startup",
        func=dr.geom_friction,
        params={
          "asset_cfg": SceneEntityCfg(
            "robot",
            geom_names=(
              f"{linkage}_groove_wall1",
              f"{linkage}_groove_wall2",
              f"{linkage}_cap",
            ),
          ),
          "operation": "scale",
          "ranges": (0.1, 1.3),
        },
      )
  # 5 mm is ~1% of this 0.476 m robot's height, arguably tighter than the real
  # build/assembly tolerance on torso payload placement; the widened arm doubles
  # it and still stays well inside the support polygon at nominal stance.
  _com = (-0.02, 0.02) if DR_WIDE else (-0.01, 0.01)
  cfg.events["base_com"].params["ranges"] = {0: _com, 1: _com, 2: _com}
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_1",)

  # Body mass: +/-20% (or +/-40% widened) on every body via pseudo_inertia's
  # alpha param -- a
  # physically-consistent density scale (mass and inertia move together, COM
  # unchanged), rather than dr.body_mass's plain mass-only scale (which the
  # function's own docstring warns leaves body_inertia stale/inconsistent).
  #
  # torso_1 is excluded here and randomized separately below with dr.body_mass
  # instead: pseudo_inertia recomputes body_ipos from the model's *default* even
  # when only alpha is set (its COM-shift params are all 0), so running it on
  # torso_1 here would silently reset base_com's ipos offset back to default --
  # regardless of which event runs first, since both "add" (base_com) and this
  # call read from the compile-time default rather than each other's output (see
  # dr's Operation.uses_defaults). dr.body_mass only touches body_mass, so it
  # can't clobber base_com's ipos no matter the order.
  # Soft joint limits = hard limits minus min(7deg, 5% of range) per side,
  # instead of the articulation's range-proportional 0.9 factor. This lets the
  # 130deg folded-elbow default sit inside the soft limit (140 - 7 = 133deg).
  # Balance assist: an external spring-damper torque that pushes a tilted
  # trunk back upright, so the policy can try lifting a foot without falling.
  # It is spread over every body as a rigid-body rotation and applied each
  # physics step -- on the 0.56 kg base link alone, or refreshed once per
  # control step, these gains make the simulation diverge.
  # A manual switch, not a curriculum:
  #   - BALANCE_ASSIST_SCALE below: 0 = off, 1 = full strength.
  #   - while training is running, write a number into BALANCE_ASSIST_SCALE_FILE
  #     (e.g. `echo 0.5 > balance_assist_scale.txt` in the repo root) to change
  #     it without restarting; it is re-read every 200 env steps and overrides
  #     the value here. Delete the file to fall back to BALANCE_ASSIST_SCALE.
  # Gains are set for a critically damped return at scale 1, modelling the
  # robot as an inverted pendulum tipping about its feet: m*g*h = 24.4 Nm/rad
  # (6.32 kg, COM 0.394 m up) and I = 1.21 kg m^2 about a ground pivot (roll
  # 1.215, pitch 1.196). With a natural frequency of 7 rad/s (~0.7 s to settle):
  #   stiffness = m*g*h + I*wn^2 = 83.5,  damping = 2*I*wn = 16.9
  # The damping ratio stays 0.92-1.02 down to scale 0.4; below scale 0.29 the
  # assist is weaker than gravity and only slows a fall.
  # Not used in play, and a policy must finish training at 0.
  if not play and BALANCE_ASSIST_SCALE is not None:
    cfg.events["balance_assist"] = EventTermCfg(
      func=balance_assist,
      mode="step",
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "scale": BALANCE_ASSIST_SCALE,
        "stiffness": 83.5,
        "damping": 16.9,
        "max_torque": 40.0,
        "scale_file": BALANCE_ASSIST_SCALE_FILE,
        "reload_every": 200,
      },
    )
  cfg.events["soft_joint_pos_limit_margin"] = EventTermCfg(
    mode="startup",
    func=set_soft_joint_pos_limit_margin,
    params={"margin": float(np.deg2rad(7.0)), "max_range_fraction": 0.05},
  )
  cfg.events["body_mass_randomization"] = EventTermCfg(
    mode="startup",
    func=dr.pseudo_inertia,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(r"^(?!torso_1$).*$",)),
      # alpha is a log-scale: mass and inertia both scale by exp(2*alpha), so
      # alpha = ln(scale)/2 -- the two arms are scale ~(0.6, 1.4) and ~(0.8, 1.2),
      # matching torso_mass_randomization below.
      "alpha_range": (-0.2554, 0.1682) if DR_WIDE else (-0.1116, 0.0912),
    },
  )
  cfg.events["torso_mass_randomization"] = EventTermCfg(
    mode="startup",
    func=dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_1",)),
      "operation": "scale",
      "ranges": (0.6, 1.4) if DR_WIDE else (0.8, 1.2),
    },
  )

  # Split armature randomization into disjoint joint groups so the four
  # *_cap and four *_actual passive joints can each use their own scale range.
  # All eight have armature=1e-3 in the current robot XML.
  _actuated_armature_range = (0.6, 1.4) if DR_WIDE else (0.9, 1.1)
  _cap_armature_range = (0.6, 1.4) if DR_WIDE else (0.5, 1.3)
  _actual_armature_range = (0.6, 1.4) if DR_WIDE else (0.5, 1.3)
  cfg.events["dof_armature_randomization"] = EventTermCfg(
    mode="startup",
    func=dr.joint_armature,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(DOFS_FILTER,)),
      "operation": "scale",
      "ranges": _actuated_armature_range,
    },
  )
  cfg.events["dof_cap_armature_randomization"] = EventTermCfg(
    mode="startup",
    func=dr.joint_armature,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*_cap$",)),
      "operation": "scale",
      "ranges": _cap_armature_range,
    },
  )
  cfg.events["dof_actual_armature_randomization"] = EventTermCfg(
    mode="startup",
    func=dr.joint_armature,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*_actual$",)),
      "operation": "scale",
      "ranges": _actual_armature_range,
    },
  )

  # BAM owns the 25 driven joints' friction. The passive cap/actual joints
  # are not BAM targets, so randomize their XML-authored frictionloss directly.
  _cap_friction_range = (0.5, 1.5) if DR_WIDE else (0.3, 1.2)
  _actual_friction_range = (0.5, 1.5) if DR_WIDE else (0.3, 1.2)
  cfg.events["dof_cap_friction_randomization"] = EventTermCfg(
    mode="startup",
    func=dr.joint_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*_cap$",)),
      "operation": "scale",
      "ranges": _cap_friction_range,
    },
  )
  cfg.events["dof_actual_friction_randomization"] = EventTermCfg(
    mode="startup",
    func=dr.joint_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*_actual$",)),
      "operation": "scale",
      "ranges": _actual_friction_range,
    },
  )

  # 2026-09-29: BAM 펌웨어 kp 를 모터 그룹마다 env 별로 흔든다 (BamActuator.set_gains).
  # duty = kp * error_gain * error 라 kp 배율은 error_gain 오차도 같이 덮는다.
  # 실기 로그를 같은 target 으로 sim 에 재생한 비교(deploy_ws kid_rl/docs/
  # SIM2REAL_OSCILLATION_2026-09-28.md):
  #   - MX-106 (hip/ankle roll): 땅에서 하중을 받으면 sim 이 덜 따라감 (오른 hip roll 43% vs
  #     실기 71%, ankle 23% vs 63%). 파라미터가 프로토콜 1.0 에서 식별됐고 1.0->2.0 duty 환산이
  #     약 2배 어긋나 있어 1.0~2.0 로 넓게 잡는다.
  #   - XH540 (다리 pitch): KP divisor 가 데이터시트 128 그대로(같은 2.0 계열 MX-64/106 은 실측
  #     141/147) -- 강한 쪽으로 1.0~1.3.
  #   - MX-64 (torso/hip yaw): 실기가 sim 보다 덜 움직임(느린 동작 추종비 0.15 vs 0.30) --
  #     약한 쪽을 포함해 0.7~1.1.
  #   - MX-28 (팔/목): 비교 근거 없음, 0.9~1.2.
  cfg.events["bam_kp_randomization"] = EventTermCfg(
    mode="startup",
    func=randomize_bam_kp,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      # 2026-09-30: 위 비대칭 범위의 근거("sim 이 하중에 약함")는 BAM motor 변환 후 남은
      # ctrlrange(관절 각도 범위)가 토크를 1~2 Nm 로 자르던 버그였다. 수정 후에는 같은 target
      # 에 대한 roll 추종이 실기와 비슷하므로 모든 모터를 기준 kp(27/46/102/149) 대비 0.8~1.2 로.
      "kp_scale_ranges": {
        "mx106v2": (0.8, 1.2),
        "xh": (0.8, 1.2),
        "mx64v2": (0.8, 1.2),
        "mx28": (0.8, 1.2),
      },
    },
  )

  # BAM writes per-environment dof_frictionloss/dof_damping, which requires those
  # model fields to be expanded per world before stepping.
  cfg.events["bam_init"] = EventTermCfg(func=bam_init, mode="startup")

  # Diagnostic only (no physics effect): logs njmax/nconmax buffer utilization
  # every step to Metrics/njmax_overflow_frac, Metrics/njmax_util_mean,
  # Metrics/nconmax_overflow, Metrics/nconmax_util -- so overflow (silently
  # dropped contacts/constraints -> energy gain -> falls, see SIM_CFG comment
  # above) can be confirmed or ruled out directly instead of inferred from
  # episode length / fell_over, which moved the *wrong* way after the
  # 150/620 -> 300/1536 bump in one run and can't be trusted alone. Remove
  # once nconmax/njmax headroom is settled.
  cfg.events["log_solver_buffer_usage"] = EventTermCfg(
    func=log_solver_buffer_usage, mode="step"
  )
  # Log the exact per-transition reward contributions used by PPO, averaged
  # over every environment and every rollout step. Unlike Episode_Reward/*,
  # these values do not wait for an episode to terminate.
  if not play:
    enable_step_reward_logging(cfg)

  # ---------------------------- Curriculum ------------------------
  # Keep the velocity-command distribution and tracking std values fixed.
  # Only the push magnitude increases through this curriculum.
  # Remove the template command curriculum so it cannot overwrite the fixed
  # command ranges above; terrain curriculum remains enabled.
  del cfg.curriculum["command_vel"]
  cfg.curriculum["staged_curriculum"] = CurriculumTermCfg(
    func=reward_based_staged_curriculum,
    params={
      "stages": make_interpolated_push_stages(
        num_stages=6,
        reward_term_name="track_linear_velocity",
        threshold_start=0.23,
        threshold_end=3.0,
        push_full_scale={
          "x": 0.5,
          "y": 0.5,
          "roll": 0.5,
          "pitch": 0.5,
          "yaw": 0.5,
        },
      ),
    },
  )

  # Both of these read state that only exists under terrain_type="generator";
  # see the Terrain section above for why each one has to go with a plane.
  if flat:
    cfg.curriculum.pop("terrain_levels", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)

  if play:
    cfg.curriculum = {}
    cfg.commands["twist"].rel_standing_envs = 0.0
    cfg.commands["twist"].rel_rotation_envs = 0.0
    cfg.events["push_robot"].params["velocity_range"] = {
      "x": (0.0, 0.0),
      "y": (0.0, 0.0),
    }
    cfg.observations["actor"].enable_corruption = False

  # Convert last so __post_init__ applies the no-DR override after all normal,
  # flat, and play settings. Tyro also calls it after a CLI field override.
  base_fields = {
    field.name: getattr(cfg, field.name) for field in fields(ManagerBasedRlEnvCfg)
  }
  return KidRLVelocityEnvCfg(
    **base_fields,
    enable_randomization=enable_randomization,
  )


##
# RL algorithm (RSL-RL PPO).
##

# hidden_dims/algorithm hyperparameters are unchanged from microban's config, not
# derived from a formula -- (512, 256, 128) is the de facto default from ETH RSL's
# legged_gym/rsl_rl locomotion work, widely reused as a starting point rather than
# tuned per-robot. Our actor observation is 84-dim / 25-dim action, on par with the
# classic legged-robot setups that size was designed for, so it's neither over- nor
# under-sized here. Revisit empirically (via ablation) if training is slow/unstable,
# not by recomputing from observation size.
# CAPS coefficients are starting values, not validated real-robot tuning.
CAPS_SPATIAL_COEF = 0.1
CAPS_TEMPORAL_COEF = 0.03
CAPS_GYRO_STD = 0.3  # rad/s, before actor observation normalization
CAPS_JOINT_VEL_STD = 0.3
"""CAPS 공간 외란: actor joint_vel 칸(rad/s, 정규화 전). 2026-09-29 실기 분석에서 CAPS 로 gyro 민감도가
1/25 로 줄자 joint_vel 이 남은 주 되먹임 경로였다(칸당·정규화 1 std 당 민감도가 gyro 의 약 2.6배).
학습 정규화 std 평균 0.60 이라 0.3 은 약 0.5 std -- gyro(0.3 = 약 1 std)보다 약하게 흔든다."""


@dataclass
class KidRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """mjlab's PPO config plus RSL-RL's symmetry extension.

  RSL-RL's PPO already accepts a ``symmetry_cfg``, but mjlab's config dataclass
  does not declare the field, and the trainer hands RSL-RL ``asdict(cfg.agent)``
  -- so without this subclass the key never reaches the algorithm and symmetry
  is silently off. See mjlab_kid_rl.tasks.symmetry for the mirror itself.
  """

  symmetry_cfg: dict | None = None
  smooth_cfg: dict | None = None
  """CAPS 공간·시간 손실 설정. mjlab_kid_rl.tasks.smooth_ppo 참고."""


KID_RL_VELOCITY_RL_CFG = RslRlOnPolicyRunnerCfg(
  actor=RslRlModelCfg(
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      # With action scale 0.5 this starts exploration at a 0.30 rad
      # (about 17 deg) joint-target standard deviation. The old value 1.0
      # started at 0.50 rad and the learned std later ran above 1.5.
      "init_std": 1.0,
      "std_type": "scalar",
    },
  ),
  critic=RslRlModelCfg(
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=True,
  ),
  algorithm=KidRlPpoAlgorithmCfg(
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
    entropy_coef=0.01,
    num_learning_epochs=5,
    num_mini_batches=4,
    # A fixed, conservative rate prevents KL spikes from repeatedly pinning
    # training to the adaptive scheduler's 1e-5 floor. 3e-4 is intentionally
    # below the previous 1e-3 initial rate while remaining large enough for the
    # rare early stepping successes to change the mean policy.
    learning_rate=3.0e-4,
    schedule="fixed",
    gamma=0.99,
    lam=0.95,
    desired_kl=0.01,
    max_grad_norm=1.0,
    # Same-time spatial mirror regularization. This enforces
    # pi(mirror(obs)) ~= mirror(pi(obs)); it does not impose a half-cycle gait.
    # Keep the coefficient small so it regularizes left/right responses without
    # overwhelming the asymmetric action needed to initiate a swing.
    # 2026-09-28: gyro 가 조금 변할 때 action 이 크게 변하지 않도록 벌한다(smooth_ppo.py).
    # CAPS 공간(gyro 외란)·시간(실제 연속 관측) 손실을 함께 적용한다.
    # slices 는 actor 관측의 base_ang_vel 칸(0:3, 첫 항목) -- 관측 순서를 바꾸면 같이 바꿀 것.
    class_name="mjlab_kid_rl.tasks.smooth_ppo:SmoothPPO",
    smooth_cfg={
      "spatial_coef": CAPS_SPATIAL_COEF,
      "temporal_coef": CAPS_TEMPORAL_COEF,
      # actor 관측 칸: base_ang_vel 0:3, joint_vel 31:56 (관측 순서를 바꾸면 같이 바꿀 것)
      "slices": [(0, 3, CAPS_GYRO_STD), (31, 56, CAPS_JOINT_VEL_STD)],
    },
    symmetry_cfg={
      "use_data_augmentation": True,
      "use_mirror_loss": True,
      "mirror_loss_coeff": 0.1,
      "data_augmentation_func": compute_symmetric_states,
    },
  ),
  wandb_project="mjlab_kid_rl_velocity",
  experiment_name="mjlab_kid_rl_velocity",
  save_interval=50,
  num_steps_per_env=24,
  # Bound the raw policy action before it enters the environment or the
  # last-action observation, preventing the unbounded feedback seen in long runs.
  # 2026-09-30: action clip 제거 (None = 자르지 않음). 실기 controllers.yaml action_clip 도 0 으로
  # 맞춰야 한다 -- last_action 관측이 학습과 같은 (자르지 않은) 값이어야 한다.
  clip_actions=None,
  max_iterations=15_000,
)
