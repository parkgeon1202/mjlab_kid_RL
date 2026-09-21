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
  UniformVelocityCommandWithRotation,
  default_joint_pose_exp,
  contralateral_arm_swing,
  feet_distance_penalty,
  base_height_penalty,
  feet_crossing_reward,
  foot_flatness_penalty,
  flatness_weighted_foot_slip_penalty,
  feet_air_time_once_reward,
  forward_step_reward,
  gait_symmetry_reward,
  velocity_shortfall_penalty,
  relative_angular_velocity_error_penalty,
  relative_linear_velocity_error_penalty,
  foot_base_heading_error_penalty,
  log_solver_buffer_usage,
  no_stepping_penalty,
  not_stepping_penalty,
  overlong_swing_penalty,
  same_foot_repeat_penalty,
  self_collision_cost_excluding_linkage,
  set_push_parameters,
  make_interpolated_push_stages,
  reward_based_staged_curriculum,
  selected_action_excess_l2,
  upper_body_excursion_penalty,
  track_angular_velocity_gated,
  track_linear_velocity_gated,
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

_DEFAULT_FOOT_SEPARATION = 0.13425
"""Horizontal distance between the two foot sites in HOME_KEYFRAME [m].

Measured from the compiled model after forward kinematics. The site-local y
coordinates cannot be subtracted directly because the sites belong to different
foot body frames.
"""

_FOOT_SEPARATION_MIN = 0.16
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
      "encoder_bias",
      "base_com",
      "body_mass_randomization",
      "torso_mass_randomization",
      "dof_armature_randomization",
      "dof_friction_randomization",
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
  )

  # ---------------------------- Terminations ----------------------
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
    cfg.sim.nconmax = 192
    cfg.sim.njmax = 768

  # ---------------------------- Actions ---------------------------
  # actuator_names (not joint names): the MJCF defines exactly 25 <position>
  # actuators, one per actuated joint, so ".*" already excludes the passive ones.
  # 25, not the old model's 19: this revision adds a shoulder yaw and a wrist pitch
  # per arm plus neck_yaw/head_pitch, and all of them are policy-driven.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = 0.5
  joint_pos_action.actuator_names = (r".*",)

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
  # delay, which counts simulation steps. The widened arm (1-6 = 20-120 ms)
  # exposes the policy to the occasional late serial-bus read instead of only
  # the nominal one-to-four-tick latency.
  cfg.observations["actor"].terms["joint_pos"] = ObservationTermCfg(
    func=mdp.joint_pos_rel,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(DOFS_FILTER,))},
    noise=Unoise(n_min=-0.002, n_max=0.002),
    delay_min_lag=1,
    delay_max_lag=6 if DR_WIDE else 4,
  )
  cfg.observations["actor"].terms["joint_vel"] = ObservationTermCfg(
    func=mdp.joint_vel_rel,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(DOFS_FILTER,))},
    noise=Unoise(n_min=-0.125, n_max=0.125),
    delay_min_lag=1,
    delay_max_lag=6 if DR_WIDE else 4,
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
    term.delay_min_lag = 1
    term.delay_max_lag = 10 if DR_WIDE else 8
    term.delay_update_period = 64
    cfg.observations["actor"].terms[term_name] = term

  # ---------------------------- Rewards ---------------------------
  # Moved up from the pose-reward block below so the in_air-gated rewards
  # (track_linear/angular_velocity, upright) can share the same "is a
  # nontrivial velocity actually commanded" cutoff as pose/foot_clearance/
  # foot_swing_height/air_time/foot_slip, instead of an independent value.
  walking_threshold = 0.01
  max_swing_time = 0.6

  # Keep the in-air-gated variants so a planted robot cannot collect tracking
  # reward for doing nothing.
  cfg.rewards["track_linear_velocity"].func = track_linear_velocity_gated
  # Keep these values fixed throughout the push curriculum, but leave enough
  # width for early, imperfect steps to receive a useful tracking signal.
  # With std=0.1, a 0.3 m/s error produced exp(-9) ~= 1.2e-4, effectively
  # flattening the reward before the policy had already learned to walk.
  cfg.rewards["track_linear_velocity"].params["std"] = 0.15
  cfg.rewards["track_linear_velocity"].params["sensor_name"] = FEET_GROUND_SENSOR_CFG.name
  cfg.rewards["track_linear_velocity"].params["height_sensor_name"] = FOOT_HEIGHT_SCAN_CFG.name
  cfg.rewards["track_linear_velocity"].params["command_threshold"] = walking_threshold
  cfg.rewards["track_linear_velocity"].params["max_air_time"] = max_swing_time
  cfg.rewards["track_linear_velocity"].weight = 3.0
  cfg.rewards["track_angular_velocity"].func = track_angular_velocity_gated
  cfg.rewards["track_angular_velocity"].params["std"] = 0.15
  cfg.rewards["track_angular_velocity"].params["sensor_name"] = FEET_GROUND_SENSOR_CFG.name
  cfg.rewards["track_angular_velocity"].params["height_sensor_name"] = FOOT_HEIGHT_SCAN_CFG.name
  cfg.rewards["track_angular_velocity"].params["command_threshold"] = walking_threshold
  cfg.rewards["track_angular_velocity"].params["max_air_time"] = max_swing_time
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
    r".*torso_yaw.*": 0.0,
    r".*shoulder_pitch.*": 0.1,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist_pitch.*": 0.1,
    r"neck_yaw": 0.1,
    r"head_pitch": 0.1,
    r".*hip_roll.*": 0.1,
    r".*hip_pitch.*": 0.15,
    r".*hip_yaw.*": 0.1,
    # Left/right knee sit asymmetrically bent (+/-30deg, see HOME_KEYFRAME).
    r".*knee.*": 0.15,
    r".*ankle_pitch.*": 0.1,
    r".*ankle_roll.*": 0.1,
  }
  std_walking = {
    r".*torso_yaw.*": 0.1,
    r".*shoulder_pitch.*": 0.4,
    r".*shoulder_roll.*": 0.2,
    r".*shoulder_yaw.*": 0.2,
    r".*elbow.*": 0.2,
    r".*wrist_pitch.*": 0.2,
    r"neck_yaw": 0.1,
    r"head_pitch": 0.1,
    r".*hip_roll.*": 0.2,
    r".*hip_pitch.*": 0.4,
    r".*hip_yaw.*": 0.2,
    r".*knee.*": 0.4,
    r".*ankle_pitch.*": 0.3,
    r".*ankle_roll.*": 0.2,
  }
  # Active from the start: with lin_vel_x widened to +/-0.8 and running_threshold
  # at 0.7, a forward command alone crosses into this band (~48% of normal envs,
  # and nearly all rotation envs). Loosened ~1.5x over
  # std_walking, with the sagittal (pitch) joints given the most slack since they
  # do the swinging; roll/yaw stay comparatively tight to discourage the legs
  # splaying sideways at speed.
  std_running = {
    r".*torso_yaw.*": 0.3,
    r".*shoulder_pitch.*": 0.6,
    r".*shoulder_roll.*": 0.3,
    r".*shoulder_yaw.*": 0.3,
    r".*elbow.*": 0.3,
    r".*wrist_pitch.*": 0.3,
    r"neck_yaw": 0.15,
    r"head_pitch": 0.15,
    r".*hip_roll.*": 0.3,
    r".*hip_pitch.*": 0.6,
    r".*hip_yaw.*": 0.3,
    r".*knee.*": 0.6,
    r".*ankle_pitch.*": 0.45,
    r".*ankle_roll.*": 0.3,
  }
  # The template's 1.5 is sized for a ~1.3 m humanoid and is unreachable here on
  # linear velocity alone: 1.5 m/s is Froude ~0.9 for this robot's 0.28 m legs,
  # i.e. a sprint. At 1.5 the "running" band was effectively a rotation-only
  # regime. 0.7 sits inside the widened lin_vel_x range (Froude ~0.18), so a pure
  # forward command crosses into it well before the curriculum stage fires.
  running_threshold = 0.7

  cfg.rewards["pose"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=(DOFS_FILTER,)
  )
  cfg.rewards["pose"].params["std_standing"] = std_standing
  cfg.rewards["pose"].params["std_walking"] = std_walking
  cfg.rewards["pose"].params["std_running"] = std_running
  cfg.rewards["pose"].params["walking_threshold"] = walking_threshold
  cfg.rewards["pose"].params["running_threshold"] = running_threshold
  cfg.rewards["pose"].weight = 1.0

  # At commands below walking_threshold, independently reward holding the
  # HOME_KEYFRAME pose. The target comes from robot.data.default_joint_pos, so
  # this stays synchronized with HOME_KEYFRAME without duplicating joint angles
  # here. At/above the threshold the reward is exactly zero and cannot resist
  # the walking motion. Tune its contribution with this term's weight.
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
  cfg.rewards["upright"].params["pitch"] = np.deg2rad(10.0)
  cfg.rewards["upright"].params["std"] = np.sqrt(0.1)
  # Same in_air gate as track_linear/angular_velocity_gated -- see upright's
  # docstring/gate comment: standing statically upright is the trivial way to
  # max this reward, so it shouldn't pay out while a nontrivial velocity is
  # commanded and no foot is off the ground.
  cfg.rewards["upright"].params["sensor_name"] = FEET_GROUND_SENSOR_CFG.name
  cfg.rewards["upright"].params["height_sensor_name"] = FOOT_HEIGHT_SCAN_CFG.name
  cfg.rewards["upright"].params["command_threshold"] = walking_threshold
  cfg.rewards["upright"].params["max_air_time"] = max_swing_time
  # Bumped 1.0 -> 2.0: the fell_over-dominated run showed upright contributing
  # almost nothing (Episode_Reward/upright ~0.03) next to termination (~-0.99)
  # and self_collisions (~-0.62) -- doubling it to push staying-upright harder
  # before the policy ever gets punished for falling.
  cfg.rewards["upright"].weight = 0.5

  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["angular_momentum"].weight = -0.02

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

  cfg.rewards["air_time"].func = feet_air_time_once_reward
  cfg.rewards["air_time"].params = {
    "sensor_name": FEET_GROUND_SENSOR_CFG.name,
    "threshold_min": 0.2,
    "threshold_max": max_swing_time,
    "command_name": "twist",
    "command_threshold": walking_threshold,
  }
  # Unlike the stock term, this pays only once at landing when the completed
  # swing lasted 0.2--0.6 s. Holding a foot up pays nothing until it lands.
  cfg.rewards["air_time"].weight = 1.0

  cfg.rewards["no_stepping"] = RewardTermCfg(
    func=no_stepping_penalty,
    weight=-0.3,
    params={
      "sensor_name": FEET_GROUND_SENSOR_CFG.name,
      "command_name": "twist",
      "command_threshold": walking_threshold,
    },
  )
  cfg.rewards["not_stepping"] = RewardTermCfg(
    func=not_stepping_penalty,
    # At -10 this charged -0.2 on every double-support transition, making a
    # normal landing far more expensive than holding one foot up forever.
    weight=-1.0,
    params={
      "sensor_name": FEET_GROUND_SENSOR_CFG.name,
      "command_name": "twist",
      "command_threshold": walking_threshold,
    },
  )
  cfg.rewards["overlong_swing"] = RewardTermCfg(
    func=overlong_swing_penalty,
    # Effective cost is -0.04 per 50 Hz step after 0.6 s. At the same instant
    # the tracking/upright gate closes, so landing is better than waiting.
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
  cfg.rewards["foot_slip"].weight = -3.0
  cfg.rewards["action_rate_l2"].weight = -0.4

  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=self_collision_cost_excluding_linkage,
    weight=-1.0,
    params={
      "sensor_name": SELF_COLLISION_SENSOR_CFG.name,
      "linkage_sensor_names": tuple(c.name for c in LINKAGE_CONTACT_SENSOR_CFGS),
    },
  )

  # Soft, continuous companion to the fell_over termination (root_height <
  # 0.2m ends the episode outright). 0.35m gives real headroom below the
  # ~0.476-0.478m home/standing height (bent-knee default included) while
  # still sitting well above the hard 0.2m cutoff -- a crouch/settle deep
  # enough to start eating into that gap gets a continuously growing penalty
  # instead of no signal at all until the episode-ending threshold hits.
  cfg.rewards["base_height"] = RewardTermCfg(
    func=base_height_penalty,
    weight=-2.0,
    params={"minimum_height": 0.4},
  )

  # A hard cap on how far the upper body may swing off zero, covering torso_yaw,
  # both arms and neck/head. These joints all sit at 0.0 in HOME_KEYFRAME, and
  # their physical ranges are far too wide for dof_pos_limits (90% of physical
  # range) to ever engage: torso_yaw and shoulder_yaw are +/-180deg, elbow_pitch
  # is -143..120deg. Without this term nothing stops the policy from throwing an
  # arm behind its back or cranking the head sideways to game a reward.
  #
  # Each limit equals that joint's std_running above, so this is a backstop and
  # not a second shaping term: `pose` does the shaping inside the band, and this
  # only fires once a joint leaves the range `pose` would ever ask for. It also
  # stays clear of arm_swing -- hip_pitch at std_running 0.6 times gain 0.8 asks
  # for shoulder_pitch ~0.48, inside the 0.6 cap.
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
      "max_excursion": {
        r".*torso_yaw.*": 0.3,
        r".*shoulder_pitch.*": 0.6,
        r".*shoulder_roll.*": 0.6,
        r".*shoulder_yaw.*": 0.3,
        r".*elbow.*": 0.3,
        r".*wrist_pitch.*": 0.3,
        r"neck_yaw": 0.15,
        r"head_pitch": 0.15,
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

  # Human-like gait: right arm reaches forward while the left leg swings, and
  # vice versa. gain=0.6 keeps the arm swing smaller than the leg swing, which is
  # closer to how people walk than a 1:1 mirror. Gated on the command so a
  # standing robot is not asked to swing.
  cfg.rewards["arm_swing"] = RewardTermCfg(
    func=contralateral_arm_swing,
    weight=0.05,
    params={
      "std": 0.25,
      "gain": 0.8,
      "command_name": "twist",
      "command_threshold": walking_threshold,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  # Fires once, on the step an episode ends by failure. `is_terminated` reads
  # termination_manager.terminated, which excludes the time_out term, so surviving
  # the full 20 s is not penalised -- only fell_over / out_of_terrain_bounds are.
  # scale_rewards_by_dt is on, so the weight is multiplied by step_dt (0.02):
  # the effective one-off penalty is -20, against a per-step budget where the
  # largest positive term (track_linear_velocity, weight 2.0) contributes 0.04.
  cfg.rewards["termination"] = RewardTermCfg(
    func=envs_mdp.is_terminated,
    weight=-200.0,
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
  cfg.rewards["same_foot_repeat"] = RewardTermCfg(
    func=same_foot_repeat_penalty,
    weight=-1.0,
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
    weight=3.0,
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
    },
  )

  cfg.rewards["feet_distance"] = RewardTermCfg(
    func=feet_distance_penalty,
    weight=-50.0,
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
  cfg.rewards["foot_flatness"] = RewardTermCfg(
    func=foot_flatness_penalty,
    weight=-0.3,
    params={
      "sensor_name": FEET_GROUND_SENSOR_CFG.name,
      "asset_cfg": SceneEntityCfg("robot", body_names=("left_foot_1", "right_foot_1")),
    },
  )

  # One-shot swing-height milestone: each foot pays once when it first clears
  # 3 cm during commanded single support. Holding it there pays nothing more;
  # landing rearms that foot for the next swing.
  cfg.rewards["feet_crossing"] = RewardTermCfg(
    func=feet_crossing_reward,
    weight=1.0,
    params={
      "sensor_name": FEET_GROUND_SENSOR_CFG.name,
      "height_sensor_name": FOOT_HEIGHT_SCAN_CFG.name,
      "min_swing_height": 0.03,
      "command_name": "twist",
      "command_threshold": walking_threshold,
    },
  )

  # One-off reward when a foot lands with an x/y displacement matching the
  # body-frame planar velocity command. This covers forward, backward, lateral,
  # and diagonal steps; see forward_step_reward for the frame conversion.
  cfg.rewards["forward_step"] = RewardTermCfg(
    func=forward_step_reward,
    weight=0.5,
    params={
      "sensor_name": FEET_GROUND_SENSOR_CFG.name,
      "height_sensor_name": FOOT_HEIGHT_SCAN_CFG.name,
      "asset_cfg": SceneEntityCfg("robot", site_names=tuple(foot_site_names)),
      "command_name": "twist",
      "command_threshold": walking_threshold,
      # Gaussian width on the 2-D step-vector error. 0.05 m is ~16% of
      # the 0.31 m stride the retargeted human walk shows for this robot at
      # 0.32 m/s, so a step has to land within a few centimetres of what the
      # command implies to score most of the term.
      "std": 0.05,
      "min_step_distance": 0.02,
      "min_swing_height": 0.02,
    },
  )

  cfg.rewards["velocity_shortfall"] = RewardTermCfg(
    func=velocity_shortfall_penalty,
    weight=-0.3,
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
    weight=-0.1,
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
    weight=-0.08,
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
  command = cfg.commands["twist"]
  command.build = lambda env, _cmd=command: UniformVelocityCommandWithRotation(
    _cmd, env
  )
  command.viz.z_offset = 0.5
  # Match the 20 s episode horizon: reset samples one fresh command and the
  # timeout resets the environment before an in-episode resample can occur.
  # Both bounds are required; ``(20.0)`` would be a float, not a one-item tuple.
  command.resampling_time_range = (3.0, 20.0)
  command.rel_standing_envs = 0.1
  command.rel_heading_envs = 0.0
  command.rel_rotation_envs = 0.1
  # Single-purpose commands, exclusive with rotation (see
  # UniformVelocityCommandWithRotation). Rotation and the three single-axis
  # modes take 40%; diagonal planar translation (vx/vy nonzero, wz=0) takes
  # another 20%. Standing takes 10%, and the remaining 30% keep the mixed
  # vx/vy/wz sample.
  command.rel_forward_only_envs = 0.1
  command.rel_backward_only_envs = 0.1
  command.rel_lateral_only_envs = 0.1
  command.rel_planar_only_envs = 0.2
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
  # Push starts disabled. The staged curriculum below ramps it to x/y +/-0.20
  # and roll/pitch +/-0.22; DR_WIDE still controls the other DR ranges.
  cfg.events["push_robot"].params["velocity_range"] = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
  }
  cfg.events["push_robot"].interval_range_s = (8.0, 10.0)
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
    r".*left_foot_collision.*",
    r".*right_foot_collision.*",
  )
  # Raised the low end from mjlab's default 0.3 -- that's wet-tile-slippery and
  # was a plausible contributor to fell_over alongside the other issues found
  # this session (double curriculum, action-std runaway). 0.6 still gives real
  # low-friction variety without handing every env a near-ice floor.
  cfg.events["foot_friction"].params["ranges"] = (0.8, 1.2)
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

  # Armature: all 33 scalar joints, actuated + passive. For the 25 BAM-driven joints this
  # scales the reflected inertia BAM computed at edit_spec time (a static model
  # constant BAM never revisits after startup, so the randomization sticks). The 4
  # *_roll_cap joints (armature=5e-8 in the XML) get scaled too; the 4 *_roll_actual
  # joints have no armature attribute (defaults to 0 in MuJoCo), so scaling is a
  # no-op there -- 0 x anything is still 0.
  cfg.events["dof_armature_randomization"] = EventTermCfg(
    mode="startup",
    func=dr.joint_armature,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "operation": "scale",
      "ranges": (0.6, 1.4) if DR_WIDE else (0.8, 1.2),
    },
  )

  # Friction: the 8 passive *_roll_cap/*_roll_actual joints only (DOFS_FILTER's
  # complement). These keep their XML-authored frictionloss untouched by BAM (see
  # preserve_joint_friction in kid_rl_constants), so unlike the 25 actuated joints
  # -- whose frictionloss BAM zeroes at build time and rewrites every step,
  # making this randomization a no-op there -- it actually sticks here.
  cfg.events["dof_friction_randomization"] = EventTermCfg(
    mode="startup",
    func=dr.joint_friction,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*_(cap|actual)$",)),
      "operation": "scale",
      "ranges": (0.5, 1.5) if DR_WIDE else (0.7, 1.3),
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
          "x": 0.20,
          "y": 0.20,
          "roll": 0.22,
          "pitch": 0.22,
          "yaw": 0.22,
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
@dataclass
class KidRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """mjlab's PPO config plus RSL-RL's symmetry extension.

  RSL-RL's PPO already accepts a ``symmetry_cfg``, but mjlab's config dataclass
  does not declare the field, and the trainer hands RSL-RL ``asdict(cfg.agent)``
  -- so without this subclass the key never reaches the algorithm and symmetry
  is silently off. See mjlab_kid_rl.tasks.symmetry for the mirror itself.
  """

  symmetry_cfg: dict | None = None


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
    # 0.01 let the entropy bonus swamp the policy gradient. Measured at
    # iteration 17k: Loss/entropy 42.15 x 0.01 = 0.42 against a surrogate loss of
    # -0.0175 -- a 24x imbalance, i.e. the update was optimising entropy, not the
    # task. That drove a self-reinforcing trap:
    #
    #   entropy bonus pushes the action std up hard each update
    #     -> the policy distribution moves a lot between updates -> large KL
    #     -> the adaptive scheduler below divides the LR by 1.5 per minibatch
    #        (20 per iteration, so it reaches its own max(1e-5, ...) floor within
    #        a single iteration)
    #     -> at 1e-5 nothing is learned, the surrogate loss stays tiny
    #     -> entropy keeps dominating
    #
    # The symptoms were an action std that rose from init_std 1.0 to ~1.39 and
    # saturated there, Loss/learning_rate pinned at the 1e-5 floor, and mean
    # reward / episode length flat for 2250 iterations.
    #
    # Keep some exploration pressure without recreating the observed
    # std=1.5 runaway. This matches the value described by the analysis above;
    # the previous literal 0.005 did not match that comment.
    entropy_coef=0.005,
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
    symmetry_cfg={
      "use_data_augmentation": False,
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
  clip_actions=5.0,
  max_iterations=15_000,
)
