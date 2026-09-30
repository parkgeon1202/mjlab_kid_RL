"""BeyondMimic-style motion-tracking task for kid_RL.

This configuration is intentionally separate from ``kid_RL_env_cfg.py``.
The velocity-command task remains unchanged and can still be selected through
its existing task id.
"""

from copy import deepcopy

from bam.mjlab import bam_init
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.config.g1.rl_cfg import (
  unitree_g1_tracking_ppo_runner_cfg,
)
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from mjlab_kid_rl.robot.kid_rl_dance.kid_rl_dance_constants import (
  get_kid_rl_dance_robot_cfg,
)
from mjlab_kid_rl.tasks.mdp import self_collision_cost_excluding_linkage


# The real robot measures the 25 actuated joints. The 8 *_roll_cap/*_roll_actual
# joints are passive followers of the hip/ankle four-bar linkages.
ACTUATED_JOINTS_FILTER = r"^(?!.*_(?:cap|actual)$).*$"

TRACKING_BODY_NAMES = (
  "base_link",
  "torso_1",
  "left_hip_slider_1",
  "left_lower_leg_1",
  "left_foot_1",
  "right_hip_slider_1",
  "right_lower_leg_1",
  "right_foot_1",
  "left_arm1",
  "left_arm3",
  "left_hand_tip",
  "right_arm1",
  "right_arm3",
  "right_hand_tip",
)

END_EFFECTOR_BODY_NAMES = (
  "left_foot_1",
  "right_foot_1",
  "left_hand_tip",
  "right_hand_tip",
)


# base_link is the free-joint root, so its subtree is the complete robot.
#
# "found" only, with no force history: the stock tracking reward
# (mdp.self_collision_cost) uses force_history to reject weak contacts, but this
# task swaps it for self_collision_cost_excluding_linkage (see below), which
# reads "found" alone. Keeping force+history_length=4 would allocate a
# [B, N, 4, 3] buffer per env that nothing reads. It would not help anyway:
# measured peak contact force on the cap<->groove pairs is ~1.3e7 N, so no
# force_threshold can separate the linkage from a real self-collision -- which
# is exactly why the reward is swapped rather than retuned.
SELF_COLLISION_SENSOR_CFG = ContactSensorCfg(
  name="self_collision",
  primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
  secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
  fields=("found",),
  reduce="none",
  num_slots=1,
)

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


# Keep the linkage-friendly solver settings independent of the velocity task.
# These can be tuned here without changing previous locomotion experiments.
KID_RL_TRACKING_SIM_CFG = SimulationCfg(
  mujoco=MujocoCfg(
    timestep=0.005,
    integrator="implicitfast",
    iterations=50,
    ls_iterations=30,
    ccd_iterations=100,
    disableflags=("nativeccd",),
  ),
  # nconmax/njmax are deliberately NOT set: left at mujoco_warp's own heuristic.
  # On flat terrain with this robot that heuristic allocates 192 per-world
  # contacts / njmax 768, which clears both the put_data floor (mujoco_warp
  # builds its initial MjData from qpos0 -- the robot buried in the ground --
  # and refuses anything below that configuration's counts) and the measured
  # random-action peaks. Unlike the velocity task, this one runs on a plane with
  # no heightfield sub-terrain, so there is no reason to hand-size these here.
  # Only revisit if the console prints an nconmax/njmax overflow.
  contact_sensor_maxmatch=256,
)

KID_RL_TRACKING_VIEWER_CFG = ViewerConfig(
  origin_type=ViewerConfig.OriginType.ASSET_BODY,
  entity_name="robot",
  body_name="torso_1",
  distance=3.0,
  elevation=-15.0,
  azimuth=90.0,
)


def make_kid_rl_tracking_env_cfg(
  play: bool = False,
  motion_file: str = "",
) -> ManagerBasedRlEnvCfg:
  """Create an independent flat-terrain motion-tracking environment.

  Args:
    play: Apply deterministic evaluation overrides.
    motion_file: Local mjlab motion NPZ. It may be left empty while registering
      the task and supplied later with
      ``--env.commands.motion.motion-file /path/to/motion.npz``.
  """
  cfg = make_tracking_env_cfg()

  cfg.scene = SceneCfg(
    terrain=TerrainEntityCfg(terrain_type="plane"),
    num_envs=1,
    extent=2.0,
    entities={"robot": get_kid_rl_dance_robot_cfg()},
    sensors=(SELF_COLLISION_SENSOR_CFG, *LINKAGE_CONTACT_SENSOR_CFGS),
  )
  cfg.sim = deepcopy(KID_RL_TRACKING_SIM_CFG)
  cfg.viewer = deepcopy(KID_RL_TRACKING_VIEWER_CFG)
  cfg.decimation = 4  # 0.005 s simulation step -> 50 Hz policy/motion rate.

  # The action controls exactly the 25 MJCF actuators. Passive linkage joints
  # remain constrained by MuJoCo and are not policy outputs.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = (r".*",)
  joint_pos_action.scale = 1.0

  # The actor observes only hardware-measurable current joint states. The motion
  # command and critic still use the full reference/model state required by
  # mjlab's MotionCommand.
  cfg.observations["actor"].terms["joint_pos"] = ObservationTermCfg(
    func=mdp.joint_pos_rel,
    params={
      "biased": True,
      "asset_cfg": SceneEntityCfg(
        "robot", joint_names=(ACTUATED_JOINTS_FILTER,)
      ),
    },
    noise=Unoise(n_min=-0.01, n_max=0.01),
  )
  cfg.observations["actor"].terms["joint_vel"] = ObservationTermCfg(
    func=mdp.joint_vel_rel,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot", joint_names=(ACTUATED_JOINTS_FILTER,)
      )
    },
    noise=Unoise(n_min=-0.5, n_max=0.5),
  )

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.motion_file = motion_file
  # The anchor is the frame the policy measures its tracking error in
  # (motion_anchor_pos_b/ori_b), the frame the anchor terminations and
  # rewards use, and the frame relative body poses are expressed in.
  # G1 anchors at torso_link, but kid_RL uses base_link: it is the
  # free-joint root, it is body_names[0] (which _resample_command already
  # writes the root state from), and the IMU site lives on it, so the
  # anchor error and the measured base_lin_vel/base_ang_vel share a frame.
  motion_cmd.anchor_body_name = "base_link"
  motion_cmd.body_names = TRACKING_BODY_NAMES

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = (
    r".*left_foot_collision.*",
    r".*right_foot_collision.*",
  )
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_1",)
  # Scale the default human-size/G1 COM perturbation down for this 0.476 m robot.
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.005, 0.005),
    1: (-0.005, 0.005),
    2: (-0.005, 0.005),
  }
  cfg.events["bam_init"] = EventTermCfg(func=bam_init, mode="startup")

  # The four cap/groove contacts are part of the linkage mechanism, not harmful
  # limb collisions. Reuse the velocity task's tested subtraction reward.
  cfg.rewards["self_collisions"].func = self_collision_cost_excluding_linkage
  cfg.rewards["self_collisions"].params = {
    "sensor_name": SELF_COLLISION_SENSOR_CFG.name,
    "linkage_sensor_names": tuple(
      sensor.name for sensor in LINKAGE_CONTACT_SENSOR_CFGS
    ),
  }

  cfg.terminations["ee_body_pos"].params[
    "body_names"
  ] = END_EFFECTOR_BODY_NAMES

  # A direct height-ratio scaling of G1's 0.25 m threshold gives 0.09 m, but a
  # random initial policy then terminates after roughly 11 steps and receives
  # too little experience to learn the motion. 0.15 m still rejects a clearly
  # collapsed 0.476 m robot while leaving enough room for early exploration.
  # anchor_ori remains 0.8 because orientation error has no length scale.
  cfg.terminations["anchor_pos"].params["threshold"] = 0.15
  cfg.terminations["ee_body_pos"].params["threshold"] = 0.15

  # The stock term passes joint_names=(".*",), which sweeps in the 8 passive
  # linkage joints. *_roll_actual has no range in the MJCF at all (unbounded)
  # and *_roll_cap's range is wider than the crank that actually drives it, so
  # penalising them is at best noise and at worst punishes the policy for joints
  # it cannot command. Restrict to the 25 actuated joints, as the velocity task
  # does.
  cfg.rewards["joint_limit"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=(ACTUATED_JOINTS_FILTER,)
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.joint_position_range = (0.0, 0.0)
    motion_cmd.sampling_mode = "start"

  return cfg


# PPO structure follows mjlab's stock BeyondMimic/G1 tracking baseline. Only
# run-identifying fields are changed; tuning comes after one motion can replay.
KID_RL_TRACKING_RL_CFG: RslRlOnPolicyRunnerCfg = (
  unitree_g1_tracking_ppo_runner_cfg()
)
KID_RL_TRACKING_RL_CFG.experiment_name = "mjlab_kid_rl_tracking"
KID_RL_TRACKING_RL_CFG.wandb_project = "mjlab_kid_rl_tracking"
