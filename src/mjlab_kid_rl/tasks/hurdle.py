"""Hurdle crossing: hurdle entity, spawn, command, camera, observation, reward, curriculum.

The floor is the same plane as the flat IRC task. The hurdle is one mocap box (the
competition bar: 2 cm thick, 1 m wide) in every world. At each reset an env becomes
a hurdle env with probability HURDLE_ENV_FRACTION: its bar is put about 1 m ahead of
the robot, square to the robot's heading, and the robot is commanded straight
forward over it. The other envs park their bar HURDLE_PARK_OFFSET away and run the
ordinary velocity task. Each hurdle env draws its bar height from its own level,
which hurdle_levels moves.

Touching the bar with any robot geom costs a penalty on every step it lasts. The
bar never moves: a mocap body takes no force.

The actor sees the bar only through hurdle_vision, a model of the robot's RealSense
pipeline (irc_vision realsense/src/refiner: /hurdle/pose) and of the task planner's
camera tilt rule (irc_task_planner fsm/hurdling.py). The critic sees the true bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import json
import math
import os
from pathlib import Path

import mujoco
import torch

from mjlab.entity import Entity, EntityCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.envs.mdp.events import reset_root_state_uniform
from mjlab.tasks.velocity.mdp.rewards import body_angular_velocity_penalty
from mjlab.utils.lab_api.math import (
  quat_apply_inverse,
  quat_from_euler_xyz,
  quat_mul,
  sample_uniform,
  wrap_to_pi,
)

from mjlab_kid_rl.tasks.mdp import (
  UniformVelocityCommandWithRotation,
  UniformVelocityCommandWithRotationCfg,
  upright,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

HURDLE_COMMAND_NAME = "twist"
HURDLE_ENTITY_NAME = "hurdle"
HURDLE_BAR_BODY = "bar"
"""The entity's mocap body; in the compiled model it is "hurdle/bar"."""

##
# Bar geometry and placement.
##

HURDLE_THICKNESS = 0.02
HURDLE_WIDTH = 1.0
HURDLE_BUILD_HEIGHT = 0.04
"""Height [m] the bar compiles with. Resets replace it per env before any use."""
HURDLE_LEVEL_RANGES = (
  (0.005, 0.015),
  (0.015, 0.025),
  (0.025, 0.035),
  (0.035, 0.045),
  (0.045, 0.053),
  (0.053, 0.063),
)
"""Bar height range [m] of each curriculum level, lowest first. A reset draws the
height uniformly within its level's range. The real bar is 5.3 cm."""
HURDLE_LEVELS = len(HURDLE_LEVEL_RANGES)
HURDLE_GEOM_GROUP = 1
"""foot_height_scan only casts against group 0, so foot heights stay above the floor."""
HURDLE_COLLISION_BITS = 2
"""Robot bit only. The sole plates are bit 2 only, so a bit-1 bar would pass through
the sole between the studs without a contact. The upper-body meshes are bit 1 only
(kid_rl_irc_constants _UPPER_BODY_MESH_PATTERN), so they never touch the bar."""

HURDLE_ENV_FRACTION = 0.7
"""Chance that an env is a hurdle env, drawn again at every reset."""
HURDLE_DISTANCE = (0.9, 1.1)
"""Spawn base to bar centre line along the spawn heading [m]."""
HURDLE_LATERAL = (-0.1, 0.1)
"""Sideways offset of the bar's middle from the robot's path [m]."""
HURDLE_YAW_NOISE = 0.15
"""Bar yaw off square to the spawn heading, uniform in +/- this [rad]."""
HURDLE_SPAWN_JITTER = 0.05
"""Spawn x and y offset from the env origin of a hurdle env [m]."""
HURDLE_PARK_OFFSET = (0.0, 50.0)
"""Where a flat env's bar waits, from its env origin [m]. Far past a 20 s walk."""

HURDLE_FOOT_RADIUS = math.hypot(0.0675, 0.0475)
"""Half diagonal of the foot sole box [m], kid_RL_irc.xml *_foot_collision_1."""
HURDLE_PASS_MARGIN = HURDLE_THICKNESS / 2 + HURDLE_FOOT_RADIUS
"""How far past the bar's centre plane every foot site must be to have crossed [m]."""
HURDLE_REACH_BAND = 0.3
"""Base this close to the bar's centre plane [m] counts as having reached the bar."""
HURDLE_CONFIRM_TIME = 5.0
"""Time [s] the episode must keep running after the pass before hurdle_complete ends
it, so a pass that ends in a fall still ends as a failure."""
HURDLE_NEAR_BAND = (-0.5, 0.3)
"""Base offset from the bar's centre plane [m], behind and past it, where the step
shaping terms (hurdle_foot_lift_reward, hurdle_stride_progress_reward) pay."""
HURDLE_TRUTH_CLIP = (-3.0, 3.0)
"""Limits of hurdle_truth's position [m]; keeps it in range once the bar is behind."""
HURDLE_HEIGHT_LOG_DIR = Path("logs/rsl_rl/mjlab_kid_rl_irc_hurdle")
"""Where env 0's height at every reset goes, one file per process."""
HURDLE_STATS_DECAY = 0.995
"""Weight kept per new attempt in hurdle_levels' running statistics."""
HURDLE_OUTCOMES = ("clean", "touched", "fell_before", "fell_after", "idle", "blocked")
"""What a hurdle episode ended as. clean and touched crossed and did not fail,
without and with touching the bar. fell_before failed before crossing, fell_after
after. idle ran out of time without reaching the bar, blocked reached it and never
crossed."""

##
# Camera model. Frames and numbers from irc_vision (irc_man.xacro, refiner_node,
# tilt_config.yaml) and irc_task_planner (params.yaml, fsm/hurdling.py).
##

CAMERA_TILT_PIVOT = (0.0386, 0.0205, 0.1152)
"""Tilt joint origin in base_link [m] (xacro joint "tilt", parent torso_1, which sits
at the base_link origin). The tilt axis is base +y; positive tilt looks down."""
CAMERA_LENS_OFFSET = (0.060892, -0.026, 0.029383)
"""Lens (camera_optical_frame origin) in the tilt link [m], xacro joint "lasek"."""
CAMERA_TILT_RUNNING_DEG = 40.0
"""pan_tilt mode 0 (init), which RUNNING and the hurdle approach use."""
CAMERA_TILT_HURDLE_DEG = 65.0
"""pan_tilt mode 4 (hurdle)."""
CAMERA_TILT_SWITCH_DIST = 0.75
"""hurdle_tilt_fix_dist: the planner switches to mode 4 and keeps it once
/vision hurdle_dist is at most this [m]."""
CAMERA_TILT_RATE_DEG = 150.0
"""Tilt slew rate [deg/s]. An assumption: the Dynamixel's profile is not known."""
CAMERA_SIZE = (640, 480)
"""camera_profile 640x480x30, color stream (depth is aligned to it)."""
CAMERA_FOCAL = (615.0, 615.0)
CAMERA_CENTER = (320.0, 240.0)
"""fx, fy, cx, cy [px]. A typical D435 color 640x480 calibration (about 55 x 43 deg),
not this camera's camera_info: replace with the robot's when it is read."""
CAMERA_MIN_DEPTH = 0.1
"""Nearest depth [m] the D435 returns at this resolution; nearer pixels read 0."""
CAMERA_MAX_RANGE = 3.0
"""Farthest optical depth [m] at which the bar counts as detected. An assumption."""
CAMERA_SAMPLES = 21
"""Points along the bar's top centre line tested against the image."""
CAMERA_MIN_VISIBLE = 0.25
"""Bar length in the image [m] needed for a pose; stands in for refiner's 30-inlier
minimum over the top 40% band of the bbox."""
CAMERA_DROPOUT = 0.2
"""Chance per camera frame that a visible bar is missed anyway."""
CAMERA_PERIOD_S = 1.0 / 30.0
"""Camera frame period [s]. Frames arrive on this clock (random phase per episode),
not by chance. Like kid_rl: a frame that arrived since the last policy step is used
once at this step (the newest if several), and a step with none sees (0, 0, 0, 0).
At 30 Hz and 50 Hz steps that is 3 steps in every 5."""
CAMERA_LAG_STEPS = (1, 3)
"""Pipeline latency [policy steps], drawn per episode."""
CAMERA_NOISE_XY = (0.01, 0.02)
"""Pose noise std in x and y [m]: base + per metre of distance."""
CAMERA_NOISE_Z = 0.02
"""Pose noise std in z [m] per frame. Real deploy logs (bar 5.3 cm) show about 1.9 cm
frame-to-frame jitter."""
CAMERA_NEAR_CUTOFF = 0.23
CAMERA_NEAR_CUTOFF_STD = 0.025
CAMERA_NEAR_CUTOFF_CLAMP = (0.13, 0.33)
"""Bar x [m, base frame] below which the bar is lost, drawn per episode from a
Gaussian (mean, std) clamped to the range. Real deploy logs: the last detection
before the bar is lost has median x 0.23 m (min 0.13 m), before the trailing foot
crosses it."""
CAMERA_OUTLIER_PROB = 0.1
"""Chance per detected frame that x and y are replaced by an unrelated value, as when
RANSAC locks on another line (floor edge, a leg) instead of the bar."""
CAMERA_OUTLIER_X = (0.1, 1.5)
CAMERA_OUTLIER_Y = (-0.5, 0.5)
"""Uniform ranges [m] an outlier's x and y are drawn from, base frame."""
VISION_FRAMES = 11
VISION_STRIDE = 5
"""hurdle_vision's stack: 11 frames 5 steps apart, the newest and 1.0 s back."""
VISION_FRAME_DIM = 4
"""(visible, x, y, z) per frame."""

OBS_HISTORY_LENGTH = 5
"""Frames of proprioception the hurdle task's actor and critic see (80 ms between
newest and oldest at 50 Hz). Terms outside STACKED_TERMS are the newest frame only."""
STACKED_TERMS = (
  "base_ang_vel",
  "projected_gravity",
  "joint_pos",
  "joint_vel",
  "actions",
)


def _hurdle_spec() -> mujoco.MjSpec:
  """One mocap body holding the bar, centred on its body frame."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name=HURDLE_BAR_BODY, mocap=True)
  body.add_geom(
    name=HURDLE_BAR_BODY,
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(HURDLE_THICKNESS / 2, HURDLE_WIDTH / 2, HURDLE_BUILD_HEIGHT / 2),
    group=HURDLE_GEOM_GROUP,
    contype=HURDLE_COLLISION_BITS,
    conaffinity=HURDLE_COLLISION_BITS,
    margin=0.0,
    rgba=(0.8, 0.3, 0.1, 1.0),
  )
  return spec


def hurdle_entity_cfg() -> EntityCfg:
  return EntityCfg(spec_fn=_hurdle_spec)


##
# Small geometry helpers, all batched.
##


def _yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
  """[N, 4] wxyz quaternion of a rotation by ``yaw`` about z."""
  half = yaw / 2
  zeros = torch.zeros_like(yaw)
  return torch.stack((torch.cos(half), zeros, zeros, torch.sin(half)), dim=-1)


def bar_frame(
  points_w: torch.Tensor, center_w: torch.Tensor, yaw: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """Each point's (along-normal, along-bar) offset from the bar's centre [m].

  points_w [B, ..., 2 or 3], center_w [B, 3], yaw [B]. The normal is the bar's yaw
  direction, so a robot spawned behind the bar starts at a negative normal offset.
  """
  shape = (-1,) + (1,) * (points_w.dim() - 2)
  dx = points_w[..., 0] - center_w[:, 0].view(shape)
  dy = points_w[..., 1] - center_w[:, 1].view(shape)
  cos, sin = torch.cos(yaw).view(shape), torch.sin(yaw).view(shape)
  return dx * cos + dy * sin, -dx * sin + dy * cos


def camera_axes(tilt: torch.Tensor) -> tuple[torch.Tensor, ...]:
  """Lens position and optical axes (forward, right, down) in base_link, [B, 3] each.

  The tilt link turns about base +y, so +tilt moves +x towards -z. The optical
  frame is the usual one: z forward, x right, y down (xacro camera_optical_joint).
  """
  cos, sin = torch.cos(tilt), torch.sin(tilt)
  zeros = torch.zeros_like(tilt)
  ones = torch.ones_like(tilt)
  pivot = torch.tensor(CAMERA_TILT_PIVOT, device=tilt.device)
  lx, ly, lz = CAMERA_LENS_OFFSET
  lens = pivot + torch.stack((cos * lx + sin * lz, ly * ones, -sin * lx + cos * lz), -1)
  forward = torch.stack((cos, zeros, -sin), dim=-1)
  right = torch.stack((zeros, -ones, zeros), dim=-1)
  down = torch.stack((-sin, zeros, -cos), dim=-1)
  return lens, forward, right, down


def project(points_b: torch.Tensor, tilt: torch.Tensor) -> tuple[torch.Tensor, ...]:
  """Pixel (u, v) and optical depth of base-frame points [B, K, 3] at camera tilt [B].

  Returns u, v, depth, each [B, K].
  """
  lens, forward, right, down = camera_axes(tilt)
  rel = points_b - lens.unsqueeze(1)
  depth = (rel * forward.unsqueeze(1)).sum(-1)
  x = (rel * right.unsqueeze(1)).sum(-1)
  y = (rel * down.unsqueeze(1)).sum(-1)
  safe = depth.clamp_min(1e-6)
  u = CAMERA_FOCAL[0] * x / safe + CAMERA_CENTER[0]
  v = CAMERA_FOCAL[1] * y / safe + CAMERA_CENTER[1]
  return u, v, depth


def in_image(u: torch.Tensor, v: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
  """[B, K] True where a point lands in the image within the detection depths."""
  width, height = CAMERA_SIZE
  return (
    (depth > CAMERA_MIN_DEPTH)
    & (depth < CAMERA_MAX_RANGE)
    & (u >= 0)
    & (u < width)
    & (v >= 0)
    & (v < height)
  )


def bar_top_points(
  center_w: torch.Tensor, yaw: torch.Tensor, height: torch.Tensor
) -> torch.Tensor:
  """[B, CAMERA_SAMPLES, 3] World points along the bar's top centre line, end to end."""
  device = center_w.device
  along = torch.linspace(-HURDLE_WIDTH / 2, HURDLE_WIDTH / 2, CAMERA_SAMPLES, device=device)
  direction = torch.stack((-torch.sin(yaw), torch.cos(yaw)), dim=-1)  # [B, 2]
  xy = center_w[:, None, :2] + along[None, :, None] * direction[:, None, :]
  z = (center_w[:, 2] - height / 2 + height).unsqueeze(-1).expand(-1, CAMERA_SAMPLES)
  return torch.cat((xy, z.unsqueeze(-1)), dim=-1)


def perpendicular_foot(points_b: torch.Tensor) -> torch.Tensor:
  """[B, 2] Foot of the perpendicular from the base origin to the line through the
  xy of the first and last of ``points_b`` [B, K, 3] (refiner's /hurdle/pose x, y)."""
  a = points_b[:, 0, :2]
  d = points_b[:, -1, :2] - a
  t = -(a * d).sum(-1) / (d * d).sum(-1).clamp_min(1e-9)
  return a + t.unsqueeze(-1) * d


##
# Per-env state, kept on the command term.
##


class HurdleVelocityCommand(UniformVelocityCommandWithRotation):
  """Velocity command that also holds every env's hurdle state.

  Hurdle envs get vx from ``hurdle_lin_vel_x`` and vy = 0 at every resample. For
  ``hurdle_straight_fraction`` of resamples wz holds a heading each step (pure
  straight: the spawn heading at reset, the robot's heading at that moment after);
  otherwise wz is a constant yaw command drawn from +/- ``hurdle_yaw_rate``
  and held until the next resample. The other envs keep the parent's sampling.
  """

  cfg: "HurdleVelocityCommandCfg"

  def __init__(self, cfg: "HurdleVelocityCommandCfg", env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    device = env.device
    num_envs = env.num_envs
    self.is_hurdle = torch.zeros(num_envs, dtype=torch.bool, device=device)
    """[B] This episode has a bar ahead. Set by reset_root_state_hurdle."""
    self.level = torch.zeros(num_envs, dtype=torch.long, device=device)
    """[B] Height level, moved by hurdle_levels."""
    self.height = torch.zeros(num_envs, device=device)
    """[B] Bar height [m]."""
    self.bar_pos_w = torch.zeros((num_envs, 3), device=device)
    """[B, 3] Bar centre, world frame; its z is half the height."""
    self.bar_yaw = torch.zeros(num_envs, device=device)
    """[B] Bar normal's yaw, world frame."""
    self.heading = torch.zeros(num_envs, device=device)
    """[B] Heading a hurdle env holds."""
    self.yaw_rate = torch.zeros(num_envs, device=device)
    """[B] Constant wz of a hurdle env that is not going straight [rad/s]."""
    self.is_straight = torch.ones(num_envs, dtype=torch.bool, device=device)
    """[B] The hurdle env holds the spawn heading instead of yaw_rate."""
    self.hurdle_geoms = env.scene[HURDLE_ENTITY_NAME].indexing.geom_ids.to(device)
    self._episode = 0

  def log_env0_height(self, env: ManagerBasedRlEnv) -> None:
    """Append env 0's new episode to this process's height log."""
    self._episode += 1
    row = {
      "step": env.common_step_counter,
      "episode": self._episode,
      "hurdle": bool(self.is_hurdle[0].item()),
      "level": self.level[0].item(),
      "height_m": self.height[0].item(),
    }
    HURDLE_HEIGHT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = HURDLE_HEIGHT_LOG_DIR / f"env0_heights_{os.getpid()}.jsonl"
    with path.open("a") as file:
      file.write(json.dumps(row) + "\n")

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    ids = env_ids[self.is_hurdle[env_ids]]
    if len(ids) == 0:
      return
    self.vel_command_b[ids, 0] = sample_uniform(
      *self.cfg.hurdle_lin_vel_x, (len(ids),), device=self.device
    )
    self.vel_command_b[ids, 1:] = 0.0
    self.yaw_rate[ids] = sample_uniform(
      -self.cfg.hurdle_yaw_rate, self.cfg.hurdle_yaw_rate, (len(ids),), device=self.device
    )
    self.is_straight[ids] = (
      torch.rand(len(ids), device=self.device) < self.cfg.hurdle_straight_fraction
    )
    # A straight command mid-episode holds the heading the robot has now, like the
    # walking envs' forward-only command. At reset (command_counter still 0) the
    # spawn heading set by reset_root_state_hurdle stays: square to the bar.
    keep = ids[self.is_straight[ids] & (self.command_counter[ids] > 0)]
    self.heading[keep] = self.robot.data.heading_w[keep]
    for mask in (
      self.is_standing_env,
      self.is_heading_env,
      self.is_world_env,
      self.is_forward_env,
      self.is_rotation_env,
      self.is_forward_only_env,
      self.is_backward_only_env,
      self.is_lateral_only_env,
      self.is_planar_only_env,
    ):
      mask[ids] = False

  def _update_command(self) -> None:
    super()._update_command()
    ids = self.is_hurdle.nonzero(as_tuple=False).flatten()
    if len(ids) == 0:
      return
    error = wrap_to_pi(self.heading[ids] - self.robot.data.heading_w[ids])
    hold = torch.clip(
      self.cfg.heading_control_stiffness * error,
      min=self.cfg.ranges.ang_vel_z[0],
      max=self.cfg.ranges.ang_vel_z[1],
    )
    self.vel_command_b[ids, 2] = torch.where(self.is_straight[ids], hold, self.yaw_rate[ids])


@dataclass(kw_only=True)
class HurdleVelocityCommandCfg(UniformVelocityCommandWithRotationCfg):
  """Configuration for HurdleVelocityCommand."""

  hurdle_lin_vel_x: tuple[float, float] = (0.0, 0.4)
  """Forward speed of hurdle envs [m/s]: the IRC task's positive vx range. Below
  about 0.05 m/s the bar (0.9-1.1 m away) is not reached inside a 20 s episode."""
  hurdle_yaw_rate: float = 0.1
  """Yaw command [rad/s] given together with vx when not going straight: uniform in
  +/- this per resample, held constant until the next one."""
  hurdle_straight_fraction: float = 0.5
  """Chance per resample that a hurdle env goes straight (holds the spawn heading)
  instead of turning at a constant yaw_rate."""

  def build(self, env: ManagerBasedRlEnv) -> HurdleVelocityCommand:
    return HurdleVelocityCommand(self, env)


def _command(env: ManagerBasedRlEnv) -> HurdleVelocityCommand:
  return env.command_manager.get_term(HURDLE_COMMAND_NAME)


##
# Reset: spawn, bar placement and height.
##


def _sample_hurdle_heights(levels: torch.Tensor) -> torch.Tensor:
  """[N] Height drawn uniformly from each level's HURDLE_LEVEL_RANGES entry."""
  ranges = torch.tensor(HURDLE_LEVEL_RANGES, device=levels.device)[levels]  # [N, 2]
  draw = torch.rand(levels.shape, device=levels.device)
  return ranges[:, 0] + (ranges[:, 1] - ranges[:, 0]) * draw


def _set_bar_heights(
  env: ManagerBasedRlEnv, ids: torch.Tensor, heights: torch.Tensor
) -> None:
  """Make the bar in the worlds ``ids`` that tall. The renderer draws env 0 from the
  C model, which gets a copy."""
  command = _command(env)
  command.height[ids] = heights
  model = env.sim.model
  geoms = command.hurdle_geoms
  worlds = ids.unsqueeze(-1)
  half = (heights / 2).unsqueeze(-1).expand(-1, len(geoms))
  model.geom_size[worlds, geoms, 2] = half
  model.geom_aabb[worlds, geoms, 1, 2] = half
  model.geom_rbound[worlds, geoms] = torch.norm(model.geom_size[worlds, geoms], dim=-1)

  first = (ids == 0).nonzero(as_tuple=False).flatten()
  if len(first) > 0:
    env.sim.mj_model.geom_size[geoms.cpu().numpy(), 2] = heights[first[0]].item() / 2


@requires_model_fields("geom_size", "geom_aabb", "geom_rbound")
def reset_root_state_hurdle(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  pose_range: dict[str, tuple[float, float]],
  velocity_range: dict[str, tuple[float, float]] | None = None,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
  """reset_root_state_uniform, then draw each env's hurdle and place its bar.

  A hurdle env spawns near its env origin at a random heading, holds that heading,
  and gets a bar about 1 m ahead. A flat env keeps reset_root_state_uniform's pose
  and parks its bar. hurdle_levels has already moved the levels.
  """
  reset_root_state_uniform(env, env_ids, pose_range, velocity_range, asset_cfg)
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  env_ids = env_ids.long()
  device = env.device
  command = _command(env)
  num = len(env_ids)

  is_hurdle = torch.rand(num, device=device) < HURDLE_ENV_FRACTION
  command.is_hurdle[env_ids] = is_hurdle
  heights = _sample_hurdle_heights(command.level[env_ids])
  _set_bar_heights(env, env_ids, heights)

  origins = env.scene.env_origins[env_ids]
  bar_pos = origins.clone()
  bar_pos[:, 0] += HURDLE_PARK_OFFSET[0]
  bar_pos[:, 1] += HURDLE_PARK_OFFSET[1]
  bar_yaw = torch.zeros(num, device=device)

  ids = env_ids[is_hurdle]
  if len(ids) > 0:
    asset: Entity = env.scene[asset_cfg.name]
    default_root_state = asset.data.default_root_state
    assert default_root_state is not None
    root_state = default_root_state[ids].clone()
    n = len(ids)
    position = root_state[:, :3] + env.scene.env_origins[ids]
    position[:, :2] += sample_uniform(
      -HURDLE_SPAWN_JITTER, HURDLE_SPAWN_JITTER, (n, 2), device
    )
    position[:, 2] += sample_uniform(*pose_range.get("z", (0.0, 0.0)), (n,), device)
    roll = sample_uniform(*pose_range.get("roll", (0.0, 0.0)), (n,), device)
    pitch = sample_uniform(*pose_range.get("pitch", (0.0, 0.0)), (n,), device)
    yaw = sample_uniform(-math.pi, math.pi, (n,), device)
    orientation = quat_mul(root_state[:, 3:7], quat_from_euler_xyz(roll, pitch, yaw))
    asset.write_root_link_pose_to_sim(torch.cat([position, orientation], dim=-1), env_ids=ids)
    command.heading[ids] = yaw

    distance = sample_uniform(*HURDLE_DISTANCE, (n,), device)
    lateral = sample_uniform(*HURDLE_LATERAL, (n,), device)
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    center = torch.zeros((n, 3), device=device)
    center[:, 0] = position[:, 0] + distance * cos - lateral * sin
    center[:, 1] = position[:, 1] + distance * sin + lateral * cos
    center[:, 2] = env.scene.env_origins[ids, 2]
    picked = is_hurdle.nonzero(as_tuple=False).flatten()
    bar_pos[picked] = center
    bar_yaw[picked] = wrap_to_pi(
      yaw + sample_uniform(-HURDLE_YAW_NOISE, HURDLE_YAW_NOISE, (n,), device)
    )

  bar_pos[:, 2] += heights / 2
  command.bar_pos_w[env_ids] = bar_pos
  command.bar_yaw[env_ids] = bar_yaw
  hurdle: Entity = env.scene[HURDLE_ENTITY_NAME]
  hurdle.write_mocap_pose_to_sim(
    torch.cat([bar_pos, _yaw_quat(bar_yaw)], dim=-1), env_ids=env_ids
  )
  if (env_ids == 0).any():
    command.log_env0_height(env)


##
# Observations.
##


def hurdle_truth(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
  """[B, 7] The bar as the critic sees it; zeros on flat envs.

  (centre x, y, z in base frame, sin and cos of the bar yaw relative to the robot's
  heading, bar height, 1).
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = _command(env)
  center_b = quat_apply_inverse(
    asset.data.root_link_quat_w, command.bar_pos_w - asset.data.root_link_pos_w
  ).clamp(*HURDLE_TRUTH_CLIP)
  relative = wrap_to_pi(command.bar_yaw - asset.data.heading_w)
  out = torch.cat(
    (
      center_b,
      torch.sin(relative).unsqueeze(-1),
      torch.cos(relative).unsqueeze(-1),
      command.height.unsqueeze(-1),
      torch.ones_like(command.height).unsqueeze(-1),
    ),
    dim=-1,
  )
  return out * command.is_hurdle.float().unsqueeze(-1)


class hurdle_vision:
  """[B, 44] What the robot's RealSense pipeline would report about the bar.

  Per policy step:
    1. Put the camera at its current tilt and test CAMERA_SAMPLES points along the
       bar's top centre line against the image (in view and within depth range).
    2. The bar is detected when CAMERA_MIN_VISIBLE of it is in view and its x is at
       least the episode's CAMERA_NEAR_CUTOFF, minus a CAMERA_DROPOUT miss. A detection reports refiner's /hurdle/pose: the foot of
       the perpendicular from base_link to the bar line (x, y) and the bar points'
       median height z, all base frame, plus noise. CAMERA_OUTLIER_PROB of the
       detections report an unrelated x, y instead. No detection reports zeros.
    3. The task planner's tilt rule: /vision hurdle_dist, the ground distance to
       the bbox centre's point (here the mean of the in-view points), at most
       CAMERA_TILT_SWITCH_DIST switches the camera from 40 to 65 deg for good.
       The tilt slews at CAMERA_TILT_RATE_DEG.
  The newest report reaches the policy CAMERA_LAG_STEPS late and only on a fresh
  camera frame (CAMERA_PERIOD_S clock); a step without one sees zeros (nothing held). The output is
  VISION_FRAMES of those, VISION_STRIDE steps apart, oldest first, each
  (visible, x, y, z).
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    del cfg
    device = env.device
    num_envs = env.num_envs
    self.depth = 1 + (VISION_FRAMES - 1) * VISION_STRIDE + CAMERA_LAG_STEPS[1]
    self.history = torch.zeros((num_envs, self.depth, VISION_FRAME_DIM), device=device)
    self.tilt = torch.full((num_envs,), math.radians(CAMERA_TILT_RUNNING_DEG), device=device)
    self.tilt_target = self.tilt.clone()
    self.tilt_fixed = torch.zeros(num_envs, dtype=torch.bool, device=device)
    self.lag = torch.full((num_envs,), CAMERA_LAG_STEPS[0], dtype=torch.long, device=device)
    self.offsets = torch.arange(VISION_FRAMES - 1, -1, -1, device=device) * VISION_STRIDE
    # Time [s] until the next camera frame arrives.
    self.to_next_frame = torch.rand(num_envs, device=device) * CAMERA_PERIOD_S
    self.near_cutoff = torch.full((num_envs,), CAMERA_NEAR_CUTOFF, device=device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.history[env_ids] = 0.0
    self.tilt[env_ids] = math.radians(CAMERA_TILT_RUNNING_DEG)
    self.tilt_target[env_ids] = math.radians(CAMERA_TILT_RUNNING_DEG)
    self.tilt_fixed[env_ids] = False
    self.to_next_frame[env_ids] = (
      torch.rand_like(self.to_next_frame[env_ids]) * CAMERA_PERIOD_S
    )
    count = self.lag[env_ids].numel()
    low, high = CAMERA_LAG_STEPS
    self.lag[env_ids] = torch.randint(low, high + 1, (count,), device=self.lag.device)
    draw = CAMERA_NEAR_CUTOFF + CAMERA_NEAR_CUTOFF_STD * torch.randn(
      count, device=self.near_cutoff.device
    )
    self.near_cutoff[env_ids] = draw.clamp(*CAMERA_NEAR_CUTOFF_CLAMP)

  def measure(
    self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """This step's (report [B, 4], detected [B], hurdle_dist [B]) at the current tilt."""
    asset: Entity = env.scene[asset_cfg.name]
    command = _command(env)
    points_w = bar_top_points(command.bar_pos_w, command.bar_yaw, command.height)
    base_pos = asset.data.root_link_pos_w.unsqueeze(1)
    base_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, CAMERA_SAMPLES, -1)
    points_b = quat_apply_inverse(base_quat, points_w - base_pos)
    u, v, depth = project(points_b, self.tilt)
    seen = in_image(u, v, depth) & command.is_hurdle.unsqueeze(-1)

    spacing = HURDLE_WIDTH / (CAMERA_SAMPLES - 1)
    count = seen.sum(-1)
    dropped = torch.rand(count.shape, device=count.device) < CAMERA_DROPOUT
    detected = (count.float() * spacing >= CAMERA_MIN_VISIBLE) & ~dropped

    weight = seen.float().unsqueeze(-1)
    seen_mean = (points_b * weight).sum(1) / weight.sum(1).clamp_min(1.0)
    hurdle_dist = torch.norm(seen_mean[:, :2], dim=-1)

    foot = perpendicular_foot(points_b)
    detected = detected & (foot[:, 0] >= self.near_cutoff)
    distance = torch.norm(foot, dim=-1, keepdim=True)
    sigma = CAMERA_NOISE_XY[0] + CAMERA_NOISE_XY[1] * distance
    foot = foot + sigma * torch.randn_like(foot)
    outlier = torch.rand_like(distance) < CAMERA_OUTLIER_PROB
    low = torch.tensor((CAMERA_OUTLIER_X[0], CAMERA_OUTLIER_Y[0]), device=foot.device)
    high = torch.tensor((CAMERA_OUTLIER_X[1], CAMERA_OUTLIER_Y[1]), device=foot.device)
    foot = torch.where(outlier, low + (high - low) * torch.rand_like(foot), foot)
    z = seen_mean[:, 2:3] + CAMERA_NOISE_Z * torch.randn_like(distance)
    report = torch.cat((torch.ones_like(distance), foot, z), dim=-1)
    report = report * detected.float().unsqueeze(-1)
    return report, detected, hurdle_dist

  def __call__(
    self, env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
  ) -> torch.Tensor:
    report, detected, hurdle_dist = self.measure(env, asset_cfg)

    switch = detected & (hurdle_dist <= CAMERA_TILT_SWITCH_DIST) & ~self.tilt_fixed
    self.tilt_fixed |= switch
    self.tilt_target = torch.where(
      self.tilt_fixed,
      torch.full_like(self.tilt, math.radians(CAMERA_TILT_HURDLE_DEG)),
      torch.full_like(self.tilt, math.radians(CAMERA_TILT_RUNNING_DEG)),
    )
    step = math.radians(CAMERA_TILT_RATE_DEG) * env.step_dt
    self.tilt = self.tilt + (self.tilt_target - self.tilt).clamp(-step, step)

    # Frames that arrived during this step; several collapse into the newest one.
    self.to_next_frame = self.to_next_frame - env.step_dt
    fresh = self.to_next_frame <= 0.0
    arrived = torch.floor(-self.to_next_frame / CAMERA_PERIOD_S) + 1.0
    self.to_next_frame = torch.where(
      fresh, self.to_next_frame + arrived * CAMERA_PERIOD_S, self.to_next_frame
    )
    self.history = torch.roll(self.history, shifts=1, dims=1)
    self.history[:, 0] = report * fresh.float().unsqueeze(-1)

    index = (self.lag.unsqueeze(-1) + self.offsets.unsqueeze(0)).clamp_max(self.depth - 1)
    frames = torch.gather(
      self.history, 1, index.unsqueeze(-1).expand(-1, -1, VISION_FRAME_DIM)
    )
    return frames.reshape(env.num_envs, -1)


##
# Rewards.
##


def _hurdle_touched(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """[B] True while any robot collision geom touches the bar."""
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  return found.reshape(env.num_envs, -1).any(dim=-1)


def hurdle_contact_penalty(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """1 on every step any robot geom touches the bar."""
  return _hurdle_touched(env, sensor_name).float()


def hurdle_scaled_body_ang_vel(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, hurdle_scale: float
) -> torch.Tensor:
  """body_angular_velocity_penalty (roll/pitch rate squared), times ``hurdle_scale``
  in hurdle envs and 1 in walking envs."""
  penalty = body_angular_velocity_penalty(env, asset_cfg)
  scale = torch.where(_command(env).is_hurdle, hurdle_scale, 1.0)
  return penalty * scale


class hurdle_scaled_upright(upright):
  """mdp.upright, times ``walking_scale`` in walking envs and 1 in hurdle envs."""

  def __call__(self, env: ManagerBasedRlEnv, walking_scale: float, **kwargs) -> torch.Tensor:
    reward = super().__call__(env, **kwargs)
    return reward * torch.where(_command(env).is_hurdle, 1.0, walking_scale)


class hurdle_pass_reward:
  """Pay once when the robot has crossed the bar.

  Crossed: base and every foot site past the bar's centre plane by their margins,
  the base within the bar's width (not around its end), with a walking command.
  Also keeps each episode's record, which hurdle_levels reads before the reset.
  ``confirmed`` turns True HURDLE_CONFIRM_TIME after the pass; hurdle_complete
  then ends the episode.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    device = env.device
    num_envs = env.num_envs
    self.passed = torch.zeros(num_envs, dtype=torch.bool, device=device)
    self.touched = torch.zeros_like(self.passed)
    self.reached = torch.zeros_like(self.passed)
    self.episode_time = torch.zeros(num_envs, device=device)
    self.pass_time = torch.zeros_like(self.episode_time)
    self.contact_time = torch.zeros_like(self.episode_time)
    self.touch_time = torch.zeros_like(self.episode_time)
    self.speed_error = torch.zeros_like(self.episode_time)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    for state in (self.passed, self.touched, self.reached):
      state[env_ids] = False
    for state in (
      self.episode_time,
      self.pass_time,
      self.contact_time,
      self.touch_time,
      self.speed_error,
    ):
      state[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
    command_name: str = HURDLE_COMMAND_NAME,
    command_threshold: float = 0.05,
  ) -> torch.Tensor:
    command_term = _command(env)
    is_hurdle = command_term.is_hurdle
    touching = _hurdle_touched(env, sensor_name) & is_hurdle
    self.touch_time = torch.where(touching & ~self.touched, self.episode_time, self.touch_time)
    self.touched |= touching
    self.contact_time += touching.float() * env.step_dt

    asset: Entity = env.scene[asset_cfg.name]
    center, yaw = command_term.bar_pos_w, command_term.bar_yaw
    feet_u, _ = bar_frame(asset.data.site_pos_w[:, asset_cfg.site_ids, :2], center, yaw)
    base_u, base_v = bar_frame(asset.data.root_link_pos_w[:, None, :2], center, yaw)
    base_u, base_v = base_u.squeeze(1), base_v.squeeze(1)

    command = env.command_manager.get_command(command_name)
    assert command is not None
    active = torch.norm(command[:, :2], dim=-1) > command_threshold

    crossed = (
      (feet_u.min(dim=-1).values > HURDLE_PASS_MARGIN)
      & (base_u > HURDLE_THICKNESS / 2)
      & (base_v.abs() < HURDLE_WIDTH / 2)
    )
    newly = is_hurdle & ~self.passed & crossed & active
    self.pass_time = torch.where(newly, self.episode_time, self.pass_time)
    self.passed |= newly
    self.reached |= is_hurdle & (base_u > -HURDLE_REACH_BAND)

    velocity = asset.data.root_link_lin_vel_b[:, :2]
    self.speed_error += torch.nan_to_num(
      torch.norm(command[:, :2] - velocity, dim=-1) * env.step_dt
    )
    self.episode_time += env.step_dt
    return newly.float()

  @property
  def confirmed(self) -> torch.Tensor:
    """[B] HURDLE_CONFIRM_TIME of steps have run after the pass step.

    pass_time is the episode time at the start of the pass step, so the gap reads
    one step more than the time since the pass: strictly more than the confirm
    time is the pass step plus HURDLE_CONFIRM_TIME / step_dt steps."""
    return self.passed & (self.episode_time - self.pass_time > HURDLE_CONFIRM_TIME + 1e-4)


def hurdle_complete(env: ManagerBasedRlEnv, reward_name: str = "hurdle_pass") -> torch.Tensor:
  """[B] End a hurdle episode once its pass is confirmed. Register with
  time_out=True: a success, so it is not a failure for the termination penalty or
  hurdle_levels, and PPO bootstraps the value."""
  record: hurdle_pass_reward = env.reward_manager.get_term_cfg(reward_name).func
  return record.confirmed


def _near_bar(env: ManagerBasedRlEnv, asset: Entity) -> torch.Tensor:
  """[B] True for a hurdle env whose base is inside HURDLE_NEAR_BAND of the bar."""
  command_term = _command(env)
  base_u, _ = bar_frame(
    asset.data.root_link_pos_w[:, None, :2], command_term.bar_pos_w, command_term.bar_yaw
  )
  base_u = base_u.squeeze(1)
  return (
    command_term.is_hurdle & (base_u > HURDLE_NEAR_BAND[0]) & (base_u < HURDLE_NEAR_BAND[1])
  )


def _feet_in_air(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """[B, 2] True for a foot (left, right) off the ground."""
  found = env.scene[sensor_name].data.found
  assert found is not None
  if found.dim() == 3:
    found = found.any(dim=-1)
  return ~found.bool()


def _walking(env: ManagerBasedRlEnv, command_name: str, threshold: float) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return torch.norm(command[:, :2], dim=-1) > threshold


class hurdle_foot_lift_reward:
  """swing_progress_reward's lift part, moved up to the bar's heights.

  swing_progress pays a swing foot's rise from about 0 to 2 cm everywhere; this
  pays the next band, ``min_height`` to ``target_height`` (2 to 6 cm), near the
  bar. Each swing, the foot's peak height since liftoff is mapped onto [0, 1]
  over that band and every step pays only the rise of that value: at most 1 per
  swing, earned as the foot goes up. Holding the foot up pays nothing more; the
  peak clears when the foot lands. Height is foot_height_scan's, above the floor
  (the bar is outside its geom group). Each airborne foot earns on its own, so a
  two-footed hop pays for both.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.peak = torch.zeros((env.num_envs, 2), device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.peak[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    asset_cfg: SceneEntityCfg,
    min_height: float = 0.02,
    target_height: float = 0.06,
    command_name: str = HURDLE_COMMAND_NAME,
    command_threshold: float = 0.05,
  ) -> torch.Tensor:
    if target_height <= min_height:
      raise ValueError("target_height must be greater than min_height")
    in_air = _feet_in_air(env, sensor_name)
    heights = env.scene[height_sensor_name].data.heights  # [B, 2]
    previous = self.peak
    peak = torch.where(
      in_air, torch.maximum(previous, heights.clamp_min(0.0)), torch.zeros_like(previous)
    )
    span = target_height - min_height
    rise = ((peak - min_height) / span).clamp(0.0, 1.0) - (
      (previous - min_height) / span
    ).clamp(0.0, 1.0)
    asset: Entity = env.scene[asset_cfg.name]
    gate = _near_bar(env, asset) & _walking(env, command_name, command_threshold)
    self.peak = peak
    return (rise * in_air.float()).sum(dim=-1) * gate.float()


class hurdle_stride_progress_reward:
  """Reward a swing foot's forward reach across the bar, a little at a time.

  The swing_progress_reward pattern applied to distance instead of height. Each
  swing, the foot's furthest travel along the bar's normal since liftoff is mapped
  onto [0, 1] between ``min_stride`` and ``target_stride``, and every step pays
  only the rise of that value. A swing therefore pays at most 1 in total, earned
  as the foot moves forward; holding it, or moving it back and forth, pays
  nothing more. Travel only counts while the foot is at least ``min_height``
  above the floor (no sliding). Each airborne foot earns on its own, so a
  two-footed hop pays for both. Paid in a hurdle env whose base is near the bar,
  with a walking command.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    shape = (env.num_envs, 2)
    self.liftoff_u = torch.zeros(shape, device=env.device)
    self.peak = torch.zeros(shape, device=env.device)
    self.was_air = torch.zeros(shape, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.liftoff_u[env_ids] = 0.0
    self.peak[env_ids] = 0.0
    self.was_air[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    asset_cfg: SceneEntityCfg,
    min_stride: float = 0.05,
    target_stride: float = 0.25,
    min_height: float = 0.01,
    command_name: str = HURDLE_COMMAND_NAME,
    command_threshold: float = 0.05,
  ) -> torch.Tensor:
    if target_stride <= min_stride:
      raise ValueError("target_stride must be greater than min_stride")
    in_air = _feet_in_air(env, sensor_name)
    asset: Entity = env.scene[asset_cfg.name]
    command_term = _command(env)
    feet_u, _ = bar_frame(
      asset.data.site_pos_w[:, asset_cfg.site_ids, :2],
      command_term.bar_pos_w,
      command_term.bar_yaw,
    )  # [B, 2]

    started = in_air & ~self.was_air
    self.liftoff_u = torch.where(started, feet_u, self.liftoff_u)
    heights = env.scene[height_sensor_name].data.heights  # [B, 2]
    travel = torch.where(heights >= min_height, feet_u - self.liftoff_u, self.peak)
    previous = torch.where(started, torch.zeros_like(self.peak), self.peak)
    peak = torch.where(in_air, torch.maximum(previous, travel), torch.zeros_like(previous))

    span = target_stride - min_stride
    rise = ((peak - min_stride) / span).clamp(0.0, 1.0) - (
      (previous - min_stride) / span
    ).clamp(0.0, 1.0)
    gate = _near_bar(env, asset) & _walking(env, command_name, command_threshold)
    self.peak = peak
    self.was_air = in_air
    return (rise * in_air.float()).sum(dim=-1) * gate.float()

##
# Curriculum.
##


class hurdle_levels:
  """Move each finishing hurdle env through the height levels and report how it went.

  Up: crossed and did not fail. Down: never crossed. Fell after crossing: stays.
  Outcomes are counted on the level the episode ran on, over about the last
  1 / (1 - HURDLE_STATS_DECAY) attempts. Runs before hurdle_pass_reward resets.
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.counts = torch.zeros((HURDLE_LEVELS, len(HURDLE_OUTCOMES)), device=env.device)
    self.attempts = torch.zeros(HURDLE_LEVELS, device=env.device)
    self.averages: dict[str, torch.Tensor] = {}

  def _average(self, name: str, values: torch.Tensor) -> None:
    """Update a running average of per-episode ``values``, if there are any."""
    if len(values) == 0:
      return
    mean = values.float().mean()
    previous = self.averages.get(name, mean)
    weight = 1.0 - HURDLE_STATS_DECAY ** len(values)
    self.averages[name] = previous + weight * (mean - previous)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice,
    reward_name: str = "hurdle_pass",
  ) -> dict[str, torch.Tensor]:
    if isinstance(env_ids, slice):
      env_ids = torch.arange(env.num_envs, device=env.device)
    command = _command(env)
    record = env.reward_manager.get_term_cfg(reward_name).func
    failed = env.termination_manager.terminated[env_ids]

    # The very first reset ends an episode that never ran.
    ran = record.episode_time[env_ids] > 0.0
    hurdle = command.is_hurdle[env_ids] & ran
    flat = ~command.is_hurdle[env_ids] & ran
    passed = record.passed[env_ids]
    touched = record.touched[env_ids]
    reached = record.reached[env_ids]

    outcomes = torch.stack(
      (
        passed & ~failed & ~touched,
        passed & ~failed & touched,
        failed & ~passed,
        failed & passed,
        ~failed & ~passed & ~reached,
        ~failed & ~passed & reached,
      ),
      dim=-1,
    )
    levels = command.level[env_ids]
    for level in range(HURDLE_LEVELS):
      on_level = hurdle & (levels == level)
      attempts = on_level.sum()
      self.attempts[level] += attempts
      self.counts[level] *= HURDLE_STATS_DECAY**attempts
      self.counts[level] += outcomes[on_level].sum(dim=0).float()

    time = record.episode_time[env_ids]
    speed_error = record.speed_error[env_ids] / time.clamp_min(env.step_dt)
    self._average("hurdle_episode_s", time[hurdle])
    self._average("flat_episode_s", time[flat])
    self._average("hurdle_speed_error", speed_error[hurdle])
    self._average("flat_speed_error", speed_error[flat])
    self._average("passed", passed[hurdle])
    self._average("pass_time_s", record.pass_time[env_ids][hurdle & passed])
    self._average("contact_s", record.contact_time[env_ids][hurdle])
    self._average("touched_episodes", touched[hurdle])
    self._average("first_touch_s", record.touch_time[env_ids][hurdle & touched])
    height = command.height[env_ids]
    for level in range(HURDLE_LEVELS):
      self._average(f"level{level}_height", height[hurdle & (levels == level)])

    up = hurdle & passed & ~failed
    down = hurdle & ~passed
    moved = levels + up.long() - down.long()
    command.level[env_ids] = moved.clamp(0, HURDLE_LEVELS - 1)

    log = {
      "mean": command.level.float().mean(),
      "max": command.level.max().float(),
      **self.averages,
    }
    totals = self.counts.sum(dim=0)
    for index, name in enumerate(HURDLE_OUTCOMES):
      log[name] = totals[index] / totals.sum().clamp_min(1.0)
    for level in range(HURDLE_LEVELS):
      counts = self.counts[level]
      log[f"level{level}_attempts"] = self.attempts[level]
      log[f"level{level}_success"] = counts[:2].sum() / counts.sum().clamp_min(1.0)
    return log


def vision_obs_cfg() -> ObservationTermCfg:
  return ObservationTermCfg(func=hurdle_vision)


def truth_obs_cfg() -> ObservationTermCfg:
  return ObservationTermCfg(func=hurdle_truth)
