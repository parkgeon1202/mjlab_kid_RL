# Copyright 2026 Marc Duclusaud

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from mjlab.entity import Entity
from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from mjlab.utils.lab_api.string import resolve_matching_names_values
from mjlab.tasks.velocity.mdp.velocity_command import (
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)


############################ COMMANDS #############################

class UniformVelocityCommandWithRotation(UniformVelocityCommand):
    """Extends UniformVelocityCommand with single-purpose command environments.

    On every resample each environment is put in at most one of these exclusive
    modes. The fractions are of all environments and are sampled only from the
    non-standing pool, so the parent's standing override does not dilute them.
    Whatever is left over keeps the parent's mixed (vx, vy, wz) sample.

      * rotation:      vx = vy = 0, |wz| >= ``rotation_min_ang_vel``, drawn from
                       ``rotation_env_ang_vel_range`` (or ``ranges.ang_vel_z``)
      * forward-only:  vx > 0, vy = wz = 0
      * backward-only: vx < 0, vy = wz = 0
      * lateral-only:  vy != 0 on a random side, vx = wz = 0
      * planar-only:   vx != 0, vy != 0, wz = 0

    Linear speeds come from the *current* ``cfg.ranges``, so the staged
    curriculum's widening applies to these modes too. The magnitude floor
    ``directional_min_lin_vel`` keeps them from degenerating into a near-zero
    command, but it is clipped to that side's range limit: a stage whose range is
    still zero on that side gets a zero command rather than one the curriculum
    has not unlocked yet.

    The parent's own ``rel_forward_envs`` forces vx >= 0.3 m/s whatever
    ``ranges`` says; leave it at 0 and use ``rel_forward_only_envs`` instead.
    """

    cfg: "UniformVelocityCommandWithRotationCfg"

    def __init__(self, cfg: "UniformVelocityCommandWithRotationCfg", env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.is_rotation_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_forward_only_env = torch.zeros_like(self.is_rotation_env)
        self.is_backward_only_env = torch.zeros_like(self.is_rotation_env)
        self.is_lateral_only_env = torch.zeros_like(self.is_rotation_env)
        self.is_planar_only_env = torch.zeros_like(self.is_rotation_env)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)

        # getattr: the velocity task hands this class the template's plain
        # UniformVelocityCommandCfg with these fields attached ad hoc, so they are
        # not guaranteed to exist on the instance.
        fractions = (
            getattr(self.cfg, "rel_rotation_envs", 0.0),
            getattr(self.cfg, "rel_forward_only_envs", 0.0),
            getattr(self.cfg, "rel_backward_only_envs", 0.0),
            getattr(self.cfg, "rel_lateral_only_envs", 0.0),
            getattr(self.cfg, "rel_planar_only_envs", 0.0),
        )
        standing_fraction = getattr(self.cfg, "rel_standing_envs", 0.0)
        moving_fraction = 1.0 - standing_fraction
        if sum(fractions) > moving_fraction + 1e-6:
            raise ValueError(
                "rel_rotation_envs + rel_forward_only_envs + rel_backward_only_envs"
                " + rel_lateral_only_envs + rel_planar_only_envs must be <= "
                f"1 - rel_standing_envs ({moving_fraction:.3f}), got "
                f"{sum(fractions):.3f}"
            )

        # Draw only from non-standing environments. Dividing the configured
        # all-environment fractions by the moving share keeps, for example,
        # rel_planar_only_envs=0.2 at 20% overall rather than letting the
        # parent's standing mask reduce it to 18% when standing is 10%.
        u = torch.rand(len(env_ids), device=self.device)
        non_standing = ~self.is_standing_env[env_ids]
        masks = []
        lower = 0.0
        for fraction in fractions:
            upper = lower + (fraction / moving_fraction if moving_fraction > 0.0 else 0.0)
            masks.append(non_standing & (u >= lower) & (u < upper))
            lower = upper
        rot, fwd, bwd, lat, planar = masks

        self.is_rotation_env[env_ids] = rot
        self.is_forward_only_env[env_ids] = fwd
        self.is_backward_only_env[env_ids] = bwd
        self.is_lateral_only_env[env_ids] = lat
        self.is_planar_only_env[env_ids] = planar

        x_lo, x_hi = self.cfg.ranges.lin_vel_x
        y_lo, y_hi = self.cfg.ranges.lin_vel_y

        fwd_ids = env_ids[fwd]
        if len(fwd_ids) > 0:
            self.vel_command_b[fwd_ids] = 0.0
            self.vel_command_b[fwd_ids, 0] = self._sample_speed(
                torch.full((len(fwd_ids),), max(x_hi, 0.0), device=self.device)
            )

        bwd_ids = env_ids[bwd]
        if len(bwd_ids) > 0:
            self.vel_command_b[bwd_ids] = 0.0
            self.vel_command_b[bwd_ids, 0] = -self._sample_speed(
                torch.full((len(bwd_ids),), max(-x_lo, 0.0), device=self.device)
            )

        lat_ids = env_ids[lat]
        if len(lat_ids) > 0:
            left = torch.rand(len(lat_ids), device=self.device) < 0.5
            # Each side is limited by its own bound, so an asymmetric lin_vel_y
            # range is respected per side rather than mirrored.
            limit = torch.where(
                left,
                torch.tensor(max(y_hi, 0.0), device=self.device),
                torch.tensor(max(-y_lo, 0.0), device=self.device),
            )
            speed = self._sample_speed(limit)
            self.vel_command_b[lat_ids] = 0.0
            self.vel_command_b[lat_ids, 1] = torch.where(left, speed, -speed)

        planar_ids = env_ids[planar]
        if len(planar_ids) > 0:
            # Sample both translation axes away from zero, then explicitly
            # remove yaw. This supplies diagonal-translation examples that the
            # parent's continuous mixed sampler almost never produces with
            # exactly zero angular velocity.
            self.vel_command_b[planar_ids] = 0.0
            self.vel_command_b[planar_ids, 0] = self._sample_signed_speed(
                len(planar_ids), x_lo, x_hi
            )
            self.vel_command_b[planar_ids, 1] = self._sample_signed_speed(
                len(planar_ids), y_lo, y_hi
            )

        rot_ids = env_ids[rot]
        if len(rot_ids) > 0:
            self.vel_command_b[rot_ids, 0] = 0.0
            self.vel_command_b[rot_ids, 1] = 0.0

            # Sample angular velocity from the rotation-specific range if provided,
            # otherwise reuse what the parent sampled from cfg.ranges.ang_vel_z.
            if self.cfg.rotation_env_ang_vel_range is not None:
                ang = torch.empty(len(rot_ids), device=self.device).uniform_(
                    *self.cfg.rotation_env_ang_vel_range
                )
            else:
                ang = self.vel_command_b[rot_ids, 2]

            # Ensure non-zero angular velocity.
            min_abs_ang = self.cfg.rotation_min_ang_vel
            too_small = ang.abs() < min_abs_ang
            if too_small.any():
                signs = torch.where(
                    torch.rand(too_small.sum(), device=self.device) > 0.5,
                    torch.ones(too_small.sum(), device=self.device),
                    -torch.ones(too_small.sum(), device=self.device),
                )
                ang[too_small] = signs * min_abs_ang
            self.vel_command_b[rot_ids, 2] = ang

        # The parent copied the mixed sample into the world-frame reference before
        # these overrides; keep it in step for any env that is also a world env.
        changed = env_ids[rot | fwd | bwd | lat | planar]
        self.vel_command_w[changed] = self.vel_command_b[changed]

    def _sample_speed(self, limit: torch.Tensor) -> torch.Tensor:
        """Uniform speed in [min(floor, limit), limit] per element; 0 where limit is 0."""
        floor = torch.clamp(
            torch.full_like(limit, getattr(self.cfg, "directional_min_lin_vel", 0.0)),
            max=limit,
        )
        return floor + torch.rand_like(limit) * (limit - floor)

    def _sample_signed_speed(self, count: int, lo: float, hi: float) -> torch.Tensor:
        """Sample a nonzero signed speed from the available sides of a range."""
        neg_limit = max(-lo, 0.0)
        pos_limit = max(hi, 0.0)
        if neg_limit == 0.0 and pos_limit == 0.0:
            return torch.zeros(count, device=self.device)
        if neg_limit == 0.0:
            positive = torch.ones(count, dtype=torch.bool, device=self.device)
        elif pos_limit == 0.0:
            positive = torch.zeros(count, dtype=torch.bool, device=self.device)
        else:
            positive = torch.rand(count, device=self.device) < 0.5
        limit = torch.where(
            positive,
            torch.full((count,), pos_limit, device=self.device),
            torch.full((count,), neg_limit, device=self.device),
        )
        speed = self._sample_speed(limit)
        return torch.where(positive, speed, -speed)

@dataclass(kw_only=True)
class UniformVelocityCommandWithRotationCfg(UniformVelocityCommandCfg):
    """Configuration for UniformVelocityCommandWithRotation."""

    rel_rotation_envs: float = 0.0
    """Fraction of environments that receive pure-rotation commands
    (zero linear velocity, non-zero angular velocity)."""

    rotation_min_ang_vel: float = 0.3
    """Minimum absolute angular velocity assigned to rotation-only environments."""

    rotation_env_ang_vel_range: tuple[float, float] | None = None
    """Angular velocity range for rotation-only environments.
    If None, uses cfg.ranges.ang_vel_z (same range as normal environments)."""

    rel_forward_only_envs: float = 0.0
    """Fraction of environments commanded straight forward only (vx > 0)."""

    rel_backward_only_envs: float = 0.0
    """Fraction of environments commanded straight backward only (vx < 0)."""

    rel_lateral_only_envs: float = 0.0
    """Fraction of environments commanded sideways only, on a random side."""

    rel_planar_only_envs: float = 0.0
    """Fraction commanded diagonally with nonzero vx/vy and zero yaw rate."""

    directional_min_lin_vel: float = 0.05
    """Speed floor for the forward/backward/lateral-only modes, clipped to that
    side's current range limit."""

    def build(self, env: ManagerBasedRlEnv) -> UniformVelocityCommandWithRotation:
        return UniformVelocityCommandWithRotation(self, env)


############################ REWARDS ##############################

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def default_joint_pose_exp(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    walking_threshold: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward the default keyframe pose only for commands below walking speed.

    The robot's ``default_joint_pos`` is initialized from ``HOME_KEYFRAME``.
    The returned reward is 1 at that pose and decays with the mean squared joint
    error as ``exp(-mean(error**2) / std**2)``.  Commands at or above
    ``walking_threshold`` receive zero so this term does not resist locomotion.
    """
    asset: Entity = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None

    command = env.command_manager.get_command(command_name)
    assert command is not None
    command_speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    below_walking_threshold = command_speed < walking_threshold

    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    target_joint_pos = default_joint_pos[:, asset_cfg.joint_ids]
    mean_squared_error = torch.mean(torch.square(joint_pos - target_joint_pos), dim=1)
    reward = torch.exp(-mean_squared_error / std**2)
    return reward * below_walking_threshold.float()


class upright:
    """Reward for keeping the base at a target pitch orientation.

    Penalizes deviation from a given pitch angle (in radians) rather than
    always rewarding a perfectly vertical posture.

    Args:
        std: Standard deviation of the Gaussian kernel (controls reward sharpness).
        pitch: Target pitch angle in radians. 0.0 = perfectly upright.
               Positive values mean leaning forward.
        asset_cfg: Scene entity configuration for the robot body to track.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        pass

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std: float,
        sensor_name: str,
        height_sensor_name: str,
        pitch: float = 0.0,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
        command_name: str = "twist",
        command_threshold: float = 0.05,
        tracking_ratio: float = 0.8,
        min_air_height: float = 0.02,
        max_air_time: float = 0.6,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]

        if asset_cfg.body_ids:
            body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
        else:
            body_quat_w = asset.data.root_link_quat_w

        gravity_w = asset.data.gravity_vec_w
        projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)

        # Normalize to unit vector so the error is scale-independent.
        gravity_norm = projected_gravity_b.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        projected_gravity_b_unit = projected_gravity_b / gravity_norm

        # At pitch angle θ, the normalised gravity unit vector in body frame has
        # x = sin(θ), y = 0 (assuming flat ground, no roll, no terrain slope).
        target_gx = math.sin(pitch)
        xy_error = (
            torch.square(projected_gravity_b_unit[:, 0] - target_gx)
            + torch.square(projected_gravity_b_unit[:, 1])
        )
        reward = torch.exp(-xy_error / std**2)

        # Same gate as track_linear_velocity_gated/track_angular_velocity_gated
        # (see _speed_tracking_gate docstring): holding a static, perfectly-
        # upright pose is the trivial way to maximize this reward, so without
        # a gate a policy commanded to walk could just stand there collecting
        # it instead of stepping.
        gate = _speed_tracking_gate(
            env, command_name, sensor_name, height_sensor_name, asset_cfg,
            command_threshold, tracking_ratio, min_air_height, max_air_time,
        )
        return reward * gate.float()

    def reset(self, env_ids: torch.Tensor) -> None:
        del env_ids  # Unused.


def feet_distance_penalty(
    env: ManagerBasedRlEnv,
    min_dist: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    max_dist: float | None = None,
) -> torch.Tensor:
    """Penalize only the distance outside [min_dist, max_dist].

    Returns ``clamp(min_dist - d, min=0) + clamp(d - max_dist, min=0)`` per
    env (use with a negative weight), where ``d`` is the horizontal (xy)
    distance between the two foot sites. Leaving max_dist at its default
    (None) skips the upper term entirely -- identical to the old min-only
    behavior.

    Args:
        min_dist: Minimum desired horizontal distance between feet, in meters.
        asset_cfg: Scene entity configuration whose ``site_names`` select exactly
            the two foot sites.
        max_dist: Maximum desired horizontal distance, in meters. None disables
            the upper bound.
    """
    asset: Entity = env.scene[asset_cfg.name]
    foot_pos_xy = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]  # [B, 2, 2]
    dist = torch.norm(foot_pos_xy[:, 0] - foot_pos_xy[:, 1], dim=-1)  # [B]
    penalty = torch.clamp(min_dist - dist, min=0.0)
    if max_dist is not None:
        penalty = penalty + torch.clamp(dist - max_dist, min=0.0)
    return penalty


def velocity_shortfall_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize base_lin_vel falling short of the commanded linear velocity.

    track_linear_velocity's exp(-error^2/std^2) kernel is symmetric: over-
    and under-shooting the commanded x/y velocity cost the same. This only
    penalizes coming up short in each axis's commanded direction -- matching
    or exceeding the command gives 0 here; lagging behind (or moving the
    wrong way) is penalized proportional to the deficit, in m/s. A sharper,
    more explicit "keep up with the command" signal than the Gaussian
    tracking reward alone, aimed at a policy that settles for near-zero
    velocity instead of actually walking.
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    actual_xy = asset.data.root_link_lin_vel_b[:, :2]
    cmd_xy = command[:, :2]
    # sign(cmd)=0 when cmd_i==0, so an axis with no command contributes
    # nothing here regardless of actual velocity on that axis.
    shortfall = torch.clamp(torch.sign(cmd_xy) * (cmd_xy - actual_xy), min=0.0)
    return torch.sum(shortfall, dim=-1)


def relative_linear_velocity_error_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float,
    max_relative_error: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize xy velocity error relative to the commanded speed.

    Normalizing by command speed makes completely missing a small nonzero
    command just as costly as completely missing a large one. Commands below
    ``command_threshold`` are excluded, and the relative error is capped before
    squaring so rare large deviations cannot dominate an update.
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    command_xy = command[:, :2]
    actual_xy = asset.data.root_link_lin_vel_b[:, :2]
    command_speed = torch.norm(command_xy, dim=-1)
    tracking_error = torch.norm(command_xy - actual_xy, dim=-1)

    relative_error = tracking_error / command_speed.clamp(min=command_threshold)
    relative_error = torch.clamp(relative_error, max=max_relative_error)
    active = command_speed >= command_threshold
    return torch.square(relative_error) * active.float()


def relative_angular_velocity_error_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float,
    max_relative_error: float = 2.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize yaw-rate error relative to the commanded yaw-rate magnitude.

    ``command_threshold`` is the minimum normalization scale rather than an
    activation gate. Consequently, unintended rotation is still penalized when
    the commanded yaw rate is zero. The relative error is capped before
    squaring so large transient errors cannot dominate an update.
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    command_yaw_rate = command[:, 2]
    actual_yaw_rate = asset.data.root_link_ang_vel_b[:, 2]
    tracking_error = torch.abs(command_yaw_rate - actual_yaw_rate)

    normalization_scale = torch.abs(command_yaw_rate).clamp(min=command_threshold)
    relative_error = tracking_error / normalization_scale
    relative_error = torch.clamp(relative_error, max=max_relative_error)
    return torch.square(relative_error)


def foot_base_heading_error_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize the feet pointing away from the base heading in the world xy plane.

    The local +x axes of the selected foot sites and the root link are rotated
    into the world frame and projected onto the horizontal plane.  For each
    foot the error is ``1 - cos(delta_yaw)``; the returned value is the mean
    across the selected feet. It is 0 when all headings are aligned, 1 when
    their mean cosine is zero, and 2 when all feet point backward.

    This term measures foot-to-base yaw alignment.  It does not measure global
    heading drift: if the base and both feet rotate together, their relative
    heading error remains zero.
    """
    asset: Entity = env.scene[asset_cfg.name]

    foot_quat_w = asset.data.site_quat_w[:, asset_cfg.site_ids, :]  # [B, F, 4]
    root_quat_w = asset.data.root_link_quat_w  # [B, 4]

    local_forward = torch.zeros(
        (*foot_quat_w.shape[:-1], 3),
        device=foot_quat_w.device,
        dtype=foot_quat_w.dtype,
    )
    local_forward[..., 0] = 1.0

    foot_forward_xy = quat_apply(foot_quat_w, local_forward)[..., :2]  # [B, F, 2]
    root_forward_xy = quat_apply(root_quat_w, local_forward[:, 0])[..., :2]  # [B, 2]

    foot_forward_xy = foot_forward_xy / foot_forward_xy.norm(
        dim=-1, keepdim=True
    ).clamp(min=1e-6)
    root_forward_xy = root_forward_xy / root_forward_xy.norm(
        dim=-1, keepdim=True
    ).clamp(min=1e-6)

    heading_cosine = torch.sum(
        foot_forward_xy * root_forward_xy.unsqueeze(1), dim=-1
    ).clamp(min=-1.0, max=1.0)
    cosine_error = 1.0 - heading_cosine
    return cosine_error.mean(dim=-1)


def _any_foot_in_air(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    min_height: float = 0.02,
) -> torch.Tensor:
    """[B] bool: at least one tracked foot is both out of contact AND has
    cleared min_height off the ground.

    Contact alone ("found=False") isn't enough -- a foot that barely broke
    contact (a millimeter of bounce, contact-sensor chatter) would otherwise
    count as "airborne" and let a policy satisfy this gate without a real
    swing. Requiring the height_sensor reading to clear min_height (mirrors
    min_swing_height elsewhere, e.g. forward_step_reward) makes it an actual
    ground-clearance check, not just a contact-boolean flicker.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    found = sensor.data.found  # (N, num_feet) or (N, num_feet, num_slots)
    if found.dim() == 3:
        found = found.any(dim=-1)
    not_in_contact = ~found.bool()

    height_sensor = env.scene[height_sensor_name]
    heights = height_sensor.data.heights  # (N, num_feet)
    cleared_height = heights >= min_height

    return (not_in_contact & cleared_height).any(dim=-1)


def _speed_tracking_gate(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    height_sensor_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float,
    tracking_ratio: float,
    min_air_height: float,
    max_air_time: float,
) -> torch.Tensor:
    """[B] bool: shared gate for track_linear_velocity_gated,
    track_angular_velocity_gated, and upright.

    cmd_speed/actual_speed are always the same combined metric --
    norm(xy linear) + abs(yaw rate) -- regardless of which single reward is
    asking, so "is there a real command right now" and "is the robot already
    keeping up" mean the same thing everywhere instead of each reward judging
    it off only its own axis (e.g. track_angular_velocity_gated checking yaw
    rate alone would treat a robot commanded to walk straight, standing dead
    still, as "no command" just because the yaw component of that command
    happens to be 0).

    True (gate open, reward pays out) when any of:
    - a foot is genuinely airborne (see _any_foot_in_air) and the swing has
      not exceeded max_air_time, or
    - the command is at/below command_threshold -- a genuinely-commanded
      stand, where staying planted is the *correct* response, or
    - actual combined speed already reaches tracking_ratio of commanded
      combined speed -- already delivering close enough that this shouldn't
      get zeroed during a normal gait's brief double-support instants just
      because no foot happens to be airborne on that exact frame.
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual_lin = asset.data.root_link_lin_vel_b
    actual_ang = asset.data.root_link_ang_vel_b
    actual_speed = torch.norm(actual_lin[:, :2], dim=-1) + torch.abs(actual_ang[:, 2])
    cmd_speed = torch.norm(command[:, :2], dim=-1) + torch.abs(command[:, 2])

    trivial_command = cmd_speed <= command_threshold
    already_tracking = actual_speed >= tracking_ratio * cmd_speed

    sensor: ContactSensor = env.scene[sensor_name]
    current_air_time = sensor.data.current_air_time
    assert current_air_time is not None, (
        f"Sensor '{sensor_name}' needs track_air_time=True for the tracking gate."
    )
    overdue_swing = (current_air_time > max_air_time).any(dim=-1)
    moving_gate = (
        _any_foot_in_air(env, sensor_name, height_sensor_name, min_air_height)
        | already_tracking
    ) & ~overdue_swing
    return trivial_command | moving_gate


def track_linear_velocity_gated(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    sensor_name: str,
    height_sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_threshold: float = 0.05,
    tracking_ratio: float = 0.8,
    min_air_height: float = 0.02,
    max_air_time: float = 0.6,
) -> torch.Tensor:
    """Same as mjlab's track_linear_velocity, but zeroed out unless at least
    one foot is currently airborne (mid-swing, cleared min_air_height off the
    ground) -- while a nontrivial velocity is commanded AND the robot isn't
    already keeping up with it.

    Without this, standing perfectly still with both feet planted already
    scores well under the Gaussian kernel for small/moderate commands (its
    std is wide relative to the command range), and velocity_shortfall_penalty
    (linear in the error) isn't steep enough at those magnitudes to cancel it
    out -- so a policy can bank a solidly positive net per-step reward for
    small-to-moderate commanded speeds while never taking a step. Requiring
    an airborne foot forces the policy to actually be mid-stride to collect
    this reward at all.

    Exempt in two cases -- see _speed_tracking_gate docstring: (1) commanded
    (combined xy + yaw) speed is at/below command_threshold -- a genuinely-
    commanded stand, where both feet planted is the *correct* response, not
    the thing being guarded against; (2) actual combined speed already
    reaches at least tracking_ratio of commanded combined speed -- the robot
    is already delivering close enough, so this reward shouldn't get zeroed
    during a normal gait's brief double-support instants (both feet
    momentarily grounded, body still carrying commanded speed from momentum)
    just because no foot happens to be airborne on that exact frame. Using a
    ratio < 1.0 rather than requiring an exact match avoids flickering the
    gate shut on ordinary residual tracking error.
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_lin_vel_b
    xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    z_error = torch.square(actual[:, 2])
    lin_vel_error = xy_error + z_error
    reward = torch.exp(-lin_vel_error / std**2)

    gate = _speed_tracking_gate(
        env, command_name, sensor_name, height_sensor_name, asset_cfg,
        command_threshold, tracking_ratio, min_air_height, max_air_time,
    )
    return reward * gate.float()


def track_angular_velocity_gated(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    sensor_name: str,
    height_sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    command_threshold: float = 0.05,
    tracking_ratio: float = 0.8,
    min_air_height: float = 0.02,
    max_air_time: float = 0.6,
) -> torch.Tensor:
    """Same as mjlab's track_angular_velocity, but zeroed out unless at least
    one foot is currently airborne (mid-swing, cleared min_air_height off the
    ground) -- while a nontrivial (combined xy + yaw) command is active AND
    the robot isn't already keeping up with it. See _speed_tracking_gate /
    track_linear_velocity_gated docstrings for the reasoning and the two
    exemptions (trivial command, and actual >= tracking_ratio * commanded) --
    identical exploit and identical fix, just applied to the angular-tracking
    reward (real in-place turning on a biped requires stepping, not just
    twisting the torso).
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_ang_vel_b
    z_error = torch.square(command[:, 2] - actual[:, 2])
    xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
    ang_vel_error = z_error + xy_error
    reward = torch.exp(-ang_vel_error / std**2)

    gate = _speed_tracking_gate(
        env, command_name, sensor_name, height_sensor_name, asset_cfg,
        command_threshold, tracking_ratio, min_air_height, max_air_time,
    )
    return reward * gate.float()


class same_foot_repeat_penalty:
    """Penalize a foot completing two swing->land cycles in a row.

    Walking alternates: left lands, then right, then left. A foot that lifts,
    swings and lands again while the other foot never left the ground is
    hopping or skipping, not walking. Nothing else in this task's reward set
    notices that -- air_time pays out per foot independently, so two hops on
    the same leg score the same as two alternating steps.

    The landing instant (sensor.compute_first_contact) is the event, not the
    liftoff: a foot that lifts and puts itself straight back down without the
    other foot moving has still completed the cycle we care about. One unit of
    penalty is emitted per repeat, so three consecutive landings on the same
    foot cost two units.

    Requires the contact sensor to have track_air_time=True.

    Args:
        sensor_name: Foot/ground contact sensor. Its slot order defines the
            foot indices; only their identity matters here, not which is which.
        command_name: Velocity command to gate on. A standing robot should not
            be stepping at all, which is no_stepping's job, not this term's.
        command_threshold: Below this commanded speed the term is switched off.
    """

    _NONE = -1  # no foot has landed yet this episode

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.last_foot = torch.full(
            (env.num_envs,), self._NONE, dtype=torch.long, device=env.device
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.last_foot[env_ids] = self._NONE

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        command_name: str = "twist",
        command_threshold: float = 0.05,
    ) -> torch.Tensor:
        sensor: ContactSensor = env.scene[sensor_name]
        just_landed = sensor.compute_first_contact(dt=env.step_dt)  # [B, F] bool

        landed_count = just_landed.sum(dim=-1)  # [B]
        single = landed_count == 1
        # argmax on the bool row gives the landing foot's index; it is only
        # meaningful where exactly one foot landed, which `single` masks for.
        landed_idx = torch.argmax(just_landed.long(), dim=-1)  # [B]

        repeat = single & (self.last_foot >= 0) & (landed_idx == self.last_foot)
        penalty = repeat.float()

        # Two feet landing on the same control step is a hop, not an
        # alternation, and leaves no meaningful "previous foot" for the next
        # landing to be compared against. Clearing to _NONE keeps the next
        # landing from being scored against a stale value; the double landing
        # itself is left to air_time/no_stepping rather than double-counted here.
        both = landed_count > 1
        self.last_foot = torch.where(single, landed_idx, self.last_foot)
        self.last_foot = torch.where(
            both, torch.full_like(self.last_foot, self._NONE), self.last_foot
        )

        command = env.command_manager.get_command(command_name)
        if command is not None:
            speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
            penalty = penalty * (speed > command_threshold).float()
        return penalty


class gait_symmetry_reward:
    """Reward matched left/right steps -- i.e. penalize a limping gait.

    same_foot_repeat_penalty already enforces the *order* of a gait (left,
    right, left), but alternating perfectly while one leg does most of the
    work still limps -- a long swing on one side and a short shuffle on the
    other alternates just fine. This term covers the missing half: the two
    legs' steps should also match each other.

    Two mismatches are measured, both only at the instant a foot lands
    (sensor.compute_first_contact):

    - swing duration: |last_air_time_left - last_air_time_right|, in seconds.
      The contact sensor latches each foot's completed swing into
      last_air_time at that foot's landing, so at any landing this compares
      the most recently completed swing of each leg.
    - step length: |step_len_left - step_len_right|, in metres, where a step
      is the foot's world xy travel from its own liftoff to its own landing.

    This is the "gait symmetry" idea from the bipedal RL literature expressed
    without a gait clock. Humanoid-Gym and similar work enforce symmetry by
    generating a reference stance mask from a fixed-period phase (left foot
    stance while sin(2*pi*phase) >= 0, right foot while < 0) and rewarding
    contacts that match it; that pins the robot to one cadence and needs the
    phase fed into the observation to stay Markovian. Comparing completed
    step events against *each other* instead of against a reference gets the
    left/right balance without either cost.

    NOTE on literal mirror symmetry. For this robot every left/right joint
    pair has opposite axis signs in robot_draft.xml (e.g. left_knee_pitch
    axis "0 1 0" vs right_knee_pitch "0 -1 0"), so the mirrored-pose
    condition is q_left + q_right == 0, not q_left - q_right == 0 -- which is
    why HOME_KEYFRAME's +/-30deg knees are in fact a physically symmetric
    stance. But penalizing |q_left + q_right| *statically* would be actively
    wrong for walking: it is satisfied by both legs swinging together (a hop)
    and violated by every normal walking pose, where the legs sit half a
    cycle apart. The correct joint-space statement is
    q_left(t) = -q_right(t + T/2), which again needs a clock or a history
    buffer. The step-event comparison here is the clock-free stand-in.

    Requires the contact sensor to have track_air_time=True.

    Args:
        sensor_name: Foot/ground contact sensor, slot order matching
            asset_cfg's two foot sites.
        asset_cfg: Scene entity config whose site_names select the two foot
            sites, in the same order as the sensor's slots.
        command_name: Velocity command to gate on.
        command_threshold: Below this commanded speed the term is off -- a
            standing robot has no gait to be symmetric about.
        duration_std: Swing-duration mismatch, in seconds, at which the
            duration half of the score falls to 1/e.
        length_std: Step-length mismatch, in metres, at which the length half
            of the score falls to 1/e.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset: Entity = env.scene[asset_cfg.name]
        num_feet = len(asset.find_sites(asset_cfg.site_names)[0])
        self.liftoff_pos_xy = torch.zeros(
            (env.num_envs, num_feet, 2), device=env.device
        )
        self.last_step_len = torch.zeros((env.num_envs, num_feet), device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.liftoff_pos_xy[env_ids] = 0.0
        self.last_step_len[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        asset_cfg: SceneEntityCfg,
        command_name: str = "twist",
        command_threshold: float = 0.05,
        duration_std: float = 0.1,
        length_std: float = 0.05,
    ) -> torch.Tensor:
        sensor: ContactSensor = env.scene[sensor_name]
        just_lifted = sensor.compute_first_air(dt=env.step_dt)  # [B, F]
        just_landed = sensor.compute_first_contact(dt=env.step_dt)  # [B, F]

        asset: Entity = env.scene[asset_cfg.name]
        foot_pos_xy = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]  # [B, F, 2]
        self.liftoff_pos_xy = torch.where(
            just_lifted.unsqueeze(-1), foot_pos_xy, self.liftoff_pos_xy
        )
        step_len = torch.norm(foot_pos_xy - self.liftoff_pos_xy, dim=-1)  # [B, F]
        self.last_step_len = torch.where(just_landed, step_len, self.last_step_len)

        last_air = sensor.data.last_air_time
        assert last_air is not None, (
            f"Sensor '{sensor_name}' needs track_air_time=True for "
            "gait_symmetry_penalty."
        )

        duration_gap = torch.abs(last_air[:, 0] - last_air[:, 1])  # [B]
        length_gap = torch.abs(
            self.last_step_len[:, 0] - self.last_step_len[:, 1]
        )  # [B]

        # Only comparable once *both* legs have a completed step on record.
        # Before that one side is still the zero it was reset to, which would
        # read as a large mismatch on the very first landing of an episode.
        both_stepped = (last_air > 0.0).all(dim=-1) & (
            self.last_step_len > 0.0
        ).all(dim=-1)
        landed = just_landed.any(dim=-1)

        # Gaussian on each gap rather than the raw sum, for two reasons. The raw
        # sum is unbounded, so one leg that never swings (a multi-second gap)
        # produces a penalty spike orders of magnitude past anything else in the
        # reward set. And the two gaps are in different units -- seconds and
        # metres -- so a shared linear scale silently weights one over the other,
        # whereas a per-gap std says outright how much mismatch is "a lot" for
        # each. Each term lands in (0, 1], so the mean does too, matching the
        # bounded exp kernels the rest of this task uses (track_*_velocity,
        # upright, pose, forward_step).
        duration_score = torch.exp(-((duration_gap / duration_std) ** 2))
        length_score = torch.exp(-((length_gap / length_std) ** 2))
        reward = 0.5 * (duration_score + length_score)
        reward = reward * (landed & both_stepped).float()

        command = env.command_manager.get_command(command_name)
        if command is not None:
            speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
            reward = reward * (speed > command_threshold).float()
        return reward


def lateral_symmetry_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str = "twist",
    lateral_command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize a stance that sits off to one side of the torso's mid-line.

    Both feet are transformed into the base frame and their lateral (body y)
    offsets are added. For a stance mirrored about the torso's sagittal plane
    the left foot's +y offset cancels the right foot's -y offset and the sum
    is zero; the sum grows as the pair drifts bodily to one side, which is the
    "left and right symmetric about the body centre" the gait is supposed to
    keep. Unlike feet_distance_penalty, which only sees the *gap* between the
    feet, this sees where that gap sits relative to the body -- a robot
    carrying both feet out to its left passes feet_distance and fails here.

    Summing (rather than differencing) is what makes this a mirror test: body
    y is +left/-right, so mirrored placements are equal and opposite.

    Gated on the lateral command: a commanded side-step legitimately puts both
    feet on one side of the mid-line for a while, so the term only holds the
    robot to a centred stance when little sideways motion is being asked for.

    Args:
        asset_cfg: Scene entity config whose site_names select the two foot
            sites.
        command_name: Velocity command to read the lateral component from.
        lateral_command_threshold: Above this |command_y| the term is off.
    """
    asset: Entity = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]  # [B, F, 3]
    root_pos_w = asset.data.root_link_pos_w  # [B, 3]
    root_quat_w = asset.data.root_link_quat_w  # [B, 4]

    rel_w = foot_pos_w - root_pos_w.unsqueeze(1)  # [B, F, 3]
    rel_b = quat_apply_inverse(
        root_quat_w.unsqueeze(1).expand(-1, rel_w.shape[1], -1), rel_w
    )
    lateral = rel_b[..., 1]  # [B, F], +y = robot's left
    penalty = torch.abs(lateral.sum(dim=-1))  # [B]

    command = env.command_manager.get_command(command_name)
    if command is not None:
        centred = torch.abs(command[:, 1]) <= lateral_command_threshold
        penalty = penalty * centred.float()
    return penalty


class forward_step_reward:
    """Reward a completed planar step that matches the commanded x/y velocity.

    At liftoff, this term stores both the foot position and the robot's planar
    body frame. At landing, the foot's world-frame displacement is projected
    onto those frozen body x/y axes and compared with the commanded step vector:

        target_xy = command_xy * (last_air_time + last_contact_time)

    This handles forward, backward, left, right, and diagonal commands with the
    same calculation. Freezing the axes at liftoff prevents yaw during a swing
    from being mistaken for lateral or forward foot progress.

    In steady gait a foot must cover one full stride during its swing -- the
    ground it gave up while the body moved over it during stance, plus the
    ground the body covers during the swing itself. That full cycle is
    exactly stance + swing, both of which the contact sensor has already
    measured for the cycle that just finished (last_contact_time and
    last_air_time are updated at the landing this term fires on). So the
    target is the robot's own measured cadence times the speed it was asked
    for; no assumed duty factor or step frequency.

    Scoring is a Gaussian on the 2-D vector error, so overshooting, moving in
    the wrong direction, and leaking into the perpendicular axis all reduce the
    reward.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset: Entity = env.scene[asset_cfg.name]
        num_feet = len(asset.find_sites(asset_cfg.site_names)[0])
        self.liftoff_pos_xy = torch.zeros((env.num_envs, num_feet, 2), device=env.device)
        self.liftoff_forward_xy = torch.zeros(
            (env.num_envs, num_feet, 2), device=env.device
        )
        self.liftoff_lateral_xy = torch.zeros_like(self.liftoff_forward_xy)
        self.liftoff_forward_xy[..., 0] = 1.0
        self.liftoff_lateral_xy[..., 1] = 1.0
        self.peak_height = torch.zeros((env.num_envs, num_feet), device=env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.liftoff_pos_xy[env_ids] = 0.0
        self.liftoff_forward_xy[env_ids] = 0.0
        self.liftoff_lateral_xy[env_ids] = 0.0
        self.liftoff_forward_xy[env_ids, :, 0] = 1.0
        self.liftoff_lateral_xy[env_ids, :, 1] = 1.0
        self.peak_height[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        height_sensor_name: str,
        asset_cfg: SceneEntityCfg,
        command_name: str = "twist",
        std: float = 0.05,
        command_threshold: float = 0.05,
        min_step_distance: float = 0.02,
        min_swing_height: float = 0.02,
    ) -> torch.Tensor:
        sensor: ContactSensor = env.scene[sensor_name]
        just_lifted = sensor.compute_first_air(dt=env.step_dt)  # [B, 2]
        just_landed = sensor.compute_first_contact(dt=env.step_dt)  # [B, 2]

        asset: Entity = env.scene[asset_cfg.name]
        foot_pos_xy = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]  # [B, 2, 2], world-frame

        # Build a yaw-only planar body frame. Project local +x into the world
        # xy plane, then rotate it 90 degrees to obtain the matching local +y.
        # This keeps roll/pitch from skewing the two planar axes.
        root_quat_w = asset.data.root_link_quat_w  # [B, 4]
        local_fwd = torch.zeros((env.num_envs, 3), device=env.device)
        local_fwd[:, 0] = 1.0
        forward_w = quat_apply(root_quat_w, local_fwd)  # [B, 3]
        forward_xy = forward_w[:, :2]
        forward_xy = forward_xy / forward_xy.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        lateral_xy = torch.stack((-forward_xy[:, 1], forward_xy[:, 0]), dim=-1)
        forward_xy = forward_xy.unsqueeze(1).expand(-1, foot_pos_xy.shape[1], -1)
        lateral_xy = lateral_xy.unsqueeze(1).expand(-1, foot_pos_xy.shape[1], -1)

        # Freeze both planar body axes for the entire swing.
        just_lifted_2 = just_lifted.unsqueeze(-1)
        self.liftoff_pos_xy = torch.where(just_lifted_2, foot_pos_xy, self.liftoff_pos_xy)
        self.liftoff_forward_xy = torch.where(
            just_lifted_2, forward_xy, self.liftoff_forward_xy
        )
        self.liftoff_lateral_xy = torch.where(
            just_lifted_2, lateral_xy, self.liftoff_lateral_xy
        )

        displacement_xy = foot_pos_xy - self.liftoff_pos_xy  # [B, 2, 2], net travel since liftoff, world-frame
        progress_x = (displacement_xy * self.liftoff_forward_xy).sum(dim=-1)
        progress_y = (displacement_xy * self.liftoff_lateral_xy).sum(dim=-1)
        progress_body_xy = torch.stack((progress_x, progress_y), dim=-1)  # [B, 2, 2]

        command = env.command_manager.get_command(command_name)
        command_xy = command[:, :2]  # body-frame planar command [B, 2]
        active = (torch.norm(command_xy, dim=-1) > command_threshold).unsqueeze(-1)

        # The cycle the foot just finished: its stance, then its swing. Both
        # are per-foot [B, 2] and both refer to the completed cycle at the
        # landing instant this term scores on.
        cycle_time = sensor.data.last_contact_time + sensor.data.last_air_time  # [B, 2]
        target_body_xy = command_xy.unsqueeze(1) * cycle_time.unsqueeze(-1)  # [B, 2, 2]

        error_xy = progress_body_xy - target_body_xy
        score = torch.exp(-torch.sum(torch.square(error_xy), dim=-1) / (std**2))

        # Require some real displacement before scoring at all. Without this a
        # standing-still foot (progress ~0) sitting under a near-zero command
        # would sit near its own target and collect the full Gaussian for doing
        # nothing. command_threshold already excludes the smallest commands;
        # this closes the same hole on the displacement side.
        cleared_distance = torch.norm(progress_body_xy, dim=-1) >= min_step_distance

        # Same idea but for height: track the highest ground-clearance
        # (foot_height_scan) reached since liftoff, reset at each new liftoff,
        # accumulate the running max while airborne. Requiring this to clear
        # min_swing_height (mirrors feet_crossing_reward) stops the policy from
        # sliding/dragging the foot forward along the ground -- which can
        # briefly break contact and satisfy compute_first_air/first_contact --
        # instead of actually lifting it clear like a real step.
        height_sensor = env.scene[height_sensor_name]
        heights = height_sensor.data.heights  # [B, 2]
        self.peak_height = torch.where(
            just_lifted, heights, torch.maximum(self.peak_height, heights)
        )
        cleared_height = self.peak_height >= min_swing_height

        valid_step = cleared_distance & cleared_height

        return (
            score * just_landed.float() * valid_step.float() * active.float()
        ).sum(dim=-1)


def _standing_recovery_needed(
    env: ManagerBasedRlEnv,
    max_tilt: float,
    max_lin_speed: float,
    max_ang_speed: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return whether a stationary robot needs an active balance recovery."""
    asset: Entity = env.scene[asset_cfg.name]
    gravity_b = quat_apply_inverse(
        asset.data.root_link_quat_w, asset.data.gravity_vec_w
    )
    gravity_b = gravity_b / gravity_b.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    tilt = torch.asin(torch.norm(gravity_b[:, :2], dim=-1).clamp(max=1.0))
    lin_speed = torch.norm(asset.data.root_link_lin_vel_b[:, :2], dim=-1)
    ang_speed = torch.norm(asset.data.root_link_ang_vel_b[:, :2], dim=-1)
    return (tilt > max_tilt) | (lin_speed > max_lin_speed) | (
        ang_speed > max_ang_speed
    )


def no_stepping_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    recovery_max_tilt: float = math.radians(15.0),
    recovery_max_lin_speed: float = 0.08,
    recovery_max_ang_speed: float = 0.08,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize lifting a foot during a stable commanded stand.

    The penalty is disabled while the base is tilted or moving enough to need
    an active recovery, so a stationary command does not suppress a recovery
    step after a push.
    """
    command = env.command_manager.get_command(command_name)  # (N, 3)
    cmd_speed = torch.norm(command[:, :2], dim=-1) + torch.abs(command[:, 2])
    below_threshold = cmd_speed < command_threshold

    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found  # (N, num_feet) or (N, num_feet, num_slots)
    if found.dim() == 3:
        found = found.any(dim=-1)  # (N, num_feet)
    in_air = ~found.bool()

    recovery_needed = _standing_recovery_needed(
        env,
        recovery_max_tilt,
        recovery_max_lin_speed,
        recovery_max_ang_speed,
        asset_cfg,
    )
    stable_standing = below_threshold & ~recovery_needed
    return in_air.float().sum(dim=-1) * stable_standing.float()


class feet_air_time_once_reward:
    """Reward a valid swing once, at landing.

    ``compute_first_air`` marks the beginning of each foot's swing. On the
    matching ``compute_first_contact`` event, ``last_air_time`` is checked
    against the requested interval and the reward is emitted for that single
    policy step only. No reward is paid while the foot remains airborne.
    Simultaneous two-foot landings are rejected to avoid rewarding hopping.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        sensor: ContactSensor = env.scene[cfg.params["sensor_name"]]
        current_air_time = sensor.data.current_air_time
        assert current_air_time is not None
        self.swing_started = torch.zeros_like(current_air_time, dtype=torch.bool)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.swing_started[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        threshold_min: float = 0.3,
        threshold_max: float = 1.0,
        command_name: str = "twist",
        command_threshold: float = 0.01,
    ) -> torch.Tensor:
        sensor: ContactSensor = env.scene[sensor_name]
        just_lifted = sensor.compute_first_air(dt=env.step_dt)
        just_landed = sensor.compute_first_contact(dt=env.step_dt)
        last_air_time = sensor.data.last_air_time
        assert last_air_time is not None

        self.swing_started |= just_lifted
        valid_duration = (last_air_time >= threshold_min) & (
            last_air_time <= threshold_max
        )
        valid_landing = just_landed & self.swing_started & valid_duration
        single_landing = just_landed.sum(dim=-1) == 1

        command = env.command_manager.get_command(command_name)
        assert command is not None
        cmd_speed = torch.norm(command[:, :2], dim=-1) + torch.abs(command[:, 2])
        active = cmd_speed > command_threshold

        reward = valid_landing.any(dim=-1) & single_landing & active
        self.swing_started = torch.where(
            just_landed, torch.zeros_like(self.swing_started), self.swing_started
        )
        return reward.float()


def overlong_swing_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    max_air_time: float = 0.6,
    command_name: str = "twist",
    command_threshold: float = 0.01,
    recovery_max_tilt: float = math.radians(15.0),
    recovery_max_lin_speed: float = 0.08,
    recovery_max_ang_speed: float = 0.08,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize a timed-out swing unless a stationary robot is recovering.

    ``max_air_time`` applies to both standing and moving commands; there is no
    immediate standing-only penalty. During a disturbed commanded stand the
    term is disabled completely, allowing a longer recovery step if necessary.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    current_air_time = sensor.data.current_air_time
    assert current_air_time is not None, (
        f"Sensor '{sensor_name}' needs track_air_time=True for overlong_swing_penalty."
    )
    overdue = (current_air_time > max_air_time).any(dim=-1)

    command = env.command_manager.get_command(command_name)
    assert command is not None
    cmd_speed = torch.norm(command[:, :2], dim=-1) + torch.abs(command[:, 2])
    standing = cmd_speed <= command_threshold
    recovery_needed = _standing_recovery_needed(
        env,
        recovery_max_tilt,
        recovery_max_lin_speed,
        recovery_max_ang_speed,
        asset_cfg,
    )
    suppress_for_recovery = standing & recovery_needed
    return (overdue & ~suppress_for_recovery).float()


def not_stepping_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str = "twist",
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """Penalize keeping both feet planted while a walking command is active.

    Command magnitude is ``norm(xy linear) + abs(yaw rate)``. The term returns
    1 when that magnitude is at or above ``command_threshold`` and both feet
    are in contact, otherwise 0. Use with a negative weight.
    """
    command = env.command_manager.get_command(command_name)  # (N, 3)
    cmd_speed = torch.norm(command[:, :2], dim=-1) + torch.abs(command[:, 2])
    walking_command = cmd_speed >= command_threshold

    sensor = env.scene.sensors[sensor_name]
    found = sensor.data.found  # (N, num_feet) or (N, num_feet, num_slots)
    if found.dim() == 3:
        found = found.any(dim=-1)  # (N, num_feet)
    both_feet_planted = found.bool().all(dim=-1)

    return both_feet_planted.float() * walking_command.float()


def swing_progress_reward(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    target_height: float = 0.03,
    command_name: str = "twist",
    command_threshold: float = 0.01,
) -> torch.Tensor:
    """Dense, graded reward for getting a foot off the ground at all.

    Every other stepping term in this task pays nothing until a *complete*
    step has happened: forward_step needs a landing that cleared both 2 cm of
    height and 2 cm of distance, feet_crossing needs 3 cm of swing height,
    air_time needs 0.3 s airborne. Between "both feet planted" and "a full
    step" the reward is therefore flat -- and not_stepping_penalty, being a
    constant -0.3 whenever both feet are down, does not break that tie either:
    it lowers the value of standing uniformly without saying which direction
    gets out of it. A policy that has never completed a step sees no gradient
    toward one and can only find it by chance, against the termination penalty
    if the attempt tips the robot over.

    This term fills that gap. It pays out in proportion to how far the highest
    airborne foot has risen, saturating at target_height, so lifting a foot 5
    mm already scores better than not lifting it, 1 cm better still, and the
    gradient runs continuously into the region where the gated terms take
    over.

    The maximum over feet -- not the sum -- is deliberate: rewarding both feet
    at once would pay for a two-footed hop. Taking only the highest foot means
    the best a robot can do is get one foot up, which is single support, which
    is walking.

    Contact is required as well as height: a foot resting on a ledge reads a
    nonzero terrain clearance while still being stood on, and should not count
    as a swing.

    Args:
        sensor_name: Foot/ground contact sensor, slots matching the height
            sensor's feet.
        height_sensor_name: Terrain height sensor giving per-foot ground
            clearance.
        target_height: Clearance at which this term saturates, in metres.
            Sits at/below the swing-height gates of the terms that take over
            (feet_crossing's 3 cm) so the hand-off is continuous.
        command_name: Velocity command to gate on.
        command_threshold: Below this commanded speed the term is off -- a
            robot asked to stand still should keep both feet down.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    found = sensor.data.found
    if found.dim() == 3:
        found = found.any(dim=-1)
    in_air = ~found.bool()  # [B, F]

    height_sensor = env.scene[height_sensor_name]
    heights = height_sensor.data.heights  # [B, F]

    progress = torch.clamp(heights, min=0.0, max=target_height) / target_height
    reward = (progress * in_air.float()).max(dim=-1).values  # [B]

    command = env.command_manager.get_command(command_name)
    if command is not None:
        speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        reward = reward * (speed > command_threshold).float()
    return reward


def base_height_penalty(
    env: ManagerBasedRlEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize base_link dropping below a soft height floor.

    Unlike root_height_below_minimum (a termination -- ends the episode once
    crossed), this is a continuous reward: 0 while the root is at or above
    minimum_height, growing linearly with how far below it the root is.
    Meant as a gentler, always-on nudge to stay tall (discourage crouch-
    walking/settling low) well before the episode-ending fell_over threshold
    actually fires.
    """
    asset: Entity = env.scene[asset_cfg.name]
    height = asset.data.root_link_pos_w[:, 2]
    return torch.clamp(minimum_height - height, min=0.0)


class selected_action_excess_l2:
    """Penalize selected raw actions only beyond a magnitude threshold."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        action_term = env.action_manager.get_term(cfg.params["action_name"])
        available_names = list(action_term.target_names)
        target_names = cfg.params["target_names"]
        missing = [name for name in target_names if name not in available_names]
        if missing:
            raise ValueError(
                "selected_action_excess_l2 could not find action targets: "
                f"{missing}"
            )
        self.action_ids = torch.tensor(
            [available_names.index(name) for name in target_names],
            dtype=torch.long,
            device=env.device,
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        action_name: str,
        target_names: tuple[str, ...],
        max_abs_action: float,
    ) -> torch.Tensor:
        del target_names  # Resolved once in __init__.
        raw_action = env.action_manager.get_term(action_name).raw_action
        selected = raw_action[:, self.action_ids]
        excess = torch.clamp(torch.abs(selected) - max_abs_action, min=0.0)
        return torch.sum(torch.square(excess), dim=1)


class upper_body_excursion_penalty:
    """Hard-style cap on how far each upper-body joint may swing off zero.

    All of torso_yaw, the arm joints and neck_yaw/head_pitch sit at 0.0 in
    HOME_KEYFRAME (only the six leg joints listed there are non-zero), so
    "off zero" and "off default" are the same thing for every joint this term
    is meant to cover.

    These joints have very wide physical ranges -- torso_yaw and shoulder_yaw
    are +/-180deg, elbow_pitch is -143..120deg -- so dof_pos_limits, which only
    engages at 90% of a joint's *physical* range, would never fire for any
    posture a gait (good or bad) could produce. This term is the separate,
    deliberately tight cap: 0 while |q| <= the joint's limit, growing linearly
    (and, at the caller's weight, steeply) beyond it.

    It is a backstop, not a shaping term. Each limit is set at the joint's
    ``std_running`` value from the ``pose`` reward, so the pose term does all
    the shaping inside the band and this term only fires once a joint leaves
    the range that reward would ever ask for. That also keeps it clear of
    ``contralateral_arm_swing``: with hip_pitch at its std_running 0.6 and
    gain 0.8, the arm swing asks for shoulder_pitch ~0.48, inside the 0.6 cap.

    Args:
        max_excursion: Joint-name regex -> limit [rad]. Every joint selected by
            asset_cfg must be matched by exactly one key; unmatched joints raise
            rather than being silently dropped (see below).

    ``resolve_matching_names_values`` returns values only for joints some key
    fullmatches, so a missing key would yield a limit tensor shorter than
    ``asset_cfg.joint_ids`` and broadcast wrongly against it. The pose reward
    has the same trap; here it is checked explicitly at construction.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
        _, matched_names, limits = resolve_matching_names_values(
            data=cfg.params["max_excursion"],
            list_of_strings=joint_names,
        )
        if list(matched_names) != list(joint_names):
            missing = [n for n in joint_names if n not in matched_names]
            raise ValueError(
                "max_excursion must cover every joint selected by asset_cfg; "
                f"no key fullmatches: {missing}"
            )
        self.max_excursion = torch.tensor(
            limits, device=env.device, dtype=torch.float32
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        max_excursion: dict[str, float],
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        del max_excursion  # Resolved once in __init__.
        asset: Entity = env.scene[asset_cfg.name]
        joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
        excess = torch.clamp(torch.abs(joint_pos) - self.max_excursion, min=0.0)
        return torch.sum(excess, dim=-1)


def foot_flatness_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize a *stance* foot's sole not being parallel to the ground.

    asset_cfg.body_names must select the foot bodies (e.g. left_foot_1,
    right_foot_1), in the same left/right order as sensor_name's contact
    sensor -- each foot body's local +z axis is verified (session testing) to
    point straight up in world frame when that foot sits flat on level
    ground, so rotating it into world frame and dotting with world-up is a
    direct flatness measure: 1.0 when flat, shrinking as the sole tilts.

    Only charged for a foot currently in contact (sensor_name's `found`).
    Without this gate, a foot mid-swing is *supposed* to be tilted (toe-off,
    mid-air, heel-strike are all naturally non-flat) and would get penalized
    for the exact motion every other stepping-related reward in this file is
    trying to encourage. Gating to stance keeps the part that's actually
    useful -- a planted foot should sit flat for full contact area/friction
    and to avoid rolling onto an edge -- without fighting the swing.

    Returns sum over feet of (1 - dot) * in_contact -- 0 for any airborne
    foot regardless of its orientation, and 0 for a flat stance foot, growing
    (unbounded above small tilt angles) as a *planted* foot tips onto an edge.
    """
    asset: Entity = env.scene[asset_cfg.name]
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, F, 4]
    up_local = torch.zeros_like(body_quat_w[..., :3])
    up_local[..., 2] = 1.0
    up_world = quat_apply(body_quat_w, up_local)  # [B, F, 3]
    # dot(up_world, world_up=(0,0,1)) is just up_world's own z-component.
    alignment = up_world[..., 2]
    tilt = 1.0 - alignment  # [B, F]

    sensor: ContactSensor = env.scene[sensor_name]
    found = sensor.data.found  # [B, F] or [B, F, num_slots]
    if found.dim() == 3:
        found = found.any(dim=-1)
    in_contact = found.bool()

    return torch.sum(tilt * in_contact.float(), dim=-1)


def flatness_weighted_foot_slip_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.01,
    flat_slip_scale: float = 10.0,
    tilted_slip_scale: float = 1.0,
    flat_alignment_threshold: float = math.cos(math.radians(15.0)),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize stance-foot slip more strongly when the sole is flat.

    The slip cost remains squared horizontal site velocity while in contact.
    Each foot's multiplier is smoothly interpolated from
    ``tilted_slip_scale`` at or below ``flat_alignment_threshold`` to
    ``flat_slip_scale`` when perfectly flat. A smooth transition avoids giving
    the policy a sharp angle boundary that it could exploit by tilting the foot
    only slightly.

    ``asset_cfg.site_names`` and ``asset_cfg.body_names`` must select the same
    feet in the same order.
    """
    asset: Entity = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene[sensor_name]

    found = sensor.data.found
    assert found is not None
    if found.dim() == 3:
        found = found.any(dim=-1)
    in_contact = found.bool()  # [B, F]

    foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
    slip_speed = torch.norm(foot_vel_xy, dim=-1)  # [B, F]

    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
    up_local = torch.zeros_like(body_quat_w[..., :3])
    up_local[..., 2] = 1.0
    alignment = quat_apply(body_quat_w, up_local)[..., 2].clamp(max=1.0)

    denominator = max(1.0 - flat_alignment_threshold, 1e-6)
    flatness = torch.clamp(
        (alignment - flat_alignment_threshold) / denominator,
        min=0.0,
        max=1.0,
    )
    slip_scale = tilted_slip_scale + (
        flat_slip_scale - tilted_slip_scale
    ) * flatness

    command = env.command_manager.get_command(command_name)
    assert command is not None
    command_speed = torch.norm(command[:, :2], dim=-1) + torch.abs(command[:, 2])
    active = command_speed > command_threshold

    cost = torch.sum(
        torch.square(slip_speed) * slip_scale * in_contact.float(), dim=-1
    )

    num_in_contact = in_contact.float().sum()
    mean_slip_speed = torch.sum(slip_speed * in_contact.float()) / torch.clamp(
        num_in_contact, min=1.0
    )
    env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_speed
    return cost * active.float()


class feet_crossing_reward:
    """Reward each foot once when it first clears the swing-height threshold.

    A foot is eligible only during commanded single support: exactly one foot
    is planted and the other is airborne.  The airborne foot pays one reward
    on the first step at or above ``min_swing_height``.  Remaining above the
    threshold pays nothing more.  Contact rearms that foot for its next swing.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        height_sensor = env.scene[cfg.params["height_sensor_name"]]
        self.rewarded_this_swing = torch.zeros(
            (env.num_envs, height_sensor.num_frames),
            device=env.device,
            dtype=torch.bool,
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self.rewarded_this_swing[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        height_sensor_name: str,
        min_swing_height: float = 0.02,
        command_name: str = "twist",
        command_threshold: float = 0.05,
    ) -> torch.Tensor:
        sensor: ContactSensor = env.scene[sensor_name]
        found = sensor.data.found  # [B, F] or [B, F, num_slots]
        assert found is not None
        if found.dim() == 3:
            found = found.any(dim=-1)
        in_contact = found.bool()  # [B, F]
        single_support = in_contact.sum(dim=-1) == 1  # [B]

        height_sensor = env.scene[height_sensor_name]
        heights = height_sensor.data.heights  # [B, F]

        command = env.command_manager.get_command(command_name)
        assert command is not None
        cmd_speed = torch.norm(command[:, :2], dim=-1) + torch.abs(command[:, 2])
        active = cmd_speed > command_threshold  # [B]

        eligible = (
            ~in_contact
            & (heights >= min_swing_height)
            & single_support.unsqueeze(-1)
            & active.unsqueeze(-1)
        )
        newly_reached = eligible & ~self.rewarded_this_swing

        # A landing ends the current swing and rearms that foot.  An airborne
        # foot that has already paid remains latched until that landing.
        self.rewarded_this_swing = torch.where(
            in_contact,
            torch.zeros_like(self.rewarded_this_swing),
            self.rewarded_this_swing | newly_reached,
        )
        return newly_reached.any(dim=-1).float()


def self_collision_cost_excluding_linkage(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    linkage_sensor_names: tuple[str, ...],
) -> torch.Tensor:
    """Self-collision count, minus the 4-bar linkage's own cap<->groove contact.

    SELF_COLLISION_SENSOR_CFG (kid_RL_env_cfg.py) is a single primary=
    secondary="base_link" subtree-vs-subtree sensor: any contact anywhere
    inside the robot counts, by design. That includes the hip/ankle 4-bar
    linkage's cap<->groove-wall contact (contype/conaffinity=16 in
    kid_RL_dance.xml), which is present continuously during normal
    articulation -- session testing found found>0 on ~99% of steps the
    instant any hip/ankle roll joint moved, even with root height nominal
    the whole time (i.e. not from falling). That contact is redundant with
    the tendon equality (*_cap_couple) for kinematics, so it's not a real
    self-collision to penalize, but it's left in physics untouched (unlike
    an earlier version of this fix that used spec.add_exclude()) since it
    may still be doing real constraining work and that wasn't worth risking
    unverified. Instead: linkage_sensor_names are small dedicated
    ContactSensors, one per known cap<->groove pair, whose count gets
    subtracted here so only genuine self-collisions get penalized.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    found = sensor.data.found
    assert found is not None
    total = found.sum(dim=-1).float()

    linkage_total = torch.zeros_like(total)
    for name in linkage_sensor_names:
        linkage_sensor: ContactSensor = env.scene[name]
        linkage_found = linkage_sensor.data.found
        assert linkage_found is not None
        linkage_total = linkage_total + linkage_found.sum(dim=-1).float()

    return torch.clamp(total - linkage_total, min=0.0)


########################## DIAGNOSTICS #############################


class log_solver_buffer_usage:
    """Log njmax/nconmax buffer utilization to env.extras["log"] every step.

    mujoco_warp silently drops contacts/constraints on overflow (writing past
    njmax/naconmax gets clamped) and its own overflow warnings are GPU-side
    wp.printf calls -- easy to lose under CUDA's printf FIFO limits, especially
    in exactly the high-contact-count moments (self-collision-heavy falls)
    where overflow is most likely. Comparing episode length / fell_over before
    and after a nconmax/njmax bump is an indirect, noisy proxy for whether
    overflow was actually happening (it can move the wrong way for unrelated
    reasons, e.g. curriculum/reward changes in the same run).

    This reads the true per-step required counts straight from mjwarp.Data
    (nefc, nacon) and compares them against the allocated capacity (njmax,
    naconmax) -- a direct, unambiguous signal instead of an inferred one.

    - njmax is a hard per-world cap (no sharing across envs -- see the
      nconmax/njmax explanation in chat): nefc is (num_envs,), so both the
      mean utilization and the *overflow fraction* (fraction of envs whose
      required constraint count exceeded njmax, this step) are meaningful
      per-step numbers.
    - nconmax/naconmax is a single pool shared across every env (naconmax =
      nconmax * num_envs), so nacon is one scalar for the whole batch -- only
      a single utilization/overflow value makes sense, not one per env.

    The mean/frac numbers above are diluted by ~num_envs * num_steps_per_env
    samples per logged iteration (e.g. ~49k at num_envs=2048): a rare overflow
    on a single env for a single step rounds to 0.0000 at the logger's
    precision and disappears. To catch exactly that case (a foot visibly
    clipping through the ground on rare envs/steps -- occasionally seen at
    higher num_envs, not observed at lower num_envs, seed/luck vs. a shared-
    pool contention artifact are both still live theories at that point) this
    also keeps a *cumulative*, never-reset event count across the whole run:
    Metrics/njmax_overflow_count_total and Metrics/nconmax_overflow_count_total.
    Since these barely move within one iteration's window, the logged value is
    effectively the running total so far -- one single overflow anywhere,
    ever, makes it visibly nonzero from that point on, instead of vanishing
    into a mean.

    Use with ``mode="step"`` (see ``bam_init`` for the same pattern).
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
        del cfg
        self._njmax_overflow_count = torch.zeros((), dtype=torch.long, device=env.device)
        self._nconmax_overflow_count = torch.zeros((), dtype=torch.long, device=env.device)
        # Sanity-check the capacities once at startup: a near-zero nconmax_util
        # is physically implausible for a legged robot batch (any box-terrain
        # contact alone should produce multiple contact points per env), so if
        # that shows up, we need the raw absolute numbers (not just the ratio)
        # to tell apart "nacon is genuinely ~0 (real contact-detection bug --
        # would also explain feet clipping through the ground with no
        # resisting force)" from "naconmax is bigger than expected (bug in this
        # instrumentation's reading of it)".
        print(
            f"[log_solver_buffer_usage] njmax={env.sim.data.njmax} "
            f"naconmax={env.sim.data.naconmax} (nconmax={env.sim.cfg.nconmax} "
            f"x num_envs={env.num_envs})"
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        del env_ids  # Unused; applies to every env unconditionally.

        data = env.sim.data

        nefc = data.nefc.float()  # (num_envs,) -- constraints actually required this step
        njmax = float(data.njmax)  # per-world capacity (hard cap, not shared)
        njmax_overflow = nefc > njmax
        self._njmax_overflow_count += njmax_overflow.sum()
        env.extras["log"]["Metrics/njmax_util_mean"] = (nefc / njmax).mean()
        env.extras["log"]["Metrics/njmax_overflow_frac"] = njmax_overflow.float().mean()
        env.extras["log"]["Metrics/njmax_overflow_count_total"] = (
            self._njmax_overflow_count.float()
        )

        nacon = data.nacon[0].float()  # scalar -- contacts actually required this step, all envs
        naconmax = float(data.naconmax)  # shared capacity across all envs
        nconmax_overflow = nacon > naconmax
        self._nconmax_overflow_count += nconmax_overflow.long()
        env.extras["log"]["Metrics/nconmax_util"] = nacon / naconmax
        env.extras["log"]["Metrics/nconmax_overflow"] = nconmax_overflow.float()
        env.extras["log"]["Metrics/nconmax_overflow_count_total"] = (
            self._nconmax_overflow_count.float()
        )
        # Raw absolute counts, not ratios -- a near-zero nconmax_util is
        # implausible for a legged-robot batch, so these disambiguate "nacon is
        # genuinely ~0" (real contact-detection bug) from "naconmax is bigger
        # than expected" (bug in reading it). Also nefc's raw mean, for the
        # same reason on the njmax side.
        env.extras["log"]["Metrics/nacon_raw"] = nacon
        env.extras["log"]["Metrics/nefc_mean_raw"] = nefc.mean()

        # nacon==0 across the whole batch is consistent with robots free-
        # falling with no ground contact at all (spawned too high / above the
        # contact margin) rather than actually walking-then-falling. Root
        # height directly distinguishes that: mean/min near the home height
        # (~0.476 m) with occasional dips = normal falls; something well above
        # that with min still high = free-fall, never touching down before the
        # fell_over height check (0.2 m) fires.
        asset = env.scene["robot"]
        root_z = asset.data.root_link_pos_w[:, 2]
        env.extras["log"]["Metrics/root_height_mean"] = root_z.mean()
        env.extras["log"]["Metrics/root_height_min"] = root_z.min()
        env.extras["log"]["Metrics/root_height_max"] = root_z.max()

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        del env_ids  # Cumulative counters intentionally never reset on episode reset.


########################## CURRICULUM #############################

class step_based_staged_curriculum:
    """
    Curriculum based on step count stages. Each stage is applied once when
    env.common_step_counter reaches the stage's step threshold.

    Stage definitions example:
    stages = [
        {
            "name": "stage 1",
            "step": 10_000 * 24,
            "apply": lambda env: env.reward_manager.get_term_cfg("term_name").weight = 1.0,
        },
        ...
    ]
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        self.current_stage = 0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        stages: list[dict],
    ) -> dict[str, torch.Tensor]:
        del env_ids
        if (
            self.current_stage < len(stages)
            and env.common_step_counter >= stages[self.current_stage]["step"]
        ):
            stage = stages[self.current_stage]
            print(
                f"Curriculum stage {self.current_stage + 1}: {stage['name']} at step {env.common_step_counter}"
            )
            stage["apply"](env)
            self.current_stage += 1

        return {"stage": self.current_stage}

class reward_based_staged_curriculum:
    """
    Curriculum based on stages ending while a reward component gets its mean
    episode reward accross all environments above a threshold.

    Stage definitions example:
    stages = [
        {
            "name": "stage 1",
            "reward_term_name": "term_name",
            "threshold": 0.5,
            "apply": lambda env: env.reward_manager.get_term_cfg("term_name").weight = 1.0,
        },
        ...
    ]
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        self.rewards = torch.zeros(env.num_envs, device=env.device)
        self.current_stage = 0
        self.stage_first_step = 0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        stages: list[dict],
    ) -> dict[str, torch.Tensor]:
        # Bounds check first: the indexing below reads stages[self.current_stage],
        # which is out of range once the last stage has been applied. Without this
        # the term raises IndexError on the first call after the final promotion.
        if self.current_stage >= len(stages):
            return {"stage": self.current_stage}

        self.rewards[env_ids] = (
            env.reward_manager._episode_sums[stages[self.current_stage]["reward_term_name"]][env_ids]
            / env.max_episode_length_s
        )
        mean_reward = self.rewards.mean().item()

        if (
            self.current_stage < len(stages)
            and mean_reward >= stages[self.current_stage]["threshold"]
            and env.common_step_counter >= self.stage_first_step + 100 * 24
        ):
            stage = stages[self.current_stage]
            print(
                f"Curriculum stage {self.current_stage + 1}: {stage['name']} at step {env.common_step_counter} (mean episode reward: {mean_reward:.4f})"
            )
            stage["apply"](env)
            self.current_stage += 1
            self.stage_first_step = env.common_step_counter
            self.rewards.zero_()  # Reset rewards to avoid immediately triggering the next stage

        return {"stage": self.current_stage}

class reward_based_curriculum:
    """
    Curriculum based on the mean episode reward of a specific term accross all environments.
    Once the mean reward across envs exceeds a threshold, a new curriculum stage is applied.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
        self.rewards = torch.zeros(env.num_envs, device=env.device)
        self.current_stage = 0
        self.stage_first_step = 0

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        reward_term_name: str,
        stages: list[dict],
    ) -> dict[str, torch.Tensor]:
        self.rewards[env_ids] = (
            env.reward_manager._episode_sums[reward_term_name][env_ids]
            / env.max_episode_length_s
        )
        mean_reward = self.rewards.mean().item()

        if (
            self.current_stage < len(stages)
            and mean_reward >= stages[self.current_stage]["threshold"]
            and env.common_step_counter >= self.stage_first_step + 100 * 24
        ):
            stage = stages[self.current_stage]
            print(
                f"Curriculum stage {self.current_stage + 1}: {stage['name']} at step {env.common_step_counter} (mean episode reward: {mean_reward:.4f})"
            )
            stage["apply"](env)
            self.current_stage += 1
            self.stage_first_step = env.common_step_counter

        return {"stage": self.current_stage}

def set_command_velocity(
        env,
        lin_vel_x=None,
        lin_vel_y=None,
        ang_vel_z=None,
        rotation_env_ang_vel_z=None,
) -> None:
    """
    Helper function to set the command velocity parameters in the environment.
    """
    cmd = env.command_manager.get_term_cfg("twist")
    if lin_vel_x is not None:
        cmd.ranges.lin_vel_x = lin_vel_x
    if lin_vel_y is not None:
        cmd.ranges.lin_vel_y = lin_vel_y
    if ang_vel_z is not None:
        cmd.ranges.ang_vel_z = ang_vel_z
    if rotation_env_ang_vel_z is not None:
        cmd.rotation_env_ang_vel_range = rotation_env_ang_vel_z

def set_stepping_parameters(
    env,
    air_time_weight: float | None = None,
    no_stepping_penalty_weight: float | None = None,
    rel_standing_envs: float | None = None,
    rel_rotation_envs: float | None = None,
) -> None:
    """
    Helper function to set stepping/standing curriculum parameters.
    """
    if air_time_weight is not None:
        env.reward_manager.get_term_cfg("air_time").weight = air_time_weight
    if no_stepping_penalty_weight is not None:
        env.reward_manager.get_term_cfg("no_stepping").weight = no_stepping_penalty_weight
    if rel_standing_envs is not None:
        env.command_manager.get_term_cfg("twist").rel_standing_envs = rel_standing_envs
    if rel_rotation_envs is not None:
        env.command_manager.get_term_cfg("twist").rel_rotation_envs = rel_rotation_envs

def set_push_parameters(
    env,
    velocity_range: dict[str, tuple[float, float]] | None = None,
    interval_range: tuple[float, float] | None = None,
) -> None:
    """Update push-event velocity and interval parameters."""
    push_event_cfg = env.event_manager.get_term_cfg("push_robot")
    if velocity_range is not None:
        push_event_cfg.params["velocity_range"] = velocity_range
    if interval_range is not None:
        push_event_cfg.params["interval_range"] = interval_range


def make_interpolated_push_stages(
    num_stages: int,
    reward_term_name: str,
    threshold_start: float,
    threshold_end: float,
    push_full_scale: dict[str, float],
    push_scale_start: float = 0.0,
    push_scale_end: float = 1.0,
) -> list[dict]:
    """Build a push-only reward-based curriculum by interpolation.

    Interpolating rather than enumerating also keeps the ramp self-consistent:
    changing num_stages re-spaces every value, so the endpoints stay put and
    only the granularity changes, and there is no way for a hand-edited stage to
    end up out of order or to skip a knob.

    Push is expressed as a scalar fraction of ``push_full_scale`` because the
    stages it replaces were exactly 25/50/75/100% of one full-scale range --
    a single number, not four independent ones. Each axis becomes the
    symmetric range (-scale * full, +scale * full).

    Args:
        num_stages: Number of promotions from start to end, inclusive of the
            final one. The start values are *not* applied as a stage -- they
            are whatever the config already set, and stage 1 is the first move
            away from them.
        reward_term_name: Reward term whose mean episode value gates promotion.
        threshold_start: Threshold that promotes stage 1, exactly.
        threshold_end: Threshold that promotes the final stage, exactly.
        push_full_scale: Per-axis magnitude at push_scale_end, e.g.
            {"x": 0.2, "y": 0.2, "roll": 0.22, "pitch": 0.22}.
        push_scale_start / push_scale_end: fraction of push_full_scale at the
            beginning and end of the ramp.

    Returns:
        A list of stage dicts for reward_based_staged_curriculum.
    """
    if num_stages < 1:
        raise ValueError(f"num_stages must be >= 1, got {num_stages}")

    def lerp(a: float, b: float, t: float) -> float:
        return a + (b - a) * t

    stages: list[dict] = []
    for i in range(1, num_stages + 1):
        # Values and thresholds run on different clocks on purpose.
        #
        # Values step *off* the starting point: stage 1 is already a change, so
        # t_value starts at 1/num_stages, and the last stage lands exactly on
        # the end values.
        #
        # Thresholds instead span the full range inclusive, so stage 1 promotes
        # at threshold_start and the last at threshold_end. Sharing one clock
        # would push the first gate up to lerp(start, end, 1/num_stages) --
        # with the numbers this replaced, 0.23 would have become 0.40, i.e. the
        # curriculum would need noticeably better walking before it moved at all.
        t_value = i / num_stages
        t_threshold = (i - 1) / (num_stages - 1) if num_stages > 1 else 0.0
        t = t_value
        push_scale = lerp(push_scale_start, push_scale_end, t)
        velocity_range = {
            axis: (-push_scale * full, push_scale * full)
            for axis, full in push_full_scale.items()
        }

        # Bind the interpolated value as a default argument. These functions are
        # called after the loop has finished; a bare closure would make every
        # stage apply the final push range.
        def apply(env, _velocity_range=velocity_range):
            set_push_parameters(env, velocity_range=_velocity_range)

        stages.append(
            {
                "name": f"{i}/{num_stages}: push {push_scale * 100:.0f}%",
                "reward_term_name": reward_term_name,
                "threshold": lerp(threshold_start, threshold_end, t_threshold),
                "apply": apply,
            }
        )
    return stages


def penalize_stepping_while_standing(
    env: ManagerBasedRlEnv,
    air_time_weight: float,
    no_stepping_penalty_weight: float,
) -> torch.Tensor:
    """
    Updating the air_time and no_stepping reward weights to penalize stepping while standing.
    """
    env.reward_manager.get_term_cfg("air_time").weight = air_time_weight
    env.reward_manager.get_term_cfg("no_stepping").weight = no_stepping_penalty_weight

def stepping_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    air_time_weight: float,
    no_stepping_penalty_weight: float,
    rel_standing_envs: float = 0.0,
    rel_rotation_envs: float = 0.0,
    step: int = 10000 * 24,
) -> dict[str, torch.Tensor]:
    """
    Updating the air_time and no_stepping reward weights to penalize stepping while standing
    after a certain number of iterations.
    """
    del env_ids  # Unused.

    if env.common_step_counter >= step:
        env.reward_manager.get_term_cfg("air_time").weight = air_time_weight
        env.reward_manager.get_term_cfg("no_stepping").weight = no_stepping_penalty_weight
        env.command_manager.get_term_cfg("twist").rel_standing_envs = rel_standing_envs
        env.command_manager.get_term_cfg("twist").rel_rotation_envs = rel_rotation_envs

    return {
        "air_time_weight": torch.tensor(env.reward_manager.get_term_cfg("air_time").weight),
        "no_stepping_penalty_weight": torch.tensor(env.reward_manager.get_term_cfg("no_stepping").weight),
        "rel_standing_envs": torch.tensor(env.command_manager.get_term_cfg("twist").rel_standing_envs),
        "rel_rotation_envs": torch.tensor(env.command_manager.get_term_cfg("twist").rel_rotation_envs),
    }


class contralateral_arm_swing:
    """Reward human-like arm swing: each arm swings with the opposite leg.

    Converts four pitch joints into a signed "forward-ness" (positive = that limb
    is reaching toward +x in the body frame) and rewards the right arm tracking
    the left leg, and the left arm tracking the right leg.

    The per-joint signs below are not a convention -- they were measured on the
    compiled model, because kid_RL_v3's left/right pitch axes are mirrored in the
    MJCF. Driving a joint +0.3 rad moves the limb tip forward for right_hip_pitch
    and right_shoulder_pitch, and backward for left_hip_pitch and
    left_shoulder_pitch. (The shoulder pair was intentionally flipped from its
    original CAD-derived axes -- both hips still mirror the original mapping.)

    Args:
        std: Gaussian kernel width on the coupling error [rad].
        gain: Arm swing amplitude relative to the leg it follows. 1.0 means the
            arm mirrors the leg angle; below 1.0 gives a subtler swing.
        command_name: Velocity command to gate on.
        command_threshold: Below this commanded speed the term is switched off,
            so a standing robot is not asked to swing its arms.
    """

    _JOINTS = (
        "left_hip_pitch",
        "right_hip_pitch",
        "left_shoulder_pitch",
        "right_shoulder_pitch",
    )
    _FORWARD_SIGN = (-1.0, 1.0, -1.0, 1.0)  # measured, see docstring

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        ids = []
        for name in self._JOINTS:
            found, _ = asset.find_joints((f"^{name}$",))
            assert len(found) == 1, f"{name!r} matched {len(found)} joints"
            ids.append(found[0])
        self.joint_ids = torch.tensor(ids, dtype=torch.long, device=env.device)
        self.forward_sign = torch.tensor(
            self._FORWARD_SIGN, dtype=torch.float32, device=env.device
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std: float,
        gain: float = 1.0,
        command_name: str | None = None,
        command_threshold: float = 0.0,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        default = asset.data.default_joint_pos[:, self.joint_ids]
        fwd = (asset.data.joint_pos[:, self.joint_ids] - default) * self.forward_sign
        left_leg, right_leg, left_arm, right_arm = fwd.unbind(dim=1)

        error = torch.square(right_arm - gain * left_leg) + torch.square(
            left_arm - gain * right_leg
        )
        reward = torch.exp(-error / std**2)

        if command_name is not None:
            command = env.command_manager.get_command(command_name)
            if command is not None:
                speed = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
                reward = reward * (speed > command_threshold).float()
        return reward

    def reset(self, env_ids: torch.Tensor) -> None:
        del env_ids  # Unused.
