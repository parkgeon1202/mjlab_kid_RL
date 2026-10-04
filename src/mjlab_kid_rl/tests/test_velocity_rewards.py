"""Focused tests for kid_RL velocity-tracking reward terms."""

import math
import types
import unittest

import torch

from mjlab_kid_rl.tasks.mdp import (
  airborne_foot_arm_swing_reward,
  default_joint_pose_exp,
  feet_air_time_continuous_reward,
  both_feet_airborne_penalty,
  gait_symmetry_reward,
  feet_crossing_reward,
  feet_distance_penalty,
  foot_landing_alignment_reward,
  gait_phase_swing_clearance_reward,
  swing_progress_reward,
  flatness_weighted_foot_slip_penalty,
  foot_base_heading_error_penalty,
  forward_step_reward,
  no_stepping_penalty,
  not_stepping_penalty,
  not_stepping_each_foot_penalty,
  overlong_swing_penalty,
  relative_angular_velocity_error_penalty,
  selected_action_excess_l2,
  track_angular_velocity_gated,
  track_linear_velocity_gated,
  upright,
)


class _TestScene(dict):
  @property
  def sensors(self):
    return self


class TestFootLandingAlignment(unittest.TestCase):
  def setUp(self):
    self.contact = torch.ones((1, 2), dtype=torch.bool)
    self.landed = torch.tensor([[True, False]])
    self.air_time = torch.zeros((1, 2))
    self.quat = torch.tensor([[[1., 0., 0., 0.], [1., 0., 0., 0.]]])
    self.terrain = types.SimpleNamespace(
      normals_w=torch.tensor([[[0., 0., 1.]] * 4]),
      hit_pos_w=torch.zeros((1, 4, 3)),
      distances=torch.ones((1, 4)),
    )
    sensor = types.SimpleNamespace(
      data=types.SimpleNamespace(found=self.contact, current_air_time=self.air_time),
      compute_first_contact=lambda dt: self.landed,
    )
    self.env = types.SimpleNamespace(
      num_envs=1, device="cpu", step_dt=0.02,
      scene=_TestScene(
        contact=sensor, height=types.SimpleNamespace(data=self.terrain),
        robot=types.SimpleNamespace(data=types.SimpleNamespace(body_link_quat_w=self.quat)),
      ),
    )
    self.term = foot_landing_alignment_reward(None, self.env)
    self.asset_cfg = types.SimpleNamespace(name="robot", body_ids=[0, 1])

  def reward(self):
    return self.term(self.env, "contact", "height", self.asset_cfg)

  def test_only_first_contact_is_paid(self):
    torch.testing.assert_close(self.reward(), torch.ones(1))
    self.landed[:] = False
    torch.testing.assert_close(self.reward(), torch.zeros(1))

  def test_sixty_degree_misalignment(self):
    self.quat[0, 0] = torch.tensor([math.cos(math.pi/6), 0., math.sin(math.pi/6), 0.])
    torch.testing.assert_close(self.reward(), torch.tensor([0.5]))

  def test_terrain_normal_on_slope(self):
    self.quat[0, 0] = torch.tensor([math.cos(math.pi/6), 0., math.sin(math.pi/6), 0.])
    self.terrain.normals_w[0, :2] = torch.tensor([math.sin(math.pi/3), 0., 0.5])
    torch.testing.assert_close(self.reward(), torch.ones(1))

  def test_highest_valid_ray_and_misses(self):
    self.terrain.hit_pos_w[0, 0, 2] = -1.
    self.terrain.normals_w[0, 0] = torch.tensor([1., 0., 0.1])
    torch.testing.assert_close(self.reward(), torch.ones(1))
    self.terrain.distances[:] = -1.
    torch.testing.assert_close(self.reward(), torch.zeros(1))

  def test_hop_and_staggered_landing_are_excluded(self):
    self.contact[:] = False
    self.air_time[:] = 0.08
    self.landed[:] = False
    torch.testing.assert_close(self.reward(), torch.zeros(1))
    self.contact[0, 0] = True
    self.landed[0, 0] = True
    torch.testing.assert_close(self.reward(), torch.zeros(1))
    self.contact[:] = True
    self.landed[:] = torch.tensor([[False, True]])
    torch.testing.assert_close(self.reward(), torch.zeros(1))
    self.landed[:] = torch.tensor([[True, False]])
    torch.testing.assert_close(self.reward(), torch.ones(1))

  def test_simultaneous_landing_and_reset(self):
    self.landed[:] = True
    torch.testing.assert_close(self.reward(), torch.zeros(1))
    self.term.hop_in_progress[:] = True
    self.term.reset(torch.tensor([0]))
    self.landed[:] = torch.tensor([[True, False]])
    torch.testing.assert_close(self.reward(), torch.ones(1))


class TestDefaultJointPoseReward(unittest.TestCase):
  def test_max_at_default_and_rewards_only_standing(self) -> None:
    default = torch.tensor([[0.1, -0.2]])
    asset = types.SimpleNamespace(
      data=types.SimpleNamespace(
        joint_pos=default.clone(),
        default_joint_pos=default,
      )
    )
    command = torch.zeros((1, 3))
    env = types.SimpleNamespace(
      scene=_TestScene(robot=asset),
      command_manager=types.SimpleNamespace(get_command=lambda _: command),
    )
    asset_cfg = types.SimpleNamespace(name="robot", joint_ids=[0, 1])

    def deviation() -> torch.Tensor:
      return default_joint_pose_exp(
        env, std=0.15, command_name="twist",
        walking_threshold=0.01, asset_cfg=asset_cfg,
      )

    torch.testing.assert_close(deviation(), torch.ones(1))
    asset.data.joint_pos[0, 0] += 0.15
    torch.testing.assert_close(deviation(), torch.tensor([math.exp(-0.5)]))
    command[0, 0] = 0.2
    torch.testing.assert_close(deviation(), torch.zeros(1))


class TestAirborneFootArmSwingReward(unittest.TestCase):
  def setUp(self) -> None:
    joint_names = ["left_shoulder_pitch", "right_shoulder_pitch"]
    self.asset = types.SimpleNamespace(
      data=types.SimpleNamespace(
        joint_pos=torch.zeros((1, 2)),
        default_joint_pos=torch.zeros((1, 2)),
      ),
      find_joints=lambda patterns: (
        [joint_names.index(patterns[0].strip("^$") )],
        [patterns[0].strip("^$")],
      ),
    )
    self.sensor = types.SimpleNamespace(
      # The real contact sensor exposes numeric 0/1 values rather than a
      # Boolean tensor.
      data=types.SimpleNamespace(found=torch.tensor([[1.0, 1.0]]))
    )
    self.command = torch.zeros((1, 3))
    self.arm_action = types.SimpleNamespace(
      target_names=airborne_foot_arm_swing_reward._ARM_JOINTS,
      raw_action=torch.zeros((1, 10)),
    )
    self.env = types.SimpleNamespace(
      device="cpu",
      scene=_TestScene(robot=self.asset, feet=self.sensor),
      command_manager=types.SimpleNamespace(get_command=lambda _: self.command),
      action_manager=types.SimpleNamespace(get_term=lambda _: self.arm_action),
    )
    cfg = types.SimpleNamespace(
      params={
        "asset_cfg": types.SimpleNamespace(name="robot"),
        "action_name": "joint_pos",
      }
    )
    self.term = airborne_foot_arm_swing_reward(cfg, self.env)

  def _reward(self) -> torch.Tensor:
    return self.term(
      self.env,
      sensor_name="feet",
      target_angle=0.25,
      std=0.25,
      min_forward_command=0.1,
      max_lateral_command=0.1,
      max_yaw_command=0.05,
      action_name="joint_pos",
      asset_cfg=types.SimpleNamespace(name="robot"),
    )

  def test_left_airborne_targets_right_arm_forward(self) -> None:
    self.command[:] = torch.tensor([[0.2, 0.0, 0.0]])
    self.sensor.data.found[:] = torch.tensor([[False, True]])
    # With mirrored shoulder axes, +0.25 rad sends the left arm backward and
    # the right arm forward.
    self.asset.data.joint_pos[:] = 0.25
    torch.testing.assert_close(self._reward(), torch.ones(1))

  def test_right_airborne_targets_left_arm_forward(self) -> None:
    self.command[:, 0] = 0.2
    self.sensor.data.found[:] = torch.tensor([[True, False]])
    self.asset.data.joint_pos[:] = -0.25
    torch.testing.assert_close(self._reward(), torch.ones(1))

  def test_one_wrong_arm_receives_penalty(self) -> None:
    self.command[:, 0] = 0.2
    self.sensor.data.found[:] = torch.tensor([[False, True]])
    # Left arm is correctly back; right arm is incorrectly back too.
    self.asset.data.joint_pos[:] = torch.tensor([[0.25, -0.25]])
    torch.testing.assert_close(self._reward(), torch.tensor([-1.0]))

  def test_stationary_arm_receives_penalty(self) -> None:
    self.command[:, 0] = 0.2
    self.sensor.data.found[:] = torch.tensor([[True, False]])
    # Right arm is correctly back; left arm stays at its default.
    self.asset.data.joint_pos[:] = torch.tensor([[0.0, -0.25]])
    torch.testing.assert_close(self._reward(), torch.tensor([-0.5]))

  def test_nonstraight_command_penalizes_all_arm_actions(self) -> None:
    self.command[:] = torch.tensor([[0.2, 0.12, 0.0]])
    self.sensor.data.found[:] = torch.tensor([[False, True]])
    self.asset.data.joint_pos[:] = 0.25
    self.arm_action.raw_action[0, 0] = 0.5  # shoulder pitch
    self.arm_action.raw_action[0, 8] = -0.3  # elbow pitch
    torch.testing.assert_close(self._reward(), torch.tensor([-0.34]))

    self.command[:] = torch.tensor([[0.2, 0.0, 0.1]])
    torch.testing.assert_close(self._reward(), torch.tensor([-0.34]))

    self.command[:] = torch.tensor([[0.0, 0.0, 0.0]])
    torch.testing.assert_close(self._reward(), torch.tensor([-0.34]))

  def test_straight_forward_uses_arm_alignment_instead_of_action_cost(self) -> None:
    self.command[:] = torch.tensor([[0.2, 0.1, 0.0]])
    self.sensor.data.found[:] = torch.tensor([[False, True]])
    self.asset.data.joint_pos[:] = 0.25
    self.arm_action.raw_action[:] = 1.0
    torch.testing.assert_close(self._reward(), torch.ones(1))

  def test_zero_when_both_feet_share_contact_state(self) -> None:
    self.command[:, 0] = 0.2
    for found in ([[True, True]], [[False, False]]):
      self.sensor.data.found[:] = torch.tensor(found)
      torch.testing.assert_close(self._reward(), torch.zeros(1))


class TestFlatnessWeightedFootSlipPenalty(unittest.TestCase):
  asset_cfg = types.SimpleNamespace(
    name="robot", site_ids=[0, 1], body_ids=[0, 1]
  )

  def _penalty(self, foot_quat_w: torch.Tensor) -> torch.Tensor:
    asset = types.SimpleNamespace(
      data=types.SimpleNamespace(
        site_lin_vel_w=torch.tensor(
          [[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
        ),
        body_link_quat_w=foot_quat_w,
      )
    )
    sensor = types.SimpleNamespace(
      data=types.SimpleNamespace(found=torch.tensor([[True, True]]))
    )
    command = torch.tensor([[0.2, 0.0, 0.0]])
    env = types.SimpleNamespace(
      scene={"robot": asset, "feet": sensor},
      command_manager=types.SimpleNamespace(get_command=lambda _: command),
      extras={"log": {}},
    )
    return flatness_weighted_foot_slip_penalty(
      env,
      sensor_name="feet",
      command_name="twist",
      flat_slip_scale=10.0,
      tilted_slip_scale=1.0,
      flat_alignment_threshold=0.95,
      asset_cfg=self.asset_cfg,
    )

  def test_flat_contact_uses_large_slip_scale(self) -> None:
    identity = torch.tensor(
      [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]
    )
    torch.testing.assert_close(self._penalty(identity), torch.tensor([10.0]))

  def test_tilted_contact_uses_small_slip_scale(self) -> None:
    # 90 degrees about x: the local foot-up axis is horizontal.
    s = 2.0**-0.5
    tilted = torch.tensor(
      [[[s, s, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]
    )
    torch.testing.assert_close(self._penalty(tilted), torch.tensor([1.0]))


class TestFeetDistancePenalty(unittest.TestCase):
  asset_cfg = types.SimpleNamespace(name="robot", site_ids=[0, 1])

  def _penalty(self, separation: float) -> torch.Tensor:
    half = separation / 2.0
    asset = types.SimpleNamespace(
      data=types.SimpleNamespace(
        site_pos_w=torch.tensor([[[0.0, half, 0.0], [0.0, -half, 0.0]]])
      )
    )
    env = types.SimpleNamespace(scene={"robot": asset})
    return feet_distance_penalty(
      env,
      min_dist=0.10,
      max_dist=0.2,
      asset_cfg=self.asset_cfg,
    )

  def test_below_minimum_is_penalized_by_excess_only(self) -> None:
    torch.testing.assert_close(self._penalty(0.08), torch.tensor([0.02]))

  def test_inside_range_has_no_penalty(self) -> None:
    torch.testing.assert_close(self._penalty(0.15), torch.zeros(1))

  def test_above_maximum_is_penalized_by_excess_only(self) -> None:
    torch.testing.assert_close(self._penalty(0.22), torch.tensor([0.02]))


class TestSelectedActionExcessL2(unittest.TestCase):
  target_names = (
    "left_hip_roll_crank",
    "right_hip_roll_crank",
    "left_ankle_roll_crank",
    "right_ankle_roll_crank",
  )

  def _term(self, raw_action: torch.Tensor) -> selected_action_excess_l2:
    action_term = types.SimpleNamespace(
      target_names=[
        "unrelated_joint",
        "right_ankle_roll_crank",
        "left_hip_roll_crank",
        "right_hip_roll_crank",
        "left_ankle_roll_crank",
      ],
      raw_action=raw_action,
    )
    self.env = types.SimpleNamespace(
      device=torch.device("cpu"),
      action_manager=types.SimpleNamespace(get_term=lambda _: action_term),
    )
    cfg = types.SimpleNamespace(
      params={"action_name": "joint_pos", "target_names": self.target_names}
    )
    return selected_action_excess_l2(cfg, self.env)

  def test_has_no_penalty_inside_threshold(self) -> None:
    term = self._term(torch.tensor([[100.0, -2.0, 1.0, 2.0, -1.5]]))
    torch.testing.assert_close(
      term(self.env, "joint_pos", self.target_names, 2.0), torch.zeros(1)
    )

  def test_squares_only_selected_excess(self) -> None:
    term = self._term(torch.tensor([[100.0, -15.0, 2.5, -3.0, 1.0]]))
    # Selected excesses are 0.5, 1.0, 0.0, and 13.0. The unrelated 100 is ignored.
    torch.testing.assert_close(
      term(self.env, "joint_pos", self.target_names, 2.0),
      torch.tensor([170.25]),
    )

  def test_missing_target_is_rejected(self) -> None:
    action_term = types.SimpleNamespace(target_names=["left_hip_roll_crank"])
    env = types.SimpleNamespace(
      device=torch.device("cpu"),
      action_manager=types.SimpleNamespace(get_term=lambda _: action_term),
    )
    cfg = types.SimpleNamespace(
      params={"action_name": "joint_pos", "target_names": self.target_names}
    )
    with self.assertRaisesRegex(ValueError, "could not find action targets"):
      selected_action_excess_l2(cfg, env)


class TestUprightReward(unittest.TestCase):
  def setUp(self) -> None:
    self.command = torch.tensor([[0.2, 0.0, 0.0]])
    self.asset = types.SimpleNamespace(data=types.SimpleNamespace(
      body_link_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
      gravity_vec_w=torch.tensor([[0.0, 0.0, -1.0]]),
    ))
    self.env = types.SimpleNamespace(
      scene={"robot": self.asset},
      command_manager=types.SimpleNamespace(get_command=lambda _: self.command),
    )
    self.term = upright(types.SimpleNamespace(params={}), self.env)

  def _reward(self) -> torch.Tensor:
    return self.term(
      self.env, std=0.2, pitch=0.0, standing_pitch=0.0,
      asset_cfg=types.SimpleNamespace(name="robot", body_ids=[0]),
      command_threshold=0.01,
    )

  def test_moving_command_pays_without_foot_or_speed_gate(self) -> None:
    torch.testing.assert_close(self._reward(), torch.ones(1))

  def test_standing_command_pays(self) -> None:
    self.command[:] = 0.0
    torch.testing.assert_close(self._reward(), torch.ones(1))

  def test_tilt_reduces_reward_at_narrower_std(self) -> None:
    angle = 0.2
    self.asset.data.gravity_vec_w[:] = torch.tensor([
      [math.sin(angle), 0.0, -math.cos(angle)]
    ])
    expected = torch.exp(torch.tensor(-math.sin(angle) ** 2 / 0.2**2))
    torch.testing.assert_close(self._reward(), expected.unsqueeze(0))


class TestSplitVelocityTracking(unittest.TestCase):
  def setUp(self) -> None:
    self.sensor = types.SimpleNamespace(
      data=types.SimpleNamespace(
        found=torch.tensor([[1.0, 1.0]]),
        current_air_time=torch.zeros((1, 2)),
        last_air_time=torch.zeros((1, 2)),
      ),
      first_air=torch.zeros((1, 2), dtype=torch.bool),
      first_contact=torch.zeros((1, 2), dtype=torch.bool),
    )
    self.sensor.compute_first_air = lambda dt: self.sensor.first_air
    self.sensor.compute_first_contact = lambda dt: self.sensor.first_contact
    self.height = types.SimpleNamespace(
      data=types.SimpleNamespace(heights=torch.zeros((1, 2)))
    )
    self.command = torch.tensor([[0.2, 0.0, 0.0]])
    asset = types.SimpleNamespace(data=types.SimpleNamespace(
      root_link_lin_vel_b=torch.tensor([[0.2, 0.0, 0.0]]),
      root_link_ang_vel_b=torch.zeros((1, 3)),
    ))
    self.env = types.SimpleNamespace(
      num_envs=1, device="cpu", step_dt=0.02,
      scene={"robot": asset, "feet": self.sensor, "height": self.height},
      command_manager=types.SimpleNamespace(get_command=lambda _: self.command),
    )
    cfg = types.SimpleNamespace(params={"sensor_name": "feet"})
    self.terms = [
      track_linear_velocity_gated(cfg, self.env),
      track_angular_velocity_gated(cfg, self.env),
    ]

  def _rewards(self) -> torch.Tensor:
    return torch.stack([
      term(
        self.env, std=0.3, command_name="twist", sensor_name="feet",
        height_sensor_name="height", command_threshold=0.01,
        max_air_time=0.5,
      ) for term in self.terms
    ]).flatten()

  def test_pays_half_at_two_centimeter_lift_and_half_at_landing(self) -> None:
    # Exact speed tracking alone does not bypass the landing requirement.
    torch.testing.assert_close(self._rewards(), torch.zeros(2))
    self.sensor.data.found[0, 0] = 0.0
    self.sensor.first_air[0, 0] = True
    self.sensor.data.current_air_time[0, 0] = 0.02
    self.height.data.heights[0, 0] = 0.01
    torch.testing.assert_close(self._rewards(), torch.zeros(2))
    self.sensor.first_air[:] = False
    self.height.data.heights[0, 0] = 0.03
    torch.testing.assert_close(self._rewards(), torch.full((2,), 0.5))
    torch.testing.assert_close(self._rewards(), torch.zeros(2))
    self.sensor.data.found[0, 0] = 1.0
    self.sensor.first_contact[0, 0] = True
    self.sensor.data.last_air_time[0, 0] = 0.2
    torch.testing.assert_close(self._rewards(), torch.full((2,), 0.5))
    self.sensor.first_contact[:] = False
    torch.testing.assert_close(self._rewards(), torch.zeros(2))

  def test_no_reward_without_height_or_after_overlong_swing(self) -> None:
    self.sensor.data.found[0, 0] = 0.0
    self.sensor.first_air[0, 0] = True
    self.sensor.data.current_air_time[0, 0] = 0.02
    self.height.data.heights[0, 0] = 0.01
    self._rewards()
    self.sensor.first_air[:] = False
    self.sensor.data.found[0, 0] = 1.0
    self.sensor.first_contact[0, 0] = True
    self.sensor.data.last_air_time[0, 0] = 0.2
    torch.testing.assert_close(self._rewards(), torch.zeros(2))
    self.sensor.first_contact[:] = False
    self.sensor.data.found[0, 0] = 0.0
    self.sensor.first_air[0, 0] = True
    self.sensor.data.current_air_time[0, 0] = 0.2
    self.height.data.heights[0, 0] = 0.03
    torch.testing.assert_close(self._rewards(), torch.full((2,), 0.5))
    self.sensor.data.found[0, 0] = 1.0
    self.sensor.first_contact[0, 0] = True
    self.sensor.data.last_air_time[0, 0] = 0.7
    torch.testing.assert_close(self._rewards(), torch.zeros(2))

  def test_standing_command_keeps_continuous_reward(self) -> None:
    self.command[:] = 0.0
    self.env.scene["robot"].data.root_link_lin_vel_b[:] = 0.0
    torch.testing.assert_close(self._rewards(), torch.ones(2))


class TestContinuousAirTimeReward(unittest.TestCase):
  def setUp(self) -> None:
    self.sensor = types.SimpleNamespace(data=types.SimpleNamespace(
      found=torch.tensor([[True, True]]),
      current_air_time=torch.tensor([[0.0, 0.0]]),
    ))
    self.command = torch.tensor([[0.2, 0.0, 0.0]])
    self.env = types.SimpleNamespace(
      scene={"feet": self.sensor},
      command_manager=types.SimpleNamespace(get_command=lambda _: self.command),
    )

  def _reward(self) -> torch.Tensor:
    return feet_air_time_continuous_reward(
      self.env, sensor_name="feet", threshold_min=0.3,
      threshold_max=0.7, command_threshold=0.01,
    )

  def test_pays_every_step_from_point_three_until_point_seven_seconds(self) -> None:
    self.sensor.data.found[0, 0] = False
    self.sensor.data.current_air_time[0, 0] = 0.29
    torch.testing.assert_close(self._reward(), torch.zeros(1))
    for duration in (0.3, 0.5, 0.7):
      self.sensor.data.current_air_time[0, 0] = duration
      torch.testing.assert_close(self._reward(), torch.ones(1))
    self.sensor.data.current_air_time[0, 0] = 0.71
    torch.testing.assert_close(self._reward(), torch.zeros(1))

  def test_landing_does_not_pay(self) -> None:
    self.sensor.data.found[0, 0] = False
    self.sensor.data.current_air_time[0, 0] = 0.3
    torch.testing.assert_close(self._reward(), torch.ones(1))
    self.sensor.data.found[0, 0] = True
    torch.testing.assert_close(self._reward(), torch.zeros(1))

  def test_two_footed_hop_does_not_pay(self) -> None:
    self.sensor.data.found[:] = False
    self.sensor.data.current_air_time[:] = 0.3
    torch.testing.assert_close(self._reward(), torch.zeros(1))

  def test_standing_command_does_not_pay(self) -> None:
    self.command[:] = 0.0
    self.sensor.data.found[0, 0] = False
    self.sensor.data.current_air_time[0, 0] = 0.3
    torch.testing.assert_close(self._reward(), torch.zeros(1))


class TestDoubleFlightPenalty(unittest.TestCase):
  def test_only_sustained_double_flight_during_walking_is_charged(self) -> None:
    sensor = types.SimpleNamespace(data=types.SimpleNamespace(
      found=torch.tensor([[False, False]]),
      current_air_time=torch.tensor([[0.04, 0.04]]),
    ))
    command = torch.tensor([[0.2, 0.0, 0.0]])
    env = types.SimpleNamespace(
      scene={"feet": sensor},
      command_manager=types.SimpleNamespace(get_command=lambda _: command),
    )
    def reward() -> torch.Tensor:
      return both_feet_airborne_penalty(
        env, sensor_name="feet", min_air_time_s=0.06,
      )
    torch.testing.assert_close(reward(), torch.zeros(1))
    sensor.data.current_air_time[:] = 0.08
    torch.testing.assert_close(reward(), torch.ones(1))
    sensor.data.found[0, 0] = True
    torch.testing.assert_close(reward(), torch.zeros(1))
    sensor.data.found[0, 0] = False
    command[:] = 0.0
    torch.testing.assert_close(reward(), torch.zeros(1))


class TestGaitSymmetryHopGate(unittest.TestCase):
  def test_alternating_walk_pays_but_staggered_hop_does_not(self) -> None:
    lifted = torch.zeros((1, 2), dtype=torch.bool)
    landed = torch.tensor([[True, False]])
    sensor = types.SimpleNamespace(
      data=types.SimpleNamespace(
        found=torch.tensor([[True, True]]),
        current_air_time=torch.zeros((1, 2)),
        last_air_time=torch.full((1, 2), 0.3),
      ),
      compute_first_air=lambda dt: lifted,
      compute_first_contact=lambda dt: landed,
    )
    asset = types.SimpleNamespace(
      data=types.SimpleNamespace(site_pos_w=torch.tensor([[[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]]])),
      find_sites=lambda names: ([0, 1], names),
    )
    env = types.SimpleNamespace(
      num_envs=1, device=torch.device("cpu"), step_dt=0.02,
      scene={"feet": sensor, "robot": asset},
      command_manager=types.SimpleNamespace(get_command=lambda _: torch.tensor([[0.2, 0.0, 0.0]])),
    )
    asset_cfg = types.SimpleNamespace(name="robot", site_names=("left", "right"), site_ids=[0, 1])
    term = gait_symmetry_reward(types.SimpleNamespace(params={"asset_cfg": asset_cfg}), env)
    def reward() -> torch.Tensor:
      return term(env, sensor_name="feet", asset_cfg=asset_cfg)

    torch.testing.assert_close(reward(), torch.zeros(1))
    landed[:] = torch.tensor([[False, True]])
    torch.testing.assert_close(reward(), torch.ones(1))

    landed[:] = False
    sensor.data.found[:] = False
    sensor.data.current_air_time[:] = 0.08
    torch.testing.assert_close(reward(), torch.zeros(1))
    landed[:] = torch.tensor([[True, False]])
    sensor.data.found[:] = torch.tensor([[True, False]])
    torch.testing.assert_close(reward(), torch.zeros(1))
    landed[:] = torch.tensor([[False, True]])
    sensor.data.found[:] = True
    torch.testing.assert_close(reward(), torch.zeros(1))


class TestDelayedNotSteppingPenalty(unittest.TestCase):
  def setUp(self) -> None:
    self.command = torch.tensor([[0.2, 0.0, 0.0]])
    self.sensor = types.SimpleNamespace(
      data=types.SimpleNamespace(found=torch.tensor([[True, True]]))
    )
    self.env = types.SimpleNamespace(
      num_envs=1,
      device=torch.device("cpu"),
      step_dt=0.02,
      scene={"feet": self.sensor},
      command_manager=types.SimpleNamespace(get_command=lambda _: self.command),
    )
    self.term = not_stepping_penalty(types.SimpleNamespace(params={}), self.env)

  def _penalty(self) -> torch.Tensor:
    return self.term(
      self.env, sensor_name="feet", command_threshold=0.01,
      min_duration_s=1.0,
    )

  def test_penalty_starts_at_one_second_and_continues(self) -> None:
    for _ in range(49):
      torch.testing.assert_close(self._penalty(), torch.zeros(1))
    torch.testing.assert_close(self._penalty(), torch.ones(1))
    torch.testing.assert_close(self._penalty(), torch.ones(1))

  def test_foot_lift_resets_timer(self) -> None:
    for _ in range(49):
      self._penalty()
    self.sensor.data.found[0, 0] = False
    torch.testing.assert_close(self._penalty(), torch.zeros(1))
    self.sensor.data.found[0, 0] = True
    torch.testing.assert_close(self._penalty(), torch.zeros(1))

  def test_standing_command_and_episode_reset_clear_timer(self) -> None:
    for _ in range(49):
      self._penalty()
    self.command[:] = 0.0
    torch.testing.assert_close(self._penalty(), torch.zeros(1))
    self.command[0, 0] = 0.2
    torch.testing.assert_close(self._penalty(), torch.zeros(1))
    for _ in range(49):
      self._penalty()
    self.term.reset(torch.tensor([0]))
    torch.testing.assert_close(self._penalty(), torch.zeros(1))


class TestNotSteppingEachFootPenalty(unittest.TestCase):
  def setUp(self) -> None:
    self.contact = types.SimpleNamespace(
      data=types.SimpleNamespace(found=torch.tensor([[True, True]]))
    )
    self.command = torch.tensor([[0.2, 0.0, 0.0]])
    self.env = types.SimpleNamespace(
      num_envs=1,
      device=torch.device("cpu"),
      step_dt=0.02,
      scene={"feet": self.contact},
      command_manager=types.SimpleNamespace(get_command=lambda _: self.command),
    )
    self.term = not_stepping_each_foot_penalty(
      types.SimpleNamespace(params={}), self.env
    )

  def _penalty(self) -> torch.Tensor:
    return self.term(
      self.env,
      sensor_name="feet",
      max_time_without_lift_s=2.0,
    )

  def test_each_overdue_foot_is_charged_every_step_until_it_lifts(self) -> None:
    for _ in range(99):
      torch.testing.assert_close(self._penalty(), torch.zeros(1))
    for _ in range(50):
      torch.testing.assert_close(self._penalty(), torch.tensor([2.0]))
    self.contact.data.found[0, 0] = False
    torch.testing.assert_close(self._penalty(), torch.tensor([1.0]))
    self.contact.data.found[0, 0] = True
    torch.testing.assert_close(self._penalty(), torch.tensor([1.0]))

  def test_contact_loss_resets_only_that_foot(self) -> None:
    for _ in range(90):
      self._penalty()
    self.contact.data.found[0, 0] = False
    torch.testing.assert_close(self._penalty(), torch.zeros(1))
    self.contact.data.found[0, 0] = True
    for _ in range(8):
      self._penalty()
    torch.testing.assert_close(self._penalty(), torch.tensor([1.0]))

  def test_standing_command_and_episode_reset_clear_timers(self) -> None:
    for _ in range(99):
      self._penalty()
    self.command[:] = 0.0
    torch.testing.assert_close(self._penalty(), torch.zeros(1))
    self.command[0, 0] = 0.2
    torch.testing.assert_close(self._penalty(), torch.zeros(1))
    for _ in range(98):
      self._penalty()
    self.term.reset(torch.tensor([0]))
    torch.testing.assert_close(self._penalty(), torch.zeros(1))


class TestOverlongSwingPenalty(unittest.TestCase):
  def setUp(self) -> None:
    self.sensor = types.SimpleNamespace(
      data=types.SimpleNamespace(
        found=torch.tensor([[True, True]]),
        current_air_time=torch.tensor([[0.0, 0.0]]),
      )
    )
    self.asset = types.SimpleNamespace(
      data=types.SimpleNamespace(
        root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        gravity_vec_w=torch.tensor([[0.0, 0.0, -1.0]]),
        root_link_lin_vel_b=torch.zeros((1, 3)),
        root_link_ang_vel_b=torch.zeros((1, 3)),
      )
    )
    self.command = torch.zeros((1, 3))
    scene = _TestScene(feet=self.sensor, robot=self.asset)
    self.env = types.SimpleNamespace(
      scene=scene,
      command_manager=types.SimpleNamespace(get_command=lambda _: self.command),
      action_manager=types.SimpleNamespace(
        action=torch.tensor([[2.0, 0.0]]),
        prev_action=torch.zeros((1, 2)),
        prev_prev_action=torch.zeros((1, 2)),
      ),
    )

  def _penalty(self) -> torch.Tensor:
    return overlong_swing_penalty(
      self.env,
      sensor_name="feet",
      max_air_time=0.6,
      command_name="twist",
      command_threshold=0.01,
    )

  def _no_stepping_penalty(self) -> torch.Tensor:
    return no_stepping_penalty(
      self.env,
      sensor_name="feet",
      command_name="twist",
      command_threshold=0.01,
    )

  def test_standing_uses_air_time_threshold(self) -> None:
    self.sensor.data.found[:] = torch.tensor([[False, True]])
    self.sensor.data.current_air_time[:] = torch.tensor([[0.02, 0.0]])
    torch.testing.assert_close(self._penalty(), torch.zeros(1))

    self.sensor.data.current_air_time[:] = torch.tensor([[0.7, 0.0]])
    torch.testing.assert_close(self._penalty(), torch.ones(1))

  def test_standing_with_both_feet_down_is_not_penalized(self) -> None:
    torch.testing.assert_close(self._penalty(), torch.zeros(1))

  def test_moving_keeps_the_air_time_threshold(self) -> None:
    self.command[:] = torch.tensor([[0.2, 0.0, 0.0]])
    self.sensor.data.found[:] = torch.tensor([[False, True]])
    self.sensor.data.current_air_time[:] = torch.tensor([[0.2, 0.0]])
    torch.testing.assert_close(self._penalty(), torch.zeros(1))

    self.sensor.data.current_air_time[:] = torch.tensor([[0.7, 0.0]])
    torch.testing.assert_close(self._penalty(), torch.ones(1))

  def test_stable_standing_still_penalizes_liftoff(self) -> None:
    self.sensor.data.found[:] = torch.tensor([[False, True]])
    torch.testing.assert_close(self._no_stepping_penalty(), torch.ones(1))

  def test_push_velocity_allows_recovery_step(self) -> None:
    self.sensor.data.found[:] = torch.tensor([[False, True]])
    self.sensor.data.current_air_time[:] = torch.tensor([[0.7, 0.0]])
    self.asset.data.root_link_lin_vel_b[:, 0] = 0.1
    torch.testing.assert_close(self._no_stepping_penalty(), torch.zeros(1))
    torch.testing.assert_close(self._penalty(), torch.zeros(1))

  def test_static_tilt_allows_recovery_step(self) -> None:
    self.sensor.data.found[:] = torch.tensor([[False, True]])
    self.sensor.data.current_air_time[:] = torch.tensor([[0.7, 0.0]])
    self.asset.data.root_link_quat_w[:] = torch.tensor(
      [[0.98480775, 0.17364818, 0.0, 0.0]]
    )
    torch.testing.assert_close(self._no_stepping_penalty(), torch.zeros(1))
    torch.testing.assert_close(self._penalty(), torch.zeros(1))

def _angular_velocity_env(command_yaw: float, actual_yaw: float):
  asset = types.SimpleNamespace(
    data=types.SimpleNamespace(
      root_link_ang_vel_b=torch.tensor([[0.0, 0.0, actual_yaw]]),
    )
  )
  command = torch.tensor([[0.0, 0.0, command_yaw]])
  command_manager = types.SimpleNamespace(get_command=lambda _: command)
  return types.SimpleNamespace(scene={"robot": asset}, command_manager=command_manager)


class TestRelativeAngularVelocityPenalty(unittest.TestCase):
  def test_penalizes_rotation_at_zero_command(self) -> None:
    env = _angular_velocity_env(command_yaw=0.0, actual_yaw=-0.4)
    torch.testing.assert_close(
      relative_angular_velocity_error_penalty(
        env,
        "twist",
        command_threshold=0.05,
        max_relative_error=2.0,
      ),
      torch.tensor([4.0]),
    )

  def test_is_zero_when_command_is_tracked(self) -> None:
    env = _angular_velocity_env(command_yaw=0.2, actual_yaw=0.2)
    torch.testing.assert_close(
      relative_angular_velocity_error_penalty(
        env,
        "twist",
        command_threshold=0.05,
        max_relative_error=2.0,
      ),
      torch.zeros(1),
    )

  def test_uses_relative_error(self) -> None:
    env = _angular_velocity_env(command_yaw=0.2, actual_yaw=0.1)
    torch.testing.assert_close(
      relative_angular_velocity_error_penalty(
        env,
        "twist",
        command_threshold=0.05,
        max_relative_error=2.0,
      ),
      torch.tensor([0.25]),
    )


def _heading_env(
  root_quat_w: tuple[float, float, float, float],
  foot_quat_w: list[tuple[float, float, float, float]],
):
  asset = types.SimpleNamespace(
    data=types.SimpleNamespace(
      root_link_quat_w=torch.tensor([root_quat_w]),
      site_quat_w=torch.tensor([foot_quat_w]),
    )
  )
  return types.SimpleNamespace(scene={"robot": asset})


class TestFootBaseHeadingPenalty(unittest.TestCase):
  asset_cfg = types.SimpleNamespace(name="robot", site_ids=[0, 1])

  def test_aligned_feet_have_zero_error(self) -> None:
    env = _heading_env(
      root_quat_w=(1.0, 0.0, 0.0, 0.0),
      foot_quat_w=[(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)],
    )
    torch.testing.assert_close(
      foot_base_heading_error_penalty(env, self.asset_cfg),
      torch.zeros(1),
    )

  def test_ninety_degree_foot_yaw_has_unit_error(self) -> None:
    half_sqrt_two = 2.0**-0.5
    env = _heading_env(
      root_quat_w=(1.0, 0.0, 0.0, 0.0),
      foot_quat_w=[
        (half_sqrt_two, 0.0, 0.0, half_sqrt_two),
        (half_sqrt_two, 0.0, 0.0, -half_sqrt_two),
      ],
    )
    torch.testing.assert_close(
      foot_base_heading_error_penalty(env, self.asset_cfg),
      torch.ones(1),
    )

  def test_averages_error_across_feet(self) -> None:
    half_sqrt_two = 2.0**-0.5
    env = _heading_env(
      root_quat_w=(1.0, 0.0, 0.0, 0.0),
      foot_quat_w=[
        (1.0, 0.0, 0.0, 0.0),
        (half_sqrt_two, 0.0, 0.0, half_sqrt_two),
      ],
    )
    torch.testing.assert_close(
      foot_base_heading_error_penalty(env, self.asset_cfg),
      torch.tensor([0.5]),
    )

  def test_common_world_yaw_does_not_create_relative_error(self) -> None:
    half_sqrt_two = 2.0**-0.5
    yaw_90 = (half_sqrt_two, 0.0, 0.0, half_sqrt_two)
    env = _heading_env(
      root_quat_w=yaw_90,
      foot_quat_w=[yaw_90, yaw_90],
    )
    torch.testing.assert_close(
      foot_base_heading_error_penalty(env, self.asset_cfg),
      torch.zeros(1),
    )


class TestSwingProgressReward(unittest.TestCase):
  def setUp(self) -> None:
    self.contact = types.SimpleNamespace(
      data=types.SimpleNamespace(found=torch.tensor([[True, False]]))
    )
    self.height = types.SimpleNamespace(
      data=types.SimpleNamespace(heights=torch.tensor([[0.0, 0.02]]))
    )
    self.command = torch.tensor([[0.2, 0.0, 0.0]])
    self.env = types.SimpleNamespace(
      scene={"feet": self.contact, "height": self.height},
      command_manager=types.SimpleNamespace(get_command=lambda _: self.command),
    )
    self.term = swing_progress_reward(
      types.SimpleNamespace(params={"sensor_name": "feet"}), self.env
    )

  def reward(self) -> torch.Tensor:
    return self.term(
      self.env, sensor_name="feet", height_sensor_name="height",
      min_height=0.005, target_height=0.02, height_std=0.01,
      pair_bonus_scale=5.0,
    )

  def land_both(self) -> torch.Tensor:
    self.contact.data.found[:] = torch.tensor([[True, True]])
    self.height.data.heights[:] = 0.0
    return self.reward()

  def test_holding_one_foot_or_repeating_it_does_not_keep_paying(self) -> None:
    torch.testing.assert_close(self.reward(), torch.ones(1))
    torch.testing.assert_close(self.reward(), torch.zeros(1))
    self.height.data.heights[0, 1] = 0.01
    torch.testing.assert_close(self.reward(), torch.zeros(1))
    torch.testing.assert_close(self.land_both(), torch.zeros(1))
    self.contact.data.found[:] = torch.tensor([[True, False]])
    self.height.data.heights[0, 1] = 0.02
    torch.testing.assert_close(self.reward(), torch.zeros(1))
    torch.testing.assert_close(self.land_both(), torch.zeros(1))

  def test_equal_alternating_peaks_pay_more_than_unequal(self) -> None:
    self.reward()
    self.land_both()
    self.contact.data.found[:] = torch.tensor([[False, True]])
    self.height.data.heights[0, 0] = 0.02
    torch.testing.assert_close(self.reward(), torch.ones(1))
    torch.testing.assert_close(self.land_both(), torch.tensor([5.0]))
    torch.testing.assert_close(self.reward(), torch.zeros(1))

    self.term.reset()
    self.contact.data.found[:] = torch.tensor([[True, False]])
    self.height.data.heights[0, 1] = 0.01
    torch.testing.assert_close(self.reward(), torch.tensor([1.0 / 3.0]))
    self.land_both()
    self.contact.data.found[:] = torch.tensor([[False, True]])
    self.height.data.heights[0, 0] = 0.02
    torch.testing.assert_close(self.reward(), torch.ones(1))
    torch.testing.assert_close(
      self.land_both(), torch.tensor([5.0 / 3.0 * math.exp(-0.5)])
    )

  def test_two_foot_hop_and_standing_receive_zero(self) -> None:
    self.contact.data.found[:] = torch.tensor([[False, False]])
    self.height.data.heights[:] = 0.02
    torch.testing.assert_close(self.reward(), torch.zeros(1))
    self.contact.data.found[:] = torch.tensor([[True, False]])
    torch.testing.assert_close(self.reward(), torch.zeros(1))
    self.land_both()
    self.contact.data.found[:] = torch.tensor([[False, True]])
    self.height.data.heights[0, 0] = 0.02
    self.command[:] = 0.0
    torch.testing.assert_close(self.reward(), torch.zeros(1))
    self.term.reset()
    self.command[0, 0] = 0.2
    torch.testing.assert_close(self.reward(), torch.ones(1))


class TestGaitPhaseSwingClearanceReward(unittest.TestCase):
  def setUp(self) -> None:
    self.phase = torch.tensor([0.25])
    self.contact = types.SimpleNamespace(
      data=types.SimpleNamespace(found=torch.tensor([[True, True]]))
    )
    self.height = types.SimpleNamespace(
      data=types.SimpleNamespace(heights=torch.zeros((1, 2)))
    )
    self.command = torch.tensor([[0.2, 0.0, 0.0]])
    self.env = types.SimpleNamespace(
      _kid_rl_gait=types.SimpleNamespace(phase=lambda _: self.phase),
      scene={"feet": self.contact, "height": self.height},
      command_manager=types.SimpleNamespace(get_command=lambda _: self.command),
    )

  def _reward(self) -> torch.Tensor:
    return gait_phase_swing_clearance_reward(
      self.env, sensor_name="feet", height_sensor_name="height"
    )

  def test_selected_foot_receives_partial_progress_before_liftoff(self) -> None:
    # At phase 0.25, the right foot is due to swing. Partial height pays even
    # before its last contact disappears; the left foot receives no credit.
    self.height.data.heights[:] = torch.tensor([[0.03, 0.0175]])
    torch.testing.assert_close(self._reward(), torch.tensor([0.5]))
    self.height.data.heights[0, 1] = 0.03
    torch.testing.assert_close(self._reward(), torch.ones(1))

  def test_requires_opposite_foot_contact_and_turns_off_in_double_support(self) -> None:
    self.height.data.heights[0, 1] = 0.03
    self.contact.data.found[0, 0] = False
    torch.testing.assert_close(self._reward(), torch.zeros(1))
    self.contact.data.found[0, 0] = True
    self.phase[:] = 0.0
    torch.testing.assert_close(self._reward(), torch.zeros(1))

  def test_other_half_cycle_and_standing_command(self) -> None:
    self.phase[:] = 0.75
    self.height.data.heights[0, 0] = 0.03
    torch.testing.assert_close(self._reward(), torch.ones(1))
    self.command[:] = 0.0
    torch.testing.assert_close(self._reward(), torch.zeros(1))


class TestOneShotSwingHeightReward(unittest.TestCase):
  def setUp(self) -> None:
    self.contact = types.SimpleNamespace(
      data=types.SimpleNamespace(
        found=torch.tensor([[1, 0]]),
        current_air_time=torch.zeros((1, 2)),
      )
    )
    self.height = types.SimpleNamespace(
      num_frames=2,
      data=types.SimpleNamespace(heights=torch.tensor([[0.0, 0.0]])),
    )
    self.command = torch.tensor([[0.2, 0.0, 0.0]])
    self.env = types.SimpleNamespace(
      num_envs=1,
      device=torch.device("cpu"),
      scene={"feet": self.contact, "height": self.height},
      command_manager=types.SimpleNamespace(get_command=lambda _: self.command),
    )
    cfg = types.SimpleNamespace(params={"height_sensor_name": "height"})
    self.term = feet_crossing_reward(cfg, self.env)

  def _reward(self) -> torch.Tensor:
    return self.term(
      self.env,
      sensor_name="feet",
      height_sensor_name="height",
      min_swing_height=0.02,
    )

  def test_pays_once_while_foot_remains_above_threshold(self) -> None:
    self.height.data.heights[0, 1] = 0.02
    torch.testing.assert_close(self._reward(), torch.ones(1))

    self.height.data.heights[0, 1] = 0.08
    torch.testing.assert_close(self._reward(), torch.zeros(1))

  def test_does_not_rearm_after_dropping_below_threshold_in_air(self) -> None:
    self.height.data.heights[0, 1] = 0.02
    torch.testing.assert_close(self._reward(), torch.ones(1))

    self.height.data.heights[0, 1] = 0.01
    torch.testing.assert_close(self._reward(), torch.zeros(1))

    self.height.data.heights[0, 1] = 0.02
    torch.testing.assert_close(self._reward(), torch.zeros(1))

  def test_landing_rearms_the_foot_for_the_next_swing(self) -> None:
    self.height.data.heights[0, 1] = 0.02
    torch.testing.assert_close(self._reward(), torch.ones(1))

    self.contact.data.found[:] = torch.tensor([[1, 1]])
    torch.testing.assert_close(self._reward(), torch.zeros(1))

    self.contact.data.found[:] = torch.tensor([[1, 0]])
    torch.testing.assert_close(self._reward(), torch.ones(1))

  def test_staggered_landing_after_hop_does_not_get_crossing_bonus(self) -> None:
    self.contact.data.found[:] = torch.tensor([[0, 0]])
    self.contact.data.current_air_time[:] = 0.08
    self.height.data.heights[:] = 0.02
    torch.testing.assert_close(self._reward(), torch.zeros(1))
    self.contact.data.found[:] = torch.tensor([[1, 0]])
    torch.testing.assert_close(self._reward(), torch.zeros(1))
    self.contact.data.found[:] = torch.tensor([[1, 1]])
    torch.testing.assert_close(self._reward(), torch.zeros(1))
    self.contact.data.current_air_time[:] = 0.0
    self.contact.data.found[:] = torch.tensor([[1, 0]])
    torch.testing.assert_close(self._reward(), torch.ones(1))

  def test_does_not_pay_below_threshold(self) -> None:
    self.height.data.heights[0, 1] = 0.019
    torch.testing.assert_close(self._reward(), torch.zeros(1))

  def test_opposite_foot_gets_bonus_then_pair_memory_resets(self) -> None:
    self.height.data.heights[0, 1] = 0.02
    torch.testing.assert_close(self._reward(), torch.ones(1))
    self.contact.data.found[:] = torch.tensor([[1, 1]])
    torch.testing.assert_close(self._reward(), torch.zeros(1))

    self.contact.data.found[:] = torch.tensor([[0, 1]])
    self.height.data.heights[:] = torch.tensor([[0.019, 0.0]])
    torch.testing.assert_close(self._reward(), torch.zeros(1))
    self.height.data.heights[0, 0] = 0.02
    torch.testing.assert_close(self._reward(), torch.tensor([2.0]))
    torch.testing.assert_close(self._reward(), torch.zeros(1))

    self.contact.data.found[:] = torch.tensor([[1, 1]])
    torch.testing.assert_close(self._reward(), torch.zeros(1))
    self.contact.data.found[:] = torch.tensor([[1, 0]])
    self.height.data.heights[:] = torch.tensor([[0.0, 0.02]])
    torch.testing.assert_close(self._reward(), torch.ones(1))

  def test_same_foot_repeat_does_not_get_bonus(self) -> None:
    self.height.data.heights[0, 1] = 0.02
    torch.testing.assert_close(self._reward(), torch.ones(1))
    self.contact.data.found[:] = torch.tensor([[1, 1]])
    self._reward()
    self.contact.data.found[:] = torch.tensor([[1, 0]])
    torch.testing.assert_close(self._reward(), torch.ones(1))
    self.contact.data.found[:] = torch.tensor([[1, 1]])
    self._reward()
    self.contact.data.found[:] = torch.tensor([[0, 1]])
    self.height.data.heights[:] = torch.tensor([[0.02, 0.0]])
    torch.testing.assert_close(self._reward(), torch.tensor([2.0]))

  def test_reset_clears_alternation_memory(self) -> None:
    self.height.data.heights[0, 1] = 0.02
    torch.testing.assert_close(self._reward(), torch.ones(1))
    self.term.reset(torch.tensor([0]))
    self.contact.data.found[:] = torch.tensor([[0, 1]])
    self.height.data.heights[:] = torch.tensor([[0.02, 0.0]])
    torch.testing.assert_close(self._reward(), torch.ones(1))

  def test_standing_command_does_not_pay(self) -> None:
    self.command[:] = 0.0
    self.height.data.heights[0, 1] = 0.02
    torch.testing.assert_close(self._reward(), torch.zeros(1))


class _StepSensor:
  def __init__(self) -> None:
    self.first_air = torch.tensor([[True, False]])
    self.first_contact = torch.tensor([[False, False]])
    self.data = types.SimpleNamespace(
      last_contact_time=torch.tensor([[0.5, 0.5]]),
      last_air_time=torch.tensor([[0.5, 0.5]]),
    )

  def compute_first_air(self, dt: float) -> torch.Tensor:
    del dt
    return self.first_air

  def compute_first_contact(self, dt: float) -> torch.Tensor:
    del dt
    return self.first_contact


class TestPlanarStepReward(unittest.TestCase):
  def _score_matching_step(
    self,
    command_xy: tuple[float, float],
    world_displacement_xy: tuple[float, float],
    root_quat_w: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
  ) -> torch.Tensor:
    asset_cfg = types.SimpleNamespace(
      name="robot", site_names=("left", "right"), site_ids=[0, 1]
    )
    asset = types.SimpleNamespace(
      find_sites=lambda _: ([0, 1], ["left", "right"]),
      data=types.SimpleNamespace(
        site_pos_w=torch.zeros((1, 2, 3)),
        root_link_quat_w=torch.tensor([root_quat_w]),
      ),
    )
    sensor = _StepSensor()
    height_sensor = types.SimpleNamespace(
      data=types.SimpleNamespace(heights=torch.tensor([[0.03, 0.0]]))
    )
    command = torch.tensor([[command_xy[0], command_xy[1], 0.0]])
    env = types.SimpleNamespace(
      num_envs=1,
      device=torch.device("cpu"),
      step_dt=0.02,
      scene={"robot": asset, "feet": sensor, "height": height_sensor},
      command_manager=types.SimpleNamespace(get_command=lambda _: command),
    )
    term = forward_step_reward(types.SimpleNamespace(params={"asset_cfg": asset_cfg}), env)

    # Liftoff snapshots the foot position and the current planar body frame.
    term(env, "feet", "height", asset_cfg)

    asset.data.site_pos_w[0, 0, :2] = torch.tensor(world_displacement_xy)
    sensor.first_air[:] = False
    sensor.first_contact[0, 0] = True
    height_sensor.data.heights[:] = 0.0
    return term(env, "feet", "height", asset_cfg)

  def test_matches_forward_backward_and_lateral_commands(self) -> None:
    for command_xy, displacement_xy in (
      ((0.2, 0.0), (0.2, 0.0)),
      ((-0.2, 0.0), (-0.2, 0.0)),
      ((0.0, 0.15), (0.0, 0.15)),
      ((0.0, -0.15), (0.0, -0.15)),
      ((0.1, -0.1), (0.1, -0.1)),
    ):
      with self.subTest(command_xy=command_xy):
        torch.testing.assert_close(
          self._score_matching_step(command_xy, displacement_xy),
          torch.ones(1),
        )

  def test_standing_command_does_not_pay(self) -> None:
    torch.testing.assert_close(
      self._score_matching_step((0.0, 0.0), (0.1, 0.0)),
      torch.zeros(1),
    )

  def test_uses_body_frame_for_lateral_command(self) -> None:
    half_sqrt_two = 2.0**-0.5
    # At +90 deg yaw, body +y points toward world -x.
    torch.testing.assert_close(
      self._score_matching_step(
        command_xy=(0.0, 0.15),
        world_displacement_xy=(-0.15, 0.0),
        root_quat_w=(half_sqrt_two, 0.0, 0.0, half_sqrt_two),
      ),
      torch.ones(1),
    )

  def test_rejects_step_in_opposite_lateral_direction(self) -> None:
    score = self._score_matching_step(
      command_xy=(0.0, 0.15),
      world_displacement_xy=(0.0, -0.15),
    )
    self.assertLess(score.item(), 1.0e-6)


if __name__ == "__main__":
  unittest.main()
