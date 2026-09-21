"""Focused tests for kid_RL velocity-tracking reward terms."""

import types
import unittest

import torch

from mjlab_kid_rl.tasks.mdp import (
  feet_air_time_once_reward,
  feet_crossing_reward,
  feet_distance_penalty,
  flatness_weighted_foot_slip_penalty,
  foot_base_heading_error_penalty,
  forward_step_reward,
  overlong_swing_penalty,
  relative_angular_velocity_error_penalty,
  selected_action_excess_l2,
)


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


class TestOneShotAirTimeReward(unittest.TestCase):
  def setUp(self) -> None:
    self.sensor = types.SimpleNamespace()
    self.sensor.first_air = torch.tensor([[False, False]])
    self.sensor.first_contact = torch.tensor([[False, False]])
    self.sensor.data = types.SimpleNamespace(
      current_air_time=torch.tensor([[0.0, 0.0]]),
      last_air_time=torch.tensor([[0.0, 0.0]]),
    )
    self.sensor.compute_first_air = lambda dt: self.sensor.first_air
    self.sensor.compute_first_contact = lambda dt: self.sensor.first_contact
    self.command = torch.tensor([[0.2, 0.0, 0.0]])
    self.env = types.SimpleNamespace(
      step_dt=0.02,
      scene={"feet": self.sensor},
      command_manager=types.SimpleNamespace(get_command=lambda _: self.command),
    )
    cfg = types.SimpleNamespace(params={"sensor_name": "feet"})
    self.term = feet_air_time_once_reward(cfg, self.env)

  def _reward(self) -> torch.Tensor:
    return self.term(
      self.env,
      sensor_name="feet",
      threshold_min=0.3,
      threshold_max=1.0,
      command_threshold=0.01,
    )

  def test_pays_only_once_at_landing(self) -> None:
    self.sensor.first_air[:] = torch.tensor([[True, False]])
    torch.testing.assert_close(self._reward(), torch.zeros(1))

    self.sensor.first_air[:] = False
    self.sensor.data.current_air_time[:] = torch.tensor([[0.5, 0.0]])
    torch.testing.assert_close(self._reward(), torch.zeros(1))

    self.sensor.data.last_air_time[:] = torch.tensor([[0.5, 0.0]])
    self.sensor.first_contact[:] = torch.tensor([[True, False]])
    torch.testing.assert_close(self._reward(), torch.ones(1))

    self.sensor.first_contact[:] = False
    torch.testing.assert_close(self._reward(), torch.zeros(1))

  def test_new_air_phase_can_pay_again(self) -> None:
    self.sensor.first_air[:] = torch.tensor([[False, True]])
    torch.testing.assert_close(self._reward(), torch.zeros(1))

    self.sensor.first_air[:] = False
    self.sensor.data.last_air_time[:] = torch.tensor([[0.0, 0.4]])
    self.sensor.first_contact[:] = torch.tensor([[False, True]])
    torch.testing.assert_close(self._reward(), torch.ones(1))

  def test_too_long_swing_does_not_pay(self) -> None:
    self.sensor.first_air[:] = torch.tensor([[True, False]])
    torch.testing.assert_close(self._reward(), torch.zeros(1))

    self.sensor.first_air[:] = False
    self.sensor.data.last_air_time[:] = torch.tensor([[1.2, 0.0]])
    self.sensor.first_contact[:] = torch.tensor([[True, False]])
    torch.testing.assert_close(self._reward(), torch.zeros(1))

  def test_two_footed_hop_does_not_pay(self) -> None:
    self.sensor.first_air[:] = True
    torch.testing.assert_close(self._reward(), torch.zeros(1))

    self.sensor.first_air[:] = False
    self.sensor.data.last_air_time[:] = torch.tensor([[0.5, 0.5]])
    self.sensor.first_contact[:] = True
    torch.testing.assert_close(self._reward(), torch.zeros(1))

  def test_standing_command_does_not_pay(self) -> None:
    self.command[:] = 0.0
    self.sensor.first_air[:] = torch.tensor([[True, False]])
    torch.testing.assert_close(self._reward(), torch.zeros(1))

    self.sensor.first_air[:] = False
    self.sensor.data.last_air_time[:] = torch.tensor([[0.5, 0.0]])
    self.sensor.first_contact[:] = torch.tensor([[True, False]])
    torch.testing.assert_close(self._reward(), torch.zeros(1))


class TestOverlongSwingPenalty(unittest.TestCase):
  def setUp(self) -> None:
    self.sensor = types.SimpleNamespace(
      data=types.SimpleNamespace(
        found=torch.tensor([[True, True]]),
        current_air_time=torch.tensor([[0.0, 0.0]]),
      )
    )
    self.command = torch.zeros((1, 3))
    self.env = types.SimpleNamespace(
      scene={"feet": self.sensor},
      command_manager=types.SimpleNamespace(get_command=lambda _: self.command),
    )

  def _penalty(self) -> torch.Tensor:
    return overlong_swing_penalty(
      self.env,
      sensor_name="feet",
      max_air_time=0.6,
      command_name="twist",
      command_threshold=0.01,
    )

  def test_standing_penalizes_liftoff_immediately(self) -> None:
    self.sensor.data.found[:] = torch.tensor([[False, True]])
    self.sensor.data.current_air_time[:] = torch.tensor([[0.02, 0.0]])
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


class TestOneShotSwingHeightReward(unittest.TestCase):
  def setUp(self) -> None:
    self.contact = types.SimpleNamespace(
      data=types.SimpleNamespace(found=torch.tensor([[1, 0]]))
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
      min_swing_height=0.03,
    )

  def test_pays_only_once_while_foot_remains_above_threshold(self) -> None:
    self.height.data.heights[0, 1] = 0.03
    torch.testing.assert_close(self._reward(), torch.ones(1))

    self.height.data.heights[0, 1] = 0.08
    torch.testing.assert_close(self._reward(), torch.zeros(1))

  def test_dropping_below_threshold_in_air_does_not_rearm(self) -> None:
    self.height.data.heights[0, 1] = 0.03
    torch.testing.assert_close(self._reward(), torch.ones(1))

    self.height.data.heights[0, 1] = 0.01
    torch.testing.assert_close(self._reward(), torch.zeros(1))

    self.height.data.heights[0, 1] = 0.03
    torch.testing.assert_close(self._reward(), torch.zeros(1))

  def test_landing_rearms_the_foot_for_the_next_swing(self) -> None:
    self.height.data.heights[0, 1] = 0.03
    torch.testing.assert_close(self._reward(), torch.ones(1))

    self.contact.data.found[:] = torch.tensor([[1, 1]])
    torch.testing.assert_close(self._reward(), torch.zeros(1))

    self.contact.data.found[:] = torch.tensor([[1, 0]])
    torch.testing.assert_close(self._reward(), torch.ones(1))

  def test_does_not_pay_below_threshold(self) -> None:
    self.height.data.heights[0, 1] = 0.029
    torch.testing.assert_close(self._reward(), torch.zeros(1))

  def test_standing_command_does_not_pay(self) -> None:
    self.command[:] = 0.0
    self.height.data.heights[0, 1] = 0.03
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
