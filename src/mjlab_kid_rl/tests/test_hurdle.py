"""Unit tests for the mocap-bar hurdle task in mjlab_kid_rl.tasks.hurdle (CPU, stubs)."""

import copy
import math
import types
import unittest
from unittest import mock

import torch

from mjlab_kid_rl.tasks import hurdle, warm_start
from mjlab_kid_rl.tasks.symmetry import _VECTOR_TERMS, _apply, _build_spec

BASE_HEIGHT = 0.45


def _env(num_envs=1, is_hurdle=(True,), bar_x=1.0, height=0.05, base_x=0.0, feet_x=None, found=None, command=(0.3, 0.0, 0.0)):
  """Robots at (base_x, 0, BASE_HEIGHT) facing +x; bar square to +x at bar_x."""
  root_pos = torch.tensor([[base_x, 0.0, BASE_HEIGHT]]).repeat(num_envs, 1)
  feet = torch.zeros((num_envs, 2, 3))
  feet[..., 0] = base_x if feet_x is None else torch.tensor(feet_x, dtype=torch.float32)
  asset = types.SimpleNamespace(
    data=types.SimpleNamespace(
      root_link_pos_w=root_pos,
      root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(num_envs, 1),
      root_link_lin_vel_b=torch.zeros((num_envs, 3)),
      heading_w=torch.zeros(num_envs),
      site_pos_w=feet,
    )
  )
  sensor = types.SimpleNamespace(
    data=types.SimpleNamespace(found=torch.zeros((num_envs, 1)) if found is None else found)
  )
  command_term = types.SimpleNamespace(
    is_hurdle=torch.tensor(is_hurdle),
    level=torch.zeros(num_envs, dtype=torch.long),
    height=torch.full((num_envs,), height),
    bar_pos_w=torch.tensor([[bar_x, 0.0, height / 2]]).repeat(num_envs, 1),
    bar_yaw=torch.zeros(num_envs),
  )

  class _Scene(types.SimpleNamespace):
    def __getitem__(self, key):
      return {"robot": asset, "hurdle_contact": sensor}[key]

  return types.SimpleNamespace(
    num_envs=num_envs,
    step_dt=0.02,
    device=torch.device("cpu"),
    termination_manager=types.SimpleNamespace(terminated=torch.zeros(num_envs, dtype=torch.bool)),
    scene=_Scene(env_origins=torch.zeros((num_envs, 3))),
    command_manager=types.SimpleNamespace(
      get_command=lambda _: torch.tensor(command).repeat(num_envs, 1),
      get_term=lambda _: command_term,
    ),
    asset=asset,
    command=command_term,
  )


class TestCameraGeometry(unittest.TestCase):
  def test_level_camera_looks_along_base_x(self) -> None:
    lens, forward, right, down = hurdle.camera_axes(torch.zeros(1))
    torch.testing.assert_close(forward[0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(right[0], torch.tensor([0.0, -1.0, 0.0]))
    torch.testing.assert_close(down[0], torch.tensor([0.0, 0.0, -1.0]))
    expected = torch.tensor(hurdle.CAMERA_TILT_PIVOT) + torch.tensor(hurdle.CAMERA_LENS_OFFSET)
    torch.testing.assert_close(lens[0], expected)

  def test_positive_tilt_looks_down(self) -> None:
    _, forward, _, down = hurdle.camera_axes(torch.tensor([math.radians(40.0)]))
    self.assertLess(forward[0, 2].item(), -0.6)
    self.assertGreater(forward[0, 0].item(), 0.7)
    self.assertLess(down[0, 0].item(), 0.0)

  def test_point_on_the_optical_axis_lands_at_the_principal_point(self) -> None:
    tilt = torch.tensor([math.radians(65.0)])
    lens, forward, _, _ = hurdle.camera_axes(tilt)
    point = (lens + 0.7 * forward).unsqueeze(1)
    u, v, depth = hurdle.project(point, tilt)
    torch.testing.assert_close(u, torch.tensor([[hurdle.CAMERA_CENTER[0]]]))
    torch.testing.assert_close(v, torch.tensor([[hurdle.CAMERA_CENTER[1]]]))
    torch.testing.assert_close(depth, torch.tensor([[0.7]]))

  def test_point_behind_or_too_near_is_not_in_the_image(self) -> None:
    u = torch.tensor([[320.0, 320.0, 320.0, -1.0]])
    v = torch.tensor([[240.0, 240.0, 240.0, 240.0]])
    depth = torch.tensor([[1.0, -1.0, 0.05, 1.0]])
    self.assertEqual(hurdle.in_image(u, v, depth).tolist(), [[True, False, False, False]])

  def test_perpendicular_foot(self) -> None:
    points = torch.tensor([[[1.0, -0.5, 0.0], [1.0, 0.5, 0.0]]])
    torch.testing.assert_close(hurdle.perpendicular_foot(points), torch.tensor([[1.0, 0.0]]))
    points = torch.tensor([[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
    torch.testing.assert_close(hurdle.perpendicular_foot(points), torch.tensor([[1.0, 1.0]]))

  def test_bar_top_points_span_the_bar_at_its_top(self) -> None:
    points = hurdle.bar_top_points(
      torch.tensor([[1.0, 0.0, 0.025]]), torch.zeros(1), torch.tensor([0.05])
    )
    self.assertEqual(points.shape, (1, hurdle.CAMERA_SAMPLES, 3))
    torch.testing.assert_close(points[0, :, 2], torch.full((hurdle.CAMERA_SAMPLES,), 0.05))
    torch.testing.assert_close(points[0, -1, 1] - points[0, 0, 1], torch.tensor(hurdle.HURDLE_WIDTH))

  def test_bar_frame_is_negative_behind_the_bar(self) -> None:
    u, v = hurdle.bar_frame(
      torch.tensor([[[0.0, 0.2], [2.0, 0.0]]]), torch.tensor([[1.0, 0.0, 0.0]]), torch.zeros(1)
    )
    torch.testing.assert_close(u, torch.tensor([[-1.0, 1.0]]))
    torch.testing.assert_close(v, torch.tensor([[0.2, 0.0]]))


@mock.patch.multiple(
  hurdle,
  CAMERA_DROPOUT=0.0,
  CAMERA_PERIOD_S=1e-6,
  CAMERA_NOISE_XY=(0.0, 0.0),
  CAMERA_NOISE_Z=0.0,
  CAMERA_OUTLIER_PROB=0.0,
)
class TestHurdleVision(unittest.TestCase):
  def _vision(self, env):
    vision = hurdle.hurdle_vision(types.SimpleNamespace(), env)
    vision.lag[:] = 1
    return vision

  def test_bar_ahead_reports_its_pose_after_the_lag(self) -> None:
    env = _env(bar_x=1.0)
    vision = self._vision(env)
    first = vision(env).view(hurdle.VISION_FRAMES, hurdle.VISION_FRAME_DIM)
    torch.testing.assert_close(first[-1], torch.zeros(4))  # One step of lag.
    frames = vision(env).view(hurdle.VISION_FRAMES, hurdle.VISION_FRAME_DIM)
    torch.testing.assert_close(
      frames[-1], torch.tensor([1.0, 1.0, 0.0, 0.05 - BASE_HEIGHT]), atol=1e-4, rtol=0
    )
    torch.testing.assert_close(frames[:-1], torch.zeros(hurdle.VISION_FRAMES - 1, 4))

  def test_outlier_replaces_x_y_only(self) -> None:
    env = _env(bar_x=1.0)
    vision = self._vision(env)
    with mock.patch.object(hurdle, "CAMERA_OUTLIER_PROB", 1.0):
      report, detected, _ = vision.measure(env, hurdle.SceneEntityCfg("robot"))
    self.assertTrue(detected.all())
    x, y = report[0, 1].item(), report[0, 2].item()
    self.assertTrue(hurdle.CAMERA_OUTLIER_X[0] <= x <= hurdle.CAMERA_OUTLIER_X[1])
    self.assertTrue(hurdle.CAMERA_OUTLIER_Y[0] <= y <= hurdle.CAMERA_OUTLIER_Y[1])
    self.assertNotEqual((x, y), (1.0, 0.0))
    torch.testing.assert_close(
      report[0, [0, 3]], torch.tensor([1.0, 0.05 - BASE_HEIGHT]), atol=1e-4, rtol=0
    )

  def test_camera_clock_gives_three_frames_in_five_steps(self) -> None:
    env = _env(bar_x=1.0)
    with mock.patch.object(hurdle, "CAMERA_PERIOD_S", 1.0 / 30.0):
      vision = self._vision(env)
      seen = []
      for _ in range(60):
        vision(env)
        seen.append(float(vision.history[0, 0, 0]))
    seen = torch.tensor(seen)
    for start in range(0, 60, 5):
      self.assertEqual(int(seen[start:start + 5].sum()), 3)
    misses = (seen == 0).int()
    self.assertEqual(int((misses[1:] * misses[:-1]).sum()), 0)  # Never two misses in a row.

  def test_frames_are_stride_steps_apart_oldest_first(self) -> None:
    env = _env(bar_x=1.0)
    vision = self._vision(env)
    for _ in range(1 + hurdle.VISION_STRIDE + 1):
      out = vision(env).view(hurdle.VISION_FRAMES, hurdle.VISION_FRAME_DIM)
    self.assertEqual(out[:, 0].tolist(), [0.0] * (hurdle.VISION_FRAMES - 2) + [1.0, 1.0])

  def test_flat_env_and_far_bar_report_nothing(self) -> None:
    for env in (_env(is_hurdle=(False,), bar_x=1.0), _env(bar_x=10.0)):
      vision = self._vision(env)
      for _ in range(3):
        out = vision(env)
      torch.testing.assert_close(out, torch.zeros(1, hurdle.VISION_FRAMES * hurdle.VISION_FRAME_DIM))

  def test_bar_under_the_camera_is_not_seen(self) -> None:
    env = _env(bar_x=0.05)
    vision = self._vision(env)
    for _ in range(3):
      out = vision(env)
    self.assertEqual(out[0, -4].item(), 0.0)

  def test_bar_inside_the_near_cutoff_is_lost(self) -> None:
    for bar_x, seen in ((0.3, True), (0.22, False)):
      env = _env(bar_x=bar_x)
      vision = self._vision(env)
      vision.tilt[:] = math.radians(hurdle.CAMERA_TILT_HURDLE_DEG)
      vision.near_cutoff[:] = 0.25
      _, detected, _ = vision.measure(env, hurdle.SceneEntityCfg("robot"))
      self.assertEqual(detected[0].item(), seen, bar_x)

  def test_reset_draws_the_near_cutoff_in_range(self) -> None:
    env = _env(num_envs=64, is_hurdle=(True,) * 64)
    vision = self._vision(env)
    vision.reset(torch.arange(64))
    low, high = hurdle.CAMERA_NEAR_CUTOFF_CLAMP
    self.assertTrue(((vision.near_cutoff >= low) & (vision.near_cutoff <= high)).all())

  def test_near_bar_switches_the_tilt_for_good(self) -> None:
    env = _env(bar_x=0.4)
    vision = self._vision(env)
    for _ in range(20):
      vision(env)
    self.assertTrue(vision.tilt_fixed[0].item())
    self.assertAlmostEqual(math.degrees(vision.tilt[0].item()), hurdle.CAMERA_TILT_HURDLE_DEG, places=4)
    env.command.bar_pos_w[:, 0] = 1.0
    vision(env)
    self.assertAlmostEqual(math.degrees(vision.tilt[0].item()), hurdle.CAMERA_TILT_HURDLE_DEG, places=4)

  def test_far_bar_keeps_the_running_tilt(self) -> None:
    env = _env(bar_x=1.0)
    vision = self._vision(env)
    for _ in range(20):
      vision(env)
    self.assertFalse(vision.tilt_fixed[0].item())
    self.assertAlmostEqual(math.degrees(vision.tilt[0].item()), hurdle.CAMERA_TILT_RUNNING_DEG, places=4)

  def test_reset_clears_the_stack_and_the_tilt(self) -> None:
    env = _env(bar_x=0.4)
    vision = self._vision(env)
    for _ in range(20):
      vision(env)
    vision.reset(torch.tensor([0]))
    self.assertFalse(vision.tilt_fixed[0].item())
    torch.testing.assert_close(vision.history, torch.zeros_like(vision.history))
    self.assertIn(vision.lag[0].item(), range(hurdle.CAMERA_LAG_STEPS[0], hurdle.CAMERA_LAG_STEPS[1] + 1))


class TestHurdleTruth(unittest.TestCase):
  def test_truth_is_the_bar_in_base_frame_and_zero_on_flat(self) -> None:
    env = _env(num_envs=2, is_hurdle=(True, False), bar_x=1.0)
    truth = hurdle.hurdle_truth(env)
    torch.testing.assert_close(
      truth[0], torch.tensor([1.0, 0.0, 0.025 - BASE_HEIGHT, 0.0, 1.0, 0.05, 1.0])
    )
    torch.testing.assert_close(truth[1], torch.zeros(7))


def _site_cfg():
  return types.SimpleNamespace(name="robot", site_ids=[0, 1])


class TestHurdleRewards(unittest.TestCase):
  def test_contact_penalty_is_one_while_touching(self) -> None:
    env = _env(num_envs=2, is_hurdle=(True, True), found=torch.tensor([[1.0], [0.0]]))
    self.assertEqual(hurdle.hurdle_contact_penalty(env, "hurdle_contact").tolist(), [1.0, 0.0])

  def test_pass_pays_once_when_base_and_feet_are_past(self) -> None:
    env = _env(bar_x=1.0, base_x=1.3, feet_x=[1.25, 1.3])
    reward = hurdle.hurdle_pass_reward(types.SimpleNamespace(), env)
    first = reward(env, "hurdle_contact", _site_cfg())
    second = reward(env, "hurdle_contact", _site_cfg())
    self.assertEqual((first.item(), second.item()), (1.0, 0.0))
    self.assertTrue(reward.passed[0].item())
    self.assertFalse(reward.confirmed[0].item())

  def test_pass_is_confirmed_after_the_confirm_time_and_completes(self) -> None:
    env = _env(bar_x=1.0, base_x=1.3, feet_x=[1.25, 1.3])
    reward = hurdle.hurdle_pass_reward(types.SimpleNamespace(), env)
    env.reward_manager = types.SimpleNamespace(get_term_cfg=lambda _: types.SimpleNamespace(func=reward))
    steps = round(hurdle.HURDLE_CONFIRM_TIME / env.step_dt)
    done = []
    # The pass step, then HURDLE_CONFIRM_TIME of steps: it ends on the last of them.
    for _ in range(steps + 1):
      reward(env, "hurdle_contact", _site_cfg())
      done.append(hurdle.hurdle_complete(env).item())
    self.assertEqual(done, [False] * steps + [True])

  def test_trailing_foot_on_the_bar_is_not_a_pass(self) -> None:
    env = _env(bar_x=1.0, base_x=1.1, feet_x=[1.0, 1.3])
    reward = hurdle.hurdle_pass_reward(types.SimpleNamespace(), env)
    self.assertEqual(reward(env, "hurdle_contact", _site_cfg()).item(), 0.0)

  def test_going_around_the_end_is_not_a_pass(self) -> None:
    env = _env(bar_x=1.0, base_x=1.3, feet_x=[1.3, 1.3])
    env.asset.data.root_link_pos_w[:, 1] = hurdle.HURDLE_WIDTH / 2 + 0.3
    reward = hurdle.hurdle_pass_reward(types.SimpleNamespace(), env)
    self.assertEqual(reward(env, "hurdle_contact", _site_cfg()).item(), 0.0)

  def test_flat_env_never_passes(self) -> None:
    env = _env(is_hurdle=(False,), bar_x=1.0, base_x=1.3, feet_x=[1.3, 1.3])
    reward = hurdle.hurdle_pass_reward(types.SimpleNamespace(), env)
    self.assertEqual(reward(env, "hurdle_contact", _site_cfg()).item(), 0.0)


class TestHurdleStepShaping(unittest.TestCase):
  """hurdle_foot_lift_reward and hurdle_stride_progress_reward, bar at x = 1."""

  def _env(self, base_x=0.8):
    env = _env(bar_x=1.0, base_x=base_x, feet_x=[base_x, base_x])
    feet = types.SimpleNamespace(data=types.SimpleNamespace(found=torch.ones((1, 2)), heights=torch.zeros((1, 2))))
    robot = env.scene["robot"]
    lookup = {"robot": robot, "feet_ground_contact": feet, "foot_height_scan": feet}
    env.scene.__class__.__getitem__ = lambda _, key: lookup[key]
    return env, feet

  def _step(self, term, env, feet, left_air, left_height, left_x):
    feet.data.found = torch.tensor([[0.0 if left_air else 1.0, 1.0]])
    feet.data.heights = torch.tensor([[left_height, 0.0]])
    env.asset.data.site_pos_w[0, 0, 0] = left_x
    return term(env, "feet_ground_contact", "foot_height_scan", _site_cfg()).item()

  def test_lift_pays_the_rise_from_two_to_six_cm(self) -> None:
    env, feet = self._env()
    term = hurdle.hurdle_foot_lift_reward(types.SimpleNamespace(), env)
    steps = [(True, 0.02), (True, 0.04), (True, 0.03), (True, 0.08), (False, 0.0), (True, 0.04)]
    paid = [self._step(term, env, feet, air, h, 0.8) for air, h in steps]
    for got, want in zip(paid, [0.0, 0.5, 0.0, 0.5, 0.0, 0.5]):
      self.assertAlmostEqual(got, want, places=5)

  def test_lift_far_from_the_bar_pays_nothing(self) -> None:
    env, feet = self._env(base_x=0.0)
    term = hurdle.hurdle_foot_lift_reward(types.SimpleNamespace(), env)
    self.assertEqual(self._step(term, env, feet, True, 0.05, 0.0), 0.0)

  def test_stride_pays_the_rise_and_sums_to_one(self) -> None:
    env, feet = self._env()
    term = hurdle.hurdle_stride_progress_reward(types.SimpleNamespace(), env)
    xs = [0.8, 0.85, 0.95, 0.9, 1.05, 1.2]
    paid = [self._step(term, env, feet, True, 0.03, x) for x in xs]
    for got, want in zip(paid, [0.0, 0.0, 0.5, 0.0, 0.5, 0.0]):
      self.assertAlmostEqual(got, want, places=5)

  def test_stride_on_the_ground_or_skimming_pays_nothing(self) -> None:
    env, feet = self._env()
    term = hurdle.hurdle_stride_progress_reward(types.SimpleNamespace(), env)
    self._step(term, env, feet, True, 0.005, 0.8)
    self.assertEqual(self._step(term, env, feet, True, 0.005, 1.1), 0.0)
    self.assertEqual(self._step(term, env, feet, False, 0.0, 1.2), 0.0)


class TestHurdleLevels(unittest.TestCase):
  def _run(self, passed, failed, levels):
    n = len(passed)
    env = _env(num_envs=n, is_hurdle=(True,) * n)
    record = hurdle.hurdle_pass_reward(types.SimpleNamespace(), env)
    record.episode_time[:] = 5.0
    record.passed[:] = torch.tensor(passed)
    env.termination_manager.terminated[:] = torch.tensor(failed)
    env.command.level[:] = torch.tensor(levels)
    env.reward_manager = types.SimpleNamespace(
      get_term_cfg=lambda _: types.SimpleNamespace(func=record)
    )
    curriculum = hurdle.hurdle_levels(types.SimpleNamespace(), env)
    log = curriculum(env, torch.arange(n))
    return env.command.level.tolist(), log

  def test_up_on_a_clean_pass_down_on_no_pass_stay_on_fall_after(self) -> None:
    top = hurdle.HURDLE_LEVELS - 1
    levels, log = self._run(
      passed=[True, False, True, True], failed=[False, False, True, False], levels=[0, 1, 1, top]
    )
    self.assertEqual(levels, [1, 0, 1, top])
    self.assertAlmostEqual(log["fell_after"].item(), 0.25)
    self.assertAlmostEqual(log["level1_success"].item(), 0.0)


class TestVisionSymmetry(unittest.TestCase):
  def test_vision_and_truth_flip_y_only(self) -> None:
    signs = torch.tensor(_VECTOR_TERMS["hurdle_vision"]).view(hurdle.VISION_FRAMES, 4)
    torch.testing.assert_close(signs, torch.tensor([1.0, 1.0, -1.0, 1.0]).repeat(hurdle.VISION_FRAMES, 1))
    self.assertEqual(_VECTOR_TERMS["hurdle_truth"], (1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0))


class TestVelocityWarmStart(unittest.TestCase):
  def setUp(self) -> None:
    self.source = _source_checkpoint()
    self.original = copy.deepcopy(self.source)
    self.wide = warm_start.widen_checkpoint(self.source)

  def _check(self, model: str, layout, flat_dim: int, wide_dim: int) -> None:
    self.assertEqual(self.wide[model]["mlp.0.weight"].shape, (512, wide_dim))
    columns = warm_start.source_columns(*layout)
    flat_obs = torch.randn(16, flat_dim)
    # Older frames and the hurdle inputs are noise: their weights are zero.
    wide_obs = torch.randn(16, wide_dim)
    wide_obs[:, columns] = flat_obs
    torch.testing.assert_close(
      _forward(self.wide[model], wide_obs), _forward(self.original[model], flat_obs)
    )

  def test_actor_acts_the_same(self) -> None:
    layout = (warm_start.ACTOR_TERM_SIZES, warm_start.ACTOR_LEAD_DIM, warm_start.FLAT_ACTOR_DIM)
    self._check("actor_state_dict", layout, 45, 213 + hurdle.VISION_FRAMES * hurdle.VISION_FRAME_DIM)

  def test_critic_values_the_same(self) -> None:
    layout = (warm_start.CRITIC_TERM_SIZES, warm_start.CRITIC_LEAD_DIM, warm_start.FLAT_CRITIC_DIM)
    self._check("critic_state_dict", layout, 76, 308 + 7)

  def test_new_inputs_take_the_given_stats_and_first_layer_moments_go(self) -> None:
    mean = self.wide["critic_state_dict"]["obs_normalizer._mean"][0]
    torch.testing.assert_close(mean[-7:], torch.tensor(warm_start.HURDLE_TRUTH_MEAN))
    mean = self.wide["actor_state_dict"]["obs_normalizer._mean"][0]
    torch.testing.assert_close(mean[-4:], torch.tensor(warm_start.VISION_FRAME_MEAN))
    self.assertEqual(sorted(self.wide["optimizer_state_dict"]["state"]), [2])
    self.assertEqual(self.wide["iter"], 193900)

  def test_other_shapes_are_refused(self) -> None:
    self.source["actor_state_dict"]["mlp.0.weight"] = torch.zeros(512, 213)
    with self.assertRaises(ValueError):
      warm_start.widen_checkpoint(self.source)


class TestStackedSymmetry(unittest.TestCase):
  def test_every_frame_is_mirrored_alike(self) -> None:
    term_cfgs = [
      types.SimpleNamespace(history_length=3),
      types.SimpleNamespace(history_length=0),
    ]
    obs_manager = types.SimpleNamespace(
      active_terms={"actor": ["base_ang_vel", "command"]},
      group_obs_term_dim={"actor": [(9,), (3,)]},
      _group_obs_term_cfgs={"actor": term_cfgs},
    )
    unwrapped = types.SimpleNamespace(
      observation_manager=obs_manager,
      scene={"robot": types.SimpleNamespace(joint_names=["a"])},
      device="cpu",
      action_manager=types.SimpleNamespace(
        _terms={"joint_pos": types.SimpleNamespace(target_names=["a"])}
      ),
    )
    spec = _build_spec(types.SimpleNamespace(unwrapped=unwrapped))["actor"]
    self.assertEqual([slice_[:2] for slice_ in spec], [(0, 3), (3, 6), (6, 9), (9, 12)])
    frame = torch.tensor([1.0, 2.0, 3.0])
    obs = torch.cat((frame, 2 * frame, 3 * frame, torch.tensor([0.3, 0.1, 0.2])))
    signs = torch.tensor(_VECTOR_TERMS["base_ang_vel"])
    expected = torch.cat(
      (frame * signs, 2 * frame * signs, 3 * frame * signs, torch.tensor([0.3, -0.1, -0.2]))
    )
    torch.testing.assert_close(_apply(obs, spec), expected)


def _source_checkpoint() -> dict:
  torch.manual_seed(0)
  def model(width: int, outputs: int) -> dict:
    return {
      "obs_normalizer._mean": torch.randn(1, width),
      "obs_normalizer._var": torch.rand(1, width) + 0.5,
      "obs_normalizer._std": torch.ones(1, width),
      "obs_normalizer.count": torch.tensor(3_000_000_000),
      "mlp.0.weight": torch.randn(512, width),
      "mlp.0.bias": torch.randn(512),
      "mlp.2.weight": torch.randn(256, 512),
      "mlp.2.bias": torch.randn(256),
      "mlp.4.weight": torch.randn(128, 256),
      "mlp.4.bias": torch.randn(128),
      "mlp.6.weight": torch.randn(outputs, 128),
      "mlp.6.bias": torch.randn(outputs),
    }

  actor = model(warm_start.FLAT_ACTOR_DIM, 12)
  critic = model(warm_start.FLAT_CRITIC_DIM, 1)
  for state in (actor, critic):
    state["obs_normalizer._std"] = torch.sqrt(state["obs_normalizer._var"])
  moments = {
    0: {"step": torch.tensor(5.0), "exp_avg": torch.ones(512, 45), "exp_avg_sq": torch.ones(512, 45)},
    1: {"step": torch.tensor(5.0), "exp_avg": torch.ones(512, 76), "exp_avg_sq": torch.ones(512, 76)},
    2: {"step": torch.tensor(5.0), "exp_avg": torch.ones(512), "exp_avg_sq": torch.ones(512)},
  }
  return {
    "actor_state_dict": actor,
    "critic_state_dict": critic,
    "optimizer_state_dict": {"state": moments, "param_groups": []},
    "iter": 193900,
    "infos": {},
  }


def _forward(state: dict, obs: torch.Tensor) -> torch.Tensor:
  # Double: the widened sum runs in another column order, so float32 rounds it apart.
  state = {key: value.double() for key, value in state.items()}
  x = (obs.double() - state["obs_normalizer._mean"]) / (state["obs_normalizer._std"] + 1e-2)
  for layer in (0, 2, 4):
    x = torch.nn.functional.elu(x @ state[f"mlp.{layer}.weight"].T + state[f"mlp.{layer}.bias"])
  return x @ state["mlp.6.weight"].T + state["mlp.6.bias"]


if __name__ == "__main__":
  unittest.main()
