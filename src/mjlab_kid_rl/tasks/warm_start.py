"""Widen an IRC velocity checkpoint's inputs for the mocap-bar hurdle task.

The velocity networks read 45 actor and 76 critic numbers; the hurdle task stacks
OBS_HISTORY_LENGTH frames of the same terms in both, appends hurdle_vision's 44 inputs
to the actor and hurdle_truth's 7 to the critic. This keeps the hidden layers and
widens each network's first layer. The old weights move to the newest frame's columns
and every new column starts at zero, so both networks output exactly what the source
ones output until training changes them. The first layers' Adam state is dropped: new
columns have no moments of their own, and a zero second moment at the source's step
count makes Adam take oversized steps on them.

  python -m mjlab_kid_rl.tasks.warm_start SOURCE_RUN_DIR model_193900.pt OUTPUT_RUN_DIR
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mjlab_kid_rl.tasks.hurdle import OBS_HISTORY_LENGTH

ACTOR_TERM_SIZES = (3, 3, 12, 12, 12)
"""Stacked terms in observation order: base_ang_vel, projected_gravity, joint_pos,
joint_vel, actions. The actor sees the 12 actuated joints."""
CRITIC_TERM_SIZES = (3, 3, 20, 20, 12)
"""The same terms in the critic, whose joint terms cover all 20 joints."""
ACTOR_LEAD_DIM = 0
CRITIC_LEAD_DIM = 3
"""Unstacked columns before the stack: the critic's base_lin_vel."""
FLAT_ACTOR_DIM = 45
FLAT_CRITIC_DIM = 76

VISION_FRAMES = 11
VISION_FRAME_MEAN = (0.6, 0.4, 0.0, -0.24)
VISION_FRAME_STD = (0.49, 0.45, 0.1, 0.2)
"""Normalizer stats of one hurdle_vision frame (visible, x, y, z): about 60% of all
frames see the bar, at about 0.7 m and 0.4 m below the base."""
HURDLE_TRUTH_MEAN = (0.5, 0.0, -0.3, 0.0, 0.7, 0.03, 0.7)
HURDLE_TRUTH_STD = (0.5, 0.15, 0.2, 0.1, 0.46, 0.025, 0.46)
"""Normalizer stats of hurdle_truth (x, y, z, sin, cos, height, is_hurdle)."""
ACTOR_EXTRA = (VISION_FRAME_MEAN * VISION_FRAMES, VISION_FRAME_STD * VISION_FRAMES)
CRITIC_EXTRA = (HURDLE_TRUTH_MEAN, HURDLE_TRUTH_STD)


def source_columns(term_sizes: tuple[int, ...], lead: int, flat_dim: int) -> list[int]:
  """Return, for each column of a flat observation, its column in the hurdle one."""
  columns = list(range(lead))
  new_start = lead
  for size in term_sizes:
    newest = new_start + (OBS_HISTORY_LENGTH - 1) * size
    columns.extend(range(newest, newest + size))
    new_start += OBS_HISTORY_LENGTH * size
  # command and everything after it follow the stack unstacked.
  columns.extend(range(new_start, new_start + flat_dim - lead - sum(term_sizes)))
  return columns


def stats_sources(term_sizes: tuple[int, ...], lead: int, flat_dim: int) -> list[int]:
  """Return the flat column each widened column takes its normalizer stats from.

  Every frame of a term inherits that term's stats. The appended hurdle inputs are
  not listed; they follow at the end.
  """
  sources = list(range(lead))
  start = lead
  for size in term_sizes:
    sources += list(range(start, start + size)) * OBS_HISTORY_LENGTH
    start += size
  return sources + list(range(start, flat_dim))


def _widen_model(
  state: dict,
  columns: list[int],
  stats: list[int],
  extra: tuple[tuple[float, ...], tuple[float, ...]],
) -> None:
  """Widen a state dict's first layer and normalizer in place; the appended
  ``extra`` inputs get zero weight and the given (mean, std)."""
  extra_mean, extra_std = extra
  weight = state["mlp.0.weight"]
  widened = torch.zeros((weight.shape[0], len(stats) + len(extra_mean)), dtype=weight.dtype)
  widened[:, columns] = weight
  state["mlp.0.weight"] = widened

  mean = state["obs_normalizer._mean"]
  var = state["obs_normalizer._var"]
  new_mean = torch.cat(
    (mean[:, stats], torch.tensor(extra_mean, dtype=mean.dtype).unsqueeze(0)), dim=-1
  )
  new_var = torch.cat(
    (var[:, stats], torch.tensor(extra_std, dtype=var.dtype).unsqueeze(0) ** 2), dim=-1
  )
  state["obs_normalizer._mean"] = new_mean
  state["obs_normalizer._var"] = new_var
  state["obs_normalizer._std"] = torch.sqrt(new_var)


def widen_checkpoint(checkpoint: dict) -> dict:
  """Return a copy of an IRC velocity checkpoint shaped for the hurdle task."""
  actor = checkpoint["actor_state_dict"]["mlp.0.weight"].shape[1]
  critic = checkpoint["critic_state_dict"]["mlp.0.weight"].shape[1]
  if (actor, critic) != (FLAT_ACTOR_DIM, FLAT_CRITIC_DIM):
    raise ValueError(f"not an IRC velocity checkpoint: inputs {actor}, {critic}")
  actor_layout = (ACTOR_TERM_SIZES, ACTOR_LEAD_DIM, FLAT_ACTOR_DIM)
  critic_layout = (CRITIC_TERM_SIZES, CRITIC_LEAD_DIM, FLAT_CRITIC_DIM)

  widened = dict(checkpoint)
  widened["actor_state_dict"] = dict(checkpoint["actor_state_dict"])
  widened["critic_state_dict"] = dict(checkpoint["critic_state_dict"])
  _widen_model(
    widened["actor_state_dict"],
    source_columns(*actor_layout),
    stats_sources(*actor_layout),
    ACTOR_EXTRA,
  )
  _widen_model(
    widened["critic_state_dict"],
    source_columns(*critic_layout),
    stats_sources(*critic_layout),
    CRITIC_EXTRA,
  )

  # Adam rebuilds a parameter's state on its next step when it finds none.
  optimizer = dict(checkpoint["optimizer_state_dict"])
  first_layers = ((512, FLAT_ACTOR_DIM), (512, FLAT_CRITIC_DIM))
  optimizer["state"] = {
    index: state
    for index, state in checkpoint["optimizer_state_dict"]["state"].items()
    if tuple(state["exp_avg"].shape) not in first_layers
  }
  widened["optimizer_state_dict"] = optimizer
  return widened


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("source_run", type=Path)
  parser.add_argument("checkpoint")
  parser.add_argument("output_run", type=Path)
  args = parser.parse_args()

  checkpoint = torch.load(
    args.source_run / args.checkpoint, map_location="cpu", weights_only=False
  )
  args.output_run.mkdir(parents=True, exist_ok=False)
  torch.save(widen_checkpoint(checkpoint), args.output_run / args.checkpoint)
  print(f"wrote {args.output_run / args.checkpoint}")


if __name__ == "__main__":
  main()
