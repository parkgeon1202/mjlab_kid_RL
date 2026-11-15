"""Record one env's joint targets under an exported velocity policy.

Runs POLICY (an exported flat IRC policy, 45 inputs) in one env of the flat IRC
training task for STEPS steps at 50 Hz, forcing a new twist command every
COMMAND_PERIOD steps, and saves an npz with
  joint_pos_rel [STEPS, 12]: action x action_scale, i.e. joint target - default, in the
                             actor joint_pos order (rad);
  episode       [STEPS]:     episode index, so windows that cross a reset can be skipped.

  python src/mjlab_kid_rl/scripts/record_onnx_actions.py policy.onnx OUT.npz
"""

from __future__ import annotations

import argparse

import numpy as np
import onnxruntime as ort
import torch
from mjlab.envs import ManagerBasedRlEnv

from mjlab_kid_rl.tasks.kid_RL_env_cfg import make_kid_rl_velocity_env_cfg


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("policy")
  parser.add_argument("out")
  parser.add_argument("--steps", type=int, default=2000)
  parser.add_argument("--command-period", type=int, default=100)
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  torch.manual_seed(args.seed)
  cfg = make_kid_rl_velocity_env_cfg(flat=True)
  cfg.scene.num_envs = 1
  cfg.seed = args.seed
  env = ManagerBasedRlEnv(cfg, device="cuda:0" if torch.cuda.is_available() else "cpu")
  session = ort.InferenceSession(args.policy)
  scale = env.action_manager.get_term("joint_pos")._scale
  command = env.command_manager.get_term("twist")

  obs, _ = env.reset()
  targets, episodes = [], []
  episode = 0
  for step in range(args.steps):
    if step > 0 and step % args.command_period == 0:
      command.time_left[:] = 0.0  # resampled on the next step
    action = session.run(None, {"obs": obs["actor"].cpu().numpy().astype(np.float32)})[0]
    action = torch.from_numpy(action).to(env.device)
    targets.append((action * scale)[0].cpu().numpy())
    episodes.append(episode)
    obs, _, terminated, truncated, _ = env.step(action)
    episode += int(terminated[0] or truncated[0])

  np.savez(args.out, joint_pos_rel=np.asarray(targets, np.float32), episode=np.asarray(episodes))
  print(f"wrote {args.out}: {len(targets)} steps, {episode + 1} episodes")


if __name__ == "__main__":
  main()
