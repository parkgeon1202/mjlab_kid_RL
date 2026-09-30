"""CAPS spatial and temporal regularization for the project's feedforward PPO.

Reference: https://ai.bu.edu/caps/CAPS_code.zip,
CAPS-GymBenchmarks/rl_smoothness/algs/ppo/ppo.py (lines 211-214).
The official PPO code uses tf.nn.l2_loss(delta)/batch_size, i.e. half
of the mean action squared norm, rather than the website's unsquared norm.
Selected observation noise is applied before the actor normalizer.
"""

from __future__ import annotations

import math

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import PPO


class SmoothPPO(PPO):
    """PPO + spatial_coef * L_S + temporal_coef * L_T.

    smooth_cfg = {"spatial_coef": 0.01, "temporal_coef": 0.01,
                  "slices": [(0, 3, 0.3)]}
    Both terms use deterministic, unclipped current-policy mean actions.
    Temporal pairs are actual env transitions, never adjacent shuffled rows.
    Terminal/time-limit reset transitions are excluded. CAPS uses original
    rollout observations; the stock PPO symmetry augmentation remains separate.
    Legacy "coef" is accepted as an alias for "spatial_coef".
    """

    def __init__(self, *args, smooth_cfg: dict | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        cfg = smooth_cfg or {}
        self._smooth_coef = float(cfg.get("spatial_coef", cfg.get("coef", 0.0)))
        self._temporal_coef = float(cfg.get("temporal_coef", 0.0))
        for name, value in [("spatial_coef", self._smooth_coef),
                            ("temporal_coef", self._temporal_coef)]:
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"smooth_cfg.{name} must be finite and nonnegative")
        self._smooth_slices: list[tuple[int, int, float]] = []
        used: set[int] = set()
        for start, stop, sigma in cfg.get("slices", []):
            if not isinstance(start, int) or not isinstance(stop, int):
                raise ValueError("smooth_cfg slice boundaries must be integers")
            sigma = float(sigma)
            if start < 0 or stop <= start:
                raise ValueError("smooth_cfg slices require 0 <= start < stop")
            if not math.isfinite(sigma) or sigma < 0:
                raise ValueError("smooth_cfg sigma must be finite and nonnegative")
            indices = set(range(start, stop))
            if used & indices:
                raise ValueError("smooth_cfg slices must not overlap")
            used |= indices
            self._smooth_slices.append((start, stop, sigma))
        self._caps_enabled = self._smooth_coef > 0 or self._temporal_coef > 0
        if not self._caps_enabled:
            return
        if self._smooth_coef > 0 and not self._smooth_slices:
            raise ValueError("positive spatial_coef requires observation slices")
        if self.actor.is_recurrent or self.critic.is_recurrent:
            raise ValueError("SmoothPPO currently supports feedforward models only")
        if list(self.actor.obs_groups) != ["actor"]:
            raise ValueError("SmoothPPO requires actor.obs_groups == ['actor']")
        width = self.storage.observations["actor"].shape[-1]
        if any(stop > width for _, stop, _ in self._smooth_slices):
            raise ValueError(f"smooth_cfg slice exceeds actor observation width {width}")
        if self._temporal_coef:
            self._next_actor_obs = torch.zeros_like(self.storage.observations["actor"])

    def process_env_step(self, obs, rewards, dones, extras) -> None:
        if self._temporal_coef:
            # Copy before the env reuses its tensors; include the last rollout step.
            with torch.no_grad():
                self._next_actor_obs[self.storage.step].copy_(obs["actor"])
        super().process_env_step(obs, rewards, dones, extras)

    def _perturb_observations(self, obs: TensorDict) -> TensorDict:
        perturbed = obs.clone()
        for start, stop, sigma in self._smooth_slices:
            if sigma:
                perturbed["actor"][:, start:stop] += (
                    torch.randn_like(perturbed["actor"][:, start:stop]) * sigma
                )
        return perturbed

    def _smooth_loss(self, obs: TensorDict) -> torch.Tensor:
        mean = self.actor(obs, stochastic_output=False)
        nearby_mean = self.actor(self._perturb_observations(obs), stochastic_output=False)
        return 0.5 * (nearby_mean - mean).square().sum(dim=-1).mean()

    def _temporal_loss(self, obs, next_actor_obs, valid) -> torch.Tensor:
        mask = valid.reshape(-1).bool()
        current_obs = obs[mask]
        next_obs = TensorDict({"actor": next_actor_obs[mask]}, current_obs.batch_size)
        mean = self.actor(current_obs, stochastic_output=False)
        next_mean = self.actor(next_obs, stochastic_output=False)
        per_pair = 0.5 * (next_mean - mean).square().sum(dim=-1)
        # Empty valid batch -> differentiable zero. Invalid/reset observations
        # never enter either forward (even NaN reset observations are excluded).
        return per_pair.sum() / max(1, per_pair.numel())

    def update(self) -> dict[str, float]:
        if not self._caps_enabled:
            return super().update()
        minibatch = None
        spatial_losses, temporal_losses = [], []
        original_observations = self.storage.observations
        original_generator = self.storage.mini_batch_generator
        had_generator = "mini_batch_generator" in self.storage.__dict__
        previous_generator = self.storage.__dict__.get("mini_batch_generator")
        original_zero_grad = self.optimizer.zero_grad
        had_zero_grad = "zero_grad" in self.optimizer.__dict__
        previous_zero_grad = self.optimizer.__dict__.get("zero_grad")

        def caps_generator(*args, **kwargs):
            nonlocal minibatch
            for batch in original_generator(*args, **kwargs):
                obs = batch.observations
                if self._temporal_coef:
                    next_actor = obs["_caps_next_actor"]
                    valid = obs["_caps_valid"]
                    obs = obs.exclude("_caps_next_actor", "_caps_valid")
                    batch.observations = obs
                else:
                    next_actor, valid = None, None
                # Captured before PPO mutates the batch for symmetry augmentation.
                minibatch = (obs, next_actor, valid)
                yield batch

        def zero_grad_then_caps(*args, **kwargs):
            nonlocal minibatch
            original_zero_grad(*args, **kwargs)
            if minibatch is None:
                raise RuntimeError("PPO update did not supply a CAPS minibatch")
            obs, next_actor, valid = minibatch
            minibatch = None
            loss = None
            if self._smooth_coef:
                spatial = self._smooth_loss(obs)
                spatial_losses.append(spatial.detach())
                loss = self._smooth_coef * spatial
            if self._temporal_coef:
                temporal = self._temporal_loss(obs, next_actor, valid)
                temporal_losses.append(temporal.detach())
                weighted = self._temporal_coef * temporal
                loss = weighted if loss is None else loss + weighted
            loss.backward()

        try:
            if self._temporal_coef:
                # Shallow copy of keys: PPO tensors and storage content are unchanged.
                enriched = original_observations.clone(recurse=False)
                enriched["_caps_next_actor"] = self._next_actor_obs
                enriched["_caps_valid"] = ~self.storage.dones.bool()
                self.storage.observations = enriched
            self.storage.mini_batch_generator = caps_generator
            self.optimizer.zero_grad = zero_grad_then_caps
            # Stock backward accumulates onto CAPS gradients. Stock distributed
            # reduction, gradient clipping, optimizer step and clear remain intact.
            loss_dict = super().update()
        finally:
            self.storage.observations = original_observations
            for obj, name, had, previous in [
                (self.storage, "mini_batch_generator", had_generator, previous_generator),
                (self.optimizer, "zero_grad", had_zero_grad, previous_zero_grad),
            ]:
                if had:
                    setattr(obj, name, previous)
                else:
                    obj.__dict__.pop(name, None)
        spatial = torch.stack(spatial_losses).mean().item() if spatial_losses else 0.0
        temporal = torch.stack(temporal_losses).mean().item() if temporal_losses else 0.0
        loss_dict.update(
            caps_spatial=spatial,
            caps_temporal=temporal,
            caps_spatial_weighted=self._smooth_coef * spatial,
            caps_temporal_weighted=self._temporal_coef * temporal,
            caps_total=self._smooth_coef * spatial + self._temporal_coef * temporal,
            smooth=spatial,  # Existing TensorBoard dashboards.
            smooth_weighted=self._smooth_coef * spatial,
        )
        return loss_dict
