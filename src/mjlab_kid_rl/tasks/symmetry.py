"""Left/right spatial mirroring for RSL-RL's symmetry extension.

This module constructs the mirrored observation and action used to encourage
the policy relation ``pi(M(o)) = M(pi(o))``. That is a same-time spatial
condition, not a temporal half-cycle gait condition. Standing, hopping, and an
irregular but spatially symmetric policy can all satisfy it. Enforcing
``state(t + T/2) = M(state(t))`` would additionally require a gait phase or
trajectory history.

RSL-RL's PPO takes a ``symmetry_cfg`` whose ``data_augmentation_func`` produces
the mirrored copy of a batch, and then either

- trains on the mirrored samples as extra data (``use_data_augmentation``), or
- adds ``mirror_loss_coeff * MSE(pi(mirror(obs)), mirror(pi(obs)))`` straight
  to the PPO loss (``use_mirror_loss``),

either of which pushes the policy toward spatially mirrored responses. This can
regularize an already-discovered gait but does not create its cadence or choose
which foot should initiate a step from a perfectly symmetric state.

Sign convention was verified by forward kinematics on every actuated
left/right pair. A mirrored pose uses q_left + q_right == 0, i.e. swap and
negate. The reason is the complete sagittal reflection, not simply that every
stored joint axis is opposite: several roll/yaw pairs use the same MJCF axis.
The resulting mirrored body positions agree to within the model's own roughly
2 mm left/right CAD asymmetry.

Centre-line joints have no partner and are handled by axis instead: under a
mirror through the sagittal plane an angular velocity transforms as
(wx, wy, wz) -> (-wx, wy, -wz), so roll (x) and yaw (z) joints negate while
pitch (y) joints are unchanged. For this robot that is torso_yaw and neck_yaw
negated, head_pitch left alone.
"""

from __future__ import annotations

import torch

# Root-frame vector mirrors, as per-component sign patterns.
# Body frame is x forward, y left, z up; the mirror is through the x-z plane.
_LINEAR_SIGNS = (1.0, -1.0, 1.0)  # a true vector: y flips
_ANGULAR_SIGNS = (-1.0, 1.0, -1.0)  # a pseudo-vector: x and z flip

# Observation terms handled by name. Everything else must be a joint-space or
# per-foot term, or the spec builder refuses to guess.
_VECTOR_TERMS = {
    "base_lin_vel": _LINEAR_SIGNS,
    "base_ang_vel": _ANGULAR_SIGNS,
    "projected_gravity": _LINEAR_SIGNS,
    # (vx, vy, wz): the linear part flips in y, the yaw rate flips outright.
    "command": (1.0, -1.0, -1.0),
    "velocity_commands": (1.0, -1.0, -1.0),
    # [sin(2*pi*phase), cos(2*pi*phase)]. Mirroring a gait
    # swaps which leg is swinging, which *is* a half-cycle shift of the clock:
    # phase -> phase + 0.5, so sin and cos both negate (sin(x+pi) = -sin x,
    # cos(x+pi) = -cos x) while the period itself is unchanged. This is the
    # half-cycle phase offset stated directly in the observation, and it is
    # what makes the mirror consistent with a walking gait rather than with
    # both legs moving together.
    "gait_phase": (-1.0, -1.0),
}
# Per-foot scalars: mirroring just swaps the two feet.
_FOOT_SCALAR_TERMS = ("foot_height", "foot_air_time", "foot_contact")
# Per-foot 3-vectors: swap the feet, then flip each vector's y.
_FOOT_VECTOR_TERMS = ("foot_contact_forces",)


def _joint_mirror_table(joint_names: list[str]) -> tuple[list[int], list[float]]:
    """Build (perm, sign) with mirrored[..., i] = sign[i] * x[..., perm[i]]."""
    index = {name: i for i, name in enumerate(joint_names)}
    perm = list(range(len(joint_names)))
    sign = [1.0] * len(joint_names)

    for i, name in enumerate(joint_names):
        if name.startswith("left_"):
            partner = "right_" + name[len("left_") :]
        elif name.startswith("right_"):
            partner = "left_" + name[len("right_") :]
        else:
            partner = None

        if partner is not None:
            if partner not in index:
                raise ValueError(
                    f"Joint '{name}' has no mirror partner '{partner}' in the "
                    "observed joint set; symmetry cannot be defined."
                )
            perm[i] = index[partner]
            # Opposite axis signs in the model make the mirrored pose q_l + q_r = 0.
            sign[i] = -1.0
        else:
            # Centre-line joint: roll/yaw flip under a sagittal mirror, pitch does not.
            sign[i] = -1.0 if ("yaw" in name or "roll" in name) else 1.0

    return perm, sign


def _build_spec(env) -> dict:
    """Per-group list of (start, stop, perm_or_None, sign_tensor) slices."""
    unwrapped = env.unwrapped
    obs_manager = unwrapped.observation_manager
    robot = unwrapped.scene["robot"]
    device = unwrapped.device

    all_joints = list(robot.joint_names)
    action_term = next(iter(unwrapped.action_manager._terms.values()))
    action_joints = list(action_term.target_names)

    tables = {
        len(all_joints): _joint_mirror_table(all_joints),
        len(action_joints): _joint_mirror_table(action_joints),
    }

    spec: dict[str, list] = {}
    for group, names in obs_manager.active_terms.items():
        dims = obs_manager.group_obs_term_dim[group]
        slices = []
        offset = 0
        for name, dim in zip(names, dims):
            size = int(dim[0])
            stop = offset + size

            if name in _VECTOR_TERMS:
                signs = _VECTOR_TERMS[name]
                if size != len(signs):
                    raise ValueError(f"'{name}' has dim {size}, expected {len(signs)}")
                slices.append((offset, stop, None, torch.tensor(signs, device=device)))
            elif name in _FOOT_SCALAR_TERMS:
                if size != 2:
                    raise ValueError(f"'{name}' has dim {size}, expected 2 (per foot)")
                slices.append((offset, stop, [1, 0], torch.ones(2, device=device)))
            elif name in _FOOT_VECTOR_TERMS:
                if size != 6:
                    raise ValueError(f"'{name}' has dim {size}, expected 6 (2 feet x 3)")
                # Swap the two feet's 3-vectors, then flip y within each.
                perm = [3, 4, 5, 0, 1, 2]
                signs = torch.tensor(
                    _LINEAR_SIGNS + _LINEAR_SIGNS, device=device, dtype=torch.float32
                )
                slices.append((offset, stop, perm, signs))
            elif size in tables:
                # joint_pos / joint_vel / actions, in whichever joint set matches.
                perm, sign = tables[size]
                slices.append(
                    (offset, stop, perm, torch.tensor(sign, device=device))
                )
            else:
                raise ValueError(
                    f"Observation term '{name}' (group '{group}', dim {size}) has no "
                    "mirror rule. Add one to mjlab_kid_rl.tasks.symmetry before "
                    "enabling symmetry, or the mirrored batch would be wrong."
                )
            offset = stop
        spec[group] = slices

    action_perm, action_sign = tables[len(action_joints)]
    spec["__actions__"] = [
        (0, len(action_joints), action_perm, torch.tensor(action_sign, device=device))
    ]
    return spec


def _apply(flat: torch.Tensor, slices: list) -> torch.Tensor:
    out = torch.empty_like(flat)
    for start, stop, perm, signs in slices:
        chunk = flat[..., start:stop]
        if perm is not None:
            chunk = chunk[..., perm]
        out[..., start:stop] = chunk * signs
    return out


def compute_symmetric_states(env, obs=None, actions=None):
    """RSL-RL ``data_augmentation_func``: returns [original; mirrored].

    Either argument may be None, which RSL-RL does when it only needs one side
    (see PPO's mirror-loss branch). The returned batch is twice as long, with
    the original copy first -- RSL-RL relies on that ordering.
    """
    cache = getattr(env, "_kid_rl_symmetry_spec", None)
    if cache is None:
        cache = _build_spec(env)
        env._kid_rl_symmetry_spec = cache

    obs_out = None
    if obs is not None:
        mirrored = obs.clone()
        for group, slices in cache.items():
            if group == "__actions__" or group not in obs.keys():
                continue
            mirrored[group] = _apply(obs[group], slices)
        obs_out = torch.cat([obs, mirrored], dim=0)

    actions_out = None
    if actions is not None:
        mirrored_actions = _apply(actions, cache["__actions__"])
        actions_out = torch.cat([actions, mirrored_actions], dim=0)

    return obs_out, actions_out
