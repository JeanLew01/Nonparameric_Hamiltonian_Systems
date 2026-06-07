from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from baseline.behavior_cloning import BehaviorCloning
from utils.control_chain import ChainPolicySpec, _rk4_step, load_saved_rollouts, success_mask_from_spec
from dynamics.Dynamics_single_pendulum import I as SINGLE_I, m as SINGLE_M, g as SINGLE_G, l as SINGLE_L
from dynamics.Dynamics_double_pendulum import (
    g as DOUBLE_G,
    l1 as DOUBLE_L1,
    l2 as DOUBLE_L2,
    m1 as DOUBLE_M1,
    m2 as DOUBLE_M2,
)

__all__ = [
    "BehaviorCloningController",
    "MLPPolicy",
    "evaluate_incremental_behavior_cloning",
    "greedy_rank_behavior_cloning_trajectories",
    "select_behavior_cloning_trajectory_subset",
    "train_behavior_cloning_policy",
]


def _expanded_path(path_like) -> Path:
    return Path(os.path.expanduser(str(path_like)))


def _resolve_device(device):
    if device is None:
        return torch.device("cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(str(device))


def _set_random_seed(seed):
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _as_2d(array_like, width):
    arr = np.asarray(array_like, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, width)
    if arr.ndim != 2 or arr.shape[1] != width:
        raise ValueError(f"Expected shape (N, {width}), got {arr.shape}")
    return arr


def _encode_states(states, angle_indices=(), use_angle_features=False):
    X = np.asarray(states, dtype=np.float32)
    was_1d = X.ndim == 1
    if was_1d:
        X = X.reshape(1, -1)

    angle_set = set(int(idx) for idx in angle_indices)
    features = []
    for idx in range(X.shape[1]):
        column = X[:, idx : idx + 1]
        if use_angle_features and idx in angle_set:
            features.append(np.sin(column))
            features.append(np.cos(column))
        else:
            features.append(column)
    encoded = np.concatenate(features, axis=1).astype(np.float32)
    return encoded[0] if was_1d else encoded


def _norm_stats(data, enabled):
    data = np.asarray(data, dtype=np.float32)
    if not enabled:
        return np.zeros(data.shape[1], dtype=np.float32), np.ones(data.shape[1], dtype=np.float32)
    mean = np.mean(data, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(data, axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std.astype(np.float32)


def _align_rollout_pairs(X, U, control_dim):
    X = np.asarray(X, dtype=np.float32)
    U = np.asarray(U, dtype=np.float32)
    if U.ndim == 1:
        U = U.reshape(-1, control_dim)
    horizon = min(max(X.shape[0] - 1, 0), U.shape[0])
    return X[:horizon], U[:horizon, :control_dim]


def _check_success_batch(spec: ChainPolicySpec, obs_batch):
    return success_mask_from_spec(spec, obs_batch)


def _single_f_batch(states, actions):
    X = np.asarray(states, dtype=np.float32)
    U = np.asarray(actions, dtype=np.float32).reshape(-1, 1)
    theta = X[:, 0]
    p = X[:, 1]
    u = U[:, 0]
    theta_dot = p / SINGLE_I
    p_dot = -SINGLE_M * SINGLE_G * SINGLE_L * np.sin(theta) + u
    return np.stack([theta_dot, p_dot], axis=1).astype(np.float32)


def _rk4_step_single_batch(states, actions, dt):
    X = np.asarray(states, dtype=np.float32)
    U = np.asarray(actions, dtype=np.float32).reshape(-1, 1)
    dt = float(dt)
    k1 = _single_f_batch(X, U)
    k2 = _single_f_batch(X + 0.5 * dt * k1, U)
    k3 = _single_f_batch(X + 0.5 * dt * k2, U)
    k4 = _single_f_batch(X + dt * k3, U)
    return (X + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)).astype(np.float32)


def _double_f_batch(states, actions):
    X = np.asarray(states, dtype=np.float32)
    U = np.asarray(actions, dtype=np.float32)
    if U.ndim == 1:
        U = U.reshape(-1, 2)
    th1 = X[:, 0]
    th2 = X[:, 1]
    p1 = X[:, 2]
    p2 = X[:, 3]
    tau1 = U[:, 0]
    tau2 = U[:, 1]

    a11 = np.float32((DOUBLE_M1 + DOUBLE_M2) * DOUBLE_L1**2)
    a22 = np.float32(DOUBLE_M2 * DOUBLE_L2**2)
    coupling = np.float32(DOUBLE_M2 * DOUBLE_L1 * DOUBLE_L2) * np.cos(th1 - th2)
    det = a11 * a22 - coupling * coupling
    det = np.where(np.abs(det) < 1e-8, 1e-8, det).astype(np.float32)

    q1_dot = (a22 * p1 - coupling * p2) / det
    q2_dot = (-coupling * p1 + a11 * p2) / det
    coupling_grad = np.float32(DOUBLE_M2 * DOUBLE_L1 * DOUBLE_L2) * np.sin(th1 - th2)
    dH_dth1 = coupling_grad * q1_dot * q2_dot + np.float32(
        (DOUBLE_M1 + DOUBLE_M2) * DOUBLE_G * DOUBLE_L1
    ) * np.sin(th1)
    dH_dth2 = -coupling_grad * q1_dot * q2_dot + np.float32(
        DOUBLE_M2 * DOUBLE_G * DOUBLE_L2
    ) * np.sin(th2)
    p1_dot = tau1 - dH_dth1
    p2_dot = tau2 - dH_dth2
    return np.stack([q1_dot, q2_dot, p1_dot, p2_dot], axis=1).astype(np.float32)


def _rk4_step_double_batch(states, actions, dt):
    X = np.asarray(states, dtype=np.float32)
    U = np.asarray(actions, dtype=np.float32)
    if U.ndim == 1:
        U = U.reshape(-1, 2)
    dt = float(dt)
    k1 = _double_f_batch(X, U)
    k2 = _double_f_batch(X + 0.5 * dt * k1, U)
    k3 = _double_f_batch(X + 0.5 * dt * k2, U)
    k4 = _double_f_batch(X + dt * k3, U)
    return (X + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)).astype(np.float32)


def _run_bc_episodes_fast_single_batch(spec: ChainPolicySpec, controller, init_states, dt, episode_seconds):
    states = np.asarray(init_states, dtype=np.float32).copy()
    total_steps = int(np.ceil(float(episode_seconds) / float(dt)))
    success = _check_success_batch(spec, states)
    steps = np.zeros(states.shape[0], dtype=np.int32)
    active = ~success

    for _ in range(total_steps):
        if not np.any(active):
            break
        idx = np.flatnonzero(active)
        actions = controller.predict_actions(states[idx])
        states[idx] = _rk4_step_single_batch(states[idx], actions, dt)
        steps[idx] += 1
        newly_succeeded = _check_success_batch(spec, states[idx])
        if np.any(newly_succeeded):
            success[idx[newly_succeeded]] = True
            active[idx[newly_succeeded]] = False

    return {"success": success, "steps": steps.astype(np.int32)}


def _run_bc_episodes_fast_double_batch(spec: ChainPolicySpec, controller, init_states, dt, episode_seconds):
    states = np.asarray(init_states, dtype=np.float32).copy()
    total_steps = int(np.ceil(float(episode_seconds) / float(dt)))
    success = _check_success_batch(spec, states)
    steps = np.zeros(states.shape[0], dtype=np.int32)
    active = ~success

    for _ in range(total_steps):
        if not np.any(active):
            break
        idx = np.flatnonzero(active)
        actions = controller.predict_actions(states[idx])
        states[idx] = _rk4_step_double_batch(states[idx], actions, dt)
        steps[idx] += 1
        newly_succeeded = _check_success_batch(spec, states[idx])
        if np.any(newly_succeeded):
            success[idx[newly_succeeded]] = True
            active[idx[newly_succeeded]] = False

    return {"success": success, "steps": steps.astype(np.int32)}


def _collect_bc_dataset(rollouts, target_names, control_dim, max_samples_per_trajectory=None):
    state_batches = []
    action_batches = []
    used_names = []

    for name in target_names:
        rollout = rollouts.get(name)
        if rollout is None:
            continue
        _, X, U = rollout
        states, actions = _align_rollout_pairs(X, U, control_dim)
        if states.shape[0] == 0:
            continue
        if max_samples_per_trajectory is not None:
            cap = max(1, int(max_samples_per_trajectory))
            if states.shape[0] > cap:
                idx = np.linspace(0, states.shape[0] - 1, num=cap, dtype=int)
                states = states[idx]
                actions = actions[idx]
        state_batches.append(states)
        action_batches.append(actions)
        used_names.append(name)

    if not state_batches:
        raise RuntimeError("No valid state-action pairs found for BC training.")

    return (
        np.vstack(state_batches).astype(np.float32),
        np.vstack(action_batches).astype(np.float32),
        used_names,
    )


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(64, 128, 64)):
        super().__init__()
        layers = []
        in_dim = int(obs_dim)
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, int(hidden_dim)))
            layers.append(nn.ReLU())
            in_dim = int(hidden_dim)
        layers.append(nn.Linear(in_dim, int(act_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)


@dataclass
class BehaviorCloningController:
    trainer: BehaviorCloning
    obs_mean: np.ndarray
    obs_std: np.ndarray
    act_mean: np.ndarray
    act_std: np.ndarray
    angle_indices: tuple[int, ...]
    use_angle_features: bool
    u_min: np.ndarray | None = None
    u_max: np.ndarray | None = None
    numpy_layers: list[tuple[np.ndarray, np.ndarray]] | None = None

    def __post_init__(self):
        self.numpy_layers = self._extract_numpy_layers()

    def _extract_numpy_layers(self):
        net = getattr(self.trainer.policy, "net", None)
        if net is None:
            return None

        layers = []
        linear_weight = None
        linear_bias = None
        for module in net:
            if isinstance(module, nn.Linear):
                linear_weight = module.weight.detach().cpu().numpy().astype(np.float32)
                linear_bias = module.bias.detach().cpu().numpy().astype(np.float32)
            elif isinstance(module, nn.ReLU):
                if linear_weight is None or linear_bias is None:
                    return None
                layers.append((linear_weight, linear_bias))
                linear_weight = None
                linear_bias = None
            else:
                return None

        if linear_weight is not None and linear_bias is not None:
            layers.append((linear_weight, linear_bias))
        return layers or None

    def _forward_numpy(self, obs_norm):
        if not self.numpy_layers:
            return None
        x = np.asarray(obs_norm, dtype=np.float32)
        for idx, (weight, bias) in enumerate(self.numpy_layers):
            x = x @ weight.T + bias
            if idx + 1 < len(self.numpy_layers):
                x = np.maximum(x, 0.0).astype(np.float32)
        return x.astype(np.float32)

    def _encode(self, states):
        return _encode_states(
            states,
            angle_indices=self.angle_indices,
            use_angle_features=self.use_angle_features,
        )

    def _normalize_obs(self, obs):
        obs_features = np.asarray(self._encode(obs), dtype=np.float32).reshape(1, -1)
        return ((obs_features - self.obs_mean[None, :]) / self.obs_std[None, :]).astype(np.float32)

    def act(self, obs):
        obs_norm = self._normalize_obs(obs)
        action_norm = self._forward_numpy(obs_norm)
        if action_norm is None:
            action_norm = np.asarray(self.trainer.act(obs_norm[0]), dtype=np.float32).reshape(1, -1)
        action_norm = np.asarray(action_norm, dtype=np.float32).reshape(-1)
        action = action_norm * self.act_std + self.act_mean
        if self.u_min is not None:
            action = np.maximum(action, self.u_min)
        if self.u_max is not None:
            action = np.minimum(action, self.u_max)
        return action.astype(np.float32)

    def predict_actions(self, states):
        states = np.asarray(states, dtype=np.float32)
        features = np.asarray(self._encode(states), dtype=np.float32)
        features_norm = ((features - self.obs_mean[None, :]) / self.obs_std[None, :]).astype(np.float32)
        pred_norm = self._forward_numpy(features_norm)
        if pred_norm is None:
            with torch.no_grad():
                self.trainer.policy.eval()
                x = torch.tensor(features_norm, dtype=torch.float32, device=self.trainer.device)
                pred_norm = self.trainer.policy(x).cpu().numpy()
        pred = pred_norm * self.act_std[None, :] + self.act_mean[None, :]
        if self.u_min is not None:
            pred = np.maximum(pred, self.u_min[None, :])
        if self.u_max is not None:
            pred = np.minimum(pred, self.u_max[None, :])
        return pred.astype(np.float32)


def train_behavior_cloning_policy(
    spec: ChainPolicySpec,
    states,
    actions,
    hidden_sizes=(64, 128, 64),
    lr=2e-3,
    weight_decay=0.0,
    batch_size=256,
    epochs=200,
    seed=0,
    device="cpu",
    normalize_obs=True,
    normalize_actions=True,
    use_angle_features=False,
    u_min=None,
    u_max=None,
    verbose=False,
):
    states = _as_2d(states, spec.state_dim)
    actions = _as_2d(actions, spec.control_dim)

    state_features = np.asarray(
        _encode_states(
            states,
            angle_indices=spec.angle_indices,
            use_angle_features=use_angle_features,
        ),
        dtype=np.float32,
    )
    obs_mean, obs_std = _norm_stats(state_features, enabled=normalize_obs)
    act_mean, act_std = _norm_stats(actions, enabled=normalize_actions)

    train_states = ((state_features - obs_mean[None, :]) / obs_std[None, :]).astype(np.float32)
    train_actions = ((actions - act_mean[None, :]) / act_std[None, :]).astype(np.float32)

    _set_random_seed(seed)

    policy_class = lambda obs_dim, act_dim: MLPPolicy(  # noqa: E731
        obs_dim,
        act_dim,
        hidden_sizes=hidden_sizes,
    )
    trainer = BehaviorCloning(
        policy_class=policy_class,
        obs_dim=train_states.shape[1],
        act_dim=spec.control_dim,
        lr=lr,
        weight_decay=weight_decay,
        device=_resolve_device(device),
    )
    loss_history = trainer.train(
        states=train_states,
        actions=train_actions,
        batch_size=batch_size,
        epochs=epochs,
        shuffle=True,
        verbose=verbose,
    )

    controller = BehaviorCloningController(
        trainer=trainer,
        obs_mean=obs_mean.astype(np.float32),
        obs_std=obs_std.astype(np.float32),
        act_mean=act_mean.astype(np.float32),
        act_std=act_std.astype(np.float32),
        angle_indices=tuple(int(idx) for idx in spec.angle_indices),
        use_angle_features=bool(use_angle_features),
        u_min=None if u_min is None else np.asarray(u_min, dtype=np.float32).reshape(spec.control_dim),
        u_max=None if u_max is None else np.asarray(u_max, dtype=np.float32).reshape(spec.control_dim),
    )

    pred_actions = controller.predict_actions(states)
    train_mse = float(np.mean((pred_actions - actions) ** 2))
    info = {
        "loss_history": [float(loss) for loss in loss_history],
        "final_loss": float(loss_history[-1]) if loss_history else np.nan,
        "train_mse": train_mse,
        "num_samples": int(states.shape[0]),
        "feature_dim": int(train_states.shape[1]),
    }
    return controller, info


def _run_bc_episode(spec: ChainPolicySpec, env, controller, x0):
    obs, _ = env.reset(options={"x0": np.asarray(x0, dtype=np.float32)})
    has_succeeded = spec.check_success(obs)
    done = False
    total_steps = 0

    while not done:
        action = controller.act(obs)
        total_steps += 1
        obs, _, terminated, truncated, _ = env.step(action)
        if spec.check_success(obs):
            has_succeeded = True
        done = bool(terminated or truncated)

    return {"success": bool(has_succeeded), "steps": int(total_steps)}


def _run_bc_episode_fast(spec: ChainPolicySpec, controller, x0, dt, episode_seconds):
    obs = np.asarray(x0, dtype=np.float32).reshape(spec.state_dim)
    has_succeeded = spec.check_success(obs)
    total_steps = 0
    horizon_steps = max(int(np.round(float(episode_seconds) / float(dt))), 1)

    for _ in range(horizon_steps):
        action = controller.act(obs)
        total_steps += 1
        obs = _rk4_step(spec.f_continuous_fn, obs, action, dt)
        if spec.check_success(obs):
            has_succeeded = True
            break

    return {"success": bool(has_succeeded), "steps": int(total_steps)}


def greedy_rank_behavior_cloning_trajectories(
    spec: ChainPolicySpec,
    save_dir,
    candidate_names,
    init_states,
    npz_filename=None,
    dt=0.02,
    episode_seconds=25.0,
    bc_kwargs=None,
    max_drop=0.05,
    stop_on_large_drop=True,
    max_trajectories=None,
    use_fast_rollout=False,
    verbose=True,
):
    npz_filename = spec.rollout_npz_filename if npz_filename is None else npz_filename
    bc_kwargs = {} if bc_kwargs is None else dict(bc_kwargs)
    rollouts = load_saved_rollouts(save_dir, candidate_names, npz_filename=npz_filename)

    env_template = spec.make_env(dt=dt, episode_seconds=episode_seconds)
    u_min = np.asarray(env_template.action_space.low, dtype=np.float32).reshape(spec.control_dim)
    u_max = np.asarray(env_template.action_space.high, dtype=np.float32).reshape(spec.control_dim)

    remaining = [name for name in candidate_names if name in rollouts]
    selected_names = []
    records = []
    prev_rate = 0.0
    limit = None if max_trajectories is None else int(max_trajectories)

    while remaining and (limit is None or len(selected_names) < limit):
        best_row = None
        max_samples_per_trajectory = bc_kwargs.get("max_samples_per_trajectory", None)
        for name in remaining:
            subset_names = selected_names + [name]
            states, actions, used_names = _collect_bc_dataset(
                rollouts=rollouts,
                target_names=subset_names,
                control_dim=spec.control_dim,
                max_samples_per_trajectory=max_samples_per_trajectory,
            )
            controller, train_info = train_behavior_cloning_policy(
                spec=spec,
                states=states,
                actions=actions,
                u_min=u_min,
                u_max=u_max,
                seed=int(bc_kwargs.get("seed", 0)) + len(subset_names),
                **{
                    key: value
                    for key, value in bc_kwargs.items()
                    if key not in {"seed", "max_samples_per_trajectory"}
                },
            )
            env = None if use_fast_rollout else spec.make_env(dt=dt, episode_seconds=episode_seconds)
            success_count = 0
            steps = []
            if use_fast_rollout and spec.name == "single_pendulum":
                batch_episode = _run_bc_episodes_fast_single_batch(
                    spec=spec,
                    controller=controller,
                    init_states=init_states,
                    dt=dt,
                    episode_seconds=episode_seconds,
                )
                success_count = int(np.sum(batch_episode["success"]))
                steps = [float(step) for step in batch_episode["steps"]]
            elif use_fast_rollout and spec.name == "double_pendulum":
                batch_episode = _run_bc_episodes_fast_double_batch(
                    spec=spec,
                    controller=controller,
                    init_states=init_states,
                    dt=dt,
                    episode_seconds=episode_seconds,
                )
                success_count = int(np.sum(batch_episode["success"]))
                steps = [float(step) for step in batch_episode["steps"]]
            else:
                for x0 in init_states:
                    if use_fast_rollout:
                        episode = _run_bc_episode_fast(
                            spec=spec,
                            controller=controller,
                            x0=x0,
                            dt=dt,
                            episode_seconds=episode_seconds,
                        )
                    else:
                        episode = _run_bc_episode(spec=spec, env=env, controller=controller, x0=x0)
                    if episode["success"]:
                        success_count += 1
                    steps.append(float(episode["steps"]))

            rate = success_count / len(init_states)
            drop = max(prev_rate - rate, 0.0)
            row = {
                "name": name,
                "success_rate": float(rate),
                "drop_vs_prev": float(drop),
                "num_samples": int(train_info["num_samples"]),
                "train_mse": float(train_info["train_mse"]),
                "final_loss": float(train_info["final_loss"]),
                "mean_steps": float(np.mean(steps)),
                "used_names": list(used_names),
            }
            accept = drop <= float(max_drop) + 1e-9
            score = (
                int(accept),
                row["success_rate"],
                -row["drop_vs_prev"],
                row["num_samples"],
                -row["train_mse"],
                -row["final_loss"],
            )
            if best_row is None or score > best_row["score"]:
                best_row = {**row, "score": score}

        if best_row is None:
            break
        if stop_on_large_drop and best_row["drop_vs_prev"] > float(max_drop) + 1e-9:
            if verbose:
                print(
                    f"[GreedyRank-BC:{spec.name}] stopping before adding {best_row['name']} "
                    f"because success would drop by {best_row['drop_vs_prev']:.3f}"
                )
            break

        selected_names.append(best_row["name"])
        remaining.remove(best_row["name"])
        prev_rate = float(best_row["success_rate"])
        records.append({key: value for key, value in best_row.items() if key != "score"})

        if verbose:
            print(
                f"[GreedyRank-BC:{spec.name}] k={len(selected_names)} add {best_row['name']} -> "
                f"rate={best_row['success_rate']:.3f}, drop={best_row['drop_vs_prev']:.3f}, "
                f"samples={best_row['num_samples']}"
            )

    return selected_names, records


def select_behavior_cloning_trajectory_subset(
    spec: ChainPolicySpec,
    save_dir,
    ordered_names,
    init_states,
    npz_filename=None,
    dt=0.02,
    episode_seconds=25.0,
    bc_kwargs=None,
    max_drop=0.05,
    use_fast_rollout=False,
    verbose=True,
):
    npz_filename = spec.rollout_npz_filename if npz_filename is None else npz_filename
    bc_kwargs = {} if bc_kwargs is None else dict(bc_kwargs)
    rollouts = load_saved_rollouts(save_dir, ordered_names, npz_filename=npz_filename)

    env_template = spec.make_env(dt=dt, episode_seconds=episode_seconds)
    u_min = np.asarray(env_template.action_space.low, dtype=np.float32).reshape(spec.control_dim)
    u_max = np.asarray(env_template.action_space.high, dtype=np.float32).reshape(spec.control_dim)

    selected_names = []
    records = []
    prev_rate = 0.0
    max_samples_per_trajectory = bc_kwargs.get("max_samples_per_trajectory", None)

    for name in ordered_names:
        if name not in rollouts:
            continue
        candidate_names = selected_names + [name]
        states, actions, used_names = _collect_bc_dataset(
            rollouts=rollouts,
            target_names=candidate_names,
            control_dim=spec.control_dim,
            max_samples_per_trajectory=max_samples_per_trajectory,
        )
        controller, train_info = train_behavior_cloning_policy(
            spec=spec,
            states=states,
            actions=actions,
            u_min=u_min,
            u_max=u_max,
            seed=int(bc_kwargs.get("seed", 0)) + len(candidate_names),
            **{
                key: value
                for key, value in bc_kwargs.items()
                if key not in {"seed", "max_samples_per_trajectory"}
            },
        )
        env = None if use_fast_rollout else spec.make_env(dt=dt, episode_seconds=episode_seconds)
        success_count = 0
        if use_fast_rollout and spec.name == "single_pendulum":
            batch_episode = _run_bc_episodes_fast_single_batch(
                spec=spec,
                controller=controller,
                init_states=init_states,
                dt=dt,
                episode_seconds=episode_seconds,
            )
            success_count = int(np.sum(batch_episode["success"]))
        elif use_fast_rollout and spec.name == "double_pendulum":
            batch_episode = _run_bc_episodes_fast_double_batch(
                spec=spec,
                controller=controller,
                init_states=init_states,
                dt=dt,
                episode_seconds=episode_seconds,
            )
            success_count = int(np.sum(batch_episode["success"]))
        else:
            for x0 in init_states:
                if use_fast_rollout:
                    episode = _run_bc_episode_fast(
                        spec=spec,
                        controller=controller,
                        x0=x0,
                        dt=dt,
                        episode_seconds=episode_seconds,
                    )
                else:
                    episode = _run_bc_episode(spec=spec, env=env, controller=controller, x0=x0)
                if episode["success"]:
                    success_count += 1

        rate = success_count / len(init_states)
        drop = max(prev_rate - rate, 0.0)
        accepted = drop <= float(max_drop) + 1e-9
        records.append(
            {
                "name": name,
                "candidate_rate": float(rate),
                "drop_vs_prev": float(drop),
                "accepted": bool(accepted),
                "selected_count_if_accepted": int(len(candidate_names)),
                "num_samples": int(train_info["num_samples"]),
                "used_names": list(used_names),
            }
        )
        if accepted:
            selected_names.append(name)
            prev_rate = float(rate)
        if verbose:
            print(
                f"[SubsetSelect-BC:{spec.name}] {name}: rate={rate:.3f}, drop={drop:.3f}, "
                f"{'ACCEPT' if accepted else 'REJECT'}"
            )

    return selected_names, records


def evaluate_incremental_behavior_cloning(
    spec: ChainPolicySpec,
    save_dir,
    target_names,
    init_states,
    npz_filename=None,
    dt=0.02,
    episode_seconds=25.0,
    bc_kwargs=None,
    result_filename=None,
    use_fast_rollout=False,
    plot=True,
    verbose=True,
):
    npz_filename = spec.rollout_npz_filename if npz_filename is None else npz_filename
    if result_filename is None:
        result_filename = f"incremental_vanilla_bc_success_rates_{spec.name}.npz"

    bc_kwargs = {} if bc_kwargs is None else dict(bc_kwargs)
    overrides_by_k = bc_kwargs.pop("overrides_by_k", None)
    if overrides_by_k is None:
        overrides_by_k = {}
    else:
        overrides_by_k = {int(k): dict(v) for k, v in dict(overrides_by_k).items()}
    rollouts = load_saved_rollouts(save_dir, target_names, npz_filename=npz_filename)

    env_template = spec.make_env(dt=dt, episode_seconds=episode_seconds)
    u_min = np.asarray(env_template.action_space.low, dtype=np.float32).reshape(spec.control_dim)
    u_max = np.asarray(env_template.action_space.high, dtype=np.float32).reshape(spec.control_dim)

    n_traj_list = []
    success_rates = []
    diag = []

    if verbose:
        print(f"\n[Eval-BC:{spec.name}] Running incremental vanilla BC evaluation")
        print("bc_kwargs =", bc_kwargs)

    for k in range(1, len(target_names) + 1):
        current_bc_kwargs = dict(bc_kwargs)
        if k in overrides_by_k:
            current_bc_kwargs.update(overrides_by_k[k])

        max_samples_per_trajectory = current_bc_kwargs.get("max_samples_per_trajectory", None)
        current_names = target_names[:k]
        states, actions, used_names = _collect_bc_dataset(
            rollouts=rollouts,
            target_names=current_names,
            control_dim=spec.control_dim,
            max_samples_per_trajectory=max_samples_per_trajectory,
        )
        controller, train_info = train_behavior_cloning_policy(
            spec=spec,
            states=states,
            actions=actions,
            u_min=u_min,
            u_max=u_max,
            seed=int(current_bc_kwargs.get("seed", 0)) + k,
            **{
                key: value
                for key, value in current_bc_kwargs.items()
                if key not in {"seed", "max_samples_per_trajectory"}
            },
        )

        env = None if use_fast_rollout else spec.make_env(dt=dt, episode_seconds=episode_seconds)
        success_count = 0
        steps = []
        if use_fast_rollout and spec.name == "single_pendulum":
            batch_episode = _run_bc_episodes_fast_single_batch(
                spec=spec,
                controller=controller,
                init_states=init_states,
                dt=dt,
                episode_seconds=episode_seconds,
            )
            success_count = int(np.sum(batch_episode["success"]))
            steps = [float(step) for step in batch_episode["steps"]]
        elif use_fast_rollout and spec.name == "double_pendulum":
            batch_episode = _run_bc_episodes_fast_double_batch(
                spec=spec,
                controller=controller,
                init_states=init_states,
                dt=dt,
                episode_seconds=episode_seconds,
            )
            success_count = int(np.sum(batch_episode["success"]))
            steps = [float(step) for step in batch_episode["steps"]]
        else:
            for x0 in init_states:
                if use_fast_rollout:
                    episode = _run_bc_episode_fast(
                        spec=spec,
                        controller=controller,
                        x0=x0,
                        dt=dt,
                        episode_seconds=episode_seconds,
                    )
                else:
                    episode = _run_bc_episode(spec=spec, env=env, controller=controller, x0=x0)
                if episode["success"]:
                    success_count += 1
                steps.append(float(episode["steps"]))

        rate = success_count / len(init_states)
        mean_steps = float(np.mean(steps))
        n_traj_list.append(k)
        success_rates.append(rate)
        diag.append(
            (
                float(train_info["train_mse"]),
                float(train_info["final_loss"]),
                float(mean_steps),
                float(train_info["num_samples"]),
                float(train_info["feature_dim"]),
            )
        )

        if verbose:
            print(
                f"[Eval-BC:{spec.name}] k={k} ({len(used_names)} trajs, {train_info['num_samples']} samples) "
                f"-> Success Rate: {rate:.2%} | Train MSE: {train_info['train_mse']:.3e} | "
                f"Final loss: {train_info['final_loss']:.3e}"
            )

    result_save_path = _expanded_path(save_dir) / result_filename
    np.savez(
        result_save_path,
        n_traj_list=np.asarray(n_traj_list, dtype=int),
        success_rates=np.asarray(success_rates, dtype=float),
        diag=np.asarray(diag, dtype=float),
    )
    if verbose:
        print(f"\n[Eval-BC:{spec.name}] Results saved to {result_save_path}")

    if plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.plot(n_traj_list, success_rates, marker="o", linewidth=2, label="Vanilla BC")
        plt.xlabel("Number of Expert Trajectories")
        plt.ylabel("Success Rate")
        plt.title(f"{spec.name}: vanilla BC success rate vs. dataset size")
        plt.ylim(-0.05, 1.05)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return n_traj_list, success_rates, diag
