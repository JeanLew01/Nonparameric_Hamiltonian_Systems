from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from baseline.behavior_cloning import BehaviorCloning
from utils.control_chain import ChainPolicySpec, load_saved_rollouts

__all__ = [
    "BehaviorCloningController",
    "MLPPolicy",
    "evaluate_incremental_behavior_cloning",
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
    if not use_angle_features or len(angle_indices) == 0:
        return X[0] if was_1d else X

    angle_set = set(int(idx) for idx in angle_indices)
    features = []
    for idx in range(X.shape[1]):
        column = X[:, idx : idx + 1]
        if idx in angle_set:
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


def _collect_bc_dataset(rollouts, target_names, control_dim):
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
        action_norm = np.asarray(self.trainer.act(obs_norm[0]), dtype=np.float32).reshape(-1)
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
    plot=True,
    verbose=True,
):
    npz_filename = spec.rollout_npz_filename if npz_filename is None else npz_filename
    if result_filename is None:
        result_filename = f"incremental_vanilla_bc_success_rates_{spec.name}.npz"

    bc_kwargs = {} if bc_kwargs is None else dict(bc_kwargs)
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
        current_names = target_names[:k]
        states, actions, used_names = _collect_bc_dataset(
            rollouts=rollouts,
            target_names=current_names,
            control_dim=spec.control_dim,
        )
        controller, train_info = train_behavior_cloning_policy(
            spec=spec,
            states=states,
            actions=actions,
            u_min=u_min,
            u_max=u_max,
            seed=int(bc_kwargs.get("seed", 0)) + k,
            **{key: value for key, value in bc_kwargs.items() if key != "seed"},
        )

        env = spec.make_env(dt=dt, episode_seconds=episode_seconds)
        success_count = 0
        steps = []
        for x0 in init_states:
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
