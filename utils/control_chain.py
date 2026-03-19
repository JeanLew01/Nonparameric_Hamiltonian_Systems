from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dynamics.Dynamics_single_pendulum import (
    I as SINGLE_I,
    damping as SINGLE_DAMPING,
    g as SINGLE_G,
    l as SINGLE_L,
    m as SINGLE_M,
    SinglePendulumEnv,
)
from dynamics.Dynamics_spring_mass import (
    damping as SPRING_DAMPING,
    k_spring as SPRING_K,
    m as SPRING_M,
    SpringMassEnv,
)

__all__ = [
    "AffineFeedbackBackupController",
    "ChainPolicySpec",
    "MPCBackupController",
    "SpringMassPDBackupController",
    "TubeController",
    "angle_normalize",
    "make_single_pendulum_spec",
    "make_single_pendulum_mpc_backup",
    "make_spring_mass_spec",
    "make_spring_mass_pd_backup",
    "build_control_alphabet_from_saved_rollouts",
    "evaluate_incremental_chain_policy",
    "load_control_alphabet",
    "load_saved_rollouts",
    "save_rollout_metrics",
]


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def _expanded_path(path_like) -> Path:
    return Path(os.path.expanduser(str(path_like)))


def _reset_history_if_available(obj) -> None:
    if hasattr(obj, "reset_history"):
        try:
            obj.reset_history()
        except TypeError:
            pass


def _ensure_control_shape(U, control_dim):
    U = np.asarray(U, dtype=float)
    if U.ndim == 1:
        U = U.reshape(-1, control_dim)
    if U.ndim != 2:
        raise ValueError(f"Control array must be 2D, got shape {U.shape}")
    if U.shape[1] < control_dim:
        pad = np.zeros((U.shape[0], control_dim - U.shape[1]), dtype=float)
        U = np.hstack([U, pad])
    return U[:, :control_dim]


def _state_error_with_angles(x, target, angle_indices):
    delta = np.asarray(x, dtype=np.float32) - np.asarray(target, dtype=np.float32)
    for idx in angle_indices:
        delta[idx] = angle_normalize(delta[idx])
    return delta.astype(np.float32)


def _sample_single_initial_states(
    num_inits=50,
    seed=0,
    theta_low=0.0,
    theta_high=2.0 * np.pi,
    theta_dot_low=0.0,
    theta_dot_high=0.0,
):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(low=theta_low, high=theta_high, size=(num_inits, 1))
    theta_dot = rng.uniform(low=theta_dot_low, high=theta_dot_high, size=(num_inits, 1))
    theta = angle_normalize(theta)
    return np.hstack([theta, theta_dot]).astype(np.float32)


def _sample_spring_initial_states(
    num_inits=50,
    seed=0,
    x_low=-2.0,
    x_high=2.0,
    x_dot_low=-2.0,
    x_dot_high=2.0,
):
    rng = np.random.default_rng(seed)
    x_pos = rng.uniform(low=x_low, high=x_high, size=(num_inits, 1))
    x_dot = rng.uniform(low=x_dot_low, high=x_dot_high, size=(num_inits, 1))
    return np.hstack([x_pos, x_dot]).astype(np.float32)


def _single_state_error(x, target):
    return _state_error_with_angles(x, target, angle_indices=(0,))


def _spring_state_error(x, target):
    return (np.asarray(x, dtype=np.float32) - np.asarray(target, dtype=np.float32)).astype(
        np.float32
    )


def _single_f_continuous(x, u):
    theta, theta_dot = np.asarray(x, dtype=float).reshape(2,)
    tau = np.asarray(u, dtype=float).reshape(-1)[0]
    theta_dd = (tau - SINGLE_M * SINGLE_G * SINGLE_L * np.sin(theta) - SINGLE_DAMPING * theta_dot) / SINGLE_I
    return np.array([theta_dot, theta_dd], dtype=float)


def _spring_f_continuous(x, u):
    x_pos, x_dot = np.asarray(x, dtype=float).reshape(2,)
    force = np.asarray(u, dtype=float).reshape(-1)[0]
    x_ddot = (force - SPRING_K * x_pos - SPRING_DAMPING * x_dot) / SPRING_M
    return np.array([x_dot, x_ddot], dtype=float)


def _make_single_energy_fn(target_state):
    target_state = np.asarray(target_state, dtype=float).reshape(2,)
    target_energy = 0.5 * SINGLE_I * target_state[1] ** 2 + SINGLE_M * SINGLE_G * SINGLE_L * (
        1.0 - np.cos(target_state[0])
    )

    def energy_from_states(X):
        X = np.asarray(X, dtype=float)
        theta = X[:, 0]
        theta_dot = X[:, 1]
        energy = 0.5 * SINGLE_I * theta_dot**2 + SINGLE_M * SINGLE_G * SINGLE_L * (
            1.0 - np.cos(theta)
        )
        return energy - target_energy

    return energy_from_states


def _single_lh_from_states(X):
    X = np.asarray(X, dtype=float)
    theta = X[:, 0]
    theta_dot = X[:, 1]
    dH_dtheta = SINGLE_M * SINGLE_G * SINGLE_L * np.sin(theta)
    dH_dtheta_dot = SINGLE_I * theta_dot
    return np.sqrt(dH_dtheta**2 + dH_dtheta_dot**2)


def _make_spring_energy_fn(target_state):
    target_state = np.asarray(target_state, dtype=float).reshape(2,)
    target_energy = 0.5 * SPRING_M * target_state[1] ** 2 + 0.5 * SPRING_K * target_state[0] ** 2

    def energy_from_states(X):
        X = np.asarray(X, dtype=float)
        x_pos = X[:, 0]
        x_dot = X[:, 1]
        energy = 0.5 * SPRING_M * x_dot**2 + 0.5 * SPRING_K * x_pos**2
        return energy - target_energy

    return energy_from_states


def _spring_lh_from_states(X):
    X = np.asarray(X, dtype=float)
    x_pos = X[:, 0]
    x_dot = X[:, 1]
    dH_dx = SPRING_K * x_pos
    dH_dx_dot = SPRING_M * x_dot
    return np.sqrt(dH_dx**2 + dH_dx_dot**2)


def _make_single_env_factory(target_state, success_tol):
    target_state = np.asarray(target_state, dtype=float).reshape(2,)
    success_tol = np.asarray(success_tol, dtype=float).reshape(2,)

    def factory(dt=0.02, episode_seconds=25.0):
        return SinglePendulumEnv(
            dt=dt,
            episode_seconds=episode_seconds,
            x_ref=target_state,
            success_tol=success_tol,
            default_x0=target_state,
            terminal_bonus=50.0,
        )

    return factory


def _make_spring_env_factory(target_state, success_tol):
    target_state = np.asarray(target_state, dtype=float).reshape(2,)
    success_tol = np.asarray(success_tol, dtype=float).reshape(2,)

    def factory(dt=0.02, episode_seconds=25.0):
        return SpringMassEnv(
            dt=dt,
            episode_seconds=episode_seconds,
            x_ref=target_state,
            success_tol=success_tol,
            default_x0=target_state,
            terminal_bonus=50.0,
        )

    return factory


@dataclass(frozen=True)
class ChainPolicySpec:
    name: str
    state_dim: int
    control_dim: int
    angle_indices: tuple[int, ...]
    target_state: np.ndarray
    success_tol: np.ndarray
    rollout_npz_filename: str
    alphabet_filename: str
    success_rate_filename: str
    sample_initial_states_fn: Callable
    state_error_fn: Callable
    f_continuous_fn: Callable
    energy_from_states_fn: Callable
    lh_from_states_fn: Callable
    env_factory_fn: Callable

    def sample_initial_states(self, num_inits=50, seed=0, **kwargs):
        return self.sample_initial_states_fn(num_inits=num_inits, seed=seed, **kwargs)

    def state_error(self, x, target=None):
        target = self.target_state if target is None else target
        return self.state_error_fn(x, target)

    def check_success(self, obs, target=None, tol=None):
        target = self.target_state if target is None else target
        tol = self.success_tol if tol is None else tol
        delta = self.state_error(obs, target=target)
        return bool(np.all(np.abs(delta) <= np.asarray(tol, dtype=np.float32)))

    def compute_energy(self, X):
        return np.asarray(self.energy_from_states_fn(X), dtype=float).reshape(-1)

    def compute_lh(self, X):
        return np.asarray(self.lh_from_states_fn(X), dtype=float).reshape(-1)

    def make_env(self, dt=0.02, episode_seconds=25.0):
        return self.env_factory_fn(dt=dt, episode_seconds=episode_seconds)


class AffineFeedbackBackupController:
    def __init__(
        self,
        gain_matrix,
        bias=None,
        u_min=None,
        u_max=None,
        angle_indices=(),
        target_state=None,
    ):
        self.K = np.asarray(gain_matrix, dtype=np.float32)
        if self.K.ndim == 1:
            self.K = self.K.reshape(1, -1)
        self.control_dim = self.K.shape[0]
        self.state_dim = self.K.shape[1]
        self.bias = (
            np.zeros(self.control_dim, dtype=np.float32)
            if bias is None
            else np.asarray(bias, dtype=np.float32).reshape(self.control_dim)
        )
        self.u_min = None if u_min is None else np.asarray(u_min, dtype=np.float32).reshape(self.control_dim)
        self.u_max = None if u_max is None else np.asarray(u_max, dtype=np.float32).reshape(self.control_dim)
        self.angle_indices = tuple(angle_indices)
        self.target_state = None if target_state is None else np.asarray(target_state, dtype=np.float32).reshape(self.state_dim)

    def reset_episode(self, x0=None):
        return None

    def get_action(self, x_current):
        x = np.asarray(x_current, dtype=np.float32).reshape(self.state_dim)
        if self.target_state is None:
            error = x
        else:
            error = _state_error_with_angles(x, self.target_state, self.angle_indices)
        u = self.bias - self.K @ error
        if self.u_min is not None:
            u = np.maximum(u, self.u_min)
        if self.u_max is not None:
            u = np.minimum(u, self.u_max)
        return np.asarray(u, dtype=np.float32).reshape(self.control_dim)


class SpringMassPDBackupController(AffineFeedbackBackupController):
    def __init__(self, kp=8.0, kd=4.0, u_min=-50.0, u_max=50.0, target_state=None):
        gain_matrix = np.array([[kp, kd]], dtype=np.float32)
        super().__init__(
            gain_matrix=gain_matrix,
            bias=np.zeros(1, dtype=np.float32),
            u_min=np.array([u_min], dtype=np.float32),
            u_max=np.array([u_max], dtype=np.float32),
            angle_indices=(),
            target_state=np.array([0.0, 0.0], dtype=np.float32) if target_state is None else target_state,
        )


class MPCBackupController:
    def __init__(self, mpc, control_dim, default_u=None):
        self.mpc = mpc
        self.control_dim = int(control_dim)
        self.default_u = (
            np.zeros(self.control_dim, dtype=np.float32)
            if default_u is None
            else np.asarray(default_u, dtype=np.float32).reshape(self.control_dim)
        )

    def reset_episode(self, x0=None):
        if x0 is not None and hasattr(self.mpc, "x0"):
            self.mpc.x0 = np.asarray(x0, dtype=float).reshape(-1)
        _reset_history_if_available(self.mpc)
        if hasattr(self.mpc, "set_initial_guess"):
            self.mpc.set_initial_guess()

    def get_action(self, x_current):
        try:
            u = np.asarray(self.mpc.make_step(np.asarray(x_current, dtype=float)), dtype=float).reshape(-1)
        except Exception:
            u = self.default_u.astype(float)
        if u.size < self.control_dim:
            pad = np.zeros(self.control_dim - u.size, dtype=float)
            u = np.hstack([u, pad])
        return np.asarray(u[: self.control_dim], dtype=np.float32)


def make_spring_mass_pd_backup(kp=8.0, kd=4.0, target_state=None):
    return SpringMassPDBackupController(
        kp=kp,
        kd=kd,
        target_state=np.array([0.0, 0.0], dtype=np.float32) if target_state is None else target_state,
    )


def make_single_pendulum_mpc_backup(
    dt=0.02,
    n_horizon=120,
    target_state=None,
):
    from utils.mpc import SinglePendulumParams, build_single_pendulum_mpc

    target_state = (
        np.array([np.pi, 0.0], dtype=np.float32)
        if target_state is None
        else np.asarray(target_state, dtype=np.float32).reshape(2,)
    )
    cfg = SinglePendulumParams(dt=dt, n_horizon=n_horizon, x_ref=np.asarray(target_state, dtype=float))
    _, mpc, _ = build_single_pendulum_mpc(cfg)

    if hasattr(mpc, "settings") and hasattr(mpc.settings, "supress_ipopt_output"):
        try:
            mpc.settings.supress_ipopt_output()
        except TypeError:
            pass

    return MPCBackupController(mpc=mpc, control_dim=1)


def make_single_pendulum_spec(
    target_state=None,
    success_tol=None,
):
    target_state = (
        np.array([np.pi, 0.0], dtype=np.float32)
        if target_state is None
        else np.asarray(target_state, dtype=np.float32).reshape(2,)
    )
    success_tol = (
        np.array([0.12, 0.20], dtype=np.float32)
        if success_tol is None
        else np.asarray(success_tol, dtype=np.float32).reshape(2,)
    )
    return ChainPolicySpec(
        name="single_pendulum",
        state_dim=2,
        control_dim=1,
        angle_indices=(0,),
        target_state=target_state,
        success_tol=success_tol,
        rollout_npz_filename="all_rollouts_single_pendulum.npz",
        alphabet_filename="control_alphabet_single_pendulum.pkl",
        success_rate_filename="incremental_tube_success_rates_single_pendulum.npz",
        sample_initial_states_fn=_sample_single_initial_states,
        state_error_fn=_single_state_error,
        f_continuous_fn=_single_f_continuous,
        energy_from_states_fn=_make_single_energy_fn(target_state),
        lh_from_states_fn=_single_lh_from_states,
        env_factory_fn=_make_single_env_factory(target_state, success_tol),
    )


def make_spring_mass_spec(
    target_state=None,
    success_tol=None,
):
    target_state = (
        np.array([0.0, 0.0], dtype=np.float32)
        if target_state is None
        else np.asarray(target_state, dtype=np.float32).reshape(2,)
    )
    success_tol = (
        np.array([0.08, 0.12], dtype=np.float32)
        if success_tol is None
        else np.asarray(success_tol, dtype=np.float32).reshape(2,)
    )
    return ChainPolicySpec(
        name="spring_mass",
        state_dim=2,
        control_dim=1,
        angle_indices=(),
        target_state=target_state,
        success_tol=success_tol,
        rollout_npz_filename="all_rollouts_spring_mass.npz",
        alphabet_filename="control_alphabet_spring_mass.pkl",
        success_rate_filename="incremental_tube_success_rates_spring_mass.npz",
        sample_initial_states_fn=_sample_spring_initial_states,
        state_error_fn=_spring_state_error,
        f_continuous_fn=_spring_f_continuous,
        energy_from_states_fn=_make_spring_energy_fn(target_state),
        lh_from_states_fn=_spring_lh_from_states,
        env_factory_fn=_make_spring_env_factory(target_state, success_tol),
    )


def load_saved_rollouts(save_dir, names, npz_filename):
    save_path = _expanded_path(save_dir)
    npz_path = save_path / npz_filename
    if not npz_path.exists():
        raise FileNotFoundError(f"Cannot find rollouts npz: {npz_path}")

    data_rollouts = np.load(npz_path, allow_pickle=True)
    loaded = {}
    for name in names:
        t_key = f"T_{name}"
        x_key = f"X_{name}"
        u_key = f"U_{name}"
        if t_key not in data_rollouts or x_key not in data_rollouts or u_key not in data_rollouts:
            continue
        loaded[name] = (
            np.asarray(data_rollouts[t_key], dtype=float),
            np.asarray(data_rollouts[x_key], dtype=float),
            np.asarray(data_rollouts[u_key], dtype=float),
        )
    return loaded


def get_dynamics_l_numerical(x, u, f_continuous_fn, eps=1e-5):
    x = np.asarray(x, dtype=float).copy().reshape(-1)
    u = np.asarray(u, dtype=float).copy().reshape(-1)
    n = x.size
    jac = np.zeros((n, n), dtype=float)

    for idx in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[idx] += eps
        x_minus[idx] -= eps
        f_plus = np.asarray(f_continuous_fn(x_plus, u), dtype=float).reshape(-1)
        f_minus = np.asarray(f_continuous_fn(x_minus, u), dtype=float).reshape(-1)
        jac[:, idx] = (f_plus - f_minus) / (2.0 * eps)
    return float(np.linalg.norm(jac, 2))


def save_rollout_metrics(
    spec: ChainPolicySpec,
    save_dir,
    names,
    npz_filename=None,
    eps_L=1e-5,
    verbose=True,
):
    npz_filename = spec.rollout_npz_filename if npz_filename is None else npz_filename
    save_path = _expanded_path(save_dir)
    rollouts = load_saved_rollouts(save_path, names, npz_filename=npz_filename)

    metrics = {}
    for name in names:
        rollout = rollouts.get(name)
        if rollout is None:
            continue
        _, X, U = rollout
        H = spec.compute_energy(X)
        LH = spec.compute_lh(X)

        control_count = max(X.shape[0] - 1, 0)
        U = _ensure_control_shape(U, spec.control_dim)
        l_dyn = np.zeros(control_count, dtype=float)
        for idx in range(control_count):
            l_dyn[idx] = get_dynamics_l_numerical(
                X[idx], U[idx], spec.f_continuous_fn, eps=eps_L
            )

        np.save(save_path / f"H_total_{name}.npy", H)
        np.save(save_path / f"LH_{name}.npy", LH)
        np.save(save_path / f"Ldyn_{name}.npy", l_dyn)

        metrics[name] = {"H": H, "LH": LH, "Ldyn": l_dyn}
        if verbose:
            print(
                f"{name} -> H[max|abs]={float(np.max(np.abs(H))):.3e}, "
                f"LH[max]={float(np.max(LH)):.3e}, "
                f"Ldyn[max]={float(np.max(l_dyn)) if l_dyn.size else 0.0:.3e}"
            )
    return metrics


def generate_control_alphabet_from_rollout(
    spec: ChainPolicySpec,
    T,
    X,
    U,
    H_array,
    LH_array,
    rho=0.99,
    H_star=0.0,
    eta=0.0,
    r_min=1e-6,
    max_lookahead=200,
    eps_L=1e-5,
):
    T = np.asarray(T, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    H_array = np.asarray(H_array, dtype=float).reshape(-1)
    LH_array = np.asarray(LH_array, dtype=float).reshape(-1)

    if T.ndim != 1 or T.size < 2:
        raise ValueError("T must be 1D with length >= 2.")
    if X.ndim != 2 or X.shape[1] != spec.state_dim:
        raise ValueError(f"X must be (N, {spec.state_dim}), got {X.shape}")

    n_eff = min(X.shape[0], H_array.size, LH_array.size)
    if n_eff < 2:
        return []

    X = X[:n_eff]
    H_array = H_array[:n_eff]
    LH_array = LH_array[:n_eff]
    T = T[:n_eff]

    dt = float(np.mean(np.diff(T))) if T.size >= 2 else 0.02
    n_ctrl_needed = n_eff - 1
    U = _ensure_control_shape(U, spec.control_dim)
    if U.shape[0] < n_ctrl_needed:
        pad = np.zeros((n_ctrl_needed - U.shape[0], spec.control_dim), dtype=float)
        U = np.vstack([U, pad])
    else:
        U = U[:n_ctrl_needed]

    l_dyns = np.zeros(n_ctrl_needed, dtype=float)
    for idx in range(n_ctrl_needed):
        l_dyns[idx] = get_dynamics_l_numerical(
            X[idx], U[idx], spec.f_continuous_fn, eps=eps_L
        )

    tubes = []
    i = 0
    while i < n_ctrl_needed:
        v_curr = abs(float(H_array[i]) - float(H_star))
        if v_curr <= 1e-10:
            i += 1
            continue

        best_r = -1.0
        best_steps = 0
        best_tau = 0.0

        lookahead = min(int(max_lookahead), n_ctrl_needed - i)
        for k in range(1, lookahead + 1):
            j = i + k
            tau = k * dt
            v_next = abs(float(H_array[j]) - float(H_star))
            numerator = rho * max(v_curr - eta, 0.0) - max(v_next - eta, 0.0)
            if numerator <= 0.0:
                continue

            l_h_interval = float(np.max(LH_array[i : j + 1]))
            l_dyn_interval = float(np.max(l_dyns[i:j]))
            denom = l_h_interval * (rho + np.exp(l_dyn_interval * tau))
            if denom <= 0.0 or not np.isfinite(denom):
                continue

            r_candidate = numerator / denom
            if np.isfinite(r_candidate) and r_candidate > best_r:
                best_r = float(r_candidate)
                best_steps = int(k)
                best_tau = float(tau)

        if best_r > r_min and best_steps > 0:
            end_idx = i + best_steps
            u_seq = U[i:end_idx].copy()
            tubes.append(
                {
                    "x_center": X[i].astype(np.float32),
                    "u_control": U[i].astype(np.float32),
                    "u_seq": u_seq.astype(np.float32),
                    "radius": float(best_r),
                    "tau": float(best_tau),
                    "start_idx": int(i),
                    "end_idx": int(end_idx),
                }
            )
            i = end_idx
        else:
            i += 1

    return tubes


def build_control_alphabet_from_saved_rollouts(
    spec: ChainPolicySpec,
    save_dir,
    names,
    npz_filename=None,
    alphabet_filename=None,
    rho=0.99,
    H_star=0.0,
    eta=0.0,
    r_min=1e-6,
    max_lookahead=200,
    eps_L=1e-5,
    verbose=True,
    save_metrics=True,
):
    npz_filename = spec.rollout_npz_filename if npz_filename is None else npz_filename
    alphabet_filename = spec.alphabet_filename if alphabet_filename is None else alphabet_filename
    save_path = _expanded_path(save_dir)
    rollouts = load_saved_rollouts(save_path, names, npz_filename=npz_filename)

    if save_metrics:
        save_rollout_metrics(
            spec=spec,
            save_dir=save_path,
            names=names,
            npz_filename=npz_filename,
            eps_L=eps_L,
            verbose=verbose,
        )

    all_alphabets = {}
    summary = []

    for name in names:
        if verbose:
            print(f"\n=== Generating Alphabet for {name} ===")

        rollout = rollouts.get(name)
        if rollout is None:
            if verbose:
                print(f"[WARN] Missing T/X/U for {name} in {npz_filename}.")
            continue

        T, X, U = rollout
        h_path = save_path / f"H_total_{name}.npy"
        lh_path = save_path / f"LH_{name}.npy"
        if not h_path.exists() or not lh_path.exists():
            if verbose:
                print(f"[WARN] Missing energy files for {name}.")
            continue

        H_array = np.load(h_path)
        LH_array = np.load(lh_path)

        tubes = generate_control_alphabet_from_rollout(
            spec=spec,
            T=T,
            X=X,
            U=U,
            H_array=H_array,
            LH_array=LH_array,
            rho=rho,
            H_star=H_star,
            eta=eta,
            r_min=r_min,
            max_lookahead=max_lookahead,
            eps_L=eps_L,
        )
        all_alphabets[name] = tubes

        radii = np.array([tube["radius"] for tube in tubes], dtype=float)
        row = {
            "name": name,
            "n_tubes": int(len(tubes)),
            "r_min": float(np.min(radii)) if radii.size else np.nan,
            "r_mean": float(np.mean(radii)) if radii.size else np.nan,
            "r_median": float(np.median(radii)) if radii.size else np.nan,
            "r_max": float(np.max(radii)) if radii.size else np.nan,
        }
        summary.append(row)
        if verbose:
            if radii.size == 0:
                print("-> Generated 0 tubes.")
            else:
                print(
                    f"-> Generated {len(tubes)} tubes | "
                    f"r[min/med/max] = {row['r_min']:.3e} / {row['r_median']:.3e} / {row['r_max']:.3e}"
                )

    alphabet_path = save_path / alphabet_filename
    with open(alphabet_path, "wb") as handle:
        pickle.dump(all_alphabets, handle)
    if verbose:
        print(f"\nAll alphabets saved to: {alphabet_path}")
    return all_alphabets, summary


def load_control_alphabet(save_dir, alphabet_filename):
    alphabet_path = _expanded_path(save_dir) / alphabet_filename
    if not alphabet_path.exists():
        raise FileNotFoundError(f"Cannot find file: {alphabet_path}")
    with open(alphabet_path, "rb") as handle:
        return pickle.load(handle)


class TubeController:
    def __init__(
        self,
        all_alphabets_dict,
        default_u=None,
        backup_controller=None,
        eps=1e-9,
        execute_full_sequence=False,
        angle_indices=(),
        distance_weights=None,
        radius_scale=1.0,
        enter_threshold=1.0,
        abort_threshold=1.5,
        outside_mode="nearest_control",
    ):
        self.eps = float(eps)
        self.execute_full_sequence = bool(execute_full_sequence)
        self.angle_indices = tuple(angle_indices)
        self.radius_scale = float(radius_scale)
        self.enter_threshold = float(enter_threshold)
        self.abort_threshold = float(abort_threshold)
        self.outside_mode = str(outside_mode)
        self.backup_controller = backup_controller

        centers = []
        radii = []
        u_controls = []
        u_seqs = []
        meta = []

        for name, tube_list in all_alphabets_dict.items():
            for tube in tube_list:
                x_center = np.asarray(tube["x_center"], dtype=np.float32).reshape(-1)
                radius = float(tube["radius"])
                u_seq = np.asarray(tube["u_seq"], dtype=np.float32)
                u_control = np.asarray(tube.get("u_control", u_seq[0]), dtype=np.float32).reshape(-1)

                if x_center.ndim != 1 or u_seq.ndim != 2 or u_seq.shape[0] < 1:
                    continue
                if not np.isfinite(x_center).all() or not np.isfinite(radius) or radius <= 0.0:
                    continue
                if not np.isfinite(u_control).all() or not np.isfinite(u_seq).all():
                    continue

                centers.append(x_center)
                radii.append(radius)
                u_controls.append(u_control)
                u_seqs.append(u_seq)
                meta.append(
                    {
                        "name": name,
                        "start_idx": tube.get("start_idx"),
                        "end_idx": tube.get("end_idx"),
                    }
                )

        if not centers:
            raise RuntimeError("No valid tubes loaded. Check your control alphabet content.")

        self.centers = np.stack(centers, axis=0)
        self.state_dim = self.centers.shape[1]
        self.control_dim = u_controls[0].shape[0]
        self.raw_radii = np.asarray(radii, dtype=np.float32)
        self.radii = np.maximum(self.raw_radii * self.radius_scale, self.eps)
        self.u_controls = np.stack(u_controls, axis=0)
        self.u_seqs = u_seqs
        self.meta = meta
        self.active_idx = None
        self.active_k = 0

        if distance_weights is None:
            self.distance_weights = np.ones(self.state_dim, dtype=np.float32)
        else:
            self.distance_weights = np.asarray(distance_weights, dtype=np.float32).reshape(self.state_dim)

        self.default_u = (
            np.zeros(self.control_dim, dtype=np.float32)
            if default_u is None
            else np.asarray(default_u, dtype=np.float32).reshape(self.control_dim)
        )

        print(f"[TubeController] loaded {self.centers.shape[0]} tubes.")
        print(
            "[TubeController] effective radius stats: min/med/max = "
            f"{float(np.min(self.radii)):.3e}/"
            f"{float(np.median(self.radii)):.3e}/"
            f"{float(np.max(self.radii)):.3e}"
        )

    def reset_episode(self, x0=None):
        self.active_idx = None
        self.active_k = 0
        if self.backup_controller is not None and hasattr(self.backup_controller, "reset_episode"):
            self.backup_controller.reset_episode(x0=x0)

    def _distance(self, x, centers):
        diff = centers - x[None, :]
        for idx in self.angle_indices:
            diff[:, idx] = angle_normalize(diff[:, idx])
        weighted = diff * self.distance_weights[None, :]
        return np.linalg.norm(weighted, axis=1)

    def select_tube(self, x_current):
        x = np.asarray(x_current, dtype=np.float32).reshape(self.state_dim)
        dists = self._distance(x, self.centers)
        rhos = dists / (self.radii + self.eps)
        best_idx = int(np.argmin(rhos))
        best_rho = float(rhos[best_idx])
        return best_idx, best_rho

    def get_action(self, x_current):
        x = np.asarray(x_current, dtype=np.float32).reshape(self.state_dim)
        if self.execute_full_sequence and self.active_idx is not None:
            active_center = self.centers[self.active_idx]
            active_rho = float(
                self._distance(x, active_center[None, :])[0] / (self.radii[self.active_idx] + self.eps)
            )
            if active_rho <= self.abort_threshold:
                u_seq = self.u_seqs[self.active_idx]
                if self.active_k < u_seq.shape[0]:
                    u = u_seq[self.active_k].copy()
                    self.active_k += 1
                    return u, "Expert", active_rho
            self.active_idx = None
            self.active_k = 0

        best_idx, best_rho = self.select_tube(x)
        if best_rho <= self.enter_threshold:
            if self.execute_full_sequence:
                self.active_idx = best_idx
                self.active_k = 1
                return self.u_seqs[best_idx][0].copy(), "Expert", best_rho
            return self.u_controls[best_idx].copy(), "Expert", best_rho

        if self.outside_mode == "nearest_control":
            return self.u_controls[best_idx].copy(), "Nearest", best_rho
        if self.outside_mode == "nearest_sequence" and self.execute_full_sequence:
            self.active_idx = best_idx
            self.active_k = 1
            return self.u_seqs[best_idx][0].copy(), "Nearest", best_rho
        if self.outside_mode == "backup" and self.backup_controller is not None:
            return self.backup_controller.get_action(x).copy(), "Backup", best_rho
        return self.default_u.copy(), "Default", best_rho


def evaluate_incremental_chain_policy(
    spec: ChainPolicySpec,
    save_dir,
    target_names,
    init_states,
    dt=0.02,
    episode_seconds=25.0,
    controller_kwargs=None,
    alphabet_filename=None,
    result_filename=None,
    plot=True,
    verbose=True,
):
    alphabet_filename = spec.alphabet_filename if alphabet_filename is None else alphabet_filename
    result_filename = spec.success_rate_filename if result_filename is None else result_filename
    full_alphabet_dict = load_control_alphabet(save_dir, alphabet_filename=alphabet_filename)

    controller_kwargs = {} if controller_kwargs is None else dict(controller_kwargs)
    controller_kwargs.setdefault("angle_indices", spec.angle_indices)
    controller_kwargs.setdefault("distance_weights", np.ones(spec.state_dim, dtype=float))
    controller_kwargs.setdefault("default_u", np.zeros(spec.control_dim, dtype=np.float32))

    n_traj_list = []
    success_rates = []
    diag = []

    if verbose:
        print(f"\n[Eval:{spec.name}] Running incremental chain-policy evaluation")
        print("controller_kwargs =", controller_kwargs)

    for k in range(1, len(target_names) + 1):
        current_names = target_names[:k]
        current_alphabet_dict = {
            name: full_alphabet_dict[name]
            for name in current_names
            if name in full_alphabet_dict
        }
        controller = TubeController(current_alphabet_dict, **controller_kwargs)
        env = spec.make_env(dt=dt, episode_seconds=episode_seconds)

        success_count = 0
        expert_ratios = []
        guided_ratios = []
        rho_mins = []

        iterator = tqdm(init_states, desc=f"{spec.name}:k={k}", leave=False) if verbose else init_states
        for x0 in iterator:
            obs, _ = env.reset(options={"x0": x0})
            controller.reset_episode(x0=obs)

            done = False
            has_succeeded = spec.check_success(obs)
            expert_steps = 0
            guided_steps = 0
            total_steps = 0
            rho_min_this = np.inf

            while not done:
                u, status, rho = controller.get_action(obs)
                if status == "Expert":
                    expert_steps += 1
                if status in {"Expert", "Nearest", "Backup"}:
                    guided_steps += 1
                total_steps += 1
                if np.isfinite(rho):
                    rho_min_this = min(rho_min_this, rho)

                obs, _, terminated, truncated, _ = env.step(u)
                if spec.check_success(obs):
                    has_succeeded = True
                done = bool(terminated or truncated)

            if has_succeeded:
                success_count += 1
            expert_ratios.append(expert_steps / max(total_steps, 1))
            guided_ratios.append(guided_steps / max(total_steps, 1))
            rho_mins.append(rho_min_this if np.isfinite(rho_min_this) else np.nan)

        rate = success_count / len(init_states)
        er_mean = float(np.nanmean(expert_ratios))
        guided_mean = float(np.nanmean(guided_ratios))
        rho_med = float(np.nanmedian(rho_mins))

        n_traj_list.append(k)
        success_rates.append(rate)
        diag.append((er_mean, guided_mean, 1.0 - guided_mean, rho_med))

        if verbose:
            print(
                f"[Eval:{spec.name}] k={k} -> Success Rate: {rate:.2%} | "
                f"Expert-step ratio(mean): {er_mean:.3f} | "
                f"Guided-step ratio(mean): {guided_mean:.3f} | "
                f"median(min_rho): {rho_med:.3f}"
            )

    result_save_path = _expanded_path(save_dir) / result_filename
    np.savez(
        result_save_path,
        n_traj_list=np.asarray(n_traj_list, dtype=int),
        success_rates=np.asarray(success_rates, dtype=float),
        diag=np.asarray(diag, dtype=float),
    )
    if verbose:
        print(f"\n[Eval:{spec.name}] Results saved to {result_save_path}")

    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(n_traj_list, success_rates, marker="o", linewidth=2, label="Chain Policy")
        plt.xlabel("Number of Expert Trajectories")
        plt.ylabel("Success Rate")
        plt.title(f"{spec.name}: success rate vs. dataset size")
        plt.ylim(-0.05, 1.05)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return n_traj_list, success_rates, diag
