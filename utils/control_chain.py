from __future__ import annotations

import os
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dynamics.Dynamics_single_pendulum import (
    I as SINGLE_I,
    g as SINGLE_G,
    l as SINGLE_L,
    m as SINGLE_M,
    SinglePendulumEnv,
)
from dynamics.Dynamics_double_pendulum import (
    DoublePendulumEnv,
    g as DOUBLE_G,
    m1 as DOUBLE_M1,
    m2 as DOUBLE_M2,
    l1 as DOUBLE_L1,
    l2 as DOUBLE_L2,
    lc1 as DOUBLE_LC1,
    lc2 as DOUBLE_LC2,
    I1 as DOUBLE_I1,
    I2 as DOUBLE_I2,
)
from dynamics.Dynamics_spring_mass import (
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
    "make_double_pendulum_spec",
    "make_double_pendulum_mpc_backup",
    "make_spring_mass_spec",
    "make_spring_mass_pd_backup",
    "build_control_alphabet_from_saved_rollouts",
    "evaluate_incremental_chain_policy",
    "greedy_rank_expert_trajectories",
    "load_control_alphabet",
    "load_saved_rollouts",
    "rank_expert_trajectories_by_utility",
    "save_rollout_metrics",
    "select_chain_trajectory_subset",
    "success_mask_from_spec",
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


def _clip_control(u, u_min=None, u_max=None):
    u = np.asarray(u, dtype=np.float32)
    if u_min is not None:
        u = np.maximum(u, np.asarray(u_min, dtype=np.float32))
    if u_max is not None:
        u = np.minimum(u, np.asarray(u_max, dtype=np.float32))
    return u.astype(np.float32)


def _rk4_step(f_continuous_fn, x, u, dt):
    x = np.asarray(x, dtype=np.float32)
    u = np.asarray(u, dtype=np.float32)
    dt = float(dt)
    k1 = np.asarray(f_continuous_fn(x, u), dtype=np.float32)
    k2 = np.asarray(f_continuous_fn(x + 0.5 * dt * k1, u), dtype=np.float32)
    k3 = np.asarray(f_continuous_fn(x + 0.5 * dt * k2, u), dtype=np.float32)
    k4 = np.asarray(f_continuous_fn(x + dt * k3, u), dtype=np.float32)
    return (x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)).astype(np.float32)


def _scale_control_to_bounds(u, u_min=None, u_max=None, eps=1e-9):
    """
    Scale the full control vector by a single factor to satisfy box constraints.

    Using a scalar scale preserves the direction of the projected control,
    so orthogonality constraints such as a(x)^T u = 0 are maintained.
    """
    u = np.asarray(u, dtype=np.float32).copy()
    alpha = 1.0

    if u_max is not None:
        u_max = np.asarray(u_max, dtype=np.float32).reshape(u.shape)
        pos_mask = u > eps
        if np.any(pos_mask):
            alpha = min(alpha, float(np.min(u_max[pos_mask] / u[pos_mask])))

    if u_min is not None:
        u_min = np.asarray(u_min, dtype=np.float32).reshape(u.shape)
        neg_mask = u < -eps
        if np.any(neg_mask):
            alpha = min(alpha, float(np.min(u_min[neg_mask] / u[neg_mask])))

    alpha = float(np.clip(alpha, 0.0, 1.0))
    return (alpha * u).astype(np.float32)


def _project_control_orthogonal_to_direction(u, a, u_min=None, u_max=None, eps=1e-9):
    """
    Project control onto the orthogonal complement of a, then scale to bounds.

    When ||a|| is near zero, any input is energy-preserving for the induced
    power term, so we fall back to ordinary clipping.
    """
    u = np.asarray(u, dtype=np.float32).reshape(-1)
    a = np.asarray(a, dtype=np.float32).reshape(u.shape)
    norm_sq = float(np.dot(a, a))
    if not np.isfinite(norm_sq) or norm_sq <= eps:
        return _clip_control(u, u_min, u_max)

    u_proj = u - (float(np.dot(a, u)) / norm_sq) * a
    return _scale_control_to_bounds(u_proj, u_min=u_min, u_max=u_max, eps=eps)


def _sample_single_initial_states(
    num_inits=50,
    seed=0,
    theta_low=0.0,
    theta_high=2.0 * np.pi,
    p_low=0.0,
    p_high=0.0,
    theta_dot_low=None,
    theta_dot_high=None,
):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(low=theta_low, high=theta_high, size=(num_inits, 1))
    if theta_dot_low is not None or theta_dot_high is not None:
        theta_dot_low = 0.0 if theta_dot_low is None else theta_dot_low
        theta_dot_high = 0.0 if theta_dot_high is None else theta_dot_high
        p_low = SINGLE_I * float(theta_dot_low)
        p_high = SINGLE_I * float(theta_dot_high)
    p = rng.uniform(low=p_low, high=p_high, size=(num_inits, 1))
    theta = angle_normalize(theta)
    return np.hstack([theta, p]).astype(np.float32)


def _sample_spring_initial_states(
    num_inits=50,
    seed=0,
    x_low=-2.0,
    x_high=2.0,
    p_low=None,
    p_high=None,
    x_dot_low=-2.0,
    x_dot_high=2.0,
):
    rng = np.random.default_rng(seed)
    x_pos = rng.uniform(low=x_low, high=x_high, size=(num_inits, 1))
    if p_low is None:
        p_low = SPRING_M * float(x_dot_low)
    if p_high is None:
        p_high = SPRING_M * float(x_dot_high)
    p = rng.uniform(low=p_low, high=p_high, size=(num_inits, 1))
    return np.hstack([x_pos, p]).astype(np.float32)


def _sample_double_initial_states(
    num_inits=50,
    seed=0,
    th1_low=0.0,
    th1_high=2.0 * np.pi,
    th2_low=0.0,
    th2_high=2.0 * np.pi,
    p1_low=None,
    p1_high=None,
    p2_low=None,
    p2_high=None,
    th1d_low=0.0,
    th1d_high=0.0,
    th2d_low=0.0,
    th2d_high=0.0,
):
    rng = np.random.default_rng(seed)
    th1 = rng.uniform(low=th1_low, high=th1_high, size=(num_inits, 1))
    th2 = rng.uniform(low=th2_low, high=th2_high, size=(num_inits, 1))
    th1 = angle_normalize(th1)
    th2 = angle_normalize(th2)
    if all(v is not None for v in (p1_low, p1_high, p2_low, p2_high)):
        p1 = rng.uniform(low=p1_low, high=p1_high, size=(num_inits, 1))
        p2 = rng.uniform(low=p2_low, high=p2_high, size=(num_inits, 1))
        return np.hstack([th1, th2, p1, p2]).astype(np.float32)

    th1d = rng.uniform(low=th1d_low, high=th1d_high, size=(num_inits, 1))
    th2d = rng.uniform(low=th2d_low, high=th2d_high, size=(num_inits, 1))
    p = []
    for idx in range(num_inits):
        q = np.array([th1[idx, 0], th2[idx, 0]], dtype=float)
        qdot = np.array([th1d[idx, 0], th2d[idx, 0]], dtype=float)
        p.append(_double_mass_matrix_np(q) @ qdot)
    p = np.asarray(p, dtype=np.float32)
    return np.hstack([th1, th2, p[:, [0]], p[:, [1]]]).astype(np.float32)


def _double_mass_matrix_np(q):
    th1, th2 = np.asarray(q, dtype=float).reshape(2,)
    coupling = DOUBLE_M2 * DOUBLE_L1 * DOUBLE_L2 * np.cos(th1 - th2)
    return np.array(
        [
            [(DOUBLE_M1 + DOUBLE_M2) * DOUBLE_L1**2, coupling],
            [coupling, DOUBLE_M2 * DOUBLE_L2**2],
        ],
        dtype=float,
    )


def _single_control_power_direction(x):
    x = np.asarray(x, dtype=np.float32).reshape(2,)
    return np.array([x[1] / SINGLE_I], dtype=np.float32)


def _spring_control_power_direction(x):
    x = np.asarray(x, dtype=np.float32).reshape(2,)
    return np.array([x[1] / SPRING_M], dtype=np.float32)


def _double_control_power_direction(x):
    x = np.asarray(x, dtype=np.float32).reshape(4,)
    q = x[:2]
    p = x[2:]
    q_dot = np.linalg.solve(_double_mass_matrix_np(q), p.astype(float))
    return np.asarray(q_dot, dtype=np.float32)


def _single_state_error(x, target):
    return _state_error_with_angles(x, target, angle_indices=(0,))


def _double_state_error(x, target):
    return _state_error_with_angles(x, target, angle_indices=(0, 1))


def _spring_state_error(x, target):
    return (np.asarray(x, dtype=np.float32) - np.asarray(target, dtype=np.float32)).astype(
        np.float32
    )


def _single_f_continuous(x, u):
    theta, p = np.asarray(x, dtype=float).reshape(2,)
    tau = np.asarray(u, dtype=float).reshape(-1)[0]
    theta_dot = p / SINGLE_I
    p_dot = tau - SINGLE_M * SINGLE_G * SINGLE_L * np.sin(theta)
    return np.array([theta_dot, p_dot], dtype=float)


def _spring_f_continuous(x, u):
    x_pos, p = np.asarray(x, dtype=float).reshape(2,)
    force = np.asarray(u, dtype=float).reshape(-1)[0]
    x_dot = p / SPRING_M
    p_dot = force - SPRING_K * x_pos
    return np.array([x_dot, p_dot], dtype=float)


def _double_f_continuous(x, u):
    th1, th2, p1, p2 = np.asarray(x, dtype=float).reshape(4,)
    tau = np.asarray(u, dtype=float).reshape(-1)
    if tau.size < 2:
        tau = np.pad(tau, (0, 2 - tau.size))
    tau1, tau2 = tau[:2]
    q = np.array([th1, th2], dtype=float)
    p = np.array([p1, p2], dtype=float)
    q_dot = np.linalg.solve(_double_mass_matrix_np(q), p)
    coupling_grad = DOUBLE_M2 * DOUBLE_L1 * DOUBLE_L2 * np.sin(th1 - th2)
    dH_dth1 = coupling_grad * q_dot[0] * q_dot[1] + (DOUBLE_M1 + DOUBLE_M2) * DOUBLE_G * DOUBLE_L1 * np.sin(th1)
    dH_dth2 = -coupling_grad * q_dot[0] * q_dot[1] + DOUBLE_M2 * DOUBLE_G * DOUBLE_L2 * np.sin(th2)
    p_dot = np.array([tau1 - dH_dth1, tau2 - dH_dth2], dtype=float)
    return np.array([q_dot[0], q_dot[1], p_dot[0], p_dot[1]], dtype=float)


def _make_single_energy_fn(target_state):
    target_state = np.asarray(target_state, dtype=float).reshape(2,)
    target_energy = 0.5 * (target_state[1] ** 2) / SINGLE_I + SINGLE_M * SINGLE_G * SINGLE_L * (
        1.0 - np.cos(target_state[0])
    )

    def energy_from_states(X):
        X = np.asarray(X, dtype=float)
        theta = X[:, 0]
        p = X[:, 1]
        energy = 0.5 * (p**2) / SINGLE_I + SINGLE_M * SINGLE_G * SINGLE_L * (
            1.0 - np.cos(theta)
        )
        return energy - target_energy

    return energy_from_states


def _make_double_energy_fn(target_state):
    target_state = np.asarray(target_state, dtype=float).reshape(4,)
    target_q = target_state[:2]
    target_p = target_state[2:]
    target_q_dot = np.linalg.solve(_double_mass_matrix_np(target_q), target_p)
    target_energy = (
        0.5 * float(target_p @ target_q_dot)
        - (DOUBLE_M1 + DOUBLE_M2) * DOUBLE_G * DOUBLE_L1 * np.cos(target_q[0])
        - DOUBLE_M2 * DOUBLE_G * DOUBLE_L2 * np.cos(target_q[1])
    )

    def energy_from_states(X):
        X = np.asarray(X, dtype=float)
        energy = np.zeros(X.shape[0], dtype=float)
        for idx, row in enumerate(X):
            q = row[:2]
            p = row[2:]
            q_dot = np.linalg.solve(_double_mass_matrix_np(q), p)
            energy[idx] = (
                0.5 * float(p @ q_dot)
                - (DOUBLE_M1 + DOUBLE_M2) * DOUBLE_G * DOUBLE_L1 * np.cos(q[0])
                - DOUBLE_M2 * DOUBLE_G * DOUBLE_L2 * np.cos(q[1])
            )
        return energy - target_energy

    return energy_from_states


def _single_lh_from_states(X):
    X = np.asarray(X, dtype=float)
    theta = X[:, 0]
    p = X[:, 1]
    dH_dtheta = SINGLE_M * SINGLE_G * SINGLE_L * np.sin(theta)
    dH_dp = p / SINGLE_I
    return np.sqrt(dH_dtheta**2 + dH_dp**2)


def _make_spring_energy_fn(target_state):
    target_state = np.asarray(target_state, dtype=float).reshape(2,)
    target_energy = 0.5 * (target_state[1] ** 2) / SPRING_M + 0.5 * SPRING_K * target_state[0] ** 2

    def energy_from_states(X):
        X = np.asarray(X, dtype=float)
        x_pos = X[:, 0]
        p = X[:, 1]
        energy = 0.5 * (p**2) / SPRING_M + 0.5 * SPRING_K * x_pos**2
        return energy - target_energy

    return energy_from_states


def _spring_lh_from_states(X):
    X = np.asarray(X, dtype=float)
    x_pos = X[:, 0]
    p = X[:, 1]
    dH_dx = SPRING_K * x_pos
    dH_dp = p / SPRING_M
    return np.sqrt(dH_dx**2 + dH_dp**2)


def _double_lh_from_states(X):
    X = np.asarray(X, dtype=float)
    lh = np.zeros(X.shape[0], dtype=float)
    for idx, row in enumerate(X):
        th1, th2, p1, p2 = row
        q = np.array([th1, th2], dtype=float)
        p = np.array([p1, p2], dtype=float)
        q_dot = np.linalg.solve(_double_mass_matrix_np(q), p)
        coupling_grad = DOUBLE_M2 * DOUBLE_L1 * DOUBLE_L2 * np.sin(th1 - th2)
        dH_dth1 = coupling_grad * q_dot[0] * q_dot[1] + (DOUBLE_M1 + DOUBLE_M2) * DOUBLE_G * DOUBLE_L1 * np.sin(th1)
        dH_dth2 = -coupling_grad * q_dot[0] * q_dot[1] + DOUBLE_M2 * DOUBLE_G * DOUBLE_L2 * np.sin(th2)
        lh[idx] = np.sqrt(dH_dth1**2 + dH_dth2**2 + q_dot[0] ** 2 + q_dot[1] ** 2)
    return lh


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


def _make_double_env_factory(target_state, success_tol):
    target_state = np.asarray(target_state, dtype=float).reshape(4,)
    success_tol = np.asarray(success_tol, dtype=float).reshape(4,)

    def factory(dt=0.02, episode_seconds=25.0):
        return DoublePendulumEnv(
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
    control_power_direction_fn: Callable | None = None
    success_norm_eps: float | None = None

    def sample_initial_states(self, num_inits=50, seed=0, **kwargs):
        return self.sample_initial_states_fn(num_inits=num_inits, seed=seed, **kwargs)

    def state_error(self, x, target=None):
        target = self.target_state if target is None else target
        return self.state_error_fn(x, target)

    def check_success(self, obs, target=None, tol=None):
        target = self.target_state if target is None else target
        delta = self.state_error(obs, target=target)
        if self.success_norm_eps is not None and tol is None:
            return bool(np.linalg.norm(delta, ord=2) <= float(self.success_norm_eps))
        tol = self.success_tol if tol is None else tol
        return bool(np.all(np.abs(delta) <= np.asarray(tol, dtype=np.float32)))

    def compute_energy(self, X):
        return np.asarray(self.energy_from_states_fn(X), dtype=float).reshape(-1)

    def compute_lh(self, X):
        return np.asarray(self.lh_from_states_fn(X), dtype=float).reshape(-1)

    def make_env(self, dt=0.02, episode_seconds=25.0):
        return self.env_factory_fn(dt=dt, episode_seconds=episode_seconds)

    def control_power_direction(self, x):
        if self.control_power_direction_fn is None:
            return None
        return np.asarray(self.control_power_direction_fn(x), dtype=np.float32).reshape(self.control_dim)


def success_mask_from_spec(spec: ChainPolicySpec, obs_batch):
    X = np.asarray(obs_batch, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    target = np.asarray(spec.target_state, dtype=np.float32).reshape(1, -1)
    delta = X - target
    for idx in spec.angle_indices:
        delta[:, idx] = angle_normalize(delta[:, idx])
    if spec.success_norm_eps is not None:
        return np.linalg.norm(delta, axis=1) <= float(spec.success_norm_eps)
    tol = np.asarray(spec.success_tol, dtype=np.float32).reshape(1, -1)
    return np.all(np.abs(delta) <= tol, axis=1)


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
        u = _clip_control(u, self.u_min, self.u_max)
        return np.asarray(u, dtype=np.float32).reshape(self.control_dim)


class SpringMassPDBackupController(AffineFeedbackBackupController):
    def __init__(self, kp=8.0, kd=4.0, u_min=-20.0, u_max=20.0, target_state=None):
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


def make_double_pendulum_mpc_backup(
    dt=0.02,
    n_horizon=120,
    target_state=None,
):
    from utils.mpc import DoublePendulumParams, build_double_pendulum_mpc

    target_state = (
        np.array([np.pi, np.pi, 0.0, 0.0], dtype=np.float32)
        if target_state is None
        else np.asarray(target_state, dtype=np.float32).reshape(4,)
    )
    cfg = DoublePendulumParams(dt=dt, n_horizon=n_horizon, x_ref=np.asarray(target_state, dtype=float))
    _, mpc, _ = build_double_pendulum_mpc(cfg)

    if hasattr(mpc, "settings") and hasattr(mpc.settings, "supress_ipopt_output"):
        try:
            mpc.settings.supress_ipopt_output()
        except TypeError:
            pass

    return MPCBackupController(mpc=mpc, control_dim=2)


def make_single_pendulum_spec(
    target_state=None,
    success_tol=None,
    success_norm_eps=None,
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
        control_power_direction_fn=_single_control_power_direction,
        success_norm_eps=success_norm_eps,
    )


def make_double_pendulum_spec(
    target_state=None,
    success_tol=None,
    success_norm_eps=None,
):
    target_state = (
        np.array([np.pi, np.pi, 0.0, 0.0], dtype=np.float32)
        if target_state is None
        else np.asarray(target_state, dtype=np.float32).reshape(4,)
    )
    success_tol = (
        np.array([0.12, 0.12, 0.25, 0.25], dtype=np.float32)
        if success_tol is None
        else np.asarray(success_tol, dtype=np.float32).reshape(4,)
    )
    return ChainPolicySpec(
        name="double_pendulum",
        state_dim=4,
        control_dim=2,
        angle_indices=(0, 1),
        target_state=target_state,
        success_tol=success_tol,
        rollout_npz_filename="all_rollouts_double_pendulum.npz",
        alphabet_filename="control_alphabet_double_pendulum.pkl",
        success_rate_filename="incremental_tube_success_rates_double_pendulum.npz",
        sample_initial_states_fn=_sample_double_initial_states,
        state_error_fn=_double_state_error,
        f_continuous_fn=_double_f_continuous,
        energy_from_states_fn=_make_double_energy_fn(target_state),
        lh_from_states_fn=_double_lh_from_states,
        env_factory_fn=_make_double_env_factory(target_state, success_tol),
        control_power_direction_fn=_double_control_power_direction,
        success_norm_eps=success_norm_eps,
    )


def make_spring_mass_spec(
    target_state=None,
    success_tol=None,
    success_norm_eps=None,
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
        control_power_direction_fn=_spring_control_power_direction,
        success_norm_eps=success_norm_eps,
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


def _compute_rollout_metric_arrays(
    spec: ChainPolicySpec,
    X,
    U,
    eps_L=1e-5,
):
    X = np.asarray(X, dtype=float)
    H = spec.compute_energy(X)
    LH = spec.compute_lh(X)

    control_count = max(X.shape[0] - 1, 0)
    U = _ensure_control_shape(U, spec.control_dim)
    if U.shape[0] < control_count:
        pad = np.zeros((control_count - U.shape[0], spec.control_dim), dtype=float)
        U = np.vstack([U, pad])
    else:
        U = U[:control_count]

    l_dyn = np.zeros(control_count, dtype=float)
    for idx in range(control_count):
        l_dyn[idx] = get_dynamics_l_numerical(
            X[idx], U[idx], spec.f_continuous_fn, eps=eps_L
        )
    return H, LH, l_dyn


def _load_rollout_metric_arrays(save_path: Path, name: str):
    h_path = save_path / f"H_total_{name}.npy"
    lh_path = save_path / f"LH_{name}.npy"
    ldyn_path = save_path / f"Ldyn_{name}.npy"
    if not h_path.exists() or not lh_path.exists() or not ldyn_path.exists():
        return None
    return (
        np.load(h_path),
        np.load(lh_path),
        np.load(ldyn_path),
    )


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
        H, LH, l_dyn = _compute_rollout_metric_arrays(
            spec=spec,
            X=X,
            U=U,
            eps_L=eps_L,
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


def _trajectory_distances_from_center(spec: ChainPolicySpec, points, center):
    points = np.asarray(points, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32).reshape(1, -1)
    diff = points - center
    for idx in spec.angle_indices:
        diff[:, idx] = angle_normalize(diff[:, idx])
    return np.linalg.norm(diff, axis=1)


def generate_control_alphabet_from_rollout(
    spec: ChainPolicySpec,
    T,
    X,
    U,
    H_array,
    LH_array,
    l_dyns=None,
    rho=0.99,
    H_star=0.0,
    eta=0.0,
    r_min=1e-6,
    max_lookahead=200,
    advance_stride=None,
    eps_L=1e-5,
    v0=0.0,
    radius_mode="paper",
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

    if l_dyns is None:
        l_dyns = np.zeros(n_ctrl_needed, dtype=float)
        for idx in range(n_ctrl_needed):
            l_dyns[idx] = get_dynamics_l_numerical(
                X[idx], U[idx], spec.f_continuous_fn, eps=eps_L
            )
    else:
        l_dyns = np.asarray(l_dyns, dtype=float).reshape(-1)
        if l_dyns.size < n_ctrl_needed:
            raise ValueError(
                f"l_dyns must have at least {n_ctrl_needed} entries, got {l_dyns.size}"
            )
        l_dyns = l_dyns[:n_ctrl_needed]

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
        l_h_interval = float(LH_array[i])
        l_dyn_interval = 0.0
        for k in range(1, lookahead + 1):
            j = i + k
            tau = k * dt
            l_h_interval = max(l_h_interval, float(LH_array[j]))
            l_dyn_interval = max(l_dyn_interval, float(l_dyns[j - 1]))
            v_next = abs(float(H_array[j]) - float(H_star))
            if str(radius_mode) == "legacy":
                numerator = rho * max(v_curr - eta, 0.0) - max(v_next - eta, 0.0)
                denom = l_h_interval * (rho + np.exp(l_dyn_interval * tau))
            else:
                numerator = max(v_curr - eta, 0.0) - max(v_next - eta, 0.0) - float(v0) * tau
                denom = l_h_interval * (1.0 + np.exp(l_dyn_interval * tau))
            if numerator <= 0.0:
                continue

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
            x_seq = X[i : end_idx + 1].copy()
            tubes.append(
                {
                    "x_center": X[i].astype(np.float32),
                    "x_end": X[end_idx].astype(np.float32),
                    "x_seq": x_seq.astype(np.float32),
                    "u_control": U[i].astype(np.float32),
                    "u_seq": u_seq.astype(np.float32),
                    "radius": float(best_r),
                    "tau": float(best_tau),
                    "start_idx": int(i),
                    "end_idx": int(end_idx),
                }
            )
            if advance_stride is None:
                segment = X[i + 1 : end_idx + 1]
                if segment.size == 0:
                    i = end_idx
                else:
                    segment_dist = _trajectory_distances_from_center(spec, segment, X[i])
                    boundary_hits = np.flatnonzero(segment_dist >= best_r)
                    if boundary_hits.size > 0:
                        i = i + 1 + int(boundary_hits[0])
                    else:
                        i = end_idx
            else:
                i += max(1, min(int(best_steps), int(advance_stride)))
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
    advance_stride=None,
    eps_L=1e-5,
    v0=0.0,
    radius_mode="paper",
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
        metric_arrays = _load_rollout_metric_arrays(save_path, name)
        if metric_arrays is None:
            if verbose:
                print(f"[INFO] Computing rollout metrics on the fly for {name}.")
            H_array, LH_array, l_dyns = _compute_rollout_metric_arrays(
                spec=spec,
                X=X,
                U=U,
                eps_L=eps_L,
            )
            if save_metrics:
                np.save(save_path / f"H_total_{name}.npy", H_array)
                np.save(save_path / f"LH_{name}.npy", LH_array)
                np.save(save_path / f"Ldyn_{name}.npy", l_dyns)
        else:
            H_array, LH_array, l_dyns = metric_arrays

        tubes = generate_control_alphabet_from_rollout(
            spec=spec,
            T=T,
            X=X,
            U=U,
            H_array=H_array,
            LH_array=LH_array,
            l_dyns=l_dyns,
            rho=rho,
            H_star=H_star,
            eta=eta,
            r_min=r_min,
            max_lookahead=max_lookahead,
            advance_stride=advance_stride,
            eps_L=eps_L,
            v0=v0,
            radius_mode=radius_mode,
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
        target_state=None,
        success_tol=None,
        distance_weights=None,
        radius_scale=1.0,
        radius_floor=0.0,
        enter_threshold=1.0,
        abort_threshold=np.inf,
        outside_mode="default",
        selection_mode="nearest",
        stitch_top_k=8,
        stitch_threshold=3.0,
        stitch_entry_weight=1.0,
        stitch_value_weight=0.35,
        tracking_gain=None,
        outside_tracking_gain=None,
        project_tracking=False,
        control_power_direction_fn=None,
        trajectory_rank_weight=0.0,
        prefer_earliest_within_support=False,
        match_sequence_points=False,
        sequence_match_stride=1,
        match_sequence_tube_limit=None,
        log_stats=True,
        u_min=None,
        u_max=None,
    ):
        self.eps = float(eps)
        self.execute_full_sequence = bool(execute_full_sequence)
        self.angle_indices = tuple(angle_indices)
        self.radius_scale = float(radius_scale)
        self.radius_floor = max(float(radius_floor), 0.0)
        self.enter_threshold = float(enter_threshold)
        self.abort_threshold = float(abort_threshold)
        self.outside_mode = str(outside_mode)
        self.backup_controller = backup_controller
        self.selection_mode = str(selection_mode)
        self.stitch_top_k = int(stitch_top_k)
        self.stitch_threshold = float(stitch_threshold)
        self.stitch_entry_weight = float(stitch_entry_weight)
        self.stitch_value_weight = float(stitch_value_weight)
        self.trajectory_rank_weight = float(trajectory_rank_weight)
        self.prefer_earliest_within_support = bool(prefer_earliest_within_support)
        self.match_sequence_points = bool(match_sequence_points)
        self.sequence_match_stride = max(int(sequence_match_stride), 1)
        self.match_sequence_tube_limit = (
            None if match_sequence_tube_limit is None else int(match_sequence_tube_limit)
        )
        self.log_stats = bool(log_stats)

        centers = []
        ends = []
        radii = []
        u_controls = []
        u_seqs = []
        x_seqs = []
        traj_ranks = []
        meta = []

        name_to_rank = {str(name): idx for idx, name in enumerate(all_alphabets_dict.keys())}
        for name, tube_list in all_alphabets_dict.items():
            for tube in tube_list:
                x_center = np.asarray(tube["x_center"], dtype=np.float32).reshape(-1)
                x_end = np.asarray(tube.get("x_end", x_center), dtype=np.float32).reshape(-1)
                radius = float(tube["radius"])
                u_seq = np.asarray(tube["u_seq"], dtype=np.float32)
                x_seq = np.asarray(tube.get("x_seq", x_center[None, :]), dtype=np.float32)
                u_control = np.asarray(tube.get("u_control", u_seq[0]), dtype=np.float32).reshape(-1)

                if (
                    x_center.ndim != 1
                    or x_end.ndim != 1
                    or u_seq.ndim != 2
                    or u_seq.shape[0] < 1
                    or x_seq.ndim != 2
                    or x_seq.shape[0] < 1
                ):
                    continue
                if not np.isfinite(x_center).all() or not np.isfinite(radius) or radius <= 0.0:
                    continue
                if not np.isfinite(x_end).all():
                    continue
                if not np.isfinite(u_control).all() or not np.isfinite(u_seq).all() or not np.isfinite(x_seq).all():
                    continue

                centers.append(x_center)
                ends.append(x_end)
                radii.append(radius)
                u_controls.append(u_control)
                u_seqs.append(u_seq)
                x_seqs.append(x_seq)
                traj_ranks.append(int(name_to_rank.get(str(name), 0)))
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
        self.ends = np.stack(ends, axis=0)
        self.state_dim = self.centers.shape[1]
        self.control_dim = u_controls[0].shape[0]
        self.raw_radii = np.asarray(radii, dtype=np.float32)
        self.radii = np.maximum(self.raw_radii * self.radius_scale, self.radius_floor)
        self.radii = np.maximum(self.radii, self.eps)
        self.u_controls = np.stack(u_controls, axis=0)
        self.u_seqs = u_seqs
        self.x_seqs = x_seqs
        self.trajectory_ranks = np.asarray(traj_ranks, dtype=np.float32)
        self.meta = meta
        self.sequence_match_points = []
        self.sequence_match_steps = []
        for u_seq, x_seq in zip(self.u_seqs, self.x_seqs):
            n_match = min(u_seq.shape[0], x_seq.shape[0])
            if n_match <= 0:
                self.sequence_match_points.append(x_seq[:1])
                self.sequence_match_steps.append(np.array([0], dtype=np.int32))
                continue
            steps = np.arange(0, n_match, self.sequence_match_stride, dtype=np.int32)
            if steps.size == 0 or int(steps[-1]) != n_match - 1:
                steps = np.append(steps, np.int32(n_match - 1))
            self.sequence_match_points.append(x_seq[steps])
            self.sequence_match_steps.append(steps)
        self.active_idx = None
        self.active_k = 0

        if distance_weights is None:
            self.distance_weights = np.ones(self.state_dim, dtype=np.float32)
        else:
            self.distance_weights = np.asarray(distance_weights, dtype=np.float32).reshape(self.state_dim)

        self.target_state = None if target_state is None else np.asarray(target_state, dtype=np.float32).reshape(self.state_dim)
        self.success_tol = None if success_tol is None else np.asarray(success_tol, dtype=np.float32).reshape(self.state_dim)
        self.default_u = (
            np.zeros(self.control_dim, dtype=np.float32)
            if default_u is None
            else np.asarray(default_u, dtype=np.float32).reshape(self.control_dim)
        )
        self.tracking_gain = None
        if tracking_gain is not None:
            gain = np.asarray(tracking_gain, dtype=np.float32)
            if gain.ndim == 1:
                gain = gain.reshape(1, -1)
            if gain.shape != (self.control_dim, self.state_dim):
                raise ValueError(
                    f"tracking_gain must have shape ({self.control_dim}, {self.state_dim}), got {gain.shape}"
                )
            self.tracking_gain = gain
        self.outside_tracking_gain = None
        if outside_tracking_gain is not None:
            gain = np.asarray(outside_tracking_gain, dtype=np.float32)
            if gain.ndim == 1:
                gain = gain.reshape(1, -1)
            if gain.shape != (self.control_dim, self.state_dim):
                raise ValueError(
                    f"outside_tracking_gain must have shape ({self.control_dim}, {self.state_dim}), got {gain.shape}"
                )
            self.outside_tracking_gain = gain
        self.project_tracking = bool(project_tracking)
        self.control_power_direction_fn = control_power_direction_fn
        self.u_min = None if u_min is None else np.asarray(u_min, dtype=np.float32).reshape(self.control_dim)
        self.u_max = None if u_max is None else np.asarray(u_max, dtype=np.float32).reshape(self.control_dim)
        if self.selection_mode == "stitch":
            self.stitch_values = self._build_stitch_values()
        else:
            self.stitch_values = np.zeros(self.centers.shape[0], dtype=np.float32)
        self.sequence_match_enabled = self.match_sequence_points and self.execute_full_sequence
        if self.match_sequence_tube_limit is not None:
            self.sequence_match_enabled = self.sequence_match_enabled and (
                self.centers.shape[0] <= self.match_sequence_tube_limit
            )

        if self.log_stats:
            print(f"[TubeController] loaded {self.centers.shape[0]} tubes.")
            print(
                "[TubeController] effective radius stats: min/med/max = "
                f"{float(np.min(self.radii)):.3e}/"
                f"{float(np.median(self.radii)):.3e}/"
                f"{float(np.max(self.radii)):.3e}"
            )
            if self.selection_mode == "stitch":
                finite_values = self.stitch_values[np.isfinite(self.stitch_values)]
                if finite_values.size:
                    print(
                        "[TubeController] stitch-value stats: min/med/max = "
                        f"{float(np.min(finite_values)):.3e}/"
                        f"{float(np.median(finite_values)):.3e}/"
                        f"{float(np.max(finite_values)):.3e}"
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

    def _pairwise_distance(self, queries, refs):
        queries = np.asarray(queries, dtype=np.float32)
        refs = np.asarray(refs, dtype=np.float32)
        diff = refs[None, :, :] - queries[:, None, :]
        for idx in self.angle_indices:
            diff[:, :, idx] = angle_normalize(diff[:, :, idx])
        weighted = diff * self.distance_weights[None, None, :]
        return np.linalg.norm(weighted, axis=2)

    def _distance_to_target(self, centers):
        if self.target_state is None:
            return np.full(centers.shape[0], np.inf, dtype=np.float32)
        return self._distance(self.target_state, centers).astype(np.float32)

    def _build_stitch_values(self):
        n_tubes = self.centers.shape[0]
        if n_tubes == 0:
            return np.array([], dtype=np.float32)
        base_cost = self._distance_to_target(self.ends)
        if self.success_tol is not None:
            tol_scale = float(np.linalg.norm(self.success_tol * self.distance_weights) + self.eps)
            base_cost = base_cost / tol_scale

        pairwise_rhos = self._pairwise_distance(self.ends, self.centers) / (
            self.radii[None, :] + self.eps
        )
        edge_weights = []
        for idx in range(n_tubes):
            neighbors = np.where(pairwise_rhos[idx] <= self.stitch_threshold)[0]
            edge_weights.append(
                [
                    (int(jdx), float(1.0 + pairwise_rhos[idx, jdx]))
                    for jdx in neighbors
                    if int(jdx) != idx
                ]
            )

        values = base_cost.astype(np.float32).copy()
        for _ in range(n_tubes):
            updated = False
            for idx in range(n_tubes):
                best_here = float(values[idx])
                for jdx, weight in edge_weights[idx]:
                    candidate = float(weight + values[jdx])
                    if candidate + 1e-8 < best_here:
                        best_here = candidate
                if best_here + 1e-8 < float(values[idx]):
                    values[idx] = best_here
                    updated = True
            if not updated:
                break
        return values

    def _tube_sequence_match(self, tube_idx, x_current):
        x = np.asarray(x_current, dtype=np.float32).reshape(self.state_dim)
        match_points = self.sequence_match_points[int(tube_idx)]
        match_steps = self.sequence_match_steps[int(tube_idx)]
        dists = self._distance(x, match_points)
        best_local = int(np.argmin(dists))
        best_rho = float(dists[best_local] / (self.radii[int(tube_idx)] + self.eps))
        best_step = int(match_steps[best_local])
        return best_step, best_rho

    def _sequence_match_all(self, x_current):
        n_tubes = self.centers.shape[0]
        rhos = np.empty(n_tubes, dtype=np.float32)
        steps = np.zeros(n_tubes, dtype=np.int32)
        for tube_idx in range(n_tubes):
            best_step, best_rho = self._tube_sequence_match(tube_idx, x_current)
            rhos[tube_idx] = best_rho
            steps[tube_idx] = best_step
        return rhos, steps

    def _selection_scores(self, rhos):
        scores = np.asarray(rhos, dtype=np.float32)
        if self.trajectory_rank_weight <= 0.0 or self.trajectory_ranks.size == 0:
            return scores
        max_rank = float(np.max(self.trajectory_ranks))
        if max_rank <= 0.0:
            return scores
        rank_norm = self.trajectory_ranks / max_rank
        return scores * (1.0 + self.trajectory_rank_weight * rank_norm)

    def select_tube(self, x_current):
        x = np.asarray(x_current, dtype=np.float32).reshape(self.state_dim)
        if self.sequence_match_enabled:
            rhos, match_steps = self._sequence_match_all(x)
        else:
            dists = self._distance(x, self.centers)
            rhos = dists / (self.radii + self.eps)
            match_steps = np.zeros_like(rhos, dtype=np.int32)

        if self.prefer_earliest_within_support and self.trajectory_ranks.size == rhos.size:
            covered = np.where(rhos <= 1.0)[0]
            if covered.size > 0:
                covered_ranks = self.trajectory_ranks[covered]
                best_rank = float(np.min(covered_ranks))
                candidate_idxs = covered[covered_ranks <= best_rank + 1e-8]
                best_local = int(np.argmin(rhos[candidate_idxs]))
                best_idx = int(candidate_idxs[best_local])
                return best_idx, float(rhos[best_idx]), int(match_steps[best_idx])

        selection_scores = self._selection_scores(rhos)
        if self.selection_mode != "stitch":
            best_idx = int(np.argmin(selection_scores))
            best_rho = float(rhos[best_idx])
            return best_idx, best_rho, int(match_steps[best_idx])

        top_k = min(max(self.stitch_top_k, 1), rhos.size)
        candidate_idxs = np.argsort(selection_scores)[:top_k]
        candidate_scores = (
            self.stitch_entry_weight * selection_scores[candidate_idxs]
            + self.stitch_value_weight * self.stitch_values[candidate_idxs]
        )
        best_local = int(np.argmin(candidate_scores))
        best_idx = int(candidate_idxs[best_local])
        best_rho = float(rhos[best_idx])
        return best_idx, best_rho, int(match_steps[best_idx])

    def _apply_tracking(self, tube_idx, step_idx, x_current, u_nominal):
        u = np.asarray(u_nominal, dtype=np.float32).reshape(self.control_dim).copy()
        if self.tracking_gain is not None:
            x = np.asarray(x_current, dtype=np.float32).reshape(self.state_dim)
            x_ref_seq = self.x_seqs[tube_idx]
            ref_idx = min(max(int(step_idx), 0), x_ref_seq.shape[0] - 1)
            x_ref = x_ref_seq[ref_idx]
            err = _state_error_with_angles(x, x_ref, self.angle_indices)
            u = u - self.tracking_gain @ err
        else:
            x = np.asarray(x_current, dtype=np.float32).reshape(self.state_dim)

        if self.project_tracking and self.control_power_direction_fn is not None:
            a = self.control_power_direction_fn(x)
            return _project_control_orthogonal_to_direction(
                u,
                a,
                u_min=self.u_min,
                u_max=self.u_max,
                eps=self.eps,
            )
        return _clip_control(u, self.u_min, self.u_max)

    def _start_sequence(self, tube_idx, x_current, status, rho, start_step=0):
        self.active_idx = int(tube_idx)
        start_step = int(np.clip(start_step, 0, self.u_seqs[self.active_idx].shape[0] - 1))
        self.active_k = start_step + 1
        u0 = self._apply_tracking(
            self.active_idx,
            start_step,
            x_current,
            self.u_seqs[self.active_idx][start_step],
        )
        return u0, status, rho

    def _apply_default_tracking(self, tube_idx, step_idx, x_current):
        x = np.asarray(x_current, dtype=np.float32).reshape(self.state_dim)
        gain = self.outside_tracking_gain if self.outside_tracking_gain is not None else self.tracking_gain
        if gain is None:
            u = self.default_u.copy()
        else:
            x_ref_seq = self.x_seqs[tube_idx]
            ref_idx = min(max(int(step_idx), 0), x_ref_seq.shape[0] - 1)
            x_ref = x_ref_seq[ref_idx]
            err = _state_error_with_angles(x, x_ref, self.angle_indices)
            u = self.default_u.copy() - gain @ err

        if self.project_tracking and self.control_power_direction_fn is not None:
            a = self.control_power_direction_fn(x)
            return _project_control_orthogonal_to_direction(
                u,
                a,
                u_min=self.u_min,
                u_max=self.u_max,
                eps=self.eps,
            )
        return _clip_control(u, self.u_min, self.u_max)

    def get_action(self, x_current):
        x = np.asarray(x_current, dtype=np.float32).reshape(self.state_dim)
        if self.execute_full_sequence and self.active_idx is not None:
            active_ref_idx = min(self.active_k, self.x_seqs[self.active_idx].shape[0] - 1)
            active_center = self.x_seqs[self.active_idx][active_ref_idx]
            active_rho = float(
                self._distance(x, active_center[None, :])[0] / (self.radii[self.active_idx] + self.eps)
            )
            if active_rho <= self.abort_threshold:
                u_seq = self.u_seqs[self.active_idx]
                if self.active_k < u_seq.shape[0]:
                    u = self._apply_tracking(self.active_idx, self.active_k, x, u_seq[self.active_k])
                    self.active_k += 1
                    return u, "Expert", active_rho
            elif self.sequence_match_enabled:
                recover_step, recover_rho = self._tube_sequence_match(self.active_idx, x)
                u_seq = self.u_seqs[self.active_idx]
                if recover_rho <= self.abort_threshold and recover_step < u_seq.shape[0]:
                    self.active_k = int(recover_step) + 1
                    u = self._apply_tracking(self.active_idx, recover_step, x, u_seq[recover_step])
                    return u, "Expert", recover_rho
            self.active_idx = None
            self.active_k = 0

        best_idx, best_rho, best_step = self.select_tube(x)
        if best_rho <= self.enter_threshold:
            if self.execute_full_sequence:
                return self._start_sequence(best_idx, x, "Expert", best_rho, start_step=best_step)
            return self.u_controls[best_idx].copy(), "Expert", best_rho

        if self.outside_mode == "nearest_control":
            return self.u_controls[best_idx].copy(), "Nearest", best_rho
        if self.outside_mode == "nearest_tracking":
            return self._apply_default_tracking(best_idx, best_step, x), "DefaultTrack", best_rho
        if self.outside_mode == "nearest_sequence" and self.execute_full_sequence:
            return self._start_sequence(best_idx, x, "Nearest", best_rho, start_step=best_step)
        if self.outside_mode in {"default", "zero"}:
            return self.default_u.copy(), "Default", best_rho
        if self.outside_mode == "backup" and self.backup_controller is not None:
            return self.backup_controller.get_action(x).copy(), "Backup", best_rho
        return self.default_u.copy(), "Default", best_rho


def _run_chain_episode(spec: ChainPolicySpec, env, controller: TubeController, x0):
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

    return {
        "success": bool(has_succeeded),
        "expert_ratio": expert_steps / max(total_steps, 1),
        "guided_ratio": guided_steps / max(total_steps, 1),
        "rho_min": rho_min_this if np.isfinite(rho_min_this) else np.nan,
    }


def _run_chain_episode_fast(
    spec: ChainPolicySpec,
    controller: TubeController,
    x0,
    dt,
    episode_seconds,
):
    obs = np.asarray(x0, dtype=np.float32).reshape(spec.state_dim)
    controller.reset_episode(x0=obs)

    horizon_steps = max(int(np.round(float(episode_seconds) / float(dt))), 1)
    has_succeeded = spec.check_success(obs)
    expert_steps = 0
    guided_steps = 0
    total_steps = 0
    rho_min_this = np.inf

    for _ in range(horizon_steps):
        u, status, rho = controller.get_action(obs)
        if status == "Expert":
            expert_steps += 1
        if status in {"Expert", "Nearest", "Backup"}:
            guided_steps += 1
        total_steps += 1
        if np.isfinite(rho):
            rho_min_this = min(rho_min_this, rho)

        obs = _rk4_step(spec.f_continuous_fn, obs, u, dt)
        if spec.check_success(obs):
            has_succeeded = True
            break

    return {
        "success": bool(has_succeeded),
        "expert_ratio": expert_steps / max(total_steps, 1),
        "guided_ratio": guided_steps / max(total_steps, 1),
        "rho_min": rho_min_this if np.isfinite(rho_min_this) else np.nan,
    }


def _evaluate_chain_policy_chunk_fast(
    spec: ChainPolicySpec,
    alphabet_dict,
    init_states,
    dt,
    episode_seconds,
    controller_kwargs,
):
    controller = TubeController(alphabet_dict, **controller_kwargs)
    success_count = 0
    expert_ratios = []
    guided_ratios = []
    rho_mins = []
    for x0 in init_states:
        episode = _run_chain_episode_fast(
            spec=spec,
            controller=controller,
            x0=x0,
            dt=dt,
            episode_seconds=episode_seconds,
        )
        if episode["success"]:
            success_count += 1
        expert_ratios.append(float(episode["expert_ratio"]))
        guided_ratios.append(float(episode["guided_ratio"]))
        rho_mins.append(float(episode["rho_min"]))
    return {
        "success_count": int(success_count),
        "expert_ratios": expert_ratios,
        "guided_ratios": guided_ratios,
        "rho_mins": rho_mins,
    }


def _single_success_batch(spec: ChainPolicySpec, states):
    return success_mask_from_spec(spec, states)


def _single_distance_batch(states, refs, weights):
    X = np.asarray(states, dtype=np.float32)
    R = np.asarray(refs, dtype=np.float32)
    diff = R[None, :, :] - X[:, None, :]
    diff[:, :, 0] = angle_normalize(diff[:, :, 0])
    diff *= np.asarray(weights, dtype=np.float32).reshape(1, 1, -1)
    return np.linalg.norm(diff, axis=2).astype(np.float32)


def _single_distance_ref_batch(states, refs, weights):
    X = np.asarray(states, dtype=np.float32)
    R = np.asarray(refs, dtype=np.float32)
    diff = R - X
    diff[:, 0] = angle_normalize(diff[:, 0])
    diff *= np.asarray(weights, dtype=np.float32).reshape(1, -1)
    return np.linalg.norm(diff, axis=1).astype(np.float32)


def _build_single_support_tree(centers, radii, weights, enter_threshold=1.0):
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return None

    centers = np.asarray(centers, dtype=np.float32)
    radii = np.asarray(radii, dtype=np.float32).reshape(-1)
    weights = np.asarray(weights, dtype=np.float32).reshape(1, -1)
    theta_offsets = np.asarray([-2.0 * np.pi, 0.0, 2.0 * np.pi], dtype=np.float32)

    tiled_centers = []
    tiled_indices = []
    tiled_radii = []
    for offset in theta_offsets:
        shifted = centers.copy()
        shifted[:, 0] = shifted[:, 0] + offset
        tiled_centers.append(shifted)
        tiled_indices.append(np.arange(centers.shape[0], dtype=np.int32))
        tiled_radii.append(radii)

    tiled_centers = np.concatenate(tiled_centers, axis=0)
    tree_points = tiled_centers * weights
    query_radius = float(np.max(radii) * max(float(enter_threshold), 1.0))
    if query_radius <= 0.0 or not np.isfinite(query_radius):
        return None
    return {
        "tree": cKDTree(tree_points),
        "tiled_centers": tiled_centers,
        "tiled_indices": np.concatenate(tiled_indices, axis=0),
        "tiled_radii": np.concatenate(tiled_radii, axis=0),
        "weights": weights.reshape(-1),
        "query_radius": query_radius,
    }


def _query_single_support_tree(support_tree, states, eps=1e-9):
    states = np.asarray(states, dtype=np.float32)
    weights = np.asarray(support_tree["weights"], dtype=np.float32).reshape(1, -1)
    query_points = states * weights
    candidate_lists = support_tree["tree"].query_ball_point(
        query_points,
        r=float(support_tree["query_radius"]),
    )
    best_idxs = np.full(states.shape[0], -1, dtype=np.int32)
    best_rhos = np.full(states.shape[0], np.inf, dtype=np.float32)

    tiled_centers = np.asarray(support_tree["tiled_centers"], dtype=np.float32)
    tiled_indices = np.asarray(support_tree["tiled_indices"], dtype=np.int32)
    tiled_radii = np.asarray(support_tree["tiled_radii"], dtype=np.float32)
    weight_vec = np.asarray(support_tree["weights"], dtype=np.float32).reshape(1, -1)

    for row_idx, candidates in enumerate(candidate_lists):
        if len(candidates) == 0:
            continue
        candidates = np.asarray(candidates, dtype=np.int32)
        diff = tiled_centers[candidates] - states[row_idx][None, :]
        diff[:, 0] = angle_normalize(diff[:, 0])
        dists = np.linalg.norm(diff * weight_vec, axis=1)
        rhos = dists / (tiled_radii[candidates] + float(eps))
        best_local = int(np.argmin(rhos))
        best_idxs[row_idx] = int(tiled_indices[candidates[best_local]])
        best_rhos[row_idx] = float(rhos[best_local])

    return best_idxs, best_rhos


def _single_rk4_step_batch(states, actions, dt):
    X = np.asarray(states, dtype=np.float32)
    U = np.asarray(actions, dtype=np.float32).reshape(-1, 1)
    dt = float(dt)

    def f_batch(x_batch, u_batch):
        theta = x_batch[:, 0]
        p = x_batch[:, 1]
        tau = u_batch[:, 0]
        theta_dot = p / SINGLE_I
        p_dot = tau - SINGLE_M * SINGLE_G * SINGLE_L * np.sin(theta)
        return np.stack([theta_dot, p_dot], axis=1).astype(np.float32)

    k1 = f_batch(X, U)
    k2 = f_batch(X + 0.5 * dt * k1, U)
    k3 = f_batch(X + 0.5 * dt * k2, U)
    k4 = f_batch(X + dt * k3, U)
    return (X + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)).astype(np.float32)


def _double_success_batch(spec: ChainPolicySpec, states):
    return success_mask_from_spec(spec, states)


def _double_distance_batch(states, refs, weights):
    X = np.asarray(states, dtype=np.float32)
    R = np.asarray(refs, dtype=np.float32)
    diff = R[None, :, :] - X[:, None, :]
    diff[:, :, 0] = angle_normalize(diff[:, :, 0])
    diff[:, :, 1] = angle_normalize(diff[:, :, 1])
    diff *= np.asarray(weights, dtype=np.float32).reshape(1, 1, -1)
    return np.linalg.norm(diff, axis=2).astype(np.float32)


def _double_distance_ref_batch(states, refs, weights):
    X = np.asarray(states, dtype=np.float32)
    R = np.asarray(refs, dtype=np.float32)
    diff = R - X
    diff[:, 0] = angle_normalize(diff[:, 0])
    diff[:, 1] = angle_normalize(diff[:, 1])
    diff *= np.asarray(weights, dtype=np.float32).reshape(1, -1)
    return np.linalg.norm(diff, axis=1).astype(np.float32)


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


def _double_rk4_step_batch(states, actions, dt):
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


def _double_state_error_ref_batch(states, refs):
    diff = np.asarray(states, dtype=np.float32) - np.asarray(refs, dtype=np.float32)
    diff[:, 0] = angle_normalize(diff[:, 0])
    diff[:, 1] = angle_normalize(diff[:, 1])
    return diff.astype(np.float32)


def _clip_control_batch(actions, u_min=None, u_max=None):
    U = np.asarray(actions, dtype=np.float32).copy()
    if u_min is not None:
        U = np.maximum(U, np.asarray(u_min, dtype=np.float32).reshape(1, -1))
    if u_max is not None:
        U = np.minimum(U, np.asarray(u_max, dtype=np.float32).reshape(1, -1))
    return U.astype(np.float32)


def _scale_controls_to_bounds_batch(actions, u_min=None, u_max=None, eps=1e-9):
    U = np.asarray(actions, dtype=np.float32).copy()
    alpha = np.ones(U.shape[0], dtype=np.float32)

    if u_max is not None:
        upper = np.asarray(u_max, dtype=np.float32).reshape(1, -1)
        pos_mask = U > float(eps)
        ratios = np.full(U.shape, np.inf, dtype=np.float32)
        np.divide(upper, U, out=ratios, where=pos_mask)
        alpha = np.minimum(alpha, np.min(ratios, axis=1))

    if u_min is not None:
        lower = np.asarray(u_min, dtype=np.float32).reshape(1, -1)
        neg_mask = U < -float(eps)
        ratios = np.full(U.shape, np.inf, dtype=np.float32)
        np.divide(lower, U, out=ratios, where=neg_mask)
        alpha = np.minimum(alpha, np.min(ratios, axis=1))

    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    return (alpha[:, None] * U).astype(np.float32)


def _double_control_power_direction_batch(states):
    X = np.asarray(states, dtype=np.float32)
    th1 = X[:, 0]
    th2 = X[:, 1]
    p1 = X[:, 2]
    p2 = X[:, 3]

    a11 = np.float32((DOUBLE_M1 + DOUBLE_M2) * DOUBLE_L1**2)
    a22 = np.float32(DOUBLE_M2 * DOUBLE_L2**2)
    coupling = np.float32(DOUBLE_M2 * DOUBLE_L1 * DOUBLE_L2) * np.cos(th1 - th2)
    det = a11 * a22 - coupling * coupling
    det = np.where(np.abs(det) < 1e-8, 1e-8, det).astype(np.float32)

    q1_dot = (a22 * p1 - coupling * p2) / det
    q2_dot = (-coupling * p1 + a11 * p2) / det
    return np.stack([q1_dot, q2_dot], axis=1).astype(np.float32)


def _project_controls_orthogonal_to_direction_batch(actions, directions, u_min=None, u_max=None, eps=1e-9):
    U = np.asarray(actions, dtype=np.float32).copy()
    A = np.asarray(directions, dtype=np.float32).reshape(U.shape)
    norm_sq = np.sum(A * A, axis=1)
    valid = np.isfinite(norm_sq) & (norm_sq > float(eps))

    out = _clip_control_batch(U, u_min=u_min, u_max=u_max)
    if not np.any(valid):
        return out

    U_proj = U[valid].copy()
    A_valid = A[valid]
    coeff = np.sum(A_valid * U_proj, axis=1) / norm_sq[valid]
    U_proj = U_proj - coeff[:, None] * A_valid
    out[valid] = _scale_controls_to_bounds_batch(U_proj, u_min=u_min, u_max=u_max, eps=eps)
    return out.astype(np.float32)


def _gather_double_refs_and_controls(controller, tube_indices, step_indices):
    refs = []
    nominals = []
    tube_indices = np.asarray(tube_indices, dtype=np.int32)
    step_indices = np.asarray(step_indices, dtype=np.int32)
    for tube_idx, step_idx in zip(tube_indices, step_indices):
        x_seq = controller.x_seqs[int(tube_idx)]
        u_seq = controller.u_seqs[int(tube_idx)]
        ref_idx = min(max(int(step_idx), 0), x_seq.shape[0] - 1)
        u_idx = min(max(int(step_idx), 0), u_seq.shape[0] - 1)
        refs.append(x_seq[ref_idx])
        nominals.append(u_seq[u_idx])
    return np.asarray(refs, dtype=np.float32), np.asarray(nominals, dtype=np.float32)


def _double_apply_tracking_batch(controller, states, refs, u_nominal):
    U = np.asarray(u_nominal, dtype=np.float32).copy()
    X = np.asarray(states, dtype=np.float32)
    R = np.asarray(refs, dtype=np.float32)
    if controller.tracking_gain is not None:
        err = _double_state_error_ref_batch(X, R)
        U = U - err @ np.asarray(controller.tracking_gain, dtype=np.float32).T

    if controller.project_tracking and controller.control_power_direction_fn is not None:
        directions = _double_control_power_direction_batch(X)
        return _project_controls_orthogonal_to_direction_batch(
            U,
            directions,
            u_min=controller.u_min,
            u_max=controller.u_max,
            eps=controller.eps,
        )
    return _clip_control_batch(U, controller.u_min, controller.u_max)


def _double_apply_default_tracking_batch(controller, states, refs):
    X = np.asarray(states, dtype=np.float32)
    R = np.asarray(refs, dtype=np.float32)
    gain = controller.outside_tracking_gain if controller.outside_tracking_gain is not None else controller.tracking_gain
    if gain is None:
        U = np.tile(controller.default_u.reshape(1, -1), (X.shape[0], 1)).astype(np.float32)
    else:
        err = _double_state_error_ref_batch(X, R)
        U = np.tile(controller.default_u.reshape(1, -1), (X.shape[0], 1)).astype(np.float32)
        U = U - err @ np.asarray(gain, dtype=np.float32).T

    if controller.project_tracking and controller.control_power_direction_fn is not None:
        directions = _double_control_power_direction_batch(X)
        return _project_controls_orthogonal_to_direction_batch(
            U,
            directions,
            u_min=controller.u_min,
            u_max=controller.u_max,
            eps=controller.eps,
        )
    return _clip_control_batch(U, controller.u_min, controller.u_max)


def _evaluate_chain_policy_subset_single_fast_batch(
    spec: ChainPolicySpec,
    alphabet_dict,
    init_states,
    dt,
    episode_seconds,
    controller_kwargs,
):
    controller = TubeController(alphabet_dict, **controller_kwargs)
    if (
        controller.state_dim != 2
        or controller.control_dim != 1
        or controller.selection_mode == "stitch"
        or controller.outside_mode not in {"default", "zero"}
        or controller.tracking_gain is not None
        or controller.project_tracking
    ):
        return None

    states = np.asarray(init_states, dtype=np.float32).copy()
    n_states = states.shape[0]
    horizon_steps = max(int(np.round(float(episode_seconds) / float(dt))), 1)
    success = _single_success_batch(spec, states)

    centers = controller.centers
    radii = controller.radii
    weights = controller.distance_weights
    eps = float(controller.eps)
    u_lengths = np.asarray([seq.shape[0] for seq in controller.u_seqs], dtype=np.int32)
    traj_ranks = controller.trajectory_ranks.astype(np.float32)
    max_rank = float(np.max(traj_ranks)) if traj_ranks.size else 0.0
    sequence_match_enabled = bool(controller.sequence_match_enabled)
    rank_scale = 1.0 + controller.trajectory_rank_weight * (
        traj_ranks / max(max_rank, 1.0)
    ) if controller.trajectory_rank_weight > 0.0 and max_rank > 0.0 else np.ones_like(traj_ranks)

    seq_points_flat = None
    seq_tube_flat = None
    seq_step_flat = None
    seq_rank_scale_flat = None
    if sequence_match_enabled:
        seq_points = []
        seq_tubes = []
        seq_steps = []
        seq_rank_scale = []
        for tube_idx, (match_points, match_steps) in enumerate(
            zip(controller.sequence_match_points, controller.sequence_match_steps)
        ):
            if match_points.size == 0:
                continue
            n_match = match_points.shape[0]
            seq_points.append(np.asarray(match_points, dtype=np.float32))
            seq_tubes.append(np.full(n_match, tube_idx, dtype=np.int32))
            seq_steps.append(np.asarray(match_steps, dtype=np.int32))
            seq_rank_scale.append(
                np.full(n_match, rank_scale[tube_idx], dtype=np.float32)
            )
        if seq_points:
            seq_points_flat = np.concatenate(seq_points, axis=0)
            seq_tube_flat = np.concatenate(seq_tubes, axis=0)
            seq_step_flat = np.concatenate(seq_steps, axis=0)
            seq_rank_scale_flat = np.concatenate(seq_rank_scale, axis=0)
        else:
            sequence_match_enabled = False

    support_tree = None
    sequence_support_tree = None
    if (
        not sequence_match_enabled
        and controller.selection_mode == "nearest"
        and controller.trajectory_rank_weight <= 0.0
        and not controller.prefer_earliest_within_support
    ):
        support_tree = _build_single_support_tree(
            centers,
            radii,
            weights,
            enter_threshold=controller.enter_threshold,
        )
    elif (
        sequence_match_enabled
        and controller.selection_mode == "nearest"
        and controller.trajectory_rank_weight <= 0.0
        and not controller.prefer_earliest_within_support
        and seq_points_flat is not None
        and seq_tube_flat is not None
    ):
        sequence_support_tree = _build_single_support_tree(
            seq_points_flat,
            radii[seq_tube_flat],
            weights,
            enter_threshold=controller.enter_threshold,
        )

    active_idx = np.full(n_states, -1, dtype=np.int32)
    active_k = np.zeros(n_states, dtype=np.int32)
    expert_steps = np.zeros(n_states, dtype=np.float32)
    guided_steps = np.zeros(n_states, dtype=np.float32)
    total_steps = np.zeros(n_states, dtype=np.float32)
    rho_min = np.full(n_states, np.inf, dtype=np.float32)

    for _ in range(horizon_steps):
        pending = ~success
        if not np.any(pending):
            break

        actions = np.tile(controller.default_u.reshape(1, -1), (n_states, 1)).astype(np.float32)
        status_expert = np.zeros(n_states, dtype=bool)
        rho_now = np.full(n_states, np.inf, dtype=np.float32)
        need_select = pending.copy()

        active_mask = pending & (active_idx >= 0)
        if np.any(active_mask):
            idxs = np.flatnonzero(active_mask)
            if np.isinf(controller.abort_threshold):
                for state_idx in idxs:
                    tube_idx = int(active_idx[state_idx])
                    seq_u = controller.u_seqs[tube_idx]
                    if int(active_k[state_idx]) < seq_u.shape[0]:
                        actions[state_idx] = controller._apply_tracking(
                            tube_idx,
                            int(active_k[state_idx]),
                            states[state_idx],
                            seq_u[int(active_k[state_idx])],
                        )
                        active_k[state_idx] += 1
                        status_expert[state_idx] = True
                        rho_now[state_idx] = 0.0
                        need_select[state_idx] = False
                    else:
                        active_idx[state_idx] = -1
                        active_k[state_idx] = 0
            else:
                refs = []
                for state_idx in idxs:
                    tube_idx = int(active_idx[state_idx])
                    seq = controller.x_seqs[tube_idx]
                    ref_idx = min(int(active_k[state_idx]), seq.shape[0] - 1)
                    refs.append(seq[ref_idx])
                refs = np.asarray(refs, dtype=np.float32)
                active_dists = _single_distance_ref_batch(states[idxs], refs, weights)
                active_rhos = active_dists / (radii[active_idx[idxs]] + eps)
                for local_j, state_idx in enumerate(idxs):
                    tube_idx = int(active_idx[state_idx])
                    seq_u = controller.u_seqs[tube_idx]
                    rho_val = float(active_rhos[local_j])
                    if rho_val <= controller.abort_threshold and int(active_k[state_idx]) < seq_u.shape[0]:
                        actions[state_idx] = controller._apply_tracking(
                            tube_idx,
                            int(active_k[state_idx]),
                            states[state_idx],
                            seq_u[int(active_k[state_idx])],
                        )
                        active_k[state_idx] += 1
                        status_expert[state_idx] = True
                        rho_now[state_idx] = rho_val
                        need_select[state_idx] = False
                    else:
                        recovered = False
                        if sequence_match_enabled:
                            match_points = controller.sequence_match_points[tube_idx]
                            match_steps = controller.sequence_match_steps[tube_idx]
                            if match_points.size > 0:
                                recover_dists = _single_distance_batch(
                                    states[state_idx : state_idx + 1], match_points, weights
                                )[0]
                                best_local = int(np.argmin(recover_dists))
                                recover_step = int(match_steps[best_local])
                                recover_rho = float(
                                    recover_dists[best_local] / (radii[tube_idx] + eps)
                                )
                                if recover_rho <= controller.abort_threshold and recover_step < seq_u.shape[0]:
                                    active_k[state_idx] = recover_step + 1
                                    actions[state_idx] = controller._apply_tracking(
                                        tube_idx,
                                        recover_step,
                                        states[state_idx],
                                        seq_u[recover_step],
                                    )
                                    status_expert[state_idx] = True
                                    rho_now[state_idx] = recover_rho
                                    need_select[state_idx] = False
                                    recovered = True
                        if not recovered:
                            active_idx[state_idx] = -1
                            active_k[state_idx] = 0

        select_mask = pending & need_select
        if np.any(select_mask):
            idxs = np.flatnonzero(select_mask)
            if sequence_match_enabled:
                if sequence_support_tree is not None:
                    best_points, best_rhos = _query_single_support_tree(
                        sequence_support_tree,
                        states[idxs],
                        eps=eps,
                    )
                    for row_j, state_idx in enumerate(idxs):
                        best_point = int(best_points[row_j])
                        best_rho = float(best_rhos[row_j])
                        if best_point >= 0 and best_rho <= controller.enter_threshold:
                            best_idx = int(seq_tube_flat[best_point])
                            best_step = int(seq_step_flat[best_point])
                            seq_u = controller.u_seqs[best_idx]
                            if best_step < seq_u.shape[0]:
                                active_idx[state_idx] = best_idx
                                active_k[state_idx] = best_step + 1
                                actions[state_idx] = controller._apply_tracking(
                                    best_idx,
                                    best_step,
                                    states[state_idx],
                                    seq_u[best_step],
                                )
                                status_expert[state_idx] = True
                                rho_now[state_idx] = best_rho
                else:
                    flat_rhos = _single_distance_batch(states[idxs], seq_points_flat, weights) / (
                        radii[seq_tube_flat][None, :] + eps
                    )
                    for row_j, state_idx in enumerate(idxs):
                        row_rhos = flat_rhos[row_j]
                        if controller.prefer_earliest_within_support and traj_ranks.size == radii.size:
                            covered_points = np.where(row_rhos <= 1.0)[0]
                            if covered_points.size > 0:
                                covered_ranks = traj_ranks[seq_tube_flat[covered_points]]
                                best_rank = float(np.min(covered_ranks))
                                candidates = covered_points[covered_ranks <= best_rank + 1e-8]
                                best_point = int(candidates[int(np.argmin(row_rhos[candidates]))])
                            else:
                                scores = row_rhos * seq_rank_scale_flat
                                best_point = int(np.argmin(scores))
                        else:
                            scores = row_rhos * seq_rank_scale_flat
                            best_point = int(np.argmin(scores))

                        best_idx = int(seq_tube_flat[best_point])
                        best_step = int(seq_step_flat[best_point])
                        best_rho = float(row_rhos[best_point])
                        if best_rho <= controller.enter_threshold:
                            seq_u = controller.u_seqs[best_idx]
                            if best_step < seq_u.shape[0]:
                                active_idx[state_idx] = best_idx
                                active_k[state_idx] = best_step + 1
                                actions[state_idx] = controller._apply_tracking(
                                    best_idx,
                                    best_step,
                                    states[state_idx],
                                    seq_u[best_step],
                                )
                                status_expert[state_idx] = True
                                rho_now[state_idx] = best_rho
            else:
                if support_tree is not None:
                    best_idxs, best_rhos = _query_single_support_tree(
                        support_tree,
                        states[idxs],
                        eps=eps,
                    )
                    for row_j, state_idx in enumerate(idxs):
                        best_idx = int(best_idxs[row_j])
                        best_rho = float(best_rhos[row_j])
                        if best_idx >= 0 and best_rho <= controller.enter_threshold:
                            seq_u = controller.u_seqs[best_idx]
                            active_idx[state_idx] = best_idx
                            active_k[state_idx] = 1
                            actions[state_idx] = controller._apply_tracking(
                                best_idx,
                                0,
                                states[state_idx],
                                seq_u[0],
                            )
                            status_expert[state_idx] = True
                            rho_now[state_idx] = best_rho
                else:
                    rhos = _single_distance_batch(states[idxs], centers, weights) / (radii[None, :] + eps)
                    for row_j, state_idx in enumerate(idxs):
                        row_rhos = rhos[row_j]
                        if controller.prefer_earliest_within_support and traj_ranks.size == row_rhos.size:
                            covered = np.where(row_rhos <= 1.0)[0]
                            if covered.size > 0:
                                covered_ranks = traj_ranks[covered]
                                best_rank = float(np.min(covered_ranks))
                                candidates = covered[covered_ranks <= best_rank + 1e-8]
                                best_idx = int(candidates[int(np.argmin(row_rhos[candidates]))])
                                best_rho = float(row_rhos[best_idx])
                            else:
                                scores = row_rhos * rank_scale
                                best_idx = int(np.argmin(scores))
                                best_rho = float(row_rhos[best_idx])
                        else:
                            scores = row_rhos * rank_scale
                            best_idx = int(np.argmin(scores))
                            best_rho = float(row_rhos[best_idx])

                        if best_rho <= controller.enter_threshold:
                            seq_u = controller.u_seqs[best_idx]
                            active_idx[state_idx] = best_idx
                            active_k[state_idx] = 1
                            actions[state_idx] = controller._apply_tracking(
                                best_idx,
                                0,
                                states[state_idx],
                                seq_u[0],
                            )
                            status_expert[state_idx] = True
                            rho_now[state_idx] = best_rho

        total_steps[pending] += 1.0
        expert_steps[status_expert] += 1.0
        guided_steps[status_expert] += 1.0
        finite_rho = np.isfinite(rho_now)
        rho_min[finite_rho] = np.minimum(rho_min[finite_rho], rho_now[finite_rho])

        states[pending] = _single_rk4_step_batch(states[pending], actions[pending], dt)
        success[pending] |= _single_success_batch(spec, states[pending])

    expert_ratios = expert_steps / np.maximum(total_steps, 1.0)
    guided_ratios = guided_steps / np.maximum(total_steps, 1.0)
    return {
        "success_rate": float(np.mean(success)),
        "expert_ratio": float(np.nanmean(expert_ratios)),
        "guided_ratio": float(np.nanmean(guided_ratios)),
        "rho_med": float(np.nanmedian(np.where(np.isfinite(rho_min), rho_min, np.nan))),
    }


def _evaluate_chain_policy_subset_double_fast_batch(
    spec: ChainPolicySpec,
    alphabet_dict,
    init_states,
    dt,
    episode_seconds,
    controller_kwargs,
):
    controller = TubeController(alphabet_dict, **controller_kwargs)
    if (
        controller.state_dim != 4
        or controller.control_dim != 2
        or controller.selection_mode == "stitch"
        or not controller.execute_full_sequence
        or controller.outside_mode not in {"default", "zero", "nearest_tracking"}
    ):
        return None

    states = np.asarray(init_states, dtype=np.float32).copy()
    n_states = states.shape[0]
    horizon_steps = max(int(np.round(float(episode_seconds) / float(dt))), 1)
    success = _double_success_batch(spec, states)

    centers = controller.centers
    radii = controller.radii
    weights = controller.distance_weights
    eps = float(controller.eps)
    u_lengths = np.asarray([seq.shape[0] for seq in controller.u_seqs], dtype=np.int32)
    traj_ranks = controller.trajectory_ranks.astype(np.float32)
    max_rank = float(np.max(traj_ranks)) if traj_ranks.size else 0.0
    sequence_match_enabled = bool(controller.sequence_match_enabled)
    rank_scale = (
        1.0 + controller.trajectory_rank_weight * (traj_ranks / max(max_rank, 1.0))
        if controller.trajectory_rank_weight > 0.0 and max_rank > 0.0
        else np.ones_like(traj_ranks)
    )

    seq_points_flat = None
    seq_tube_flat = None
    seq_step_flat = None
    seq_rank_scale_flat = None
    if sequence_match_enabled:
        seq_points = []
        seq_tubes = []
        seq_steps = []
        seq_rank_scale = []
        for tube_idx, (match_points, match_steps) in enumerate(
            zip(controller.sequence_match_points, controller.sequence_match_steps)
        ):
            if match_points.size == 0:
                continue
            n_match = match_points.shape[0]
            seq_points.append(np.asarray(match_points, dtype=np.float32))
            seq_tubes.append(np.full(n_match, tube_idx, dtype=np.int32))
            seq_steps.append(np.asarray(match_steps, dtype=np.int32))
            seq_rank_scale.append(np.full(n_match, rank_scale[tube_idx], dtype=np.float32))
        if seq_points:
            seq_points_flat = np.concatenate(seq_points, axis=0)
            seq_tube_flat = np.concatenate(seq_tubes, axis=0)
            seq_step_flat = np.concatenate(seq_steps, axis=0)
            seq_rank_scale_flat = np.concatenate(seq_rank_scale, axis=0)
        else:
            sequence_match_enabled = False

    trajectory_ref_paths = []
    ref_limit = int(max(controller.match_sequence_tube_limit, 1))
    ref_stride = max(int(controller.sequence_match_stride), 1)
    for tube_list in alphabet_dict.values():
        ordered = sorted(
            tube_list,
            key=lambda tube: int(tube.get("start_idx", 0) or 0),
        )
        if not ordered:
            continue
        ref_points = [
            np.asarray(tube["x_center"], dtype=np.float32).reshape(-1)
            for tube in ordered
        ]
        ref_points.append(
            np.asarray(ordered[-1].get("x_end", ref_points[-1]), dtype=np.float32).reshape(-1)
        )
        ref_points = np.asarray(ref_points, dtype=np.float32)
        if ref_stride > 1 and ref_points.shape[0] > 1:
            steps = np.arange(0, ref_points.shape[0], ref_stride, dtype=np.int32)
            if steps.size == 0 or int(steps[-1]) != ref_points.shape[0] - 1:
                steps = np.append(steps, np.int32(ref_points.shape[0] - 1))
            ref_points = ref_points[steps]
        if ref_points.shape[0] > ref_limit:
            idx = np.linspace(0, ref_points.shape[0] - 1, num=ref_limit, dtype=int)
            ref_points = ref_points[idx]
        trajectory_ref_paths.append(ref_points)
    trajectory_ref_points = (
        np.concatenate(trajectory_ref_paths, axis=0).astype(np.float32)
        if trajectory_ref_paths
        else None
    )

    active_idx = np.full(n_states, -1, dtype=np.int32)
    active_k = np.zeros(n_states, dtype=np.int32)
    expert_steps = np.zeros(n_states, dtype=np.float32)
    guided_steps = np.zeros(n_states, dtype=np.float32)
    total_steps = np.zeros(n_states, dtype=np.float32)
    rho_min = np.full(n_states, np.inf, dtype=np.float32)

    for _ in range(horizon_steps):
        pending = ~success
        if not np.any(pending):
            break

        actions = np.tile(controller.default_u.reshape(1, -1), (n_states, 1)).astype(np.float32)
        status_expert = np.zeros(n_states, dtype=bool)
        rho_now = np.full(n_states, np.inf, dtype=np.float32)
        need_select = pending.copy()

        active_mask = pending & (active_idx >= 0)
        if np.any(active_mask):
            idxs = np.flatnonzero(active_mask)
            refs = []
            for state_idx in idxs:
                tube_idx = int(active_idx[state_idx])
                seq = controller.x_seqs[tube_idx]
                ref_idx = min(int(active_k[state_idx]), seq.shape[0] - 1)
                refs.append(seq[ref_idx])
            refs = np.asarray(refs, dtype=np.float32)
            active_dists = _double_distance_ref_batch(states[idxs], refs, weights)
            active_rhos = active_dists / (radii[active_idx[idxs]] + eps)
            for local_j, state_idx in enumerate(idxs):
                tube_idx = int(active_idx[state_idx])
                seq_u = controller.u_seqs[tube_idx]
                rho_val = float(active_rhos[local_j])
                if rho_val <= controller.abort_threshold and int(active_k[state_idx]) < seq_u.shape[0]:
                    step_idx = int(active_k[state_idx])
                    refs_now, nominals_now = _gather_double_refs_and_controls(
                        controller,
                        np.array([tube_idx], dtype=np.int32),
                        np.array([step_idx], dtype=np.int32),
                    )
                    actions[state_idx : state_idx + 1] = _double_apply_tracking_batch(
                        controller,
                        states[state_idx : state_idx + 1],
                        refs_now,
                        nominals_now,
                    )
                    active_k[state_idx] = step_idx + 1
                    status_expert[state_idx] = True
                    rho_now[state_idx] = rho_val
                    need_select[state_idx] = False
                else:
                    recovered = False
                    if sequence_match_enabled:
                        match_points = controller.sequence_match_points[tube_idx]
                        match_steps = controller.sequence_match_steps[tube_idx]
                        if match_points.size > 0:
                            recover_dists = _double_distance_batch(
                                states[state_idx : state_idx + 1],
                                match_points,
                                weights,
                            )[0]
                            best_local = int(np.argmin(recover_dists))
                            recover_step = int(match_steps[best_local])
                            recover_rho = float(recover_dists[best_local] / (radii[tube_idx] + eps))
                            if recover_rho <= controller.abort_threshold and recover_step < seq_u.shape[0]:
                                refs_now, nominals_now = _gather_double_refs_and_controls(
                                    controller,
                                    np.array([tube_idx], dtype=np.int32),
                                    np.array([recover_step], dtype=np.int32),
                                )
                                active_k[state_idx] = recover_step + 1
                                actions[state_idx : state_idx + 1] = _double_apply_tracking_batch(
                                    controller,
                                    states[state_idx : state_idx + 1],
                                    refs_now,
                                    nominals_now,
                                )
                                status_expert[state_idx] = True
                                rho_now[state_idx] = recover_rho
                                need_select[state_idx] = False
                                recovered = True
                    if not recovered:
                        active_idx[state_idx] = -1
                        active_k[state_idx] = 0

        select_mask = pending & need_select
        if np.any(select_mask):
            idxs = np.flatnonzero(select_mask)
            if sequence_match_enabled:
                flat_rhos = _double_distance_batch(states[idxs], seq_points_flat, weights) / (
                    radii[seq_tube_flat][None, :] + eps
                )
                enter_local = np.zeros(idxs.shape[0], dtype=bool)
                for row_j, state_idx in enumerate(idxs):
                    row_rhos = flat_rhos[row_j]
                    if controller.prefer_earliest_within_support and traj_ranks.size == radii.size:
                        covered_points = np.where(row_rhos <= 1.0)[0]
                        if covered_points.size > 0:
                            covered_ranks = traj_ranks[seq_tube_flat[covered_points]]
                            best_rank = float(np.min(covered_ranks))
                            candidates = covered_points[covered_ranks <= best_rank + 1e-8]
                            best_point = int(candidates[int(np.argmin(row_rhos[candidates]))])
                        else:
                            scores = row_rhos * seq_rank_scale_flat
                            best_point = int(np.argmin(scores))
                    else:
                        scores = row_rhos * seq_rank_scale_flat
                        best_point = int(np.argmin(scores))

                    best_idx = int(seq_tube_flat[best_point])
                    best_step = int(seq_step_flat[best_point])
                    best_rho = float(row_rhos[best_point])
                    rho_now[state_idx] = best_rho
                    if best_rho <= controller.enter_threshold:
                        seq_u = controller.u_seqs[best_idx]
                        if best_step < seq_u.shape[0]:
                            refs_now, nominals_now = _gather_double_refs_and_controls(
                                controller,
                                np.array([best_idx], dtype=np.int32),
                                np.array([best_step], dtype=np.int32),
                            )
                            active_idx[state_idx] = best_idx
                            active_k[state_idx] = best_step + 1
                            actions[state_idx : state_idx + 1] = _double_apply_tracking_batch(
                                controller,
                                states[state_idx : state_idx + 1],
                                refs_now,
                                nominals_now,
                            )
                            status_expert[state_idx] = True
                            enter_local[row_j] = True
                if controller.outside_mode == "nearest_tracking" and trajectory_ref_points is not None:
                    default_local = ~enter_local
                    if np.any(default_local):
                        default_state_idxs = idxs[default_local]
                        dists = _double_distance_batch(
                            states[default_state_idxs],
                            trajectory_ref_points,
                            weights,
                        )
                        best_ref = np.argmin(dists, axis=1).astype(np.int32)
                        default_refs = trajectory_ref_points[best_ref]
                        actions[default_state_idxs] = _double_apply_default_tracking_batch(
                            controller,
                            states[default_state_idxs],
                            default_refs,
                        )
            else:
                rhos = _double_distance_batch(states[idxs], centers, weights) / (radii[None, :] + eps)
                scores = rhos * rank_scale.reshape(1, -1)
                best_idxs = np.argmin(scores, axis=1).astype(np.int32)
                row_idx = np.arange(best_idxs.shape[0], dtype=np.int32)
                best_rhos = rhos[row_idx, best_idxs].astype(np.float32)
                rho_now[idxs] = best_rhos

                enter_local = best_rhos <= controller.enter_threshold
                if np.any(enter_local):
                    enter_state_idxs = idxs[enter_local]
                    enter_tubes = best_idxs[enter_local]
                    enter_steps = np.zeros(enter_state_idxs.shape[0], dtype=np.int32)
                    enter_refs, enter_nominals = _gather_double_refs_and_controls(
                        controller,
                        enter_tubes,
                        enter_steps,
                    )
                    actions[enter_state_idxs] = _double_apply_tracking_batch(
                        controller,
                        states[enter_state_idxs],
                        enter_refs,
                        enter_nominals,
                    )
                    active_idx[enter_state_idxs] = enter_tubes
                    active_k[enter_state_idxs] = 1
                    status_expert[enter_state_idxs] = True

                if controller.outside_mode == "nearest_tracking" and trajectory_ref_points is not None:
                    default_local = ~enter_local
                    if np.any(default_local):
                        default_state_idxs = idxs[default_local]
                        dists = _double_distance_batch(
                            states[default_state_idxs],
                            trajectory_ref_points,
                            weights,
                        )
                        best_ref = np.argmin(dists, axis=1).astype(np.int32)
                        default_refs = trajectory_ref_points[best_ref]
                        actions[default_state_idxs] = _double_apply_default_tracking_batch(
                            controller,
                            states[default_state_idxs],
                            default_refs,
                        )

        total_steps[pending] += 1.0
        expert_steps[status_expert] += 1.0
        guided_steps[status_expert] += 1.0
        finite_rho = np.isfinite(rho_now)
        rho_min[finite_rho] = np.minimum(rho_min[finite_rho], rho_now[finite_rho])

        states[pending] = _double_rk4_step_batch(states[pending], actions[pending], dt)
        success[pending] |= _double_success_batch(spec, states[pending])

    expert_ratios = expert_steps / np.maximum(total_steps, 1.0)
    guided_ratios = guided_steps / np.maximum(total_steps, 1.0)
    return {
        "success_rate": float(np.mean(success)),
        "expert_ratio": float(np.nanmean(expert_ratios)),
        "guided_ratio": float(np.nanmean(guided_ratios)),
        "rho_med": float(np.nanmedian(np.where(np.isfinite(rho_min), rho_min, np.nan))),
    }


def _prepare_chain_controller_kwargs(spec: ChainPolicySpec, controller_kwargs=None, verbose=True):
    kwargs = {} if controller_kwargs is None else dict(controller_kwargs)
    kwargs.setdefault("angle_indices", spec.angle_indices)
    kwargs.setdefault("target_state", spec.target_state)
    kwargs.setdefault("success_tol", spec.success_tol)
    kwargs.setdefault("control_power_direction_fn", spec.control_power_direction_fn)
    kwargs.setdefault("distance_weights", np.ones(spec.state_dim, dtype=float))
    kwargs.setdefault("default_u", np.zeros(spec.control_dim, dtype=np.float32))
    kwargs.setdefault("log_stats", bool(verbose))
    return kwargs


def _evaluate_chain_policy_subset(
    spec: ChainPolicySpec,
    alphabet_dict,
    init_states,
    dt=0.02,
    episode_seconds=25.0,
    controller_kwargs=None,
    use_fast_rollout=False,
    fast_num_workers=1,
):
    if use_fast_rollout and spec.name == "single_pendulum":
        if int(fast_num_workers) > 1 and len(init_states) > 0:
            chunks = [
                np.asarray(chunk, dtype=np.float32)
                for chunk in np.array_split(np.asarray(init_states, dtype=np.float32), int(fast_num_workers))
                if len(chunk) > 0
            ]
            if len(chunks) > 1:
                with ThreadPoolExecutor(max_workers=int(fast_num_workers)) as executor:
                    futures = [
                        executor.submit(
                            _evaluate_chain_policy_subset_single_fast_batch,
                            spec,
                            alphabet_dict,
                            chunk,
                            dt,
                            episode_seconds,
                            controller_kwargs,
                        )
                        for chunk in chunks
                    ]
                    results = [future.result() for future in futures]

                success_rates = [float(row["success_rate"]) for row in results]
                expert_ratios = [float(row["expert_ratio"]) for row in results]
                guided_ratios = [float(row["guided_ratio"]) for row in results]
                rho_vals = [float(row["rho_med"]) for row in results if np.isfinite(row["rho_med"])]
                weights = np.asarray([len(chunk) for chunk in chunks], dtype=float)
                rate = float(np.average(success_rates, weights=weights))
                return {
                    "success_rate": rate,
                    "expert_ratio": float(np.average(expert_ratios, weights=weights)),
                    "guided_ratio": float(np.average(guided_ratios, weights=weights)),
                    "rho_med": float(np.nanmedian(rho_vals)) if rho_vals else float("nan"),
                }

        single_batch_metrics = _evaluate_chain_policy_subset_single_fast_batch(
            spec=spec,
            alphabet_dict=alphabet_dict,
            init_states=init_states,
            dt=dt,
            episode_seconds=episode_seconds,
            controller_kwargs=controller_kwargs,
        )
        if single_batch_metrics is not None:
            return single_batch_metrics

    if use_fast_rollout and spec.name == "double_pendulum":
        if int(fast_num_workers) > 1 and len(init_states) > 0:
            chunks = [
                np.asarray(chunk, dtype=np.float32)
                for chunk in np.array_split(np.asarray(init_states, dtype=np.float32), int(fast_num_workers))
                if len(chunk) > 0
            ]
            if len(chunks) > 1:
                with ThreadPoolExecutor(max_workers=int(fast_num_workers)) as executor:
                    futures = [
                        executor.submit(
                            _evaluate_chain_policy_subset_double_fast_batch,
                            spec,
                            alphabet_dict,
                            chunk,
                            dt,
                            episode_seconds,
                            controller_kwargs,
                        )
                        for chunk in chunks
                    ]
                    results = [future.result() for future in futures]

                success_rates = [float(row["success_rate"]) for row in results]
                expert_ratios = [float(row["expert_ratio"]) for row in results]
                guided_ratios = [float(row["guided_ratio"]) for row in results]
                rho_vals = [float(row["rho_med"]) for row in results if np.isfinite(row["rho_med"])]
                weights = np.asarray([len(chunk) for chunk in chunks], dtype=float)
                rate = float(np.average(success_rates, weights=weights))
                return {
                    "success_rate": rate,
                    "expert_ratio": float(np.average(expert_ratios, weights=weights)),
                    "guided_ratio": float(np.average(guided_ratios, weights=weights)),
                    "rho_med": float(np.nanmedian(rho_vals)) if rho_vals else float("nan"),
                }

        double_batch_metrics = _evaluate_chain_policy_subset_double_fast_batch(
            spec=spec,
            alphabet_dict=alphabet_dict,
            init_states=init_states,
            dt=dt,
            episode_seconds=episode_seconds,
            controller_kwargs=controller_kwargs,
        )
        if double_batch_metrics is not None:
            return double_batch_metrics

    if use_fast_rollout and int(fast_num_workers) > 1 and len(init_states) > 0:
        chunks = [
            np.asarray(chunk, dtype=np.float32)
            for chunk in np.array_split(np.asarray(init_states, dtype=np.float32), int(fast_num_workers))
            if len(chunk) > 0
        ]
        if len(chunks) > 1:
            with ProcessPoolExecutor(max_workers=int(fast_num_workers)) as executor:
                futures = [
                    executor.submit(
                        _evaluate_chain_policy_chunk_fast,
                        spec,
                        alphabet_dict,
                        chunk,
                        dt,
                        episode_seconds,
                        controller_kwargs,
                    )
                    for chunk in chunks
                ]
                results = [future.result() for future in futures]

            success_count = sum(int(row["success_count"]) for row in results)
            expert_ratios = [value for row in results for value in row["expert_ratios"]]
            guided_ratios = [value for row in results for value in row["guided_ratios"]]
            rho_mins = [value for row in results for value in row["rho_mins"]]
            rate = success_count / len(init_states)
            return {
                "success_rate": float(rate),
                "expert_ratio": float(np.nanmean(expert_ratios)),
                "guided_ratio": float(np.nanmean(guided_ratios)),
                "rho_med": float(np.nanmedian(rho_mins)),
            }

    controller = TubeController(alphabet_dict, **controller_kwargs)
    env = None if use_fast_rollout else spec.make_env(dt=dt, episode_seconds=episode_seconds)

    success_count = 0
    expert_ratios = []
    guided_ratios = []
    rho_mins = []
    for x0 in init_states:
        if use_fast_rollout:
            episode = _run_chain_episode_fast(
                spec=spec,
                controller=controller,
                x0=x0,
                dt=dt,
                episode_seconds=episode_seconds,
            )
        else:
            episode = _run_chain_episode(spec=spec, env=env, controller=controller, x0=x0)
        if episode["success"]:
            success_count += 1
        expert_ratios.append(float(episode["expert_ratio"]))
        guided_ratios.append(float(episode["guided_ratio"]))
        rho_mins.append(float(episode["rho_min"]))

    rate = success_count / len(init_states)
    return {
        "success_rate": float(rate),
        "expert_ratio": float(np.nanmean(expert_ratios)),
        "guided_ratio": float(np.nanmean(guided_ratios)),
        "rho_med": float(np.nanmedian(rho_mins)),
    }


def rank_expert_trajectories_by_utility(
    spec: ChainPolicySpec,
    save_dir,
    candidate_names,
    init_states,
    dt=0.02,
    episode_seconds=25.0,
    controller_kwargs=None,
    alphabet_filename=None,
    use_fast_rollout=False,
    fast_num_workers=1,
    verbose=True,
):
    alphabet_filename = spec.alphabet_filename if alphabet_filename is None else alphabet_filename
    full_alphabet_dict = load_control_alphabet(save_dir, alphabet_filename=alphabet_filename)
    controller_kwargs = _prepare_chain_controller_kwargs(
        spec=spec,
        controller_kwargs=controller_kwargs,
        verbose=False,
    )

    records = []
    for name in candidate_names:
        if name not in full_alphabet_dict:
            continue
        tube_list = full_alphabet_dict[name]
        if not tube_list:
            continue
        subset = {name: tube_list}
        metrics = _evaluate_chain_policy_subset(
            spec=spec,
            alphabet_dict=subset,
            init_states=init_states,
            dt=dt,
            episode_seconds=episode_seconds,
            controller_kwargs=controller_kwargs,
            use_fast_rollout=use_fast_rollout,
            fast_num_workers=fast_num_workers,
        )
        radii = np.array([tube["radius"] for tube in tube_list], dtype=float)
        records.append(
            {
                "name": name,
                "success_rate": float(metrics["success_rate"]),
                "guided_ratio": float(metrics["guided_ratio"]),
                "expert_ratio": float(metrics["expert_ratio"]),
                "rho_med": float(metrics["rho_med"]),
                "n_tubes": int(len(tube_list)),
                "r_mean": float(np.mean(radii)) if radii.size else 0.0,
            }
        )

    records.sort(
        key=lambda row: (
            row["success_rate"],
            row["guided_ratio"],
            row["n_tubes"],
            row["r_mean"],
            -row["rho_med"] if np.isfinite(row["rho_med"]) else -np.inf,
        ),
        reverse=True,
    )
    ordered_names = [row["name"] for row in records]
    if verbose:
        print(f"[Rank:{spec.name}] ranked names = {ordered_names}")
        for row in records:
            print(
                f"  {row['name']}: rate={row['success_rate']:.3f}, guided={row['guided_ratio']:.3f}, "
                f"n_tubes={row['n_tubes']}, r_mean={row['r_mean']:.3e}"
    )
    return ordered_names, records


def greedy_rank_expert_trajectories(
    spec: ChainPolicySpec,
    save_dir,
    candidate_names,
    init_states,
    dt=0.02,
    episode_seconds=25.0,
    controller_kwargs=None,
    alphabet_filename=None,
    max_drop=0.05,
    stop_on_large_drop=True,
    max_trajectories=None,
    use_fast_rollout=False,
    fast_num_workers=1,
    verbose=True,
):
    alphabet_filename = spec.alphabet_filename if alphabet_filename is None else alphabet_filename
    full_alphabet_dict = load_control_alphabet(save_dir, alphabet_filename=alphabet_filename)
    controller_kwargs = _prepare_chain_controller_kwargs(
        spec=spec,
        controller_kwargs=controller_kwargs,
        verbose=False,
    )

    remaining = [name for name in candidate_names if name in full_alphabet_dict and full_alphabet_dict[name]]
    selected_names = []
    records = []
    prev_rate = 0.0
    limit = None if max_trajectories is None else int(max_trajectories)

    while remaining and (limit is None or len(selected_names) < limit):
        best_row = None
        for name in remaining:
            subset_names = selected_names + [name]
            subset_dict = {
                subset_name: full_alphabet_dict[subset_name]
                for subset_name in subset_names
                if subset_name in full_alphabet_dict
            }
            metrics = _evaluate_chain_policy_subset(
                spec=spec,
                alphabet_dict=subset_dict,
                init_states=init_states,
                dt=dt,
                episode_seconds=episode_seconds,
                controller_kwargs=controller_kwargs,
                use_fast_rollout=use_fast_rollout,
                fast_num_workers=fast_num_workers,
            )
            tube_list = full_alphabet_dict[name]
            radii = np.array([tube["radius"] for tube in tube_list], dtype=float)
            rate = float(metrics["success_rate"])
            drop = max(prev_rate - rate, 0.0)
            row = {
                "name": name,
                "success_rate": rate,
                "drop_vs_prev": float(drop),
                "guided_ratio": float(metrics["guided_ratio"]),
                "expert_ratio": float(metrics["expert_ratio"]),
                "rho_med": float(metrics["rho_med"]),
                "n_tubes": int(len(tube_list)),
                "r_mean": float(np.mean(radii)) if radii.size else 0.0,
            }
            accept = drop <= float(max_drop) + 1e-9
            score = (
                int(accept),
                row["success_rate"],
                row["guided_ratio"],
                -row["drop_vs_prev"],
                row["n_tubes"],
                row["r_mean"],
                -row["rho_med"] if np.isfinite(row["rho_med"]) else -np.inf,
            )
            if best_row is None or score > best_row["score"]:
                best_row = {**row, "score": score}

        if best_row is None:
            break
        if stop_on_large_drop and best_row["drop_vs_prev"] > float(max_drop) + 1e-9:
            if verbose:
                print(
                    f"[GreedyRank:{spec.name}] stopping before adding {best_row['name']} "
                    f"because success would drop by {best_row['drop_vs_prev']:.3f}"
                )
            break

        selected_names.append(best_row["name"])
        remaining.remove(best_row["name"])
        prev_rate = float(best_row["success_rate"])
        records.append({key: value for key, value in best_row.items() if key != "score"})

        if verbose:
            print(
                f"[GreedyRank:{spec.name}] k={len(selected_names)} add {best_row['name']} -> "
                f"rate={best_row['success_rate']:.3f}, drop={best_row['drop_vs_prev']:.3f}, "
                f"guided={best_row['guided_ratio']:.3f}"
            )

    return selected_names, records


def select_chain_trajectory_subset(
    spec: ChainPolicySpec,
    save_dir,
    ordered_names,
    init_states,
    dt=0.02,
    episode_seconds=25.0,
    controller_kwargs=None,
    alphabet_filename=None,
    max_drop=0.05,
    use_fast_rollout=False,
    fast_num_workers=1,
    verbose=True,
):
    alphabet_filename = spec.alphabet_filename if alphabet_filename is None else alphabet_filename
    full_alphabet_dict = load_control_alphabet(save_dir, alphabet_filename=alphabet_filename)
    controller_kwargs = _prepare_chain_controller_kwargs(
        spec=spec,
        controller_kwargs=controller_kwargs,
        verbose=False,
    )

    selected_names = []
    records = []
    prev_rate = 0.0

    for name in ordered_names:
        if name not in full_alphabet_dict or not full_alphabet_dict[name]:
            continue
        candidate_names = selected_names + [name]
        subset_dict = {
            subset_name: full_alphabet_dict[subset_name]
            for subset_name in candidate_names
            if subset_name in full_alphabet_dict
        }
        metrics = _evaluate_chain_policy_subset(
            spec=spec,
            alphabet_dict=subset_dict,
            init_states=init_states,
            dt=dt,
            episode_seconds=episode_seconds,
            controller_kwargs=controller_kwargs,
            use_fast_rollout=use_fast_rollout,
            fast_num_workers=fast_num_workers,
        )
        rate = float(metrics["success_rate"])
        drop = max(prev_rate - rate, 0.0)
        accepted = drop <= float(max_drop) + 1e-9
        records.append(
            {
                "name": name,
                "candidate_rate": rate,
                "drop_vs_prev": float(drop),
                "accepted": bool(accepted),
                "selected_count_if_accepted": int(len(candidate_names)),
            }
        )
        if accepted:
            selected_names.append(name)
            prev_rate = rate
        if verbose:
            print(
                f"[SubsetSelect:{spec.name}] {name}: rate={rate:.3f}, drop={drop:.3f}, "
                f"{'ACCEPT' if accepted else 'REJECT'}"
            )

    return selected_names, records


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
    use_fast_rollout=False,
    fast_num_workers=1,
    plot=True,
    verbose=True,
):
    alphabet_filename = spec.alphabet_filename if alphabet_filename is None else alphabet_filename
    result_filename = spec.success_rate_filename if result_filename is None else result_filename
    full_alphabet_dict = load_control_alphabet(save_dir, alphabet_filename=alphabet_filename)

    controller_kwargs = _prepare_chain_controller_kwargs(
        spec=spec,
        controller_kwargs=controller_kwargs,
        verbose=verbose,
    )

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
        if use_fast_rollout:
            subset_metrics = _evaluate_chain_policy_subset(
                spec=spec,
                alphabet_dict=current_alphabet_dict,
                init_states=init_states,
                dt=dt,
                episode_seconds=episode_seconds,
                controller_kwargs=controller_kwargs,
                use_fast_rollout=use_fast_rollout,
                fast_num_workers=fast_num_workers,
            )
            rate = float(subset_metrics["success_rate"])
            er_mean = float(subset_metrics["expert_ratio"])
            guided_mean = float(subset_metrics["guided_ratio"])
            rho_med = float(subset_metrics["rho_med"])
        elif verbose:
            controller = TubeController(current_alphabet_dict, **controller_kwargs)
            env = None if use_fast_rollout else spec.make_env(dt=dt, episode_seconds=episode_seconds)
            success_count = 0
            expert_ratios = []
            guided_ratios = []
            rho_mins = []
            iterator = tqdm(init_states, desc=f"{spec.name}:k={k}", leave=False)
            for x0 in iterator:
                if use_fast_rollout:
                    episode = _run_chain_episode_fast(
                        spec=spec,
                        controller=controller,
                        x0=x0,
                        dt=dt,
                        episode_seconds=episode_seconds,
                    )
                else:
                    episode = _run_chain_episode(spec=spec, env=env, controller=controller, x0=x0)
                if episode["success"]:
                    success_count += 1
                expert_ratios.append(float(episode["expert_ratio"]))
                guided_ratios.append(float(episode["guided_ratio"]))
                rho_mins.append(float(episode["rho_min"]))
            rate = success_count / len(init_states)
            er_mean = float(np.nanmean(expert_ratios))
            guided_mean = float(np.nanmean(guided_ratios))
            rho_med = float(np.nanmedian(rho_mins))
        else:
            subset_metrics = _evaluate_chain_policy_subset(
                spec=spec,
                alphabet_dict=current_alphabet_dict,
                init_states=init_states,
                dt=dt,
                episode_seconds=episode_seconds,
                controller_kwargs=controller_kwargs,
                use_fast_rollout=use_fast_rollout,
                fast_num_workers=fast_num_workers,
            )
            rate = float(subset_metrics["success_rate"])
            er_mean = float(subset_metrics["expert_ratio"])
            guided_mean = float(subset_metrics["guided_ratio"])
            rho_med = float(subset_metrics["rho_med"])

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
