from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np

from Dynamics_double_pendulum import DoublePendulumEnv

DEFAULT_DOUBLE_PENDULUM_ROLLOUT_NAMES = (
    "x0_pi_0_0_0",
    "x1_pi_0_pi_0",
    "x2_0_0_pi_0",
    "x3_0_0_0_0",
    "x9_pi4_0_pi_0",
    "x11_5pi4_0_0_0",
)

TARGET_STATE = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
SUCCESS_TOL = np.array([0.02, 0.05, 0.02, 0.05], dtype=np.float32)

g = 9.81
m1 = 1.0
m2 = 1.0
l1 = 1.0
l2 = 1.0
lc1 = 0.5
lc2 = 0.5
I1 = 0.2
I2 = 0.2


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def check_success(obs, target=TARGET_STATE, tol=SUCCESS_TOL):
    delta = np.asarray(obs, dtype=np.float32) - np.asarray(target, dtype=np.float32)
    delta[0] = angle_normalize(delta[0])
    delta[2] = angle_normalize(delta[2])
    return bool(np.all(np.abs(delta) < tol))


def sample_initial_states(num_inits=50, seed=0):
    rng = np.random.default_rng(seed)
    th1 = rng.uniform(low=0.0, high=2 * np.pi, size=(num_inits, 1))
    th2 = rng.uniform(low=0.0, high=2 * np.pi, size=(num_inits, 1))
    th1d = np.zeros_like(th1)
    th2d = np.zeros_like(th2)
    th1 = angle_normalize(th1)
    th2 = angle_normalize(th2)
    return np.hstack([th1, th1d, th2, th2d]).astype(np.float32)


def two_link_ddq_np(x, u):
    th1, th1d, th2, th2d = x
    tau1, tau2 = u

    c2 = np.cos(th2)
    s2 = np.sin(th2)

    d11 = I1 + I2 + m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * c2)
    d12 = I2 + m2 * (lc2**2 + l1 * lc2 * c2)
    d21 = d12
    d22 = I2 + m2 * lc2**2

    det_d = d11 * d22 - d12 * d21
    if abs(det_d) < 1e-12:
        return np.array([0.0, 0.0], dtype=float)

    h = m2 * l1 * lc2 * s2
    c1 = -2.0 * h * th1d * th2d - h * th2d**2
    c2_term = h * th1d**2

    g1 = -((m1 * lc1 + m2 * l1) * g * np.sin(th1) + m2 * lc2 * g * np.sin(th1 + th2))
    g2 = -(m2 * lc2 * g * np.sin(th1 + th2))

    rhs1 = tau1 - c1 - g1
    rhs2 = tau2 - c2_term - g2

    ddq1 = (d22 * rhs1 - d12 * rhs2) / det_d
    ddq2 = (-d21 * rhs1 + d11 * rhs2) / det_d
    return np.array([ddq1, ddq2], dtype=float)


def f_continuous_np(x, u):
    ddq = two_link_ddq_np(x, u)
    return np.array([x[1], ddq[0], x[3], ddq[1]], dtype=float)


def get_dynamics_l_numerical(x, u, eps=1e-5):
    x = np.asarray(x, dtype=float).copy()
    u = np.asarray(u, dtype=float).copy().reshape(-1)
    if u.size < 2:
        u = np.pad(u, (0, 2 - u.size), constant_values=0.0)
    u = u[:2]

    n = x.size
    jac = np.zeros((n, n), dtype=float)
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        f_plus = f_continuous_np(x_plus, u)
        f_minus = f_continuous_np(x_minus, u)
        jac[:, i] = (f_plus - f_minus) / (2.0 * eps)
    return float(np.linalg.norm(jac, 2))


def _expanded_path(path_like):
    return Path(os.path.expanduser(str(path_like)))


def _ensure_control_shape(u, n_ctrl_needed):
    u = np.asarray(u, dtype=float)
    if u.ndim == 1:
        u = u.reshape(-1, 1)
    if u.shape[1] < 2:
        pad = np.zeros((u.shape[0], 2 - u.shape[1]), dtype=float)
        u = np.hstack([u, pad])
    u = u[:, :2]
    if u.shape[0] < n_ctrl_needed:
        pad = np.zeros((n_ctrl_needed - u.shape[0], 2), dtype=float)
        u = np.vstack([u, pad])
    else:
        u = u[:n_ctrl_needed]
    return u


def load_saved_rollouts(save_dir, names, npz_filename="all_rollouts_double_pendulum.npz"):
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


def generate_control_alphabet_from_rollout(
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
    torque_clip=None,
):
    T = np.asarray(T, dtype=float).squeeze()
    X = np.asarray(X, dtype=float)
    H_array = np.asarray(H_array, dtype=float).squeeze()
    LH_array = np.asarray(LH_array, dtype=float).squeeze()

    if T.ndim != 1 or T.size < 2:
        raise ValueError("T must be 1D with length >= 2.")
    if X.ndim != 2 or X.shape[1] != 4:
        raise ValueError(f"X must be (N, 4), got {X.shape}")

    n_states = X.shape[0]
    n_eff = min(n_states, H_array.size, LH_array.size)
    if n_eff < 2:
        return []

    X = X[:n_eff]
    H_array = H_array[:n_eff]
    LH_array = LH_array[:n_eff]
    T = T[:n_eff] if T.size >= n_eff else T

    dt = float(np.mean(np.diff(T[:2]))) if T.size >= 2 else 0.02
    n_ctrl_needed = n_eff - 1
    U = _ensure_control_shape(U, n_ctrl_needed)

    if torque_clip is not None:
        clip_val = float(torque_clip)
        U = np.clip(U, -clip_val, clip_val)

    l_dyns = np.zeros(n_ctrl_needed, dtype=float)
    for idx in range(n_ctrl_needed):
        l_dyns[idx] = get_dynamics_l_numerical(X[idx], U[idx], eps=eps_L)

    tubes = []
    i = 0
    while i < n_ctrl_needed:
        v_curr = abs(float(H_array[i]) - float(H_star))
        if v_curr <= 1e-8:
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
    save_dir,
    names=DEFAULT_DOUBLE_PENDULUM_ROLLOUT_NAMES,
    npz_filename="all_rollouts_double_pendulum.npz",
    alphabet_filename="control_alphabet.pkl",
    rho=0.99,
    H_star=0.0,
    eta=0.0,
    r_min=1e-6,
    max_lookahead=200,
    eps_L=1e-5,
    torque_clip=12.0,
    verbose=True,
):
    save_path = _expanded_path(save_dir)
    rollouts = load_saved_rollouts(save_path, names, npz_filename=npz_filename)

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
                print(f"[WARN] Missing energy files for {name}:")
                print(f"       {h_path}")
                print(f"       {lh_path}")
            continue

        H_array = np.load(h_path)
        LH_array = np.load(lh_path)
        if verbose:
            print(
                "Shapes: "
                f"T={np.shape(T)}, X={np.shape(X)}, U={np.shape(U)}, "
                f"H={np.shape(H_array)}, LH={np.shape(LH_array)}"
            )

        tubes = generate_control_alphabet_from_rollout(
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
            torque_clip=torque_clip,
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
                print(
                    "   tube[0] keys:",
                    all_alphabets[name][0].keys(),
                    "u_seq shape:",
                    all_alphabets[name][0]["u_seq"].shape,
                )

    alphabet_path = save_path / alphabet_filename
    with open(alphabet_path, "wb") as handle:
        pickle.dump(all_alphabets, handle)
    if verbose:
        print(f"\nAll alphabets saved to: {alphabet_path}")

    return all_alphabets, summary


def load_control_alphabet(save_dir, alphabet_filename="control_alphabet.pkl"):
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
        eps=1e-9,
        execute_full_sequence=False,
        angle_indices=(0, 2),
        distance_weights=(1.0, 0.2, 1.0, 0.2),
        radius_scale=10.0,
        enter_threshold=1.0,
        abort_threshold=1.5,
        outside_mode="nearest_control",
    ):
        self.eps = float(eps)
        self.execute_full_sequence = bool(execute_full_sequence)
        self.angle_indices = tuple(angle_indices)
        self.distance_weights = np.asarray(distance_weights, dtype=np.float32).reshape(4,)
        self.radius_scale = float(radius_scale)
        self.enter_threshold = float(enter_threshold)
        self.abort_threshold = float(abort_threshold)
        self.outside_mode = str(outside_mode)
        self.default_u = (
            np.zeros(2, dtype=np.float32)
            if default_u is None
            else np.asarray(default_u, dtype=np.float32).reshape(2,)
        )

        centers = []
        radii = []
        u_controls = []
        u_seqs = []
        meta = []
        missing_controls = 0

        for name, tube_list in all_alphabets_dict.items():
            for tube in tube_list:
                x_center = np.asarray(tube["x_center"], dtype=np.float32).reshape(4,)
                radius = float(tube["radius"])
                u_seq = np.asarray(tube.get("u_seq"), dtype=np.float32)
                u_control_raw = tube.get("u_control")
                if u_control_raw is None:
                    u_control_raw = u_seq[0]
                u_control = np.asarray(u_control_raw, dtype=np.float32).reshape(2,)

                if u_seq.ndim != 2 or u_seq.shape[1] != 2 or u_seq.shape[0] < 1:
                    missing_controls += 1
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

        if missing_controls > 0 and not centers:
            raise RuntimeError("No valid tubes loaded. Rebuild the control alphabet from saved rollouts.")
        if not centers:
            raise RuntimeError("No valid tubes loaded (total=0). Check your control_alphabet.pkl content.")

        self.centers = np.stack(centers, axis=0)
        self.raw_radii = np.asarray(radii, dtype=np.float32)
        self.radii = np.maximum(self.raw_radii * self.radius_scale, self.eps)
        self.u_controls = np.stack(u_controls, axis=0)
        self.u_seqs = u_seqs
        self.meta = meta
        self.active_idx = None
        self.active_k = 0

        print(f"[TubeController] loaded {self.centers.shape[0]} tubes.")
        print(
            "[TubeController] effective radius stats: min/med/max = "
            f"{float(np.min(self.radii)):.3e}/"
            f"{float(np.median(self.radii)):.3e}/"
            f"{float(np.max(self.radii)):.3e}"
        )
        print(
            "[TubeController] config: "
            f"execute_full_sequence={self.execute_full_sequence}, "
            f"radius_scale={self.radius_scale:.2f}, "
            f"enter_threshold={self.enter_threshold:.2f}, "
            f"outside_mode={self.outside_mode}"
        )

    def reset_episode(self):
        self.active_idx = None
        self.active_k = 0

    def _distance(self, x, centers):
        diff = centers - x[None, :]
        for idx in self.angle_indices:
            diff[:, idx] = angle_normalize(diff[:, idx])
        weighted = diff * self.distance_weights[None, :]
        return np.linalg.norm(weighted, axis=1)

    def select_tube(self, x_current):
        x = np.asarray(x_current, dtype=np.float32).reshape(4,)
        dists = self._distance(x, self.centers)
        rhos = dists / (self.radii + self.eps)
        best_idx = int(np.argmin(rhos))
        best_rho = float(rhos[best_idx])
        return best_idx, best_rho

    def get_action(self, x_current):
        if self.execute_full_sequence and self.active_idx is not None:
            active_center = self.centers[self.active_idx]
            active_rho = float(
                self._distance(np.asarray(x_current, dtype=np.float32).reshape(4,), active_center[None, :])[0]
                / (self.radii[self.active_idx] + self.eps)
            )
            if active_rho <= self.abort_threshold:
                u_seq = self.u_seqs[self.active_idx]
                if self.active_k < u_seq.shape[0]:
                    u = u_seq[self.active_k].copy()
                    self.active_k += 1
                    return u, "Expert", active_rho
            self.active_idx = None
            self.active_k = 0

        best_idx, best_rho = self.select_tube(x_current)
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
        return self.default_u.copy(), "Default", best_rho

