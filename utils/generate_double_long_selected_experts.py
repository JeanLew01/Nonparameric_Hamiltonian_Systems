from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from utils.control_chain import build_control_alphabet_from_saved_rollouts, make_double_pendulum_spec
from utils.generate_double_long_grid_experts import _run_zero_prefix_then_mpc
from utils.mpc import DoublePendulumParams, _reset_history_if_available, build_double_pendulum_mpc
from utils.regenerate_expert_rollouts import _double_error, _save_per_rollout_arrays, _save_rollout_archive


def _expanded(path_like: str | os.PathLike[str]) -> Path:
    return Path(os.path.expanduser(str(path_like)))


def _max_wrapped_angle_span(angle_series: np.ndarray) -> float:
    wrapped = ((np.asarray(angle_series, dtype=float) + np.pi) % (2.0 * np.pi)) - np.pi
    return float(np.max(wrapped) - np.min(wrapped))


def generate_double_long_selected_experts(
    save_dir: str | os.PathLike[str] = "~/exp/Nonparameric_Hamiltonian_Systems/data",
    npz_filename: str = "all_rollouts_double_pendulum_longselected5.npz",
    alphabet_filename: str = "control_alphabet_double_pendulum_longselected5.pkl",
    prefix_seconds_candidates: tuple[float, ...] = (12.0, 10.0, 8.0, 6.0, 4.0, 2.0, 0.0),
    post_seconds: float = 24.0,
):
    save_path = _expanded(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    cfg = DoublePendulumParams(
        dt=0.02,
        n_horizon=240,
        Q=np.diag([120.0, 120.0, 6.0, 6.0]).astype(float),
        QT=np.diag([260.0, 260.0, 15.0, 15.0]).astype(float),
        R=np.diag([2e-3, 2e-3]).astype(float),
        x_ref=np.array([np.pi, np.pi, 0.0, 0.0], dtype=float),
    )
    _, mpc, simulator = build_double_pendulum_mpc(cfg)

    selected_initial_states = {
        "ls00": np.array([-np.pi / 2.0, -np.pi / 2.0, 0.0, 0.0], dtype=float),
        "ls01": np.array([0.0, -np.pi / 2.0, 0.0, 0.0], dtype=float),
        "ls02": np.array([np.pi / 2.0, 0.0, 0.0, 0.0], dtype=float),
        "ls03": np.array([0.0, np.pi / 2.0, 0.0, 0.0], dtype=float),
        "ls04": np.array([-np.pi / 2.0, 0.0, 0.0, 0.0], dtype=float),
    }

    target_state = cfg.x_ref
    success_tol = np.array([0.12, 0.12, 0.25, 0.25], dtype=float)
    rollouts: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    records = []

    for idx, (name, x0) in enumerate(selected_initial_states.items(), start=1):
        print(f"[double-longselected] {idx}/{len(selected_initial_states)} {name} x0={np.round(x0, 3).tolist()}", flush=True)
        best = None
        for prefix_seconds in prefix_seconds_candidates:
            mpc.x0 = x0.copy()
            simulator.x0 = x0.copy()
            _reset_history_if_available(mpc)
            _reset_history_if_available(simulator)
            T, X, U = _run_zero_prefix_then_mpc(
                mpc=mpc,
                simulator=simulator,
                x0=x0,
                prefix_seconds=float(prefix_seconds),
                post_seconds=float(post_seconds),
                target_state=target_state,
                success_tol=success_tol,
                hold_steps=5,
            )
            err = _double_error(X[-1], target_state)
            ok = bool(np.all(np.abs(err) <= success_tol))
            if not ok:
                continue

            span_score = _max_wrapped_angle_span(X[:, 0]) + _max_wrapped_angle_span(X[:, 1])
            score = (float(prefix_seconds), float(span_score), float(len(X)))
            candidate = {
                "name": name,
                "prefix_seconds": float(prefix_seconds),
                "span_score": float(span_score),
                "num_steps": int(len(X)),
                "terminal_error_norm": float(np.linalg.norm(err)),
                "rollout": (T, X, U),
                "score": score,
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate

        if best is not None:
            rollouts[name] = best["rollout"]
            records.append({k: v for k, v in best.items() if k not in {"rollout", "score"}})
        else:
            print(f"[double-longselected] skip {name}: no successful zero-prefix + MPC rollout found", flush=True)

    if not rollouts:
        raise RuntimeError("No successful double long-selected rollouts were generated.")

    _save_per_rollout_arrays(save_path, rollouts)
    npz_path = _save_rollout_archive(save_path, npz_filename, rollouts, target_state=target_state)

    spec = make_double_pendulum_spec(success_tol=np.array([0.4, 0.4, 0.8, 0.8], dtype=np.float32))
    build_control_alphabet_from_saved_rollouts(
        spec=spec,
        save_dir=save_path,
        names=list(rollouts.keys()),
        npz_filename=npz_filename,
        alphabet_filename=alphabet_filename,
        rho=0.999,
        H_star=0.0,
        eta=0.0,
        r_min=1e-8,
        max_lookahead=120,
        advance_stride=1,
        eps_L=1e-5,
        verbose=False,
        save_metrics=False,
    )

    return {
        "npz_path": str(npz_path),
        "alphabet_path": str(save_path / alphabet_filename),
        "names": list(rollouts.keys()),
        "records": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate long selected double-pendulum expert demos with zero-input prefixes.")
    parser.add_argument("--save-dir", default="~/exp/Nonparameric_Hamiltonian_Systems/data")
    parser.add_argument("--npz-filename", default="all_rollouts_double_pendulum_longselected5.npz")
    parser.add_argument("--alphabet-filename", default="control_alphabet_double_pendulum_longselected5.pkl")
    parser.add_argument("--post-seconds", type=float, default=24.0)
    parser.add_argument(
        "--prefix-seconds",
        nargs="+",
        type=float,
        default=[12.0, 10.0, 8.0, 6.0, 4.0, 2.0, 0.0],
    )
    args = parser.parse_args()

    result = generate_double_long_selected_experts(
        save_dir=args.save_dir,
        npz_filename=args.npz_filename,
        alphabet_filename=args.alphabet_filename,
        prefix_seconds_candidates=tuple(float(v) for v in args.prefix_seconds),
        post_seconds=float(args.post_seconds),
    )
    print(result)
