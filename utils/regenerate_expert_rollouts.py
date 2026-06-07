from __future__ import annotations

import os
import argparse
from pathlib import Path

import numpy as np

from utils.mpc import (
    DoublePendulumParams,
    SinglePendulumParams,
    SpringMassParams,
    _as_1d_array,
    _as_column_array,
    _reset_history_if_available,
    _simulator_t_step,
    build_double_pendulum_mpc,
    build_single_pendulum_mpc,
    build_spring_mass_mpc,
)


def _expanded(path_like: str | os.PathLike[str]) -> Path:
    return Path(os.path.expanduser(str(path_like)))


def _save_rollout_archive(save_dir: Path, npz_filename: str, rollouts: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], target_state=None) -> Path:
    payload = {
        **{f"T_{k}": v[0] for k, v in rollouts.items()},
        **{f"X_{k}": v[1] for k, v in rollouts.items()},
        **{f"U_{k}": v[2] for k, v in rollouts.items()},
    }
    if target_state is not None:
        payload["target_state"] = np.asarray(target_state, dtype=float)

    npz_path = save_dir / npz_filename
    np.savez(npz_path, **payload)
    return npz_path


def _save_per_rollout_arrays(save_dir: Path, rollouts: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
    for name, (T, X, U) in rollouts.items():
        np.save(save_dir / f"T_{name}.npy", T)
        np.save(save_dir / f"X_{name}.npy", X)
        np.save(save_dir / f"U_{name}.npy", U)


def _run_named_rollouts(mpc, simulator, initial_states, sim_time: float, label: str):
    raise RuntimeError("Use _run_named_rollouts_until_target with explicit target/tolerance.")


def _angle_wrap_np(x: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi


def _single_error(x: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.array([
        _angle_wrap_np(x[0] - target[0]),
        x[1] - target[1],
    ], dtype=float)


def _spring_error(x: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float) - np.asarray(target, dtype=float)


def _double_error(x: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.array([
        _angle_wrap_np(x[0] - target[0]),
        _angle_wrap_np(x[1] - target[1]),
        x[2] - target[2],
        x[3] - target[3],
    ], dtype=float)


def _is_success_error(error: np.ndarray, success_tol: np.ndarray) -> bool:
    error = np.asarray(error, dtype=float).reshape(-1)
    success_tol = np.asarray(success_tol, dtype=float).reshape(-1)
    if success_tol.size == 1 or np.allclose(success_tol, success_tol[0]):
        return bool(np.linalg.norm(error, ord=2) <= float(success_tol[0]))
    return bool(np.all(np.abs(error) <= success_tol))


def _rollout_mpc_until_target(
    mpc,
    simulator,
    x0,
    sim_time: float,
    target_state: np.ndarray,
    success_tol: np.ndarray,
    error_fn,
    hold_steps: int = 5,
):
    dt = _simulator_t_step(simulator)
    num_steps = int(sim_time / dt)

    x = _as_1d_array(x0)
    mpc.x0 = x.copy()
    simulator.x0 = x.copy()
    _reset_history_if_available(mpc)
    _reset_history_if_available(simulator)
    mpc.set_initial_guess()

    rollouts: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    X = [x.copy()]
    U = []
    T = [0.0]
    success_count = 0

    for _ in range(num_steps):
        u = _as_1d_array(mpc.make_step(x))
        x = _as_1d_array(simulator.make_step(_as_column_array(u)))
        U.append(u.copy())
        X.append(x.copy())
        T.append(T[-1] + dt)

        error = error_fn(x, target_state)
        if _is_success_error(error, success_tol):
            success_count += 1
            if success_count >= int(hold_steps):
                break
        else:
            success_count = 0

    return (
        np.asarray(T, dtype=float),
        np.asarray(X, dtype=float),
        np.asarray(U, dtype=float),
    )


def _run_named_rollouts_until_target(
    mpc,
    simulator,
    initial_states,
    sim_time: float,
    label: str,
    target_state: np.ndarray,
    success_tol: np.ndarray,
    error_fn,
    hold_steps: int = 5,
):
    rollouts: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    total = len(initial_states)
    for idx, (name, x0) in enumerate(initial_states.items(), start=1):
        print(f"[{label}] rollout {idx}/{total}: {name} x0={np.asarray(x0).tolist()}", flush=True)
        rollouts[name] = _rollout_mpc_until_target(
            mpc,
            simulator,
            x0,
            sim_time=sim_time,
            target_state=np.asarray(target_state, dtype=float),
            success_tol=np.asarray(success_tol, dtype=float),
            error_fn=error_fn,
            hold_steps=hold_steps,
        )
    return rollouts


def _clear_stale_derived_artifacts(save_dir: Path) -> list[str]:
    removed: list[str] = []
    prefixes = ("H_total_", "LH_", "Ldyn_")
    exact_names = {
        "Ldyn_all.npy",
        "control_alphabet.pkl",
        "control_alphabet_double_pendulum.pkl",
        "control_alphabet_double_pendulum_overlap.pkl",
        "control_alphabet_double_pendulum_overlap_hi.pkl",
        "control_alphabet_double_pendulum_sparse.pkl",
        "control_alphabet_single_pendulum.pkl",
        "control_alphabet_single_pendulum_dense.pkl",
        "control_alphabet_single_pendulum_dense_overlap.pkl",
        "control_alphabet_single_pendulum_dense_overlap_hi.pkl",
        "control_alphabet_spring_mass.pkl",
        "control_alphabet_single_pendulum_components_paper.pkl",
        "control_alphabet_spring_mass_paper.pkl",
    }
    suffixes = ("_chain_vs_bc.png", "_chain_vs_bc_bar.png")

    for path in save_dir.iterdir():
        name = path.name
        if path.is_dir():
            continue
        should_remove = (
            name in exact_names
            or name.startswith(prefixes)
            or name.startswith("incremental_")
            or name.endswith(suffixes)
        )
        if should_remove:
            path.unlink()
            removed.append(name)
    return sorted(removed)


def _generate_single_sparse_rollouts(save_dir: Path) -> Path:
    cfg = SinglePendulumParams(
        dt=0.02,
        n_horizon=120,
        x_ref=np.array([np.pi, 0.0], dtype=float),
    )
    _, mpc, simulator = build_single_pendulum_mpc(cfg)
    sim_time = 10.0

    initial_states = {
        "x0_0_0": np.array([0.0, 0.0], dtype=float),
        "x1_pi4_0": np.array([np.pi / 4.0, 0.0], dtype=float),
        "x2_pi2_0": np.array([np.pi / 2.0, 0.0], dtype=float),
        "x3_3pi4_0": np.array([3.0 * np.pi / 4.0, 0.0], dtype=float),
        "x4_pi_0": np.array([np.pi, 0.0], dtype=float),
        "x5_5pi4_0": np.array([5.0 * np.pi / 4.0, 0.0], dtype=float),
        "x6_3pi2_0": np.array([3.0 * np.pi / 2.0, 0.0], dtype=float),
        "x7_7pi4_0": np.array([7.0 * np.pi / 4.0, 0.0], dtype=float),
    }

    rollouts = _run_named_rollouts_until_target(
        mpc,
        simulator,
        initial_states,
        sim_time=sim_time,
        label="single-sparse",
        target_state=cfg.x_ref,
        success_tol=np.array([0.1, 0.1], dtype=float),
        error_fn=_single_error,
    )

    _save_per_rollout_arrays(save_dir, rollouts)
    return _save_rollout_archive(save_dir, "all_rollouts_single_pendulum.npz", rollouts, target_state=cfg.x_ref)


def _generate_single_dense_rollouts(save_dir: Path) -> Path:
    cfg = SinglePendulumParams(
        dt=0.02,
        n_horizon=120,
        x_ref=np.array([np.pi, 0.0], dtype=float),
    )
    _, mpc, simulator = build_single_pendulum_mpc(cfg)
    sim_time = 10.0

    thetas = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False, dtype=float)
    initial_states = {
        f"th{idx:02d}": np.array([theta0, 0.0], dtype=float)
        for idx, theta0 in enumerate(thetas)
    }
    rollouts = _run_named_rollouts_until_target(
        mpc,
        simulator,
        initial_states,
        sim_time=sim_time,
        label="single-dense",
        target_state=cfg.x_ref,
        success_tol=np.array([0.1, 0.1], dtype=float),
        error_fn=_single_error,
    )

    _save_per_rollout_arrays(save_dir, rollouts)
    return _save_rollout_archive(save_dir, "all_rollouts_single_pendulum_dense.npz", rollouts, target_state=cfg.x_ref)


def _generate_spring_rollouts(save_dir: Path) -> Path:
    cfg = SpringMassParams(
        dt=0.02,
        n_horizon=120,
        x_ref=np.array([0.0, 0.0], dtype=float),
    )
    _, mpc, simulator = build_spring_mass_mpc(cfg)
    sim_time = 10.0

    initial_states = {
        "x0_m2_0": np.array([-2.0, 0.0], dtype=float),
        "x1_m1_0": np.array([-1.0, 0.0], dtype=float),
        "x2_0_0": np.array([0.0, 0.0], dtype=float),
        "x3_1_0": np.array([1.0, 0.0], dtype=float),
        "x4_2_0": np.array([2.0, 0.0], dtype=float),
        "x5_0_2": np.array([0.0, 2.0], dtype=float),
        "x6_0_m2": np.array([0.0, -2.0], dtype=float),
        "x7_1p5_m1": np.array([1.5, -1.0], dtype=float),
    }

    rollouts = _run_named_rollouts_until_target(
        mpc,
        simulator,
        initial_states,
        sim_time=sim_time,
        label="spring",
        target_state=cfg.x_ref,
        success_tol=np.array([0.1, 0.1], dtype=float),
        error_fn=_spring_error,
    )

    _save_per_rollout_arrays(save_dir, rollouts)
    return _save_rollout_archive(save_dir, "all_rollouts_spring_mass.npz", rollouts, target_state=cfg.x_ref)


def _generate_double_rollouts(save_dir: Path) -> Path:
    cfg = DoublePendulumParams(
        dt=0.02,
        n_horizon=180,
        Q=np.diag([80.0, 80.0, 3.0, 3.0]).astype(float),
        QT=np.diag([180.0, 180.0, 8.0, 8.0]).astype(float),
        R=np.diag([5e-3, 5e-3]).astype(float),
        x_ref=np.array([np.pi, np.pi, 0.0, 0.0], dtype=float),
    )
    _, mpc, simulator = build_double_pendulum_mpc(cfg)
    sim_time = 14.0

    theta_grid = np.linspace(-np.pi, np.pi, 6, endpoint=False, dtype=float)
    initial_states = {}
    for i, th1 in enumerate(theta_grid):
        for j, th2 in enumerate(theta_grid):
            name = f"g{i:02d}_{j:02d}"
            initial_states[name] = np.array([th1, th2, 0.0, 0.0], dtype=float)

    rollouts = _run_named_rollouts_until_target(
        mpc,
        simulator,
        initial_states,
        sim_time=sim_time,
        label="double",
        target_state=cfg.x_ref,
        success_tol=np.array([0.12, 0.12, 0.25, 0.25], dtype=float),
        error_fn=_double_error,
    )

    _save_per_rollout_arrays(save_dir, rollouts)
    return _save_rollout_archive(save_dir, "all_rollouts_double_pendulum.npz", rollouts, target_state=cfg.x_ref)


def regenerate_expert_rollouts(
    save_dir: str | os.PathLike[str] = "~/exp/Nonparameric_Hamiltonian_Systems/data",
    clear_derived: bool = True,
    systems: tuple[str, ...] = ("single_sparse", "single_dense", "spring", "double"),
) -> dict[str, object]:
    save_path = _expanded(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    removed = _clear_stale_derived_artifacts(save_path) if clear_derived else []

    generated_files: list[str] = []
    if "single_sparse" in systems:
        generated_files.append(str(_generate_single_sparse_rollouts(save_path)))
    if "single_dense" in systems:
        generated_files.append(str(_generate_single_dense_rollouts(save_path)))
    if "spring" in systems:
        generated_files.append(str(_generate_spring_rollouts(save_path)))
    if "double" in systems:
        generated_files.append(str(_generate_double_rollouts(save_path)))

    return {
        "save_dir": str(save_path),
        "removed_artifacts": removed,
        "generated_files": generated_files,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate expert rollout archives from the aligned Hamiltonian dynamics.")
    parser.add_argument(
        "--save-dir",
        default="~/exp/Nonparameric_Hamiltonian_Systems/data",
        help="Directory where rollout archives are written.",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["single_sparse", "single_dense", "spring", "double"],
        choices=["single_sparse", "single_dense", "spring", "double"],
        help="Subset of rollout archives to regenerate.",
    )
    parser.add_argument(
        "--keep-derived",
        action="store_true",
        help="Keep derived metrics/alphabets/results instead of deleting stale caches.",
    )
    args = parser.parse_args()

    summary = regenerate_expert_rollouts(
        save_dir=args.save_dir,
        clear_derived=not args.keep_derived,
        systems=tuple(args.systems),
    )
    print("Regenerated expert rollouts in:", summary["save_dir"])
    print("Generated files:")
    for path in summary["generated_files"]:
        print(" -", path)
    if summary["removed_artifacts"]:
        print("Removed stale derived artifacts:")
        for name in summary["removed_artifacts"]:
            print(" -", name)
