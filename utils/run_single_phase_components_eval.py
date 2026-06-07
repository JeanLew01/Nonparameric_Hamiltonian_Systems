from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dynamics.Dynamics_single_pendulum import I as SINGLE_I
from dynamics.Dynamics_single_pendulum import g as SINGLE_G
from dynamics.Dynamics_single_pendulum import l as SINGLE_L
from dynamics.Dynamics_single_pendulum import m as SINGLE_M
from utils.behavior_cloning_eval import evaluate_incremental_behavior_cloning
from utils.control_chain import (
    build_control_alphabet_from_saved_rollouts,
    evaluate_incremental_chain_policy,
    make_single_pendulum_spec,
)
from utils.final_uniform_eval import _save_chain_vs_bc_figure


def _angle_wrap(x):
    return (float(x) + np.pi) % (2.0 * np.pi) - np.pi


def _single_energy(x):
    theta, p = np.asarray(x, dtype=float).reshape(2,)
    return 0.5 * (p**2) / SINGLE_I + SINGLE_M * SINGLE_G * SINGLE_L * (1.0 - np.cos(theta))


def _single_f_continuous(x, u):
    theta, p = np.asarray(x, dtype=float).reshape(2,)
    tau = float(np.asarray(u, dtype=float).reshape(-1)[0])
    theta_dot = p / SINGLE_I
    p_dot = tau - SINGLE_M * SINGLE_G * SINGLE_L * np.sin(theta)
    return np.array([theta_dot, p_dot], dtype=float)


def _rk4_step(x, u, dt):
    k1 = _single_f_continuous(x, u)
    k2 = _single_f_continuous(x + 0.5 * dt * k1, u)
    k3 = _single_f_continuous(x + 0.5 * dt * k2, u)
    k4 = _single_f_continuous(x + dt * k3, u)
    return np.asarray(x, dtype=float) + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _expert_single_control(x, target_state, u_max, mode: str, step_idx: int):
    theta, p = np.asarray(x, dtype=float).reshape(2,)
    theta_dot = p / SINGLE_I
    target_theta = float(np.asarray(target_state, dtype=float).reshape(2,)[0])
    target_energy = SINGLE_M * SINGLE_G * SINGLE_L * (1.0 - np.cos(target_theta))
    err_theta = _angle_wrap(theta - target_theta)

    if mode == "stabilize":
        tau = -20.0 * err_theta - 5.0 * theta_dot
        return float(np.clip(tau, -u_max, u_max))

    if mode == "capture":
        if abs(err_theta) < 0.6 and abs(theta_dot) < 0.8:
            tau = -20.0 * err_theta - 5.0 * theta_dot
        else:
            tau = 0.2 * (target_energy - _single_energy(x)) * theta_dot * np.cos(theta)
        return float(np.clip(tau, -u_max, u_max))

    # Libration component: kick from the bottom rest state, then use
    # a standard energy-shaping swing-up law until the upright PD region.
    if mode == "lib":
        if abs(err_theta) < 0.6 and abs(theta_dot) < 0.5:
            tau = -20.0 * err_theta - 5.0 * theta_dot
        elif step_idx < 60:
            tau = u_max
        else:
            signature = theta_dot * np.cos(theta)
            if abs(signature) < 1e-4:
                signature = 1.0
            tau = 0.1 * (target_energy - _single_energy(x)) * np.sign(signature)
        return float(np.clip(tau, -u_max, u_max))

    # Rotational components: use a stronger energy-shaping law so that
    # clockwise/counterclockwise demonstrations actually converge to the
    # upright target, instead of only slowly dissipating momentum.
    if abs(err_theta) < 0.2 and abs(theta_dot) < 0.5:
        tau = -20.0 * err_theta - 5.0 * theta_dot
    else:
        signature = theta_dot * np.cos(theta)
        if abs(signature) < 1e-4:
            signature = np.sign(theta_dot) if abs(theta_dot) > 1e-4 else 1.0
        tau = 0.5 * (target_energy - _single_energy(x)) * np.sign(signature)

    return float(np.clip(tau, -u_max, u_max))


def _rollout_expert_trajectory(
    x0,
    target_state,
    mode,
    dt=0.02,
    episode_seconds=30.0,
    success_tol=(0.12, 0.25),
    hold_steps=5,
    u_max=20.0,
):
    x = np.asarray(x0, dtype=float).reshape(2,)
    target_state = np.asarray(target_state, dtype=float).reshape(2,)
    success_tol = np.asarray(success_tol, dtype=float).reshape(2,)

    X = [x.copy()]
    U = []
    T = [0.0]
    n_steps = int(np.round(float(episode_seconds) / float(dt)))
    success_count = 0

    for step_idx in range(n_steps):
        u = _expert_single_control(
            x,
            target_state=target_state,
            u_max=float(u_max),
            mode=str(mode),
            step_idx=int(step_idx),
        )
        x = _rk4_step(x, np.array([u], dtype=float), dt)
        X.append(x.copy())
        U.append([u])
        T.append(T[-1] + float(dt))

        err = np.array([_angle_wrap(x[0] - target_state[0]), x[1] - target_state[1]], dtype=float)
        if np.all(np.abs(err) <= success_tol):
            success_count += 1
            if success_count >= int(hold_steps):
                break
        else:
            success_count = 0

    return np.asarray(T, dtype=float), np.asarray(X, dtype=float), np.asarray(U, dtype=float)


def _rollout_rotation_component_trajectory(
    x0,
    target_state,
    direction,
    coast_steps=0,
    dt=0.02,
    episode_seconds=40.0,
    success_tol=(0.12, 0.25),
    hold_steps=5,
    u_max=20.0,
    brake_energy_margin=8.0,
    brake_torque=None,
    mpc_bundle=None,
):
    x = np.asarray(x0, dtype=float).reshape(2,)
    target_state = np.asarray(target_state, dtype=float).reshape(2,)
    success_tol = np.asarray(success_tol, dtype=float).reshape(2,)

    direction = str(direction)
    if direction not in {"ccw", "cw"}:
        raise ValueError(f"direction must be 'ccw' or 'cw', got {direction!r}")

    X = [x.copy()]
    U = []
    T = [0.0]
    n_steps = int(np.round(float(episode_seconds) / float(dt)))
    success_count = 0
    coast_steps = max(int(coast_steps), 0)
    target_energy = _single_energy(target_state)
    brake_energy_threshold = float(target_energy + brake_energy_margin)
    brake_torque = float(u_max if brake_torque is None else brake_torque)
    desired_sign = +1.0 if direction == "ccw" else -1.0

    if mpc_bundle is not None:
        coast_steps = min(coast_steps, n_steps)
        for step_idx in range(coast_steps):
            u = 0.0
            x = _rk4_step(x, np.array([u], dtype=float), dt)
            X.append(x.copy())
            U.append([u])
            T.append(T[-1] + float(dt))

        remaining_seconds = max(0.0, float(episode_seconds) - float(coast_steps) * float(dt))
        if remaining_seconds > 0.0:
            from utils.mpc import rollout_mpc

            mpc, simulator = mpc_bundle
            T_tail, X_tail, U_tail = rollout_mpc(mpc, simulator, x, sim_time=remaining_seconds, save_U=False)
            for u_tail, x_tail in zip(U_tail, X_tail[1:]):
                x = np.asarray(x_tail, dtype=float).reshape(2,)
                X.append(x.copy())
                U.append(np.asarray(u_tail, dtype=float).reshape(1,))
                T.append(T[-1] + float(dt))

        success_mask = []
        for state in X[1:]:
            err = np.array([_angle_wrap(state[0] - target_state[0]), state[1] - target_state[1]], dtype=float)
            success_mask.append(bool(np.all(np.abs(err) <= success_tol)))
        success_mask = np.asarray(success_mask, dtype=bool)
        if success_mask.size >= int(hold_steps):
            window = np.convolve(success_mask.astype(np.int32), np.ones(int(hold_steps), dtype=np.int32), mode="valid")
            hit_idxs = np.where(window >= int(hold_steps))[0]
            if hit_idxs.size > 0:
                end_idx = int(hit_idxs[0] + int(hold_steps))
                return (
                    np.asarray(T[: end_idx + 1], dtype=float),
                    np.asarray(X[: end_idx + 1], dtype=float),
                    np.asarray(U[:end_idx], dtype=float),
                )

        return np.asarray(T, dtype=float), np.asarray(X, dtype=float), np.asarray(U, dtype=float)

    for step_idx in range(n_steps):
        if step_idx < coast_steps:
            # Let the system traverse the high-energy rotational component
            # under zero input before steering to the upright target.
            u = 0.0
        else:
            theta_dot = float(x[1] / SINGLE_I)
            err_theta = _angle_wrap(float(x[0]) - float(target_state[0]))
            energy_now = _single_energy(x)

            if abs(err_theta) < 0.35 and abs(theta_dot) < 1.2:
                u = -20.0 * err_theta - 5.0 * theta_dot
            elif energy_now > brake_energy_threshold:
                # Dissipate energy while preserving the intended rotational
                # direction until the trajectory energy is low enough to hand
                # over to the target-reaching swing-up law.
                vel_sign = np.sign(theta_dot) if abs(theta_dot) > 1e-6 else desired_sign
                u = -brake_torque * vel_sign
            else:
                u = _expert_single_control(
                    x,
                    target_state=target_state,
                    u_max=float(u_max),
                    mode="capture",
                    step_idx=int(step_idx - coast_steps),
                )
            u = float(np.clip(u, -u_max, u_max))

        x = _rk4_step(x, np.array([u], dtype=float), dt)
        X.append(x.copy())
        U.append([u])
        T.append(T[-1] + float(dt))

        err = np.array([_angle_wrap(x[0] - target_state[0]), x[1] - target_state[1]], dtype=float)
        if np.all(np.abs(err) <= success_tol):
            success_count += 1
            if success_count >= int(hold_steps):
                break
        else:
            success_count = 0

    return np.asarray(T, dtype=float), np.asarray(X, dtype=float), np.asarray(U, dtype=float)


def _save_rollout_archive(npz_path: Path, rollouts: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], target_state):
    payload = {"target_state": np.asarray(target_state, dtype=float)}
    for name, (T, X, U) in rollouts.items():
        payload[f"T_{name}"] = T
        payload[f"X_{name}"] = X
        payload[f"U_{name}"] = U
    np.savez(npz_path, **payload)


def make_uniform_energy_bounded_single_states(num_inits, H_bar, seed=123):
    rng = np.random.default_rng(int(seed))
    num_inits = int(num_inits)
    H_bar = float(H_bar)
    p_max = np.sqrt(2.0 * SINGLE_I * H_bar)

    states = []
    while len(states) < num_inits:
        theta = rng.uniform(-np.pi, np.pi)
        p = rng.uniform(-p_max, p_max)
        if _single_energy((theta, p)) <= H_bar + 1e-9:
            states.append([theta, p])
    return np.asarray(states, dtype=np.float32)


def _single_component_label(x):
    x = np.asarray(x, dtype=float).reshape(2,)
    sep_energy = 2.0 * SINGLE_M * SINGLE_G * SINGLE_L
    energy = _single_energy(x)
    if energy < sep_energy - 1e-8:
        return "lib"
    if x[1] > 0.0:
        return "ccw"
    if x[1] < 0.0:
        return "cw"
    return "lib"


def _covered_single_components(names):
    covered = set()
    for name in names:
        if name.startswith("lib"):
            covered.add("lib")
        elif "ccw" in name:
            covered.add("ccw")
        elif "cw" in name:
            covered.add("cw")
    return covered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="data")
    parser.add_argument("--num-inits", type=int, default=500)
    parser.add_argument("--hbar", type=float, default=160.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--rotation-coast-steps", type=int, default=100)
    parser.add_argument("--expert-episode-seconds", type=float, default=80.0)
    parser.add_argument("--eval-episode-seconds", type=float, default=150.0)
    parser.add_argument("--radius-scale", type=float, default=1.0)
    parser.add_argument("--enter-threshold", type=float, default=1.0)
    parser.add_argument("--abort-threshold", type=float, default=np.inf)
    parser.add_argument("--momentum-weight", type=float, default=0.2)
    parser.add_argument("--fast-num-workers", type=int, default=1)
    parser.add_argument("--bc-epochs", type=int, default=40)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    target_state = np.array([np.pi, 0.0], dtype=float)
    H_bar = float(args.hbar)
    p_bar = np.sqrt(2.0 * SINGLE_I * H_bar)
    rotation_anchor = 0.5 * np.pi

    def p_from_energy(theta, energy, sign):
        potential = SINGLE_M * SINGLE_G * SINGLE_L * (1.0 - np.cos(float(theta)))
        return float(sign) * np.sqrt(max(0.0, 2.0 * SINGLE_I * (float(energy) - potential)))

    # Incremental order follows the phase-space component story:
    # bottom-rest libration, high-energy clockwise rotation, high-energy
    # counterclockwise rotation, then flexible side libration rollouts.
    rollout_names = ["lib", "cw_long", "ccw_long", "lib_right", "lib_left"]
    initial_states = {
        "lib": np.array([0.0, 0.0], dtype=float),
        "cw_long": np.array(
            [rotation_anchor, p_from_energy(rotation_anchor, H_bar, -1.0)],
            dtype=float,
        ),
        "ccw_long": np.array(
            [-rotation_anchor, p_from_energy(-rotation_anchor, H_bar, +1.0)],
            dtype=float,
        ),
        "lib_left": np.array([-3.0, 0.0], dtype=float),
        "lib_right": np.array([3.0, 0.0], dtype=float),
    }

    print(f"[single-components] H_bar={H_bar:.3f}, p_bar={p_bar:.3f}", flush=True)

    rollouts = {}
    for name in rollout_names:
        print(f"[single-components] generating rollout: {name} x0={initial_states[name].tolist()}", flush=True)
        if name in {"lib_left", "lib_right"}:
            rollouts[name] = _rollout_expert_trajectory(
                initial_states[name],
                target_state=target_state,
                mode="stabilize",
                dt=0.02,
                episode_seconds=float(args.expert_episode_seconds),
                success_tol=(0.1, 0.1),
                hold_steps=5,
                u_max=20.0,
            )
        elif name.startswith("lib"):
            rollouts[name] = _rollout_expert_trajectory(
                initial_states[name],
                target_state=target_state,
                mode="stabilize",
                dt=0.02,
                episode_seconds=float(args.expert_episode_seconds),
                success_tol=(0.1, 0.1),
                hold_steps=5,
                u_max=20.0,
            )
        else:
            direction = "ccw" if "ccw" in name else "cw"
            coast_steps = int(args.rotation_coast_steps)
            if "extra" in name:
                coast_steps = int(1.5 * int(args.rotation_coast_steps))
            rollouts[name] = _rollout_rotation_component_trajectory(
                initial_states[name],
                target_state=target_state,
                direction=direction,
                coast_steps=coast_steps,
                dt=0.02,
                episode_seconds=float(args.expert_episode_seconds),
                success_tol=(0.1, 0.1),
                hold_steps=5,
                u_max=20.0,
            )
        T, X, _ = rollouts[name]
        print(
            f"  end_time={float(T[-1]):.3f}s, x_end={X[-1].tolist()}, "
            f"H0={_single_energy(X[0]):.3f}, Hf={_single_energy(X[-1]):.3f}",
            flush=True,
        )

    rollout_npz = save_dir / "all_rollouts_single_pendulum_components.npz"
    _save_rollout_archive(rollout_npz, rollouts, target_state=target_state)
    print(f"[single-components] saved rollouts -> {rollout_npz}", flush=True)

    spec = make_single_pendulum_spec(
        success_tol=np.array([0.1, 0.1], dtype=np.float32),
        success_norm_eps=0.1,
    )
    build_control_alphabet_from_saved_rollouts(
        spec=spec,
        save_dir=save_dir,
        names=rollout_names,
        npz_filename=rollout_npz.name,
        alphabet_filename="control_alphabet_single_pendulum_components_paper.pkl",
        rho=1.0,
        H_star=0.0,
        eta=0.0,
        r_min=1e-10,
        max_lookahead=300,
        advance_stride=None,
        eps_L=1e-6,
        v0=1e-6,
        radius_mode="paper",
        verbose=False,
        save_metrics=False,
    )
    print("[single-components] rebuilt alphabet", flush=True)

    eval_init_states = make_uniform_energy_bounded_single_states(
        num_inits=args.num_inits,
        H_bar=H_bar,
        seed=args.seed,
    )
    print(f"[single-components] sampled {len(eval_init_states)} initial states with H <= {H_bar:.3f}", flush=True)

    chain_cfg = {
        "execute_full_sequence": True,
        "outside_mode": "default",
        "radius_scale": float(args.radius_scale),
        "radius_floor": 0.055,
        "enter_threshold": float(args.enter_threshold),
        "abort_threshold": float(args.abort_threshold),
        "distance_weights": (1.0, float(args.momentum_weight)),
        "match_sequence_points": False,
        "trajectory_rank_weight": 0.0,
        "prefer_earliest_within_support": False,
        "u_min": np.array([-20.0], dtype=np.float32),
        "u_max": np.array([20.0], dtype=np.float32),
    }
    bc_cfg = {
        "hidden_sizes": (24, 24, 16),
        "lr": 1.2e-3,
        "weight_decay": 5e-4,
        "batch_size": 256,
        "epochs": max(int(args.bc_epochs), 40),
        "seed": 7,
        "device": "cpu",
        "normalize_obs": True,
        "normalize_actions": True,
        "use_angle_features": True,
    }

    n_traj_chain, success_rates_chain, diag_chain = evaluate_incremental_chain_policy(
        spec=spec,
        save_dir=save_dir,
        target_names=rollout_names,
        init_states=eval_init_states,
        dt=0.02,
        episode_seconds=float(args.eval_episode_seconds),
        controller_kwargs=chain_cfg,
        alphabet_filename="control_alphabet_single_pendulum_components_paper.pkl",
        result_filename="incremental_tube_success_rates_single_pendulum_final.npz",
        use_fast_rollout=True,
        fast_num_workers=int(args.fast_num_workers),
        plot=False,
        verbose=False,
    )
    print(f"[single-components] Chain Policy: {success_rates_chain}", flush=True)

    bc_names = ["lib", "cw_long", "ccw_long", "lib_right", "lib_left"]

    n_traj_bc, success_rates_bc, _ = evaluate_incremental_behavior_cloning(
        spec=spec,
        save_dir=save_dir,
        target_names=bc_names,
        init_states=eval_init_states,
        npz_filename=rollout_npz.name,
        dt=0.02,
        episode_seconds=float(args.eval_episode_seconds),
        bc_kwargs=bc_cfg,
        result_filename="incremental_vanilla_bc_success_rates_single_pendulum_final.npz",
        use_fast_rollout=True,
        plot=False,
        verbose=False,
    )
    print(f"[single-components] Vanilla BC: {success_rates_bc}", flush=True)

    print(f"[single-components] n_traj_chain={n_traj_chain}", flush=True)
    print(f"[single-components] n_traj_bc={n_traj_bc}", flush=True)

    _save_chain_vs_bc_figure(
        n_traj_chain=n_traj_chain,
        success_rates_chain=success_rates_chain,
        n_traj_bc=n_traj_bc,
        success_rates_bc=success_rates_bc,
        figure_path=save_dir / "single_pendulum_chain_vs_bc_bar.png",
        show_plot=False,
    )
    print(
        f"[single-components] saved figure -> {save_dir / 'single_pendulum_chain_vs_bc_bar.png'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
