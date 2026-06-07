import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dynamics.Dynamics_single_pendulum import I as SINGLE_I
from dynamics.Dynamics_single_pendulum import g as SINGLE_G
from dynamics.Dynamics_single_pendulum import l as SINGLE_L
from dynamics.Dynamics_single_pendulum import m as SINGLE_M
from dynamics.Dynamics_spring_mass import k_spring as SPRING_K
from dynamics.Dynamics_spring_mass import m as SPRING_M
from utils.behavior_cloning_eval import (
    evaluate_incremental_behavior_cloning,
    select_behavior_cloning_trajectory_subset,
)
from utils.control_chain import (
    ChainPolicySpec,
    TubeController,
    _double_apply_default_tracking_batch,
    _double_apply_tracking_batch,
    _double_distance_batch,
    _double_distance_ref_batch,
    _double_rk4_step_batch,
    _double_success_batch,
    _gather_double_refs_and_controls,
    _prepare_chain_controller_kwargs,
    _build_single_support_tree,
    _query_single_support_tree,
    _rk4_step,
    _single_distance_batch,
    _single_distance_ref_batch,
    _single_rk4_step_batch,
    _single_success_batch,
    build_control_alphabet_from_saved_rollouts,
    evaluate_incremental_chain_policy,
    greedy_rank_expert_trajectories,
    load_control_alphabet,
    make_double_pendulum_spec,
    make_single_pendulum_spec,
    make_spring_mass_spec,
)


def _expand_path(path_like):
    return Path(os.path.expanduser(str(path_like)))


def _best_grid_factors(num_points):
    num_points = int(num_points)
    best_pair = (1, num_points)
    best_gap = num_points - 1
    for rows in range(1, int(np.sqrt(num_points)) + 1):
        if num_points % rows != 0:
            continue
        cols = num_points // rows
        gap = abs(cols - rows)
        if gap < best_gap:
            best_gap = gap
            best_pair = (rows, cols)
    return best_pair


def make_uniform_position_initial_states(system_name, num_inits):
    if system_name == "single_pendulum":
        states = np.zeros((int(num_inits), 2), dtype=np.float32)
        states[:, 0] = np.linspace(-np.pi, np.pi, int(num_inits), endpoint=False, dtype=np.float32)
        return states

    if system_name == "spring_mass":
        states = np.zeros((int(num_inits), 2), dtype=np.float32)
        states[:, 0] = np.linspace(-2.0, 2.0, int(num_inits), endpoint=True, dtype=np.float32)
        return states

    if system_name == "double_pendulum":
        rows, cols = _best_grid_factors(int(num_inits))
        th1 = np.linspace(-np.pi, np.pi, rows, endpoint=False, dtype=np.float32)
        th2 = np.linspace(-np.pi, np.pi, cols, endpoint=False, dtype=np.float32)
        grid_th1, grid_th2 = np.meshgrid(th1, th2, indexing="ij")
        return np.stack(
            [
                grid_th1.reshape(-1),
                grid_th2.reshape(-1),
                np.zeros(grid_th1.size, dtype=np.float32),
                np.zeros(grid_th1.size, dtype=np.float32),
            ],
            axis=1,
        )

    raise ValueError(f"Unsupported system_name: {system_name}")


def _single_pendulum_energy(states):
    X = np.asarray(states, dtype=float)
    theta = X[:, 0]
    p = X[:, 1]
    return 0.5 * (p**2) / SINGLE_I + SINGLE_M * SINGLE_G * SINGLE_L * (1.0 - np.cos(theta))


def _spring_mass_energy(states):
    X = np.asarray(states, dtype=float)
    q = X[:, 0]
    p = X[:, 1]
    return 0.5 * (p**2) / SPRING_M + 0.5 * SPRING_K * q**2


def make_uniform_energy_bounded_initial_states(system_name, num_inits, H_bar, seed=123):
    """Rejection-sample states uniformly from a simple bounding box for H(x)<=H_bar."""
    rng = np.random.default_rng(int(seed))
    num_inits = int(num_inits)
    H_bar = float(H_bar)

    if system_name == "spring_mass":
        q_max = np.sqrt(2.0 * H_bar / SPRING_K)
        p_max = np.sqrt(2.0 * SPRING_M * H_bar)
        states = []
        while len(states) < num_inits:
            batch = max(4 * (num_inits - len(states)), 256)
            q = rng.uniform(-q_max, q_max, size=batch)
            p = rng.uniform(-p_max, p_max, size=batch)
            candidates = np.stack([q, p], axis=1)
            accepted = candidates[_spring_mass_energy(candidates) <= H_bar + 1e-9]
            states.extend(accepted[: num_inits - len(states)].tolist())
        return np.asarray(states, dtype=np.float32)

    if system_name == "single_pendulum":
        p_max = np.sqrt(2.0 * SINGLE_I * H_bar)
        states = []
        while len(states) < num_inits:
            batch = max(4 * (num_inits - len(states)), 256)
            theta = rng.uniform(-np.pi, np.pi, size=batch)
            p = rng.uniform(-p_max, p_max, size=batch)
            candidates = np.stack([theta, p], axis=1)
            accepted = candidates[_single_pendulum_energy(candidates) <= H_bar + 1e-9]
            states.extend(accepted[: num_inits - len(states)].tolist())
        return np.asarray(states, dtype=np.float32)

    return make_uniform_position_initial_states(system_name, num_inits)


def _make_eval_initial_states(config, num_inits, seed):
    H_bar = config.get("H_bar")
    if H_bar is None:
        return make_uniform_position_initial_states(config["system_name"], num_inits)
    return make_uniform_energy_bounded_initial_states(
        config["system_name"],
        num_inits,
        H_bar=H_bar,
        seed=seed,
    )


def _save_chain_vs_bc_figure(
    n_traj_chain,
    success_rates_chain,
    n_traj_bc,
    success_rates_bc,
    figure_path,
    show_plot=True,
):
    figure_path = _expand_path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    n_traj_chain = np.asarray(n_traj_chain, dtype=int)
    success_rates_chain = np.asarray(success_rates_chain, dtype=float)
    n_traj_bc = np.asarray(n_traj_bc, dtype=int)
    success_rates_bc = np.asarray(success_rates_bc, dtype=float)
    x = np.arange(1, max(int(np.max(n_traj_chain)), int(np.max(n_traj_bc))) + 1)
    chain_map = {int(k): float(v) for k, v in zip(n_traj_chain, success_rates_chain)}
    bc_map = {int(k): float(v) for k, v in zip(n_traj_bc, success_rates_bc)}
    chain_vals = np.asarray([chain_map.get(int(k), np.nan) for k in x], dtype=float)
    bc_vals = np.asarray([bc_map.get(int(k), np.nan) for k in x], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))
    width = 0.36
    ax.bar(x - width / 2.0, chain_vals, width, label="Chain Policy", color="#4C78A8", edgecolor="none")
    ax.bar(x + width / 2.0, bc_vals, width, label="Vanilla BC", color="#F58518", edgecolor="none")
    ax.set_xlabel("Number of Expert Trajectories")
    ax.set_ylabel("Success Rate")
    ax.set_xticks(x)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="y", color="black", linestyle="--", alpha=0.45)
    ax.legend(frameon=True, facecolor="white", edgecolor="black", framealpha=1.0)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _save_chain_time_figure(n_traj_list, mean_times, std_times, figure_path, show_plot=True):
    figure_path = _expand_path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(
        n_traj_list,
        mean_times,
        width=0.55,
        color="#4C78A8",
        alpha=0.85,
        edgecolor="none",
    )
    ax.errorbar(
        n_traj_list,
        mean_times,
        yerr=std_times,
        fmt="none",
        ecolor="black",
        elinewidth=1.6,
        capsize=5,
        capthick=1.6,
    )
    ax.set_xlabel("Number of Expert Trajectories")
    ax.set_ylabel("Average Time to Target (s)")
    ax.grid(axis="y", color="black", linestyle="--", alpha=0.45)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def _single_eval_config(save_dir):
    # Component order: libration from bottom rest, clockwise rotation,
    # counterclockwise rotation, then two auxiliary libration-side rollouts.
    names = ["lib", "cw_long", "ccw_long", "lib_right", "lib_left"]
    return {
        "system_name": "single_pendulum",
        "spec": make_single_pendulum_spec(
            success_tol=np.array([0.1, 0.1], dtype=np.float32),
            success_norm_eps=0.1,
        ),
        "candidate_names": names,
        "npz_filename": "all_rollouts_single_pendulum_components.npz",
        "alphabet_filename": "control_alphabet_single_pendulum_components_paper.pkl",
        "chain_result_filename": "incremental_tube_success_rates_single_pendulum_final.npz",
        "bc_result_filename": "incremental_vanilla_bc_success_rates_single_pendulum_final.npz",
        "figure_filename": "single_pendulum_chain_vs_bc_bar.png",
        "H_bar": 160.0,
        "dt": 0.02,
        "episode_seconds": 150.0,
        "chain_episode_seconds": 150.0,
        "bc_episode_seconds": 150.0,
        "selection_num_inits": 20,
        "chain_max_trajectories": 5,
        "bc_max_trajectories": 5,
        "chain_names": names,
        "bc_names": names,
        "generation_cfg": {
            "rho": 1.0,
            "H_star": 0.0,
            "eta": 0.0,
            "r_min": 1e-8,
            "max_lookahead": 200,
            "advance_stride": None,
            "eps_L": 1e-5,
            "v0": 1e-6,
            "radius_mode": "paper",
        },
        "chain_cfg": {
            "execute_full_sequence": True,
            "outside_mode": "default",
            "radius_scale": 1.0,
            "radius_floor": 0.055,
            "enter_threshold": 1.08,
            "abort_threshold": np.inf,
            "distance_weights": (1.0, 0.2),
            "match_sequence_points": False,
            "sequence_match_stride": 1,
            "match_sequence_tube_limit": None,
            "trajectory_rank_weight": 0.0,
            "prefer_earliest_within_support": False,
            "u_min": np.array([-20.0], dtype=np.float32),
            "u_max": np.array([20.0], dtype=np.float32),
        },
        "bc_cfg": {
            "hidden_sizes": (24, 24, 16),
            "lr": 1.2e-3,
            "weight_decay": 5e-4,
            "batch_size": 256,
            "epochs": 40,
            "seed": 7,
            "device": "cpu",
            "normalize_obs": True,
            "normalize_actions": True,
            "use_angle_features": True,
            "overrides_by_k": {
                2: {
                    "hidden_sizes": (32, 32, 24),
                    "lr": 1.0e-3,
                    "weight_decay": 2e-4,
                    "epochs": 80,
                    # evaluate_incremental_behavior_cloning adds k to the seed;
                    # seed=8 gives actual training seed 10 for k=2.
                    "seed": 8,
                },
                4: {
                    # seed=0 gives actual training seed 4 for k=4, which avoids
                    # the unstable single-side trajectory local minimum.
                    "seed": 0,
                },
            },
        },
    }


def _spring_eval_config(save_dir):
    names = [
        "x0_m2_0",
        "x1_m1_0",
        "x2_0_0",
        "x3_1_0",
        "x4_2_0",
        "x5_0_2",
        "x6_0_m2",
        "x7_1p5_m1",
    ]
    return {
        "system_name": "spring_mass",
        "spec": make_spring_mass_spec(
            success_tol=np.array([0.1, 0.1], dtype=np.float32),
            success_norm_eps=0.1,
        ),
        "candidate_names": names,
        "npz_filename": "all_rollouts_spring_mass.npz",
        "alphabet_filename": "control_alphabet_spring_mass_paper.pkl",
        "chain_result_filename": "incremental_tube_success_rates_spring_mass_final.npz",
        "bc_result_filename": "incremental_vanilla_bc_success_rates_spring_mass_final.npz",
        "figure_filename": "spring_mass_chain_vs_bc_bar.png",
        "H_bar": 2.0,
        "dt": 0.02,
        "episode_seconds": 20.0,
        "chain_episode_seconds": 20.0,
        "bc_episode_seconds": 20.0,
        "selection_num_inits": 20,
        "chain_max_trajectories": 5,
        "bc_max_trajectories": 5,
        "chain_names": ["x4_2_0", "x0_m2_0", "x6_0_m2", "x5_0_2", "x7_1p5_m1"],
        "bc_names": ["x4_2_0", "x6_0_m2", "x5_0_2", "x3_1_0", "x0_m2_0"],
        "generation_cfg": {
            "rho": 1.0,
            "H_star": 0.0,
            "eta": 0.0,
            "r_min": 1e-10,
            "max_lookahead": 80,
            "advance_stride": None,
            "eps_L": 1e-6,
            "v0": 1e-6,
            "radius_mode": "paper",
        },
        "chain_cfg": {
            "execute_full_sequence": True,
            "outside_mode": "default",
            "radius_scale": 1.0,
            "enter_threshold": 1.0,
            "abort_threshold": np.inf,
            "distance_weights": (1.0, 1.0),
            "match_sequence_points": False,
            "sequence_match_stride": 1,
            "match_sequence_tube_limit": None,
            "trajectory_rank_weight": 0.0,
            "prefer_earliest_within_support": False,
            "u_min": np.array([-20.0], dtype=np.float32),
            "u_max": np.array([20.0], dtype=np.float32),
        },
        "bc_cfg": {
            "hidden_sizes": (24, 24, 16),
            "lr": 1.2e-3,
            "weight_decay": 5e-4,
            "batch_size": 256,
            "epochs": 40,
            "seed": 7,
            "device": "cpu",
            "normalize_obs": True,
            "normalize_actions": True,
            "use_angle_features": False,
        },
    }


def _double_eval_config(save_dir):
    names = ["o4", "o1", "o0", "o3", "o2"]
    return {
        "system_name": "double_pendulum",
        "spec": make_double_pendulum_spec(success_tol=np.array([0.4, 0.4, 0.8, 0.8], dtype=np.float32)),
        "candidate_names": names,
        "npz_filename": "all_rollouts_double_pendulum_orbit5.npz",
        "alphabet_filename": "control_alphabet_double_pendulum_orbit5.pkl",
        "chain_result_filename": "incremental_tube_success_rates_double_pendulum_final.npz",
        "bc_result_filename": "incremental_vanilla_bc_success_rates_double_pendulum_final.npz",
        "figure_filename": "double_pendulum_chain_vs_bc.png",
        "dt": 0.02,
        "episode_seconds": 20.0,
        "chain_episode_seconds": 20.0,
        "bc_episode_seconds": 20.0,
        "selection_num_inits": 24,
        "fast_num_workers": 8,
        "chain_max_trajectories": 5,
        "bc_max_trajectories": 5,
        "chain_names": ["o4", "o1", "o0", "o3", "o2"],
        "bc_names": ["o4", "o1", "o0", "o3", "o2"],
        "generation_cfg": {
            "rho": 0.999,
            "H_star": 0.0,
            "eta": 0.0,
            "r_min": 1e-8,
            "max_lookahead": 120,
            "advance_stride": 1,
            "eps_L": 1e-5,
        },
        "chain_cfg": {
            "execute_full_sequence": True,
            "outside_mode": "nearest_tracking",
            "radius_scale": 5.0,
            "abort_threshold": 60.0,
            "distance_weights": (1.0, 1.0, 0.01, 0.01),
            "outside_tracking_gain": np.array(
                [[12.0, 0.0, 4.0, 0.0], [0.0, 12.0, 0.0, 4.0]],
                dtype=np.float32,
            ),
            "project_tracking": True,
            "match_sequence_points": True,
            "sequence_match_stride": 1,
            "match_sequence_tube_limit": 160,
            "trajectory_rank_weight": 1.0,
            "u_min": np.array([-12.0, -12.0], dtype=np.float32),
            "u_max": np.array([12.0, 12.0], dtype=np.float32),
        },
        "bc_cfg": {
            "hidden_sizes": (192, 128, 64),
            "lr": 8.0e-4,
            "weight_decay": 1e-5,
            "batch_size": 256,
            "epochs": 100,
            "seed": 7,
            "device": "cpu",
            "normalize_obs": True,
            "normalize_actions": True,
            "use_angle_features": True,
            "max_samples_per_trajectory": 400,
            "overrides_by_k": {
                5: {
                    "hidden_sizes": (128, 128, 64),
                    "lr": 1.0e-3,
                    "weight_decay": 5e-5,
                    "batch_size": 256,
                    "epochs": 60,
                    "seed": 7,
                    "device": "cpu",
                    "normalize_obs": True,
                    "normalize_actions": True,
                    "use_angle_features": True,
                }
            },
        },
    }


def _get_eval_config(system_name, save_dir):
    if system_name == "single_pendulum":
        return _single_eval_config(save_dir)
    if system_name == "spring_mass":
        return _spring_eval_config(save_dir)
    if system_name == "double_pendulum":
        return _double_eval_config(save_dir)
    raise ValueError(f"Unsupported system_name: {system_name}")


def _ensure_alphabet_if_missing(config, save_dir):
    alphabet_path = _expand_path(save_dir) / config["alphabet_filename"]
    if alphabet_path.exists():
        return
    build_kwargs = dict(config["generation_cfg"])
    build_control_alphabet_from_saved_rollouts(
        spec=config["spec"],
        save_dir=save_dir,
        names=config["candidate_names"],
        npz_filename=config["npz_filename"],
        alphabet_filename=config["alphabet_filename"],
        verbose=False,
        save_metrics=False,
        **build_kwargs,
    )


def run_final_uniform_position_eval(
    system_name,
    save_dir="~/exp/Nonparameric_Hamiltonian_Systems/data",
    num_inits=500,
    selection_num_inits=None,
    max_drop=0.05,
    show_plot=True,
    verbose=True,
):
    save_dir = _expand_path(save_dir)
    config = _get_eval_config(system_name, save_dir)
    selection_num_inits = (
        config["selection_num_inits"]
        if selection_num_inits is None
        else int(selection_num_inits)
    )

    _ensure_alphabet_if_missing(config, save_dir)

    selection_init_states = _make_eval_initial_states(config, selection_num_inits, seed=321)
    final_init_states = _make_eval_initial_states(config, num_inits, seed=123)
    chain_episode_seconds = float(config.get("chain_episode_seconds", config["episode_seconds"]))
    bc_episode_seconds = float(config.get("bc_episode_seconds", config["episode_seconds"]))

    selected_chain_names = list(config.get("chain_names") or [])
    chain_selection_records = []
    if not selected_chain_names:
        selected_chain_names, chain_selection_records = greedy_rank_expert_trajectories(
            spec=config["spec"],
            save_dir=save_dir,
            candidate_names=config["candidate_names"],
            init_states=selection_init_states,
            dt=config["dt"],
            episode_seconds=chain_episode_seconds,
            controller_kwargs=config["chain_cfg"],
            alphabet_filename=config["alphabet_filename"],
            max_drop=max_drop,
            stop_on_large_drop=True,
            max_trajectories=config["chain_max_trajectories"],
            use_fast_rollout=True,
            fast_num_workers=1,
            verbose=verbose,
        )

    selected_bc_names = list(config.get("bc_names") or [])
    bc_selection_records = []
    if not selected_bc_names:
        bc_ordered_names = selected_chain_names + [
            name for name in config["candidate_names"] if name not in selected_chain_names
        ]
        selected_bc_names, bc_selection_records = select_behavior_cloning_trajectory_subset(
            spec=config["spec"],
            save_dir=save_dir,
            ordered_names=bc_ordered_names,
            init_states=selection_init_states,
            npz_filename=config["npz_filename"],
            dt=config["dt"],
            episode_seconds=bc_episode_seconds,
            bc_kwargs=config["bc_cfg"],
            max_drop=max_drop,
            use_fast_rollout=True,
            verbose=verbose,
        )
        if config["bc_max_trajectories"] is not None:
            selected_bc_names = selected_bc_names[: int(config["bc_max_trajectories"])]

    n_traj_chain, success_rates_chain, diag_chain = evaluate_incremental_chain_policy(
        spec=config["spec"],
        save_dir=save_dir,
        target_names=selected_chain_names,
        init_states=final_init_states,
        dt=config["dt"],
        episode_seconds=chain_episode_seconds,
        controller_kwargs=config["chain_cfg"],
        alphabet_filename=config["alphabet_filename"],
        result_filename=config["chain_result_filename"],
        use_fast_rollout=True,
        fast_num_workers=int(config.get("fast_num_workers", 1)),
        plot=False,
        verbose=verbose,
    )

    n_traj_bc, success_rates_bc, diag_bc = evaluate_incremental_behavior_cloning(
        spec=config["spec"],
        save_dir=save_dir,
        target_names=selected_bc_names,
        init_states=final_init_states,
        npz_filename=config["npz_filename"],
        dt=config["dt"],
        episode_seconds=bc_episode_seconds,
        bc_kwargs=config["bc_cfg"],
        result_filename=config["bc_result_filename"],
        use_fast_rollout=True,
        plot=False,
        verbose=verbose,
    )

    figure_path = save_dir / config["figure_filename"]
    _save_chain_vs_bc_figure(
        n_traj_chain=n_traj_chain,
        success_rates_chain=success_rates_chain,
        n_traj_bc=n_traj_bc,
        success_rates_bc=success_rates_bc,
        figure_path=figure_path,
        show_plot=show_plot,
    )

    return {
        "system_name": system_name,
        "num_inits": int(num_inits),
        "selection_num_inits": int(selection_num_inits),
        "selected_chain_names": list(selected_chain_names),
        "selected_bc_names": list(selected_bc_names),
        "chain_success_rates": list(success_rates_chain),
        "bc_success_rates": list(success_rates_bc),
        "chain_result_path": str(save_dir / config["chain_result_filename"]),
        "bc_result_path": str(save_dir / config["bc_result_filename"]),
        "figure_path": str(figure_path),
        "chain_selection_records": chain_selection_records,
        "bc_selection_records": bc_selection_records,
        "chain_diag": np.asarray(diag_chain).tolist(),
        "bc_diag": np.asarray(diag_bc).tolist(),
    }


def run_paper_numerical_eval(
    save_dir="~/exp/Nonparameric_Hamiltonian_Systems/data",
    num_inits=500,
    show_plot=True,
    verbose=True,
):
    """Run the numerical experiments described in the current manuscript."""
    results = {}
    for system_name in ("spring_mass", "single_pendulum"):
        results[system_name] = run_final_uniform_position_eval(
            system_name=system_name,
            save_dir=save_dir,
            num_inits=num_inits,
            show_plot=show_plot,
            verbose=verbose,
        )
    return results


def _run_chain_episode_hitting_time_fast(spec, controller, x0, dt, episode_seconds):
    obs = np.asarray(x0, dtype=np.float32).reshape(spec.state_dim)
    controller.reset_episode(x0=obs)

    if spec.check_success(obs):
        return {"success": True, "steps_to_success": 0}

    horizon_steps = max(int(np.round(float(episode_seconds) / float(dt))), 1)
    for step_idx in range(1, horizon_steps + 1):
        u, _, _ = controller.get_action(obs)
        obs = _rk4_step(spec.f_continuous_fn, obs, u, dt)
        if spec.check_success(obs):
            return {"success": True, "steps_to_success": int(step_idx)}

    return {"success": False, "steps_to_success": int(horizon_steps)}


def _run_chain_hitting_time_single_fast_batch(
    spec: ChainPolicySpec,
    controller,
    init_states,
    dt,
    episode_seconds,
):
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
    steps_to_success = np.full(n_states, horizon_steps, dtype=np.int32)
    steps_to_success[success] = 0

    centers = controller.centers
    radii = controller.radii
    weights = controller.distance_weights
    eps = float(controller.eps)
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

    support_tree = None
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

    active_idx = np.full(n_states, -1, dtype=np.int32)
    active_k = np.zeros(n_states, dtype=np.int32)

    for step_idx in range(1, horizon_steps + 1):
        pending = ~success
        if not np.any(pending):
            break

        actions = np.tile(controller.default_u.reshape(1, -1), (n_states, 1)).astype(np.float32)
        status_expert = np.zeros(n_states, dtype=bool)
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
                                recover_rho = float(recover_dists[best_local] / (radii[tube_idx] + eps))
                                if recover_rho <= controller.abort_threshold and recover_step < seq_u.shape[0]:
                                    active_k[state_idx] = recover_step + 1
                                    actions[state_idx] = controller._apply_tracking(
                                        tube_idx,
                                        recover_step,
                                        states[state_idx],
                                        seq_u[recover_step],
                                    )
                                    status_expert[state_idx] = True
                                    need_select[state_idx] = False
                                    recovered = True
                        if not recovered:
                            active_idx[state_idx] = -1
                            active_k[state_idx] = 0

        select_mask = pending & need_select
        if np.any(select_mask):
            idxs = np.flatnonzero(select_mask)
            if sequence_match_enabled:
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

        states[pending] = _single_rk4_step_batch(states[pending], actions[pending], dt)
        newly_success = pending & _single_success_batch(spec, states)
        steps_to_success[newly_success] = step_idx
        success |= newly_success

    return {
        "success": success.astype(bool),
        "steps_to_success": steps_to_success.astype(np.int32),
    }


def _run_chain_hitting_time_double_fast_batch(
    spec: ChainPolicySpec,
    controller,
    alphabet_dict,
    init_states,
    dt,
    episode_seconds,
):
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
    steps_to_success = np.full(n_states, horizon_steps, dtype=np.int32)
    steps_to_success[success] = 0

    centers = controller.centers
    radii = controller.radii
    weights = controller.distance_weights
    eps = float(controller.eps)
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

    for step_idx in range(1, horizon_steps + 1):
        pending = ~success
        if not np.any(pending):
            break

        actions = np.tile(controller.default_u.reshape(1, -1), (n_states, 1)).astype(np.float32)
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
                    seq_step = int(active_k[state_idx])
                    refs_now, nominals_now = _gather_double_refs_and_controls(
                        controller,
                        np.array([tube_idx], dtype=np.int32),
                        np.array([seq_step], dtype=np.int32),
                    )
                    actions[state_idx : state_idx + 1] = _double_apply_tracking_batch(
                        controller,
                        states[state_idx : state_idx + 1],
                        refs_now,
                        nominals_now,
                    )
                    active_k[state_idx] = seq_step + 1
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

        states[pending] = _double_rk4_step_batch(states[pending], actions[pending], dt)
        newly_success = pending & _double_success_batch(spec, states)
        steps_to_success[newly_success] = step_idx
        success |= newly_success

    return {
        "success": success.astype(bool),
        "steps_to_success": steps_to_success.astype(np.int32),
    }


def evaluate_chain_hitting_times(
    system_name,
    save_dir="~/exp/Nonparameric_Hamiltonian_Systems/data",
    num_inits=500,
    show_plot=True,
    verbose=True,
    episode_seconds_override=None,
    result_filename_override=None,
    figure_filename_override=None,
    plot_all_times=True,
):
    save_dir = _expand_path(save_dir)
    config = _get_eval_config(system_name, save_dir)
    _ensure_alphabet_if_missing(config, save_dir)

    init_states = _make_eval_initial_states(config, num_inits, seed=123)
    dt = float(config["dt"])
    episode_seconds = float(
        config.get("chain_episode_seconds", config["episode_seconds"])
        if episode_seconds_override is None
        else episode_seconds_override
    )
    selected_chain_names = list(config.get("chain_names") or [])

    full_alphabet_dict = load_control_alphabet(save_dir, alphabet_filename=config["alphabet_filename"])
    controller_kwargs = _prepare_chain_controller_kwargs(
        spec=config["spec"],
        controller_kwargs=config["chain_cfg"],
        verbose=False,
    )

    n_traj_list = []
    success_rates = []
    mean_time_success = []
    mean_time_all = []
    std_time_success = []
    std_time_all = []
    diag = []
    time_samples_success_by_k = []
    time_samples_all_by_k = []

    for k in range(1, len(selected_chain_names) + 1):
        subset_names = selected_chain_names[:k]
        subset_alphabet = {
            name: full_alphabet_dict[name]
            for name in subset_names
            if name in full_alphabet_dict
        }
        controller = TubeController(subset_alphabet, **controller_kwargs)

        if system_name == "single_pendulum":
            batch_episode = _run_chain_hitting_time_single_fast_batch(
                spec=config["spec"],
                controller=controller,
                init_states=init_states,
                dt=dt,
                episode_seconds=episode_seconds,
            )
        elif system_name == "double_pendulum":
            batch_episode = _run_chain_hitting_time_double_fast_batch(
                spec=config["spec"],
                controller=controller,
                alphabet_dict=subset_alphabet,
                init_states=init_states,
                dt=dt,
                episode_seconds=episode_seconds,
            )
        else:
            batch_episode = None

        if batch_episode is not None:
            success_flags = np.asarray(batch_episode["success"], dtype=bool)
            steps_to_success = np.asarray(batch_episode["steps_to_success"], dtype=np.int32)
            times_all = steps_to_success.astype(float) * dt
            times_success = times_all[success_flags]
        else:
            success_flags = []
            times_success = []
            times_all = []
            for x0 in init_states:
                episode = _run_chain_episode_hitting_time_fast(
                    spec=config["spec"],
                    controller=controller,
                    x0=x0,
                    dt=dt,
                    episode_seconds=episode_seconds,
                )
                t_hit = float(episode["steps_to_success"]) * dt
                success_flags.append(bool(episode["success"]))
                times_all.append(t_hit)
                if episode["success"]:
                    times_success.append(t_hit)

        success_count = int(np.sum(success_flags))
        times_success_arr = np.asarray(times_success, dtype=float)
        times_all_arr = np.asarray(times_all, dtype=float)
        rate = float(np.mean(success_flags))
        avg_success = float(np.mean(times_success_arr)) if times_success_arr.size else np.nan
        avg_all = float(np.mean(times_all_arr)) if times_all_arr.size else np.nan

        n_traj_list.append(k)
        success_rates.append(rate)
        mean_time_success.append(avg_success)
        mean_time_all.append(avg_all)
        std_time_success.append(float(np.std(times_success_arr)) if times_success_arr.size else np.nan)
        std_time_all.append(float(np.std(times_all_arr)) if times_all_arr.size else np.nan)
        diag.append((float(success_count), float(num_inits)))
        time_samples_success_by_k.append(times_success_arr)
        time_samples_all_by_k.append(times_all_arr)

        if verbose:
            print(
                f"[Eval-Time:{system_name}] k={k} -> success={rate:.2%}, "
                f"avg_success_time={avg_success:.3f}s, avg_all_time={avg_all:.3f}s"
            )

    result_filename = (
        f"incremental_chain_hitting_times_{system_name}_final.npz"
        if result_filename_override is None
        else str(result_filename_override)
    )
    result_path = save_dir / result_filename
    payload = {
        "n_traj_list": np.asarray(n_traj_list, dtype=int),
        "success_rates": np.asarray(success_rates, dtype=float),
        "mean_time_success": np.asarray(mean_time_success, dtype=float),
        "mean_time_all": np.asarray(mean_time_all, dtype=float),
        "std_time_success": np.asarray(std_time_success, dtype=float),
        "std_time_all": np.asarray(std_time_all, dtype=float),
        "diag": np.asarray(diag, dtype=float),
    }
    for idx, samples in enumerate(time_samples_success_by_k, start=1):
        payload[f"times_success_k{idx}"] = np.asarray(samples, dtype=float)
    for idx, samples in enumerate(time_samples_all_by_k, start=1):
        payload[f"times_all_k{idx}"] = np.asarray(samples, dtype=float)
    np.savez(result_path, **payload)

    figure_filename = (
        f"{system_name}_time_bar.png"
        if figure_filename_override is None
        else str(figure_filename_override)
    )
    figure_path = save_dir / figure_filename
    _save_chain_time_figure(
        n_traj_list=n_traj_list,
        mean_times=mean_time_all if plot_all_times else mean_time_success,
        std_times=std_time_all if plot_all_times else std_time_success,
        figure_path=figure_path,
        show_plot=show_plot,
    )

    return {
        "system_name": system_name,
        "num_inits": int(num_inits),
        "selected_chain_names": list(selected_chain_names),
        "n_traj_list": list(n_traj_list),
        "success_rates": list(success_rates),
        "mean_time_success": list(mean_time_success),
        "mean_time_all": list(mean_time_all),
        "std_time_success": list(std_time_success),
        "std_time_all": list(std_time_all),
        "result_path": str(result_path),
        "figure_path": str(figure_path),
    }
