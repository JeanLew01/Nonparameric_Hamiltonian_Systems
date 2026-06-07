from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from utils.behavior_cloning_eval import (
    _collect_bc_dataset,
    _run_bc_episodes_fast_double_batch,
    _run_bc_episodes_fast_single_batch,
    train_behavior_cloning_policy,
)
from utils.control_chain import (
    ChainPolicySpec,
    load_saved_rollouts,
    make_double_pendulum_spec,
    make_single_pendulum_spec,
    make_spring_mass_spec,
    success_mask_from_spec,
)
from utils.final_uniform_eval import make_uniform_energy_bounded_initial_states, make_uniform_position_initial_states


def _spring_f_batch(states, actions):
    X = np.asarray(states, dtype=np.float32)
    U = np.asarray(actions, dtype=np.float32).reshape(-1, 1)
    q = X[:, 0]
    p = X[:, 1]
    u = U[:, 0]
    m = np.float32(1.0)
    k = np.float32(1.0)
    q_dot = p / m
    p_dot = -k * q + u
    return np.stack([q_dot, p_dot], axis=1).astype(np.float32)


def _rk4_step_spring_batch(states, actions, dt):
    X = np.asarray(states, dtype=np.float32)
    U = np.asarray(actions, dtype=np.float32).reshape(-1, 1)
    dt = float(dt)
    k1 = _spring_f_batch(X, U)
    k2 = _spring_f_batch(X + 0.5 * dt * k1, U)
    k3 = _spring_f_batch(X + 0.5 * dt * k2, U)
    k4 = _spring_f_batch(X + dt * k3, U)
    return (X + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)).astype(np.float32)


def _check_success_batch(spec: ChainPolicySpec, obs_batch):
    return success_mask_from_spec(spec, obs_batch)


def _run_bc_episodes_fast_spring_batch(spec: ChainPolicySpec, controller, init_states, dt, episode_seconds):
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
        states[idx] = _rk4_step_spring_batch(states[idx], actions, dt)
        steps[idx] += 1
        newly_succeeded = _check_success_batch(spec, states[idx])
        if np.any(newly_succeeded):
            success[idx[newly_succeeded]] = True
            active[idx[newly_succeeded]] = False

    return {"success": success, "steps": steps.astype(np.int32)}


def _save_bar_figure(means, stds, figure_path: Path):
    preferred_fonts = [
        "Times New Roman",
        "Times",
        "Nimbus Roman",
        "TeX Gyre Termes",
        "STIX Two Text",
        "DejaVu Serif",
    ]
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    chosen_font = next((f for f in preferred_fonts if f in available_fonts), "DejaVu Serif")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [chosen_font],
            "mathtext.fontset": "stix",
            "axes.labelsize": 22,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    pos = np.arange(1, len(means) + 1)
    ax.bar(pos, means, width=0.55, color="#F58518", alpha=0.85, edgecolor="none")
    ax.errorbar(
        pos,
        means,
        yerr=stds,
        fmt="none",
        ecolor="black",
        elinewidth=1.6,
        capsize=5,
        capthick=1.6,
    )
    ax.set_xticks(pos)
    ax.set_xticklabels([str(k) for k in pos])
    ax.set_xlabel("Number of Trajectories")
    ax.set_ylabel("Average Time (s)")
    ax.grid(axis="y", color="black", alpha=0.5)
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)


def _single_bc_config(save_dir: Path):
    return {
        "spec": make_single_pendulum_spec(
            success_tol=np.array([0.1, 0.1], dtype=np.float32),
            success_norm_eps=0.1,
        ),
        "target_names": ["lib", "cw_long", "ccw_long", "lib_right", "lib_left"],
        "npz_filename": "all_rollouts_single_pendulum_components.npz",
        "init_states": make_uniform_energy_bounded_initial_states(
            "single_pendulum",
            num_inits=500,
            H_bar=160.0,
            seed=123,
        ),
        "dt": 0.02,
        "episode_seconds": 150.0,
        "bc_kwargs": {
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
                    # The runner adds k to the seed; seed=8 gives actual seed 10 for k=2.
                    "seed": 8,
                },
                4: {
                    # The runner adds k to the seed; seed=0 gives actual seed 4 for k=4.
                    "seed": 0,
                },
            },
        },
        "result_filename": "incremental_vanilla_bc_hitting_times_single_pendulum_final.npz",
        "figure_filename": "single_pendulum_bc_avg_time.png",
    }


def _spring_bc_config(save_dir: Path):
    return {
        "spec": make_spring_mass_spec(
            success_tol=np.array([0.1, 0.1], dtype=np.float32),
            success_norm_eps=0.1,
        ),
        "target_names": ["x4_2_0", "x6_0_m2", "x5_0_2", "x3_1_0", "x0_m2_0"],
        "npz_filename": "all_rollouts_spring_mass.npz",
        "init_states": make_uniform_energy_bounded_initial_states(
            "spring_mass",
            num_inits=500,
            H_bar=2.0,
            seed=123,
        ),
        "dt": 0.02,
        "episode_seconds": 20.0,
        "bc_kwargs": {
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
        "result_filename": "incremental_vanilla_bc_hitting_times_spring_mass_final.npz",
        "figure_filename": "spring_mass_bc_avg_time.png",
    }


def _double_bc_config(save_dir: Path):
    return {
        "spec": make_double_pendulum_spec(success_tol=np.array([0.4, 0.4, 0.8, 0.8], dtype=np.float32)),
        "target_names": ["o4", "o1", "o0", "o3", "o2"],
        "npz_filename": "all_rollouts_double_pendulum_orbit5.npz",
        "init_states": make_uniform_position_initial_states("double_pendulum", 500),
        "dt": 0.02,
        "episode_seconds": 20.0,
        "bc_kwargs": {
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
        "result_filename": "incremental_vanilla_bc_hitting_times_double_pendulum_final.npz",
        "figure_filename": "double_pendulum_bc_avg_time.png",
    }


def _get_config(system_name: str, save_dir: Path):
    if system_name == "single_pendulum":
        return _single_bc_config(save_dir)
    if system_name == "spring_mass":
        return _spring_bc_config(save_dir)
    if system_name == "double_pendulum":
        return _double_bc_config(save_dir)
    raise ValueError(f"Unsupported system_name: {system_name}")


def run_bc_hitting_times(
    system_name: str,
    save_dir: str | Path = "data",
    episode_seconds_override: float | None = None,
    result_filename_override: str | None = None,
    figure_filename_override: str | None = None,
):
    save_dir = Path(save_dir)
    config = _get_config(system_name, save_dir)
    spec = config["spec"]
    target_names = list(config["target_names"])
    init_states = np.asarray(config["init_states"], dtype=np.float32)
    dt = float(config["dt"])
    episode_seconds = float(
        config["episode_seconds"] if episode_seconds_override is None else episode_seconds_override
    )
    bc_kwargs = dict(config["bc_kwargs"])
    overrides_by_k = {
        int(k): dict(v) for k, v in dict(bc_kwargs.pop("overrides_by_k", {})).items()
    }

    rollouts = load_saved_rollouts(save_dir, target_names, npz_filename=config["npz_filename"])
    env_template = spec.make_env(dt=dt, episode_seconds=episode_seconds)
    u_min = np.asarray(env_template.action_space.low, dtype=np.float32).reshape(spec.control_dim)
    u_max = np.asarray(env_template.action_space.high, dtype=np.float32).reshape(spec.control_dim)
    n_traj_list = []
    success_rates = []
    mean_time_success = []
    mean_time_all = []
    std_time_success = []
    std_time_all = []
    payload = {}

    for k in range(1, len(target_names) + 1):
        current_bc_kwargs = dict(bc_kwargs)
        if k in overrides_by_k:
            current_bc_kwargs.update(overrides_by_k[k])

        max_samples_per_trajectory = current_bc_kwargs.get("max_samples_per_trajectory", None)
        current_names = target_names[:k]
        states, actions, _ = _collect_bc_dataset(
            rollouts=rollouts,
            target_names=current_names,
            control_dim=spec.control_dim,
            max_samples_per_trajectory=max_samples_per_trajectory,
        )
        controller, _ = train_behavior_cloning_policy(
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

        if system_name == "single_pendulum":
            batch = _run_bc_episodes_fast_single_batch(spec, controller, init_states, dt, episode_seconds)
        elif system_name == "double_pendulum":
            batch = _run_bc_episodes_fast_double_batch(spec, controller, init_states, dt, episode_seconds)
        else:
            batch = _run_bc_episodes_fast_spring_batch(spec, controller, init_states, dt, episode_seconds)

        success = np.asarray(batch["success"], dtype=bool)
        steps = np.asarray(batch["steps"], dtype=np.int32)
        times_all = steps.astype(float) * dt
        times_success = times_all[success]

        success_rates.append(float(np.mean(success)))
        n_traj_list.append(k)
        mean_time_success.append(float(np.mean(times_success)) if times_success.size else np.nan)
        mean_time_all.append(float(np.mean(times_all)) if times_all.size else np.nan)
        std_time_success.append(float(np.std(times_success)) if times_success.size else np.nan)
        std_time_all.append(float(np.std(times_all)) if times_all.size else np.nan)
        payload[f"times_success_k{k}"] = np.asarray(times_success, dtype=float)
        payload[f"times_all_k{k}"] = np.asarray(times_all, dtype=float)

    payload.update(
        {
            "n_traj_list": np.asarray(n_traj_list, dtype=int),
            "success_rates": np.asarray(success_rates, dtype=float),
            "mean_time_success": np.asarray(mean_time_success, dtype=float),
            "mean_time_all": np.asarray(mean_time_all, dtype=float),
            "std_time_success": np.asarray(std_time_success, dtype=float),
            "std_time_all": np.asarray(std_time_all, dtype=float),
        }
    )

    result_path = save_dir / str(
        config["result_filename"] if result_filename_override is None else result_filename_override
    )
    np.savez(result_path, **payload)

    figure_path = save_dir / str(
        config["figure_filename"] if figure_filename_override is None else figure_filename_override
    )
    _save_bar_figure(
        means=np.asarray(mean_time_all, dtype=float),
        stds=np.asarray(std_time_all, dtype=float),
        figure_path=figure_path,
    )

    return {
        "system_name": system_name,
        "result_path": str(result_path),
        "figure_path": str(figure_path),
        "n_traj_list": list(n_traj_list),
        "success_rates": list(success_rates),
        "mean_time_success": list(mean_time_success),
        "std_time_success": list(std_time_success),
        "mean_time_all": list(mean_time_all),
        "std_time_all": list(std_time_all),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["single_pendulum", "spring_mass", "double_pendulum"], required=True)
    parser.add_argument("--save-dir", type=str, default="data")
    args = parser.parse_args()

    result = run_bc_hitting_times(system_name=args.system, save_dir=args.save_dir)
    print(result, flush=True)


if __name__ == "__main__":
    main()
