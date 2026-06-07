from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.final_uniform_eval import evaluate_chain_hitting_times, run_final_uniform_position_eval
from utils.run_bc_hitting_times import run_bc_hitting_times


METHOD_COLORS = {
    "chain": "#4C78A8",
    "bc": "#F58518",
}


def _as_float_array(values):
    return np.asarray(values, dtype=float).reshape(-1)


def _bar_pair(ax, x, chain_values, bc_values, ylabel, ylim=None, chain_err=None, bc_err=None):
    width = 0.36
    ax.bar(
        x - width / 2.0,
        chain_values,
        width,
        yerr=chain_err,
        capsize=4 if chain_err is not None else 0,
        label="Chain Policy",
        color=METHOD_COLORS["chain"],
        edgecolor="none",
        ecolor="black",
    )
    ax.bar(
        x + width / 2.0,
        bc_values,
        width,
        yerr=bc_err,
        capsize=4 if bc_err is not None else 0,
        label="Vanilla BC",
        color=METHOD_COLORS["bc"],
        edgecolor="none",
        ecolor="black",
    )
    ax.set_xticks(x)
    ax.set_xlabel("Number of Expert Trajectories")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", color="black", linestyle="--", alpha=0.45)


def save_system_success_time_figure(system_name, result, figure_path):
    n_traj = np.asarray(result["n_traj_list"], dtype=int)
    chain_success = _as_float_array(result["chain_success_rates"])
    bc_success = _as_float_array(result["bc_success_rates"])
    chain_mean_time = _as_float_array(result["chain_mean_time_all"])
    bc_mean_time = _as_float_array(result["bc_mean_time_all"])
    chain_std_time = _as_float_array(result["chain_std_time_all"])
    bc_std_time = _as_float_array(result["bc_std_time_all"])

    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    _bar_pair(
        axes[0],
        n_traj,
        chain_success,
        bc_success,
        ylabel="Success Rate",
        ylim=(-0.05, 1.05),
    )
    _bar_pair(
        axes[1],
        n_traj,
        chain_mean_time,
        bc_mean_time,
        ylabel="Average Time to Target (s)",
        chain_err=chain_std_time,
        bc_err=bc_std_time,
    )
    axes[0].legend(frameon=True, facecolor="white", edgecolor="black", framealpha=1.0)
    fig.suptitle(system_name.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_all_systems_success_time_figure(results, figure_path):
    system_names = list(results.keys())
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(system_names), 2, figsize=(13, 5.4 * len(system_names)))
    if len(system_names) == 1:
        axes = np.asarray([axes])

    for row, system_name in enumerate(system_names):
        result = results[system_name]
        n_traj = np.asarray(result["n_traj_list"], dtype=int)
        chain_success = _as_float_array(result["chain_success_rates"])
        bc_success = _as_float_array(result["bc_success_rates"])
        chain_mean_time = _as_float_array(result["chain_mean_time_all"])
        bc_mean_time = _as_float_array(result["bc_mean_time_all"])
        chain_std_time = _as_float_array(result["chain_std_time_all"])
        bc_std_time = _as_float_array(result["bc_std_time_all"])

        _bar_pair(
            axes[row, 0],
            n_traj,
            chain_success,
            bc_success,
            ylabel="Success Rate",
            ylim=(-0.05, 1.05),
        )
        _bar_pair(
            axes[row, 1],
            n_traj,
            chain_mean_time,
            bc_mean_time,
            ylabel="Average Time to Target (s)",
            chain_err=chain_std_time,
            bc_err=bc_std_time,
        )
        axes[row, 0].set_title(f"{system_name.replace('_', ' ').title()}: Success")
        axes[row, 1].set_title(f"{system_name.replace('_', ' ').title()}: Reach Time")

    axes[0, 0].legend(frameon=True, facecolor="white", edgecolor="black", framealpha=1.0)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_one_system(system_name, save_dir, num_inits, verbose):
    success_result = run_final_uniform_position_eval(
        system_name=system_name,
        save_dir=save_dir,
        num_inits=num_inits,
        show_plot=False,
        verbose=verbose,
    )
    chain_time_result = evaluate_chain_hitting_times(
        system_name=system_name,
        save_dir=save_dir,
        num_inits=num_inits,
        show_plot=False,
        verbose=verbose,
        plot_all_times=True,
    )
    bc_time_result = run_bc_hitting_times(
        system_name=system_name,
        save_dir=save_dir,
    )

    result = {
        "system_name": system_name,
        "n_traj_list": np.arange(1, len(success_result["chain_success_rates"]) + 1, dtype=int),
        "chain_success_rates": success_result["chain_success_rates"],
        "bc_success_rates": success_result["bc_success_rates"],
        "chain_mean_time_all": chain_time_result["mean_time_all"],
        "chain_std_time_all": chain_time_result["std_time_all"],
        "bc_mean_time_all": bc_time_result["mean_time_all"],
        "bc_std_time_all": bc_time_result["std_time_all"],
        "success_chain_result_path": success_result["chain_result_path"],
        "success_bc_result_path": success_result["bc_result_path"],
        "time_chain_result_path": chain_time_result["result_path"],
        "time_bc_result_path": bc_time_result["result_path"],
    }

    out_path = Path(save_dir) / f"{system_name}_success_time_combined.png"
    save_system_success_time_figure(system_name, result, out_path)
    result["combined_figure_path"] = str(out_path)

    np.savez(
        Path(save_dir) / f"{system_name}_success_time_combined_results.npz",
        n_traj_list=np.asarray(result["n_traj_list"], dtype=int),
        chain_success_rates=_as_float_array(result["chain_success_rates"]),
        bc_success_rates=_as_float_array(result["bc_success_rates"]),
        chain_mean_time_all=_as_float_array(result["chain_mean_time_all"]),
        chain_std_time_all=_as_float_array(result["chain_std_time_all"]),
        bc_mean_time_all=_as_float_array(result["bc_mean_time_all"]),
        bc_std_time_all=_as_float_array(result["bc_std_time_all"]),
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="data")
    parser.add_argument("--num-inits", type=int, default=500)
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["spring_mass", "single_pendulum"],
        choices=["spring_mass", "single_pendulum"],
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for system_name in args.systems:
        results[system_name] = run_one_system(
            system_name=system_name,
            save_dir=save_dir,
            num_inits=int(args.num_inits),
            verbose=not args.quiet,
        )
        print(
            f"[paper] {system_name}: saved combined figure -> "
            f"{results[system_name]['combined_figure_path']}",
            flush=True,
        )

    overview_path = save_dir / "paper_success_time_combined.png"
    save_all_systems_success_time_figure(results, overview_path)
    print(f"[paper] saved overview figure -> {overview_path}", flush=True)

    for system_name, result in results.items():
        print(f"\n[{system_name}]", flush=True)
        print("Chain success:", np.round(_as_float_array(result["chain_success_rates"]), 6).tolist(), flush=True)
        print("BC success:", np.round(_as_float_array(result["bc_success_rates"]), 6).tolist(), flush=True)
        print("Chain mean time:", np.round(_as_float_array(result["chain_mean_time_all"]), 6).tolist(), flush=True)
        print("BC mean time:", np.round(_as_float_array(result["bc_mean_time_all"]), 6).tolist(), flush=True)


if __name__ == "__main__":
    main()
