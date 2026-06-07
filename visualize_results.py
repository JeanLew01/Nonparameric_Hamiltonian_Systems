from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


METHOD_COLORS = {
    "chain": "#4C78A8",
    "bc": "#F58518",
}

SYSTEM_LABELS = {
    "spring_mass": "Spring-Mass",
    "single_pendulum": "Single Pendulum",
    "double_pendulum": "Double Pendulum",
}

FINAL_RESULT_FILES = [
    "final_results_manifest.json",
    "spring_mass_success_time_combined_results.npz",
    "single_pendulum_success_time_combined_results.npz",
    "incremental_tube_success_rates_spring_mass_final.npz",
    "incremental_vanilla_bc_success_rates_spring_mass_final.npz",
    "incremental_chain_hitting_times_spring_mass_final.npz",
    "incremental_vanilla_bc_hitting_times_spring_mass_final.npz",
    "incremental_tube_success_rates_single_pendulum_final.npz",
    "incremental_vanilla_bc_success_rates_single_pendulum_final.npz",
    "incremental_chain_hitting_times_single_pendulum_final.npz",
    "incremental_vanilla_bc_hitting_times_single_pendulum_final.npz",
    "incremental_tube_success_rates_double_pendulum_final.npz",
    "incremental_vanilla_bc_success_rates_double_pendulum_final.npz",
    "incremental_chain_hitting_times_double_pendulum_final.npz",
    "incremental_vanilla_bc_hitting_times_double_pendulum_final.npz",
]


def _array(values):
    return np.asarray(values, dtype=float).reshape(-1)


def _load_npz(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required result file: {path}")
    return np.load(path, allow_pickle=True)


def _configure_matplotlib() -> None:
    preferred_fonts = [
        "Times New Roman",
        "Times",
        "Nimbus Roman",
        "TeX Gyre Termes",
        "STIX Two Text",
        "DejaVu Serif",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    chosen_font = next((font for font in preferred_fonts if font in available_fonts), "DejaVu Serif")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [chosen_font],
            "mathtext.fontset": "stix",
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 15,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )


def _load_combined_result(data_dir: Path, system_name: str):
    path = data_dir / f"{system_name}_success_time_combined_results.npz"
    data = _load_npz(path)
    return {
        "n_traj_list": _array(data["n_traj_list"]),
        "chain_success_rates": _array(data["chain_success_rates"]),
        "bc_success_rates": _array(data["bc_success_rates"]),
        "chain_mean_time_all": _array(data["chain_mean_time_all"]),
        "chain_std_time_all": _array(data["chain_std_time_all"]),
        "bc_mean_time_all": _array(data["bc_mean_time_all"]),
        "bc_std_time_all": _array(data["bc_std_time_all"]),
    }


def _load_double_result(data_dir: Path):
    chain_success = _load_npz(data_dir / "incremental_tube_success_rates_double_pendulum_final.npz")
    bc_success = _load_npz(data_dir / "incremental_vanilla_bc_success_rates_double_pendulum_final.npz")
    chain_time = _load_npz(data_dir / "incremental_chain_hitting_times_double_pendulum_final.npz")
    bc_time = _load_npz(data_dir / "incremental_vanilla_bc_hitting_times_double_pendulum_final.npz")
    return {
        "n_traj_list": _array(chain_success["n_traj_list"]),
        "chain_success_rates": _array(chain_success["success_rates"]),
        "bc_success_rates": _array(bc_success["success_rates"]),
        "chain_mean_time_all": _array(chain_time["mean_time_all"]),
        "chain_std_time_all": _array(chain_time["std_time_all"]),
        "bc_mean_time_all": _array(bc_time["mean_time_all"]),
        "bc_std_time_all": _array(bc_time["std_time_all"]),
    }


def load_results(data_dir: str | Path = "data"):
    data_dir = Path(data_dir)
    return {
        "spring_mass": _load_combined_result(data_dir, "spring_mass"),
        "single_pendulum": _load_combined_result(data_dir, "single_pendulum"),
        "double_pendulum": _load_double_result(data_dir),
    }


def _bar_pair(ax, x, chain_values, bc_values, ylabel, ylim=None, chain_err=None, bc_err=None):
    width = 0.36
    ax.bar(
        x - width / 2.0,
        chain_values,
        width,
        yerr=chain_err,
        capsize=4 if chain_err is not None else 0,
        color=METHOD_COLORS["chain"],
        edgecolor="none",
        ecolor="black",
        label="Chain Policy",
    )
    ax.bar(
        x + width / 2.0,
        bc_values,
        width,
        yerr=bc_err,
        capsize=4 if bc_err is not None else 0,
        color=METHOD_COLORS["bc"],
        edgecolor="none",
        ecolor="black",
        label="Vanilla BC",
    )
    ax.set_xticks(x)
    ax.set_xlabel("Number of Expert Trajectories")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", color="black", linestyle="--", alpha=0.45)


def save_system_figure(system_name: str, result: dict[str, np.ndarray], output_dir: Path):
    n_traj = result["n_traj_list"].astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    _bar_pair(
        axes[0],
        n_traj,
        result["chain_success_rates"],
        result["bc_success_rates"],
        ylabel="Success Rate",
        ylim=(-0.05, 1.05),
    )
    _bar_pair(
        axes[1],
        n_traj,
        result["chain_mean_time_all"],
        result["bc_mean_time_all"],
        ylabel="Average Time to Target (s)",
        chain_err=result["chain_std_time_all"],
        bc_err=result["bc_std_time_all"],
    )
    axes[0].legend(frameon=True, facecolor="white", edgecolor="black", framealpha=1.0)
    fig.suptitle(SYSTEM_LABELS.get(system_name, system_name.replace("_", " ").title()))
    fig.tight_layout()
    out_path = output_dir / f"{system_name}_success_time_combined.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_overview_figure(results: dict[str, dict[str, np.ndarray]], output_dir: Path):
    system_names = list(results.keys())
    fig, axes = plt.subplots(len(system_names), 2, figsize=(13, 5.2 * len(system_names)))
    if len(system_names) == 1:
        axes = np.asarray([axes])

    for row, system_name in enumerate(system_names):
        result = results[system_name]
        n_traj = result["n_traj_list"].astype(int)
        _bar_pair(
            axes[row, 0],
            n_traj,
            result["chain_success_rates"],
            result["bc_success_rates"],
            ylabel="Success Rate",
            ylim=(-0.05, 1.05),
        )
        _bar_pair(
            axes[row, 1],
            n_traj,
            result["chain_mean_time_all"],
            result["bc_mean_time_all"],
            ylabel="Average Time to Target (s)",
            chain_err=result["chain_std_time_all"],
            bc_err=result["bc_std_time_all"],
        )
        label = SYSTEM_LABELS.get(system_name, system_name.replace("_", " ").title())
        axes[row, 0].set_title(f"{label}: Success")
        axes[row, 1].set_title(f"{label}: Reach Time")

    axes[0, 0].legend(frameon=True, facecolor="white", edgecolor="black", framealpha=1.0)
    fig.tight_layout()
    out_path = output_dir / "all_systems_success_time_combined.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def copy_final_result_files(data_dir: Path, output_dir: Path):
    copied = []
    for filename in FINAL_RESULT_FILES:
        src = data_dir / filename
        if not src.exists():
            continue
        dst = output_dir / filename
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def write_summary(results: dict[str, dict[str, np.ndarray]], output_dir: Path):
    payload = {}
    for system_name, result in results.items():
        payload[system_name] = {
            key: _array(value).tolist()
            for key, value in result.items()
        }
    out_path = output_dir / "visualized_results_summary.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def visualize_results(data_dir: str | Path = "data", output_dir: str | Path = "results"):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()

    results = load_results(data_dir)
    copied_files = copy_final_result_files(data_dir, output_dir)
    figure_paths = [save_system_figure(name, result, output_dir) for name, result in results.items()]
    figure_paths.append(save_overview_figure(results, output_dir))
    summary_path = write_summary(results, output_dir)

    return {
        "output_dir": output_dir,
        "copied_files": copied_files,
        "figure_paths": figure_paths,
        "summary_path": summary_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    artifacts = visualize_results(data_dir=args.data_dir, output_dir=args.output_dir)
    print(f"[visualize] saved results to {artifacts['output_dir']}", flush=True)
    print(f"[visualize] copied {len(artifacts['copied_files'])} result files", flush=True)
    for path in artifacts["figure_paths"]:
        print(f"[visualize] figure -> {path}", flush=True)
    print(f"[visualize] summary -> {artifacts['summary_path']}", flush=True)


if __name__ == "__main__":
    main()
