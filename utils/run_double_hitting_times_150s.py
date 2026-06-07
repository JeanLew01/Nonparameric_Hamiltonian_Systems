from __future__ import annotations

from pathlib import Path

from utils.final_uniform_eval import evaluate_chain_hitting_times
from utils.run_bc_hitting_times import run_bc_hitting_times


def main():
    save_dir = Path("data")

    chain_result = evaluate_chain_hitting_times(
        system_name="double_pendulum",
        save_dir=save_dir,
        num_inits=500,
        show_plot=False,
        verbose=True,
        episode_seconds_override=150.0,
        result_filename_override="incremental_chain_hitting_times_double_pendulum_150s.npz",
        figure_filename_override="double_pendulum_chain_avg_time_150s.png",
        plot_all_times=True,
    )

    bc_result = run_bc_hitting_times(
        system_name="double_pendulum",
        save_dir=save_dir,
        episode_seconds_override=150.0,
        result_filename_override="incremental_vanilla_bc_hitting_times_double_pendulum_150s.npz",
        figure_filename_override="double_pendulum_bc_avg_time_150s.png",
    )

    print("CHAIN_RESULT", chain_result["result_path"], flush=True)
    print("CHAIN_FIGURE", chain_result["figure_path"], flush=True)
    print("CHAIN_MEAN_ALL", chain_result["mean_time_all"], flush=True)
    print("CHAIN_STD_ALL", chain_result["std_time_all"], flush=True)
    print("BC_RESULT", bc_result["result_path"], flush=True)
    print("BC_FIGURE", bc_result["figure_path"], flush=True)
    print("BC_MEAN_ALL", bc_result["mean_time_all"], flush=True)
    print("BC_STD_ALL", bc_result["std_time_all"], flush=True)


if __name__ == "__main__":
    main()
