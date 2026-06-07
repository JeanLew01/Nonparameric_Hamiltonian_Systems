from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


EXPECTED_RESULTS = {
    "spring_mass": {
        "file": "spring_mass_success_time_combined_results.npz",
        "values": {
            "n_traj_list": [1, 2, 3, 4, 5],
            "chain_success_rates": [0.962, 0.962, 1.0, 1.0, 1.0],
            "bc_success_rates": [0.096, 0.992, 1.0, 1.0, 1.0],
            "chain_mean_time_all": [5.18752, 3.5588, 2.82336, 2.53688, 2.51764],
            "chain_std_time_all": [
                3.480238303564858,
                3.429089465149604,
                1.196544988874217,
                1.0974887086435103,
                1.0675557270700204,
            ],
            "bc_mean_time_all": [18.188, 4.78128, 2.09288, 3.1872, 1.33412],
            "bc_std_time_all": [
                5.561364580748147,
                2.17169020847818,
                0.7221493651593138,
                0.9859595123533217,
                0.3063720378885776,
            ],
        },
    },
    "single_pendulum": {
        "file": "single_pendulum_success_time_combined_results.npz",
        "values": {
            "n_traj_list": [1, 2, 3, 4, 5],
            "chain_success_rates": [0.352, 0.692, 1.0, 1.0, 1.0],
            "bc_success_rates": [0.056, 0.418, 1.0, 1.0, 1.0],
            "chain_mean_time_all": [103.70216, 58.16152, 16.98016, 16.98016, 16.974],
            "chain_std_time_all": [
                62.9430810314716,
                61.415553581235436,
                4.385877879558436,
                4.385877879558436,
                4.403037951233217,
            ],
            "bc_mean_time_all": [142.27168, 91.48392, 17.62408, 14.08044, 22.95212],
            "bc_std_time_all": [
                31.782807773662793,
                69.0798279256224,
                3.857878504255934,
                3.3229375868950655,
                7.793181090260895,
            ],
        },
    },
    "double_pendulum_success": {
        "file": "incremental_tube_success_rates_double_pendulum_final.npz",
        "values": {
            "n_traj_list": [1, 2, 3, 4, 5],
            "success_rates": [0.044, 0.034, 0.036, 0.038, 0.036],
        },
    },
    "double_pendulum_bc_success": {
        "file": "incremental_vanilla_bc_success_rates_double_pendulum_final.npz",
        "values": {
            "n_traj_list": [1, 2, 3, 4, 5],
            "success_rates": [0.024, 0.04, 0.046, 0.152, 0.258],
        },
    },
    "double_pendulum_chain_time": {
        "file": "incremental_chain_hitting_times_double_pendulum_final.npz",
        "values": {
            "n_traj_list": [1, 2, 3, 4, 5],
            "success_rates": [0.044, 0.034, 0.036, 0.038, 0.036],
            "mean_time_all": [19.36504, 19.46756, 19.48752, 19.46756, 19.4612],
            "std_time_all": [
                3.225845160326205,
                3.031386423140409,
                2.9761566238355135,
                3.027082761736124,
                3.0390959445203434,
            ],
        },
    },
    "double_pendulum_bc_time": {
        "file": "incremental_vanilla_bc_hitting_times_double_pendulum_final.npz",
        "values": {
            "n_traj_list": [1, 2, 3, 4, 5],
            "success_rates": [0.024, 0.04, 0.046, 0.152, 0.034],
            "mean_time_all": [19.57844, 19.41328, 19.2488, 18.13544, 19.4032],
            "std_time_all": [
                2.768430596276526,
                3.119738649566659,
                3.554907053637268,
                4.866587470332779,
                3.242216180330978,
            ],
        },
    },
}


FINAL_FIGURES: list[str] = []


def _as_array(values):
    return np.asarray(values, dtype=float).reshape(-1)


def _check_array(actual, expected, atol):
    actual_arr = _as_array(actual)
    expected_arr = _as_array(expected)
    return actual_arr.shape == expected_arr.shape and np.allclose(actual_arr, expected_arr, atol=atol, rtol=0.0)


def load_expected_summary(save_dir: str | Path = "data"):
    save_dir = Path(save_dir)
    summary = {}
    for name, record in EXPECTED_RESULTS.items():
        result_path = save_dir / record["file"]
        values = {}
        if result_path.exists():
            loaded = np.load(result_path, allow_pickle=True)
            for key in record["values"]:
                values[key] = _as_array(loaded[key]).tolist()
        summary[name] = {
            "file": str(result_path),
            "values": values,
        }
    return summary


def verify_final_results(save_dir: str | Path = "data", atol: float = 1e-8, check_figures: bool = True):
    save_dir = Path(save_dir)
    failures = []

    for name, record in EXPECTED_RESULTS.items():
        result_path = save_dir / record["file"]
        if not result_path.exists():
            failures.append(f"{name}: missing {result_path}")
            continue

        loaded = np.load(result_path, allow_pickle=True)
        for key, expected in record["values"].items():
            if key not in loaded.files:
                failures.append(f"{name}: missing key {key} in {result_path}")
                continue
            if not _check_array(loaded[key], expected, atol=atol):
                failures.append(
                    f"{name}: {key} changed. "
                    f"expected={expected}, actual={_as_array(loaded[key]).tolist()}"
                )

    if check_figures:
        for figure_name in FINAL_FIGURES:
            figure_path = save_dir / figure_name
            if not figure_path.exists():
                failures.append(f"missing figure {figure_path}")

    return failures


def write_manifest(save_dir: str | Path = "data", output_path: str | Path | None = None):
    save_dir = Path(save_dir)
    output_path = save_dir / "final_results_manifest.json" if output_path is None else Path(output_path)
    manifest = {
        "description": "Final saved numerical results. Visualization files are intentionally not tracked.",
        "data_dir": str(save_dir),
        "results": load_expected_summary(save_dir),
        "figures": [str(save_dir / name) for name in FINAL_FIGURES],
    }
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="data")
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--no-figure-check", action="store_true")
    args = parser.parse_args()

    failures = verify_final_results(
        save_dir=args.save_dir,
        atol=float(args.atol),
        check_figures=not bool(args.no_figure_check),
    )
    if failures:
        print("[verify] final results changed:", flush=True)
        for failure in failures:
            print(f"  - {failure}", flush=True)
        raise SystemExit(1)

    print("[verify] final results unchanged.", flush=True)
    if args.write_manifest:
        manifest_path = write_manifest(save_dir=args.save_dir)
        print(f"[verify] wrote manifest -> {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
