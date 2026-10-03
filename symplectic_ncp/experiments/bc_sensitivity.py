"""Sensitivity of the vanilla BC baseline to the training choices the paper leaves unspecified.

The paper fixes the architecture (24, 24, 16), Adam, lr 1.2e-3, weight decay
5e-4 and 40 epochs, but not the mini-batch size, the sampling of the (x, u)
pairs along the demonstrations or the seed.  This script retrains BC for every
combination of

    sample_grid in {control, simulation}  x  batch_size in {16, 64, full}  x  seeds

on the same first-M demonstrations and test initial states as the main
pipeline, and reports success rate and average reach time (mean +- std over
seeds).

CLI:  python -m symplectic_ncp.experiments.bc_sensitivity --systems spring_mass single_pendulum --out outputs
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from symplectic_ncp.config import get_config

FULL_BATCH = 10**9


def _run_one(job: tuple) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from symplectic_ncp.baselines.behavior_cloning import train_behavior_cloning
    from symplectic_ncp.evaluation.initial_states import sample_initial_states
    from symplectic_ncp.experts.demonstration import load_demonstrations
    from symplectic_ncp.simulation import simulate_feedback_policy

    system_name, out_dir, grid, batch, seed, num_inits = job
    cfg = get_config(system_name)
    cfg = dataclasses.replace(cfg, num_inits=num_inits, bc=dataclasses.replace(cfg.bc, sample_grid=grid, batch_size=batch))
    system = cfg.make_system()
    target = cfg.make_target(system)
    demos = load_demonstrations(Path(out_dir) / system_name / "demonstrations.npz")
    x0 = sample_initial_states(system, cfg)
    rows = []
    for M in cfg.num_demos:
        policy = train_behavior_cloning(system, demos[:M], cfg.bc, seed=seed)
        res = simulate_feedback_policy(system, target, policy, x0, cfg.horizon, cfg.sim_dt, cfg.control_period)
        rows.append({"M": M, **res.summary()})
    return {"grid": grid, "batch_size": "full" if batch == FULL_BATCH else batch, "seed": seed, "per_M": rows}


def run(systems, out_dir, seeds, num_inits, workers) -> dict:
    out = {}
    for name in systems:
        jobs = [
            (name, str(out_dir), grid, batch, seed, num_inits)
            for grid, batch, seed in itertools.product(("control", "simulation"), (16, 64, FULL_BATCH), seeds)
        ]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_run_one, jobs))
        table = {}
        for r in results:
            key = f"{r['grid']}/batch={r['batch_size']}"
            table.setdefault(key, []).append(r["per_M"])
        summary = {}
        for key, runs in table.items():
            succ = np.array([[row["success_rate"] for row in per_M] for per_M in runs])
            time = np.array([[row["mean_reach_time"] for row in per_M] for per_M in runs])
            summary[key] = {
                "success_mean": succ.mean(axis=0).tolist(),
                "success_std": succ.std(axis=0).tolist(),
                "time_mean": time.mean(axis=0).tolist(),
                "time_std": time.std(axis=0).tolist(),
            }
        out[name] = {"seeds": list(seeds), "num_inits": num_inits, "runs": results, "summary": summary}
        path = Path(out_dir) / name / "bc_sensitivity.json"
        path.write_text(json.dumps(out[name], indent=2))
        print(f"[{name}] wrote {path}")
        print(format_markdown(name, summary))
    return out


def format_markdown(name: str, summary: dict) -> str:
    lines = [f"### {name}: BC success rate (mean +- std over seeds), M = 1..5", "", "| setting | " + " | ".join(
        f"M={m}" for m in range(1, 6)) + " |", "|---|" + "---|" * 5]
    for key, s in summary.items():
        cells = [f"{m:.3f} +- {d:.3f}" for m, d in zip(s["success_mean"], s["success_std"])]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--systems", nargs="+", default=["spring_mass", "single_pendulum"])
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--num-inits", type=int, default=500)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = parser.parse_args(argv)
    run(args.systems, args.out, tuple(args.seeds), args.num_inits, args.workers)


if __name__ == "__main__":
    main()
