"""Command-line entry point reproducing Section IV (Figs. 2-3).

Example::

    python -m symplectic_ncp.experiments.run --systems spring_mass single_pendulum --out outputs
    python -m symplectic_ncp.experiments.run --quick            # 50 initial states, smoke run
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from symplectic_ncp.config import CONFIGS, ExperimentConfig, get_config

QUICK_NUM_INITS = 50


def build_config(system_name: str, args: argparse.Namespace) -> ExperimentConfig:
    """Paper configuration of ``system_name`` with the CLI overrides applied."""
    cfg = get_config(system_name)
    num_inits = args.num_inits if args.num_inits is not None else (QUICK_NUM_INITS if args.quick else None)
    if num_inits is not None:
        cfg = replace(cfg, num_inits=int(num_inits))
    if args.bc_seeds is not None:
        cfg = replace(cfg, bc=replace(cfg.bc, seeds=tuple(int(s) for s in args.bc_seeds)))
    if args.num_demos is not None:
        cfg = replace(cfg, num_demos=tuple(int(m) for m in args.num_demos))
    return cfg


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the numerical results of Section IV.")
    parser.add_argument("--systems", nargs="+", choices=sorted(CONFIGS), default=["spring_mass", "single_pendulum"])
    parser.add_argument("--out", default="outputs", help="output directory")
    parser.add_argument("--num-inits", type=int, default=None, help="number of test initial states (paper: 500)")
    parser.add_argument("--quick", action="store_true", help=f"smoke run with {QUICK_NUM_INITS} initial states")
    parser.add_argument("--force-experts", action="store_true", help="regenerate the NMPC demonstrations")
    parser.add_argument("--bc-seeds", nargs="+", type=int, default=None, help="BC training seeds (default: config)")
    parser.add_argument("--num-demos", nargs="+", type=int, default=None, help="values of M (default: 1 2 3 4 5)")
    parser.add_argument("--no-plots", action="store_true", help="skip the figures")
    parser.add_argument("--show-paper", action="store_true", help="overlay the paper's numbers on the figures")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> dict:
    from symplectic_ncp.experiments.pipeline import load_results, run_system
    from symplectic_ncp.experiments.plotting import available_systems, make_all_figures, summary_markdown, write_summary

    args = parse_args(argv)
    results = {}
    for name in args.systems:
        results[name] = run_system(build_config(name, args), args.out, args.force_experts, verbose=not args.quiet)
    print(summary_markdown(results))
    # summary.md also keeps the systems of earlier runs into the same directory
    combined = {name: load_results(args.out, name) for name in available_systems(args.out)}
    combined.update(results)
    print(f"summary written to {write_summary(args.out, combined)}")
    if not args.no_plots:
        for path in make_all_figures(args.out, list(results), show_paper=args.show_paper):
            print(f"saved {path}")
    return results


if __name__ == "__main__":
    main()
