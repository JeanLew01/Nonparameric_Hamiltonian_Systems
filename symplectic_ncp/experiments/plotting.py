"""Figures and tables of the reproduction.

* :func:`plot_results` -- the layout of the paper's Figs. 2-3: (a) success rate,
  (b) average reach time (unsuccessful runs count as the horizon) with one
  standard deviation error bars, x = number of trajectories; bars from left to
  right: Vanilla BC, Diffusion Policy, Chain Policy (PPO, which uses no
  demonstrations, is reported in the tables only).
* :func:`plot_assignment_set` -- phase portrait with the demonstrations, the
  support balls B_{r_i}(x_i) of K drawn to scale in data coordinates, the
  target S_tgt and a few zero-input energy levels; the second panel zooms on
  the target.
* :func:`summary_markdown` -- paper-vs-reproduction table (``summary.md``).

Usage: ``python -m symplectic_ncp.experiments.plotting --out outputs``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from symplectic_ncp.experiments.paper_reference import METHODS as PAPER_METHODS
from symplectic_ncp.experiments.paper_reference import reference_value
from symplectic_ncp.experiments.pipeline import ASSIGNMENTS_FILE, DEMOS_FILE, RESULTS_FILE, load_results, system_dir

METHODS = ("chain", "bc", "dp", "ppo")  # tables
FIGURE_METHODS = ("bc", "dp", "chain")  # bar order in the figures (left to right); PPO is reported in tables only
METHOD_LABELS = {"chain": "Chain Policy", "bc": "Vanilla BC", "dp": "Diffusion Policy", "ppo": "PPO (no demos)"}
COLORS = {"chain": "#1f77b4", "bc": "#ff7f0e", "dp": "#2ca02c", "ppo": "#9467bd"}
SYSTEM_TITLES = {"spring_mass": "Spring-mass", "single_pendulum": "Single pendulum"}
FIGURE_FORMATS = ("png", "pdf")


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, fig_dir: Path, stem: str) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in FIGURE_FORMATS:
        path = fig_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        paths.append(path)
    return paths


# --------------------------------------------------------------- data access
def method_summary(entry: dict, method: str, results: dict | None = None) -> dict:
    """Summary used in figures/tables: the chain rollout, or a learned method pooled over its seeds.

    PPO uses no demonstrations, so its (single) summary is repeated for every M.
    """
    if method == "chain":
        return entry.get("chain") or {}
    if method == "ppo":
        return ((results or {}).get("ppo") or {}).get("pooled") or {}
    return (entry.get(method) or {}).get("pooled") or {}


def available_methods(results: dict) -> list[str]:
    present = [m for m in METHODS if m != "ppo" and any(e.get(m) for e in results["per_M"])]
    return present + (["ppo"] if results.get("ppo") else [])


def series(results: dict, method: str, quantity: str) -> tuple[np.ndarray, np.ndarray]:
    """(M values, quantity) over ``results['per_M']``; missing values are NaN."""
    Ms = np.asarray([e["M"] for e in results["per_M"]], dtype=int)
    vals = [method_summary(e, method, results).get(quantity) for e in results["per_M"]]
    return Ms, np.asarray([np.nan if v is None else v for v in vals], dtype=float)


def training_times(results: dict, method: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(M values, mean, std) of the wall-clock training time; PPO is repeated for every M."""
    Ms = np.asarray([e["M"] for e in results["per_M"]], dtype=int)
    if method == "chain":
        mean = [(e.get("chain") or {}).get("train_seconds") for e in results["per_M"]]
        return Ms, np.asarray([np.nan if v is None else v for v in mean], dtype=float), np.zeros(len(Ms))
    blocks = [results.get("ppo")] * len(Ms) if method == "ppo" else [e.get(method) for e in results["per_M"]]
    mean = [b["mean_over_seeds"]["train_seconds"] if b else np.nan for b in blocks]
    std = [b["mean_over_seeds"]["train_seconds_std"] if b else np.nan for b in blocks]
    return Ms, np.asarray(mean, dtype=float), np.asarray(std, dtype=float)


# ---------------------------------------------------------- Figs. 2-3 layout
def plot_results(results: dict, fig_dir, show_paper: bool = False) -> list[Path]:
    """Two-panel figure of Section IV; optionally overlays the paper's numbers as black markers."""
    plt = _pyplot()
    name = results["system"]
    methods = [m for m in FIGURE_METHODS if m in available_methods(results)]
    width = 0.8 / len(methods)
    with plt.rc_context({"font.family": "serif", "axes.axisbelow": True}):
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
        panels = [("success_rate", None, "Success Rate", "(a) Success rate"),
                  ("mean_reach_time", "std_reach_time", "Average Reach Time", "(b) Average reach time")]
        for ax, (quantity, err_q, ylabel, caption) in zip(axes, panels):
            for k, method in enumerate(methods):
                Ms, vals = series(results, method, quantity)
                err = series(results, method, err_q)[1] if err_q else None
                offset = (k - (len(methods) - 1) / 2) * width
                ax.bar(Ms + offset, vals, width, color=COLORS[method], label=METHOD_LABELS[method],
                       yerr=err, capsize=2, error_kw={"elinewidth": 0.8, "ecolor": "black"})
                if show_paper and method in PAPER_METHODS:
                    ref = [reference_value(name, method, quantity, int(M)) for M in Ms]
                    pts = [(M + offset, r[0]) for M, r in zip(Ms, ref) if r is not None]
                    if pts:
                        x, y = zip(*pts)
                        ax.plot(x, y, "k_", ms=12, mew=2, label="Paper" if k == 1 else None)
            ax.set_xticks(Ms)
            ax.set_xlabel("Number of Trajectories")
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.6)
            ax.set_title(caption, y=-0.42, fontsize=10)
        axes[0].set_ylim(0.0, 1.05)
        axes[1].set_ylim(bottom=0.0)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(methods), bbox_to_anchor=(0.5, 1.0),
                   frameon=False, fontsize=9)
        fig.suptitle(f"{SYSTEM_TITLES.get(name, name)} results", fontsize=11, y=1.08)
        fig.tight_layout()
        paths = _save(fig, Path(fig_dir), f"{name}_results")
        plt.close(fig)
    return paths


def plot_training_time(results_by_system: dict[str, dict], fig_dir) -> list[Path]:
    """Wall-clock training time per method and M (log scale) of the demonstration-based methods."""
    plt = _pyplot()
    names = list(results_by_system)
    with plt.rc_context({"font.family": "serif", "axes.axisbelow": True}):
        fig, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 3.2), squeeze=False)
        for ax, name in zip(axes[0], names):
            res = results_by_system[name]
            methods = [m for m in FIGURE_METHODS if m in available_methods(res)]
            width = 0.8 / max(len(methods), 1)
            for k, method in enumerate(methods):
                Ms, mean, std = training_times(res, method)
                offset = (k - (len(methods) - 1) / 2) * width
                ax.bar(Ms + offset, mean, width, color=COLORS[method], label=METHOD_LABELS[method],
                       yerr=std if np.any(std > 0) else None, capsize=2,
                       error_kw={"elinewidth": 0.8, "ecolor": "black"})
            ax.set_yscale("log")
            ax.set_xticks(Ms)
            ax.set_xlabel("Number of Trajectories")
            ax.set_ylabel("Training time [s]")
            ax.set_title(SYSTEM_TITLES.get(name, name), fontsize=10)
            ax.grid(axis="y", which="both", alpha=0.4)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.06),
                   frameon=False, fontsize=9)
        fig.tight_layout()
        paths = _save(fig, Path(fig_dir), "training_time")
        plt.close(fig)
    return paths


# ------------------------------------------------------ assignment-set figure
def _chart(system, X, origin: np.ndarray) -> np.ndarray:
    """Coordinates of X in the chart centred at ``origin``: angles mapped to [origin - pi, origin + pi)."""
    return origin + system.difference(X, origin)


def _break_wraps(states: np.ndarray, angle_indices) -> np.ndarray:
    """Insert NaN rows where a charted angle jumps so lines are not drawn across the plot."""
    if not angle_indices or len(states) < 2:
        return states
    jumps = np.any(np.abs(np.diff(states[:, list(angle_indices)], axis=0)) > np.pi, axis=1)
    return np.insert(states, np.nonzero(jumps)[0] + 1, np.nan, axis=0)


def _periodic_copies(center: np.ndarray, radius: float, angle_indices, origin: np.ndarray) -> list[np.ndarray]:
    """``center`` (already charted) plus its 2*pi shifts whose ball meets the chart window."""
    copies = [center]
    for idx in angle_indices:
        for c in list(copies):
            for shift in (-2.0 * np.pi, 2.0 * np.pi):
                if abs(c[idx] + shift - origin[idx]) <= np.pi + radius:
                    shifted = c.copy()
                    shifted[idx] += shift
                    copies.append(shifted)
    return copies


def _draw_phase_portrait(ax, plt, system, results, demos, K, H_levels, origin, half_widths):
    """Energy levels, demonstrations, balls B_{r_i}(x_i) and S_tgt in the chart centred at ``origin``."""
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Circle

    angles = system.angle_indices
    lims = [(o - w, o + w) for o, w in zip(origin, half_widths)]
    qq, pp = np.meshgrid(np.linspace(*lims[0], 400), np.linspace(*lims[1], 400))
    HH = system.hamiltonian(np.stack([qq.ravel(), pp.ravel()], axis=1)).reshape(qq.shape)
    ax.contour(qq, pp, HH, levels=H_levels, colors="0.6", linewidths=0.7, linestyles="-")

    cmap = plt.get_cmap("tab10")
    for j, d in enumerate(demos):
        color = cmap((j + 2) % 10)
        ax.plot(*_break_wraps(_chart(system, d.states, origin), angles).T, color=color, lw=0.9,
                label=f"demo {j + 1}: {d.name}")
        mask = K.demo_ids == j
        centers = _chart(system, K.centers[mask], origin)
        patches = [Circle(c, r) for c0, r in zip(centers, K.radii[mask])
                   for c in _periodic_copies(c0, r, angles, origin)]
        ax.add_collection(PatchCollection(patches, facecolor=color, edgecolor=color, alpha=0.35, lw=0.5))
        ax.plot(*centers.T, ".", color=color, ms=1.5)

    tgt = results["target"]
    tc = _chart(system, np.asarray(tgt["center"], dtype=float)[None, :], origin)[0]
    for c in _periodic_copies(tc, tgt["radius"], angles, origin):
        ax.add_patch(Circle(c, tgt["radius"], fill=False, edgecolor="red", lw=1.2, ls="--"))
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])
    ax.set_xlabel("q")
    ax.set_ylabel("p")


def plot_assignment_set(out_dir, system_name: str, fig_dir=None) -> list[Path]:
    """Phase portrait of the demonstrations and Supp(K) (balls to scale) for one system."""
    from symplectic_ncp.chain import AssignmentSet
    from symplectic_ncp.experts import load_demonstrations
    from symplectic_ncp.systems import make_system

    plt = _pyplot()
    sdir = system_dir(out_dir, system_name)
    results = load_results(out_dir, system_name)
    demos = load_demonstrations(sdir / DEMOS_FILE)
    K = AssignmentSet.load(sdir / ASSIGNMENTS_FILE)
    system = make_system(system_name)

    H_bar = float(results["config"]["H_bar"])
    H_X = float(results["lipschitz"]["H_X"])
    levels = list(H_bar * np.array([0.05, 0.25, 0.5, 0.75, 1.0]))
    if hasattr(system, "separatrix_energy") and system.separatrix_energy < H_X:
        levels.append(system.separatrix_energy)
    levels = sorted(set(levels))
    low, high = system.bounding_box(H_X)
    full_half = 0.55 * (high - low)
    full_origin = 0.5 * (high + low)
    for idx in system.angle_indices:
        full_origin[idx], full_half[idx] = 0.0, np.pi

    center = np.asarray(results["target"]["center"], dtype=float)
    zoom_half = np.full(system.state_dim, 4.0 * float(results["target"]["radius"]))
    zoom_levels = sorted(set([results["target"]["H_min"], results["target"]["H_max"]] + levels))

    with plt.rc_context({"font.family": "serif"}):
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), gridspec_kw={"width_ratios": [1.6, 1.0]})
        _draw_phase_portrait(axes[0], plt, system, results, demos, K, levels, full_origin, full_half)
        _draw_phase_portrait(axes[1], plt, system, results, demos, K, zoom_levels, center, zoom_half)
        if not system.angle_indices:
            axes[0].set_aspect("equal")
        axes[1].set_aspect("equal", adjustable="box")
        axes[0].set_title(f"(a) Demonstrations and Supp(K), N = {len(K)}", fontsize=10)
        axes[1].set_title("(b) Zoom on S_tgt (dashed)", fontsize=10)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), fontsize=8,
                   bbox_to_anchor=(0.5, -0.06), frameon=False)
        fig.suptitle(f"{SYSTEM_TITLES.get(system_name, system_name)}: assignment set (balls to scale)", fontsize=11)
        fig.tight_layout()
        paths = _save(fig, Path(fig_dir) if fig_dir else Path(out_dir) / "figures", f"{system_name}_assignment_set")
        plt.close(fig)
    return paths


# --------------------------------------------------------------------- tables
def _fmt(value, digits: int = 3) -> str:
    return "-" if value is None or not np.isfinite(value) else f"{value:.{digits}g}"


def _ref(system: str, method: str, quantity: str, M: int, digits: int = 3) -> str:
    ref = reference_value(system, method, quantity, M)
    if ref is None:
        return "-"
    value, exact = ref
    return ("" if exact else "~") + _fmt(value, digits)


def summary_markdown(results_by_system: dict[str, dict]) -> str:
    """Markdown tables of every method, the Theorem 2 diagnostics and the training times."""
    lines = ["# Results (Section IV + ablations)", ""]
    lines.append("Success rate / average reach time in seconds (unsuccessful runs count as the horizon; mean over "
                 "the test states, learned methods pooled over their seeds).  Paper values in parentheses "
                 "(`~` = read off the paper's figure).  PPO uses no demonstrations, so its numbers do not depend "
                 "on M.  Theorem 2 diagnostics for K_M: C1 viol. = sampled violation fraction of the local energy "
                 "decrease (Condition 1); C2 cov. = covered fraction of {E : Delta H(E) <= c} (Condition 2, "
                 "target band included); C3 cov. = smallest per-ergodic-component coverage (Condition 3).")
    for name, res in results_by_system.items():
        cfg = res.get("config", {})
        methods = available_methods(res)
        lines += [
            "",
            f"## {SYSTEM_TITLES.get(name, name)}",
            "",
            f"{res['num_initial_states']} initial states with H <= {cfg.get('H_bar')}, horizon {cfg.get('horizon')} s; "
            f"seeds: BC {cfg.get('bc', {}).get('seeds')}, DP {cfg.get('dp_seeds')}, PPO {cfg.get('ppo_seeds')}.",
            "",
            "| M | N(K_M) | " + " | ".join(METHOD_LABELS[m] for m in methods) + " | C1 viol. | C2 cov. | C3 cov. |",
            "|---|---|" + "---|" * len(methods) + "---|---|---|",
        ]
        for e in res["per_M"]:
            M = int(e["M"])
            chain = e.get("chain") or {}
            row = [str(M), str(chain.get("num_assignments", "-"))]
            for method in methods:
                sm = method_summary(e, method, res)
                cell = f"{_fmt(sm.get('success_rate'))} / {_fmt(sm.get('mean_reach_time'))}"
                if method in PAPER_METHODS:
                    cell += f" ({_ref(name, method, 'success_rate', M)} / {_ref(name, method, 'mean_reach_time', M)})"
                row.append(cell)
            theory = chain.get("theory") or {}
            cond1 = (theory.get("condition1") or {}).get("empirical") or {}
            cond2 = theory.get("condition2") or {}
            cond3 = {k: v for k, v in (theory.get("condition3") or {}).items()
                     if isinstance(v, dict) and v.get("covered_fraction") is not None}
            worst = min(cond3, key=lambda k: cond3[k]["covered_fraction"]) if cond3 else None
            c3 = f"{_fmt(cond3[worst]['covered_fraction'], 4)} ({worst})" if worst else "-"
            row += [_fmt(cond1.get("violation_fraction")), _fmt(cond2.get("covered_fraction_including_band"), 4), c3]
            lines.append("| " + " | ".join(row) + " |")
    lines += ["", training_time_markdown(results_by_system)]
    return "\n".join(lines) + "\n"


def training_time_markdown(results_by_system: dict[str, dict]) -> str:
    """Training-time table (seconds, mean +- std over seeds; chain policy: Algorithm 1, deterministic)."""
    lines = ["## Training time [s]", "",
             "Wall-clock time of the training call on the same machine (chain policy: Lipschitz constants + "
             "Algorithm 1 on the first M demonstrations, CPU, one process per demonstration; BC: CPU; "
             "diffusion policy and PPO: as reported in the device column).  Demonstration generation (shared "
             "by chain, BC and DP) is excluded; PPO uses no demonstrations.", ""]
    for name, res in results_by_system.items():
        methods = available_methods(res)
        Ms = [int(e["M"]) for e in res["per_M"]]
        lines += [f"**{SYSTEM_TITLES.get(name, name)}**", "",
                  "| Method | device | " + " | ".join(f"M={M}" for M in Ms) + " |",
                  "|---|---|" + "---|" * len(Ms)]
        for method in methods:
            _, mean, std = training_times(res, method)
            if method == "chain":
                device = "cpu"
            elif method == "ppo":
                device = res["ppo"]["seeds"][0].get("device", "-")
            else:
                device = next(((e.get(method) or {}).get("seeds", [{}])[0].get("device", "-")
                               for e in res["per_M"] if e.get(method)), "-")
            cells = [_fmt(m, 3) if sd == 0 or not np.isfinite(sd) else f"{_fmt(m, 3)} ± {_fmt(sd, 2)}"
                     for m, sd in zip(mean, std)]
            if method == "ppo":
                cells = [cells[0]] + ["(same)"] * (len(cells) - 1)
            lines.append(f"| {METHOD_LABELS[method]} | {device} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def write_summary(out_dir, results_by_system: dict[str, dict]) -> Path:
    path = Path(out_dir) / "summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary_markdown(results_by_system))
    return path


# ------------------------------------------------------------------------ CLI
def available_systems(out_dir) -> list[str]:
    return sorted(p.parent.name for p in Path(out_dir).glob(f"*/{RESULTS_FILE}"))


def make_all_figures(out_dir, systems=None, show_paper: bool = False, assignment_figure: bool = True) -> list[Path]:
    fig_dir = Path(out_dir) / "figures"
    paths = []
    for name in systems or available_systems(out_dir):
        paths += plot_results(load_results(out_dir, name), fig_dir, show_paper=show_paper)
        if assignment_figure:
            paths += plot_assignment_set(out_dir, name, fig_dir)
    names = available_systems(out_dir)
    if names:
        paths += plot_training_time({n: load_results(out_dir, n) for n in names}, fig_dir)
    return paths


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Plot the Section IV figures from saved results.")
    parser.add_argument("--out", default="outputs", help="output directory used by run.py")
    parser.add_argument("--systems", nargs="+", default=None, help="systems to plot (default: all with results)")
    parser.add_argument("--show-paper", action="store_true", help="overlay the paper's numbers as black markers")
    parser.add_argument("--no-assignment-figure", action="store_true", help="skip the phase-portrait figure")
    args = parser.parse_args(argv)
    systems = args.systems or available_systems(args.out)
    for path in make_all_figures(args.out, systems, args.show_paper, not args.no_assignment_figure):
        print(f"saved {path}")
    print(f"saved {write_summary(args.out, {s: load_results(args.out, s) for s in systems})}")


if __name__ == "__main__":
    main()
