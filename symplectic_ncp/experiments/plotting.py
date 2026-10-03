"""Figures and tables of the reproduction.

* :func:`plot_results` -- the layout of the paper's Figs. 2-3: (a) success rate,
  (b) average reach time (unsuccessful runs count as the horizon) with one
  standard deviation error bars, x = number of trajectories, Chain Policy
  (#1f77b4) vs Vanilla BC (#ff7f0e).
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

from symplectic_ncp.experiments.paper_reference import METHOD_LABELS, METHODS, reference_value
from symplectic_ncp.experiments.pipeline import ASSIGNMENTS_FILE, DEMOS_FILE, RESULTS_FILE, load_results, system_dir

COLORS = {"chain": "#1f77b4", "bc": "#ff7f0e"}
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
def method_summary(entry: dict, method: str) -> dict:
    """Summary used in figures/tables: chain rollout, or BC pooled over its seeds."""
    return entry["chain"] if method == "chain" else entry["bc"]["pooled"]


def series(results: dict, method: str, quantity: str) -> tuple[np.ndarray, np.ndarray]:
    """(M values, quantity) over ``results['per_M']``; missing values are NaN."""
    Ms = np.asarray([e["M"] for e in results["per_M"]], dtype=int)
    vals = [method_summary(e, method).get(quantity) for e in results["per_M"]]
    return Ms, np.asarray([np.nan if v is None else v for v in vals], dtype=float)


# ---------------------------------------------------------- Figs. 2-3 layout
def plot_results(results: dict, fig_dir, show_paper: bool = False) -> list[Path]:
    """Two-panel figure of Section IV; optionally overlays the paper's numbers as black markers."""
    plt = _pyplot()
    name = results["system"]
    width = 0.35
    with plt.rc_context({"font.family": "serif", "axes.axisbelow": True}):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
        panels = [("success_rate", None, "Success Rate", "(a) Success rate"),
                  ("mean_reach_time", "std_reach_time", "Average Reach Time", "(b) Average reach time")]
        for ax, (quantity, err_q, ylabel, caption) in zip(axes, panels):
            for k, method in enumerate(METHODS):
                Ms, vals = series(results, method, quantity)
                err = series(results, method, err_q)[1] if err_q else None
                offset = (k - 0.5) * width
                ax.bar(Ms + offset, vals, width, color=COLORS[method], label=METHOD_LABELS[method],
                       yerr=err, capsize=3, error_kw={"elinewidth": 0.9, "ecolor": "black"})
                if show_paper:
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
        axes[1].legend(loc="upper right")
        fig.suptitle(f"{SYSTEM_TITLES.get(name, name)} results", fontsize=11)
        fig.tight_layout()
        paths = _save(fig, Path(fig_dir), f"{name}_results")
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
    """Markdown tables: reproduction vs paper (``~`` marks values read off the paper's figures)."""
    lines = ["# Reproduction vs paper (Section IV)", ""]
    lines.append("Reach times in seconds, unsuccessful runs count as the horizon; mean +- std over the "
                 "test states (BC: pooled over seeds).  `~` = read off the paper's figure (approximate).  "
                 "Theorem 2 diagnostics for K_M: C1 viol. = sampled violation fraction of the local energy "
                 "decrease (Condition 1); C2 cov. = covered fraction of {E : Delta H(E) <= c} (Condition 2, "
                 "target band included); C3 cov. = smallest per-ergodic-component coverage (Condition 3), "
                 "with the component that attains it.")
    for name, res in results_by_system.items():
        lip = res.get("lipschitz", {})
        cfg = res.get("config", {})
        lines += [
            "",
            f"## {SYSTEM_TITLES.get(name, name)}",
            "",
            f"{res['num_initial_states']} initial states with H <= {cfg.get('H_bar')}, horizon {cfg.get('horizon')} s, "
            f"BC seeds {cfg.get('bc', {}).get('seeds')}; L_H = {_fmt(lip.get('L_H'), 4)}, L = {_fmt(lip.get('L'), 4)} "
            f"on H <= {_fmt(lip.get('H_X'), 4)}; |K| = {res['assignment_set']['N']}.",
            "",
            "| M | N(K_M) | Chain success | (paper) | Chain time | (paper) | BC success | (paper) | BC time | (paper) "
            "| C1 viol. | C2 cov. | C3 cov. |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for e in res["per_M"]:
            M = int(e["M"])
            row = [str(M), str(e["chain"].get("num_assignments", "-"))]
            theory = e["chain"].get("theory") or {}
            cond1 = (theory.get("condition1") or {}).get("empirical") or {}
            cond2 = theory.get("condition2") or {}
            cond3 = {k: v for k, v in (theory.get("condition3") or {}).items()
                     if isinstance(v, dict) and v.get("covered_fraction") is not None}
            worst = min(cond3, key=lambda k: cond3[k]["covered_fraction"]) if cond3 else None
            c3 = f"{_fmt(cond3[worst]['covered_fraction'], 4)} ({worst})" if worst else "-"
            checks = [_fmt(cond1.get("violation_fraction")),
                      _fmt(cond2.get("covered_fraction_including_band"), 4), c3]
            for method in METHODS:
                s = method_summary(e, method)
                row += [
                    _fmt(s.get("success_rate")),
                    _ref(name, method, "success_rate", M),
                    f"{_fmt(s.get('mean_reach_time'))} +- {_fmt(s.get('std_reach_time'))}",
                    f"{_ref(name, method, 'mean_reach_time', M)} +- {_ref(name, method, 'std_reach_time', M)}",
                ]
            lines.append("| " + " | ".join(row + checks) + " |")
    return "\n".join(lines) + "\n"


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
