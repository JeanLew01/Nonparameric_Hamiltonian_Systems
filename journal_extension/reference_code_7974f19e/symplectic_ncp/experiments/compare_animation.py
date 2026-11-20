"""Animations comparing vanilla BC, the diffusion policy, the chain policy and PPO on the single pendulum.

Three GIFs of the same closed-loop runs: ``comparison.gif`` (2 x 2 pendulums),
``comparison_phase_orbit.gif`` (2 x 2 phase portraits with each method's
orbit) and ``energy_flow.gif`` (energy versus time of all four methods).

The three imitation methods use the same first M demonstrations; PPO uses none.
The initial state is taken from the Section IV test set: by default a
libration state where pi_{K_M} reaches S_tgt while BC fails for the animated
seed and for at least 80% of its seeds (the one with the median chain reach
time among those states), so the
example is representative rather than a lucky pick.  The diffusion-policy and
PPO models are the first-seed models saved by the pipeline
(``outputs/<system>/models/``), i.e. the ones evaluated in ``results.json``.

CLI:  python -m symplectic_ncp.experiments.compare_animation --out outputs [--num-demos 3] [--index I]
      [--only pendulum phase energy]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from symplectic_ncp.chain import AssignmentSet
from symplectic_ncp.config import get_config
from symplectic_ncp.experiments.animate import PendulumPanel, frame_indices, phase_segments, save_animation, simulate_trace
from symplectic_ncp.experiments.pipeline import ASSIGNMENTS_M_FILE, MODELS_DIR, RESULTS_FILE
from symplectic_ncp.experts.demonstration import load_demonstrations
from symplectic_ncp.simulation import simulate_feedback_policy

METHOD_TITLES = {
    "chain": "Chain Policy",
    "bc": "Vanilla BC (MLP)",
    "dp": "Diffusion Policy",
    "ppo": "PPO",
}
METHOD_COLORS = {"chain": "#1f77b4", "bc": "#ff7f0e", "dp": "#2ca02c", "ppo": "#9467bd"}
CHAIN_IDLE_COLOR = "#9ecae1"


def pick_example(out_dir, cfg, M: int, min_bc_fail: float = 0.8) -> int:
    """Median-reach-time libration test state where the chain policy succeeds and BC fails.

    BC must fail for the animated (first) seed and for at least ``min_bc_fail`` of its seeds.
    """
    system = cfg.make_system()
    r = np.load(Path(out_dir) / cfg.system_name / "rollouts.npz")
    chain_ok, bc = r[f"chain_success_M{M}"], r[f"bc_success_M{M}"]
    bc_bad = (1.0 - bc.mean(axis=0) >= min_bc_fail) & ~bc[0]
    comp = system.ergodic_component(r["initial_states"])
    cand = np.flatnonzero(chain_ok & bc_bad & (comp == 0))
    if cand.size == 0:
        cand = np.flatnonzero(chain_ok & bc_bad)
    if cand.size == 0:
        raise RuntimeError(f"no test state where the chain policy succeeds and BC fails for M = {M}")
    times = r[f"chain_reach_time_M{M}"][cand]
    return int(cand[np.argsort(times)[cand.size // 2]])


def feedback_trace(cfg, policy, x0) -> dict:
    """Closed-loop trajectory of a feedback policy (BC, diffusion policy or PPO) from ``x0``."""
    system = cfg.make_system()
    target = cfg.make_target(system)
    if hasattr(policy, "reset"):
        policy.reset(0)
    res = simulate_feedback_policy(
        system, target, policy, np.asarray(x0)[None], cfg.horizon, cfg.sim_dt, cfg.control_period, record=True
    )
    trace = res.trace[0]
    trace["success"] = bool(res.success[0])
    trace["reach_time"] = float(res.reach_time[0])
    trace["min_distance"] = float(np.min(target.distance(trace["x"])))
    return trace


def load_policies(out_dir, cfg, demos, M: int, seed: int) -> dict:
    """BC is retrained (about 1 s); the diffusion policy and PPO are loaded from the pipeline's models."""
    from symplectic_ncp.baselines.behavior_cloning import train_behavior_cloning
    from symplectic_ncp.baselines.diffusion_policy import DiffusionPolicy
    from symplectic_ncp.baselines.ppo import PPOPolicy

    mdir = Path(out_dir) / cfg.system_name / MODELS_DIR
    system = cfg.make_system()
    return {
        "bc": train_behavior_cloning(system, demos, cfg.bc, seed),
        "dp": DiffusionPolicy.load(mdir / f"dp_M{M}_seed{cfg.dp_seeds[0]}.pt"),
        "ppo": PPOPolicy.load(mdir / f"ppo_seed{cfg.ppo_seeds[0]}.npz"),
    }


def training_seconds(out_dir, cfg, M: int) -> dict:
    """Wall-clock training time of each method (first seed for the learned ones) from results.json."""
    with open(Path(out_dir) / cfg.system_name / RESULTS_FILE) as fh:
        res = json.load(fh)
    entry = next(e for e in res["per_M"] if int(e["M"]) == M)
    return {
        "chain": entry["chain"]["train_seconds"],
        "bc": entry["bc"]["seeds"][0]["timing"]["train_s"],
        "dp": entry["dp"]["seeds"][0]["timing"]["train_s"],
        "ppo": res["ppo"]["seeds"][0]["timing"]["train_s"],
    }


ORDER = ("bc", "dp", "chain", "ppo")  # panel / legend order, as in the result figures


def _frame_runs(cfg, traces: dict, show_seconds: float, fps: int, hold: float):
    """Shared frame clock: every trajectory is shown until it enters S_tgt (then frozen)."""
    system = cfg.make_system()
    reach = [tr["reach_time"] for tr in traces.values() if tr["success"]]
    t_show = max([show_seconds, *reach])
    frame_times = np.concatenate([np.arange(0.0, t_show, 1.0 / fps), np.full(int(hold * fps), t_show)])
    runs = {}
    for name, tr in traces.items():
        t_stop = tr["reach_time"] if tr["success"] else np.inf
        idx, nxt = frame_indices(tr["t"], np.minimum(frame_times, t_stop))
        runs[name] = dict(tr=tr, idx=idx, nxt=nxt, H=system.hamiltonian(tr["x"]), t_stop=t_stop)
    return frame_times, runs, t_show


def _status(cfg, runs, frame_times, name, f, hold_frames) -> str:
    run = runs[name]
    tr = run["tr"]
    if tr["success"] and frame_times[f] >= run["t_stop"]:
        return f"S_tgt reached at t = {tr['reach_time']:.2f} s"
    if not tr["success"] and f >= len(frame_times) - hold_frames:
        return f"not reached within {cfg.horizon:.0f} s (closest distance to x*: {tr['min_distance']:.2f})"
    return ""


def _mode(name, tr, j, demo_names) -> tuple[str, str]:
    """Drawing colour and a short description of the input applied after trace row j."""
    if name == "chain":
        if tr["demo"][j] >= 0:
            return METHOD_COLORS["chain"], f"snippet of {demo_names[tr['demo'][j]]}"
        return CHAIN_IDLE_COLOR, "zero input (u0)"
    return METHOD_COLORS[name], "learned feedback"


def _panel_title(name, M, train_s) -> str:
    data = "no demos" if name == "ppo" else f"M = {M} demos"
    return f"{METHOD_TITLES[name]}  ({data}, training {train_s[name]:.1f} s)"


def _suptitle(x0, M, demo_names) -> str:
    return (f"Single pendulum, x0 = ({x0[0]:.2f}, {x0[1]:.2f}): BC, diffusion policy and chain policy use the "
            f"same {M} demonstrations\n({', '.join(demo_names[:M])}); PPO learns from 10M environment steps "
            "without demonstrations")


def compare(cfg, traces: dict, train_s: dict, demo_names, x0, M: int, path, show_seconds: float = 12.0,
            fps: int = 25, hold: float = 2.0) -> Path:
    """2 x 2 pendulum animation (BC, diffusion policy, chain policy, PPO)."""
    system = cfg.make_system()
    target = cfg.make_target(system)
    frame_times, runs, _ = _frame_runs(cfg, traces, show_seconds, fps, hold)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 10.0))
    fig.suptitle(_suptitle(x0, M, demo_names), fontsize=10)
    panels = {}
    for ax, name in zip(axes.ravel(), ORDER):
        panels[name] = PendulumPanel(ax, system, target, _panel_title(name, M, train_s))
        ax.title.set_fontsize(10)
        ax.title.set_color(METHOD_COLORS[name])

    def update(f):
        artists = []
        for name in ORDER:
            run = runs[name]
            tr, i, j = run["tr"], run["idx"][f], run["nxt"][f]
            col, mode = _mode(name, tr, j, demo_names)
            t_now = min(frame_times[f], run["t_stop"])
            status = _status(cfg, runs, frame_times, name, f, int(hold * fps)).replace(" (", "\n(")
            text = f"t = {t_now:6.2f} s\nH = {run['H'][i]:7.2f}\nu = {tr['u'][j, 0]:+6.2f}\n{mode}\n{status}"
            artists += panels[name].update(tr["x"][i, 0], tr["u"][j, 0], col, text)
        return artists

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save_animation(fig, update, len(frame_times), path, fps)


def phase_orbit(cfg, traces: dict, train_s: dict, demos, x0, M: int, path, show_seconds: float = 12.0,
                fps: int = 25, hold: float = 2.0) -> Path:
    """2 x 2 phase-portrait animation: each method's closed-loop orbit (q, p) over zero-input energy levels.

    Grey curves are the M demonstrations the imitation methods learn from (none for PPO).
    """
    system = cfg.make_system()
    target = cfg.make_target(system)
    frame_times, runs, _ = _frame_runs(cfg, traces, show_seconds, fps, hold)
    E_max = max(cfg.H_bar, max(float(r["H"][: r["idx"][-1] + 1].max()) for r in runs.values())) * 1.05
    p_max = np.sqrt(2 * system.inertia * E_max)
    q_grid, p_grid = np.meshgrid(np.linspace(-np.pi, np.pi, 400), np.linspace(-p_max, p_max, 400))
    E_grid = system.hamiltonian(np.c_[q_grid.ravel(), p_grid.ravel()]).reshape(q_grid.shape)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.6), sharex=True, sharey=True)
    fig.suptitle(_suptitle(x0, M, demo_names=[d.name for d in demos]), fontsize=10)
    markers, lines, texts = {}, {name: [] for name in ORDER}, {}
    for ax, name in zip(axes.ravel(), ORDER):
        ax.contour(q_grid, p_grid, E_grid, levels=[E for E in (5, 20, 60, 100, 140) if E < E_max],
                   colors="0.88", linewidths=0.7)
        ax.contour(q_grid, p_grid, E_grid, levels=[system.separatrix_energy], colors="0.55", linewidths=1.0)
        if name != "ppo":
            for k, d in enumerate(demos[:M]):
                for seg in phase_segments(system.wrap(d.states)):
                    ax.plot(seg[:, 0], seg[:, 1], color="0.6", lw=2.0, alpha=0.6,
                            label="demonstrations" if k == 0 and seg is not None else None)
        for qc in (np.pi, -np.pi):
            ax.add_patch(Circle((qc, 0.0), target.radius, fc="#d62728", ec="#d62728", alpha=0.6))
        ax.plot([x0[0]], [x0[1]], "k*", ms=10, zorder=6)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-p_max, p_max)
        ax.set_title(_panel_title(name, M, train_s), fontsize=10, color=METHOD_COLORS[name])
        (markers[name],) = ax.plot([], [], "o", ms=8, mec="k", color=METHOD_COLORS[name], zorder=7)
        texts[name] = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=8, family="monospace",
                              bbox={"fc": "white", "ec": "0.8", "alpha": 0.85})
    for ax in axes[1]:
        ax.set_xlabel("q [rad]")
    for ax in axes[:, 0]:
        ax.set_ylabel("p")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles[:1], labels[:1], loc="lower left", fontsize=8)

    def update(f):
        artists = []
        for name in ORDER:
            run = runs[name]
            tr, i, j = run["tr"], run["idx"][f], run["nxt"][f]
            ax = axes.ravel()[ORDER.index(name)]
            for ln in lines[name]:
                ln.remove()
            lines[name] = []
            for seg in phase_segments(tr["x"][: i + 1]):
                lines[name].extend(ax.plot(seg[:, 0], seg[:, 1], color=METHOD_COLORS[name], lw=1.3, zorder=5))
            markers[name].set_data([tr["x"][i, 0]], [tr["x"][i, 1]])
            t_now = min(frame_times[f], run["t_stop"])
            status = _status(cfg, runs, frame_times, name, f, int(hold * fps)).replace(" (", "\n(")
            texts[name].set_text(f"t = {t_now:5.2f} s   H = {run['H'][i]:6.2f}" + (f"\n{status}" if status else ""))
            artists += [markers[name], texts[name], *lines[name]]
        return artists

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save_animation(fig, update, len(frame_times), path, fps)


def energy_flow(cfg, traces: dict, train_s: dict, demo_names, x0, M: int, path, show_seconds: float = 12.0,
                fps: int = 25, hold: float = 2.0) -> Path:
    """Energy H(x(t)) of the four closed loops on one axis, with the target band and the separatrix."""
    system = cfg.make_system()
    target = cfg.make_target(system)
    frame_times, runs, t_show = _frame_runs(cfg, traces, show_seconds, fps, hold)
    E_max = max(float(r["H"][: r["idx"][-1] + 1].max()) for r in runs.values()) * 1.15

    fig, (ax, side) = plt.subplots(1, 2, figsize=(12.0, 5.2), gridspec_kw={"width_ratios": [2.6, 1.0]})
    fig.suptitle(_suptitle(x0, M, demo_names), fontsize=10)
    side.axis("off")
    ax.axhspan(target.H_min, target.H_max, color="#d62728", alpha=0.35, lw=0, label="target energy band H(S_tgt)")
    ax.axhline(system.separatrix_energy, color="0.5", lw=1, ls=":", label="separatrix 2mgl")
    ax.set_xlim(0.0, t_show * 1.01)
    ax.set_ylim(0.0, E_max)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("energy H(x(t))")
    ax.grid(alpha=0.3)
    curves, dots = {}, {}
    for name in ORDER:
        (curves[name],) = ax.plot([], [], color=METHOD_COLORS[name], lw=1.8, label=_panel_title(name, M, train_s))
        (dots[name],) = ax.plot([], [], "o", ms=6, color=METHOD_COLORS[name])
    handles, labels = ax.get_legend_handles_labels()
    side.legend(handles, [lb.replace("  (", "\n(") for lb in labels], loc="upper left", fontsize=8, frameon=False)
    clock = side.text(0.0, 0.32, "", transform=side.transAxes, va="top", family="monospace", fontsize=8)

    def update(f):
        lines = []
        for name in ORDER:
            run = runs[name]
            tr, i = run["tr"], run["idx"][f]
            curves[name].set_data(tr["t"][: i + 1], run["H"][: i + 1])
            dots[name].set_data([tr["t"][i]], [run["H"][i]])
            status = _status(cfg, runs, frame_times, name, f, int(hold * fps)).replace(" (", "\n  (")
            if status:
                lines.append(f"{METHOD_TITLES[name]}:\n  {status}")
        clock.set_text(f"t = {frame_times[f]:5.2f} s" + "".join(f"\n{s}" for s in lines))
        return [*curves.values(), *dots.values(), clock]

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return save_animation(fig, update, len(frame_times), path, fps)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--num-demos", type=int, default=3)
    parser.add_argument("--index", type=int, help="index of the test initial state (default: automatic pick)")
    parser.add_argument("--seconds", type=float, default=12.0, help="simulated time shown")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--only", nargs="+", choices=["pendulum", "phase", "energy"],
                        default=["pendulum", "phase", "energy"], help="animations to render")
    args = parser.parse_args(argv)

    cfg = get_config("single_pendulum")
    sdir = Path(args.out) / cfg.system_name
    M = args.num_demos
    index = pick_example(args.out, cfg, M) if args.index is None else args.index
    x0 = np.load(sdir / "rollouts.npz")["initial_states"][index]
    all_demos = load_demonstrations(sdir / "demonstrations.npz")
    demo_names = [d.name for d in all_demos]
    K = AssignmentSet.load(sdir / ASSIGNMENTS_M_FILE.format(M))
    policies = load_policies(args.out, cfg, all_demos[:M], M, cfg.bc.seeds[0])

    traces = {"chain": simulate_trace(cfg, K, x0)}
    traces.update({name: feedback_trace(cfg, policy, x0) for name, policy in policies.items()})
    train_s = training_seconds(args.out, cfg, M)
    out = Path(args.out)
    kw = dict(show_seconds=args.seconds, fps=args.fps)
    paths = []
    if "pendulum" in args.only:
        paths.append(compare(cfg, traces, train_s, demo_names, x0, M, out / "comparison.gif", **kw))
    if "phase" in args.only:
        paths.append(phase_orbit(cfg, traces, train_s, all_demos, x0, M, out / "comparison_phase_orbit.gif", **kw))
    if "energy" in args.only:
        paths.append(energy_flow(cfg, traces, train_s, demo_names, x0, M, out / "energy_flow.gif", **kw))
    for name in ORDER:
        tr = traces[name]
        print(f"{METHOD_TITLES[name]:>18s}: " + (f"reached S_tgt at t = {tr['reach_time']:.2f} s" if tr["success"]
                                              else f"did not reach S_tgt within {cfg.horizon:.0f} s"))
    print(f"test state #{index}, x0 = {np.round(x0, 3).tolist()}; saved " + ", ".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
