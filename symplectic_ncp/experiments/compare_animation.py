"""Side-by-side animation: chain policy vs vanilla BC trained on the same M demonstrations.

The initial state is taken from the Section IV test set: by default a
libration state where pi_{K_M} reaches S_tgt but BC fails for every training
seed (the one with the median chain reach time among those states), so the
example is representative rather than a lucky pick.

CLI:  python -m symplectic_ncp.experiments.compare_animation --out outputs [--num-demos 3] [--index I]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from symplectic_ncp.baselines.behavior_cloning import train_behavior_cloning
from symplectic_ncp.chain import AssignmentSet
from symplectic_ncp.config import get_config
from symplectic_ncp.experiments.animate import (
    PendulumPanel,
    draw_energy_axes,
    draw_phase_background,
    frame_indices,
    phase_segments,
    save_animation,
    simulate_trace,
)
from symplectic_ncp.experts.demonstration import load_demonstrations
from symplectic_ncp.simulation import simulate_feedback_policy

CHAIN_COLOR, CHAIN_IDLE_COLOR, BC_COLOR = "#1f77b4", "#9ecae1", "#ff7f0e"


def pick_example(out_dir, cfg, M: int) -> int:
    """Median-reach-time libration test state where the chain policy succeeds and BC fails for all seeds."""
    system = cfg.make_system()
    r = np.load(Path(out_dir) / cfg.system_name / "rollouts.npz")
    chain_ok, bc_ok = r[f"chain_success_M{M}"], r[f"bc_success_M{M}"].any(axis=0)
    comp = system.ergodic_component(r["initial_states"])
    cand = np.flatnonzero(chain_ok & ~bc_ok & (comp == 0))
    if cand.size == 0:
        cand = np.flatnonzero(chain_ok & ~bc_ok)
    if cand.size == 0:
        raise RuntimeError(f"no test state where the chain policy succeeds and BC fails for M = {M}")
    times = r[f"chain_reach_time_M{M}"][cand]
    return int(cand[np.argsort(times)[cand.size // 2]])


def bc_trace(cfg, demos, x0, seed: int) -> dict:
    system = cfg.make_system()
    target = cfg.make_target(system)
    policy = train_behavior_cloning(system, demos, cfg.bc, seed)
    res = simulate_feedback_policy(
        system, target, policy, np.asarray(x0)[None], cfg.horizon, cfg.sim_dt, cfg.control_period, record=True
    )
    trace = res.trace[0]
    trace["success"] = bool(res.success[0])
    trace["reach_time"] = float(res.reach_time[0])
    trace["min_distance"] = float(np.min(target.distance(trace["x"])))
    return trace


def compare(cfg, K: AssignmentSet, demo_names, x0, chain: dict, bc: dict, M: int, seed: int, path,
            show_seconds: float = 12.0, fps: int = 25, hold: float = 2.0) -> Path:
    system = cfg.make_system()
    target = cfg.make_target(system)
    t_show = max(show_seconds, chain["reach_time"] if chain["success"] else 0.0)
    frame_times = np.concatenate([np.arange(0.0, t_show, 1.0 / fps), np.full(int(hold * fps), t_show)])
    runs = {}
    for name, tr in (("chain", chain), ("bc", bc)):
        t_stop = tr["reach_time"] if tr["success"] else np.inf
        idx, nxt = frame_indices(tr["t"], np.minimum(frame_times, t_stop))
        runs[name] = dict(tr=tr, idx=idx, nxt=nxt, H=system.hamiltonian(tr["x"]), t_stop=t_stop)
    E_max = max(float(runs["chain"]["H"].max()), float(runs["bc"]["H"][runs["bc"]["tr"]["t"] <= t_show].max())) * 1.15

    fig = plt.figure(figsize=(14, 6.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.3], height_ratios=[1.3, 1.0], wspace=0.15, hspace=0.38)
    fig.suptitle(
        f"Same {M} expert demonstrations ({', '.join(demo_names[:M])}),  x0 = ({x0[0]:.2f}, {x0[1]:.2f})",
        fontsize=12,
    )
    panels = {
        "chain": PendulumPanel(fig.add_subplot(gs[:, 0]), system, target, "Chain policy (Algorithm 1 + NCP)"),
        "bc": PendulumPanel(fig.add_subplot(gs[:, 1]), system, target, f"Vanilla BC (MLP 24-24-16, seed {seed})"),
    }
    ax_ph = fig.add_subplot(gs[0, 2])
    draw_phase_background(ax_ph, system, target, K, demo_names, E_max)
    ax_ph.set_title("phase portrait; dots = Supp(K) along the demonstrations", fontsize=9)
    ax_ph.legend(loc="lower left", fontsize=6, markerscale=5, framealpha=0.85)
    ax_e = fig.add_subplot(gs[1, 2])
    draw_energy_axes(ax_e, system, target, t_show, E_max)
    colors = {"chain": CHAIN_COLOR, "bc": BC_COLOR}
    markers, e_lines, path_lines = {}, {}, []
    for name, label in (("chain", "chain policy"), ("bc", "vanilla BC")):
        (markers[name],) = ax_ph.plot([], [], "o", ms=7, mec="k", color=colors[name], zorder=6)
        (e_lines[name],) = ax_e.plot([], [], color=colors[name], lw=1.4, label=label)
    ax_e.legend(loc="lower right", fontsize=7, ncol=2)

    def status(name, f):
        run = runs[name]
        tr = run["tr"]
        if tr["success"] and frame_times[f] >= run["t_stop"]:
            return f"S_tgt reached at t = {tr['reach_time']:.2f} s"
        if not tr["success"] and f >= len(frame_times) - int(hold * fps):
            return f"not reached within {cfg.horizon:.0f} s\n(closest distance to x*: {tr['min_distance']:.2f})"
        return ""

    def update(f):
        artists = []
        for ln in path_lines:
            ln.remove()
        path_lines.clear()
        for name, run in runs.items():
            tr, i, j = run["tr"], run["idx"][f], run["nxt"][f]
            q, p = tr["x"][i]
            u = tr["u"][j, 0]
            if name == "chain":
                executing = tr["demo"][j] >= 0
                col = CHAIN_COLOR if executing else CHAIN_IDLE_COLOR
                mode = f"snippet of {demo_names[tr['demo'][j]]}" if executing else "zero input (u0)"
            else:
                col, mode = BC_COLOR, "u = MLP(x)"
            t_now = min(frame_times[f], run["t_stop"])
            text = f"t = {t_now:6.2f} s\nH = {run['H'][i]:7.2f}\nu = {u:+6.2f}\n{mode}\n{status(name, f)}"
            artists += panels[name].update(q, u, col, text)
            for seg in phase_segments(tr["x"][: i + 1]):
                path_lines.extend(ax_ph.plot(seg[:, 0], seg[:, 1], color=colors[name], lw=1.1, zorder=5))
            markers[name].set_data([q], [p])
            e_lines[name].set_data(tr["t"][: i + 1], run["H"][: i + 1])
        return [*artists, *markers.values(), *e_lines.values(), *path_lines]

    return save_animation(fig, update, len(frame_times), path, fps)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--num-demos", type=int, default=3)
    parser.add_argument("--index", type=int, help="index of the test initial state (default: automatic pick)")
    parser.add_argument("--seed", type=int, help="BC training seed (default: first configured seed)")
    parser.add_argument("--seconds", type=float, default=12.0, help="simulated time shown")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--name", default="comparison.gif")
    args = parser.parse_args(argv)

    cfg = get_config("single_pendulum")
    sdir = Path(args.out) / cfg.system_name
    M = args.num_demos
    index = pick_example(args.out, cfg, M) if args.index is None else args.index
    x0 = np.load(sdir / "rollouts.npz")["initial_states"][index]
    seed = cfg.bc.seeds[0] if args.seed is None else args.seed
    K = AssignmentSet.load(sdir / "assignments.npz").from_demos(range(M))
    demos = load_demonstrations(sdir / "demonstrations.npz")[:M]
    demo_names = [d.name for d in load_demonstrations(sdir / "demonstrations.npz")]

    chain = simulate_trace(cfg, K, x0)
    bc = bc_trace(cfg, demos, x0, seed)
    path = compare(cfg, K, demo_names, x0, chain, bc, M, seed, Path(args.out) / args.name,
                   show_seconds=args.seconds, fps=args.fps)
    for name, tr in (("chain policy", chain), ("vanilla BC", bc)):
        print(f"{name}: " + (f"reached S_tgt at t = {tr['reach_time']:.2f} s" if tr["success"]
                             else f"did not reach S_tgt within {cfg.horizon:.0f} s"))
    print(f"test state #{index}, x0 = {np.round(x0, 3).tolist()}; saved {path}")


if __name__ == "__main__":
    main()
