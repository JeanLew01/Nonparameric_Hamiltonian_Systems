"""Animation of the single pendulum under the nonparametric chain policy pi_K.

Left: the pendulum (rod coloured by the demonstration whose snippet is being
executed, grey under the default zero input; arrow = applied torque).  Top
right: phase portrait with Supp(K), zero-input energy levels, S_tgt and the
trajectory.  Bottom right: energy H(x(t)) with the target band H(S_tgt) and
the separatrix 2 m g l.

CLI:  python -m symplectic_ncp.experiments.animate --out outputs [--num-demos 5] [--x0 0.0 -28.0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch  # noqa: E402

from symplectic_ncp.chain import DEFAULT, AssignmentSet, NonparametricChainPolicy  # noqa: E402
from symplectic_ncp.config import get_config  # noqa: E402
from symplectic_ncp.simulation import simulate_chain_policy  # noqa: E402

DEMO_COLORS = ["#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
IDLE_COLOR = "#7f7f7f"

# Default initial states: one per ergodic component of the pendulum.
DEFAULT_STATES = {
    "rotation_cw": (0.0, -28.0),
    "rotation_ccw": (2.0, 25.0),
    "libration": (0.6, 4.0),
}


# ------------------------------------------------------------- simulation
def simulate_trace(cfg, K: AssignmentSet, x0, horizon: float | None = None) -> dict:
    """Closed-loop trajectory of pi_K from ``x0`` with its applied inputs and snippet indices."""
    system = cfg.make_system()
    target = cfg.make_target(system)
    policy = NonparametricChainPolicy(K, system, membership_tol=cfg.chain.membership_tol)
    res = simulate_chain_policy(
        system, target, policy, np.asarray(x0, dtype=float)[None], horizon or cfg.horizon, cfg.sim_dt, record=True
    )
    trace = res.trace[0]
    trace["success"] = bool(res.success[0])
    trace["reach_time"] = float(res.reach_time[0])
    trace["demo"] = np.where(trace["snippet"] == DEFAULT, -1, K.demo_ids[np.maximum(trace["snippet"], 0)])
    return trace


def frame_indices(t: np.ndarray, frame_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trace row shown at every frame and the row whose input is applied next."""
    idx = np.clip(np.searchsorted(t, frame_times, side="right") - 1, 0, len(t) - 1)
    return idx, np.minimum(idx + 1, len(t) - 1)


def phase_segments(x: np.ndarray):
    """Split a wrapped (q, p) path where the angle jumps across +-pi."""
    breaks = np.flatnonzero(np.abs(np.diff(x[:, 0])) > np.pi) + 1
    return np.split(x, breaks)


# ------------------------------------------------------------------ panels
class PendulumPanel:
    """Pendulum drawing (q = 0 hanging down, q = pi upright) with a torque arrow and a text box."""

    def __init__(self, ax, system, target, title: str):
        self.ax, self.system = ax, system
        l = self.l = system.l
        ax.set_xlim(-1.35 * l, 1.35 * l)
        ax.set_ylim(-1.35 * l, 1.35 * l)
        ax.set_aspect("equal")
        ax.axis("off")
        for dq in (-target.radius, target.radius):  # angular tolerance of S_tgt around the upright
            q = np.pi + dq
            ax.plot([0, l * np.sin(q)], [0, -l * np.cos(q)], color="#ff9896", lw=1, ls="--", zorder=0)
        ax.add_patch(Circle((0, l), 0.09 * l, fc="none", ec="#d62728", ls="--", lw=1))
        ax.plot(0, 0, "ko", ms=6, zorder=5)
        (self.rod,) = ax.plot([], [], lw=4, solid_capstyle="round", zorder=3)
        self.bob = Circle((0, 0), 0.08 * l, zorder=4)
        ax.add_patch(self.bob)
        self.arrow = FancyArrowPatch((0, 0), (0, 0), mutation_scale=16, lw=2, color="k", zorder=6)
        ax.add_patch(self.arrow)
        self.info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", family="monospace", fontsize=9)
        ax.set_title(title)

    def update(self, q: float, u: float, color: str, text: str) -> list:
        l = self.l
        tip = (l * np.sin(q), -l * np.cos(q))
        self.rod.set_data([0, tip[0]], [0, tip[1]])
        self.rod.set_color(color)
        self.bob.center = tip
        self.bob.set_facecolor(color)
        # torque arrow: tangential at the bob, length proportional to u / u_max
        length = 0.6 * l * u / self.system.u_max[0]
        self.arrow.set_positions(tip, (tip[0] + length * np.cos(q), tip[1] + length * np.sin(q)))
        self.arrow.set_visible(abs(u) > 1e-9)
        self.info.set_text(text)
        return [self.rod, self.bob, self.arrow, self.info]


def draw_phase_background(ax, system, target, K: AssignmentSet, demo_names, E_max: float) -> None:
    """Zero-input energy levels, separatrix, Supp(K) centers coloured by demonstration and S_tgt."""
    p_max = np.sqrt(2 * system.inertia * E_max)
    q_grid, p_grid = np.meshgrid(np.linspace(-np.pi, np.pi, 400), np.linspace(-p_max, p_max, 400))
    E_grid = system.hamiltonian(np.c_[q_grid.ravel(), p_grid.ravel()]).reshape(q_grid.shape)
    levels = [E for E in (5, 20, 60, 100, 140) if E < E_max]
    ax.contour(q_grid, p_grid, E_grid, levels=levels, colors="0.85", linewidths=0.8)
    ax.contour(q_grid, p_grid, E_grid, levels=[system.separatrix_energy], colors="0.55", linewidths=1.0)
    for j in np.unique(K.demo_ids):
        c = K.centers[K.demo_ids == j]
        ax.scatter(c[:, 0], c[:, 1], s=1.5, color=DEMO_COLORS[j % len(DEMO_COLORS)], alpha=0.5,
                   label=f"Supp(K): {demo_names[j]}")
    for qc in (np.pi, -np.pi):
        ax.add_patch(Circle((qc, 0.0), target.radius, fc="none", ec="#d62728", lw=1.2))
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-p_max, p_max)
    ax.set_xlabel("q")
    ax.set_ylabel("p")


def draw_energy_axes(ax, system, target, t_max: float, E_max: float) -> None:
    ax.axhspan(target.H_min, target.H_max, color="#d62728", alpha=0.25, lw=0, label="H(S_tgt)")
    ax.axhline(system.separatrix_energy, color="0.55", lw=1, ls=":", label="separatrix 2mgl")
    ax.set_xlim(0, max(t_max, 1e-3) * 1.02)
    ax.set_ylim(0, E_max)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("H(x)")


def save_animation(fig, update, num_frames: int, path, fps: int) -> Path:
    anim = FuncAnimation(fig, update, frames=num_frames, interval=1000 / fps, blit=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


# --------------------------------------------------------------- animation
def animate(cfg, K: AssignmentSet, demo_names, trace: dict, path: Path, fps: int = 25, speed: float = 1.0,
            hold: float = 1.5, title: str = "", max_seconds: float | None = None) -> Path:
    """Animate ``trace`` until it reaches S_tgt (or for ``max_seconds`` of simulated time if it never does)."""
    system = cfg.make_system()
    target = cfg.make_target(system)
    t, X, U, demo = trace["t"], trace["x"], trace["u"][:, 0], trace["demo"]
    t_end = trace["reach_time"] if trace["success"] else t[-1]
    if max_seconds is not None:
        t_end = min(t_end, max_seconds)
    frame_times = np.concatenate([np.arange(0.0, t_end, speed / fps), np.full(int(hold * fps), t_end)])
    idx, nxt = frame_indices(t, frame_times)
    H = system.hamiltonian(X)
    E_max = max(float(H.max()), cfg.H_bar) * 1.05

    fig = plt.figure(figsize=(12, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.35], height_ratios=[1.4, 1.0], wspace=0.22, hspace=0.35)
    pend = PendulumPanel(fig.add_subplot(gs[:, 0]), system, target, title or "Chain policy on the single pendulum")
    ax_ph = fig.add_subplot(gs[0, 1])
    draw_phase_background(ax_ph, system, target, K, demo_names, E_max)
    ax_ph.legend(loc="upper left", fontsize=7, markerscale=5, ncol=2, framealpha=0.8)
    ax_e = fig.add_subplot(gs[1, 1])
    draw_energy_axes(ax_e, system, target, t_end, E_max)
    ax_e.legend(loc="upper right", fontsize=8)
    path_lines = []
    (cur,) = ax_ph.plot([], [], "o", ms=7, mec="k", zorder=5)
    (e_line,) = ax_e.plot([], [], color="k", lw=1.2)
    (e_cur,) = ax_e.plot([], [], "o", ms=5, color="k")

    def update(f):
        i, j = idx[f], nxt[f]
        q, p = X[i]
        d = demo[j]
        col = IDLE_COLOR if d < 0 else DEMO_COLORS[d % len(DEMO_COLORS)]
        mode = "zero input (u0)" if d < 0 else f"snippet of {demo_names[d]}"
        final = f >= len(frame_times) - int(hold * fps)
        if trace["success"] and frame_times[f] >= trace["reach_time"]:
            note = "\nS_tgt reached"
        elif not trace["success"] and final:
            note = f"\nS_tgt not reached within {cfg.horizon:.0f} s"
        else:
            note = ""
        text = f"t = {frame_times[f]:6.2f} s\nH = {H[i]:7.2f}\nu = {U[j]:+6.2f}\n{mode}"
        artists = pend.update(q, U[j], col, text + note)
        for ln in path_lines:
            ln.remove()
        path_lines.clear()
        for seg in phase_segments(X[: i + 1]):
            path_lines.extend(ax_ph.plot(seg[:, 0], seg[:, 1], color="k", lw=1.0))
        cur.set_data([q], [p])
        cur.set_color(col)
        e_line.set_data(t[: i + 1], H[: i + 1])
        e_cur.set_data([t[i]], [H[i]])
        return [*artists, cur, e_line, e_cur, *path_lines]

    return save_animation(fig, update, len(frame_times), path, fps)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--num-demos", type=int, default=5, help="use K_M built from the first M demonstrations")
    parser.add_argument("--x0", type=float, nargs=2, metavar=("Q", "P"), help="initial state (default: one per component)")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--speed", type=float, default=1.0, help="simulated seconds per real second")
    parser.add_argument("--max-seconds", type=float, default=10.0,
                        help="simulated time shown when the trajectory never reaches S_tgt")
    parser.add_argument("--name", help="output file name (default: single_pendulum_<state>_M<M>.gif)")
    args = parser.parse_args(argv)

    cfg = get_config("single_pendulum")
    sdir = Path(args.out) / cfg.system_name
    K = AssignmentSet.load(sdir / f"assignments_M{args.num_demos}.npz")  # built from the first M demos only
    demo_names = [e.name for e in cfg.experts]
    states = {"custom": tuple(args.x0)} if args.x0 else DEFAULT_STATES
    for name, x0 in states.items():
        trace = simulate_trace(cfg, K, x0)
        path = Path(args.out) / "figures" / (args.name or f"single_pendulum_{name}_M{args.num_demos}.gif")
        title = f"Chain policy, M = {args.num_demos}, x0 = ({x0[0]:.2f}, {x0[1]:.2f})"
        animate(cfg, K, demo_names, trace, path, fps=args.fps, speed=args.speed, title=title,
                max_seconds=None if trace["success"] else args.max_seconds)
        status = f"reached S_tgt at t = {trace['reach_time']:.2f} s" if trace["success"] else "did not reach S_tgt"
        print(f"{name}: {status}; saved {path}")


if __name__ == "__main__":
    main()
