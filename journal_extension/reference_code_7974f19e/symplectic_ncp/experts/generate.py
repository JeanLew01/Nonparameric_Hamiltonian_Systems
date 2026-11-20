"""Offline generation of the expert demonstrations (paper, Sections III-D and IV).

Each demonstration D_j = (x_j, u_j(.), T_j) is a closed-loop rollout of the
NMPC expert on the true dynamics: the plant is the RK4 integrator of the
system at ``cfg.sim_dt``, the expert input is recomputed every
``cfg.control_period`` and held in between (zero-order hold), and the rollout
stops at the first simulation step whose state lies in
S_tgt^delta intersected with H_tgt^eps (``target.in_certified_target``), as
Section III-D requires of every demonstration.

CLI:  python -m symplectic_ncp.experts.generate --systems spring_mass single_pendulum --out outputs
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from symplectic_ncp.config import ExperimentConfig, get_config
from symplectic_ncp.experts.demonstration import Demonstration, save_demonstrations
from symplectic_ncp.experts.nmpc import NMPCExpert
from symplectic_ncp.systems.base import HamiltonianSystem
from symplectic_ncp.target import TargetSet


def rollout_expert(
    system: HamiltonianSystem,
    target: TargetSet,
    expert,
    x0,
    name: str,
    cfg: ExperimentConfig,
) -> Demonstration:
    """Closed-loop rollout of ``expert`` from ``x0`` until the certified target is reached.

    ``expert`` is any callable x -> u with an optional ``reset(x0)`` method.
    States are recorded at every simulation step (angles wrapped, except the
    initial state, which is stored exactly as given).  Raises RuntimeError if
    the certified target is not reached within ``cfg.nmpc.max_duration``.
    """
    x0 = np.asarray(x0, dtype=float).reshape(system.state_dim)
    dt = float(cfg.sim_dt)
    stride = int(round(cfg.control_period / dt))
    if stride < 1 or not np.isclose(stride * dt, cfg.control_period):
        raise ValueError("control_period must be a positive multiple of sim_dt")
    if target.in_certified_target(x0, cfg.energy_eps, cfg.demo_delta)[0]:
        raise ValueError(f"demonstration {name!r} starts inside the certified target")
    max_steps = int(np.ceil(cfg.nmpc.max_duration / dt - 1e-9))

    if hasattr(expert, "reset"):
        expert.reset(x0)
    x = x0.copy()  # plant state (angles left unwrapped during integration)
    states, controls = [x0.copy()], []
    u = np.zeros(system.control_dim)
    for k in range(max_steps):
        if k % stride == 0:
            u = system.clip_control(expert(system.wrap(x)))[0]
        x = system.rk4_step(x, u, dt)[0]
        states.append(system.wrap(x))
        controls.append(u.copy())
        if target.in_certified_target(x, cfg.energy_eps, cfg.demo_delta)[0]:
            return Demonstration(name, np.asarray(states), np.asarray(controls), dt, cfg.control_period)
    raise RuntimeError(
        f"expert demonstration {name!r} did not reach the certified target within "
        f"{cfg.nmpc.max_duration} s (final state {system.wrap(x)})"
    )


def make_expert(
    cfg: ExperimentConfig, system: HamiltonianSystem, target: TargetSet, energy_cap: float | None = None
) -> NMPCExpert:
    return NMPCExpert(system, target, cfg.nmpc, cfg.control_period, energy_cap=energy_cap)


def demonstration_summary(system: HamiltonianSystem, target: TargetSet, demo: Demonstration) -> dict:
    """Duration, input and energy statistics of one demonstration (JSON-serializable)."""
    energy = system.hamiltonian(demo.states)
    final = demo.states[-1]
    return {
        "name": demo.name,
        "x0": demo.states[0].tolist(),
        "duration": demo.duration,
        "num_steps": demo.num_steps,
        "max_abs_u": float(np.max(np.abs(demo.controls))),
        "H0": float(energy[0]),
        "H_min": float(energy.min()),
        "H_max": float(energy.max()),
        "final_state": final.tolist(),
        "final_distance": float(target.distance(final)[0]),
        "final_delta_H": float(target.energy_distance(final)[0]),
    }


def generate_demonstrations(cfg: ExperimentConfig, verbose: bool = True) -> list[Demonstration]:
    """NMPC demonstrations from every ``cfg.experts`` initial state, in that order."""
    system = cfg.make_system()
    target = cfg.make_target(system)
    experts = {}  # one NMPC per energy cap
    demos = []
    for spec in cfg.experts:
        start = time.perf_counter()
        if spec.energy_cap not in experts:
            experts[spec.energy_cap] = make_expert(cfg, system, target, spec.energy_cap)
        demo = rollout_expert(system, target, experts[spec.energy_cap], spec.x0, spec.name, cfg)
        demos.append(demo)
        if verbose:
            s = demonstration_summary(system, target, demo)
            print(
                f"[{cfg.system_name}] {s['name']:>16s}: T={s['duration']:6.3f}s  max|u|={s['max_abs_u']:5.2f}  "
                f"H in [{s['H_min']:.3f}, {s['H_max']:.3f}]  x_T={np.round(s['final_state'], 4)}  "
                f"dH_T={s['final_delta_H']:.2e}  ({time.perf_counter() - start:.1f}s wall)",
                flush=True,
            )
    return demos


def demonstrations_path(out_dir, system_name: str) -> Path:
    return Path(out_dir) / system_name / "demonstrations.npz"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Generate NMPC expert demonstrations.")
    parser.add_argument("--systems", nargs="+", default=["spring_mass", "single_pendulum"])
    parser.add_argument("--out", default="outputs")
    args = parser.parse_args(argv)
    for name in args.systems:
        cfg = get_config(name)
        system = cfg.make_system()
        target = cfg.make_target(system)
        demos = generate_demonstrations(cfg)
        path = demonstrations_path(args.out, name)
        save_demonstrations(path, demos, fingerprint=cfg.expert_fingerprint())
        summary = [demonstration_summary(system, target, d) for d in demos]
        path.with_name("demonstrations_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"saved {len(demos)} demonstrations to {path}")


if __name__ == "__main__":
    main()
