"""Experiment configurations reproducing Section IV of the paper.

Every parameter stated in the paper is set to the paper's value; parameters
the paper leaves unspecified are marked ``# not stated in the paper``.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field

import numpy as np

from symplectic_ncp.systems import HamiltonianSystem, make_system
from symplectic_ncp.target import TargetSet


@dataclass(frozen=True)
class ExpertSpec:
    """Initial state x_j of an expert demonstration."""

    name: str
    x0: tuple[float, ...]
    energy_cap: float | None = None  # optional soft NMPC constraint H(x) <= energy_cap


@dataclass
class NMPCConfig:
    """Nonlinear MPC expert (CasADi + do-mpc).  Weights are not stated in the paper."""

    horizon: int = 100  # not stated in the paper
    Q: tuple[float, ...] = (10.0, 1.0)  # stage weight on the state error, not stated in the paper
    Q_terminal: tuple[float, ...] = (50.0, 5.0)  # not stated in the paper
    R: float = 1e-3  # input weight, not stated in the paper
    max_duration: float = 30.0  # demonstration time limit [s], not stated in the paper


@dataclass
class ChainConfig:
    """Algorithm 1 and the nonparametric chain policy."""

    v0: float = 1e-3  # guaranteed decrease rate, "a small v0" in the paper
    lipschitz: str = "global"  # "global": L_H, L of Assumptions 1-2 on X; "local": max along the snippet
    membership_tol: float = 1e-6  # numerical slack on rho_K(x) <= 1
    min_anchor_advance: float = 1e-4  # floor [s] on sigma_i in Algorithm 1 (prevents Zeno anchor sequences)
    energy_margin: float = 0.05  # X = {H <= (1 + margin) max(H_bar, demo energies)} for L_H, L
    workers: int = 0  # Algorithm 1 processes in parallel over demonstrations (0 = one per demo)


@dataclass
class BCConfig:
    """Vanilla behavior cloning baseline (Section IV)."""

    hidden_sizes: tuple[int, ...] = (24, 24, 16)
    lr: float = 1.2e-3
    weight_decay: float = 5e-4
    epochs: int = 40
    batch_size: int = 64  # not stated in the paper
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)  # not stated in the paper (it reports one run)
    angle_features: bool = True  # encode angles as (sin, cos); not stated in the paper
    sample_grid: str = "control"  # (x, u) pairs at expert updates ("control") or every sim step; not stated


@dataclass
class ExperimentConfig:
    system_name: str
    target_center: tuple[float, ...]
    experts: tuple[ExpertSpec, ...]
    horizon: float  # simulation horizon [s] (20 s spring-mass, 150 s pendulum)
    H_bar: float  # test initial states are uniform on {H(x) <= H_bar}; value not stated in the paper
    target_radius: float = 0.1  # S_tgt = {||x - x*|| <= 0.1}
    energy_eps: float = 1e-3  # epsilon of H_tgt^eps, not stated in the paper
    demo_delta: float = 0.1  # demonstrations end in S_tgt^delta, delta <= target radius
    num_inits: int = 500
    init_seed: int = 2026
    control_period: float = 0.02  # ZOH period of NMPC and BC, not stated in the paper
    sim_dt: float = 0.005  # RK4 step; also the snippet sampling step, not stated in the paper
    num_demos: tuple[int, ...] = (1, 2, 3, 4, 5)
    u_bound: float = 20.0
    nmpc: NMPCConfig = field(default_factory=NMPCConfig)
    chain: ChainConfig = field(default_factory=ChainConfig)
    bc: BCConfig = field(default_factory=BCConfig)
    dp_seeds: tuple[int, ...] = (0, 1, 2)  # diffusion-policy ablation (symplectic_ncp.baselines.diffusion_policy)
    ppo_seeds: tuple[int, ...] = (0, 1, 2)  # PPO ablation (symplectic_ncp.baselines.ppo)

    def expert_fingerprint(self) -> str:
        """JSON string of every setting that determines the expert demonstrations (cache key)."""
        return json.dumps(
            {
                "system": self.system_name,
                "experts": [dataclasses.asdict(e) for e in self.experts],
                "nmpc": dataclasses.asdict(self.nmpc),
                "target": [list(self.target_center), self.target_radius],
                "energy_eps": self.energy_eps,
                "demo_delta": self.demo_delta,
                "control_period": self.control_period,
                "sim_dt": self.sim_dt,
                "u_bound": self.u_bound,
            },
            sort_keys=True,
            default=float,
        )

    def make_system(self) -> HamiltonianSystem:
        return make_system(self.system_name, u_min=-self.u_bound, u_max=self.u_bound)

    def make_target(self, system: HamiltonianSystem | None = None) -> TargetSet:
        system = self.make_system() if system is None else system
        return TargetSet(system, np.asarray(self.target_center, dtype=float), self.target_radius)


def _pendulum_rotation_state(theta: float, energy: float, direction: int) -> tuple[float, float]:
    pend = make_system("single_pendulum")
    potential = pend.mgl * (1.0 - np.cos(theta))
    return float(theta), float(direction * np.sqrt(2.0 * pend.inertia * (energy - potential)))


def spring_mass_config() -> ExperimentConfig:
    return ExperimentConfig(
        system_name="spring_mass",
        target_center=(0.0, 0.0),
        horizon=20.0,
        H_bar=2.0,
        experts=(
            ExpertSpec("q2_p0", (2.0, 0.0)),
            ExpertSpec("qm2_p0", (-2.0, 0.0)),
            ExpertSpec("q0_pm2", (0.0, -2.0)),
            ExpertSpec("q0_p2", (0.0, 2.0)),
            ExpertSpec("q1.5_pm1", (1.5, -1.0)),
        ),
        nmpc=NMPCConfig(horizon=50, Q=(10.0, 1.0), Q_terminal=(100.0, 10.0), R=1e-3, max_duration=10.0),
    )


def single_pendulum_config() -> ExperimentConfig:
    H_bar = 160.0
    separatrix = make_system("single_pendulum").separatrix_energy
    return ExperimentConfig(
        system_name="single_pendulum",
        target_center=(np.pi, 0.0),
        horizon=150.0,
        H_bar=H_bar,
        # Ordered so that the first three demonstrations visit the three
        # ergodic components of the pendulum one at a time: clockwise
        # rotation, counter-clockwise rotation, libration; the last two add
        # libration coverage.  Libration demonstrations are kept below the
        # separatrix (energy cap 2 m g l) so that each demonstration lies in a
        # single component.
        experts=(
            ExpertSpec("rotation_cw", _pendulum_rotation_state(0.5 * np.pi, H_bar, -1)),
            ExpertSpec("rotation_ccw", _pendulum_rotation_state(-0.5 * np.pi, H_bar, +1)),
            ExpertSpec("libration_rest", (0.0, 0.0), energy_cap=separatrix),
            ExpertSpec("libration_right", (3.0, 0.0), energy_cap=separatrix),
            ExpertSpec("libration_left", (-3.0, 0.0), energy_cap=separatrix),
        ),
        nmpc=NMPCConfig(horizon=100, Q=(10.0, 0.1), Q_terminal=(100.0, 1.0), R=1e-3, max_duration=40.0),
    )


CONFIGS = {
    "spring_mass": spring_mass_config,
    "single_pendulum": single_pendulum_config,
}


def get_config(system_name: str) -> ExperimentConfig:
    try:
        return CONFIGS[system_name]()
    except KeyError as exc:
        raise ValueError(f"unknown system {system_name!r}; choose from {sorted(CONFIGS)}") from exc
