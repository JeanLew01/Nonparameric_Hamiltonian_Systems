"""Nonlinear MPC expert (paper, Section IV: "NMPC implemented in Python using CasADi and do-mpc").

The prediction model is built from the same port-Hamiltonian structure as the
plant (Definition 1): x' = J grad H(x) + G u, with H written symbolically in
CasADi for each system and grad H obtained by automatic differentiation.  The
paper does not state the NMPC formulation; we use the simplest standard one:

    min  sum_k  e(x_k)^T Q e(x_k) + R |u_k|^2  +  e(x_N)^T Q_N e(x_N)
    s.t. x' = J grad H(x) + G u (orthogonal collocation),  u_min <= u <= u_max,

where e(x) = x - x* for Euclidean coordinates and, for angles, the smooth
chordal error e_q^2 = 2 (1 - cos(q - q*)) (no wrapped difference, so the cost
is smooth on the circle).
"""

from __future__ import annotations

import warnings
from typing import Callable

import casadi as ca
import numpy as np

from symplectic_ncp.config import NMPCConfig
from symplectic_ncp.systems.base import HamiltonianSystem
from symplectic_ncp.target import TargetSet

with warnings.catch_warnings():  # optional do-mpc features (ONNX, OPC UA) warn on import
    warnings.simplefilter("ignore")
    import do_mpc

# --------------------------------------------------------------------------
# Symbolic Hamiltonians, keyed by ``system.name`` (parameters read from the system object).
# --------------------------------------------------------------------------


def _spring_mass_hamiltonian(system, x):
    q, p = x[0], x[1]
    return p**2 / (2.0 * system.m) + 0.5 * system.k * q**2


def _single_pendulum_hamiltonian(system, x):
    q, p = x[0], x[1]
    return p**2 / (2.0 * system.inertia) + system.mgl * (1.0 - ca.cos(q))


CASADI_HAMILTONIANS: dict[str, Callable] = {
    "spring_mass": _spring_mass_hamiltonian,
    "single_pendulum": _single_pendulum_hamiltonian,
}


def casadi_hamiltonian(system: HamiltonianSystem, x):
    """H(x) as a CasADi expression of the symbolic state ``x`` (n x 1)."""
    try:
        builder = CASADI_HAMILTONIANS[system.name]
    except KeyError as exc:
        raise ValueError(f"no CasADi Hamiltonian for system {system.name!r}") from exc
    return builder(system, x)


def casadi_vector_field(system: HamiltonianSystem, x, u):
    """f(x, u) = J grad H(x) + G u with grad H from CasADi automatic differentiation."""
    J, G = system.structure_matrices()
    grad_H = ca.gradient(casadi_hamiltonian(system, x), x)
    return ca.mtimes(ca.DM(J), grad_H) + ca.mtimes(ca.DM(G), u)


def tracking_error_squared(system: HamiltonianSystem, x, x_ref: np.ndarray):
    """Per-coordinate squared error; angles use the smooth chordal error 2 (1 - cos(q - q*))."""
    terms = []
    for i in range(system.state_dim):
        if i in system.angle_indices:
            terms.append(2.0 * (1.0 - ca.cos(x[i] - float(x_ref[i]))))
        else:
            terms.append((x[i] - float(x_ref[i])) ** 2)
    return ca.vertcat(*terms)


# --------------------------------------------------------------------------
# Expert
# --------------------------------------------------------------------------


class NMPCExpert:
    """do-mpc NMPC steering ``system`` to the target centre x*.

    ``__call__(x)`` solves one NMPC problem (warm-started from the previous
    solution) and returns the first input, clipped to [u_min, u_max].  Inputs
    are piecewise constant over ``control_period``, which is also the
    prediction step.  Solves are quiet (IPOPT output suppressed) and
    deterministic: :meth:`reset` restores an identical initial guess.

    ``energy_cap`` (optional) adds the soft state constraint H(x) <= energy_cap
    (penalty ``cap_penalty`` on the slack).  It keeps a demonstration inside
    one family of energy layers, e.g. a pendulum libration demonstration below
    the separatrix, so that each demonstration covers a single ergodic
    component (Theorem 2, Condition 3).

    ``initial_input`` is the input guess used by :meth:`reset` (default 0).
    At the pendulum's bottom rest state the swing-up problem is symmetric in
    u; with this zero guess the direction is fixed (deterministically) by the
    floating-point value of sin(0 - pi) in the cost gradient.
    """

    def __init__(
        self,
        system: HamiltonianSystem,
        target: TargetSet,
        cfg: NMPCConfig,
        control_period: float,
        collocation_deg: int = 3,
        initial_input: np.ndarray | float = 0.0,
        energy_cap: float | None = None,
        cap_penalty: float = 1e4,
    ):
        self.system = system
        self.target = target
        self.cfg = cfg
        self.control_period = float(control_period)
        self.initial_input = np.broadcast_to(np.asarray(initial_input, dtype=float), (system.control_dim,)).copy()
        self.energy_cap = None if energy_cap is None else float(energy_cap)
        self.cap_penalty = float(cap_penalty)
        self._model = self._build_model()
        self._mpc = self._build_mpc(collocation_deg)
        self._last_x: np.ndarray | None = None
        self.reset(np.asarray(target.center, dtype=float))

    # ------------------------------------------------------------ setup
    def _build_model(self):
        n, m = self.system.state_dim, self.system.control_dim
        model = do_mpc.model.Model("continuous", "SX")
        x = model.set_variable("_x", "x", shape=(n, 1))
        u = model.set_variable("_u", "u", shape=(m, 1))
        model.set_rhs("x", casadi_vector_field(self.system, x, u))
        model.set_expression("H", casadi_hamiltonian(self.system, x))
        model.setup()
        return model

    def _build_mpc(self, collocation_deg: int):
        n, m = self.system.state_dim, self.system.control_dim
        cfg = self.cfg
        Q = np.broadcast_to(np.asarray(cfg.Q, dtype=float), (n,))
        Q_N = np.broadcast_to(np.asarray(cfg.Q_terminal, dtype=float), (n,))
        R = np.broadcast_to(np.asarray(cfg.R, dtype=float), (m,))

        mpc = do_mpc.controller.MPC(self._model)
        mpc.settings.n_horizon = int(cfg.horizon)
        mpc.settings.t_step = self.control_period
        mpc.settings.n_robust = 0
        mpc.settings.state_discretization = "collocation"
        mpc.settings.collocation_type = "radau"
        mpc.settings.collocation_deg = int(collocation_deg)
        mpc.settings.collocation_ni = 1
        mpc.settings.store_full_solution = False
        mpc.settings.supress_ipopt_output()

        x = self._model.x["x"]
        u = self._model.u["u"]
        err2 = tracking_error_squared(self.system, x, np.asarray(self.target.center, dtype=float))
        lterm = ca.dot(ca.DM(Q), err2) + ca.dot(ca.DM(R), u**2)
        mterm = ca.dot(ca.DM(Q_N), err2)
        mpc.set_objective(lterm=lterm, mterm=mterm)
        mpc.set_rterm(u=0.0)  # no input-rate penalty

        mpc.bounds["lower", "_u", "u"] = self.system.u_min
        mpc.bounds["upper", "_u", "u"] = self.system.u_max
        if self.energy_cap is not None:
            mpc.set_nl_cons(
                "energy_cap",
                self._model.aux["H"],
                ub=self.energy_cap,
                soft_constraint=True,
                penalty_term_cons=self.cap_penalty,
            )
        mpc.setup()
        return mpc

    # ------------------------------------------------------------ runtime
    def reset(self, x0) -> None:
        """Reset time, history and warm start; the initial guess is x0 / ``initial_input`` everywhere."""
        x0 = np.asarray(x0, dtype=float).reshape(self.system.state_dim)
        mpc = self._mpc
        mpc.reset_history()
        mpc.t0 = 0.0
        mpc.x0 = x0
        mpc.u0 = self.initial_input
        mpc.set_initial_guess()
        mpc.flags["initial_run"] = False  # do not pass multipliers of a previous episode
        self._last_x = x0.copy()

    def _unwrap(self, x: np.ndarray) -> np.ndarray:
        """Shift angle coordinates by multiples of 2 pi to the branch of the previous state.

        The NMPC is invariant to this shift (the cost and H are 2 pi periodic);
        it only keeps the warm start consistent when the caller wraps angles.
        """
        x = x.copy()
        if self._last_x is not None:
            for i in self.system.angle_indices:
                x[i] = self._last_x[i] + self.system.difference(x, self._last_x)[i]
        return x

    def __call__(self, x) -> np.ndarray:
        x = self._unwrap(np.asarray(x, dtype=float).reshape(self.system.state_dim))
        self._last_x = x
        u = np.asarray(self._mpc.make_step(x.reshape(-1, 1)), dtype=float).reshape(self.system.control_dim)
        return np.clip(u, self.system.u_min, self.system.u_max)

    @property
    def solver_stats(self) -> dict:
        return dict(self._mpc.solver_stats)
