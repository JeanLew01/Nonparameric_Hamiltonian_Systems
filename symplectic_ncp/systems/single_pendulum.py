"""Single pendulum (paper, Section IV-2).

x = [q, p] = [theta, m l^2 theta'],  H(x) = p^2 / (2 m l^2) + m g l (1 - cos q),
x' = [[0, 1], [-1, 0]] grad H(x) + [0, 1]^T u.

Energy layers below the separatrix energy 2 m g l are single libration
orbits; above it each layer splits into a counter-clockwise (p > 0) and a
clockwise (p < 0) rotation, i.e. two ergodic components.  The pendulum
therefore violates Assumption 6, which the paper's numerical section
explores on purpose.
"""

from __future__ import annotations

import numpy as np

from symplectic_ncp.systems.base import HamiltonianSystem, as_batch

LIBRATION, ROTATION_CCW, ROTATION_CW = 0, 1, 2


class SinglePendulum(HamiltonianSystem):
    name = "single_pendulum"
    state_dim = 2
    control_dim = 1
    angle_indices = (0,)

    def __init__(self, m: float = 1.0, l: float = 2.0, g: float = 9.81, u_min=-20.0, u_max=20.0):
        self.m = float(m)
        self.l = float(l)
        self.g = float(g)
        self.inertia = self.m * self.l**2
        self.mgl = self.m * self.g * self.l
        super().__init__(u_min, u_max)

    @property
    def separatrix_energy(self) -> float:
        return 2.0 * self.mgl

    def hamiltonian(self, x):
        X = as_batch(x, 2)
        return X[:, 1] ** 2 / (2.0 * self.inertia) + self.mgl * (1.0 - np.cos(X[:, 0]))

    def grad_hamiltonian(self, x):
        X = as_batch(x, 2)
        out = np.empty_like(X)
        out[:, 0] = self.mgl * np.sin(X[:, 0])
        out[:, 1] = X[:, 1] / self.inertia
        return out

    def structure_matrices(self):
        J = np.array([[0.0, 1.0], [-1.0, 0.0]])
        G = np.array([[0.0], [1.0]])
        return J, G

    def state_jacobian(self, x):
        X = as_batch(x, 2)
        jac = np.zeros((X.shape[0], 2, 2))
        jac[:, 0, 1] = 1.0 / self.inertia
        jac[:, 1, 0] = -self.mgl * np.cos(X[:, 0])
        return jac

    def bounding_box(self, energy_bound):
        p_max = np.sqrt(2.0 * self.inertia * energy_bound)
        return np.array([-np.pi, -p_max]), np.array([np.pi, p_max])

    def ergodic_component(self, x):
        X = as_batch(x, 2)
        rotating = self.hamiltonian(X) > self.separatrix_energy
        labels = np.full(X.shape[0], LIBRATION, dtype=int)
        labels[rotating & (X[:, 1] > 0.0)] = ROTATION_CCW
        labels[rotating & (X[:, 1] < 0.0)] = ROTATION_CW
        return labels

    def component_names(self):
        return {LIBRATION: "libration", ROTATION_CCW: "rotation_ccw", ROTATION_CW: "rotation_cw"}
