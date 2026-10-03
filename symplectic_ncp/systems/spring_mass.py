"""Spring-mass system (paper, Section IV-1).

x = [q, p],  H(x) = p^2 / (2m) + k q^2 / 2,  x' = [[0, 1], [-1, 0]] grad H(x) + [0, 1]^T u.
"""

from __future__ import annotations

import numpy as np

from symplectic_ncp.systems.base import HamiltonianSystem, as_batch


class SpringMass(HamiltonianSystem):
    name = "spring_mass"
    state_dim = 2
    control_dim = 1
    angle_indices = ()

    def __init__(self, m: float = 1.0, k: float = 1.0, u_min=-20.0, u_max=20.0):
        self.m = float(m)
        self.k = float(k)
        super().__init__(u_min, u_max)

    def hamiltonian(self, x):
        X = as_batch(x, 2)
        return X[:, 1] ** 2 / (2.0 * self.m) + 0.5 * self.k * X[:, 0] ** 2

    def grad_hamiltonian(self, x):
        X = as_batch(x, 2)
        return np.stack([self.k * X[:, 0], X[:, 1] / self.m], axis=1)

    def structure_matrices(self):
        J = np.array([[0.0, 1.0], [-1.0, 0.0]])
        G = np.array([[0.0], [1.0]])
        return J, G

    def state_jacobian(self, x):
        X = as_batch(x, 2)
        jac = np.array([[0.0, 1.0 / self.m], [-self.k, 0.0]])
        return np.broadcast_to(jac, (X.shape[0], 2, 2)).copy()

    def bounding_box(self, energy_bound):
        q_max = np.sqrt(2.0 * energy_bound / self.k)
        p_max = np.sqrt(2.0 * self.m * energy_bound)
        return np.array([-q_max, -p_max]), np.array([q_max, p_max])
