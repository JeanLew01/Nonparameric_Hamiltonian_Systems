"""Nonparametric chain policy pi_K (paper, Definition 7).

The index map iota_K(x) selects the assignment with the smallest normalized
distance rho_K(x) = min_i ||x - x_i|| / r_i whenever rho_K(x) <= 1, and the
default control u_0 (the zero input) otherwise.  Index ``-1`` denotes the
default control in code (index 0 in the paper).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from symplectic_ncp.chain.assignment_set import AssignmentSet
from symplectic_ncp.systems.base import HamiltonianSystem, as_batch

DEFAULT = -1


class NonparametricChainPolicy:
    def __init__(self, assignments: AssignmentSet, system: HamiltonianSystem, membership_tol: float = 0.0):
        if len(assignments) == 0:
            raise ValueError("the assignment set is empty")
        self.K = assignments
        self.system = system
        self.membership_tol = float(membership_tol)
        self.r_max = float(np.max(assignments.radii))
        self._tree = cKDTree(system.to_periodic_box(assignments.centers), boxsize=system.periodic_boxsize())
        self.default_control = np.zeros(system.control_dim)

    def __len__(self) -> int:
        return len(self.K)

    # ----------------------------------------------------------- geometry
    def normalized_distance(self, x, indices) -> np.ndarray:
        """||x - x_i|| / r_i for a single state ``x`` and assignment ``indices``."""
        indices = np.asarray(indices, dtype=int)
        d = self.system.distance(np.asarray(x, dtype=float)[None, :], self.K.centers[indices])
        return d / self.K.radii[indices]

    def candidates(self, points, query_radii) -> list[np.ndarray]:
        """Indices of centers within ``query_radii`` of each point (periodic-aware KD-tree)."""
        P = self.system.to_periodic_box(as_batch(points, self.system.state_dim))
        radii = np.broadcast_to(np.asarray(query_radii, dtype=float), (P.shape[0],))
        lists = self._tree.query_ball_point(P, radii, return_sorted=False)
        return [np.asarray(lst, dtype=int) for lst in lists]

    # --------------------------------------------------------- Definition 7
    def select(self, x) -> np.ndarray:
        """iota_K for every state in the batch (DEFAULT outside Supp(K))."""
        X = as_batch(x, self.system.state_dim)
        out = np.full(X.shape[0], DEFAULT, dtype=int)
        for b, cand in enumerate(self.candidates(X, self.r_max * (1.0 + self.membership_tol))):
            if cand.size == 0:
                continue
            rho = self.normalized_distance(X[b], cand)
            j = int(np.argmin(rho))
            if rho[j] <= 1.0 + self.membership_tol:
                out[b] = int(cand[j])
        return out

    def in_support(self, x) -> np.ndarray:
        return self.select(x) != DEFAULT
