"""Target set S_tgt = {x : ||x - x*|| <= radius} and its energy geometry (Section III-A).

Implements the energy band H(S_tgt) = [H_min, H_max], the energy signed
distance Delta H (Definition 8) and the shrunken sets H_tgt^eps and
S_tgt^delta used to certify expert demonstrations (Section III-D).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from symplectic_ncp.systems.base import HamiltonianSystem, as_batch


def _ball_samples(dim: int, radius: float, num_dirs: int = 4096, num_shells: int = 64) -> np.ndarray:
    """Deterministic samples of the closed ball of the given radius centred at 0."""
    if dim == 2:
        angles = np.linspace(0.0, 2.0 * np.pi, num_dirs, endpoint=False)
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    else:
        rng = np.random.default_rng(0)
        dirs = rng.normal(size=(num_dirs, dim))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    shells = np.linspace(0.0, radius, num_shells + 1)[1:]
    pts = (shells[:, None, None] * dirs[None, :, :]).reshape(-1, dim)
    return np.vstack([np.zeros((1, dim)), pts])


@dataclass
class TargetSet:
    """Closed Euclidean ball around ``center`` (angles wrapped) and its energy band."""

    system: HamiltonianSystem
    center: np.ndarray
    radius: float = 0.1
    H_min: float = field(init=False)
    H_max: float = field(init=False)

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=float).reshape(self.system.state_dim)
        self.radius = float(self.radius)
        energies = self.system.hamiltonian(self.center[None, :] + _ball_samples(self.system.state_dim, self.radius))
        self.H_min = float(np.min(energies))
        self.H_max = float(np.max(energies))

    # Definition 8 -----------------------------------------------------------
    @property
    def H_plus(self) -> float:
        """H*_+ = (H_max + H_min) / 2."""
        return 0.5 * (self.H_max + self.H_min)

    @property
    def H_minus(self) -> float:
        """H*_- = (H_max - H_min) / 2."""
        return 0.5 * (self.H_max - self.H_min)

    def energy_distance(self, x) -> np.ndarray:
        """Delta H(x) = |H(x) - H*_+| - H*_-  (<= 0 iff H(x) in H(S_tgt))."""
        return self.energy_distance_from_energy(self.system.hamiltonian(x))

    def energy_distance_from_energy(self, energy) -> np.ndarray:
        return np.abs(np.asarray(energy, dtype=float) - self.H_plus) - self.H_minus

    # Sets ---------------------------------------------------------------------
    def distance(self, x) -> np.ndarray:
        return self.system.distance(as_batch(x, self.system.state_dim), self.center[None, :])

    def contains(self, x) -> np.ndarray:
        """Membership in S_tgt."""
        return self.distance(x) <= self.radius

    def in_energy_target(self, x, eps: float = 0.0) -> np.ndarray:
        """Membership in H_tgt^eps = {Delta H <= -eps} (H_tgt for eps = 0)."""
        return self.energy_distance(x) <= -float(eps)

    def in_certified_target(self, x, eps: float, delta: float) -> np.ndarray:
        """Membership in S_tgt^delta = {||x - x*|| <= delta} intersected with H_tgt^eps.

        Expert demonstrations are required to end in this set (Section III-D),
        which is contained in H_tgt^eps and in S_tgt whenever delta <= radius.
        """
        return (self.distance(x) <= float(delta)) & self.in_energy_target(x, eps)
