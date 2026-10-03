"""Port-Hamiltonian control systems  x' = J grad H(x) + G u   (paper, Definition 1).

All numerical methods are batched: states have shape ``(B, n)`` and controls
``(B, m)``.  A single state of shape ``(n,)`` is accepted wherever it is
unambiguous and is treated as a batch of size one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def wrap_angle(theta):
    """Wrap angles to [-pi, pi)."""
    return (np.asarray(theta, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi


def as_batch(x, dim: int) -> np.ndarray:
    """Return ``x`` as a float array of shape ``(B, dim)``."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, dim)
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise ValueError(f"expected shape (B, {dim}), got {arr.shape}")
    return arr


class HamiltonianSystem(ABC):
    """Lossless Hamiltonian system with affine input (Definition 1).

    Subclasses provide the Hamiltonian ``H``, its gradient, the constant
    structure matrices ``J`` (skew-symmetric) and ``G``, and the state
    Jacobian of the vector field.  Coordinates listed in ``angle_indices``
    live on the circle; distances and differences wrap them.
    """

    name: str = "hamiltonian_system"
    state_dim: int
    control_dim: int
    angle_indices: tuple[int, ...] = ()

    def __init__(self, u_min, u_max):
        self.u_min = np.asarray(u_min, dtype=float).reshape(self.control_dim)
        self.u_max = np.asarray(u_max, dtype=float).reshape(self.control_dim)

    # ------------------------------------------------------------------ model
    @abstractmethod
    def hamiltonian(self, x) -> np.ndarray:
        """H(x), shape ``(B,)``."""

    @abstractmethod
    def grad_hamiltonian(self, x) -> np.ndarray:
        """grad H(x), shape ``(B, n)``."""

    @abstractmethod
    def structure_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the constant ``(J, G)`` with J skew-symmetric (n, n), G (n, m)."""

    @abstractmethod
    def state_jacobian(self, x) -> np.ndarray:
        """d f / d x at ``x`` (independent of u because G is constant), shape ``(B, n, n)``."""

    @abstractmethod
    def bounding_box(self, energy_bound: float) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned box ``(low, high)`` containing the sublevel set {H <= energy_bound}."""

    def ergodic_component(self, x) -> np.ndarray:
        """Label of the zero-input ergodic component containing each state (Theorem 1).

        The default assumes every energy layer is a single ergodic component
        (Assumption 6) and returns zeros.
        """
        return np.zeros(as_batch(x, self.state_dim).shape[0], dtype=int)

    def component_names(self) -> dict[int, str]:
        return {0: "single"}

    # ------------------------------------------------------------- dynamics
    def clip_control(self, u) -> np.ndarray:
        return np.clip(as_batch(u, self.control_dim), self.u_min, self.u_max)

    def dynamics(self, x, u) -> np.ndarray:
        """f(x, u) = J grad H(x) + G u, shape ``(B, n)``."""
        X = as_batch(x, self.state_dim)
        U = np.asarray(u, dtype=float)
        JT, GT = self._structure_transposed()
        GU = np.full((1, self.control_dim), float(U)) @ GT if U.ndim == 0 else as_batch(U, self.control_dim) @ GT
        return self.grad_hamiltonian(X) @ JT + GU  # (1, n) or (B, n) input term broadcasts over rows

    def _structure_transposed(self) -> tuple[np.ndarray, np.ndarray]:
        """Cached (J^T, G^T); the structure matrices are constant (Definition 1)."""
        if getattr(self, "_JT_GT", None) is None:
            J, G = self.structure_matrices()
            self._JT_GT = (np.ascontiguousarray(J.T), np.ascontiguousarray(G.T))
        return self._JT_GT

    def rk4_step(self, x, u, h) -> np.ndarray:
        """One RK4 step with zero-order-hold input; ``h`` is a scalar or ``(B,)``."""
        X = as_batch(x, self.state_dim)
        H = np.asarray(h, dtype=float)
        H = H.reshape(-1, 1) if H.ndim == 1 else H
        k1 = self.dynamics(X, u)
        k2 = self.dynamics(X + 0.5 * H * k1, u)
        k3 = self.dynamics(X + 0.5 * H * k2, u)
        k4 = self.dynamics(X + H * k3, u)
        return X + (H / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # ------------------------------------------------------------- geometry
    def wrap(self, x) -> np.ndarray:
        """Wrap angle coordinates of ``x`` to [-pi, pi)."""
        X = np.array(x, dtype=float, copy=True)
        for idx in self.angle_indices:
            X[..., idx] = wrap_angle(X[..., idx])
        return X

    def difference(self, x, y) -> np.ndarray:
        """x - y with angle coordinates wrapped (broadcasting over leading axes)."""
        D = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
        for idx in self.angle_indices:
            D[..., idx] = wrap_angle(D[..., idx])
        return D

    def distance(self, x, y) -> np.ndarray:
        """Euclidean distance ||x - y|| on the state manifold (angles wrapped)."""
        return np.linalg.norm(self.difference(x, y), axis=-1)

    def periodic_boxsize(self) -> np.ndarray:
        """``boxsize`` argument for :class:`scipy.spatial.cKDTree` (0 = non-periodic)."""
        box = np.zeros(self.state_dim, dtype=float)
        for idx in self.angle_indices:
            box[idx] = 2.0 * np.pi
        return box

    def to_periodic_box(self, x) -> np.ndarray:
        """Map angle coordinates into [0, 2*pi) for periodic KD-tree queries."""
        X = np.array(x, dtype=float, copy=True)
        for idx in self.angle_indices:
            X[..., idx] = np.mod(X[..., idx], 2.0 * np.pi)
            X[..., idx] = np.where(X[..., idx] >= 2.0 * np.pi, 0.0, X[..., idx])
        return X

    # ------------------------------------------------- Lipschitz constants
    def lipschitz_constants(self, energy_bound: float, grid_points: int = 801) -> tuple[float, float]:
        """Constants of Assumptions 1-2 on the compact set X = {H <= energy_bound}.

        Returns ``(L_H, L)`` with ``L_H = sup ||grad H||`` and
        ``L = sup ||d f / d x||_2`` evaluated on a dense grid of the bounding
        box restricted to X.
        """
        low, high = self.bounding_box(energy_bound)
        axes = [np.linspace(lo, hi, grid_points) for lo, hi in zip(low, high)]
        mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, self.state_dim)
        inside = mesh[self.hamiltonian(mesh) <= energy_bound + 1e-12]
        L_H = float(np.max(np.linalg.norm(self.grad_hamiltonian(inside), axis=1)))
        L = float(np.max(np.linalg.norm(self.state_jacobian(inside), ord=2, axis=(1, 2))))
        return L_H, L

    def sample_energy_sublevel(self, num: int, energy_bound: float, rng: np.random.Generator) -> np.ndarray:
        """Uniform samples from {H(x) <= energy_bound} by rejection from the bounding box."""
        low, high = self.bounding_box(energy_bound)
        out = np.empty((0, self.state_dim))
        while out.shape[0] < num:
            batch = rng.uniform(low, high, size=(max(4 * (num - out.shape[0]), 256), self.state_dim))
            out = np.vstack([out, batch[self.hamiltonian(batch) <= energy_bound]])
        return out[:num]
