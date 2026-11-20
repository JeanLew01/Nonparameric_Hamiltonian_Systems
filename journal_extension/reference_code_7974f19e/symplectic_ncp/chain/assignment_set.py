"""Control alphabet and assignment set (paper, Definitions 5 and 6).

An assignment set K = {(x_i, r_i, u_i)}_{i=1}^N stores, for every
verification triple, the center state x_i, the certified radius r_i and the
open-loop control snippet u_i : (0, tau_i] -> U.  Snippets are stored as
zero-order-hold sequences: ``controls[i][0]`` is held for the first
``leads[i]`` in (0, dt] seconds and ``controls[i][k]``, k >= 1, on the
following full steps of length ``dt``, so
tau_i = leads[i] + (len(controls[i]) - 1) dt.  A lead shorter than dt arises
when Algorithm 1 places the anchor between two sampling instants of the
demonstration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class AssignmentSet:
    centers: np.ndarray  # (N, n) verification centers x_i
    radii: np.ndarray  # (N,) certified radii r_i > 0
    controls: list[np.ndarray]  # N arrays of shape (n_i, m): snippet u_i on the dt grid
    dt: float  # sampling step of the snippets
    demo_ids: np.ndarray = field(default=None)  # (N,) index of the source demonstration
    anchor_times: np.ndarray = field(default=None)  # (N,) anchor time s inside the source demonstration
    leads: np.ndarray = field(default=None)  # (N,) duration of the first ZOH step, in (0, dt]

    def __post_init__(self):
        n = len(self.controls)
        centers = np.asarray(self.centers, dtype=float)
        self.centers = centers.reshape(n, centers.shape[-1]) if centers.ndim == 2 else centers.reshape(n, -1)
        self.radii = np.asarray(self.radii, dtype=float).reshape(-1)
        self.controls = [np.asarray(u, dtype=float).reshape(len(u), -1) for u in self.controls]
        self.demo_ids = np.zeros(n, dtype=int) if self.demo_ids is None else np.asarray(self.demo_ids, dtype=int)
        self.anchor_times = (
            np.zeros(n, dtype=float) if self.anchor_times is None else np.asarray(self.anchor_times, dtype=float)
        )
        self.leads = np.full(n, float(self.dt)) if self.leads is None else np.asarray(self.leads, dtype=float).reshape(-1)
        if not (self.radii.shape[0] == self.demo_ids.shape[0] == self.anchor_times.shape[0] == self.leads.shape[0] == n):
            raise ValueError("inconsistent assignment-set lengths")
        if n and (np.any(self.leads <= 0.0) or np.any(self.leads > self.dt * (1.0 + 1e-9))):
            raise ValueError("snippet leads must lie in (0, dt]")
        if n and np.any(self.radii <= 0.0):
            raise ValueError("certified radii must be positive")

    # ----------------------------------------------------------------- basic
    def __len__(self) -> int:
        return len(self.controls)

    @property
    def steps(self) -> np.ndarray:
        """Number of dt-steps of every snippet, shape (N,)."""
        return np.asarray([u.shape[0] for u in self.controls], dtype=int)

    @property
    def durations(self) -> np.ndarray:
        """Snippet durations tau_i, shape (N,)."""
        return self.leads + (self.steps - 1) * self.dt

    @classmethod
    def empty(cls, state_dim: int, dt: float) -> "AssignmentSet":
        return cls(np.zeros((0, state_dim)), np.zeros(0), [], dt)

    # ------------------------------------------------------------ algebra
    @classmethod
    def concatenate(cls, sets: list["AssignmentSet"]) -> "AssignmentSet":
        sets = [s for s in sets if len(s)]
        if not sets:
            raise ValueError("cannot concatenate only empty assignment sets")
        dt = sets[0].dt
        if any(abs(s.dt - dt) > 1e-12 for s in sets):
            raise ValueError("assignment sets use different snippet steps")
        return cls(
            centers=np.vstack([s.centers for s in sets]),
            radii=np.concatenate([s.radii for s in sets]),
            controls=[u for s in sets for u in s.controls],
            dt=dt,
            demo_ids=np.concatenate([s.demo_ids for s in sets]),
            anchor_times=np.concatenate([s.anchor_times for s in sets]),
            leads=np.concatenate([s.leads for s in sets]),
        )

    def subset(self, mask_or_indices) -> "AssignmentSet":
        idx = np.arange(len(self))[mask_or_indices]
        return AssignmentSet(
            centers=self.centers[idx],
            radii=self.radii[idx],
            controls=[self.controls[i] for i in idx],
            dt=self.dt,
            demo_ids=self.demo_ids[idx],
            anchor_times=self.anchor_times[idx],
            leads=self.leads[idx],
        )

    def from_demos(self, demo_ids) -> "AssignmentSet":
        """Triples extracted from the given demonstrations (incremental K for the first M demos)."""
        return self.subset(np.isin(self.demo_ids, np.asarray(list(demo_ids), dtype=int)))

    # ----------------------------------------------------------------- I/O
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        steps = self.steps
        flat = np.vstack(self.controls) if len(self) else np.zeros((0, 1))
        np.savez(
            path,
            centers=self.centers,
            radii=self.radii,
            control_flat=flat,
            control_steps=steps,
            dt=self.dt,
            demo_ids=self.demo_ids,
            anchor_times=self.anchor_times,
            leads=self.leads,
        )

    @classmethod
    def load(cls, path) -> "AssignmentSet":
        data = np.load(Path(path))
        offsets = np.concatenate([[0], np.cumsum(data["control_steps"])])
        flat = data["control_flat"]
        controls = [flat[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)]
        return cls(
            centers=data["centers"],
            radii=data["radii"],
            controls=controls,
            dt=float(data["dt"]),
            demo_ids=data["demo_ids"],
            anchor_times=data["anchor_times"],
            leads=data["leads"] if "leads" in data else None,
        )
