"""Expert demonstration D_j = (x_j, u_j(.), T_j) sampled on the simulation grid."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Demonstration:
    """One expert trajectory phi(t, x_j, u_j), t in [0, T_j].

    ``states[k]`` is the state at time ``k * dt`` and ``controls[k]`` is the
    input held on (k dt, (k + 1) dt].  The expert itself updates its input
    every ``control_period`` seconds (a multiple of ``dt``).
    """

    name: str
    states: np.ndarray  # (n_steps + 1, n)
    controls: np.ndarray  # (n_steps, m)
    dt: float
    control_period: float

    def __post_init__(self):
        self.states = np.asarray(self.states, dtype=float)
        self.controls = np.asarray(self.controls, dtype=float).reshape(self.states.shape[0] - 1, -1)

    @property
    def initial_state(self) -> np.ndarray:
        return self.states[0]

    @property
    def num_steps(self) -> int:
        return self.controls.shape[0]

    @property
    def duration(self) -> float:
        return self.num_steps * self.dt

    @property
    def times(self) -> np.ndarray:
        return np.arange(self.states.shape[0]) * self.dt

    def control_samples(self) -> tuple[np.ndarray, np.ndarray]:
        """(state, input) pairs at the expert's own update instants (used to train BC)."""
        stride = int(round(self.control_period / self.dt))
        idx = np.arange(0, self.num_steps, stride)
        return self.states[idx], self.controls[idx]


def save_demonstrations(path, demos: list[Demonstration], fingerprint: str | None = None) -> None:
    """Save demonstrations; ``fingerprint`` records the configuration that generated them."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"names": np.asarray([d.name for d in demos])}
    if fingerprint is not None:
        payload["fingerprint"] = np.asarray(fingerprint)
    for j, d in enumerate(demos):
        payload[f"states_{j}"] = d.states
        payload[f"controls_{j}"] = d.controls
        payload[f"dt_{j}"] = d.dt
        payload[f"control_period_{j}"] = d.control_period
    np.savez(path, **payload)


def load_demonstrations(path) -> list[Demonstration]:
    data = np.load(Path(path))
    return [
        Demonstration(
            name=str(name),
            states=data[f"states_{j}"],
            controls=data[f"controls_{j}"],
            dt=float(data[f"dt_{j}"]),
            control_period=float(data[f"control_period_{j}"]),
        )
        for j, name in enumerate(data["names"])
    ]


def load_fingerprint(path) -> str | None:
    """Generation fingerprint stored by :func:`save_demonstrations` (None if absent)."""
    data = np.load(Path(path))
    return str(data["fingerprint"]) if "fingerprint" in data else None
