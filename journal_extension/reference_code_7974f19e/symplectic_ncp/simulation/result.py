"""Closed-loop rollout outcome and the summary statistics reported in Section IV."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RolloutResult:
    success: np.ndarray  # (B,) bool: the trajectory entered S_tgt before the horizon
    reach_time: np.ndarray  # (B,) first hitting time of S_tgt, np.inf when unsuccessful
    horizon: float
    extras: dict = field(default_factory=dict)  # per-trajectory diagnostics, arrays of shape (B,)
    trace: list | None = None  # optional per-trajectory time series (see simulate_chain_policy(record=True))

    def __post_init__(self):
        self.success = np.asarray(self.success, dtype=bool)
        self.reach_time = np.asarray(self.reach_time, dtype=float)

    @property
    def reach_time_or_horizon(self) -> np.ndarray:
        """Reach times where unsuccessful trajectories are assigned the horizon (Section IV)."""
        return np.where(self.success, np.minimum(self.reach_time, self.horizon), self.horizon)

    def summary(self) -> dict:
        times = self.reach_time_or_horizon
        succ = self.reach_time[self.success]
        return {
            "num_trajectories": int(self.success.size),
            "success_rate": float(np.mean(self.success)),
            "mean_reach_time": float(np.mean(times)),
            "std_reach_time": float(np.std(times)),
            "mean_reach_time_successful": float(np.mean(succ)) if succ.size else None,
        }
