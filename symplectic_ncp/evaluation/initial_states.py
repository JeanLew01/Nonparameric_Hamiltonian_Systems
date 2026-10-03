"""Test initial states (paper, Section IV).

"For each system, we uniformly sample 500 initial states from the set of
states whose total energy is no greater than H_bar for testing."  The same
seeded sample is used for every method and every number M of demonstrations.
"""

from __future__ import annotations

import numpy as np

from symplectic_ncp.config import ExperimentConfig
from symplectic_ncp.systems import HamiltonianSystem


def sample_initial_states(system: HamiltonianSystem, cfg: ExperimentConfig) -> np.ndarray:
    """``cfg.num_inits`` states uniform on S_0 = {H(x) <= cfg.H_bar}, angles wrapped; shape (num_inits, n)."""
    rng = np.random.default_rng(cfg.init_seed)
    return system.wrap(system.sample_energy_sublevel(cfg.num_inits, cfg.H_bar, rng))


def component_fractions(system: HamiltonianSystem, X: np.ndarray) -> dict[str, float]:
    """Fraction of the states in each zero-input ergodic component (e.g. pendulum libration/rotation)."""
    labels = system.ergodic_component(X)
    return {name: float(np.mean(labels == k)) for k, name in system.component_names().items()}
