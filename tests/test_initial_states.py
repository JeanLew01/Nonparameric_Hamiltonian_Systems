from dataclasses import replace

import numpy as np
import pytest

from symplectic_ncp.config import get_config
from symplectic_ncp.evaluation import component_fractions, sample_initial_states


@pytest.mark.parametrize("name", ["spring_mass", "single_pendulum"])
def test_shape_bound_and_determinism(name):
    cfg = get_config(name)
    system = cfg.make_system()
    X = sample_initial_states(system, cfg)
    assert X.shape == (cfg.num_inits, system.state_dim) == (500, 2)
    assert np.all(system.hamiltonian(X) <= cfg.H_bar)
    np.testing.assert_array_equal(X, sample_initial_states(system, cfg))
    other = sample_initial_states(system, replace(cfg, init_seed=cfg.init_seed + 1))
    assert not np.array_equal(X, other)
    for idx in system.angle_indices:
        assert np.all((X[:, idx] >= -np.pi) & (X[:, idx] < np.pi))


def test_spring_mass_uniform_in_energy():
    # Uniform on the ellipse {H <= H_bar}: H / H_bar ~ U(0, 1) (area of a sublevel set is linear in H).
    cfg = replace(get_config("spring_mass"), num_inits=20000)
    system = cfg.make_system()
    ratio = system.hamiltonian(sample_initial_states(system, cfg)) / cfg.H_bar
    assert abs(ratio.mean() - 0.5) < 0.01
    assert abs(np.mean(ratio < 0.25) - 0.25) < 0.01


def test_pendulum_libration_fraction():
    # Area of {H <= 2 m g l} / area of {H <= 160} for m = 1, l = 2, g = 9.81 is about 0.34.
    cfg = replace(get_config("single_pendulum"), num_inits=50000)
    system = cfg.make_system()
    X = sample_initial_states(system, cfg)
    libration = np.mean(system.hamiltonian(X) < system.separatrix_energy)
    assert libration == pytest.approx(0.336, abs=0.01)
    frac = component_fractions(system, X)
    assert frac["libration"] == pytest.approx(libration)
    assert frac["rotation_ccw"] == pytest.approx(frac["rotation_cw"], abs=0.015)
    assert sum(frac.values()) == pytest.approx(1.0)
