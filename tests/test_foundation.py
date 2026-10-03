import numpy as np
import pytest

from symplectic_ncp.chain import DEFAULT, AssignmentSet, NonparametricChainPolicy
from symplectic_ncp.config import get_config
from symplectic_ncp.experts import Demonstration, load_demonstrations, save_demonstrations
from symplectic_ncp.simulation import RolloutResult
from symplectic_ncp.systems import SinglePendulum, SpringMass, make_system
from symplectic_ncp.target import TargetSet

SYSTEMS = [SpringMass(), SinglePendulum()]


@pytest.mark.parametrize("system", SYSTEMS, ids=lambda s: s.name)
def test_structure_is_hamiltonian(system):
    J, G = system.structure_matrices()
    assert np.allclose(J, -J.T)
    x = system.sample_energy_sublevel(64, 10.0, np.random.default_rng(0))
    # zero input is power conserving: dH/dt = grad H . J grad H = 0
    power = np.einsum("bi,bi->b", system.grad_hamiltonian(x), system.dynamics(x, 0.0))
    assert np.allclose(power, 0.0, atol=1e-10)
    # control enters through G only
    u = np.full((64, 1), 3.0)
    assert np.allclose(system.dynamics(x, u) - system.dynamics(x, 0.0), u @ G.T)


@pytest.mark.parametrize("system", SYSTEMS, ids=lambda s: s.name)
def test_gradient_and_jacobian_match_finite_differences(system):
    x = system.sample_energy_sublevel(16, 10.0, np.random.default_rng(1))
    eps = 1e-6
    for i in range(system.state_dim):
        e = np.zeros(system.state_dim)
        e[i] = eps
        dH = (system.hamiltonian(x + e) - system.hamiltonian(x - e)) / (2 * eps)
        assert np.allclose(dH, system.grad_hamiltonian(x)[:, i], atol=1e-5)
        df = (system.dynamics(x + e, 0.0) - system.dynamics(x - e, 0.0)) / (2 * eps)
        assert np.allclose(df, system.state_jacobian(x)[:, :, i], atol=1e-5)


@pytest.mark.parametrize("system", SYSTEMS, ids=lambda s: s.name)
def test_rk4_conserves_energy_under_zero_input(system):
    x = system.sample_energy_sublevel(32, 5.0, np.random.default_rng(2))
    H0 = system.hamiltonian(x)
    for _ in range(400):
        x = system.rk4_step(x, 0.0, 0.005)
    assert np.max(np.abs(system.hamiltonian(x) - H0)) < 1e-6


def test_rk4_accepts_per_row_steps():
    system = SpringMass()
    x = np.array([[1.0, 0.0], [1.0, 0.0]])
    out = system.rk4_step(x, 0.0, np.array([0.0, 0.1]))
    assert np.allclose(out[0], x[0])
    assert np.allclose(out[1], system.rk4_step(x[1], 0.0, 0.1)[0])


def test_lipschitz_constants_spring_mass_are_analytic():
    L_H, L = SpringMass().lipschitz_constants(2.0)
    assert L_H == pytest.approx(2.0, rel=1e-6)  # sup sqrt(q^2 + p^2) on {H <= 2}
    assert L == pytest.approx(1.0)


def test_pendulum_geometry_wraps_angles():
    pend = SinglePendulum()
    a = np.array([np.pi - 0.05, 0.0])
    b = np.array([-np.pi + 0.05, 0.0])
    assert pend.distance(a, b) == pytest.approx(0.1)
    assert np.allclose(pend.difference(a, b), [-0.1, 0.0])
    comps = pend.ergodic_component(np.array([[0.0, 1.0], [0.0, 30.0], [0.0, -30.0]]))
    assert comps.tolist() == [0, 1, 2]


def test_energy_band_and_signed_distance():
    pend = SinglePendulum()
    target = TargetSet(pend, np.array([np.pi, 0.0]), 0.1)
    # band of the 0.1-ball around the upright equilibrium
    assert target.H_min == pytest.approx(pend.mgl * (1.0 - np.cos(np.pi - 0.1)), rel=1e-6)
    assert target.H_max == pytest.approx(2 * pend.mgl + 0.1**2 / (2 * pend.inertia), rel=1e-6)
    E = np.linspace(0.0, 80.0, 1001)
    inside = (E >= target.H_min) & (E <= target.H_max)
    assert np.array_equal(target.energy_distance_from_energy(E) <= 0.0, inside)
    assert target.contains(np.array([-np.pi + 0.05, 0.05]))[0]
    spring_target = get_config("spring_mass").make_target()
    assert spring_target.H_min == pytest.approx(0.0, abs=1e-12)
    assert spring_target.H_max == pytest.approx(0.005)
    x = np.array([[0.06, 0.0], [0.0, 0.0], [0.2, 0.0]])
    assert spring_target.in_certified_target(x, 1e-3, 0.1).tolist() == [True, False, False]


def test_assignment_set_roundtrip_and_selection(tmp_path):
    pend = make_system("single_pendulum")
    K = AssignmentSet(
        centers=np.array([[3.1, 0.0], [-3.1, 1.0], [0.0, 0.0]]),
        radii=np.array([0.1, 0.2, 0.05]),
        controls=[np.ones((3, 1)), np.zeros((2, 1)), np.ones((1, 1))],
        dt=0.005,
        demo_ids=[0, 0, 1],
    )
    assert np.allclose(K.durations, [0.015, 0.01, 0.005])
    K.save(tmp_path / "K.npz")
    K2 = AssignmentSet.load(tmp_path / "K.npz")
    assert np.allclose(K2.centers, K.centers) and K2.steps.tolist() == [3, 2, 1]
    for field in ("radii", "anchor_times", "leads", "demo_ids"):
        assert np.array_equal(getattr(K2, field), getattr(K, field))
    # snippets whose anchor lies between grid points start with a shorter lead step
    K4 = AssignmentSet(K.centers, K.radii, K.controls, 0.005, leads=[0.002, 0.005, 0.001])
    assert np.allclose(K4.durations, [0.012, 0.01, 0.001])
    K4.save(tmp_path / "K4.npz")
    assert np.array_equal(AssignmentSet.load(tmp_path / "K4.npz").leads, K4.leads)
    with pytest.raises(ValueError):
        AssignmentSet(K.centers, K.radii, K.controls, 0.005, leads=[0.006, 0.005, 0.001])
    assert len(K.from_demos([1])) == 1 and len(K.from_demos([0, 1])) == 3

    policy = NonparametricChainPolicy(K, pend)
    queries = np.array([[-3.15, 0.0], [3.12, 0.95], [1.0, 1.0], [0.01, 0.0]])
    assert policy.select(queries).tolist() == [0, 1, DEFAULT, 2]
    # overlapping balls: the smallest normalized distance wins (Definition 7)
    K3 = AssignmentSet(np.array([[0.0, 0.0], [0.3, 0.0]]), np.array([1.0, 0.5]), [np.zeros((1, 1))] * 2, 0.005)
    assert NonparametricChainPolicy(K3, pend).select(np.array([0.28, 0.0])).tolist() == [1]


def test_demonstration_roundtrip(tmp_path):
    states = np.cumsum(np.ones((9, 2)), axis=0)
    demo = Demonstration("d", states, np.arange(8.0).reshape(8, 1), dt=0.005, control_period=0.02)
    save_demonstrations(tmp_path / "demos.npz", [demo, demo])
    loaded = load_demonstrations(tmp_path / "demos.npz")
    assert [d.name for d in loaded] == ["d", "d"]
    assert np.allclose(loaded[1].states, states) and loaded[0].duration == pytest.approx(0.04)
    xs, us = demo.control_samples()
    assert us[:, 0].tolist() == [0.0, 4.0]


def test_rollout_summary_assigns_horizon_to_failures():
    res = RolloutResult(np.array([True, False]), np.array([2.0, np.inf]), horizon=20.0)
    s = res.summary()
    assert s["success_rate"] == 0.5 and s["mean_reach_time"] == pytest.approx(11.0)
    assert s["mean_reach_time_successful"] == pytest.approx(2.0)
