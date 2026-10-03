import time

import numpy as np
import pytest
import torch

from symplectic_ncp.baselines import BehaviorCloningPolicy, state_features, train_behavior_cloning
from symplectic_ncp.config import BCConfig, get_config
from symplectic_ncp.experts.demonstration import Demonstration


def _damping_demo(system, x0, duration=4.0, dt=0.005, control_period=0.02, gain=2.0):
    """Synthetic expert: ZOH damping u = -gain * p (not an NMPC; enough to test BC)."""
    stride = int(round(control_period / dt))
    n = int(round(duration / dt))
    states = [np.asarray(x0, dtype=float)]
    controls = []
    u = np.zeros(1)
    for k in range(n):
        if k % stride == 0:
            u = system.clip_control(-gain * states[-1][1:2])[0]
        controls.append(u)
        states.append(system.rk4_step(states[-1], u, dt)[0])
    return Demonstration("damp", np.array(states), np.array(controls), dt, control_period)


@pytest.fixture(scope="module")
def spring():
    cfg = get_config("spring_mass")
    system = cfg.make_system()
    demos = [_damping_demo(system, x0) for x0 in [(2.0, 0.0), (0.0, -2.0)]]
    return cfg, system, demos


@pytest.fixture(scope="module")
def pendulum():
    cfg = get_config("single_pendulum")
    system = cfg.make_system()
    demos = [_damping_demo(system, (2.5, 0.0), gain=1.0)]
    return cfg, system, demos


def test_config_matches_paper():
    bc = BCConfig()
    assert tuple(bc.hidden_sizes) == (24, 24, 16)
    assert bc.lr == pytest.approx(1.2e-3)
    assert bc.weight_decay == pytest.approx(5e-4)
    assert bc.epochs == 40


def test_architecture_and_shapes(spring):
    cfg, system, demos = spring
    policy = train_behavior_cloning(system, demos, cfg.bc, seed=0)
    assert [W.shape for W in policy.weights] == [(2, 24), (24, 24), (24, 16), (16, 1)]
    assert len(policy.train_losses) == cfg.bc.epochs
    assert policy.train_losses[-1] < policy.train_losses[0]
    U = policy(np.random.default_rng(0).normal(size=(7, 2)))
    assert U.shape == (7, 1)
    assert policy(np.array([0.3, -0.1])).shape == (1, 1)


def test_imitates_linear_expert(spring):
    cfg, system, demos = spring
    policy = train_behavior_cloning(system, demos, cfg.bc, seed=0)
    X = np.concatenate([d.control_samples()[0] for d in demos])
    U = np.concatenate([d.control_samples()[1] for d in demos])
    rmse = np.sqrt(np.mean((policy(X) - U) ** 2))
    assert rmse < 0.2 * U.std()


def test_outputs_clipped(spring):
    cfg, system, demos = spring
    policy = train_behavior_cloning(system, demos, cfg.bc, seed=0)
    U = policy(np.array([[0.0, 1e4], [0.0, -1e4], [1e4, 1e4]]))
    assert np.all(U <= system.u_max) and np.all(U >= system.u_min)
    assert np.isclose(np.abs(U).max(), 20.0)


def test_deterministic_given_seed(spring):
    cfg, system, demos = spring
    a = train_behavior_cloning(system, demos, cfg.bc, seed=3)
    b = train_behavior_cloning(system, demos, cfg.bc, seed=3)
    c = train_behavior_cloning(system, demos, cfg.bc, seed=4)
    for Wa, Wb in zip(a.weights, b.weights):
        np.testing.assert_array_equal(Wa, Wb)
    assert any(not np.array_equal(Wa, Wc) for Wa, Wc in zip(a.weights, c.weights))


def test_numpy_forward_matches_torch(pendulum):
    cfg, system, demos = pendulum
    policy = train_behavior_cloning(system, demos, cfg.bc, seed=0)
    X = system.sample_energy_sublevel(200, 160.0, np.random.default_rng(1))
    F = (state_features(X, policy.angle_indices, policy.angle_features) - policy.x_mean) / policy.x_std
    with torch.no_grad():
        out = policy.to_torch()(torch.from_numpy(F)).numpy() * policy.u_std + policy.u_mean
    np.testing.assert_allclose(policy(X), np.clip(out, -20.0, 20.0), atol=1e-10)


def test_angle_features_periodic(pendulum):
    cfg, system, demos = pendulum
    policy = train_behavior_cloning(system, demos, cfg.bc, seed=0)
    assert policy.weights[0].shape[0] == 3  # (sin q, cos q, p)
    X = np.array([[3.1, 1.0], [-2.0, -4.0]])
    np.testing.assert_allclose(policy(X), policy(X + [2.0 * np.pi, 0.0]), atol=1e-9)


def test_save_load_roundtrip(pendulum, tmp_path):
    cfg, system, demos = pendulum
    policy = train_behavior_cloning(system, demos, cfg.bc, seed=0)
    policy.save(tmp_path / "bc.npz")
    loaded = BehaviorCloningPolicy.load(tmp_path / "bc.npz")
    X = system.sample_energy_sublevel(50, 160.0, np.random.default_rng(2))
    np.testing.assert_array_equal(policy(X), loaded(X))
    assert loaded.train_losses == policy.train_losses
    assert loaded.angle_indices == (0,)


def test_forward_is_fast(pendulum):
    cfg, system, demos = pendulum
    policy = train_behavior_cloning(system, demos, cfg.bc, seed=0)
    X = system.sample_energy_sublevel(500, 160.0, np.random.default_rng(3))
    policy(X)
    timings = []
    for _ in range(50):
        t0 = time.perf_counter()
        policy(X)
        timings.append(time.perf_counter() - t0)
    assert np.median(timings) < 2e-3
