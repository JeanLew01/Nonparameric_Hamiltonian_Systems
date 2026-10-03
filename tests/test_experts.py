import dataclasses
from pathlib import Path

import casadi as ca
import numpy as np
import pytest

from symplectic_ncp.config import get_config
from symplectic_ncp.experts import load_demonstrations
from symplectic_ncp.experts.generate import demonstration_summary, demonstrations_path, rollout_expert
from symplectic_ncp.experts.nmpc import NMPCExpert, casadi_hamiltonian, casadi_vector_field

SYSTEM_NAMES = ["spring_mass", "single_pendulum"]
OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"


@pytest.mark.parametrize("name", SYSTEM_NAMES)
def test_casadi_model_matches_numpy_system(name):
    system = get_config(name).make_system()
    x = ca.SX.sym("x", system.state_dim)
    u = ca.SX.sym("u", system.control_dim)
    H = ca.Function("H", [x], [casadi_hamiltonian(system, x)])
    f = ca.Function("f", [x, u], [casadi_vector_field(system, x, u)])
    X = system.sample_energy_sublevel(32, 50.0, np.random.default_rng(0))
    U = np.linspace(-20.0, 20.0, 32)[:, None]
    H_cas = np.array([float(H(xi)) for xi in X])
    f_cas = np.array([np.array(f(xi, ui)).ravel() for xi, ui in zip(X, U)])
    assert np.allclose(H_cas, system.hamiltonian(X), atol=1e-10)
    assert np.allclose(f_cas, system.dynamics(X, U), atol=1e-10)


class _Damping:
    """u = -c p (energy dissipating), with a call counter to check the ZOH period."""

    def __init__(self, c):
        self.c, self.calls = c, 0

    def reset(self, x0):
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        return np.array([-self.c * x[1]])


def test_rollout_records_zoh_and_stops_in_certified_target():
    cfg = get_config("spring_mass")
    system = cfg.make_system()
    target = cfg.make_target(system)
    expert = _Damping(2.0)
    x0 = (0.5, 0.0)
    demo = rollout_expert(system, target, expert, x0, "damped", cfg)
    stride = int(round(cfg.control_period / cfg.sim_dt))
    assert np.array_equal(demo.states[0], np.asarray(x0))
    assert expert.calls == -(-demo.num_steps // stride)
    blocks = demo.controls[: (demo.num_steps // stride) * stride].reshape(-1, stride)
    assert np.all(blocks == blocks[:, :1])  # input held between expert updates
    inside = target.in_certified_target(demo.states, cfg.energy_eps, cfg.demo_delta)
    assert inside[-1] and not inside[:-1].any()  # stops at the first certified step
    # recorded states follow the plant integrator exactly
    replay = system.rk4_step(demo.states[:-1], demo.controls, cfg.sim_dt)
    assert np.allclose(replay, demo.states[1:], atol=1e-12)


def test_rollout_raises_when_target_not_reached():
    cfg = get_config("spring_mass")
    cfg.nmpc = dataclasses.replace(cfg.nmpc, max_duration=0.5)
    system = cfg.make_system()
    with pytest.raises(RuntimeError):
        rollout_expert(system, cfg.make_target(system), lambda x: np.zeros(1), (1.0, 0.0), "idle", cfg)


def test_nmpc_spring_mass_demonstration():
    cfg = get_config("spring_mass")
    system = cfg.make_system()
    target = cfg.make_target(system)
    expert = NMPCExpert(system, target, cfg.nmpc, cfg.control_period)
    spec = cfg.experts[2]  # (0, -2): pure braking, about 0.6 s
    demo = rollout_expert(system, target, expert, spec.x0, spec.name, cfg)
    assert np.array_equal(demo.states[0], np.asarray(spec.x0))
    assert np.max(np.abs(demo.controls)) <= cfg.u_bound
    assert target.in_certified_target(demo.states[-1], cfg.energy_eps, cfg.demo_delta)[0]
    assert demo.duration < 2.0
    # deterministic: a reset reproduces the demonstration exactly
    again = rollout_expert(system, target, expert, spec.x0, spec.name, cfg)
    assert np.array_equal(again.controls, demo.controls)


def test_nmpc_pendulum_is_invariant_to_angle_wrapping():
    cfg = get_config("single_pendulum")
    system = cfg.make_system()
    nmpc_cfg = dataclasses.replace(cfg.nmpc, horizon=25)
    expert = NMPCExpert(system, cfg.make_target(system), nmpc_cfg, cfg.control_period)
    x = np.array([3.0, 5.0])
    expert.reset(x)
    u1 = expert(x)
    expert.reset(x + np.array([2.0 * np.pi, 0.0]))
    u2 = expert(x + np.array([2.0 * np.pi, 0.0]))
    assert np.allclose(u1, u2, atol=1e-6)
    assert np.all(np.abs(u1) <= cfg.u_bound)
    # a wrapped state fed after an unwrapped one continues on the same branch
    expert.reset(np.array([np.pi - 0.01, 3.0]))
    expert(np.array([np.pi - 0.01, 3.0]))
    expert(np.array([-np.pi + 0.01, 3.0]))
    assert expert._last_x[0] == pytest.approx(np.pi + 0.01)


@pytest.mark.parametrize("name", SYSTEM_NAMES)
def test_saved_demonstrations_are_valid(name):
    """Checks the generated outputs/<system>/demonstrations.npz (skipped if not generated yet)."""
    path = demonstrations_path(OUTPUTS, name)
    if not path.exists():
        pytest.skip(f"{path} not generated")
    cfg = get_config(name)
    system = cfg.make_system()
    target = cfg.make_target(system)
    demos = load_demonstrations(path)
    assert [d.name for d in demos] == [s.name for s in cfg.experts]
    for demo, spec in zip(demos, cfg.experts):
        assert np.array_equal(demo.states[0], np.asarray(spec.x0))
        assert np.max(np.abs(demo.controls)) <= cfg.u_bound
        assert target.in_certified_target(demo.states[-1], cfg.energy_eps, cfg.demo_delta)[0]
        assert not target.in_certified_target(demo.states[:-1], cfg.energy_eps, cfg.demo_delta).any()
        assert demo.dt == cfg.sim_dt and demo.control_period == cfg.control_period
        replay = system.rk4_step(demo.states[:-1], demo.controls, cfg.sim_dt)
        assert np.allclose(system.difference(replay, demo.states[1:]), 0.0, atol=1e-9)
        summary = demonstration_summary(system, target, demo)
        assert summary["final_delta_H"] <= -cfg.energy_eps
    if name == "single_pendulum":
        for demo in demos:
            if demo.name.startswith("rotation"):
                # rotation demos brake without reversing their direction of motion
                p = demo.states[:, 1]
                rotating = system.hamiltonian(demo.states) > system.separatrix_energy
                assert np.all(np.sign(p[rotating]) == np.sign(p[0]))
                assert np.all(np.sign(p) == np.sign(p[0]))
        rest = next(d for d in demos if d.name == "libration_rest")
        assert np.array_equal(rest.states[0], np.zeros(2))
        # libration demonstrations stay below the separatrix (NMPC energy cap)
        for demo in demos:
            if demo.name.startswith("libration"):
                assert system.hamiltonian(demo.states).max() <= system.separatrix_energy + 1e-3
