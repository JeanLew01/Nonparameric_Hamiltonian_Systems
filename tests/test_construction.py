"""Tests of Algorithm 1 (Section III-D) on synthetic energy-shaping demonstrations."""

import warnings

import numpy as np
import pytest

from symplectic_ncp.chain.assignment_set import AssignmentSet
from symplectic_ncp.chain.construction import (
    LIPSCHITZ_MARGIN,
    build_certified_assignment_set,
    max_energy_on_balls,
    build_assignment_set,
    certified_radius_profile,
    extract_assignments,
    lipschitz_for_demos,
)
from symplectic_ncp.config import ChainConfig, get_config
from symplectic_ncp.experts.demonstration import Demonstration

DT = 0.005
V0 = 1e-3
FLOOR = 1e-3  # coarser Zeno floor than the default keeps the tests fast


def energy_shaping_demo(system, target, x0, H_ref, gain, max_time=30.0, name="demo"):
    """u = -gain (H - H_ref) dH/dp (clipped), RK4 at DT, stopped in S_tgt^delta (Section III-D)."""
    X, U = [np.asarray(x0, dtype=float)], []
    for _ in range(int(max_time / DT)):
        x = X[-1]
        if target.in_certified_target(x, 1e-3, 0.1)[0]:
            return Demonstration(name, np.asarray(X), np.asarray(U), DT, DT)
        u = system.clip_control(-gain * (system.hamiltonian(x) - H_ref) * system.grad_hamiltonian(x)[:, 1:2])
        U.append(u[0])
        X.append(system.rk4_step(x, u, DT)[0])
    raise RuntimeError("synthetic demonstration did not reach the target")


def make_case(name):
    cfg = get_config(name)
    system = cfg.make_system()
    target = cfg.make_target(system)
    if name == "spring_mass":  # H_ref below the band: damping-like energy decrease
        demos = [energy_shaping_demo(system, target, x0, -1.0, 2.0) for x0 in [(2.0, 0.0), (0.0, -2.0), (1.5, -1.0)]]
    else:  # swing-up to the middle of the target energy band
        demos = [energy_shaping_demo(system, target, (3.0, 0.0), target.H_plus, 0.5)]
    L_H, L, H_X = lipschitz_for_demos(system, demos, cfg.H_bar)
    return cfg, system, target, demos, L_H, L, H_X


@pytest.fixture(scope="module", params=["spring_mass", "single_pendulum"])
def case(request):
    return make_case(request.param)


def test_lipschitz_constants_on_energy_sublevel(case):
    cfg, system, target, demos, L_H, L, H_X = case
    E_max = max(cfg.H_bar, max(system.hamiltonian(d.states).max() for d in demos))
    assert H_X == pytest.approx((1.0 + LIPSCHITZ_MARGIN) * E_max)
    assert (L_H, L) == pytest.approx(system.lipschitz_constants(H_X))
    if system.name == "spring_mass":  # ||grad H|| = ||x|| <= sqrt(2 H_X), df/dx = J (norm 1)
        assert L_H == pytest.approx(np.sqrt(2.0 * H_X), rel=1e-3)
        assert L == pytest.approx(1.0)


def test_radius_profile_matches_formula(case):
    _, system, target, demos, L_H, L, _ = case
    demo = demos[0]
    dH = target.energy_distance(demo.states)
    for start in [0, 7, demo.num_steps // 2, demo.num_steps - 1]:
        r = certified_radius_profile(system, target, demo, start, V0, L_H, L)
        t = DT * np.arange(1, demo.num_steps - start + 1)
        with np.errstate(over="ignore"):
            direct = (dH[start] - dH[start + 1 :] - V0 * t) / (L_H + L_H * np.exp(L * t))
        assert r.shape == (demo.num_steps - start,)
        np.testing.assert_allclose(r, direct, rtol=1e-12, atol=1e-300)


def test_local_radius_profile_matches_formula(case):
    _, system, target, demos, L_H, L, _ = case
    demo, start = demos[0], 11
    r = certified_radius_profile(system, target, demo, start, V0, L_H, L, lipschitz="local")
    dH = target.energy_distance(demo.states)
    for k in [1, 5, 40]:
        seg = demo.states[start : start + k + 1]
        lh = np.max(np.linalg.norm(system.grad_hamiltonian(seg), axis=1))
        lf = np.max(np.linalg.norm(system.state_jacobian(seg), ord=2, axis=(1, 2)))
        direct = (dH[start] - dH[start + k] - V0 * k * DT) / (lh * (1.0 + np.exp(lf * k * DT)))
        assert r[k - 1] == pytest.approx(direct, rel=1e-12)
    with pytest.raises(ValueError):
        certified_radius_profile(system, target, demo, start, V0, L_H, L, lipschitz="bogus")


def test_radius_profile_no_overflow_warning(case):
    _, system, target, demos, L_H, _, _ = case
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = certified_radius_profile(system, target, demos[0], 0, V0, L_H, 5e3)
    assert np.all(np.isfinite(r))
    assert np.all(r[200:] == 0.0)  # e^{-L t} underflowed


def _demo_state(system, demo, s):
    """phi(s) on the continuous demonstration (partial RK4 step from the preceding grid point)."""
    k = min(int(np.floor(s / DT + 1e-9)), demo.num_steps)
    h = s - k * DT
    if h <= 1e-12 * DT:
        return demo.states[k]
    return system.rk4_step(demo.states[k], demo.controls[k], h)[0]


def _check_algorithm1(system, target, demo, K, L_H, L, lipschitz="global", min_advance=1e-4):
    """Invariants of Algorithm 1 (continuous anchors) on one demonstration."""
    dH = target.energy_distance(demo.states)
    s = K.anchor_times
    assert len(K) > 0 and s[0] == 0.0  # line 5: s = 0
    assert np.all(K.radii > 0.0)
    floor = np.minimum(min_advance, K.durations[:-1])
    assert np.all(np.diff(s) >= floor * (1 - 1e-9))  # anchors advance by sigma_i, floored
    checked = set(np.linspace(0, len(K) - 1, min(len(K), 40)).astype(int).tolist())  # keep the test fast
    for i, (s_i, x_i, r_i, u_i, lead) in enumerate(zip(s, K.centers, K.radii, K.controls, K.leads)):
        if i not in checked:
            continue
        k = int(np.floor(s_i / DT + 1e-9))
        theta = s_i / DT - k
        theta = 0.0 if theta < 1e-9 else theta
        assert not target.contains(x_i)[0]  # line 6
        np.testing.assert_allclose(x_i, _demo_state(system, demo, s_i), atol=1e-12)  # line 7
        assert lead == pytest.approx((1.0 - theta) * DT)
        steps = u_i.shape[0]
        np.testing.assert_array_equal(u_i, demo.controls[k : k + steps])  # u_{i,t} restriction
        t = (np.arange(1, demo.num_steps - k + 1) - theta) * DT
        if lipschitz == "global":
            direct = (target.energy_distance(x_i)[0] - dH[k + 1 :] - V0 * t) / (L_H * (1.0 + np.exp(L * t)))
            assert r_i == pytest.approx(np.max(direct), rel=1e-9)  # line 8
            assert steps == int(np.argmax(direct)) + 1
            tau = K.durations[i]  # Condition 1 holds with equality (Theorem 2)
            lhs = dH[k + steps] + V0 * tau + L_H * r_i * np.exp(L * tau)
            assert lhs == pytest.approx(target.energy_distance(x_i)[0] - L_H * r_i, rel=1e-9, abs=1e-10)
        # line 14: the next anchor is the exit point of the continuous trajectory (or the floor / tau_i)
        if i + 1 < len(K):
            advance = s[i + 1] - s_i
            assert advance <= K.durations[i] + 1e-12
            if advance > min_advance * (1 + 1e-9) and advance < K.durations[i] - 1e-12:
                assert system.distance(K.centers[i + 1], x_i) == pytest.approx(r_i, rel=1e-6)
                fine = np.linspace(s_i, s[i + 1], 11)[1:-1]
                inside = [system.distance(_demo_state(system, demo, u), x_i) for u in fine]
                assert np.all(np.asarray(inside) < r_i * (1 + 1e-6))


@pytest.mark.parametrize("lipschitz", ["global", "local"])
def test_algorithm1_invariants(case, lipschitz):
    _, system, target, demos, L_H, L, _ = case
    for j, demo in enumerate(demos):
        K = extract_assignments(system, target, demo, j, V0, L_H, L, lipschitz, FLOOR)
        assert isinstance(K, AssignmentSet) and K.dt == DT
        assert np.all(K.demo_ids == j)
        _check_algorithm1(system, target, demo, K, L_H, L, lipschitz, FLOOR)


def test_build_assignment_set_concatenates_in_demo_order(case):
    _, system, target, demos, L_H, L, _ = case
    K = build_assignment_set(system, target, demos, ChainConfig(v0=V0, min_anchor_advance=FLOOR), L_H, L)
    parts = [extract_assignments(system, target, d, j, V0, L_H, L, min_advance=FLOOR) for j, d in enumerate(demos)]
    assert len(K) == sum(len(p) for p in parts)
    np.testing.assert_array_equal(K.demo_ids, np.concatenate([p.demo_ids for p in parts]))
    np.testing.assert_array_equal(K.radii, np.concatenate([p.radii for p in parts]))
    assert len(K.from_demos([0])) == len(parts[0])


def test_build_assignment_set_local_lipschitz_runs(case):
    _, system, target, demos, L_H, L, _ = case
    K_local = build_assignment_set(system, target, demos[:1], ChainConfig(v0=V0, min_anchor_advance=FLOOR, lipschitz="local"), L_H, L)
    K_global = build_assignment_set(system, target, demos[:1], ChainConfig(v0=V0, min_anchor_advance=FLOOR), L_H, L)
    assert len(K_local) > 0 and np.all(K_local.radii > 0)
    # local constants are no larger than the global ones -> radii of the first anchor are no smaller
    assert K_local.radii[0] >= K_global.radii[0] * (1 - 1e-12)


def test_build_assignment_set_rejects_mixed_steps(case):
    _, system, target, demos, L_H, L, _ = case
    d = demos[0]
    other = Demonstration("other", d.states[::2], d.controls[::2][: d.states[::2].shape[0] - 1], 2 * DT, 2 * DT)
    with pytest.raises(ValueError):
        build_assignment_set(system, target, [d, other], ChainConfig(v0=V0, min_anchor_advance=FLOOR), L_H, L)


def test_demo_starting_in_target_gives_empty_set():
    assert len(AssignmentSet.empty(2, DT)) == 0
    cfg, system, target, _, L_H, L, _ = make_case("spring_mass")
    demo = Demonstration("in_target", np.zeros((3, 2)), np.zeros((2, 1)), DT, DT)
    K = extract_assignments(system, target, demo, 0, V0, L_H, L)
    assert len(K) == 0
    assert len(build_assignment_set(system, target, [demo], cfg.chain, L_H, L)) == 0


def test_certified_assignment_set_fits_inside_X(case):
    cfg, system, target, demos, _, _, _ = case
    chain = ChainConfig(v0=V0, min_anchor_advance=FLOOR)
    K, (L_H, L, H_X) = build_certified_assignment_set(system, target, demos, chain, cfg.H_bar)
    assert max_energy_on_balls(system, K.centers, K.radii) <= H_X  # Assumption 1 holds on Supp(K)
    assert (L_H, L) == pytest.approx(system.lipschitz_constants(H_X))
