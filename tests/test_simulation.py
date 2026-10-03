"""Tests of the continuous-time crossing detection and the closed-loop executors."""

from __future__ import annotations

import numpy as np
import pytest

from symplectic_ncp.chain import DEFAULT, AssignmentSet, NonparametricChainPolicy
from symplectic_ncp.simulation import (
    BallIndex,
    RolloutResult,
    earliest_per_row,
    first_entry,
    simulate_chain_policy,
    simulate_feedback_policy,
    unwrap_relative,
)
from symplectic_ncp.simulation.closed_loop import RK4Flow, refine_on_flow
from symplectic_ncp.systems import SinglePendulum, SpringMass
from symplectic_ncp.target import TargetSet

DT = 0.005


# ----------------------------------------------------------------- helpers
def fine_first_entry(system, x0, centers, radii, t_max, dt_fine=2e-4):
    """Brute force under zero input: fine RK4 from every x0 (B, n), exact chord/ball intersection
    between consecutive fine samples; returns the first entry time into each ball, shape (B, M)."""
    X = system.wrap(np.atleast_2d(np.asarray(x0, dtype=float)))
    C = np.atleast_2d(np.asarray(centers, dtype=float))
    R = np.atleast_1d(np.asarray(radii, dtype=float))
    first = np.full((X.shape[0], C.shape[0]), np.inf)
    zero = np.zeros((X.shape[0], system.control_dim))
    for k in range(int(np.ceil(t_max / dt_fine))):
        Xn = system.rk4_step(X, zero, dt_fine)
        m = system.difference(X[:, None, :], C[None, :, :])  # (B, M, n)
        d = (Xn - X)[:, None, :]
        a = np.sum(d * d, axis=2)
        b = np.sum(m * d, axis=2)
        c = np.sum(m * m, axis=2) - R[None, :] ** 2
        disc = b * b - a * c
        tau = (-b - np.sqrt(np.maximum(disc, 0.0))) / a
        hit = (disc >= 0.0) & (tau >= 0.0) & (tau <= 1.0) & ~np.isfinite(first)
        first[hit] = (k + tau[hit]) * dt_fine
        X = system.wrap(Xn)
    return first


def zero_orbit(system, x0, t, dt=1e-3):
    """State of the zero-input flow from x0 at time t (RK4)."""
    x = np.asarray(x0, dtype=float)[None, :]
    n = int(np.ceil(t / dt))
    for _ in range(n):
        x = system.rk4_step(x, np.zeros((1, 1)), t / n)
    return system.wrap(x[0])


def random_arcs(system, rng, B=64, h=DT):
    flow = RK4Flow(system)
    low, high = system.bounding_box(160.0 if isinstance(system, SinglePendulum) else 2.0)
    xa = system.wrap(rng.uniform(low, high, size=(B, 2)))
    U = rng.uniform(-20.0, 20.0, size=(B, 1))
    xb, arc = flow.arc(xa, U, np.full(B, h))
    return xa, U, xb, arc


# ------------------------------------------------------------ crossing.py
def test_hermite_arc_interpolates_the_step():
    sys_ = SinglePendulum()
    rng = np.random.default_rng(0)
    xa, U, xb, arc = random_arcs(sys_, rng)
    rows = np.arange(len(arc))
    pts = arc.points(rows, np.broadcast_to([0.0, 1.0], (len(arc), 2)))
    np.testing.assert_allclose(pts[:, 0], xa, atol=1e-12)
    np.testing.assert_allclose(sys_.wrap(pts[:, 1]), sys_.wrap(xb), atol=1e-12)
    eps = 1e-6
    deriv = (arc.points(rows, np.full((len(arc), 1), eps))[:, 0] - xa) / eps
    np.testing.assert_allclose(deriv, arc.h[:, None] * arc.fa, rtol=1e-4, atol=1e-6)
    # The Hermite arc agrees with the RK4 flow sigma -> RK4(xa, u, sigma h) to O(h^4).
    flow = RK4Flow(sys_)
    for sig in (0.25, 0.5, 0.8):
        x_flow, _ = flow.step(xa, U, sig * arc.h)
        np.testing.assert_allclose(arc.points(rows, np.full((len(arc), 1), sig))[:, 0], x_flow, atol=1e-7)
    # Every arc point stays within the enclosing ball used by the prefilter.
    dense = arc.points(rows, np.broadcast_to(np.linspace(0, 1, 201), (len(arc), 201)))
    dist = np.linalg.norm(dense - arc.midpoint()[:, None, :], axis=2)
    assert np.all(dist <= arc.enclosing_radius()[:, None] + 1e-12)


def test_flow_is_bitwise_identical_to_system_rk4():
    for sys_ in (SpringMass(), SinglePendulum()):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(50, 2)) * 3
        U = rng.uniform(-20, 20, size=(50, 1))
        h = rng.uniform(0.0, DT, size=50)
        xb, fa = RK4Flow(sys_).step(X, U, h)
        assert np.array_equal(xb, sys_.rk4_step(X, U, h))
        assert np.array_equal(fa, sys_.dynamics(X, U))


@pytest.mark.parametrize("system", [SpringMass(), SinglePendulum()])
def test_first_entry_matches_brute_force(system):
    rng = np.random.default_rng(2)
    xa, U, xb, arc = random_arcs(system, rng, B=80)
    B = len(arc)
    rows = np.repeat(np.arange(B), 3)
    s_true = rng.uniform(0.0, 1.0, size=rows.size)
    on_arc = arc.points(rows, s_true[:, None])[:, 0]
    radii = np.exp(rng.uniform(np.log(1e-4), np.log(5e-2), size=rows.size))
    direction = rng.normal(size=(rows.size, 2))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    offset = rng.uniform(0.0, 1.6, size=rows.size)[:, None] * radii[:, None]  # about half of them miss
    centers_wrapped = system.wrap(on_arc + offset * direction)
    centers = unwrap_relative(system, centers_wrapped, xa[rows])
    s = first_entry(arc, rows, centers, radii)

    grid = np.linspace(0.0, 1.0, 100001)
    checked = 0
    for p in range(rows.size):
        pts = arc.points(rows[p : p + 1], grid[None, :])[0]
        g = np.linalg.norm(pts - centers[p], axis=1) / radii[p] - 1.0
        inside = np.flatnonzero(g <= 0.0)
        if inside.size == 0:
            # Brute force sees no entry; only a graze below the sampling resolution may differ.
            assert not np.isfinite(s[p]) or g.min() < 1e-6
            continue
        if g.min() < -1e-4:  # genuine (non-grazing) crossing: must be found accurately
            assert np.isfinite(s[p])
            assert abs(s[p] - grid[inside[0]]) <= 2e-5
            checked += 1
        if inside[0] == 0:
            assert s[p] == 0.0
    assert checked > rows.size // 4


def test_first_entry_across_angle_wrap():
    sys_ = SinglePendulum()
    flow = RK4Flow(sys_)
    xa = np.array([[np.pi - 0.005, 20.0]])  # rotating counter-clockwise through q = pi
    xb, arc = flow.arc(xa, np.zeros((1, 1)), np.array([DT]))
    assert xb[0, 0] > np.pi  # raw step crosses the seam
    target_on_far_side = sys_.wrap(arc.points(np.array([0]), np.array([[0.7]]))[0, 0])
    assert target_on_far_side[0] < 0.0
    centers = unwrap_relative(sys_, target_on_far_side[None, :], xa)
    s = first_entry(arc, np.array([0]), centers, np.array([1e-3]))
    assert 0.6 < s[0] < 0.7


def test_ball_index_pairs_are_exact():
    sys_ = SinglePendulum()
    rng = np.random.default_rng(3)
    N = 600
    centers = sys_.wrap(rng.uniform([-np.pi, -30], [np.pi, 30], size=(N, 2)))
    radii = np.exp(rng.uniform(np.log(1e-4), np.log(0.05), size=N))
    radii[:10] = rng.uniform(0.2, 1.0, size=10)
    index = BallIndex(sys_, centers, radii)
    pts = sys_.wrap(rng.uniform([-np.pi, -30], [np.pi, 30], size=(300, 2)))
    pts[:20, 0] = np.pi - 1e-3  # near the seam
    q = rng.uniform(0.0, 0.3, size=300)
    pr, pb = index.pairs(pts, q)
    got = set(zip(pr.tolist(), pb.tolist()))
    d = sys_.distance(pts[:, None, :], centers[None, :, :])
    want = set(zip(*np.nonzero(d <= q[:, None] + radii[None, :])))
    assert got == {(int(a), int(b)) for a, b in want}


def test_earliest_per_row():
    s, lab = earliest_per_row(4, [0, 2, 0, 2, 3], [0.5, np.inf, 0.2, 0.7, np.inf], [10, 11, 12, 13, 14])
    np.testing.assert_array_equal(s, [0.2, np.inf, 0.7, np.inf])
    np.testing.assert_array_equal(lab, [12, -1, 13, -1])


def test_refine_on_flow_lands_on_the_boundary():
    sys_ = SinglePendulum()
    rng = np.random.default_rng(4)
    xa, U, xb, arc = random_arcs(sys_, rng, B=100)
    rows = np.arange(100)
    s_true = rng.uniform(0.1, 0.9, size=100)
    radii = np.exp(rng.uniform(np.log(1e-4), np.log(1e-2), size=100))
    centers = arc.points(rows, s_true[:, None])[:, 0] + np.array([0.0, 0.5]) * radii[:, None]
    s = first_entry(arc, rows, centers, radii)
    assert np.all(np.isfinite(s))
    starts_inside = np.linalg.norm(xa - centers, axis=1) <= radii
    assert np.all(s[starts_inside] == 0.0)
    s_ref, x_e = refine_on_flow(RK4Flow(sys_), xa, U, arc.h, centers, radii, s)
    rho = np.linalg.norm(x_e - centers, axis=1) / radii
    assert np.max(np.abs(rho[~starts_inside] - 1.0)) < 1e-8


# ------------------------------------------------------- feedback executor
def test_feedback_target_entry_time_matches_fine_simulation():
    sys_ = SinglePendulum()
    x0 = np.array([[0.3, 18.0], [-2.0, -25.0], [1.0, 2.0], [3.0, 20.0]])
    times = np.array([0.37, 0.81, 0.6, 0.2])
    for i in range(4):
        center = zero_orbit(sys_, x0[i], times[i]) + np.array([0.0, 2e-3])
        target = TargetSet(sys_, center, 5e-3)
        res = simulate_feedback_policy(sys_, target, lambda X: np.zeros((X.shape[0], 1)), x0[i], 1.0, DT, 0.02)
        ref = fine_first_entry(sys_, x0[i], center, 5e-3, times[i] + 0.05)[0, 0]
        assert res.success[0]
        assert abs(res.reach_time[0] - ref) <= 1e-4 * ref


def test_feedback_zoh_clipping_and_call_period():
    sys_ = SpringMass()
    target = TargetSet(sys_, np.array([50.0, 50.0]), 0.1)  # never reached
    calls = []

    def policy(X):
        calls.append(X.shape[0])
        return np.full((X.shape[0], 1), 1000.0)  # clipped to u_max = 20

    x0 = np.array([[1.0, 0.0], [0.0, 1.0]])
    res = simulate_feedback_policy(sys_, target, policy, x0, horizon=0.1, dt=DT, control_period=0.02)
    assert len(calls) == 5 and not res.success.any()
    x = x0.copy()
    for _ in range(20):
        x = sys_.rk4_step(x, np.full((2, 1), 20.0), DT)
    np.testing.assert_allclose(res.extras["final_energy"], sys_.hamiltonian(x), rtol=1e-12)


def test_feedback_pd_law_reaches_target():
    sys_ = SpringMass()
    target = TargetSet(sys_, np.zeros(2), 0.1)
    rng = np.random.default_rng(5)
    x0 = sys_.sample_energy_sublevel(100, 2.0, rng)
    res = simulate_feedback_policy(sys_, target, lambda X: -2.0 * X[:, :1] - 3.0 * X[:, 1:], x0, 20.0, DT, 0.02)
    assert isinstance(res, RolloutResult)
    assert res.success.all() and np.all(res.reach_time < 10.0)
    inside0 = target.contains(x0)
    assert np.all(res.reach_time[inside0] == 0.0)


# ---------------------------------------------------------- chain executor
def braking_snippet(system, x_start, steps, gain=3.0):
    """Open-loop snippet recorded from the damping law u = -gain p started at x_start."""
    x, us = np.asarray(x_start, dtype=float)[None, :], []
    for _ in range(steps):
        u = system.clip_control(-gain * x[:, 1:])
        us.append(u[0])
        x = system.rk4_step(x, u, DT)
    return np.array(us)


def test_chain_policy_support_entry_and_execution():
    sys_ = SpringMass()
    target = TargetSet(sys_, np.zeros(2), 0.1)
    x0 = np.array([[1.0, 0.0]])  # q = cos t, p = -sin t
    c0 = np.array([0.0, -1.0])  # reached at t = pi / 2
    snippet = braking_snippet(sys_, c0, 1600, gain=1.5)
    # A second ball sitting on the braking path: ignored while the snippet runs (Remark 2).
    path = [c0[None, :]]
    for u in snippet[:60]:
        path.append(sys_.rk4_step(path[-1], u[None, :], DT))
    K = AssignmentSet(
        centers=np.vstack([c0, path[40][0]]),
        radii=np.array([0.05, 0.05]),
        controls=[snippet, np.zeros((5, 1))],
        dt=DT,
    )
    policy = NonparametricChainPolicy(K, sys_, membership_tol=1e-6)
    res = simulate_chain_policy(sys_, target, policy, x0, 20.0, DT)
    entry = fine_first_entry(sys_, x0[0], c0, 0.05, 2.0)[0, 0]
    assert res.success[0]
    assert res.extras["n_snippets"][0] == 1
    assert res.extras["fallback_entries"][0] == 0
    assert abs(res.extras["max_idle_interval"][0] - entry) <= 1e-4 * entry
    np.testing.assert_allclose(res.extras["idle_time"][0], entry, rtol=1e-4)
    # The target is entered during the snippet, which ends the episode before tau_1 = 8 s.
    assert res.extras["exec_time"][0] < len(snippet) * DT
    np.testing.assert_allclose(res.reach_time[0], res.extras["idle_time"][0] + res.extras["exec_time"][0])

    # Brute force of the concatenated signal u(t) = 0 on [0, t_e), snippet(t - t_e) afterwards.
    n1 = 4000
    x = x0.copy()
    for _ in range(n1):
        x = sys_.rk4_step(x, np.zeros((1, 1)), entry / n1)
    sub, t_ref = 10, np.inf
    for k, u in enumerate(snippet):
        for j in range(sub):
            xn = sys_.rk4_step(x, u[None, :], DT / sub)
            d = xn - x
            a, b, c = np.sum(d * d), np.sum(x * d), np.sum(x * x) - 0.1**2
            disc = b * b - a * c
            if disc >= 0 and 0.0 <= (-b - np.sqrt(disc)) / a <= 1.0:
                t_ref = entry + (k * sub + j + (-b - np.sqrt(disc)) / a) * DT / sub
                break
            x = xn
        if np.isfinite(t_ref):
            break
    assert abs(res.reach_time[0] - t_ref) <= 1e-4 * t_ref


def test_chain_policy_detects_tiny_balls_between_samples():
    sys_ = SinglePendulum()
    target = TargetSet(sys_, np.array([np.pi, 0.0]), 0.1)
    rng = np.random.default_rng(6)
    x0 = np.array([[0.5, 15.0], [-1.0, -20.0], [0.2, 5.0]])
    entry_times = rng.uniform(0.3, 1.0, size=3)
    centers, radii = [], []
    for i in range(3):
        on = zero_orbit(sys_, x0[i], entry_times[i])
        r = 1e-4
        centers.append(on + rng.normal(size=2) * 0.3 * r)
        radii.append(r)
    centers = np.array(centers)
    K = AssignmentSet(centers, np.array(radii), [np.zeros((3, 1))] * 3, DT)
    policy = NonparametricChainPolicy(K, sys_, membership_tol=1e-6)
    res = simulate_chain_policy(sys_, target, policy, x0, 1.2, DT)
    ref = fine_first_entry(sys_, x0, centers, np.array(radii), 1.2).min(axis=1)
    assert np.all(np.isfinite(ref))
    assert np.all(res.extras["n_snippets"] >= 1)
    np.testing.assert_allclose(res.extras["max_idle_interval"], ref, rtol=1e-4)
    assert np.all(res.extras["fallback_entries"] == 0)


def test_chain_policy_zero_input_conserves_energy_and_initial_selection():
    sys_ = SpringMass()
    target = TargetSet(sys_, np.zeros(2), 0.1)
    K = AssignmentSet(np.array([[10.0, 0.0]]), np.array([0.5]), [np.ones((4, 1))], DT)
    policy = NonparametricChainPolicy(K, sys_)
    rng = np.random.default_rng(7)
    x0 = sys_.sample_energy_sublevel(64, 2.0, rng)
    x0 = x0[~target.contains(x0)]
    x0 = np.vstack([x0, [[10.1, 0.0]], [[0.01, 0.0]]])  # one starts inside Supp(K), one inside S_tgt
    res = simulate_chain_policy(sys_, target, policy, x0, 20.0, DT)
    idle = slice(0, x0.shape[0] - 2)
    assert not res.success[idle].any()
    np.testing.assert_allclose(res.extras["final_energy"][idle], sys_.hamiltonian(x0[idle]), rtol=1e-8)
    np.testing.assert_allclose(res.extras["idle_time"][idle], 20.0, rtol=1e-12)
    assert np.all(res.extras["n_snippets"][idle] == 0)
    assert res.extras["n_snippets"][-2] >= 1  # pi_K(x0) selected the snippet at t = 0
    assert res.success[-1] and res.reach_time[-1] == 0.0
    np.testing.assert_allclose(res.reach_time_or_horizon[idle], 20.0)


def test_chain_policy_rejects_mismatched_step_and_handles_partial_horizon():
    sys_ = SpringMass()
    target = TargetSet(sys_, np.zeros(2), 0.1)
    K = AssignmentSet(np.array([[1.0, 0.0]]), np.array([0.05]), [np.zeros((2, 1))], 0.01)
    with pytest.raises(ValueError):
        simulate_chain_policy(sys_, target, NonparametricChainPolicy(K, sys_), np.array([[1.0, 1.0]]), 1.0, DT)
    K = AssignmentSet(np.array([[5.0, 5.0]]), np.array([0.05]), [np.zeros((2, 1))], DT)
    res = simulate_chain_policy(sys_, target, NonparametricChainPolicy(K, sys_), np.array([[1.0, 1.0]]), 0.0123, DT)
    np.testing.assert_allclose(res.extras["idle_time"], 0.0123, rtol=1e-12)
    assert DEFAULT == -1


def test_chain_policy_first_support_entry_matches_brute_force_many_balls():
    """Zero snippets keep the zero-input flow, so the first Supp(K) entry must match brute force:
    a horizon just after it sees one snippet, a horizon just before it sees none."""
    sys_ = SinglePendulum()
    target = TargetSet(sys_, np.array([np.pi, 0.0]), 0.1)
    rng = np.random.default_rng(8)
    B, per_env = 12, 12
    x0 = sys_.wrap(sys_.sample_energy_sublevel(B, 160.0, rng))
    x0 = x0[~target.contains(x0)]
    B = x0.shape[0]
    grid_steps = 800  # orbit samples every 1 ms up to 0.8 s
    orbit = [x0]
    for _ in range(grid_steps):
        orbit.append(sys_.rk4_step(orbit[-1], np.zeros((B, 1)), 1e-3))
    orbit = np.stack(orbit, axis=1)  # (B, grid_steps + 1, 2)
    idx = rng.integers(150, grid_steps + 1, size=(B, per_env))
    pts = orbit[np.arange(B)[:, None], idx].reshape(-1, 2)
    radii = np.exp(rng.uniform(np.log(1e-4), np.log(2e-2), size=pts.shape[0]))
    direction = rng.normal(size=pts.shape)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    centers = sys_.wrap(pts + rng.uniform(0.0, 1.5, size=(pts.shape[0], 1)) * radii[:, None] * direction)
    K = AssignmentSet(centers, radii, [np.zeros((5, 1))] * centers.shape[0], DT)
    policy = NonparametricChainPolicy(K, sys_, membership_tol=1e-6)
    ref = fine_first_entry(sys_, x0, centers, radii, 0.85).min(axis=1)
    hit = np.isfinite(ref) & ~policy.in_support(x0) & (ref > 0.01)
    assert hit.sum() >= B // 2
    for i in np.flatnonzero(hit):
        late = simulate_chain_policy(sys_, target, policy, x0[i], ref[i] * (1.0 + 1e-4), DT)
        early = simulate_chain_policy(sys_, target, policy, x0[i], ref[i] * (1.0 - 1e-4), DT)
        assert late.extras["n_snippets"][0] == 1 and early.extras["n_snippets"][0] == 0
        assert abs(late.extras["max_idle_interval"][0] - ref[i]) <= 1e-4 * ref[i]
        assert late.extras["fallback_entries"][0] == 0


def test_chain_policy_applies_the_snippet_lead():
    """A snippet anchored between grid points: its first ZOH step lasts lead_i < dt (AssignmentSet)."""
    sys_ = SpringMass()
    target = TargetSet(sys_, np.array([-5.0, -5.0]), 0.1)  # far away: no target event
    x0 = np.array([[1.0, 0.0]])
    controls = np.array([[5.0], [-7.0], [3.0]])
    lead = 0.3 * DT
    K = AssignmentSet(np.array([[1.0, 0.0]]), np.array([0.01]), [controls], DT, leads=[lead])
    tau = float(K.durations[0])
    assert tau == pytest.approx(lead + 2 * DT)
    res = simulate_chain_policy(sys_, target, NonparametricChainPolicy(K, sys_), x0, tau, DT)
    x = x0.copy()
    for u, h in zip(controls, [lead, DT, DT]):
        x = sys_.rk4_step(x, u[None, :], h)
    assert res.extras["n_snippets"][0] == 1
    np.testing.assert_allclose(res.extras["exec_time"][0], tau, rtol=1e-12)
    np.testing.assert_allclose(res.extras["final_energy"][0], sys_.hamiltonian(x)[0], rtol=1e-12)
    # applying full dt steps instead would give a different end state
    y = x0.copy()
    for u in controls:
        y = sys_.rk4_step(y, u[None, :], DT)
    assert abs(sys_.hamiltonian(y)[0] - sys_.hamiltonian(x)[0]) > 1e-6


def test_chain_policy_trace_is_consistent_with_the_rollout():
    sys_ = SpringMass()
    target = TargetSet(sys_, np.zeros(2), 0.1)
    x0 = np.array([[1.0, 0.0], [0.0, -1.2]])
    c0 = np.array([0.0, -1.0])
    K = AssignmentSet(np.array([c0]), np.array([0.05]), [braking_snippet(sys_, c0, 1600, gain=1.5)], DT)
    policy = NonparametricChainPolicy(K, sys_)
    res = simulate_chain_policy(sys_, target, policy, x0, 20.0, DT, record=True)
    plain = simulate_chain_policy(sys_, target, policy, x0, 20.0, DT)
    np.testing.assert_array_equal(res.reach_time, plain.reach_time)  # recording does not change the rollout
    for b, tr in enumerate(res.trace):
        np.testing.assert_allclose(tr["x"][0], x0[b])
        assert np.all(np.diff(tr["t"]) > 0.0)
        if res.success[b]:
            assert tr["t"][-1] == pytest.approx(res.reach_time[b])
            assert target.contains(tr["x"][-1])[0] or target.distance(tr["x"][-1])[0] == pytest.approx(0.1)
        idle = tr["snippet"][1:] == DEFAULT
        assert np.all(tr["u"][1:][idle] == 0.0)  # zero input outside the snippets
    assert plain.trace is None


# ------------------------------------------- feedback executor: action chunks
def test_feedback_chunk_of_length_one_is_bitwise_identical_to_2d():
    sys_ = SinglePendulum()
    target = TargetSet(sys_, np.array([np.pi, 0.0]), 0.1)
    x0 = sys_.wrap(np.random.default_rng(7).uniform([-np.pi, -20.0], [np.pi, 20.0], size=(40, 2)))

    def law(X):
        return -3.0 * np.sin(X[:, :1]) - 0.5 * X[:, 1:] + 25.0 * np.cos(X[:, :1])  # saturates sometimes

    a = simulate_feedback_policy(sys_, target, law, x0, 3.0, DT, 0.02, record=True)
    b = simulate_feedback_policy(sys_, target, lambda X: law(X)[:, None, :], x0, 3.0, DT, 0.02, record=True)
    assert np.array_equal(a.success, b.success) and np.array_equal(a.reach_time, b.reach_time)
    assert np.array_equal(a.extras["final_energy"], b.extras["final_energy"])
    for ta, tb in zip(a.trace, b.trace):
        assert np.array_equal(ta["u"], tb["u"]) and np.array_equal(ta["x"], tb["x"])


def test_feedback_chunk_schedule_clipping_and_queries():
    sys_ = SpringMass()
    target = TargetSet(sys_, np.array([50.0, 50.0]), 0.1)  # never reached
    calls = []

    def policy(X):
        calls.append(X.shape[0])
        k = len(calls)
        # chunk (B, 3, 1): elements 5k + j, the last one far above u_max (clipped to 20).
        chunk = np.array([5.0 * k, 5.0 * k + 1.0, 1000.0])
        return np.broadcast_to(chunk[None, :, None], (X.shape[0], 3, 1)).copy()

    x0 = np.array([[1.0, 0.0], [0.0, 1.0]])
    res = simulate_feedback_policy(sys_, target, policy, x0, horizon=0.2, dt=DT, control_period=0.02, record=True)
    # 10 control periods, a query every 3 periods: at periods 0, 3, 6, 9.
    assert calls == [2, 2, 2, 2]
    per_period = np.array([5.0, 6.0, 20.0, 10.0, 11.0, 20.0, 15.0, 16.0, 20.0, 20.0])
    expected_u = np.repeat(per_period, 4)
    for tr in res.trace:
        np.testing.assert_array_equal(tr["u"][1:, 0], expected_u)
    x = x0.copy()
    for u in expected_u:
        x = sys_.rk4_step(x, np.full((2, 1), u), DT)
    np.testing.assert_allclose(res.extras["final_energy"], sys_.hamiltonian(x), rtol=1e-12)


def test_feedback_chunk_positions_survive_shrinking_running_set():
    sys_ = SpringMass()
    # The first trajectory starts next to the target and enters it during its first chunk.
    target = TargetSet(sys_, np.array([0.0, 0.0]), 0.1)
    x0 = np.array([[0.0, -0.12], [1.5, 0.0], [0.0, -1.5]])
    calls = []

    def policy(X):
        calls.append(X.copy())
        # Ta = 5 copies of a PD law evaluated at the query state (open loop for 5 periods).
        u = sys_.clip_control(-2.0 * X[:, :1] - 3.0 * X[:, 1:])
        return np.repeat(u[:, None, :], 5, axis=1)

    res = simulate_feedback_policy(sys_, target, policy, x0, horizon=0.3, dt=DT, control_period=0.02, record=True)
    assert res.success[0] and 0.0 < res.reach_time[0] < 0.1
    assert not res.success[1:].any()
    # Queries at periods 0, 5, 10: the first one on all three, then only on the two still running.
    assert [c.shape[0] for c in calls] == [3, 2, 2]
    # Reference: the two remaining trajectories, simulated by hand with the same chunks.
    x = x0[1:].copy()
    for period in range(15):
        if period % 5 == 0:
            u = sys_.clip_control(-2.0 * x[:, :1] - 3.0 * x[:, 1:])
            np.testing.assert_allclose(calls[period // 5][-2:], x, rtol=0, atol=0)
        for _ in range(4):
            x = sys_.rk4_step(x, u, DT)
    np.testing.assert_allclose(res.extras["final_energy"][1:], sys_.hamiltonian(x), rtol=1e-12)


def test_feedback_chunk_rejects_bad_shape():
    sys_ = SpringMass()
    target = TargetSet(sys_, np.array([50.0, 50.0]), 0.1)
    with pytest.raises(ValueError):
        simulate_feedback_policy(
            sys_, target, lambda X: np.zeros((X.shape[0] + 1, 4, 1)), np.array([[1.0, 0.0]]), 0.1, DT, 0.02
        )
