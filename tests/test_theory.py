"""Tests of the Section III diagnostics (Theorem 2 conditions, Lemma 1, Theorems 3-4)."""

import json
import math

import numpy as np
import pytest

from symplectic_ncp.chain.assignment_set import AssignmentSet
from symplectic_ncp.chain.construction import build_assignment_set, lipschitz_for_demos
from symplectic_ncp.chain.theory import (
    _coverage,
    ball_energy_images,
    component_energy_ranges,
    condition1_empirical,
    demo_decrease_rates,
    state_space_diameter,
    strong_convexity_modulus,
    theory_report,
    vector_field_bound,
)
from symplectic_ncp.config import get_config
from symplectic_ncp.experts.demonstration import Demonstration
from symplectic_ncp.systems import SinglePendulum, SpringMass

DT = 0.005


def energy_shaping_demo(system, target, x0, H_ref, gain, max_time=30.0):
    """u = -gain (H - H_ref) dH/dp (clipped), RK4 at DT, stopped in S_tgt^delta."""
    X, U = [np.asarray(x0, dtype=float)], []
    for _ in range(int(max_time / DT)):
        x = X[-1]
        if target.in_certified_target(x, 1e-3, 0.1)[0]:
            return Demonstration("demo", np.asarray(X), np.asarray(U), DT, DT)
        u = system.clip_control(-gain * (system.hamiltonian(x) - H_ref) * system.grad_hamiltonian(x)[:, 1:2])
        U.append(u[0])
        X.append(system.rk4_step(x, u, DT)[0])
    raise RuntimeError("synthetic demonstration did not reach the target")


@pytest.fixture(scope="module", params=["spring_mass", "single_pendulum"])
def case(request):
    cfg = get_config(request.param)
    system = cfg.make_system()
    target = cfg.make_target(system)
    if request.param == "spring_mass":
        demos = [energy_shaping_demo(system, target, x0, -1.0, 2.0) for x0 in [(2.0, 0.0), (0.0, -2.0), (1.5, -1.0)]]
    else:
        demos = [energy_shaping_demo(system, target, (3.0, 0.0), target.H_plus, 0.5)]
    L_H, L, H_X = lipschitz_for_demos(system, demos, cfg.H_bar)
    K = build_assignment_set(system, target, demos, cfg.chain, L_H, L)
    report = theory_report(system, target, K, demos, cfg, L_H, L, H_X, T1=2.0, T2=3.0)
    return cfg, system, target, demos, K, L_H, L, H_X, report


def test_report_is_strict_json(case):
    report = case[-1]
    text = json.dumps(report, allow_nan=False)
    assert json.loads(text) == report
    for key in ["N", "radii", "durations", "tau_min", "condition1", "condition2", "condition3",
                "lemma1", "theorem2", "theorem3", "theorem4"]:
        assert key in report


def test_condition1_holds_with_global_constants(case):
    *_, K, L_H, L, H_X, report = case
    assert report["N"] == len(K)
    alg, emp = report["condition1"]["algebraic"], report["condition1"]["empirical"]
    assert alg["num_violations"] == 0
    assert emp["num_violations"] == 0 and emp["violation_fraction"] == 0.0
    assert emp["num_triples_checked"] == min(len(K), 300)
    assert emp["num_points"] == emp["num_triples_checked"] * emp["points_per_triple"]
    assert emp["max_margin"] <= 0.0


def test_condition1_check_detects_inflated_radii(case):
    cfg, system, target, _, K, *_ = case
    big = AssignmentSet(K.centers, K.radii * 1e3 + 0.5, K.controls, K.dt, K.demo_ids, K.anchor_times)
    out = condition1_empirical(system, target, big, cfg.chain.v0)
    assert out["num_violations"] > 0 and out["max_margin"] > 0.0


def test_energy_coverage_report(case):
    cfg, system, target, _, K, *_, report = case
    cov = report["condition2"]
    c = report["constants"]["c"]
    assert c == pytest.approx(max(target.H_min, cfg.H_bar - target.H_max), rel=1e-6)
    assert 0.0 <= cov["covered_fraction"] <= 1.0
    assert cov["uncovered_measure"] == pytest.approx((1.0 - cov["covered_fraction"]) * cov["required_measure"])
    names = set(system.component_names().values())
    assert set(report["condition3"]) == names
    if system.name == "spring_mass":
        assert cov["required_intervals"] == [[pytest.approx(target.H_max), pytest.approx(cfg.H_bar)]]
        assert cov["covered_fraction"] > 0.99  # demos start on H = H_bar and end in the band
        assert report["condition3"]["single"]["covered_fraction"] == pytest.approx(cov["covered_fraction"])
    else:
        # one libration swing-up demo: no rotation coverage, partial libration coverage
        assert report["condition3"]["rotation_ccw"]["covered_fraction"] == 0.0
        assert report["condition3"]["rotation_cw"]["num_balls"] == 0
        assert 0.0 < report["condition3"]["libration"]["covered_fraction"] < 1.0
        assert cov["largest_gaps"][0] == [pytest.approx(target.H_max), pytest.approx(cfg.H_bar)]


def test_lemma1_and_theorems(case):
    cfg, system, target, demos, K, L_H, L, H_X, report = case
    lem = report["lemma1"]
    assert lem["rates_below_bound"] and lem["hitting_times_above_bound"]
    assert 0.0 < lem["v_eps_hat"] <= lem["v_eps_median"] <= lem["v_eps_max"] <= lem["v_upper_bound_LH_Cf"]
    tau_min = float(K.durations.min())
    c = report["constants"]["c"]
    assert report["theorem2"]["executions_bound"] == math.ceil(c / (cfg.chain.v0 * tau_min))
    th4 = report["theorem4"]
    D_X = report["constants"]["D_X"]
    T_bar = L_H * D_X / cfg.chain.v0 * (1.0 + 2.0 / tau_min)
    assert th4["T_bar_bound"] == pytest.approx(T_bar)
    assert th4["T_max_bound"] == pytest.approx(T_bar + 3.0)
    th3 = report["theorem3"]
    if system.name == "spring_mass":
        assert th3["applicable"] and th3["alg1_N_le_canonical_bound"]
    else:
        assert th3["applicable"] is False


def test_demo_decrease_rates_match_definition(case):
    cfg, system, target, demos, K, *_ = case
    eps = cfg.energy_eps
    rates, times = demo_decrease_rates(target, K, demos, eps)
    i = 0  # first anchor of demo 0 is its initial state
    dH = target.energy_distance(demos[0].states)
    k_hit = int(np.flatnonzero(dH <= -eps)[0])
    assert times[i] == pytest.approx(k_hit * DT)
    assert rates[i] == pytest.approx((dH[0] + eps) / (k_hit * DT))


def test_constants_spring_mass():
    s = SpringMass()
    # f(x, u) = (p, -q + u) on the disc of radius 2, |u| <= 20: max at q = -2, u = 20
    assert vector_field_bound(s, 2.0) == pytest.approx(22.0, rel=1e-3)
    assert state_space_diameter(s, 2.0) == pytest.approx(4.0, rel=1e-2)
    assert strong_convexity_modulus(s, 2.0) == pytest.approx(1.0)
    assert strong_convexity_modulus(SpringMass(m=2.0, k=3.0), 2.0) == pytest.approx(0.5)
    assert strong_convexity_modulus(SinglePendulum(), 160.0) is None


def test_ball_energy_images_and_components():
    s = SpringMass()
    lo, hi, comp = ball_energy_images(s, np.array([[1.0, 0.0], [0.0, 1.5]]), np.array([0.1, 0.2]))
    np.testing.assert_allclose(lo, [0.9**2 / 2, 1.3**2 / 2], rtol=1e-9)
    np.testing.assert_allclose(hi, [1.1**2 / 2, 1.7**2 / 2], rtol=1e-9)
    np.testing.assert_allclose(comp[0][0], lo)
    p = SinglePendulum()
    q_sep = np.array([[0.0, np.sqrt(2.0 * p.inertia * p.separatrix_energy)]])  # on the separatrix
    lo, hi, comp = ball_energy_images(p, q_sep, np.array([0.05]))
    assert comp[0][1][0] <= p.separatrix_energy <= comp[1][0][0] + 1e-9  # libration below, ccw rotation above
    assert np.isnan(comp[2][0][0])  # never meets the clockwise rotation
    ranges = component_energy_ranges(p, 0.0, 160.0, num_samples=50_000)
    assert ranges[0][-1][1] == pytest.approx(p.separatrix_energy, abs=0.5)
    assert ranges[1][0][0] == pytest.approx(p.separatrix_energy, abs=0.5)
    assert ranges[2][-1][1] == pytest.approx(160.0, abs=0.5)


def test_coverage_intervals():
    out = _coverage([(0.0, 10.0)], [(1.0, 2.0), (1.5, 4.0), (6.0, 7.0)])
    assert out["covered_fraction"] == pytest.approx(0.4)
    assert out["largest_gaps"] == [[7.0, 10.0], [4.0, 6.0], [0.0, 1.0]]
    assert _coverage([], [(0.0, 1.0)])["covered_fraction"] is None
