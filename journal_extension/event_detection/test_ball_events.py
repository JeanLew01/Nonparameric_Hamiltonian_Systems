"""Focused numerical regressions; no reference-repository code is executed."""
import numpy as np

from .ball_events import (
    BallIndex, HermiteArc, first_entry, interpolation_padding,
    support_candidates, verified_first_entry,
)


def arc_from_polynomial(coefficients):
    a = np.asarray(coefficients, float)
    xa = a[0]
    xb = a.sum(axis=0)
    fa = a[1]
    fb = a[1]+2*a[2]+3*a[3]
    return HermiteArc(xa, xb, fa, fb, np.ones(1))


def evaluator(arc):
    def evaluate(row, s):
        a = arc.coeffs[row]
        return a[0]+s*(a[1]+s*(a[2]+s*a[3]))
    return evaluate


def test_fixed_newton_counterexample():
    c = 1/32
    arc = arc_from_polynomial([[c*c, 0], [-2*c, 0], [1, 0], [0, 0]])
    result = first_entry(arc, [0], [[0, 0]], [1e-6])
    assert abs(result[0]-(c-.001)) < 2e-11
    # The reference's four Newton iterations leave distance 1/26244.
    assert (c*(2/3)**4)**2 > 1e-6
    actual = verified_first_entry(arc, [0], [[0, 0]], [1e-6], evaluator(arc))
    assert abs(actual.s[0]-(c-.001)) < 2e-11
    assert np.linalg.norm(actual.states[0]) <= 1e-6


def test_crossing_with_both_step_endpoints_outside():
    arc = arc_from_polynomial([[-2, 0], [4, 0], [0, 0], [0, 0]])
    for radius in [1., .001, 1e-9]:
        result = verified_first_entry(arc, [0], [[0, 0]], [radius], evaluator(arc))
        assert abs(result.s[0]-(2-radius)/4) < 2e-12
        assert np.linalg.norm(result.states[0]) <= radius


def test_inside_grazing_and_no_hit():
    arc = arc_from_polynomial([[-1, 1], [2, 0], [0, 0], [0, 0]])
    result = first_entry(arc, [0, 0, 0], [[0, 0], [0, -.01], [-1, 1]], [1, 1, .1])
    assert abs(result[0]-.5) < 3e-8
    assert np.isinf(result[1])
    assert result[2] == 0
    actual = verified_first_entry(arc, [0], [[0, 0]], [1.], evaluator(arc))
    assert np.linalg.norm(actual.states[0]) <= 1.


def test_periodic_angle_crossing():
    xa, xb = np.array([[np.pi-.02, 0.]]), np.array([[-np.pi+.02, 0.]])
    arc = HermiteArc(xa, xb, [[.04, 0]], [[.04, 0]], [1.], angle_indices=(0,))
    index = BallIndex([[-np.pi, 0]], [.005], angle_indices=(0,))
    rows, balls, centers = support_candidates(index, arc)
    assert rows.tolist() == [0] and balls.tolist() == [0]
    result = verified_first_entry(arc, rows, centers, index.radii[balls], evaluator(arc))
    assert abs(result.s[0]-.375) < 1e-11
    assert np.linalg.norm(result.states[0]-centers[0]) <= .005


def test_no_outside_ball_fallback():
    arc = arc_from_polynomial([[-1, 0], [2, 0], [0, 0], [0, 0]])
    base = evaluator(arc)
    shifted = lambda row, s: base(row, s)+[0, .02]
    result = verified_first_entry(arc, [0], [[0, 0]], [.01], shifted)
    assert np.isinf(result.s[0])
    assert np.isnan(result.states[0]).all()
    assert result.diagnostics['rejected_subarcs'] > 0


def test_padding_recovers_displaced_actual_crossing():
    arc = arc_from_polynomial([[-1, .011], [2, 0], [0, 0], [0, 0]])
    base = evaluator(arc)
    actual = lambda row, s: base(row, s)-[0, .002]
    assert np.isinf(first_entry(arc, [0], [[0, 0]], [.01])[0])
    result = verified_first_entry(arc, [0], [[0, 0]], [.01], actual, interpolation_padding=.0021)
    expected = (1-np.sqrt(.01**2-.009**2))/2
    assert abs(result.s[0]-expected) < 2e-11
    assert np.linalg.norm(result.states[0]) <= .01


def _rk4(x, h):
    def f(y):
        return np.array([y[1], -y[0]])
    a = f(x); b = f(x+h*a/2); c = f(x+h*b/2); d = f(x+h*c)
    return x+h*(a+2*b+2*c+d)/6


def oscillator_entry(h):
    x = np.array([1., 0.]); t = 0.
    center = np.array([0., -1.]); radius = .05
    for _ in range(int(np.ceil(2/h))):
        y = _rk4(x, h)
        arc = HermiteArc(x, y, [x[1], -x[0]], [y[1], -y[0]], [h])
        actual = lambda row, s: _rk4(x, s*h)
        pad = interpolation_padding(arc, actual)
        event = verified_first_entry(arc, [0], [center], [radius], actual,
                                     interpolation_padding=pad)
        if np.isfinite(event.s[0]):
            assert np.linalg.norm(event.states[0]-center) <= radius
            return t+h*event.s[0]
        x, t = y, t+h
    raise AssertionError('oscillator target event missed')


def test_timestep_convergence_to_exact_oscillator_entry():
    exact = np.pi/2-2*np.arcsin(.05/2)
    errors = [abs(oscillator_entry(h)-exact) for h in [.2, .1, .05]]
    assert errors[1] < errors[0]/8
    assert errors[2] < errors[1]/8
    assert errors[-1] < 1e-7


def test_large_library_prefilter_retains_known_crossing():
    rng = np.random.default_rng(81)
    centers = rng.uniform(-20, 20, (50000, 4))
    centers[0] = 0.
    radii = np.full(50000, 1e-4)
    index = BallIndex(centers, radii, angle_indices=())
    arc = arc_from_polynomial([[-.001, 0, 0, 0], [.002, 0, 0, 0],
                               [0, 0, 0, 0], [0, 0, 0, 0]])
    rows, balls, lifted = support_candidates(index, arc)
    assert 0 in balls
    assert len(balls) < 10
    entry = first_entry(arc, rows, lifted, radii[balls])
    assert abs(entry[np.flatnonzero(balls == 0)[0]]-.45) < 2e-11
