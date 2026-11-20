"""Ball entry on cubic arcs, followed by membership in an actual numerical flow.

This is numerical event localization, not a validated ODE certificate.
Every stationary point of the cubic's squared distance is considered;
there is no fixed-iteration Newton approximation to closest approach.
Conservative chord filters enclose the *cubic*, not the true ODE solution.
The verified API checks actual numerical-flow membership before returning
an event. Its optional interpolation padding is an explicit practical
error allowance, not a rigorously validated interpolation bound.
"""
from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial import polynomial as poly
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree


def unwrap_relative(points, reference, angle_indices=()):
    delta = np.asarray(points, dtype=float) - np.asarray(reference, dtype=float)
    for j in angle_indices:
        delta[..., j] -= 2 * np.pi * np.round(delta[..., j] / (2 * np.pi))
    return np.asarray(reference, dtype=float) + delta


def _norm(x):
    return np.sqrt(np.sum(np.asarray(x) ** 2, axis=-1))


@dataclass
class HermiteArc:
    xa: np.ndarray
    xb: np.ndarray
    fa: np.ndarray
    fb: np.ndarray
    h: np.ndarray
    angle_indices: tuple = ()
    coeffs: np.ndarray = field(init=False)
    chord_deviation: np.ndarray = field(init=False)

    def __post_init__(self):
        self.xa = np.atleast_2d(np.asarray(self.xa, float))
        self.xb = unwrap_relative(np.atleast_2d(self.xb), self.xa, self.angle_indices)
        self.fa, self.fb = np.atleast_2d(self.fa), np.atleast_2d(self.fb)
        self.h = np.broadcast_to(self.h, (len(self.xa),)).astype(float)
        dx = self.xb - self.xa
        ha, hb = self.h[:, None] * self.fa, self.h[:, None] * self.fb
        self.coeffs = np.stack((self.xa, ha, 3 * dx - 2 * ha - hb,
                                ha + hb - 2 * dx), axis=1)
        a2, a3 = self.coeffs[:, 2], self.coeffs[:, 3]
        self.chord_deviation = _norm(np.maximum(abs(2*a2), abs(2*a2+6*a3))) / 8

    def __len__(self):
        return len(self.xa)

    def points(self, rows, s):
        a = self.coeffs[np.asarray(rows), None, :, :]
        t = np.asarray(s)[..., None]
        return a[..., 0, :] + t*(a[..., 1, :] + t*(a[..., 2, :] + t*a[..., 3, :]))

    def second_derivative(self, rows, s):
        a = self.coeffs[np.asarray(rows)]
        return 2*a[:, None, 2] + 6*np.asarray(s)[..., None]*a[:, None, 3]

    def midpoint(self):
        return (self.xa + self.xb) / 2

    def enclosing_radius(self):
        return _norm(self.xb-self.xa)/2 + self.chord_deviation


def _value(a, t):
    return a[0] + t*(a[1] + t*(a[2] + t*a[3]))


def _segment_distance(start, delta):
    length2 = np.sum(delta*delta, axis=-1)
    t = np.clip(-np.sum(start*delta, axis=-1)/np.where(length2>0, length2, 1), 0, 1)
    return _norm(start + t[..., None]*delta)


def _candidate_intervals(arc, rows, centers, radii, num_chords, padding):
    """Return pair indices and conservative candidate subarc endpoints."""
    S = int(num_chords)
    if S < 1:
        raise ValueError('num_chords must be positive')
    rows = np.asarray(rows, int)
    radii = np.asarray(radii, float)
    padding = np.broadcast_to(padding, radii.shape)
    if np.any(radii < 0) or np.any(padding < 0):
        raise ValueError('radii and interpolation padding must be nonnegative')
    chord = _segment_distance(arc.xa[rows]-centers, arc.xb[rows]-arc.xa[rows])
    keep = np.flatnonzero(chord <= radii+padding+arc.chord_deviation[rows])
    if not len(keep):
        return np.empty(0, int), np.empty(0), np.empty(0)
    unique, inverse = np.unique(rows[keep], return_inverse=True)
    grid = np.broadcast_to(np.linspace(0, 1, S+1), (len(unique), S+1))
    points = arc.points(unique, grid)
    acc = abs(arc.second_derivative(unique, grid))
    deviation = _norm(np.maximum(acc[:, :-1], acc[:, 1:]))/(8*S*S)
    rel = points[inverse]-centers[keep, None, :]
    distance = _segment_distance(rel[:, :-1], np.diff(rel, axis=1))
    pair, interval = np.nonzero(distance <= (radii[keep]+padding[keep])[:, None]+deviation[inverse])
    return keep[pair], interval/S, (interval+1)/S


def _stationary_knots(a, center, lo, hi):
    """All numerically real stationary roots, scaled onto a unit subinterval.

    Squared distance has degree at most six. Local scaling reduces the
    conditioning problems from powers of a small subinterval width.
    Direct distance evaluation is used after companion-matrix root finding.
    """
    width = hi-lo
    b = np.stack((_value(a, lo)-center,
                  width*(a[1]+2*lo*a[2]+3*lo*lo*a[3]),
                  width*width*(a[2]+3*lo*a[3]), width**3*a[3]))
    scale = float(np.max(abs(b)))
    if scale == 0:
        return np.array([lo, hi])
    b /= scale
    squared = np.zeros(7)
    for j in range(b.shape[1]):
        term = poly.polymul(b[:, j], b[:, j])
        squared[:len(term)] += term
    derivative = poly.polyder(squared)
    # Only exact trailing zeros are trimmed; small high-order terms remain.
    derivative = poly.polytrim(derivative)
    if len(derivative) <= 1:
        return np.array([lo, hi])
    roots = poly.polyroots(derivative/np.max(abs(derivative)))
    # Near-real roots can arise from repeated stationary points. Their real
    # parts are harmless extra knots; subsequent membership checks decide.
    real = roots.real[abs(roots.imag) <= 1e-5*(1+abs(roots.real))]
    real = real[(real > -1e-10) & (real < 1+1e-10)]
    return lo+width*np.unique(np.r_[0., np.clip(real, 0, 1), 1.])


def _inside_bisection(gap, outside, inside, tolerance):
    """Retain an actually inside upper bracket, not an unverified secant root."""
    if gap(outside) <= 0:
        return outside
    if gap(inside) > 0:
        raise ValueError('entry bracket has no inside endpoint')
    for _ in range(80):
        if inside-outside <= tolerance:
            break
        mid = (outside+inside)/2
        if gap(mid) <= 0:
            inside = mid
        else:
            outside = mid
    return inside


def first_entry(arc, rows, centers, radii, num_chords=16,
                newton_iters=None, bisection_iters=None, parameter_tolerance=1e-12):
    """Drop-in cubic-arc entry; uses polynomial stationary roots, not Newton.

    Centers must be in the arc's unwrapped coordinates. Infinity denotes no
    detected intersection. The unused iteration keywords allow old callers
    to migrate without preserving their fixed-iteration Newton behavior.
    """
    rows, centers, radii = np.asarray(rows, int), np.asarray(centers, float), np.asarray(radii, float)
    out = np.full(len(rows), np.inf)
    pair, left, right = _candidate_intervals(arc, rows, centers, radii, num_chords, 0.)
    for j, lo, hi in zip(pair, left, right):
        if lo >= out[j]:
            continue
        a, c, r = arc.coeffs[rows[j]], centers[j], radii[j]
        knots = _stationary_knots(a, c, lo, hi)
        def gap(t):
            return float(_norm(_value(a, t)-c)-r)
        for k, t in enumerate(knots):
            if gap(t) <= 0:
                hit = t if k == 0 else _inside_bisection(gap, knots[k-1], t, parameter_tolerance)
                out[j] = min(out[j], hit)
                break
    return out


@dataclass
class EventBatch:
    s: np.ndarray
    states: np.ndarray
    diagnostics: dict


def verified_first_entry(arc, rows, centers, radii, evaluate, num_chords=16,
                         interpolation_padding=0., parameter_tolerance=1e-12):
    """Locate entry and require membership in ``evaluate(row,s)``.

    ``evaluate`` returns an unwrapped state of the actual numerical flow
    used by the caller (partial RK4, or dense ODE output). It must agree
    with the arc's local coordinates. Returned finite events always have
    distance <= the requested radius, without an outside-support fallback.

    Stationary knots of the Hermite distance partition each candidate
    subarc. A bounded closest-approach search in those small intervals
    catches displacement of a minimum between the interpolant and the
    actual numerical flow. This extra search and finite-precision polynomial
    roots are numerical safeguards, not a global root-isolation certificate
    for a general nonlinear flow. Audit with step refinement.
    """
    rows, centers, radii = np.asarray(rows, int), np.asarray(centers, float), np.asarray(radii, float)
    out = np.full(len(rows), np.inf)
    states = np.full((len(rows), arc.xa.shape[1]), np.nan)
    pair, left, right = _candidate_intervals(arc, rows, centers, radii, num_chords, interpolation_padding)
    calls = rejected = searches = 0
    for j, lo, hi in zip(pair, left, right):
        if lo >= out[j]:
            continue
        a, c, r, row = arc.coeffs[rows[j]], centers[j], radii[j], int(rows[j])
        cache = {}
        def state(t):
            nonlocal calls
            key = float(t)
            if key not in cache:
                cache[key] = np.asarray(evaluate(row, key), float).reshape(-1)
                calls += 1
            return cache[key]
        def gap(t):
            return float(_norm(state(t)-c)-r)
        knots = _stationary_knots(a, c, lo, hi)
        hit = None
        for start, end in zip(knots[:-1], knots[1:]):
            if gap(start) <= 0:
                hit = start
                break
            if gap(end) <= 0:
                hit = _inside_bisection(gap, start, end, parameter_tolerance)
                break
            searches += 1
            opt = minimize_scalar(gap, bounds=(start, end), method='bounded',
                                  options={'xatol': parameter_tolerance, 'maxiter': 100})
            if gap(opt.x) <= 0:
                hit = _inside_bisection(gap, start, opt.x, parameter_tolerance)
                break
        if hit is None:
            rejected += 1
            continue
        actual = state(hit)
        if _norm(actual-c) > r:
            raise RuntimeError('internal error: event state is outside requested ball')
        if hit < out[j]:
            out[j], states[j] = hit, actual
    return EventBatch(out, states, dict(candidate_subarcs=len(pair), flow_evaluations=calls,
                                       rejected_subarcs=rejected, closest_approach_searches=searches))


def interpolation_padding(arc, evaluate, factor=4., nodes=(.25, .5, .75)):
    """Practical sampled interpolation-error padding; explicitly not validated."""
    pad = np.zeros(len(arc.xa))
    for row in range(len(pad)):
        for s in nodes:
            delta = np.asarray(evaluate(row, s))-_value(arc.coeffs[row], s)
            pad[row] = max(pad[row], float(_norm(delta)))
    return factor*pad


class BallIndex:
    """Periodic radius-class KD-tree for candidate support balls."""
    def __init__(self, centers, radii, angle_indices=(), class_ratio=8., max_classes=3):
        self.centers, self.radii = np.asarray(centers, float), np.asarray(radii, float)
        self.angle_indices = tuple(angle_indices)
        if np.any(self.radii <= 0):
            raise ValueError('support radii must be positive')
        box = np.zeros(self.centers.shape[1])
        box[list(self.angle_indices)] = 2*np.pi
        self.classes = []
        if not len(self.radii):
            return
        level = np.minimum(np.floor(-np.log(self.radii/self.radii.max())/np.log(class_ratio)).astype(int), max_classes-1)
        for lv in np.unique(level):
            ids = np.flatnonzero(level == lv)
            self.classes.append((ids, self.radii[ids].max(), cKDTree(self._box(self.centers[ids]), boxsize=box)))

    def _box(self, x):
        x = np.array(x, float, copy=True)
        for j in self.angle_indices:
            x[..., j] %= 2*np.pi
            x[..., j] = np.where(x[..., j] >= 2*np.pi, 0., x[..., j])
        return x

    def pairs(self, points, query_radii):
        points = np.asarray(points, float)
        query_radii = np.broadcast_to(query_radii, (len(points),))
        rows, balls = [], []
        for ids, maximum, tree in self.classes:
            lists = tree.query_ball_point(self._box(points), query_radii+maximum)
            for row, members in enumerate(lists):
                if not members:
                    continue
                selected = ids[np.asarray(members, int)]
                centers = unwrap_relative(self.centers[selected], points[row], self.angle_indices)
                selected = selected[_norm(centers-points[row]) <= query_radii[row]+self.radii[selected]]
                rows.extend([row]*len(selected)); balls.extend(selected)
        return np.asarray(rows, int), np.asarray(balls, int)


def support_candidates(index, arc, padding=0.):
    """Return row/ball pairs and centers in a consistent small-step lift.

    Large angular steps are rejected rather than silently dropping a
    periodic image. Subdivide such steps before event localization.
    """
    extent = arc.enclosing_radius()+np.broadcast_to(padding, (len(arc.xa),))
    if index.angle_indices and len(index.radii) and np.any(extent+index.radii.max() >= np.pi):
        raise ValueError('angular event step is too large for one local lift; subdivide')
    rows, balls = index.pairs(arc.midpoint(), extent)
    centers = unwrap_relative(index.centers[balls], arc.midpoint()[rows], index.angle_indices)
    return rows, balls, centers
