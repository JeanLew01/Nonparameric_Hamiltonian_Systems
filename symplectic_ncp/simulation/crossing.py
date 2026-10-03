"""Continuous-time first entry of a simulated trajectory into Euclidean balls.

Certified radii of the assignment set are far smaller than the distance a
state travels in one simulation step, so the support Supp(K) (Definition 6)
and the target S_tgt must be detected on the continuous trajectory, not only
at sample instants (Remark 2, Step 2 of Theorem 2).  Inside one RK4 step
x_a -> x_b of length h the trajectory is represented by the cubic Hermite arc

    p(s) = h00(s) x_a + h10(s) h f_a + h01(s) x_b + h11(s) h f_b,   s in [0, 1],

with f_a = f(x_a, u), f_b = f(x_b, u) and x_b unwrapped relative to x_a.

A (row, ball) pair survives only when the chord [x_a, x_b], and then one of S
sub-chords, comes within ``r + dev`` of the center, where ``dev`` is a
rigorous bound on the distance between the arc and that chord (linear
interpolation error with the affine p'').  These filters therefore never
discard a crossing of the arc.  Candidate sub-arcs are resolved on the arc
itself: Newton's method on ||p(s) - c||^2, started at the chord's closest
point, finds the sub-arc's closest approach; if it lies in the ball, a
bisection between the sub-arc start (outside) and that point, finished by a
secant step, gives the entry parameter.  A sub-arc is 1/S of an integration
step and practically straight, so ||p(s) - c|| is unimodal on it.

Distances wrap angle coordinates (``system.angle_indices``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from symplectic_ncp.systems.base import HamiltonianSystem

_TWO_PI = 2.0 * np.pi


def _norm(D) -> np.ndarray:
    """Euclidean norm over the last axis (faster than np.linalg.norm for small n)."""
    return np.sqrt(np.einsum("...i,...i->...", D, D))


def wrapped_difference(system: HamiltonianSystem, x, y) -> np.ndarray:
    """x - y with angle coordinates mapped to [-pi, pi] (same distances as ``system.difference``)."""
    D = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    for idx in system.angle_indices:
        D[..., idx] -= _TWO_PI * np.round(D[..., idx] / _TWO_PI)
    return D


def unwrap_relative(system: HamiltonianSystem, points, reference) -> np.ndarray:
    """Representatives of ``points`` closest to ``reference`` (angles unwrapped)."""
    reference = np.asarray(reference, dtype=float)
    return reference + wrapped_difference(system, points, reference)


@dataclass
class HermiteArc:
    """Cubic Hermite interpolants of a batch of integration steps (one row per step)."""

    xa: np.ndarray  # (B, n) step start states
    xb: np.ndarray  # (B, n) step end states, unwrapped relative to xa
    fa: np.ndarray  # (B, n) vector field at xa
    fb: np.ndarray  # (B, n) vector field at xb (same held input)
    h: np.ndarray  # (B,) step lengths
    chord_deviation: np.ndarray = field(init=False)  # (B,) bound on max_s ||p(s) - chord(s)||
    coeffs: np.ndarray = field(init=False)  # (B, 4, n) monomial coefficients of p(s)

    def __post_init__(self):
        dx = self.xb - self.xa
        hfa = self.h[:, None] * self.fa
        hfb = self.h[:, None] * self.fb
        self.coeffs = np.stack([self.xa, hfa, 3.0 * dx - 2.0 * hfa - hfb, hfa + hfb - 2.0 * dx], axis=1)
        # p'' = 2 a2 + 6 a3 s is affine in s, so |p''| is maximal at s = 0 or s = 1 (componentwise).
        a2, a3 = self.coeffs[:, 2], self.coeffs[:, 3]
        acc = np.maximum(np.abs(2.0 * a2), np.abs(2.0 * a2 + 6.0 * a3))
        self.chord_deviation = _norm(acc) / 8.0

    @classmethod
    def from_step(cls, system: HamiltonianSystem, xa, xb, u, h, fa=None, fb=None) -> "HermiteArc":
        """Arc of the step xa -> xb under the held input ``u``; ``fa`` / ``fb`` may be passed if known."""
        xa = np.asarray(xa, dtype=float)
        fa = system.dynamics(xa, u) if fa is None else np.asarray(fa, dtype=float)
        fb = system.dynamics(xb, u) if fb is None else np.asarray(fb, dtype=float)
        xb = unwrap_relative(system, xb, xa)
        h = np.broadcast_to(np.asarray(h, dtype=float), (xa.shape[0],)).copy()
        return cls(xa, xb, fa, fb, h)

    def __len__(self) -> int:
        return self.xa.shape[0]

    def points(self, rows, s) -> np.ndarray:
        """p(s) for arc ``rows`` (P,) at parameters ``s`` (P, Q); returns (P, Q, n)."""
        return _horner(self.coeffs[rows], s)

    def second_derivative(self, rows, s) -> np.ndarray:
        """d^2 p / ds^2 for ``rows`` (P,) at ``s`` (P, Q); returns (P, Q, n)."""
        a = self.coeffs[rows]
        return 2.0 * a[:, None, 2] + 6.0 * np.asarray(s, dtype=float)[..., None] * a[:, None, 3]

    def midpoint(self) -> np.ndarray:
        """Chord midpoint (x_a + x_b) / 2, shape (B, n)."""
        return 0.5 * (self.xa + self.xb)

    def enclosing_radius(self) -> np.ndarray:
        """Radius around :meth:`midpoint` that contains the whole arc, shape (B,)."""
        return 0.5 * _norm(self.xb - self.xa) + self.chord_deviation


def _horner(coeffs, s) -> np.ndarray:
    """Evaluate cubics with coefficients (P, 4, n) at parameters (P, Q); returns (P, Q, n)."""
    s = np.asarray(s, dtype=float)[..., None]
    a = coeffs[:, None]
    return a[..., 0, :] + s * (a[..., 1, :] + s * (a[..., 2, :] + s * a[..., 3, :]))


def _segment_distance(p0, d) -> tuple[np.ndarray, np.ndarray]:
    """Distance from the origin to the segments p0 + tau d, tau in [0, 1], and the closest tau."""
    dd = np.einsum("...i,...i->...", d, d)
    md = np.einsum("...i,...i->...", p0, d)
    tau = np.clip(-md / np.where(dd > 0.0, dd, 1.0), 0.0, 1.0)
    return _norm(p0 + tau[..., None] * d), tau


def first_entry(
    arc: HermiteArc,
    rows,
    centers,
    radii,
    num_chords: int = 16,
    newton_iters: int = 4,
    bisection_iters: int = 12,
) -> np.ndarray:
    """Earliest s in [0, 1] with ||p(s) - c|| <= r for every (row, ball) pair.

    ``rows`` (P,) index the arc, ``centers`` (P, n) must already be unwrapped
    relative to ``arc.xa[rows]`` and ``radii`` is (P,).  Returns (P,) with
    ``np.inf`` where the arc does not meet the ball; ``s = 0`` means the arc
    starts inside the ball.
    """
    rows = np.asarray(rows, dtype=int)
    centers = np.asarray(centers, dtype=float)
    radii = np.asarray(radii, dtype=float)
    out = np.full(rows.shape[0], np.inf)
    if rows.size == 0:
        return out

    # Stage 0: the arc stays within chord_deviation of the chord [x_a, x_b].
    dist0, _ = _segment_distance(arc.xa[rows] - centers, arc.xb[rows] - arc.xa[rows])
    keep = np.flatnonzero(dist0 <= radii + arc.chord_deviation[rows])
    if keep.size == 0:
        return out
    rows, centers, radii = rows[keep], centers[keep], radii[keep]

    # Stage 1: S sub-chords, each with its own deviation bound (vertices once per distinct row).
    S = int(num_chords)
    urows, inv = np.unique(rows, return_inverse=True)
    G = np.broadcast_to(np.linspace(0.0, 1.0, S + 1), (urows.size, S + 1))
    verts = arc.points(urows, G)  # (U, S+1, n)
    acc = np.abs(arc.second_derivative(urows, G))
    dev = _norm(np.maximum(acc[:, :-1], acc[:, 1:])) / (8.0 * S * S)  # (U, S)
    rel = verts[inv] - centers[:, None, :]
    closest, tau = _segment_distance(rel[:, :-1], rel[:, 1:] - rel[:, :-1])  # (P', S)
    pi, ji = np.nonzero(closest <= radii[:, None] + dev[inv])
    if pi.size == 0:
        return out

    # Stage 2: on each candidate sub-arc locate the point closest to the center (Newton on
    # ||p(s) - c||^2 started at the chord's closest point); if it lies in the ball, the entry
    # is the boundary crossing between the sub-arc start (outside) and that point.
    c, r = centers[pi], radii[pi]
    coeffs = arc.coeffs[rows[pi]].copy()
    coeffs[:, 0] -= c  # cubic of p(s) - c
    lo, hi = ji / S, (ji + 1.0) / S
    s_min = (ji + tau[pi, ji]) / S
    for _ in range(int(newton_iters)):
        val, d1, d2 = _cubic_derivatives(coeffs, s_min)
        grad = np.einsum("ci,ci->c", val, d1)
        curv = np.einsum("ci,ci->c", d1, d1) + np.einsum("ci,ci->c", val, d2)
        ok = curv > 0.0
        s_min = np.where(ok, np.clip(s_min - grad / np.where(ok, curv, 1.0), lo, hi), s_min)
    g_min = _norm(_horner(coeffs, s_min[:, None])[:, 0]) - r
    g_lo = _norm(_horner(coeffs, lo[:, None])[:, 0]) - r
    hit = np.flatnonzero((g_min <= 0.0) | (g_lo <= 0.0))
    if hit.size == 0:
        return out
    pi, coeffs, r, lo, s_min, g_lo = pi[hit], coeffs[hit], r[hit], lo[hit], s_min[hit], g_lo[hit]
    s_hit = lo.copy()  # sub-arc starts inside the ball (only possible at s = 0 for the first entry)
    todo = np.flatnonzero(g_lo > 0.0)
    if todo.size:
        s_hit[todo] = _bisect_entry(coeffs[todo], r[todo], lo[todo], s_min[todo], bisection_iters)
    np.minimum.at(out, keep[pi], s_hit)
    return out


def _cubic_derivatives(coeffs, s) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """p, dp/ds and d^2p/ds^2 of cubics (P, 4, n) at parameters (P,)."""
    s = np.asarray(s, dtype=float)[:, None]
    a0, a1, a2, a3 = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2], coeffs[:, 3]
    val = a0 + s * (a1 + s * (a2 + s * a3))
    d1 = a1 + s * (2.0 * a2 + 3.0 * s * a3)
    d2 = 2.0 * a2 + 6.0 * s * a3
    return val, d1, d2


def _bisect_entry(coeffs, radii, lo, hi, iters: int) -> np.ndarray:
    """Crossing of ||q(s)|| = r between ``lo`` (outside) and ``hi`` (inside) for cubics q = p - c:
    bisection, then one secant step on the signed distance."""

    def gap(s):
        return _norm(_horner(coeffs, s[:, None])[:, 0]) - radii

    g_lo, g_hi = gap(lo), gap(hi)
    for _ in range(int(iters)):
        mid = 0.5 * (lo + hi)
        g = gap(mid)
        ins = g <= 0.0
        hi, g_hi = np.where(ins, mid, hi), np.where(ins, g, g_hi)
        lo, g_lo = np.where(ins, lo, mid), np.where(ins, g_lo, g)
    denom = g_lo - g_hi
    return np.where(denom > 0.0, lo + (hi - lo) * g_lo / np.where(denom > 0.0, denom, 1.0), hi)


def earliest_per_row(num_rows: int, rows, s, labels) -> tuple[np.ndarray, np.ndarray]:
    """Per row: the smallest finite ``s`` over its pairs and its label (``inf`` / -1 if none)."""
    rows = np.asarray(rows, dtype=int)
    s = np.asarray(s, dtype=float)
    labels = np.asarray(labels, dtype=int)
    best = np.full(num_rows, np.inf)
    label = np.full(num_rows, -1, dtype=int)
    finite = np.isfinite(s)
    if not np.any(finite):
        return best, label
    rows, s, labels = rows[finite], s[finite], labels[finite]
    order = np.lexsort((s, rows))  # by row, then by s
    rows, s, labels = rows[order], s[order], labels[order]
    first = np.ones(rows.size, dtype=bool)
    first[1:] = rows[1:] != rows[:-1]
    best[rows[first]] = s[first]
    label[rows[first]] = labels[first]
    return best, label


class BallIndex:
    """Candidate search for the balls B_{r_i}(c_i) that can meet query balls B_q(p).

    Balls are split by radius into at most ``max_classes`` classes (radius
    ratio ``class_ratio`` between consecutive class bounds, the last class
    takes all smaller balls), each with its own periodic KD-tree queried with
    ``q + max radius of the class``.  This keeps query radii tight even when a
    few certified radii are large (spring-mass radii reach ~1).  Returned pairs
    satisfy ||p - c_i|| <= q + r_i (angles wrapped) and no such pair is omitted.
    """

    def __init__(self, system: HamiltonianSystem, centers, radii, class_ratio: float = 8.0, max_classes: int = 2):
        self.system = system
        self.centers = np.asarray(centers, dtype=float)
        self.radii = np.asarray(radii, dtype=float)
        self.classes: list[tuple[np.ndarray, float, cKDTree]] = []
        if self.radii.size == 0:
            return
        level = np.floor(-np.log(self.radii / self.radii.max()) / np.log(class_ratio)).astype(int)
        level = np.minimum(level, int(max_classes) - 1)
        for lv in np.unique(level):
            members = np.flatnonzero(level == lv)
            tree = cKDTree(system.to_periodic_box(self.centers[members]), boxsize=system.periodic_boxsize())
            self.classes.append((members, float(self.radii[members].max()), tree))

    def pairs(self, points, query_radii) -> tuple[np.ndarray, np.ndarray]:
        """All (point index, ball index) with ||p - c_i|| <= q + r_i."""
        P = np.asarray(points, dtype=float)
        q = np.asarray(query_radii, dtype=float)
        box = self.system.to_periodic_box(P)
        rows_out, balls_out = [], []
        for members, r_class, tree in self.classes:
            qr = q + r_class
            counts = tree.query_ball_point(box, qr, return_length=True)
            hit = np.flatnonzero(counts)
            if hit.size == 0:
                continue
            lists = tree.query_ball_point(box[hit], qr[hit], return_sorted=False)
            pr = np.repeat(hit, counts[hit])
            pb = members[np.fromiter((j for lst in lists for j in lst), dtype=int, count=pr.size)]
            ok = _norm(wrapped_difference(self.system, P[pr], self.centers[pb])) <= q[pr] + self.radii[pb]
            rows_out.append(pr[ok])
            balls_out.append(pb[ok])
        if not rows_out:
            return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
        return np.concatenate(rows_out), np.concatenate(balls_out)
