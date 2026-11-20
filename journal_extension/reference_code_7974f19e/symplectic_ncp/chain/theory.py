"""Diagnostics of the theory of Section III for a constructed assignment set.

* Theorem 2, Condition 1 (local energy decrease): the algebraic inequality on
  every stored triple and an empirical check of its consequence (7),
  Delta H(phi(tau_i, y, u_i)) + v0 tau_i <= Delta H(y), by simulating snippets
  from points y of the certified balls;
* Theorem 2, Condition 2 (energy coverage) and Condition 3 (ergodic coverage),
  measured on the energy axis from numerically computed ball images H(B_{r_i}(x_i));
* Lemma 1, Theorem 3 (existence / sample complexity) and Theorem 4 (finite time).

Every quantity in :func:`theory_report` is a python float/int/bool/list or
``None`` when not applicable, so the report is JSON-serializable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from symplectic_ncp.chain.assignment_set import AssignmentSet
from symplectic_ncp.experts.demonstration import Demonstration
from symplectic_ncp.systems.base import HamiltonianSystem
from symplectic_ncp.target import TargetSet

MAX_GAPS = 5  # number of largest uncovered energy intervals reported


def _num(x) -> float | None:
    """Python float, or None for missing / non-finite values (strict JSON)."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _stats(values: np.ndarray) -> dict:
    if values.size == 0:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": _num(np.min(values)),
        "max": _num(np.max(values)),
        "mean": _num(np.mean(values)),
        "median": _num(np.median(values)),
    }


# ------------------------------------------------------------ geometry of X
def _grid(system: HamiltonianSystem, energy_bound: float, points_per_axis: int | None = None):
    """Regular grid of the bounding box of {H <= energy_bound}; returns (mesh (g,..,g,n), inside mask)."""
    n = system.state_dim
    g = points_per_axis or max(11, int(round(4e5 ** (1.0 / n))))
    low, high = system.bounding_box(energy_bound)
    axes = [np.linspace(lo, hi, g) for lo, hi in zip(low, high)]
    mesh = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    inside = system.hamiltonian(mesh.reshape(-1, n)).reshape(mesh.shape[:-1]) <= energy_bound + 1e-12
    return mesh, inside


def vector_field_bound(system: HamiltonianSystem, energy_bound: float) -> float:
    """C_f = sup_{x in X, u in U} ||f(x, u)|| (Remark 1) on a grid of X.

    f is affine in u, so ||f|| is convex in u and the sup over the box U is
    attained at one of its vertices.
    """
    mesh, inside = _grid(system, energy_bound)
    X = mesh[inside]
    m = system.control_dim
    corners = np.array(np.meshgrid(*[[lo, hi] for lo, hi in zip(system.u_min, system.u_max)], indexing="ij"))
    corners = corners.reshape(m, -1).T
    return float(max(np.max(np.linalg.norm(system.dynamics(X, np.broadcast_to(u, (X.shape[0], m))), axis=1))
                     for u in corners))


def state_space_diameter(system: HamiltonianSystem, energy_bound: float) -> float:
    """D_X = sup_{x, y in X} ||x - y|| (angles wrapped), from the boundary points of a grid of X.

    The farthest pair lies on the boundary of X, so only grid points of X with a
    neighbour outside X (or on the edge of the box) are compared.
    """
    n = system.state_dim
    mesh, inside = _grid(system, energy_bound, max(11, int(round(4e4 ** (1.0 / n)))))
    padded = np.pad(inside, 1, constant_values=False)
    interior = inside.copy()
    for axis in range(n):
        for shift in (-1, 1):
            interior &= np.roll(padded, shift, axis=axis)[tuple(slice(1, -1) for _ in range(n))]
    P = mesh[inside & ~interior]
    best = 0.0
    for start in range(0, P.shape[0], 512):
        d = system.distance(P[start : start + 512, None, :], P[None, :, :])
        best = max(best, float(np.max(d)))
    return best


def strong_convexity_modulus(system: HamiltonianSystem, energy_bound: float) -> float | None:
    """mu_H of Assumption 7 on X, or None when H is not strongly convex there.

    With constant J and G, df/dx = J Hess H, hence Hess H = J^{-1} df/dx; mu_H is
    its smallest eigenvalue over a grid of X.  Spring-mass: min(k, 1/m);
    pendulum: Hess H = diag(mgl cos q, 1/(m l^2)) is indefinite -> None.
    """
    J, _ = system.structure_matrices()
    if abs(np.linalg.det(J)) < 1e-12:
        return None
    mesh, inside = _grid(system, energy_bound)
    hess = np.linalg.solve(J, system.state_jacobian(mesh[inside]))
    mu = float(np.min(np.linalg.eigvalsh(0.5 * (hess + np.swapaxes(hess, 1, 2)))))
    return mu if mu > 1e-12 else None


# -------------------------------------------------------- energy intervals
def _merge(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted((float(a), float(b)) for a, b in intervals if b >= a):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _coverage(required: Sequence[tuple[float, float]], covered: Sequence[tuple[float, float]]) -> dict:
    """Measure of ``required`` covered by ``covered`` (both unions of intervals) and the largest gaps."""
    required, covered = _merge(required), _merge(covered)
    total = sum(b - a for a, b in required)
    gaps: list[tuple[float, float]] = []
    for a, b in required:
        cursor = a
        for c0, c1 in covered:
            if c1 <= cursor or c0 >= b:
                continue
            if c0 > cursor:
                gaps.append((cursor, c0))
            cursor = max(cursor, c1)
            if cursor >= b:
                break
        if cursor < b:
            gaps.append((cursor, b))
    uncovered = sum(b - a for a, b in gaps)
    gaps.sort(key=lambda ab: ab[0] - ab[1])
    return {
        "required_intervals": [[_num(a), _num(b)] for a, b in required],
        "required_measure": _num(total),
        "covered_fraction": _num(1.0 - uncovered / total) if total > 0 else None,
        "uncovered_measure": _num(uncovered),
        "largest_gaps": [[_num(a), _num(b)] for a, b in gaps[:MAX_GAPS]],
    }


def _unit_ball_offsets(dim: int, num_dirs: int, num_shells: int) -> np.ndarray:
    """Center, a dense boundary sphere and ``num_shells`` interior spheres of the unit ball."""
    if dim == 2:
        ang = np.linspace(0.0, 2.0 * np.pi, num_dirs, endpoint=False)
        dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    else:
        dirs = np.random.default_rng(0).normal(size=(num_dirs, dim))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    shells = np.arange(1, num_shells + 2) / (num_shells + 1)  # ..., 1 (boundary)
    return np.vstack([np.zeros((1, dim)), (shells[:, None, None] * dirs[None]).reshape(-1, dim)])


def ball_energy_images(
    system: HamiltonianSystem,
    centers: np.ndarray,
    radii: np.ndarray,
    num_dirs: int = 128,
    num_shells: int = 2,
    chunk: int = 2048,
) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """Numerical energy images H(B_{r_i}(x_i)) = [lo_i, hi_i], overall and per ergodic component.

    Ball images are intervals (connected ball, continuous H).  In 2-D the
    extrema lie on the boundary circle unless the ball contains an
    equilibrium, so the dense boundary circle plus the center and a few
    interior circles (which catch interior equilibria up to O(r^2)) suffice;
    in higher dimension the same samples use random directions.  Per
    component alpha, [lo, hi] is the range over the samples of the ball with
    label alpha (NaN when the ball does not meet alpha).
    """
    offsets = _unit_ball_offsets(system.state_dim, num_dirs, num_shells)
    N = centers.shape[0]
    lo, hi = np.empty(N), np.empty(N)
    comps = {c: (np.full(N, np.nan), np.full(N, np.nan)) for c in system.component_names()}
    for s in range(0, N, chunk):
        pts = centers[s : s + chunk, None, :] + radii[s : s + chunk, None, None] * offsets[None]
        flat = pts.reshape(-1, system.state_dim)
        E = system.hamiltonian(flat).reshape(pts.shape[:2])
        lab = system.ergodic_component(flat).reshape(pts.shape[:2])
        lo[s : s + chunk], hi[s : s + chunk] = E.min(axis=1), E.max(axis=1)
        for c, (clo, chi) in comps.items():
            mask = lab == c
            has = mask.any(axis=1)
            cmin = np.where(mask, E, np.inf).min(axis=1)
            cmax = np.where(mask, E, -np.inf).max(axis=1)
            clo[s : s + chunk] = np.where(has, cmin, np.nan)
            chi[s : s + chunk] = np.where(has, cmax, np.nan)
    return lo, hi, comps


def component_energy_ranges(
    system: HamiltonianSystem, energy_low: float, energy_high: float, num_samples: int = 200_000,
    num_bins: int = 400, seed: int = 0,
) -> dict[int, list[tuple[float, float]]]:
    """Energies E in [energy_low, energy_high] at which each ergodic component exists.

    Estimated from uniform samples of {H <= energy_high} binned in energy: a
    component exists on a whole bin when it is also present in both
    neighbouring bins, and only on the sampled sub-range of the bin at a
    transition (e.g. the pendulum separatrix).
    """
    rng = np.random.default_rng(seed)
    X = system.sample_energy_sublevel(num_samples, energy_high, rng)
    E = system.hamiltonian(X)
    keep = E >= energy_low
    E, lab = E[keep], system.ergodic_component(X[keep])
    edges = np.linspace(energy_low, energy_high, num_bins + 1)
    idx = np.clip(np.searchsorted(edges, E, side="right") - 1, 0, num_bins - 1)
    out: dict[int, list[tuple[float, float]]] = {}
    for c in system.component_names():
        mask = lab == c
        emin = np.full(num_bins, np.inf)
        emax = np.full(num_bins, -np.inf)
        np.minimum.at(emin, idx[mask], E[mask])
        np.maximum.at(emax, idx[mask], E[mask])
        present = np.isfinite(emin)
        intervals = []
        for b in np.flatnonzero(present):
            a = edges[b] if b == 0 or present[b - 1] else emin[b]
            z = edges[b + 1] if b == num_bins - 1 or present[b + 1] else emax[b]
            intervals.append((a, z))
        out[c] = _merge(intervals)
    return out


def _intersect(A: Sequence[tuple[float, float]], B: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    out = []
    for a0, a1 in A:
        for b0, b1 in B:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi > lo:
                out.append((lo, hi))
    return _merge(out)


# ------------------------------------------------------- Condition 1 checks
def _snippet_endpoints(assignments: AssignmentSet, demos: Sequence[Demonstration]) -> np.ndarray:
    """x_i(tau_i) = phi(s_i + tau_i) read from the source demonstrations."""
    out = np.empty_like(assignments.centers)
    ends = assignments.anchor_times + assignments.durations  # snippets end on the demonstration grid
    for i, (j, t_end) in enumerate(zip(assignments.demo_ids, ends)):
        demo = demos[int(j)]
        out[i] = demo.states[int(round(t_end / demo.dt))]
    return out


def _simulate_snippets(
    system: HamiltonianSystem, Y: np.ndarray, controls: list[np.ndarray], dt: float, leads: np.ndarray
) -> np.ndarray:
    """phi(tau, y, u) for every row y with its own ZOH snippet (RK4 on the snippet grid, first step = lead)."""
    steps = np.asarray([u.shape[0] for u in controls])
    U = np.zeros((len(controls), int(steps.max()), system.control_dim))
    for b, u in enumerate(controls):
        U[b, : u.shape[0]] = u
    X = Y.copy()
    for k in range(int(steps.max())):
        active = steps > k
        h = np.asarray(leads, dtype=float)[active] if k == 0 else dt
        X[active] = system.rk4_step(X[active], U[active, k], h)
    return X


def condition1_empirical(
    system: HamiltonianSystem,
    target: TargetSet,
    assignments: AssignmentSet,
    v0: float,
    max_triples: int = 300,
    boundary_dirs: int = 8,
    interior_points: int = 4,
    seed: int = 0,
    tol: float = 1e-9,
) -> dict:
    """Empirical check of (7): Delta H(phi(tau_i, y, u_i)) + v0 tau_i <= Delta H(y) for y in B_{r_i}(x_i).

    Triples: all, or a seeded subset of ``max_triples``.  Points per triple:
    the center, ``boundary_dirs`` boundary points (evenly spaced angles in
    2-D, random directions otherwise) and ``interior_points`` uniform interior
    points.  Snippets are simulated with ``system.rk4_step`` on their own ZOH
    grid (first step ``K.leads[i]``, then K.dt), i.e. exactly the snippet the
    closed-loop simulator applies.  A point
    violates (7) when the left side exceeds the right side by more than
    ``tol * max(1, |Delta H(y)|)`` (floating-point slack only).
    """
    rng = np.random.default_rng(seed)
    N, n = len(assignments), system.state_dim
    idx = np.arange(N) if N <= max_triples else np.sort(rng.choice(N, max_triples, replace=False))
    if n == 2:
        ang = np.arange(boundary_dirs) * 2.0 * np.pi / boundary_dirs
        bdirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    else:
        bdirs = rng.normal(size=(boundary_dirs, n))
        bdirs /= np.linalg.norm(bdirs, axis=1, keepdims=True)
    Y, rows, ctrl = [], [], []
    for i in idx:
        x, r = assignments.centers[i], assignments.radii[i]
        idirs = rng.normal(size=(interior_points, n))
        idirs /= np.linalg.norm(idirs, axis=1, keepdims=True)
        rad = r * rng.uniform(size=(interior_points, 1)) ** (1.0 / n)
        pts = np.vstack([x[None], x + r * bdirs, x + rad * idirs])
        Y.append(pts)
        rows.extend([i] * pts.shape[0])
        ctrl.extend([assignments.controls[i]] * pts.shape[0])
    Y, rows = np.vstack(Y), np.asarray(rows)
    end = _simulate_snippets(system, Y, ctrl, assignments.dt, assignments.leads[rows])
    dH_y = target.energy_distance(Y)
    margin = target.energy_distance(end) + v0 * assignments.durations[rows] - dH_y
    violated = margin > tol * np.maximum(1.0, np.abs(dH_y))
    bad_triples = np.unique(rows[violated])
    return {
        "num_triples_checked": int(idx.size),
        "points_per_triple": int(1 + boundary_dirs + interior_points),
        "num_points": int(Y.shape[0]),
        "num_violations": int(violated.sum()),
        "violation_fraction": _num(violated.mean()),
        "triples_violated_fraction": _num(bad_triples.size / idx.size),
        "max_margin": _num(margin.max()),  # max of lhs - rhs of (7); <= 0 means satisfied
        "boundary_max_margin": _num(margin.reshape(idx.size, -1)[:, 1 : 1 + boundary_dirs].max()),
    }


def condition1_algebraic(
    target: TargetSet, assignments: AssignmentSet, demos: Sequence[Demonstration], v0: float, L_H: float, L: float,
    tol: float = 1e-9,
) -> dict:
    """Condition 1 of Theorem 2 on the stored triples with the given (global) constants:

    Delta H(x_i(tau_i)) + v0 tau_i + L_H r_i e^{L tau_i} <= Delta H(x_i) - L_H r_i.
    Algorithm 1 with global constants makes this an equality up to round-off.
    """
    tau, r = assignments.durations, assignments.radii
    dH0 = target.energy_distance(assignments.centers)
    dH1 = target.energy_distance(_snippet_endpoints(assignments, demos))
    with np.errstate(over="ignore"):  # L_H r e^{L tau} without overflowing e^{L tau} alone
        growth = np.exp(np.log(L_H * r) + L * tau)
    residual = dH1 + v0 * tau + growth - (dH0 - L_H * r)
    violated = residual > tol * np.maximum(1.0, np.abs(dH0))
    return {
        "num_violations": int(violated.sum()),
        "violation_fraction": _num(violated.mean()),
        "max_residual": _num(residual.max()),
    }


# ------------------------------------------------------- decrease rates
def demo_decrease_rates(
    target: TargetSet, assignments: AssignmentSet, demos: Sequence[Demonstration], eps: float
) -> tuple[np.ndarray, np.ndarray]:
    """Demonstrated rates v_eps(x_i, u_j) = (Delta H(x_i) + eps) / T_eps (eq. 3) and hitting times T_eps.

    For every anchor outside H_tgt^eps, T_eps is the remaining time until the
    source demonstration first enters H_tgt^eps (on its grid).  Since
    v_eps(x) is a sup over inputs, min_i of these rates estimates (an upper
    bound of) the uniform rate underline{v_eps} of Assumption 5 only up to the
    expert's suboptimality.
    """
    rates, times = [], []
    for j, demo in enumerate(demos):
        sel = np.flatnonzero(assignments.demo_ids == j)
        if sel.size == 0:
            continue
        dH = target.energy_distance(demo.states)
        hit = dH <= -eps
        n = dH.shape[0]
        nxt = np.where(hit, np.arange(n), n)
        nxt = np.minimum.accumulate(nxt[::-1])[::-1]  # first hit index >= k
        s = assignments.anchor_times[sel]
        k = np.minimum(np.ceil(s / demo.dt - 1e-9).astype(int), n - 1)  # first grid point at or after s
        dH_anchor = target.energy_distance(assignments.centers[sel])
        ok = (dH_anchor > -eps) & (nxt[k] < n)
        T = nxt[k[ok]] * demo.dt - s[ok]
        ok_pos = T > 0.0
        T = T[ok_pos]
        rates.append((dH_anchor[ok][ok_pos] + eps) / T)
        times.append(T)
    if not rates:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(rates), np.concatenate(times)


# ---------------------------------------------------------------- report
def theory_report(
    system: HamiltonianSystem,
    target: TargetSet,
    assignments: AssignmentSet,
    demos: Sequence[Demonstration],
    cfg,
    L_H: float,
    L: float,
    H_X: float,
    T1: float | None = None,
    T2: float | None = None,
    seed: int = 0,
    T1_censored: bool = False,
) -> dict:
    """JSON-serializable diagnostics of Theorems 2-4 and Lemma 1 for K built from ``demos``.

    ``cfg`` is an :class:`~symplectic_ncp.config.ExperimentConfig` (v0, eps,
    H_bar).  ``T1``/``T2`` are the return/reaching times of Assumption 8
    (e.g. observed in closed loop); Theorem 4 is evaluated when T1 is given
    and flagged applicable only if Theorem 2's conditions hold and T1 is not
    right-censored (``T1_censored``: some zero-input excursion never returned
    to Supp(K) before the horizon, so T1 is only a lower bound).
    """
    v0, eps, H_bar = float(cfg.chain.v0), float(cfg.energy_eps), float(cfg.H_bar)
    N = len(assignments)

    # ----- constants on X
    C_f = vector_field_bound(system, H_X)
    D_X = state_space_diameter(system, H_X)
    mu_H = strong_convexity_modulus(system, H_X)
    mesh, inside = _grid(system, H_X)
    E_floor = float(np.min(system.hamiltonian(mesh[inside])))  # min H on S_0 (and X)

    # ----- energy range of Condition 2: {E : Delta H(E) <= c}, c = sup_{S_0} Delta H
    c = float(max(target.energy_distance_from_energy(E_floor), target.energy_distance_from_energy(H_bar)))
    lower = (max(E_floor, target.H_min - c), target.H_min)
    upper = (target.H_max, target.H_max + c)
    required = [iv for iv in (lower, upper) if iv[1] > iv[0]]  # Delta H in (0, c]
    literal = [(max(E_floor, target.H_min - c), target.H_max + c)]  # Delta H <= c (incl. the band)

    report: dict = {
        "N": int(N),
        "per_demo": {str(int(j)): int(np.sum(assignments.demo_ids == j)) for j in np.unique(assignments.demo_ids)},
        "constants": {
            "L_H": _num(L_H), "L": _num(L), "H_X": _num(H_X), "C_f": _num(C_f), "D_X": _num(D_X),
            "mu_H": _num(mu_H), "v0": _num(v0), "eps": _num(eps), "H_bar": _num(H_bar),
            "H_min": _num(target.H_min), "H_max": _num(target.H_max), "c": _num(c),
            "lipschitz_mode": str(getattr(cfg.chain, "lipschitz", "global")),
        },
        "radii": _stats(assignments.radii),
        "durations": _stats(assignments.durations),
        "tau_min": _num(np.min(assignments.durations)) if N else None,
    }
    if N == 0:
        report.update({"condition1": None, "condition2": _coverage(required, []), "condition3": None,
                       "lemma1": None, "theorem2": None, "theorem3": None, "theorem4": None})
        return report
    tau_min = float(np.min(assignments.durations))

    # ----- Condition 1
    report["condition1"] = {
        "algebraic": condition1_algebraic(target, assignments, demos, v0, L_H, L),
        "empirical": condition1_empirical(system, target, assignments, v0, seed=seed),
    }

    # ----- Conditions 2 and 3
    lo, hi, comp = ball_energy_images(system, assignments.centers, assignments.radii)
    cov = _coverage(required, list(zip(lo, hi)))
    cov["covered_fraction_including_band"] = _coverage(literal, list(zip(lo, hi)))["covered_fraction"]
    cov["support_energy_range"] = [_num(lo.min()), _num(hi.max())]
    report["condition2"] = cov
    exist = component_energy_ranges(system, E_floor, max(H_X, target.H_max + c), seed=seed)
    names = system.component_names()
    cond3 = {}
    for cid, name in names.items():
        clo, chi = comp[cid]
        have = np.isfinite(clo)
        entry = _coverage(_intersect(required, exist[cid]), list(zip(clo[have], chi[have])))
        entry["num_balls"] = int(have.sum())
        cond3[name] = entry
    report["condition3"] = cond3

    # ----- Lemma 1 and the demonstrated decrease rate
    rates, times = demo_decrease_rates(target, assignments, demos, eps)
    v_hat = float(rates.min()) if rates.size else None
    report["lemma1"] = {
        "v_upper_bound_LH_Cf": _num(L_H * C_f),
        "T_eps_lower_bound": _num(eps / (L_H * C_f)),
        "v_eps_hat": _num(v_hat),
        "v_eps_median": _num(np.median(rates)) if rates.size else None,
        "v_eps_max": _num(rates.max()) if rates.size else None,
        "num_rate_samples": int(rates.size),
        "rates_below_bound": bool(np.all(rates <= L_H * C_f)) if rates.size else None,
        "min_hitting_time": _num(times.min()) if times.size else None,
        "hitting_times_above_bound": bool(np.all(times >= eps / (L_H * C_f))) if times.size else None,
        "v0_below_v_eps_hat": bool(v0 < v_hat) if v_hat is not None else None,
    }

    # ----- Theorem 2: number of controlled executions to reach H_tgt
    cov_tol = 1e-3  # numerical slack on the sampled coverage fractions
    c1 = report["condition1"]["empirical"]["violation_fraction"]
    c2 = report["condition2"].get("covered_fraction_including_band")
    c3 = [v.get("covered_fraction") for v in (report.get("condition3") or {}).values() if isinstance(v, dict)]
    c3 = [x for x in c3 if x is not None]
    report["theorem2"] = {
        "executions_bound": int(math.ceil(c / (v0 * tau_min))) if c > 0 else 0,
        "condition1_holds": bool(c1 == 0),
        "condition2_holds": bool(c2 is not None and c2 >= 1.0 - cov_tol),
        "condition3_holds": bool(not c3 or min(c3) >= 1.0 - cov_tol),
    }
    report["theorem2"]["conditions_hold"] = all(
        report["theorem2"][k] for k in ("condition1_holds", "condition2_holds", "condition3_holds")
    )

    # ----- Theorem 3 (requires Assumption 7 and v0 < underline{v_eps})
    H1, H2 = float(lo.min()), float(hi.max())
    th3: dict = {"H1": _num(H1), "H2": _num(H2), "N": int(N)}
    if mu_H is None:
        th3.update(applicable=False, reason="H is not strongly convex on X (Assumption 7 fails)")
    elif v_hat is None or not v0 < v_hat:
        th3.update(applicable=False, reason="v0 < v_eps_hat does not hold")
    else:
        ratio = 1.0 - v0 / v_hat
        expo = L * (L_H * D_X + eps) / v_hat
        log10_N = (math.log10((H2 - H1) * 16.0 * L_H**2 / (mu_H * ratio**2 * eps**2))
                   + 2.0 * expo / math.log(10.0))
        log10_r = math.log10(eps * ratio / (2.0 * L_H)) - expo / math.log(10.0)
        th3.update(
            applicable=True,
            log10_N_bound=_num(log10_N),
            N_bound=_num(10.0**log10_N) if log10_N < 300 else None,
            # Theorem 3 bounds the size of its canonical construction (optimal controls, r(x),
            # greedy energy cover), not Algorithm 1's K; v_eps_hat (demonstrated rates) is a
            # plug-in for underline{v_eps}.  The comparison below is informative only.
            note="existence bound for the canonical construction of Theorem 3 with v_eps_hat plugged in; "
            "not a bound on Algorithm 1's N",
            alg1_N_le_canonical_bound=bool(math.log10(max(N, 1)) <= log10_N),
            log10_radius_lower_bound=_num(log10_r),  # eq. (8)
        )
    report["theorem3"] = th3

    # ----- Theorem 4 (Assumption 8 times T1, T2)
    th4: dict = {"T1": _num(T1), "T1_censored": bool(T1_censored), "T2": _num(T2), "tau_min": _num(tau_min)}
    reasons = []
    if not report["theorem2"]["conditions_hold"]:
        reasons.append("Theorem 2's conditions do not hold for this K")
    if T1 is None or T1_censored:
        reasons.append("T1 of Assumption 8 is unknown or right-censored (some excursions never returned)")
    th4.update(applicable=not reasons, reason="; ".join(reasons) or None)
    if T1 is not None:
        T_bar = L_H * D_X / v0 * (1.0 + T1 / tau_min)  # eq. (14)
        T_bar_13 = c / v0 + math.floor(c / (v0 * tau_min)) * T1  # eq. (13) with Delta H(x0) <= c
        th4.update(T_bar_bound=_num(T_bar), T_bar_bound_eq13=_num(T_bar_13),
                   T_max_bound=_num(T_bar + T2) if T2 is not None else None)
    else:
        th4.update(T_bar_bound=None, T_bar_bound_eq13=None, T_max_bound=None)
    report["theorem4"] = th4
    return report
