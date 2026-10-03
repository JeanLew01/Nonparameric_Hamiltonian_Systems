"""Assignment-set construction from expert demonstrations (paper, Section III-D, Algorithm 1).

Along a demonstration phi(t, x_j, u_j), t in [0, T_j], sampled on its step
``dt``, an anchor time s gives the center x_i = phi(s) and every candidate
duration t ending on a grid point, the certified radius

    r_i(t) = (Delta H(x_i) - Delta H(phi(s + t)) - v0 t) / (L_H + L_H e^{L t}),

which is exactly the largest r satisfying Condition 1 of Theorem 2 for the
snippet u_{i,t} = u_j restricted to (s, s + t].
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from symplectic_ncp.chain.assignment_set import AssignmentSet
from symplectic_ncp.experts.demonstration import Demonstration
from symplectic_ncp.systems.base import HamiltonianSystem
from symplectic_ncp.target import TargetSet

LIPSCHITZ_MODES = ("global", "local")


# --------------------------------------------------------------- constants
LIPSCHITZ_MARGIN = 5e-2  # default energy padding of X (ChainConfig.energy_margin)


def lipschitz_for_demos(
    system: HamiltonianSystem,
    demos: Sequence[Demonstration],
    H_bar: float,
    energy_floor: float = 0.0,
    margin: float = LIPSCHITZ_MARGIN,
) -> tuple[float, float, float]:
    """Constants of Assumptions 1-2 on X = {H <= H_X}.

    H_X = (1 + margin) max(H_bar, max_j max_t H(phi_j(t)), energy_floor):
    X must contain S_0 = {H <= H_bar}, every demonstrated state and (via
    ``energy_floor``, see :func:`build_certified_assignment_set`) every
    certified ball, so that the Grönwall and Lipschitz arguments of Theorem 2
    apply.  The margin keeps the grid-based supremum of
    :meth:`HamiltonianSystem.lipschitz_constants` an upper bound on the
    unpadded set and usually leaves room for the balls, so that
    :func:`build_certified_assignment_set` needs a single pass.  Returns
    ``(L_H, L, H_X)``.
    """
    H_X = max(float(H_bar), float(energy_floor))
    for demo in demos:
        H_X = max(H_X, float(np.max(system.hamiltonian(demo.states))))
    H_X *= 1.0 + margin
    L_H, L = system.lipschitz_constants(H_X)
    return float(L_H), float(L), H_X


def max_energy_on_balls(system: HamiltonianSystem, centers: np.ndarray, radii: np.ndarray, num_dirs: int = 64) -> float:
    """max_i max_{y in B_{r_i}(x_i)} H(y), sampled on the ball boundaries and centers."""
    if len(radii) == 0:
        return -np.inf
    n = system.state_dim
    if n == 2:
        ang = np.linspace(0.0, 2.0 * np.pi, num_dirs, endpoint=False)
        dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    else:
        dirs = np.random.default_rng(0).normal(size=(num_dirs, n))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    pts = centers[:, None, :] + radii[:, None, None] * np.vstack([np.zeros((1, n)), dirs])[None]
    return float(np.max(system.hamiltonian(pts.reshape(-1, n))))


# ---------------------------------------------------------- radius profile
def _local_constants(system: HamiltonianSystem, states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """||grad H|| and ||df/dx||_2 at every state of a trajectory (for the local ablation)."""
    grad_norm = np.linalg.norm(system.grad_hamiltonian(states), axis=1)
    jac_norm = np.linalg.norm(system.state_jacobian(states), ord=2, axis=(1, 2))
    return grad_norm, jac_norm


def _denominator_weight(t: np.ndarray, L_H, L) -> np.ndarray:
    """1 / (L_H (1 + e^{Lt})) evaluated as e^{-Lt} / (L_H (1 + e^{-Lt})).

    Never overflows: e^{-Lt} underflows silently to 0 for large Lt, giving r = 0.
    A local L_H can vanish only on a sub-trajectory resting at an equilibrium.
    """
    L_H = np.maximum(np.asarray(L_H, dtype=float), np.finfo(float).tiny)
    with np.errstate(over="ignore", under="ignore"):
        decay = np.exp(-np.asarray(L, dtype=float) * t)
        return decay / (L_H * (1.0 + decay))


class _RadiusProfiler:
    """r_i(t) for an anchor phi(s), s = (k + theta) dt, and the snippet ends s + t = (k + m) dt.

    The anchor may lie between grid points (Algorithm 1 places it where the
    demonstration leaves the previous ball); the candidate durations end on
    the demonstration grid, t_m = (m - theta) dt, m = 1, ..., n - k, so the
    snippet u_j|(s, s + t_m] is a ZOH sequence whose first step lasts
    (1 - theta) dt.

    :meth:`argmax` skips the tail of the demonstration where no duration can
    beat the best radius found in a first block: since Delta H >= -H*_-
    everywhere and the constants (global, or running maxima for "local") are
    at least their values L_H0, L0 at the anchor,

        r_i(t) <= (Delta H(x_i) + H*_-) / (L_H0 (1 + e^{L0 t})),

    which is decreasing in t.  Durations whose bound is below the best radius
    are strictly worse, so the argmax is exact.
    """

    HEAD = 256  # length of the first block evaluated by argmax

    def __init__(self, system, target, demo, v0: float, L_H: float, L: float, lipschitz: str):
        if lipschitz not in LIPSCHITZ_MODES:
            raise ValueError(f"lipschitz must be one of {LIPSCHITZ_MODES}, got {lipschitz!r}")
        self.system, self.target = system, target
        self.lipschitz = lipschitz
        self.dt = demo.dt
        self.v0 = float(v0)
        self.energy_dist = target.energy_distance(demo.states)
        self.H_minus = target.H_minus
        self.L_H, self.L = float(L_H), float(L)
        if lipschitz == "local":
            self.grad_norm, self.jac_norm = _local_constants(system, demo.states)

    def profile(self, k: int, theta: float, x_i, stop: int | None = None) -> np.ndarray:
        """r_i(t_m) for m = 1..(n - k) (or up to grid index ``stop``)."""
        dH = self.energy_dist
        end = dH.shape[0] - 1 if stop is None else min(stop, dH.shape[0] - 1)
        t = (np.arange(1, end - k + 1) - theta) * self.dt
        num = float(self.target.energy_distance(x_i)[0]) - dH[k + 1 : end + 1] - self.v0 * t
        if self.lipschitz == "global":
            return num * _denominator_weight(t, self.L_H, self.L)
        # max over the sub-trajectory phi([s, s + t])
        g0, j0 = (float(a[0]) for a in _local_constants(self.system, np.asarray(x_i).reshape(1, -1)))
        L_H_t = np.maximum(g0, np.maximum.accumulate(self.grad_norm[k + 1 : end + 1]))
        L_t = np.maximum(j0, np.maximum.accumulate(self.jac_norm[k + 1 : end + 1]))
        return num * _denominator_weight(t, L_H_t, L_t)

    def _last_useful_step(self, x_i, best: float) -> int | None:
        """Largest m whose upper bound on r_i(t_m) can reach ``best`` (None: no finite cutoff)."""
        if self.lipschitz == "global":
            L_H0, L0 = self.L_H, self.L
        else:
            L_H0, L0 = (float(a[0]) for a in _local_constants(self.system, np.asarray(x_i).reshape(1, -1)))
        ratio = (float(self.target.energy_distance(x_i)[0]) + self.H_minus) / (max(L_H0, np.finfo(float).tiny) * best)
        if L0 <= 0.0 or not np.isfinite(ratio):
            return None
        if ratio <= 1.0:  # 1 + e^{L0 t} > ratio for every t > 0
            return 0
        return int(np.log(ratio - 1.0) / (L0 * self.dt)) + 2 if ratio > 2.0 else 2

    def argmax(self, k: int, theta: float, x_i) -> tuple[int, float]:
        """(m*, r_i(t_m*)) maximizing r_i over {m : r_i(t_m) > 0}; (0, 0.0) if that set is empty."""
        n = self.energy_dist.shape[0] - 1
        r = self.profile(k, theta, x_i, k + self.HEAD)
        best = float(np.max(r))
        if k + self.HEAD < n:
            last = self._last_useful_step(x_i, best) if best > 0.0 else None
            if last is None or last > self.HEAD:
                r = self.profile(k, theta, x_i, None if last is None else k + last)
        valid = r > 0.0
        if not np.any(valid):
            return 0, 0.0
        m = int(np.argmax(np.where(valid, r, -np.inf)))  # first maximizer: shortest duration on ties
        return m + 1, float(r[m])


def certified_radius_profile(
    system: HamiltonianSystem,
    target: TargetSet,
    demo: Demonstration,
    start: int,
    v0: float,
    L_H: float,
    L: float,
    lipschitz: str = "global",
) -> np.ndarray:
    """Certified radius r_i(t) of Section III-D for the anchor ``demo.states[start]``.

    Returns an array of length ``demo.num_steps - start`` whose entry k-1 is
    r_i(k dt).  ``lipschitz="global"`` uses the constants of Assumptions 1-2
    (the paper); ``"local"`` (ablation only, not certified by the theory)
    replaces them by the maxima of ||grad H|| and ||df/dx|| along phi([s, s+t]).
    """
    start = int(start)
    if not 0 <= start < demo.num_steps:
        raise ValueError(f"start index {start} outside [0, {demo.num_steps})")
    return _RadiusProfiler(system, target, demo, v0, L_H, L, lipschitz).profile(start, 0.0, demo.states[start])


# ------------------------------------------------------------ Algorithm 1
_SNAP = 1e-9  # anchors within this fraction of dt of a grid point are snapped to it


def _flow(system, x, u, h: float) -> np.ndarray:
    """phi(h, x, u) for a constant input over one (partial) demonstration step."""
    if h <= 0.0:
        return np.asarray(x, dtype=float).copy()
    return system.rk4_step(np.asarray(x, dtype=float).reshape(1, -1), u, h)[0]


def _demo_state(system, demo: Demonstration, s: float) -> np.ndarray:
    """phi(s) on the continuous demonstration: partial RK4 step from the preceding grid point."""
    k = min(int(np.floor(s / demo.dt + _SNAP)), demo.num_steps)
    h = s - k * demo.dt
    if k >= demo.num_steps or h <= _SNAP * demo.dt:
        return demo.states[k]
    return _flow(system, demo.states[k], demo.controls[k], h)


def _exit_time(system, x_start, u, h: float, x_i, r_i: float, grid: int = 32, iters: int = 60) -> float:
    """First delta in (0, h] with ||phi(delta, x_start, u) - x_i|| = r_i (x_start inside the ball).

    The one-step flow is evaluated on ``grid`` points in one batched RK4 call to bracket the
    first exit, which is then located by the Illinois (modified regula falsi) method on the exact
    one-step flow.  Returns the upper end of the final bracket (a point on or outside the sphere);
    if the sampled flow never leaves the ball, returns h.
    """
    deltas = h * np.arange(1, grid + 1) / grid
    X = system.rk4_step(np.repeat(np.asarray(x_start, dtype=float)[None, :], grid, axis=0), u, deltas)
    gaps = system.distance(X, x_i) - r_i
    out = np.flatnonzero(gaps >= 0.0)
    if out.size == 0:
        return h
    j = int(out[0])
    lo, g_lo = (0.0, float(system.distance(x_start, x_i)) - r_i) if j == 0 else (deltas[j - 1], gaps[j - 1])
    hi, g_hi = deltas[j], gaps[j]
    side = 0
    tol = 1e-12 * h
    for _ in range(iters):
        if hi - lo <= tol or g_hi <= 1e-14 * max(r_i, np.finfo(float).tiny):
            break
        c = hi - g_hi * (hi - lo) / (g_hi - g_lo)
        if not lo < c < hi:
            c = 0.5 * (lo + hi)
        g_c = float(system.distance(_flow(system, x_start, u, c), x_i)) - r_i
        if g_c >= 0.0:
            hi, g_hi = c, g_c
            if side == +1:
                g_lo *= 0.5
            side = +1
        else:
            lo, g_lo = c, g_c
            if side == -1:
                g_hi *= 0.5
            side = -1
    return hi


def _anchor_states(system, demo: Demonstration, s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized anchor geometry: grid index k, fraction theta and state phi(s) for every time in ``s``.

    Same arithmetic as the scalar loop of :func:`_algorithm1_triples` (one partial RK4 step of
    length theta dt from the preceding grid point), so both paths give identical anchors.
    """
    dt, n = demo.dt, demo.num_steps
    k = np.floor(s / dt + _SNAP).astype(int)
    theta = s / dt - k
    theta = np.where((theta < _SNAP) | (k >= n), 0.0, theta)
    kc = np.minimum(k, n)
    X = demo.states[kc].copy()
    rows = np.flatnonzero(theta > 0.0)
    if rows.size:
        X[rows] = system.rk4_step(demo.states[kc[rows]], demo.controls[kc[rows]], theta[rows] * dt)
    return k, theta, X


def _demo_states(system, demo: Demonstration, s: np.ndarray) -> np.ndarray:
    """Vectorized :func:`_demo_state` (identical arithmetic)."""
    dt, n = demo.dt, demo.num_steps
    k = np.minimum(np.floor(s / dt + _SNAP).astype(int), n)
    h = s - k * dt
    X = demo.states[k].copy()
    rows = np.flatnonzero((k < n) & (h > _SNAP * dt))
    if rows.size:
        X[rows] = system.rk4_step(demo.states[k[rows]], demo.controls[k[rows]], h[rows])
    return X


def _block_argmax(profiler: "_RadiusProfiler", k: np.ndarray, theta: np.ndarray, X: np.ndarray):
    """Vectorized :meth:`_RadiusProfiler.argmax` for global constants over a block of anchors.

    Returns (m_best, r_best, exact) where ``exact`` marks rows whose maximizer is certainly within
    the first HEAD durations; the others must be recomputed with the scalar profiler.
    """
    dH = profiler.energy_dist
    n = dH.shape[0] - 1
    head = profiler.HEAD
    m = np.arange(1, head + 1)
    idx = k[:, None] + m[None, :]
    valid_len = idx <= np.minimum(k + head, n)[:, None]
    t = (m[None, :] - theta[:, None]) * profiler.dt
    dHx = profiler.target.energy_distance(X)
    num = dHx[:, None] - dH[np.minimum(idx, n)] - profiler.v0 * t
    r = num * _denominator_weight(t, profiler.L_H, profiler.L)
    r = np.where(valid_len, r, -np.inf)
    best = np.max(r, axis=1)
    # rows whose maximizer could lie beyond the first block (same test as the scalar argmax)
    exact = (k + head >= n)
    pos = best > 0.0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratio = (dHx + profiler.H_minus) / (max(profiler.L_H, np.finfo(float).tiny) * np.where(pos, best, 1.0))
        last = np.where(ratio <= 1.0, 0, np.where(ratio > 2.0, np.floor(np.log(ratio - 1.0) / (profiler.L * profiler.dt)) + 2, 2))
    exact |= pos & np.isfinite(ratio) & (profiler.L > 0.0) & (last <= head)
    masked = np.where(r > 0.0, r, -np.inf)
    m_best = np.where(np.any(r > 0.0, axis=1), np.argmax(masked, axis=1) + 1, 0)
    r_best = np.where(m_best > 0, r[np.arange(len(k)), np.maximum(m_best - 1, 0)], 0.0)
    return m_best, r_best, exact


def _algorithm1_triples(system, target, demo, v0, L_H, L, lipschitz, min_advance: float) -> list[tuple]:
    """Lines 5-19 of Algorithm 1 on one demonstration; returns (x_i, r_i, u_i, lead_i, s_i) tuples.

    The demonstration is the continuous trajectory phi(t, x_j, u_j) whose
    states are stored on the grid t = k dt; between grid points it is
    recovered exactly by one partial RK4 step under the held input.

    * line 6: loop while s < T_j and phi(s) is not in S_tgt;
    * line 8: t_i is the argmax of r_i(t) over the durations ending on grid
      points with r_i(t) > 0 (ties -> the shortest); if none, stop (line 10);
    * line 14: sigma_i = inf{delta in (0, tau_i] : ||phi(s + delta) - x_i|| = r_i},
      located on the continuous trajectory (first grid point outside the
      ball, then the first exit inside that step, see :func:`_exit_time`), so
      consecutive balls touch.
      sigma_i = tau_i if the trajectory stays in the ball (line 16).
    * The advance is floored at ``min_advance`` (seconds).  Where r_i is
      extremely small (r_i -> 0 is possible, e.g. where Delta H barely
      decreases) the exact recursion s <- s + sigma_i would need an unbounded
      number of anchors; the floor is the only departure from line 14 and
      leaves gaps only where r_i < ||f|| min_advance.
    """
    states, controls, dt, n = demo.states, demo.controls, demo.dt, demo.num_steps
    profiler = _RadiusProfiler(system, target, demo, v0, L_H, L, lipschitz)
    triples = []
    s = 0.0  # anchor time
    block = 8  # adaptive size of the vectorized block of floor-advanced anchors
    while True:
        if lipschitz == "global" and min_advance > 0.0:
            s, block, stop = _floored_block(system, target, demo, profiler, s, min_advance, block, triples)
            if stop:
                break
        k = int(np.floor(s / dt + _SNAP))
        theta = s / dt - k
        if theta < _SNAP or k >= n:
            theta = 0.0
        if k >= n:
            break
        x_i = states[k] if theta == 0.0 else _flow(system, states[k], controls[k], theta * dt)
        if target.contains(x_i)[0]:  # line 6
            break
        m_best, r_i = profiler.argmax(k, theta, x_i)  # line 8: snippet ends at grid index k + m_best
        if m_best == 0:  # lines 9-10: no t with r_i(t) > 0
            break
        lead = (1.0 - theta) * dt
        tau_i = (m_best - theta) * dt
        triples.append((x_i.copy(), r_i, controls[k : k + m_best].copy(), lead, s))  # lines 12-13

        # Line 14: first exit of the continuous trajectory from B_{r_i}(x_i).
        floor = min(min_advance, tau_i)
        if floor > 0.0 and float(system.distance(_demo_state(system, demo, s + floor), x_i)) >= r_i:
            s += floor  # the trajectory has left the ball by s + floor: sigma_i <= floor
            continue
        dist = system.distance(states[k + 1 : k + m_best + 1], x_i[None, :])
        out = np.flatnonzero(dist >= r_i)
        if out.size:
            j = int(out[0]) + 1  # first grid point k + j outside the ball
            seg_start, seg_time = (x_i, s) if j == 1 else (states[k + j - 1], (k + j - 1) * dt)
            seg_len = (k + j) * dt - seg_time
            sigma = seg_time + _exit_time(system, seg_start, controls[k + j - 1], seg_len, x_i, r_i) - s
        else:
            sigma = tau_i  # line 16
        s += min(max(sigma, min_advance), tau_i)  # line 18 (with the Zeno floor)
    return triples


def _floored_block(system, target, demo, profiler, s: float, min_advance: float, block: int, triples: list):
    """Vectorized run of consecutive anchors whose advance is the floor ``min_advance``.

    Candidate anchors s, s + f, s + 2f, ... (accumulated exactly like the scalar loop) are
    evaluated in one batch; they are accepted in order while every accepted anchor would be
    floor-advanced by the scalar loop.  Returns (s, next block size, stop) where ``s`` is the
    first anchor left to the scalar loop and ``stop`` signals a loop termination (lines 6/10).
    """
    dt, n = demo.dt, demo.num_steps
    s_arr = np.add.accumulate(np.concatenate([[s], np.full(block, min_advance)]))  # block + 1 times
    k, theta, X = _anchor_states(system, demo, s_arr)
    m_best, r_best, exact = _block_argmax(profiler, np.minimum(k, n - 1), theta, X)
    inside = target.contains(X)
    tau = (m_best - theta) * dt
    nxt = _demo_states(system, demo, s_arr[:-1] + np.minimum(min_advance, np.maximum(tau[:-1], 0.0)))
    left = system.distance(nxt, X[:-1]) >= r_best[:-1]
    accepted = 0
    for j in range(block):
        if k[j] >= n or inside[j]:
            return s_arr[j], block, True
        if not exact[j]:
            break  # the maximizer may lie in the tail: leave this anchor to the scalar loop
        if m_best[j] == 0:
            return s_arr[j], block, True
        if tau[j] < min_advance or not left[j]:
            break  # not a plain floor advance: the scalar loop handles this anchor
        triples.append((X[j].copy(), float(r_best[j]), demo.controls[k[j] : k[j] + m_best[j]].copy(),
                        (1.0 - theta[j]) * dt, float(s_arr[j])))
        accepted += 1
    block = min(2 * block, 512) if accepted == block else max(4, accepted)
    return s_arr[accepted], block, False


def _to_assignment_set(triples: list[tuple], demo_ids: list[int], state_dim: int, dt: float) -> AssignmentSet:
    if not triples:
        return AssignmentSet.empty(state_dim, dt)
    return AssignmentSet(
        centers=np.asarray([tr[0] for tr in triples]),
        radii=np.asarray([tr[1] for tr in triples]),
        controls=[tr[2] for tr in triples],
        dt=dt,
        demo_ids=np.asarray(demo_ids, dtype=int),
        anchor_times=np.asarray([tr[4] for tr in triples]),
        leads=np.asarray([tr[3] for tr in triples]),
    )


def extract_assignments(
    system: HamiltonianSystem,
    target: TargetSet,
    demo: Demonstration,
    demo_id: int,
    v0: float,
    L_H: float,
    L: float,
    lipschitz: str = "global",
    min_advance: float = 1e-4,
) -> AssignmentSet:
    """Inner loop of Algorithm 1 (lines 5-19) on one demonstration (see :func:`_algorithm1_triples`)."""
    triples = _algorithm1_triples(system, target, demo, v0, L_H, L, lipschitz, min_advance)
    return _to_assignment_set(triples, [int(demo_id)] * len(triples), system.state_dim, demo.dt)


def build_certified_assignment_set(
    system: HamiltonianSystem,
    target: TargetSet,
    demos: Sequence[Demonstration],
    chain_cfg,
    H_bar: float,
    max_iter: int = 10,
) -> tuple[AssignmentSet, tuple[float, float, float]]:
    """Algorithm 1 with constants valid on a set X that contains Supp(K).

    Assumption 1 is used on every ball B_{r_i}(x_i) (eqs. (5)-(6)), so X must
    contain them.  Starting from X = {H <= H_X} of :func:`lipschitz_for_demos`,
    X is enlarged to the largest energy reached on the balls and K is rebuilt
    until it fits (larger constants shrink the radii, so this terminates
    quickly).  Returns ``(K, (L_H, L, H_X))``.
    """
    floor = 0.0
    margin = getattr(chain_cfg, "energy_margin", LIPSCHITZ_MARGIN)
    for _ in range(max_iter):
        L_H, L, H_X = lipschitz_for_demos(system, demos, H_bar, floor, margin)
        K = build_assignment_set(system, target, demos, chain_cfg, L_H, L)
        reach = max_energy_on_balls(system, K.centers, K.radii)
        if reach <= H_X:
            return K, (L_H, L, H_X)
        floor = reach
    raise RuntimeError("could not find an energy sublevel set X containing Supp(K)")


def build_assignment_set(
    system: HamiltonianSystem,
    target: TargetSet,
    demos: Sequence[Demonstration],
    chain_cfg,
    L_H: float,
    L: float,
) -> AssignmentSet:
    """Algorithm 1: K = union over demonstrations j (demo_ids = list position) of their triples.

    ``chain_cfg`` provides ``v0``, ``lipschitz``, ``min_anchor_advance`` and
    ``workers`` (:class:`~symplectic_ncp.config.ChainConfig`).  Algorithm 1
    treats every demonstration independently (its outer loop), so with
    ``workers != 1`` the demonstrations are processed in parallel processes
    (0 = one per demonstration, up to the CPU count); the result is identical.
    """
    if not demos:
        raise ValueError("no demonstrations given")
    dt = demos[0].dt
    if any(abs(d.dt - dt) > 1e-12 for d in demos):
        raise ValueError("all demonstrations must share the same sampling step")
    args = (chain_cfg.v0, L_H, L, chain_cfg.lipschitz, chain_cfg.min_anchor_advance)
    workers = int(getattr(chain_cfg, "workers", 1))
    workers = min(len(demos), os.cpu_count() or 1) if workers == 0 else min(workers, len(demos))
    if workers > 1:
        ctx = multiprocessing.get_context("fork")  # children only run numpy code
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            parts = list(pool.map(_algorithm1_triples, *zip(*[(system, target, d, *args) for d in demos])))
    else:
        parts = [_algorithm1_triples(system, target, d, *args) for d in demos]
    triples, ids = [], []
    for j, part in enumerate(parts):
        triples += part
        ids += [j] * len(part)
    return _to_assignment_set(triples, ids, system.state_dim, dt)
