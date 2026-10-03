"""Batched closed-loop rollouts of the chain policy and of state-feedback baselines.

``simulate_chain_policy`` executes the nonparametric chain policy pi_K
(Definition 7) by concatenation (Remark 2) with the default control u_0 = 0:

* EXEC: a selected snippet u_i runs open loop for its full duration tau_i;
  the policy is not re-evaluated before the snippet ends.  Target entry still
  ends the episode.
* IDLE: outside Supp(K) the zero input is applied and pi_K is evaluated
  continuously in time: the first instant the zero-input flow enters a ball
  B_{r_i}(x_i) is detected on the continuous trajectory (``crossing.py``),
  the state is integrated exactly to that instant and the snippet
  u_{iota_K(x)} starts there (Step 2 of the proof of Theorem 2).  The paper's
  default snippet u_0 : (0, tau_0] -> U is thus taken in the limit tau_0 -> 0.

Every environment keeps its own clock, so after an event the following steps
of that environment follow the snippet's ZOH grid (a first step of length
``K.leads[i]``, then full ``dt`` steps).
Within one step the continuous trajectory is sigma -> RK4(x_a, u, sigma h),
sigma in [0, 1], whose end point is the ordinary RK4 step.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from symplectic_ncp.chain.policy import DEFAULT, NonparametricChainPolicy
from symplectic_ncp.simulation.crossing import BallIndex, HermiteArc, earliest_per_row, first_entry, unwrap_relative
from symplectic_ncp.simulation.result import RolloutResult
from symplectic_ncp.systems.base import HamiltonianSystem, as_batch
from symplectic_ncp.target import TargetSet

_TIME_TOL = 1e-9  # relative to dt: an environment with less remaining time has finished


class RK4Flow:
    """Vector field and RK4 step of ``system`` with J, G cached.

    Bitwise identical to ``system.dynamics`` / ``system.rk4_step`` (same
    operations in the same order); it only avoids per-call overhead in the
    inner loop and returns k1 = f(x_a, u), which the Hermite arc reuses.
    """

    def __init__(self, system: HamiltonianSystem):
        J, G = system.structure_matrices()
        self.system = system
        self.JT = np.asarray(J, dtype=float).T
        self.GT = np.asarray(G, dtype=float).T

    def field(self, X, U) -> np.ndarray:
        return self.system.grad_hamiltonian(X) @ self.JT + U @ self.GT

    def step(self, X, U, h) -> tuple[np.ndarray, np.ndarray]:
        """RK4 step with per-row lengths ``h`` (B,); returns (x_next, f(X, U))."""
        H = np.asarray(h, dtype=float).reshape(-1, 1)
        k1 = self.field(X, U)
        k2 = self.field(X + 0.5 * H * k1, U)
        k3 = self.field(X + 0.5 * H * k2, U)
        k4 = self.field(X + H * k3, U)
        return X + (H / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), k1

    def arc(self, xa, U, h) -> tuple[np.ndarray, HermiteArc]:
        """One RK4 step and its Hermite arc."""
        xb, fa = self.step(xa, U, h)
        return xb, HermiteArc.from_step(self.system, xa, xb, U, h, fa=fa, fb=self.field(xb, U))


def refine_on_flow(flow: RK4Flow, xa, u, h, centers, radii, s, iters: int = 1):
    """Newton-refine entry parameters on the RK4 flow sigma -> RK4(xa, u, sigma h).

    ``s`` comes from the Hermite arc, which differs from the RK4 flow by its
    O(h^4) interpolation error; Newton steps on g(sigma) = ||x(sigma) - c||^2 - r^2
    with dx/dsigma ~ h f(x, u) put the state on the ball boundary to round-off.
    ``centers`` must be unwrapped relative to ``xa``.  Returns the refined
    parameters and the unwrapped states at those instants.
    """
    s = np.asarray(s, dtype=float).copy()
    for _ in range(iters):
        x, _ = flow.step(xa, u, s * h)
        d = x - centers
        g = np.sum(d * d, axis=1) - radii**2
        dg = 2.0 * h * np.sum(d * flow.field(x, u), axis=1)
        ok = (s > 0.0) & (dg < 0.0)
        s = np.where(ok, np.clip(s - g / np.where(ok, dg, -1.0), 0.0, 1.0), s)
    return s, flow.step(xa, u, s * h)[0]


def _target_centers(system, target: TargetSet, xa) -> np.ndarray:
    return unwrap_relative(system, np.broadcast_to(target.center, xa.shape), xa)


def target_entry(system, target: TargetSet, arc: HermiteArc, num_chords: int = 16) -> np.ndarray:
    """First entry parameter of every arc into S_tgt (inf if none), shape (B,)."""
    rows = np.arange(len(arc))
    centers = _target_centers(system, target, arc.xa)
    return first_entry(arc, rows, centers, np.full(rows.size, target.radius), num_chords=num_chords)


def support_entry(system, index: BallIndex, arc: HermiteArc, rows, num_chords: int = 16):
    """Earliest entry of the arcs ``rows`` into any ball of K: (s, ball index or -1), each (len(rows),)."""
    rows = np.asarray(rows, dtype=int)
    if rows.size == 0:
        return np.zeros(0), np.zeros(0, dtype=int)
    # Conservative prefilter: the arc lies in B_q(midpoint), q = half chord + chord deviation.
    pair_row, pair_ball = index.pairs(arc.midpoint()[rows], arc.enclosing_radius()[rows])
    if pair_row.size == 0:
        return np.full(rows.size, np.inf), np.full(rows.size, -1, dtype=int)
    centers = unwrap_relative(system, index.centers[pair_ball], arc.xa[rows[pair_row]])
    s = first_entry(arc, rows[pair_row], centers, index.radii[pair_ball], num_chords=num_chords)
    return earliest_per_row(rows.size, pair_row, s, pair_ball)


def simulate_chain_policy(
    system: HamiltonianSystem,
    target: TargetSet,
    policy: NonparametricChainPolicy,
    x0,
    horizon: float,
    dt: float,
    num_chords: int = 16,
    record: bool = False,
) -> RolloutResult:
    """Event-driven execution of pi_K (Remark 2) from every initial state in ``x0`` (B, n).

    ``record=True`` also returns ``trace``: for every trajectory a dict of arrays
    ``t`` (T,), ``x`` (T, n), ``u`` (T, m) and ``snippet`` (T,), where row j is
    the state at time t[j] and u / snippet are the input and assignment index
    (DEFAULT = zero input) applied on (t[j-1], t[j]].

    ``extras`` (arrays of shape (B,)): ``n_snippets`` executed snippets,
    ``exec_time`` / ``idle_time`` time spent in EXEC / IDLE,
    ``max_idle_interval`` the longest zero-input excursion that ended by
    entering Supp(K) (an empirical T_1 of Theorem 4; the excursion from t = 0
    counts), ``open_idle_interval`` the length of a zero-input excursion still
    open at the horizon (right-censored return time; 0 otherwise),
    ``fallback_entries`` support entries at which iota_K of the
    integrated entry state returned the default index for numerical reasons
    (the crossed ball is executed instead), ``final_energy`` H at the end of
    the episode (target entry or horizon).
    """
    K = policy.K
    if abs(K.dt - dt) > 1e-12:
        raise ValueError(f"snippet step {K.dt} differs from the simulation step {dt}")
    index = BallIndex(system, K.centers, K.radii)
    flow = RK4Flow(system)
    X = system.wrap(as_batch(x0, system.state_dim))
    B, n, m = X.shape[0], system.state_dim, system.control_dim
    steps = K.steps
    offsets = np.concatenate([[0], np.cumsum(steps)[:-1]]).astype(int)
    flat = np.vstack(K.controls).reshape(-1, m)
    t_end = horizon - _TIME_TOL * dt

    t = np.zeros(B)
    success = np.zeros(B, dtype=bool)
    reach_time = np.full(B, np.inf)
    active = np.full(B, DEFAULT, dtype=int)  # running snippet, DEFAULT = IDLE (zero input)
    k = np.zeros(B, dtype=int)  # step inside the running snippet
    idle_start = np.zeros(B)
    extras = {
        "n_snippets": np.zeros(B, dtype=int),
        "exec_time": np.zeros(B),
        "idle_time": np.zeros(B),
        "max_idle_interval": np.zeros(B),
        "fallback_entries": np.zeros(B, dtype=int),
    }

    def switch(envs: np.ndarray, selected: np.ndarray) -> None:
        """Apply u_{selected} from the current time of ``envs`` (DEFAULT -> zero input)."""
        active[envs] = selected
        k[envs] = 0
        execs = selected != DEFAULT
        extras["n_snippets"][envs[execs]] += 1
        idle_start[envs[~execs]] = t[envs[~execs]]

    # t = 0: target membership, then u_0 = pi_K(x_0).
    done = target.contains(X)
    success[done] = True
    reach_time[done] = 0.0
    envs = np.flatnonzero(~done)
    if envs.size:
        switch(envs, policy.select(X[envs]))
    trace = [[(0.0, X[b].copy(), np.zeros(m), DEFAULT)] for b in range(B)] if record else None

    while True:
        run = np.flatnonzero(~done & (t < t_end))
        if run.size == 0:
            break
        xa = X[run]
        is_exec = active[run] != DEFAULT
        h = np.full(run.size, float(dt))
        first = np.flatnonzero(is_exec & (k[run] == 0))
        h[first] = K.leads[active[run[first]]]  # the first ZOH step of a snippet lasts lead_i <= dt
        h = np.minimum(h, horizon - t[run])
        U = np.zeros((run.size, m))
        U[is_exec] = flat[offsets[active[run][is_exec]] + k[run][is_exec]]
        applied = active[run].copy()
        xb, arc = flow.arc(xa, U, h)

        # Earliest events on the continuous trajectory of this step.
        s_tgt = target_entry(system, target, arc, num_chords)
        s_sup = np.full(run.size, np.inf)
        ball = np.full(run.size, -1, dtype=int)
        idle = np.flatnonzero(~is_exec)
        s_sup[idle], ball[idle] = support_entry(system, index, arc, idle, num_chords)
        tgt_event = np.isfinite(s_tgt) & (s_tgt <= s_sup)
        sup_event = ~tgt_event & np.isfinite(s_sup)

        ev = np.flatnonzero(tgt_event | sup_event)
        if ev.size:
            is_tgt = tgt_event[ev]
            centers = np.empty((ev.size, n))
            radii = np.empty(ev.size)
            centers[is_tgt] = _target_centers(system, target, xa[ev[is_tgt]])
            radii[is_tgt] = target.radius
            b = ball[ev[~is_tgt]]
            centers[~is_tgt] = unwrap_relative(system, K.centers[b], xa[ev[~is_tgt]])
            radii[~is_tgt] = K.radii[b]
            s_ev = np.where(is_tgt, s_tgt[ev], s_sup[ev])
            s_ev, x_ev = refine_on_flow(flow, xa[ev], U[ev], h[ev], centers, radii, s_ev)
            e = run[ev]
            elapsed = s_ev * h[ev]
            ex = is_exec[ev]
            extras["exec_time"][e[ex]] += elapsed[ex]
            extras["idle_time"][e[~ex]] += elapsed[~ex]
            t[e] += elapsed
            X[e] = system.wrap(x_ev)

            # Target entry ends the episode at the entry instant.
            et = e[is_tgt]
            reach_time[et] = t[et]
            success[et] = True
            done[et] = True

            # Support entry from IDLE: switch to iota_K at the integrated entry state.
            es = e[~is_tgt]
            if es.size:
                idle_len = t[es] - idle_start[es]
                extras["max_idle_interval"][es] = np.maximum(extras["max_idle_interval"][es], idle_len)
                selected = policy.select(X[es])
                fallback = selected == DEFAULT
                extras["fallback_entries"][es[fallback]] += 1
                selected[fallback] = b[fallback]
                switch(es, selected)

        # Ordinary full step.
        adv = np.flatnonzero(~tgt_event & ~sup_event)
        if adv.size:
            e = run[adv]
            X[e] = system.wrap(xb[adv])
            t[e] += h[adv]
            ex = is_exec[adv]
            extras["exec_time"][e[ex]] += h[adv][ex]
            extras["idle_time"][e[~ex]] += h[adv][~ex]
            ee = e[ex]
            k[ee] += 1
            finished = ee[k[ee] >= steps[active[ee]]]
            if finished.size:  # snippet completed: evaluate pi_K at its end state (Remark 2)
                switch(finished, policy.select(X[finished]))
        if record:
            for row, e in enumerate(run):
                trace[e].append((float(t[e]), X[e].copy(), U[row].copy(), int(applied[row])))

    # Zero-input excursions still open at the horizon never returned to Supp(K): right-censored.
    open_idle = ~done & (active == DEFAULT)
    extras["open_idle_interval"] = np.where(open_idle, t - idle_start, 0.0)
    extras["final_energy"] = system.hamiltonian(X)
    if record:
        trace = [
            {
                "t": np.asarray([r[0] for r in rows]),
                "x": np.asarray([r[1] for r in rows]),
                "u": np.asarray([r[2] for r in rows]),
                "snippet": np.asarray([r[3] for r in rows], dtype=int),
            }
            for rows in trace
        ]
    return RolloutResult(
        success=success, reach_time=reach_time, horizon=float(horizon), extras=extras, trace=trace
    )


def simulate_feedback_policy(
    system: HamiltonianSystem,
    target: TargetSet,
    policy_fn: Callable[[np.ndarray], np.ndarray],
    x0,
    horizon: float,
    dt: float,
    control_period: float,
    num_chords: int = 16,
    record: bool = False,
) -> RolloutResult:
    """Closed loop with a state-feedback law (e.g. behavior cloning).

    ``u = policy_fn(X)`` is recomputed every ``control_period`` and held
    (ZOH), clipped to the input bounds; target entry is detected on the
    continuous trajectory of every ``dt`` step, exactly as for the chain policy.

    Action chunks (e.g. diffusion policy): if ``policy_fn`` returns a 3-D
    array (B, Ta, m), chunk element j is held during the j-th control period
    after the query and the trajectory queries again after Ta periods.  The
    chunk position is tracked per trajectory, so each call receives only the
    running trajectories whose chunk is exhausted.  A 2-D output (B, m) is a
    chunk of length one: it is queried on all running trajectories every
    control period (the original behavior).
    ``extras["final_energy"]`` holds H at the end of each episode.
    ``record=True`` also returns ``trace`` (per trajectory: ``t``, ``x``, ``u``
    with u applied on (t[j-1], t[j]], as in :func:`simulate_chain_policy`).
    """
    ratio = int(round(control_period / dt))
    if ratio < 1 or abs(ratio * dt - control_period) > 1e-9:
        raise ValueError("control_period must be a positive multiple of dt")
    flow = RK4Flow(system)
    X = system.wrap(as_batch(x0, system.state_dim))
    B, m = X.shape[0], system.control_dim
    success = target.contains(X)
    reach_time = np.where(success, 0.0, np.inf)
    U = np.zeros((B, m))
    chunk = np.zeros((B, 1, m))  # current (clipped) action chunk of every trajectory
    pos = np.zeros(B, dtype=int)  # index of the next chunk element to apply
    length = np.zeros(B, dtype=int)  # chunk length (0: query at the next control instant)
    num_steps = int(np.ceil(horizon / dt - _TIME_TOL))
    trace = [[(0.0, X[b].copy(), np.zeros(m))] for b in range(B)] if record else None
    for step in range(num_steps):
        run = np.flatnonzero(~success)
        if run.size == 0:
            break
        t0 = step * dt
        h = np.full(run.size, min(dt, horizon - t0))
        if step % ratio == 0:
            need = run[pos[run] >= length[run]]
            if need.size:
                out = np.asarray(policy_fn(X[need]), dtype=float)
                if out.ndim == 3:
                    if out.shape[0] != need.size or out.shape[2] != m or out.shape[1] < 1:
                        raise ValueError(f"action chunk must have shape ({need.size}, Ta, {m}), got {out.shape}")
                    L = out.shape[1]
                    out = system.clip_control(out.reshape(-1, m)).reshape(need.size, L, m)
                else:
                    L = 1
                    out = system.clip_control(out.reshape(need.size, m))[:, None, :]
                if chunk.shape[1] < L:
                    chunk = np.concatenate([chunk, np.zeros((B, L - chunk.shape[1], m))], axis=1)
                chunk[need, :L] = out
                length[need] = L
                pos[need] = 0
            U[run] = chunk[run, pos[run]]
            pos[run] += 1
        xa, Ur = X[run], U[run]
        xb, arc = flow.arc(xa, Ur, h)
        s_tgt = target_entry(system, target, arc, num_chords)
        hit = np.flatnonzero(np.isfinite(s_tgt))
        if hit.size:
            centers = _target_centers(system, target, xa[hit])
            radii = np.full(hit.size, target.radius)
            s_hit, xb[hit] = refine_on_flow(flow, xa[hit], Ur[hit], h[hit], centers, radii, s_tgt[hit])
            reach_time[run[hit]] = t0 + s_hit * h[hit]
            success[run[hit]] = True
        X[run] = system.wrap(xb)
        if record:
            for row, e in enumerate(run):
                t_e = reach_time[e] if success[e] else t0 + h[row]
                trace[e].append((float(t_e), X[e].copy(), Ur[row].copy()))
    if record:
        trace = [
            {"t": np.asarray([r[0] for r in rows]), "x": np.asarray([r[1] for r in rows]),
             "u": np.asarray([r[2] for r in rows])}
            for rows in trace
        ]
    return RolloutResult(
        success=success, reach_time=reach_time, horizon=float(horizon),
        extras={"final_energy": system.hamiltonian(X)}, trace=trace,
    )
