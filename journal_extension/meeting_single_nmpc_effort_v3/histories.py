"""Reconstruct sampled states from executed actions, never endpoint chords."""
from common import *


def passive_values(x, times):
    if not len(times):
        return np.empty((0, 2))
    try:
        return np.asarray(b.passive(x, times))
    except ValueError:
        # Only an exact equilibrium can be held constant. A separatrix needs
        # its own integrator and must not be silently treated as stationary.
        assert np.linalg.norm([x[1] / b.I, -b.A * np.sin(x[0])]) < 1e-12
        return np.repeat(np.asarray(x)[None, :], len(times), axis=0)


def reconstruct(initial, row, actions, payload, subdivisions, times):
    states = np.full((len(times), 2), np.nan)
    clock = 0.
    current = np.asarray(initial).copy()
    max_passive_energy_error = 0.
    max_start_error = 0.
    max_endpoint_error = 0.
    for action in actions:
        start, _, duration, source = action[:4]
        source = int(source)
        before = np.asarray(action[7:9])
        after = np.asarray(action[9:11])
        mask = (times >= clock) & (times < start)
        states[mask] = passive_values(current, times[mask] - clock)
        if np.any(mask):
            err = np.max(np.abs(np.array([b.energy(x) for x in states[mask]]) - b.energy(current)))
            max_passive_energy_error = max(max_passive_energy_error, float(err))
        predicted = passive_values(current, np.array([max(0., start - clock)]))[0]
        max_start_error = max(max_start_error, float(b.distance(predicted, before)))
        parent = int(payload['parent'][source])
        u = payload['control_flat'][payload['offsets'][parent]:payload['offsets'][parent + 1]]
        trace, hs, us = b.primitive_trace(before, u, float(payload['leads'][source]), float(payload['dt']), subdivisions)
        ts = np.r_[0., np.cumsum(hs)]

        def active_state(elapsed):
            if elapsed >= ts[-1]:
                assert elapsed - ts[-1] < 2e-10
                return trace[-1]
            k = min(len(us) - 1, int(np.searchsorted(ts, elapsed, side='right') - 1))
            return b.step(trace[k], us[k], max(0., elapsed - ts[k]))

        mask = (times >= start) & (times < start + duration)
        for j in np.flatnonzero(mask):
            states[j] = active_state(float(times[j] - start))
        max_endpoint_error = max(max_endpoint_error, float(b.distance(active_state(duration), after)))
        clock = start + duration
        current = after
    completion = float(row['time'])
    if row['success']:
        mask = (times >= clock) & (times < completion)
        states[mask] = passive_values(current, times[mask] - clock)
        states[times >= completion] = np.asarray(row['final'])
    else:
        mask = times >= clock
        states[mask] = passive_values(current, times[mask] - clock)
    if np.any(mask):
        err = np.max(np.abs(np.array([b.energy(x) for x in states[mask]]) - b.energy(current)))
        max_passive_energy_error = max(max_passive_energy_error, float(err))
    assert np.isfinite(states).all()
    assert b.distance(states[0], initial) < 1e-10
    assert max_start_error < 1e-8, max_start_error
    assert max_endpoint_error < 1e-8, max_endpoint_error
    signed = np.array([b.score(x) for x in states])
    gaps = np.array([b.target_gap(x) for x in states])
    final_error = float(b.distance(states[-1], np.asarray(row['final'])))
    assert final_error < 1e-8, final_error
    return states, signed, gaps, np.array([max_start_error, max_endpoint_error, max_passive_energy_error, final_error])
