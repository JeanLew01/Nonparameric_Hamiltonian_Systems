"""Equivalent grouped orbit/ball intersections for the frozen event policy.

The numerical root solves, membership tests, event selection and tie breaking
are unchanged. Only repeated label-array scans and scalar midpoint elliptic
evaluations are replaced by a single sort and vectorized elliptic evaluation.
"""
from common import *
from scipy.special import ellipkinc
from numba import njit


@njit(cache=True)
def membership(states, centers, radii, labels):
    answer = np.empty(len(states), np.bool_)
    for i in range(len(states)):
        answer[i] = b.distance(states[i], centers[labels[i]]) <= radii[labels[i]]
    return answer


class FastLibrary(b.Library):
    def first_access(self, x):
        idx = self.inside(x)
        if idx >= 0:
            return 0., x.copy(), idx, 0
        par = b.orbit_parameters(x)
        e, m, period, rate, z, direction = par
        ids = self.candidates(e, direction)
        if not len(ids):
            return np.inf, x.copy(), -1, 0
        points, labels = b.intersection_points(e, self.centers, self.radii,
                                               self.angles, self.values, self.counts, ids)
        if direction:
            keep = direction * points[:, 1] > 0
            points, labels = points[keep], labels[keep]
        if not len(labels):
            return np.inf, x.copy(), -1, len(ids)
        if direction:
            phases = (direction * ellipkinc(points[:, 0] / 2, m)) % period
        else:
            alpha = np.arctan2(np.sin(points[:, 0] / 2) / np.sqrt(m),
                               points[:, 1] / (2 * b.I * b.OMEGA * np.sqrt(m)))
            phases = ellipkinc(alpha, m) % period
        order = np.lexsort((phases, labels))
        phases, labels = phases[order], labels[order]
        unique = np.r_[True, (labels[1:] != labels[:-1]) | (phases[1:] != phases[:-1])]
        phases, labels = phases[unique], labels[unique]
        first = np.r_[True, labels[1:] != labels[:-1]]
        start = np.maximum.accumulate(np.where(first, np.arange(len(labels)), 0))
        last = np.r_[labels[:-1] != labels[1:], True]
        following = np.roll(phases, -1)
        following[last] = phases[start[last]] + period
        midpoints = (phases + following) / 2
        inside_mid = membership(b.orbit_state(par, midpoints), self.centers, self.radii, labels)
        inside_endpoint = membership(b.orbit_state(par, phases), self.centers, self.radii, labels)
        retain = inside_mid | inside_endpoint
        entries = (phases - z) % period / rate
        inside = np.where(inside_mid, entries + (following - phases) / (2 * rate), entries)
        entries, inside, labels = entries[retain], inside[retain], labels[retain]
        order = np.lexsort((labels, inside, entries))
        for k in order:
            entry, high, idx = float(entries[k]), float(inside[k]), int(labels[k])
            r = self.radii[idx]
            low = entry
            if b.distance(b.orbit_state(par, z + rate * low), self.centers[idx]) <= r:
                low = max(0, entry - max(1e-12, 1e-4 * r / max(1, np.linalg.norm(x))))
            if b.distance(b.orbit_state(par, z + rate * high), self.centers[idx]) > r:
                continue
            if b.distance(b.orbit_state(par, z + rate * low), self.centers[idx]) > r:
                for _ in range(48):
                    mid = (low + high) / 2
                    if b.distance(b.orbit_state(par, z + rate * mid), self.centers[idx]) <= r:
                        high = mid
                    else:
                        low = mid
            else:
                high = low
            y = b.orbit_state(par, z + rate * high)
            if b.distance(y, self.centers[idx]) > r:
                raise RuntimeError('Orbit entry lost membership')
            selected = self.inside(y)
            return high, y, selected if selected >= 0 else idx, len(ids)
        return np.inf, x.copy(), -1, len(ids)
