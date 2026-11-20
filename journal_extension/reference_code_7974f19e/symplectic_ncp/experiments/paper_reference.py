"""Numbers reported in Section IV of the paper (Figs. 2 and 3).

Values stated in the text are exact; all other values were read off the bar
charts of Figs. 2-3 and are approximate (flagged by ``exact=False``).  The
error bars of Fig. 2b/3b are read as one standard deviation of the reach time
(the paper does not say what they are).

Known inconsistency in the paper: the protocol assigns the horizon to
unsuccessful runs, so the mean reach time is at least
``(1 - success_rate) * horizon``.  Vanilla BC on the spring-mass system has
success 0.062 / 0.19 (text) for M = 1 / 2, i.e. a mean reach time of at least
18.8 s / 16.2 s, yet Fig. 2b shows about 14.1 s / 9.9 s.  The pendulum values
are consistent with the protocol (e.g. chain M = 1: 114.71 s >= 0.652 * 150 s).
Fig. 3b shows about 108 s for the chain policy at M = 1 where the text says
114.71 s; the text value is used here.
"""

from __future__ import annotations

from dataclasses import dataclass

NUM_DEMOS = (1, 2, 3, 4, 5)
METHODS = ("chain", "bc")
METHOD_LABELS = {"chain": "Chain Policy", "bc": "Vanilla BC"}


@dataclass(frozen=True)
class ReferenceSeries:
    """One reported quantity for M = 1..5; ``exact[k]`` is True for values stated in the text."""

    values: tuple[float, ...]
    exact: tuple[bool, ...]

    def to_dict(self) -> dict:
        return {"values": list(self.values), "exact": list(self.exact)}


def _series(values, exact) -> ReferenceSeries:
    if isinstance(exact, bool):
        exact = (exact,) * len(values)
    return ReferenceSeries(tuple(float(v) for v in values), tuple(bool(e) for e in exact))


# Method -> quantity -> series.  Quantities: success_rate, mean_reach_time, std_reach_time.
PAPER_RESULTS: dict[str, dict] = {
    "spring_mass": {
        "figure": "Fig. 2",
        "horizon": 20.0,
        "chain": {
            "success_rate": _series([1.0, 1.0, 1.0, 1.0, 1.0], True),
            # 5.54 s at M = 1 and "about 2.94 s" (saturated, M >= 3) are in the text.
            "mean_reach_time": _series([5.54, 4.0, 2.94, 2.94, 2.94], (True, False, True, True, True)),
            "std_reach_time": _series([2.3, 1.45, 1.35, 1.4, 1.4], False),
        },
        "bc": {
            "success_rate": _series([0.062, 0.19, 1.0, 1.0, 1.0], (True, True, False, False, False)),
            "mean_reach_time": _series([14.1, 9.95, 3.6, 3.6, 1.4], False),
            "std_reach_time": _series([3.45, 5.4, 3.0, 1.7, 0.4], False),
        },
    },
    "single_pendulum": {
        "figure": "Fig. 3",
        "horizon": 150.0,
        "chain": {
            "success_rate": _series([0.348, 0.678, 1.0, 1.0, 1.0], True),
            "mean_reach_time": _series([114.71, 68.0, 22.0, 18.0, 13.16], (True, False, False, False, True)),
            "std_reach_time": _series([40.0, 53.0, 5.5, 8.0, 7.5], False),
        },
        "bc": {
            "success_rate": _series([0.008, 0.53, 0.418, 0.65, 0.744], True),
            "mean_reach_time": _series([149.0, 78.0, 90.0, 61.5, 47.5], False),
            "std_reach_time": _series([12.5, 61.0, 65.0, 45.5, 40.5], False),
        },
    },
}


def paper_reference(system_name: str) -> dict | None:
    """JSON-serializable reference numbers for ``system_name`` (None if the paper has none)."""
    ref = PAPER_RESULTS.get(system_name)
    if ref is None:
        return None
    out = {"figure": ref["figure"], "horizon": ref["horizon"], "num_demos": list(NUM_DEMOS)}
    for method in METHODS:
        out[method] = {q: s.to_dict() for q, s in ref[method].items()}
    return out


def reference_value(system_name: str, method: str, quantity: str, M: int) -> tuple[float, bool] | None:
    """(value, exact) reported for ``M`` demonstrations, or None when not reported."""
    ref = PAPER_RESULTS.get(system_name)
    if ref is None or M not in NUM_DEMOS:
        return None
    series: ReferenceSeries = ref[method][quantity]
    k = NUM_DEMOS.index(M)
    return series.values[k], series.exact[k]
