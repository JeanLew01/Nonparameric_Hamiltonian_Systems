Numerical ball-entry localization

This module detects between-step target or assignment-ball crossings using
cubic Hermite interpolation. Squared-distance stationary points partition
each candidate cubic subarc. The verified interface checks a supplied
actual numerical-flow evaluator and returns only states inside the ball.
It never executes an assignment merely because its interpolating arc hit.

Public interfaces
  HermiteArc(xa, xb, fa, fb, h, angle_indices=())
  first_entry(arc, rows, centers, radii, ...)
  verified_first_entry(arc, rows, centers, radii, evaluate, ...)
  interpolation_padding(arc, evaluate, ...)
  BallIndex(centers, radii, angle_indices=(), ...)
  support_candidates(index, arc, padding=0)
  unwrap_relative(points, reference, angle_indices=())

Rows identify an arc for each center/radius pair. The evaluator signature
is evaluate(row, s), with s in [0,1], and must return the actual partial
numerical-flow state in the same angular lift as the arc and centers.
verified_first_entry returns EventBatch(s, states, diagnostics). Infinite
s and NaN states denote no numerically verified hit. Large angular steps
must be subdivided before using periodic support_candidates.

The supplied padding estimator samples the interpolation error. It is not
an all-time error bound. Polynomial root solvers and minimum searches are
floating-point routines, not interval certificates. Thus this module
provides numerical event localization and requires step-refinement audits.
It does not prove complete root detection for arbitrary nonlinear flows.

Focused tests, from the project root:
  PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider \
    experiments/event_detection/test_ball_events.py

The tests include a crossing missed by four fixed Newton iterations,
both-endpoints-outside crossings down to radius 1e-9, grazing and no-hit
cases, wrapped angles, actual-flow membership, displacement padding,
oscillator time-step convergence, and a 50,000-ball prefilter fixture.
See review/round3_independent_review.txt for observed checks and limitations.

The reference_single benchmark uses this module for controlled target
events. Its passive support search uses a separate exact elliptic-flow
parameterization with numerical orbit-ball root localization; this avoids
repeatedly querying tens of thousands of supports at every passive step.
