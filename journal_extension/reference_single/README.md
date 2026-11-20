# Single-pendulum model and initial states

- `benchmark.py`: plant, target, signed Reeb-graph distance, exact passive flow
  (Jacobi elliptic functions) and event detection of the single pendulum.
- `initial_states.npz`: the 500 initial states of the single-pendulum trials.

Both are read by `../meeting_single_nmpc_effort_v3`, whose `protocol.json`
records their SHA-256, so they are kept unchanged and under this name. The
folder originally also held an earlier reference study, which is not part of
this release.
