# Double pendulum, NMPC demonstrations simulated with RK4 (16 Nov 2026)

Requested on 16 Nov 2026: the two pendulums should use the same integrator
for the demonstrations (the single pendulum used RK4, this study DOP853).

Copy of `../meeting_double_nmpc` with three changes:

- `demonstrations.py`: the closed-loop plant is RK4 at a 5 ms step (five steps
  per 25 ms control interval), as for the single-pendulum expert. The 5 ms
  states are stored (`fine_states`).
- `construct.py`: the 5 ms grid of a demonstration is that stored simulation
  (recomputed and compared bit by bit), and the passive shifts of the anchors
  use RK4 at a 1 ms step (`drift_rk4`) instead of DOP853.
- `verify_demonstrations.py`: adds a replay with RK4 at 0.625 ms, the check
  used for the single pendulum.

DOP853 remains only in two extra checks that the paper no longer cites
(`verify_demonstrations.py`, `check.py`). The paper cites `check_rk4.py`
(17 Nov 2026): replay of all 26,137 finished actions with RK4 at one eighth of
the execution step, same result (end-of-action V within 3.2e-5 J, no violation). Controller, start states, support test, executor, guard,
exploration draws and the 500 initial states are unchanged.

The demonstrations barely move (same durations 1.2 / 1.4 / 1.2 s; inputs
differ from the DOP853 ones by at most 2e-4 N m, states by 8e-6), and the
libraries have the same sizes (223/669/1115/2007 and 152/450/743/1252).
Outcomes shift as far as the long chaotic trials are sensitive:

| | up-up hits D, D+E, D+T, D+E+T | restricted mean time (s) | down-down hits | restricted mean time (s) |
| --- | --- | --- | --- | --- |
| DOP853 demonstrations | 344, 500, 397, 500 | 3644, 400, 2582, 113 | 227, 500, 325, 500 | 6004, 459, 3740, 87 |
| RK4 demonstrations (this study) | 344, 500, 395, 500 | 3643, 412, 2668, 116 | 222, 500, 356, 500 | 6054, 474, 3190, 89 |

Step halving on twelve states per target: 10 of 96 labels change (1 up D+T,
6 down D, 3 down D+T). Independent DOP853 replay of 26,137 finished actions:
no violation.

    python demonstrations.py && ./run_all.sh     # about 45 min on 16 cores

Figure 6: `../sync_figures.sh`. Number checker: `../verify_nmpc_text.py`.
