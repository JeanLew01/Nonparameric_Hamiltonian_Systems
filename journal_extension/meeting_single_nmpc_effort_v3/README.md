# Single pendulum, third version: one push per period, transport of whole pushes (18 Nov 2026)

Remark of 18 Nov 2026 on Figure 4(b): for K = 1, 2, 4 the success curves rose
at different rates at first and then shared one slow tail; only K = 8 reached
100% sooner. A more reasonable result would be a different rate for each K.

## What was wrong (measured in `../meeting_single_nmpc_effort_v2`, M = 3)

The curve mixed two kinds of states. Rotation states gained with every
doubling of K (mean 26.3, 16.5, 11.0, 8.3 s). Libration states did not
(22.7, 21.8, 22.2, 10.7 s), and they are the last to arrive.

- K = 2: the swing-up demonstration pushed at both passes through the bottom,
  so the half-period placement sat where the demonstration's next push
  already was. Bursts per trial 10.6 and 10.6, wait between bursts 1.69 and
  1.68 s.
- K = 4: a visit to a quarter-period placement completed 1.7 of the 4.5
  snippets of a push; the rest was done at the bottom. Pushes were split,
  not added (28.4 bursts with waits of 0.62 s against 10.6 with 1.69 s).

## Two changes

1. `expert.py`: in the libration demonstrations the torque has one sign
   (`TORQUE_SIGN`), so every demonstration pushes once per period, as the
   braking rotations already did. From rest at the bottom the one-sided
   problem is degenerate (the optimizer postpones the first push from solve to
   solve); the first push is applied by its known form.
2. `construct.py`: a transported assignment replays the remainder of the push
   that contains its anchor (`push_remainder`), by one constant torque that
   matches the energy at the end of the push. The original assignments are
   still those of Algorithm 1.

Unchanged: expert bound 2 N m, eta, start states, Algorithm 1 with V,
selector, guard, initial states, evaluation.

## Results (500 states, M = 3)

| K | first wait | wait between actions | mean reaching time | last state arrives | effort |
| --- | --- | --- | --- | --- | --- |
| 1 (D) | 1.030 s | 24.83 s | 31.53 s | 102.5 s | 14.8 |
| 2 | 0.507 s | 11.50 s | 17.73 s | 53.7 s | 33.4 |
| 4 | 0.261 s | 4.22 s | 10.07 s | 28.6 s | 108.7 |
| 8 | 0.132 s | 2.10 s | 7.64 s | 18.1 s | 83.7 |

The success curve of a larger K is never below that of a smaller one.
Rotation states: 26.3, 15.7, 8.8, 7.6 s; libration states: 41.0, 21.4, 12.3,
7.7 s. D+T (K = 8) is faster than D for 492 states and slower for 3 (at most
0.13 s). Against K = 4, K = 8 is slower for 136 single trials (at most
1.6 s) although its curve is never below. Most of the effort at K = 4 and 8
is spent at the quarter-period placements (85% and 70%).

D by 20 s for M = 1..5: 43, 101, 138, 145, 201; D+T (K = 8): 154, 324, 500,
500, 500; by 120 s both 154, 324, 500, 500, 500. D alone is slower than in
the second version (libration demonstrations now take 105.5 and 40.4 s).

Demonstrations: 42.2, 42.2, 105.5, 40.4, 40.4 s with 32, 32, 32, 10, 10
pushes. Assignments 64,563; transport 451,941 requested, 448,528 accepted.
Step halving on 500 trials: no label change, times within 1.3e-10 s.

    ./run_all.sh && python refinement_report.py

Figures: `../sync_figures.sh` (20 s / 120 s, time axis 110 s). Number
checker: `../verify_nmpc_text.py` (49 statements); the checker of the previous
text is `../verify_nmpc_text_6d5b034.py`.
