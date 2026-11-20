# Section coverage for eight half-decade gains (15 Nov 2026, evening)

Rerun of `../meeting_dynamics_kappa_log` with the gains requested on
15 Nov 2026: kappa in {0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1}. The gain 10
is dropped and 0.003, 0.03, 0.3 are added. Nothing else changes: H = 15 J,
section q2 = 0 with p2 > 0, 64 x 128 grid (5740 allowed cells), horizon
5000 s, 40 plotted times, 5 s passive prefix, the same six initial states with
bounded passive orbits, and the same recorded exploration stream per state
(modes 0, +kappa, -kappa with probability 1/3, durations uniform on
[0.25, 0.75] s). See the README of the earlier study for how the initial
states were chosen.

All 96 trajectories (8 gains x 6 states x 2 step sizes) were computed anew
(`run.py`, one minute on 8 workers). The 60 trajectories whose gain also
appears in the earlier study are bit-identical to it.

## Result (5000 s)

| kappa | Mean coverage, 1 ms | Range over six states | Largest force (N m) | Mean coverage, 2 ms |
| --- | --- | --- | --- | --- |
| 0 | 5.1% | 3.4-6.7% | 0 | 5.2% |
| 0.001 | 8.5% | 3.7-24.1% | 0.01 | 6.0% |
| 0.003 | 9.7% | 4.5-25.3% | 0.02 | 8.7% |
| 0.01 | 14.3% | 6.7-27.5% | 0.08 | 12.5% |
| 0.03 | 23.4% | 12.5-29.3% | 0.23 | 29.2% |
| 0.1 | 30.5% | 27.4-32.5% | 0.77 | 29.8% |
| 0.3 | 31.6% | 30.3-32.5% | 2.32 | 32.3% |
| 1 | 32.3% | 31.7-32.6% | 7.75 | 32.1% |

At the 1 ms step the mean coverage increases with every gain. The force bound
of 20 N m is never reached.

Read the middle gains with care. For kappa between 0.001 and 0.03 a
trajectory first leaks slowly out of its passive region and then covers the
section at the full rate, so the coverage at 5000 s depends on when each of
the six trajectories happens to leave, and that time is sensitive to the
integration step. The clearest case is kappa = 0.03: 23.4% at 1 ms and 29.2%
at 2 ms. The ordering between neighbouring gains in this range is therefore
supported by the means but not by every single trajectory. From kappa = 0.1
on, all six trajectories fill the section and the two steps agree within
1 percentage point.

## Files

- `run.py`, `diagnostic.py` (byte-identical kernel), `protocol.json`,
  `initial_states.npz`, `fine/` (1 ms), `main/` (2 ms).
- `summarize.py` -> `summary.json`, `coverage_checkpoints.csv`,
  `poincare_diagnostic.pdf/png` (maps for kappa = 0, 0.01, 1).
- `../paper_figures.py` draws the manuscript version of the figure from this
  folder (set `HCP_KAPPA_STUDY=meeting_dynamics_kappa_log` for the earlier
  six gains).
