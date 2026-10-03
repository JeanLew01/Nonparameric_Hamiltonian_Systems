# Results (Section IV + ablations)

Success rate / average reach time in seconds (unsuccessful runs count as the horizon; mean over the test states, learned methods pooled over their seeds).  Paper values in parentheses (`~` = read off the paper's figure).  PPO uses no demonstrations, so its numbers do not depend on M.  Theorem 2 diagnostics for K_M: C1 viol. = sampled violation fraction of the local energy decrease (Condition 1); C2 cov. = covered fraction of {E : Delta H(E) <= c} (Condition 2, target band included); C3 cov. = smallest per-ergodic-component coverage (Condition 3).

## Single pendulum

500 initial states with H <= 160.0, horizon 150.0 s; seeds: BC [0, 1, 2, 3, 4], DP [0, 1, 2], PPO [0, 1, 2].

| M | N(K_M) | Chain Policy | Vanilla BC | Diffusion Policy | PPO (no demos) | C1 viol. | C2 cov. | C3 cov. |
|---|---|---|---|---|---|---|---|---|
| 1 | 22715 | 0.308 / 105 (0.348 / 115) | 0.0172 / 147 (0.008 / ~149) | 0.006 / 149 | 1 / 2.91 | 0 | 0.7548 | 0 (libration) |
| 2 | 45430 | 0.648 / 55.4 (0.678 / ~68) | 0.0176 / 147 (0.53 / ~78) | 0.00467 / 149 | 1 / 2.91 | 0 | 0.7548 | 0 (libration) |
| 3 | 46473 | 1 / 4.52 (1 / ~22) | 0.244 / 115 (0.418 / ~90) | 0.983 / 10.6 | 1 / 2.91 | 0 | 0.9997 | 1 (libration) |
| 4 | 47607 | 1 / 4.38 (1 / ~18) | 0.296 / 108 (0.65 / ~61.5) | 1 / 5.24 | 1 / 2.91 | 0 | 0.9997 | 1 (libration) |
| 5 | 48741 | 1 / 4.38 (1 / 13.2) | 0.729 / 46.2 (0.744 / ~47.5) | 1 / 4.18 | 1 / 2.91 | 0 | 0.9997 | 1 (libration) |

## Spring-mass

500 initial states with H <= 2.0, horizon 20.0 s; seeds: BC [0, 1, 2, 3, 4], DP [0, 1, 2], PPO [0, 1, 2].

| M | N(K_M) | Chain Policy | Vanilla BC | Diffusion Policy | PPO (no demos) | C1 viol. | C2 cov. | C3 cov. |
|---|---|---|---|---|---|---|---|---|
| 1 | 412 | 1 / 3.95 (1 / 5.54) | 0.837 / 4.51 (0.062 / ~14.1) | 0.943 / 3.28 | 1 / 0.382 | 0 | 0.9975 | 1 (single) |
| 2 | 824 | 1 / 2.39 (1 / ~4) | 1 / 1.16 (0.19 / ~9.95) | 1 / 1.25 | 1 / 0.382 | 0 | 0.9975 | 1 (single) |
| 3 | 1615 | 1 / 2.17 (1 / 2.94) | 1 / 1.17 (~1 / ~3.6) | 1 / 1.1 | 1 / 0.382 | 0 | 0.9975 | 1 (single) |
| 4 | 2406 | 1 / 1.99 (1 / 2.94) | 1 / 1.15 (~1 / ~3.6) | 1 / 1.04 | 1 / 0.382 | 0 | 0.9975 | 1 (single) |
| 5 | 2806 | 1 / 1.96 (1 / 2.94) | 1 / 1.12 (~1 / ~1.4) | 1 / 1.04 | 1 / 0.382 | 0 | 0.9975 | 1 (single) |

## Training time [s]

Wall-clock time of the training call on the same machine (chain policy: Lipschitz constants + Algorithm 1 on the first M demonstrations, CPU, one process per demonstration; BC: CPU; diffusion policy and PPO: as reported in the device column).  Demonstration generation (shared by chain, BC and DP) is excluded; PPO uses no demonstrations.

**Single pendulum**

| Method | device | M=1 | M=2 | M=3 | M=4 | M=5 |
|---|---|---|---|---|---|---|
| Chain Policy | cpu | 10.9 | 12.5 | 12.4 | 14.1 | 12.4 |
| Vanilla BC | cpu | 0.324 ± 0.054 | 0.45 ± 0.016 | 0.51 ± 0.035 | 0.687 ± 0.062 | 0.683 ± 0.049 |
| Diffusion Policy | cuda | 17.4 ± 0.093 | 17.2 ± 0.082 | 17.3 ± 0.094 | 17.3 ± 0.043 | 17.2 ± 0.045 |
| PPO (no demos) | cuda | 147 ± 3.7 | (same) | (same) | (same) | (same) |

**Spring-mass**

| Method | device | M=1 | M=2 | M=3 | M=4 | M=5 |
|---|---|---|---|---|---|---|
| Chain Policy | cpu | 3.42 | 3.98 | 4.62 | 4.81 | 4.89 |
| Vanilla BC | cpu | 0.11 ± 0.0052 | 0.168 ± 0.0087 | 0.172 ± 0.023 | 0.232 ± 0.026 | 0.283 ± 0.027 |
| Diffusion Policy | cuda | 18.2 ± 1.3 | 17.5 ± 0.28 | 17.7 ± 0.33 | 17.6 ± 0.36 | 17.7 ± 0.33 |
| PPO (no demos) | cpu | 87.4 ± 2 | (same) | (same) | (same) | (same) |

