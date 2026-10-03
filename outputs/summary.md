# Reproduction vs paper (Section IV)

Reach times in seconds, unsuccessful runs count as the horizon; mean +- std over the test states (BC: pooled over seeds).  `~` = read off the paper's figure (approximate).  Theorem 2 diagnostics for K_M: C1 viol. = sampled violation fraction of the local energy decrease (Condition 1); C2 cov. = covered fraction of {E : Delta H(E) <= c} (Condition 2, target band included); C3 cov. = smallest per-ergodic-component coverage (Condition 3), with the component that attains it.

## Single pendulum

500 initial states with H <= 160.0, horizon 150.0 s, BC seeds [0, 1, 2, 3, 4]; L_H = 21.38, L = 19.62 on H <= 164; |K| = 48717.

| M | N(K_M) | Chain success | (paper) | Chain time | (paper) | BC success | (paper) | BC time | (paper) | C1 viol. | C2 cov. | C3 cov. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1041 | 0.184 | 0.348 | 123 +- 56.4 | 115 +- ~40 | 0.004 | 0.008 | 149 +- 9.46 | ~149 +- ~12.5 | 0 | 0.2449 | 0 (rotation_ccw) |
| 2 | 23746 | 0.488 | 0.678 | 78.9 +- 72.9 | ~68 +- ~53 | 0.014 | 0.53 | 148 +- 17.4 | ~78 +- ~61 | 0 | 0.9997 | 0 (rotation_ccw) |
| 3 | 46451 | 1 | 1 | 4.37 +- 1.85 | ~22 +- ~5.5 | 0.0784 | 0.418 | 139 +- 39.4 | ~90 +- ~65 | 0 | 0.9997 | 1 (libration) |
| 4 | 47584 | 1 | 1 | 4.29 +- 1.68 | ~18 +- ~8 | 0.1 | 0.65 | 135 +- 43.8 | ~61.5 +- ~45.5 | 0 | 0.9997 | 1 (libration) |
| 5 | 48717 | 1 | 1 | 4.29 +- 1.68 | 13.2 +- ~7.5 | 0.605 | 0.744 | 63.7 +- 69.8 | ~47.5 +- ~40.5 | 0 | 0.9997 | 1 (libration) |

## Spring-mass

500 initial states with H <= 2.0, horizon 20.0 s, BC seeds [0, 1, 2, 3, 4]; L_H = 5.163, L = 1 on H <= 13.33; |K| = 2689.

| M | N(K_M) | Chain success | (paper) | Chain time | (paper) | BC success | (paper) | BC time | (paper) | C1 viol. | C2 cov. | C3 cov. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 391 | 1 | 1 | 3.95 +- 1.85 | 5.54 +- ~2.3 | 0.837 | 0.062 | 4.51 +- 6.85 | ~14.1 +- ~3.45 | 0 | 0.9975 | 1 (single) |
| 2 | 782 | 1 | 1 | 2.39 +- 0.946 | ~4 +- ~1.45 | 1 | 0.19 | 1.16 +- 0.33 | ~9.95 +- ~5.4 | 0 | 0.9975 | 1 (single) |
| 3 | 1546 | 1 | 1 | 2.18 +- 0.972 | 2.94 +- ~1.35 | 1 | ~1 | 1.17 +- 0.35 | ~3.6 +- ~3 | 0 | 0.9975 | 1 (single) |
| 4 | 2310 | 1 | 1 | 2 +- 0.939 | 2.94 +- ~1.4 | 1 | ~1 | 1.15 +- 0.344 | ~3.6 +- ~1.7 | 0 | 0.9975 | 1 (single) |
| 5 | 2689 | 1 | 1 | 1.97 +- 0.927 | 2.94 +- ~1.4 | 1 | ~1 | 1.12 +- 0.325 | ~1.4 +- ~0.4 | 0 | 0.9975 | 1 (single) |
