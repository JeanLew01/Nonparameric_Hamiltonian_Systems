"""End-to-end reproduction of the Section IV protocol for one system.

Steps (paper, Section IV):

1. NMPC expert demonstrations D = {(x_j, u_j, T_j)}_{j=1}^5 (loaded from
   ``<out>/<system>/demonstrations.npz`` when present, otherwise generated).
2. Lipschitz constants L_H, L on X = {H <= H_X} (Assumptions 1-2) and the
   assignment set K of all demonstrations (Algorithm 1), saved to
   ``assignments.npz``.  K_M, built from the first M demonstrations, is the
   subset of K with demo id < M because Algorithm 1 treats every
   demonstration independently.
3. ``num_inits`` test states uniform on {H <= H_bar}.
4. For M in ``cfg.num_demos``: the NCP pi_{K_M} (Definition 7, executed as in
   Remark 2) and vanilla BC trained on the same first M demonstrations are
   rolled out from the same initial states; success rate and the average
   reach time (unsuccessful runs count as the horizon) are recorded, together
   with the empirical checks of Theorems 2-4 for K_M.

Outputs under ``<out>/<system>/``: ``demonstrations.npz``, ``assignments.npz``,
``results.json`` and ``rollouts.npz`` (per-trajectory outcomes).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

from symplectic_ncp.chain import AssignmentSet, NonparametricChainPolicy
from symplectic_ncp.config import ExperimentConfig
from symplectic_ncp.experiments.paper_reference import paper_reference
from symplectic_ncp.experts import Demonstration, load_demonstrations, save_demonstrations
from symplectic_ncp.experts.demonstration import load_fingerprint
from symplectic_ncp.simulation import RolloutResult
from symplectic_ncp.systems import HamiltonianSystem
from symplectic_ncp.target import TargetSet

DEMOS_FILE = "demonstrations.npz"
ASSIGNMENTS_FILE = "assignments.npz"
RESULTS_FILE = "results.json"
ROLLOUTS_FILE = "rollouts.npz"


# ----------------------------------------------------------------- utilities
class _Logger:
    def __init__(self, prefix: str, verbose: bool):
        self.prefix, self.verbose, self.t0 = prefix, verbose, time.perf_counter()

    def __call__(self, msg: str) -> None:
        if self.verbose:
            print(f"[{self.prefix} {time.perf_counter() - self.t0:7.1f}s] {msg}", flush=True)


def to_jsonable(obj):
    """Recursively convert numpy / dataclass objects; non-finite floats become None."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(float(obj)) else None
    return obj


def system_dir(out_dir, system_name: str) -> Path:
    return Path(out_dir) / system_name


def load_results(out_dir, system_name: str) -> dict:
    with open(system_dir(out_dir, system_name) / RESULTS_FILE) as fh:
        return json.load(fh)


# --------------------------------------------------------------- step 1: data
def load_or_generate_demonstrations(cfg: ExperimentConfig, path: Path, force: bool, log) -> list[Demonstration]:
    """Reuse saved demonstrations generated with the same expert settings (fingerprint); else run NMPC."""
    fingerprint = cfg.expert_fingerprint()
    if path.exists() and not force:
        if load_fingerprint(path) == fingerprint:
            demos = load_demonstrations(path)
            log(f"loaded {len(demos)} demonstrations from {path}")
            return demos
        log(f"{path} was generated with different expert settings; regenerating")
    from symplectic_ncp.experts.generate import generate_demonstrations

    log("generating NMPC expert demonstrations")
    demos = generate_demonstrations(cfg, verbose=log.verbose)
    save_demonstrations(path, demos, fingerprint=fingerprint)
    log(f"saved {len(demos)} demonstrations to {path}")
    return demos


def describe_demonstrations(system: HamiltonianSystem, target: TargetSet, demos: list[Demonstration]) -> list[dict]:
    from symplectic_ncp.experts.generate import demonstration_summary

    return [{"id": j, **demonstration_summary(system, target, d)} for j, d in enumerate(demos)]


# ------------------------------------------------------------ step 4: rollouts
def _extras_summary(extras: dict) -> dict:
    out = {}
    for key, val in extras.items():
        arr = np.asarray(val, dtype=float)
        finite = arr[np.isfinite(arr)]
        out[key] = {
            "mean": float(np.mean(finite)) if finite.size else None,
            "max": float(np.max(finite)) if finite.size else None,
            "sum": float(np.sum(finite)) if finite.size else None,
        }
    return out


def evaluate_chain(system, target, K_M: AssignmentSet, demos_M, cfg: ExperimentConfig, X0, lip: dict, log) -> tuple:
    """Roll out pi_{K_M} from X0 and attach the Theorem 2-4 report.  Returns (summary dict, RolloutResult)."""
    from symplectic_ncp.chain.theory import theory_report
    from symplectic_ncp.simulation.closed_loop import simulate_chain_policy

    policy = NonparametricChainPolicy(K_M, system, membership_tol=cfg.chain.membership_tol)
    t0 = time.perf_counter()
    res: RolloutResult = simulate_chain_policy(system, target, policy, X0, cfg.horizon, cfg.sim_dt)
    rollout_time = time.perf_counter() - t0

    # Theorem 4: T1 bounds the zero-input return time to Supp(K) (Assumption 8); we pass its empirical
    # value, the largest completed idle interval.  Excursions still open at the horizon never returned,
    # so T1 is then right-censored (only a lower bound) and flagged.  T2 is not estimated.
    idle = np.asarray(res.extras.get("max_idle_interval", []), dtype=float)
    idle = idle[np.isfinite(idle)]
    T1 = float(np.max(idle)) if idle.size else None
    open_idle = np.asarray(res.extras.get("open_idle_interval", []), dtype=float)
    T1_censored = bool(np.any(open_idle > 0.0))
    t0 = time.perf_counter()
    theory = theory_report(
        system, target, K_M, demos_M, cfg, lip["L_H"], lip["L"], lip["H_X"], T1=T1, T2=None, T1_censored=T1_censored
    )
    theory_time = time.perf_counter() - t0

    summary = {
        **res.summary(),
        "num_assignments": len(K_M),
        "assignments_per_demo": np.bincount(K_M.demo_ids, minlength=len(demos_M)),
        "rollout_extras": _extras_summary(res.extras),
        "T1_empirical": T1,
        "T1_censored": T1_censored,
        "theory": theory,
        "timing": {"rollout_s": rollout_time, "theory_s": theory_time},
    }
    log(
        f"  chain  N={len(K_M):5d}  success={summary['success_rate']:.3f}  "
        f"time={summary['mean_reach_time']:.2f}+-{summary['std_reach_time']:.2f}s  ({rollout_time:.1f}s)"
    )
    return summary, res


def evaluate_bc(system, target, demos_M, cfg: ExperimentConfig, X0, log) -> tuple:
    """Train vanilla BC on the first M demos for every seed and roll it out from X0.

    Returns (summary dict, success (S, B), reach_time (S, B)).  ``pooled`` pools the S x B
    trajectories (its mean equals the mean of the per-seed means); it is what the figures show.
    """
    from symplectic_ncp.baselines.behavior_cloning import imitation_dataset, train_behavior_cloning
    from symplectic_ncp.simulation.closed_loop import simulate_feedback_policy

    per_seed, results = [], []
    for seed in cfg.bc.seeds:
        t0 = time.perf_counter()
        policy = train_behavior_cloning(system, demos_M, cfg.bc, seed)
        train_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        res = simulate_feedback_policy(system, target, policy, X0, cfg.horizon, cfg.sim_dt, cfg.control_period)
        rollout_time = time.perf_counter() - t0
        results.append(res)
        per_seed.append({"seed": int(seed), **res.summary(), "timing": {"train_s": train_time, "rollout_s": rollout_time}})
        log(
            f"  bc[{seed}]  success={per_seed[-1]['success_rate']:.3f}  "
            f"time={per_seed[-1]['mean_reach_time']:.2f}+-{per_seed[-1]['std_reach_time']:.2f}s  "
            f"(train {train_time:.1f}s, rollout {rollout_time:.1f}s)"
        )
    success = np.stack([r.success for r in results])
    reach = np.stack([r.reach_time for r in results])
    pooled = RolloutResult(success.reshape(-1), reach.reshape(-1), cfg.horizon).summary()
    rates = np.asarray([s["success_rate"] for s in per_seed])
    times = np.asarray([s["mean_reach_time"] for s in per_seed])
    summary = {
        "seeds": per_seed,
        "mean_over_seeds": {
            "success_rate": float(rates.mean()),
            "success_rate_std": float(rates.std()),
            "mean_reach_time": float(times.mean()),
            "mean_reach_time_std": float(times.std()),
        },
        "pooled": pooled,
        "num_samples": int(imitation_dataset(demos_M, cfg.bc.sample_grid)[0].shape[0]),
        "sample_grid": cfg.bc.sample_grid,
    }
    return summary, success, reach


# ------------------------------------------------------------------- driver
def run_system(cfg: ExperimentConfig, out_dir, force_experts: bool = False, verbose: bool = True) -> dict:
    """Run the Section IV protocol for ``cfg`` and write the outputs under ``out_dir/<system>/``."""
    from symplectic_ncp.chain.construction import build_certified_assignment_set
    from symplectic_ncp.evaluation.initial_states import sample_initial_states

    log = _Logger(cfg.system_name, verbose)
    sdir = system_dir(out_dir, cfg.system_name)
    sdir.mkdir(parents=True, exist_ok=True)
    system = cfg.make_system()
    target = cfg.make_target(system)
    timing: dict[str, float] = {}

    # 1. demonstrations
    t0 = time.perf_counter()
    demos = load_or_generate_demonstrations(cfg, sdir / DEMOS_FILE, force_experts, log)
    timing["demonstrations_s"] = time.perf_counter() - t0
    if max(cfg.num_demos) > len(demos):
        raise ValueError(f"num_demos={cfg.num_demos} exceeds the {len(demos)} available demonstrations")

    # 2. Lipschitz constants and Algorithm 1 on all demonstrations
    t0 = time.perf_counter()
    K, (L_H, L, H_X) = build_certified_assignment_set(system, target, demos, cfg.chain, cfg.H_bar)
    lip = {"L_H": float(L_H), "L": float(L), "H_X": float(H_X)}
    log(f"Lipschitz on X={{H<={H_X:.4g}}} (contains S_0, the demos and Supp(K)): L_H={L_H:.4g}, L={L:.4g}")
    timing["assignment_set_s"] = time.perf_counter() - t0
    if abs(K.dt - cfg.sim_dt) > 1e-12:
        raise ValueError(f"snippet step {K.dt} differs from sim_dt {cfg.sim_dt}")
    K.save(sdir / ASSIGNMENTS_FILE)
    log(
        f"assignment set: N={len(K)} per demo {np.bincount(K.demo_ids, minlength=len(demos)).tolist()}, "
        f"r in [{K.radii.min() if len(K) else 0:.3g}, {K.radii.max() if len(K) else 0:.3g}] "
        f"({timing['assignment_set_s']:.1f}s)"
    )

    # 3. test initial states
    X0 = sample_initial_states(system, cfg)
    log(f"{X0.shape[0]} initial states uniform on {{H <= {cfg.H_bar}}}, horizon {cfg.horizon}s")

    # 4. per-M evaluation
    per_M, arrays = [], {"initial_states": X0, "num_demos": np.asarray(cfg.num_demos), "bc_seeds": np.asarray(cfg.bc.seeds)}
    for M in cfg.num_demos:
        log(f"M={M}")
        demos_M = demos[:M]
        K_M = K.from_demos(range(M))
        entry: dict = {"M": int(M)}
        if len(K_M):
            entry["chain"], chain_res = evaluate_chain(system, target, K_M, demos_M, cfg, X0, lip, log)
            arrays[f"chain_success_M{M}"] = chain_res.success
            arrays[f"chain_reach_time_M{M}"] = chain_res.reach_time
            for key, val in chain_res.extras.items():
                arrays[f"chain_{key}_M{M}"] = np.asarray(val)
        else:
            # Algorithm 1 extracted nothing from these demos: pi_K is the default u_0 = 0 everywhere.
            from symplectic_ncp.simulation.closed_loop import simulate_feedback_policy

            log("  chain  K_M is empty; pi_K is the zero input everywhere")
            zero = lambda X: np.zeros((np.shape(X)[0], system.control_dim))  # noqa: E731
            empty = simulate_feedback_policy(system, target, zero, X0, cfg.horizon, cfg.sim_dt, cfg.control_period)
            entry["chain"] = {**empty.summary(), "num_assignments": 0, "theory": None}
            arrays[f"chain_success_M{M}"] = empty.success
            arrays[f"chain_reach_time_M{M}"] = empty.reach_time
        entry["bc"], bc_succ, bc_reach = evaluate_bc(system, target, demos_M, cfg, X0, log)
        arrays[f"bc_success_M{M}"] = bc_succ
        arrays[f"bc_reach_time_M{M}"] = bc_reach
        per_M.append(entry)
    timing["total_s"] = time.perf_counter() - log.t0

    results = {
        "system": cfg.system_name,
        "config": cfg,
        "target": {
            "center": target.center,
            "radius": target.radius,
            "H_min": target.H_min,
            "H_max": target.H_max,
            "H_plus": target.H_plus,
            "H_minus": target.H_minus,
        },
        "demonstrations": describe_demonstrations(system, target, demos),
        "lipschitz": lip,
        "assignment_set": {
            "N": len(K),
            "per_demo": np.bincount(K.demo_ids, minlength=len(demos)),
            "radius_min": float(K.radii.min()) if len(K) else None,
            "radius_max": float(K.radii.max()) if len(K) else None,
            "tau_min": float(K.durations.min()) if len(K) else None,
            "tau_max": float(K.durations.max()) if len(K) else None,
        },
        "num_initial_states": int(X0.shape[0]),
        "per_M": per_M,
        "paper_reference": paper_reference(cfg.system_name),
        "timing": timing,
    }
    results = to_jsonable(results)
    with open(sdir / RESULTS_FILE, "w") as fh:
        json.dump(results, fh, indent=2)
    np.savez(sdir / ROLLOUTS_FILE, **arrays)
    log(f"wrote {sdir / RESULTS_FILE} and {sdir / ROLLOUTS_FILE} (total {timing['total_s']:.1f}s)")
    return results
