"""Smoke tests of the Section IV pipeline, CLI, figures and summary table.

``test_pipeline_with_fake_components`` substitutes every module the pipeline
depends on (experts, Algorithm 1, theory, simulators, BC, initial states) with
small fakes honouring the module contracts, so it isolates the pipeline's own
logic.  ``test_pipeline_real_tiny`` runs the real modules on the spring-mass
system with tiny settings.
"""

from __future__ import annotations

import json
import shutil
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from symplectic_ncp.chain import AssignmentSet
from symplectic_ncp.config import ExperimentConfig, get_config
from symplectic_ncp.experiments import paper_reference, pipeline, plotting, run
from symplectic_ncp.experts import Demonstration, load_demonstrations, save_demonstrations
from symplectic_ncp.simulation import RolloutResult

REPO = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ helpers
def damping_demonstration(cfg: ExperimentConfig, x0, name: str, gain: float = 2.0) -> Demonstration:
    """Energy-shaping expert u = -gain * dH/dp (ZOH every control_period) until S_tgt^delta cap H_tgt^eps."""
    system, target = cfg.make_system(), cfg.make_target()
    stride = int(round(cfg.control_period / cfg.sim_dt))
    x = np.asarray(x0, dtype=float)[None, :]
    states, controls = [x[0]], []
    for k in range(int(30.0 / cfg.sim_dt)):
        if target.in_certified_target(x, cfg.energy_eps, cfg.demo_delta)[0]:
            break
        if k % stride == 0:
            u = system.clip_control(-gain * system.grad_hamiltonian(x)[:, 1:2])
        x = system.rk4_step(x, u, cfg.sim_dt)
        states.append(x[0])
        controls.append(u[0])
    else:
        raise RuntimeError("synthetic expert did not reach the target")
    return Demonstration(name, np.asarray(states), np.asarray(controls), cfg.sim_dt, cfg.control_period)


def synthetic_demos(cfg: ExperimentConfig) -> list[Demonstration]:
    return [damping_demonstration(cfg, spec.x0, spec.name) for spec in cfg.experts]


def tiny_config(**overrides) -> ExperimentConfig:
    cfg = get_config("spring_mass")
    settings = {"num_inits": 8, "horizon": 4.0, "num_demos": (1, 2), "bc": replace(cfg.bc, seeds=(0,)), **overrides}
    return replace(cfg, **settings)


def install_fakes(monkeypatch) -> dict:
    """Replace the pipeline's dependencies by contract-compatible fakes; returns a call log."""
    calls: dict[str, list] = {}

    def record(name, *args):
        calls.setdefault(name, []).append(args)

    def generate_demonstrations(cfg, verbose=True):
        record("generate")
        return synthetic_demos(cfg)

    def lipschitz_for_demos(system, demos, H_bar):
        H_X = max(H_bar, max(float(system.hamiltonian(d.states).max()) for d in demos))
        return (*system.lipschitz_constants(H_X, grid_points=101), H_X)

    def build_assignment_set(system, target, demos, chain_cfg, L_H, L):
        sets = []
        for j, d in enumerate(demos):
            idx = np.arange(0, d.num_steps - 40, 200)
            sets.append(AssignmentSet(d.states[idx], np.full(idx.size, 0.05), [d.controls[i:i + 40] for i in idx],
                                      d.dt, demo_ids=np.full(idx.size, j), anchor_times=idx * d.dt))
        return AssignmentSet.concatenate(sets)

    def build_certified_assignment_set(system, target, demos, chain_cfg, H_bar):
        L_H, L, H_X = lipschitz_for_demos(system, demos, H_bar)
        return build_assignment_set(system, target, demos, chain_cfg, L_H, L), (L_H, L, H_X)

    def demonstration_summary(system, target, demo):
        return {"name": demo.name, "duration": demo.duration}

    def imitation_dataset(demos, sample_grid="control"):
        pairs = [d.control_samples() for d in demos]
        return np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs])

    def theory_report(system, target, assignments, demos, cfg, L_H, L, H_X, T1=None, T2=None, T1_censored=False):
        record("theory", len(assignments), len(demos), T1)
        return {"N": len(assignments), "T1": T1}

    def simulate_chain_policy(system, target, policy, x0, horizon, dt):
        B = x0.shape[0]
        success = np.arange(B) % 2 == 0
        extras = {"n_snippets": np.ones(B), "max_idle_interval": np.full(B, 0.5), "fallback_entries": np.zeros(B)}
        return RolloutResult(success, np.where(success, 1.0, np.inf), horizon, extras)

    def simulate_feedback_policy(system, target, policy_fn, x0, horizon, dt, control_period):
        assert policy_fn(x0).shape == (x0.shape[0], system.control_dim)
        success = np.arange(x0.shape[0]) % 4 == 0
        return RolloutResult(success, np.where(success, 2.0, np.inf), horizon)

    def train_behavior_cloning(system, demos, cfg, seed):
        record("bc", len(demos), seed)
        return lambda X: np.zeros((np.shape(X)[0], system.control_dim))

    def sample_initial_states(system, cfg):
        return system.sample_energy_sublevel(cfg.num_inits, cfg.H_bar, np.random.default_rng(cfg.init_seed))

    fakes = {
        "symplectic_ncp.experts.generate": {"generate_demonstrations": generate_demonstrations,
                                            "demonstration_summary": demonstration_summary},
        "symplectic_ncp.chain.construction": {"build_certified_assignment_set": build_certified_assignment_set},
        "symplectic_ncp.chain.theory": {"theory_report": theory_report},
        "symplectic_ncp.simulation.closed_loop": {"simulate_chain_policy": simulate_chain_policy,
                                                  "simulate_feedback_policy": simulate_feedback_policy},
        "symplectic_ncp.baselines.behavior_cloning": {"train_behavior_cloning": train_behavior_cloning,
                                                      "imitation_dataset": imitation_dataset},
        "symplectic_ncp.evaluation.initial_states": {"sample_initial_states": sample_initial_states},
    }
    for mod_name, attrs in fakes.items():
        module = types.ModuleType(mod_name)
        for k, v in attrs.items():
            setattr(module, k, v)
        monkeypatch.setitem(sys.modules, mod_name, module)
    return calls


def check_outputs(out: Path, cfg: ExperimentConfig, results: dict) -> None:
    sdir = out / cfg.system_name
    for f in ("demonstrations.npz", "assignments.npz", "results.json", "rollouts.npz"):
        assert (sdir / f).exists(), f
    on_disk = json.loads((sdir / "results.json").read_text())
    assert on_disk == json.loads(json.dumps(results))
    assert on_disk["config"]["bc"]["lr"] == pytest.approx(1.2e-3)
    assert on_disk["config"]["horizon"] == cfg.horizon
    assert set(on_disk["lipschitz"]) == {"L_H", "L", "H_X"}
    assert on_disk["paper_reference"] == json.loads(json.dumps(paper_reference.paper_reference(cfg.system_name)))
    assert [e["M"] for e in on_disk["per_M"]] == list(cfg.num_demos)
    K = AssignmentSet.load(sdir / "assignments.npz")
    roll = np.load(sdir / "rollouts.npz")
    assert roll["initial_states"].shape == (cfg.num_inits, 2)
    for e in on_disk["per_M"]:
        M = e["M"]
        assert e["chain"]["num_assignments"] == int(np.sum(K.demo_ids < M))
        assert 0.0 <= e["chain"]["success_rate"] <= 1.0
        assert e["chain"]["mean_reach_time"] <= cfg.horizon + 1e-9
        assert len(e["bc"]["seeds"]) == len(cfg.bc.seeds)
        assert roll[f"chain_success_M{M}"].shape == (cfg.num_inits,)
        assert roll[f"bc_success_M{M}"].shape == (len(cfg.bc.seeds), cfg.num_inits)
        rate = roll[f"chain_success_M{M}"].mean()
        assert rate == pytest.approx(e["chain"]["success_rate"])


def check_reporting(out: Path, cfg: ExperimentConfig, results: dict) -> None:
    fig_dir = out / "figures"
    paths = plotting.plot_results(results, fig_dir, show_paper=True) + plotting.plot_assignment_set(out, cfg.system_name)
    assert {p.suffix for p in paths} == {".png", ".pdf"}
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)
    table = plotting.summary_markdown({cfg.system_name: results})
    assert "| M |" in table and table.count("\n| 1 |") == 1
    assert plotting.write_summary(out, {cfg.system_name: results}).read_text() == table


# -------------------------------------------------------------------- tests
def test_paper_reference_values():
    assert paper_reference.reference_value("spring_mass", "bc", "success_rate", 1) == (0.062, True)
    assert paper_reference.reference_value("single_pendulum", "chain", "mean_reach_time", 5) == (13.16, True)
    assert paper_reference.reference_value("single_pendulum", "chain", "mean_reach_time", 1) == (114.71, True)
    assert paper_reference.reference_value("single_pendulum", "bc", "success_rate", 3) == (0.418, True)
    assert paper_reference.reference_value("spring_mass", "bc", "mean_reach_time", 1)[1] is False
    ref = paper_reference.paper_reference("single_pendulum")
    assert json.loads(json.dumps(ref)) == ref
    assert paper_reference.paper_reference("unknown") is None


def test_cli_config_overrides():
    args = run.parse_args(["--systems", "single_pendulum", "--quick", "--bc-seeds", "0", "1"])
    cfg = run.build_config("single_pendulum", args)
    assert cfg.num_inits == run.QUICK_NUM_INITS and cfg.bc.seeds == (0, 1) and cfg.horizon == 150.0
    args = run.parse_args(["--num-inits", "7", "--quick"])
    assert args.systems == ["spring_mass", "single_pendulum"]
    assert run.build_config("spring_mass", args).num_inits == 7
    assert run.build_config("spring_mass", run.parse_args([])).num_inits == 500


def test_pipeline_with_fake_components(tmp_path, monkeypatch):
    calls = install_fakes(monkeypatch)
    cfg = tiny_config()
    results = pipeline.run_system(cfg, tmp_path, verbose=False)
    assert len(calls["generate"]) == 1
    assert [c[1] for c in calls["theory"]] == [1, 2]  # theory on the first M demonstrations
    assert all(c[2] == 0.5 for c in calls["theory"])  # T1 = largest observed idle interval
    assert [c[0] for c in calls["bc"]] == [1, 2]  # BC trained on the same first M demonstrations
    check_outputs(tmp_path, cfg, results)
    check_reporting(tmp_path, cfg, results)

    # saved demonstrations are reused, not regenerated
    pipeline.run_system(cfg, tmp_path, verbose=False)
    assert len(calls["generate"]) == 1
    pipeline.run_system(cfg, tmp_path, force_experts=True, verbose=False)
    assert len(calls["generate"]) == 2

    # CLI end to end (writes summary.md and the figures)
    out = tmp_path / "cli"
    run.main(["--systems", "spring_mass", "--out", str(out), "--num-inits", "6", "--num-demos", "1", "2", "--quiet"])
    assert (out / "summary.md").exists()
    assert (out / "figures" / "spring_mass_results.pdf").exists()
    assert (out / "figures" / "spring_mass_assignment_set.png").exists()


def test_pipeline_real_tiny(tmp_path):
    cfg = tiny_config()
    sdir = tmp_path / cfg.system_name
    sdir.mkdir()
    saved = REPO / "outputs" / cfg.system_name / "demonstrations.npz"
    if saved.exists() and [d.name for d in load_demonstrations(saved)] == [s.name for s in cfg.experts]:
        shutil.copy(saved, sdir / "demonstrations.npz")
    else:
        save_demonstrations(sdir / "demonstrations.npz", synthetic_demos(cfg))
    results = pipeline.run_system(cfg, tmp_path, verbose=False)
    check_outputs(tmp_path, cfg, results)
    check_reporting(tmp_path, cfg, results)
    assert results["per_M"][0]["chain"]["theory"] is not None
