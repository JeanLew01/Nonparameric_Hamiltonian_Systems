"""Fast tests of the PPO baseline (environment, policy, a tiny training run)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from symplectic_ncp.baselines.ppo import PPOConfig, PPOPolicy, ReachEnv, evaluate_in_env, ppo_config, train_ppo
from symplectic_ncp.config import get_config


def _setup(name):
    cfg = get_config(name)
    system = cfg.make_system()
    return cfg, system, cfg.make_target(system)


def _tiny(name, **kw):
    base = dict(num_envs=32, rollout_steps=64, total_env_steps=32 * 64 * 2, update_epochs=2, num_minibatches=4,
                device="cpu")
    base.update(kw)
    return dataclasses.replace(ppo_config(name), **base)


@pytest.fixture(scope="module")
def trained_spring():
    """A short spring-mass run (about 15 s on one CPU thread)."""
    cfg, system, target = _setup("spring_mass")
    pc = dataclasses.replace(
        ppo_config("spring_mass"), num_envs=64, rollout_steps=64, total_env_steps=64 * 64 * 30,
        num_minibatches=4, eval_every=10, eval_envs=64, device="cpu",
    )
    policy, stats = train_ppo(system, target, cfg, pc, seed=0)
    return cfg, system, target, pc, policy, stats


@pytest.mark.parametrize("name", ["spring_mass", "single_pendulum"])
def test_config_sanity(name):
    pc = ppo_config(name)
    assert isinstance(pc, PPOConfig)
    assert pc.batch_size % pc.num_minibatches == 0
    assert 0.0 < pc.gamma < 1.0 and 0.0 < pc.gae_lambda <= 1.0
    assert pc.total_env_steps >= pc.batch_size
    assert pc.reward_mode in ("cost", "potential")
    cfg = get_config(name)
    assert pc.episode_seconds <= cfg.horizon
    with pytest.raises(ValueError):
        ppo_config("double_pendulum")


@pytest.mark.parametrize("name,obs_dim", [("spring_mass", 2), ("single_pendulum", 3)])
def test_env_reset_and_step_shapes(name, obs_dim):
    cfg, system, target = _setup(name)
    env = ReachEnv(system, target, cfg, ppo_config(name), num_envs=16, seed=0)
    obs = env.reset()
    assert obs.shape == (16, obs_dim) and env.obs_dim == obs_dim
    assert np.all(system.hamiltonian(env.X) <= cfg.H_bar + 1e-9)
    assert not np.any(target.contains(env.X))
    obs, r, term, trunc, info = env.step(np.zeros((16, 1)))
    assert obs.shape == (16, obs_dim)
    assert r.shape == term.shape == trunc.shape == (16,)
    assert info["final_state"].shape == (16, 2) and info["final_obs"].shape == (16, obs_dim)
    # reset uses the training seed, not the test states' seed
    env2 = ReachEnv(system, target, cfg, ppo_config(name), num_envs=16, seed=0)
    env2.reset()
    env3 = ReachEnv(system, target, cfg, ppo_config(name), num_envs=16, seed=1)
    env3.reset()
    assert not np.allclose(env2.X, env3.X)


def test_env_termination_and_truncation():
    cfg, system, target = _setup("spring_mass")
    pc = dataclasses.replace(ppo_config("spring_mass"), episode_seconds=5 * cfg.control_period)
    env = ReachEnv(system, target, cfg, pc, num_envs=2, seed=0)
    env.reset()
    # env 0 heads into S_tgt within one control period; env 1 is far away
    env.X = np.array([[0.0, 0.12], [1.5, 0.0]])
    obs, r, term, trunc, info = env.step(np.array([[-20.0], [0.0]]))
    assert term.tolist() == [True, False] and not trunc.any()
    assert target.contains(info["final_state"][0:1])[0]
    assert r[0] > pc.success_bonus - 1.0 and r[1] < 0.0
    assert env.t[0] == 0 and not target.contains(env.X[0:1])[0]  # auto reset
    for _ in range(3):
        _, _, term, trunc, _ = env.step(np.zeros((2, 1)))
    assert not trunc[1]
    _, _, term, trunc, info = env.step(np.zeros((2, 1)))
    assert trunc[1] and not term[1] and info["ep_len"][1] == 5


def test_training_improves_spring_mass(trained_spring):
    cfg, system, target, pc, policy, stats = trained_spring
    assert stats["env_steps"] == 64 * 64 * 30
    assert stats["train_seconds"] > 0.0 and stats["device"] == "cpu"
    curve = stats["curve"]
    assert len(curve) == 30
    evals = [(c["eval_success"], c["eval_time"]) for c in curve if "eval_success" in c]
    assert len(evals) == 3
    assert evals[-1][0] > evals[0][0] and evals[-1][1] < evals[0][1]  # deterministic policy improves
    assert np.mean([c["ep_return"] for c in curve[-3:]]) > 0.0  # episodes end with the success bonus
    X0 = ReachEnv(system, target, cfg, pc, 1, seed=123).sample_states(64)
    ev = evaluate_in_env(policy, system, target, cfg, X0, seconds=5.0)
    assert ev["success_rate"] > 0.5


def test_policy_deterministic_and_clipped(trained_spring):
    cfg, system, target, pc, policy, stats = trained_spring
    X = np.random.default_rng(0).uniform(-3, 3, size=(50, 2))
    U = policy(X)
    assert U.shape == (50, 1)
    assert np.array_equal(U, policy(X))
    assert policy(np.array([0.3, -0.1])).shape == (1, 1)
    big = dataclasses.replace(policy, weights=[W * 1e3 for W in policy.weights])
    Ub = big(X)
    assert np.all(Ub <= system.u_max) and np.all(Ub >= system.u_min)
    assert np.any(np.abs(Ub) == system.u_max[0])


def test_training_deterministic_given_seed():
    cfg, system, target = _setup("single_pendulum")
    pc = _tiny("single_pendulum")
    p1, s1 = train_ppo(system, target, cfg, pc, seed=3)
    p2, s2 = train_ppo(system, target, cfg, pc, seed=3)
    p3, _ = train_ppo(system, target, cfg, pc, seed=4)
    for W1, W2 in zip(p1.weights, p2.weights):
        assert np.array_equal(W1, W2)
    assert np.array_equal(p1.obs_mean, p2.obs_mean)
    assert not np.array_equal(p1.weights[0], p3.weights[0])
    assert s1["env_steps"] == pc.batch_size * 2


def test_save_load_round_trip(trained_spring, tmp_path):
    cfg, system, target, pc, policy, stats = trained_spring
    path = tmp_path / "ppo.npz"
    policy.save(path)
    loaded = PPOPolicy.load(path)
    X = np.random.default_rng(1).uniform(-2, 2, size=(20, 2))
    assert np.array_equal(policy(X), loaded(X))
    assert loaded.meta["env_steps"] == stats["env_steps"]
    assert loaded.system_name == "spring_mass"


def test_pendulum_policy_angle_features(tmp_path):
    cfg, system, target = _setup("single_pendulum")
    policy, _ = train_ppo(system, target, cfg, _tiny("single_pendulum", energy_feature=True), seed=0)
    X = np.array([[0.5, 1.0], [0.5 + 2 * np.pi, 1.0]])
    U = policy(X)
    assert np.allclose(U[0], U[1])  # periodic in the angle
    policy.save(tmp_path / "p.npz")
    assert np.array_equal(PPOPolicy.load(tmp_path / "p.npz")(X), U)
