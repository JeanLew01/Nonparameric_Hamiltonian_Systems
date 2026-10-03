"""Tests of the diffusion-policy baseline (small CPU models; the CUDA-graph test is skipped without a GPU)."""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pytest
import torch
from torch import nn

from symplectic_ncp.baselines.diffusion_policy import (
    DiffusionPolicy,
    DiffusionPolicyConfig,
    chunk_dataset,
    cosine_alpha_bar,
    ddim_timesteps,
    diffusion_policy_config,
    train_diffusion_policy,
)
from symplectic_ncp.config import get_config
from symplectic_ncp.experts.demonstration import Demonstration
from symplectic_ncp.simulation import simulate_feedback_policy

TINY = DiffusionPolicyConfig(
    hidden=64, num_blocks=1, step_embed_dim=16, cond_dim=32, pred_horizon=6, action_horizon=3,
    train_steps=50, inference_steps=5, iterations=800, batch_size=64, lr=3e-3, warmup=20,
    ema_decay=0.99, log_every=100, device="cpu",
)


def _damping_demo(system, x0, duration=2.0, dt=0.005, control_period=0.02, gain=2.0):
    """Synthetic expert: ZOH damping u = -gain * p."""
    stride = int(round(control_period / dt))
    n = int(round(duration / dt))
    states, controls, u = [np.asarray(x0, dtype=float)], [], np.zeros(1)
    for k in range(n):
        if k % stride == 0:
            u = system.clip_control(-gain * states[-1][1:2])[0]
        controls.append(u)
        states.append(system.rk4_step(states[-1], u, dt)[0])
    return Demonstration("damp", np.array(states), np.array(controls), dt, control_period)


@pytest.fixture(scope="module")
def spring():
    cfg = get_config("spring_mass")
    system = cfg.make_system()
    demos = [_damping_demo(system, (2.0, 0.0), gain=15.0), _damping_demo(system, (0.0, -2.0), gain=15.0)]
    return cfg, system, demos


@pytest.fixture(scope="module")
def trained(spring):
    cfg, system, demos = spring
    return train_diffusion_policy(system, demos, TINY, seed=0)


# --------------------------------------------------------------------- data
def test_chunk_dataset_alignment_and_padding(spring):
    _, _, demos = spring
    Tp = 7
    X, A = chunk_dataset(demos, Tp)
    Xs = [d.control_samples()[0] for d in demos]
    Us = [d.control_samples()[1] for d in demos]
    assert X.shape == (sum(len(x) for x in Xs), 2) and A.shape == (X.shape[0], Tp, 1)
    np.testing.assert_array_equal(X, np.concatenate(Xs))
    K0 = len(Us[0])
    np.testing.assert_array_equal(A[0], Us[0][:Tp])  # the next Tp expert inputs
    np.testing.assert_array_equal(A[5, :, 0], Us[0][5 : 5 + Tp, 0])
    # Past the end of a demonstration: padded with its last input, no leakage into the next one.
    np.testing.assert_array_equal(A[K0 - 2, :, 0], np.r_[Us[0][-2:, 0], np.full(Tp - 2, Us[0][-1, 0])])
    np.testing.assert_array_equal(A[K0 - 1, :, 0], np.full(Tp, Us[0][-1, 0]))
    np.testing.assert_array_equal(A[K0], Us[1][:Tp])
    # Expert inputs at the control period: the input held on the period after the sample.
    stride = int(round(demos[0].control_period / demos[0].dt))
    np.testing.assert_array_equal(A[3, 1], demos[0].controls[4 * stride])


# ----------------------------------------------------------------- schedule
def test_schedule_sanity():
    ab = cosine_alpha_bar(100)
    assert ab.shape == (100,)
    assert np.all(np.diff(ab) < 0) and 0.99 < ab[0] < 1.0 and 0.0 < ab[-1] < 1e-3
    ts = ddim_timesteps(100, 10)
    np.testing.assert_array_equal(ts, np.arange(90, -1, -10))
    assert ddim_timesteps(100, 100)[0] == 99 and ddim_timesteps(100, 1).tolist() == [0]
    with pytest.raises(ValueError):
        ddim_timesteps(10, 20)
    for name in ("spring_mass", "single_pendulum"):
        c = diffusion_policy_config(name)
        assert 1 <= c.action_horizon <= c.pred_horizon and c.inference_steps <= c.train_steps


def test_ddim_sampler_recovers_target_with_an_oracle(trained):
    """With the exact noise predictor for a fixed clean chunk, DDIM returns that chunk."""
    policy, _ = trained
    oracle_policy = dataclasses.replace(policy)  # independent buffers / generator
    D = policy.cfg.pred_horizon * policy.control_dim
    target = torch.linspace(-1.0, 1.0, D)  # standardized chunk inside the bounds
    ab = torch.tensor(cosine_alpha_bar(policy.cfg.train_steps), dtype=torch.float32)

    class Oracle(nn.Module):
        def forward(self, a, k, obs):
            s = ab[k][:, None]
            return (a - s.sqrt() * target) / (1.0 - s).sqrt()

    oracle_policy.model = Oracle()
    X = np.random.default_rng(0).normal(size=(5, 2))
    U = oracle_policy.sample_chunk(X)
    expected = np.clip(target.double().numpy() * policy.u_std + policy.u_mean, policy.u_min, policy.u_max)
    np.testing.assert_allclose(U[:, :, 0], np.broadcast_to(expected, (5, D)), atol=1e-4)


# ----------------------------------------------------------------- training
def test_training_loss_decreases_and_info(trained):
    policy, info = trained
    assert info["iterations"] == TINY.iterations and info["device"] == "cpu"
    assert info["train_seconds"] > 0 and info["num_samples"] == 2 * 100
    hist = info["loss_history"]
    assert len(hist) == TINY.iterations // TINY.log_every
    assert info["final_loss"] == hist[-1]
    assert hist[-1] < 0.6 * hist[0]


def test_training_is_deterministic(spring):
    _, system, demos = spring
    cfg = dataclasses.replace(TINY, iterations=60)
    a, ia = train_diffusion_policy(system, demos, cfg, seed=3)
    b, ib = train_diffusion_policy(system, demos, cfg, seed=3)
    c, _ = train_diffusion_policy(system, demos, cfg, seed=4)
    assert ia["loss_history"] == ib["loss_history"]
    for (ka, va), (_, vb) in zip(a.model.state_dict().items(), b.model.state_dict().items()):
        assert torch.equal(va, vb), ka
    assert not all(torch.equal(va, vc) for va, vc in zip(a.model.state_dict().values(), c.model.state_dict().values()))


# ----------------------------------------------------------------- sampling
def test_output_shape_and_clipping(trained):
    policy, _ = trained
    X = np.random.default_rng(1).normal(size=(9, 2)) * 3.0
    U = policy(X)
    assert U.shape == (9, TINY.action_horizon, 1)
    assert policy.sample_chunk(X).shape == (9, TINY.pred_horizon, 1)
    assert policy(np.array([0.3, -0.1])).shape == (1, TINY.action_horizon, 1)
    assert np.all(U >= policy.u_min) and np.all(U <= policy.u_max)
    # The damping expert saturates at |u| = 20 far from the origin; samples respect the bounds exactly.
    far = policy.sample_chunk(np.array([[0.0, 5.0], [0.0, -5.0]]))
    assert np.all(np.abs(far) <= 20.0)
    assert policy(np.zeros((0, 2))).shape == (0, TINY.action_horizon, 1)


def test_sampling_is_deterministic(trained):
    policy, _ = trained
    X = np.random.default_rng(2).normal(size=(6, 2))
    policy.reset()
    first, second = policy(X), policy(X)
    assert not np.array_equal(first, second)  # "stream" noise: fresh noise per query
    policy.reset()
    np.testing.assert_array_equal(policy(X), first)
    np.testing.assert_array_equal(policy(X), second)
    policy.reset(seed=123)
    assert not np.array_equal(policy(X), first)

    fixed = dataclasses.replace(policy, cfg=dataclasses.replace(policy.cfg, noise="fixed"))
    u_all = fixed(X)
    np.testing.assert_allclose(fixed(X[2:3]), u_all[2:3], atol=1e-5)  # a function of the state only
    np.testing.assert_array_equal(fixed(X), u_all)


def test_policy_tracks_the_expert_on_training_states(spring, trained):
    _, _, demos = spring
    policy, _ = trained
    X, A = chunk_dataset(demos, TINY.pred_horizon)
    policy.reset()
    err = np.abs(policy.sample_chunk(X) - A).mean()
    assert err < 0.3 * np.abs(A - A.mean(axis=0)).mean()  # far better than the best constant chunk


def test_closed_loop_queries_every_action_horizon(spring, trained):
    cfg, system, _ = spring
    policy, _ = trained
    target = cfg.make_target(system)
    calls = []

    def counted(X):
        calls.append(X.shape[0])
        return policy(X)

    x0 = np.array([[1.5, 0.0], [0.0, 1.5], [-1.0, -1.0]])
    policy.reset()
    res = simulate_feedback_policy(system, target, counted, x0, 0.3, cfg.sim_dt, cfg.control_period)
    periods = int(round(0.3 / cfg.control_period))
    assert len(calls) == -(-periods // TINY.action_horizon) and calls[0] == 3
    policy.reset()
    res2 = simulate_feedback_policy(system, target, policy, x0, 0.3, cfg.sim_dt, cfg.control_period)
    np.testing.assert_array_equal(res.extras["final_energy"], res2.extras["final_energy"])


def test_save_load_roundtrip(tmp_path, trained):
    policy, _ = trained
    path = tmp_path / "dp.pt"
    policy.save(path)
    loaded = DiffusionPolicy.load(path, device="cpu")
    assert loaded.cfg == policy.cfg and loaded.train_losses == policy.train_losses
    X = np.random.default_rng(3).normal(size=(4, 2))
    policy.reset()
    loaded.reset()
    np.testing.assert_array_equal(loaded(X), policy(X))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_graph_sampler_matches_eager(trained):
    policy, _ = trained
    X = np.random.default_rng(4).normal(size=(37, 2))
    on_gpu = dataclasses.replace(policy, model=copy.deepcopy(policy.model), device="cuda")
    eager = dataclasses.replace(on_gpu, cfg=dataclasses.replace(policy.cfg, cuda_graph=False))
    graphed = dataclasses.replace(on_gpu, cfg=dataclasses.replace(policy.cfg, cuda_graph=True))
    np.testing.assert_allclose(graphed(X), eager(X), atol=1e-3)
    graphed.reset()
    a = graphed(X)
    graphed.reset()
    np.testing.assert_array_equal(graphed(X), a)
