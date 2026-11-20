"""Proximal policy optimization (PPO) baseline for target reachability.

A model-free reinforcement-learning baseline that is *not* in the paper.  It
learns a state-feedback law u = pi(x) by interacting with a simulator built
only from the package's :class:`HamiltonianSystem` and :class:`TargetSet`:

Environment (:class:`ReachEnv`, batched, numpy)
    * reset: x_0 uniform on {H(x) <= reset_energy_scale * H_bar}, sampled with
      ``system.sample_energy_sublevel`` and an rng seeded from the training
      seed (never the test states of ``sample_initial_states``).
    * step: one control period of zero-order-hold input, integrated with
      ``system.rk4_step`` at ``cfg.sim_dt``; S_tgt membership is checked at
      every RK4 sub-step (a conservative discretization of the evaluator's
      continuous-time entry detection).
    * terminated: the state entered S_tgt (success bonus paid);
      truncated: the episode reached ``episode_seconds``.

Reward (per control period T = cfg.control_period, all terms documented in
:class:`PPOConfig`)::

    c(x)  = w_dist * ||x - x*|| / dist_scale + w_energy * |H(x) - H(x*)| / energy_scale
    r     = - T * (w_time + c(x'))  - T * w_u * ||u / u_max||^2  + success_bonus * [x' entered S_tgt]

or, with ``reward_mode="potential"``, the potential-based shaping
(Ng et al., 1999) r = gamma * Phi(x') - Phi(x) - T * w_time - T * w_u ||u/u_max||^2
+ success_bonus * hit with Phi = -c and Phi = 0 at the terminal state, which
leaves the optimal policy of the sparse "reach fast" task unchanged.

Learner: PPO with clipped surrogate (Schulman et al., 2017), GAE(lambda),
separate tanh-MLP actor and critic, state-independent Gaussian log-std,
actions sampled in normalized units a ~ N(mu(x), sigma) and applied as
u = clip(a, -1, 1) * u_half + u_center (the log-probability is that of the
unclipped sample, as in CleanRL), running observation normalization (angles
as (sin, cos)), optional running return normalization of the reward, linear
learning-rate annealing, value bootstrapping at truncation.

The deployed :class:`PPOPolicy` is the deterministic mean action clipped to
the input bounds, evaluated with a numpy forward pass.
"""

from __future__ import annotations

import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from symplectic_ncp.baselines.behavior_cloning import state_features
from symplectic_ncp.config import ExperimentConfig
from symplectic_ncp.systems import HamiltonianSystem
from symplectic_ncp.systems.base import as_batch
from symplectic_ncp.target import TargetSet


# ---------------------------------------------------------------------- config
@dataclass
class PPOConfig:
    """PPO hyperparameters, environment and reward settings (none are from the paper)."""

    # networks
    actor_hidden: tuple[int, ...] = (64, 64)
    critic_hidden: tuple[int, ...] = (64, 64)
    init_log_std: float = -0.5  # initial std of the normalized action (u / u_max)
    # optimization
    lr: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 10
    num_minibatches: int = 8
    target_kl: float | None = None  # early stop of the epochs when approx KL exceeds this
    # data collection
    num_envs: int = 64
    rollout_steps: int = 256  # env steps per env per update (batch = num_envs * rollout_steps)
    total_env_steps: int = 2_000_000
    episode_seconds: float = 10.0  # truncation; env steps = episode_seconds / control_period
    reset_energy_scale: float = 1.0  # reset uniform on {H <= reset_energy_scale * H_bar}
    # observation / reward normalization
    norm_obs: bool = True
    obs_clip: float = 10.0
    norm_reward: bool = True
    energy_feature: bool = False  # append H(x) to the observation
    # reward (see module docstring)
    reward_mode: str = "cost"  # "cost" or "potential"
    w_dist: float = 1.0
    w_energy: float = 0.0
    w_time: float = 1.0
    w_u: float = 0.0
    success_bonus: float = 10.0
    dist_scale: float = 1.0
    energy_scale: float = 1.0
    # monitoring / model selection on training-distribution states (never the test states)
    eval_every: int = 0  # updates between deterministic evaluations (0 = never)
    eval_envs: int = 256
    keep_best: bool = False  # return the best evaluated policy instead of the last one
    # runtime
    device: str = "cpu"
    torch_threads: int = 1  # 1 = bitwise reproducible CPU kernels

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.rollout_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def ppo_config(system_name: str) -> PPOConfig:
    """Per-system defaults (chosen with monitoring on training-distribution states, never the test states).

    spring_mass: "cost" reward with the distance only, 1M steps on one CPU
    thread (~2.5 min).  single_pendulum: potential-based shaping of distance
    plus energy error |H - H(x*)| (normalized by ~ the separatrix energy 2mgl),
    10M steps (~3.5 min on the GPU); the "cost" variant with the same weights
    also reaches 100 % on the training distribution.
    """
    if system_name == "spring_mass":
        return PPOConfig(
            gamma=0.99,
            num_envs=64,
            rollout_steps=128,
            total_env_steps=1_000_000,
            episode_seconds=8.0,
            reward_mode="cost",
            w_dist=1.0,
            w_energy=0.0,
            w_time=1.0,
            success_bonus=10.0,
            dist_scale=1.0,
        )
    if system_name == "single_pendulum":
        # Large batches: the update cost dominates, and on an RTX 4070 these
        # run ~3x faster on the GPU than on the CPU (64x64 MLPs, 16k minibatches).
        return PPOConfig(
            gamma=0.995,
            gae_lambda=0.95,
            num_envs=1024,
            rollout_steps=64,
            num_minibatches=4,
            total_env_steps=10_000_000,
            episode_seconds=20.0,
            reward_mode="potential",
            w_dist=1.0,
            w_energy=1.0,
            w_time=1.0,
            success_bonus=10.0,
            dist_scale=3.0,
            energy_scale=40.0,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    raise ValueError(f"no PPO defaults for system {system_name!r}")


# ------------------------------------------------------------------ utilities
class RunningMeanStd:
    """Parallel (Chan et al.) running mean / variance of batches of vectors."""

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=float)
        self.var = np.ones(shape, dtype=float)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=float)
        b_mean, b_var, b_count = x.mean(axis=0), x.var(axis=0), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_count
        self.mean = self.mean + delta * b_count / tot
        m2 = self.var * self.count + b_var * b_count + delta**2 * self.count * b_count / tot
        self.var = m2 / tot
        self.count = tot


def observation_features(system: HamiltonianSystem, X: np.ndarray, energy_feature: bool) -> np.ndarray:
    """Raw observation: angles as (sin, cos), other coordinates kept, optionally H(x) appended."""
    F = state_features(X, tuple(system.angle_indices), True)
    if energy_feature:
        F = np.concatenate([F, system.hamiltonian(X)[:, None]], axis=1)
    return F


def _mlp(in_dim: int, hidden: tuple[int, ...], out_dim: int, out_gain: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for width in hidden:
        lin = nn.Linear(prev, width)
        nn.init.orthogonal_(lin.weight, np.sqrt(2.0))
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.Tanh()]
        prev = width
    out = nn.Linear(prev, out_dim)
    nn.init.orthogonal_(out.weight, out_gain)
    nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, cfg: PPOConfig):
        super().__init__()
        self.actor = _mlp(obs_dim, tuple(cfg.actor_hidden), act_dim, 0.01)
        self.critic = _mlp(obs_dim, tuple(cfg.critic_hidden), 1, 1.0)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(cfg.init_log_std)))

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor(obs)
        return torch.distributions.Normal(mean, self.log_std.expand_as(mean).exp())


# ---------------------------------------------------------------- environment
class ReachEnv:
    """Batched target-reaching environment on ``system`` (auto-resets finished episodes).

    ``step`` takes physical inputs U (B, m) (clipped to the bounds) and returns
    ``(obs, reward, terminated, truncated, info)`` where ``obs`` already
    belongs to the reset state for finished environments; ``info`` holds
    ``final_obs`` / ``final_state`` (B, .) of the step's end state (entry
    state on termination), ``ep_return`` / ``ep_len`` of finished episodes
    (NaN otherwise) and ``hit`` (= terminated).
    """

    def __init__(
        self,
        system: HamiltonianSystem,
        target: TargetSet,
        cfg: ExperimentConfig,
        ppo_cfg: PPOConfig,
        num_envs: int,
        seed: int,
    ):
        self.system, self.target, self.cfg, self.ppo_cfg = system, target, cfg, ppo_cfg
        self.num_envs = int(num_envs)
        self.rng = np.random.default_rng(seed)
        self.substeps = int(round(cfg.control_period / cfg.sim_dt))
        if self.substeps < 1 or abs(self.substeps * cfg.sim_dt - cfg.control_period) > 1e-9:
            raise ValueError("control_period must be a positive multiple of sim_dt")
        self.max_steps = int(round(ppo_cfg.episode_seconds / cfg.control_period))
        self.reset_energy = ppo_cfg.reset_energy_scale * cfg.H_bar
        self.H_star = float(system.hamiltonian(target.center[None, :])[0])
        self.u_max_abs = np.maximum(np.abs(system.u_min), np.abs(system.u_max))
        self.X = np.zeros((self.num_envs, system.state_dim))
        self.t = np.zeros(self.num_envs, dtype=int)
        self.ep_return = np.zeros(self.num_envs)

    @property
    def obs_dim(self) -> int:
        return self.observe(np.zeros((1, self.system.state_dim))).shape[1]

    def observe(self, X: np.ndarray) -> np.ndarray:
        return observation_features(self.system, X, self.ppo_cfg.energy_feature)

    def sample_states(self, num: int) -> np.ndarray:
        """Uniform on {H <= reset energy} minus S_tgt (episodes never start inside the target)."""
        out = np.empty((0, self.system.state_dim))
        while out.shape[0] < num:
            X = self.system.wrap(self.system.sample_energy_sublevel(num - out.shape[0], self.reset_energy, self.rng))
            out = np.vstack([out, X[~self.target.contains(X)]])
        return out

    def reset(self) -> np.ndarray:
        self.X = self.sample_states(self.num_envs)
        self.t[:] = 0
        self.ep_return[:] = 0.0
        return self.observe(self.X)

    def cost(self, X: np.ndarray) -> np.ndarray:
        c = self.ppo_cfg
        out = c.w_dist * self.target.distance(X) / c.dist_scale
        if c.w_energy:
            out = out + c.w_energy * np.abs(self.system.hamiltonian(X) - self.H_star) / c.energy_scale
        return out

    def integrate(self, X: np.ndarray, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One control period of ZOH input; returns (end or entry state, entered S_tgt)."""
        hit = np.zeros(X.shape[0], dtype=bool)
        for _ in range(self.substeps):
            Xn = self.system.wrap(self.system.rk4_step(X, U, self.cfg.sim_dt))
            X = np.where(hit[:, None], X, Xn)  # freeze at the entry sub-step
            hit |= self.target.contains(X)
        return X, hit

    def step(self, U: np.ndarray):
        c = self.ppo_cfg
        T = self.cfg.control_period
        U = self.system.clip_control(U)
        X0 = self.X
        X1, hit = self.integrate(X0, U)
        self.t += 1
        effort = np.sum((U / self.u_max_abs) ** 2, axis=1)
        reward = -T * (c.w_time + c.w_u * effort) + c.success_bonus * hit
        if c.reward_mode == "cost":
            reward = reward - T * self.cost(X1)
        elif c.reward_mode == "potential":
            phi1 = np.where(hit, 0.0, -self.cost(X1))
            reward = reward + c.gamma * phi1 + self.cost(X0)
        else:
            raise ValueError(f"unknown reward_mode {c.reward_mode!r}")
        terminated = hit
        truncated = ~hit & (self.t >= self.max_steps)
        done = terminated | truncated
        self.ep_return += reward
        info = {
            "final_state": X1.copy(),
            "final_obs": self.observe(X1),
            "hit": hit,
            "ep_return": np.where(done, self.ep_return, np.nan),
            "ep_len": np.where(done, self.t, -1),
        }
        self.X = X1
        idx = np.flatnonzero(done)
        if idx.size:
            self.X[idx] = self.sample_states(idx.size)
            self.t[idx] = 0
            self.ep_return[idx] = 0.0
        return self.observe(self.X), reward, terminated, truncated, info


# --------------------------------------------------------------------- policy
@dataclass
class PPOPolicy:
    """Deterministic PPO policy U = clip(u_center + u_half * clip(mu(x), -1, 1)) (numpy forward).

    ``weights[k]`` has shape (in_k, out_k): a layer is ``h @ W + b``, tanh on hidden layers.
    """

    weights: list[np.ndarray]
    biases: list[np.ndarray]
    obs_mean: np.ndarray
    obs_std: np.ndarray
    obs_clip: float
    state_dim: int
    angle_indices: tuple[int, ...]
    energy_feature: bool
    u_min: np.ndarray
    u_max: np.ndarray
    system_name: str = ""
    meta: dict = field(default_factory=dict)  # hyperparameters and training statistics (JSON-able)

    def __post_init__(self):
        self.weights = [np.asarray(W, dtype=float) for W in self.weights]
        self.biases = [np.asarray(b, dtype=float) for b in self.biases]
        self.obs_mean = np.asarray(self.obs_mean, dtype=float)
        self.obs_std = np.asarray(self.obs_std, dtype=float)
        self.u_min = np.asarray(self.u_min, dtype=float)
        self.u_max = np.asarray(self.u_max, dtype=float)
        self.angle_indices = tuple(int(i) for i in self.angle_indices)
        self._u_center = 0.5 * (self.u_max + self.u_min)
        self._u_half = 0.5 * (self.u_max - self.u_min)
        self._system = None

    @property
    def control_dim(self) -> int:
        return self.u_min.shape[0]

    def _features(self, X: np.ndarray) -> np.ndarray:
        F = state_features(X, self.angle_indices, True)
        if self.energy_feature:
            if self._system is None:
                from symplectic_ncp.systems import make_system

                self._system = make_system(self.system_name, u_min=self.u_min, u_max=self.u_max)
            F = np.concatenate([F, self._system.hamiltonian(X)[:, None]], axis=1)
        return F

    def normalized_action(self, X) -> np.ndarray:
        """Unclipped actor mean mu(x) in normalized input units, shape (B, m)."""
        X = as_batch(X, self.state_dim)
        h = np.clip((self._features(X) - self.obs_mean) / self.obs_std, -self.obs_clip, self.obs_clip)
        last = len(self.weights) - 1
        for k, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ W + b
            if k < last:
                h = np.tanh(h)
        return h

    def __call__(self, X) -> np.ndarray:
        a = np.clip(self.normalized_action(X), -1.0, 1.0)
        return np.clip(self._u_center + self._u_half * a, self.u_min, self.u_max)

    # ------------------------------------------------------------ persistence
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {f"W{k}": W for k, W in enumerate(self.weights)}
        payload.update({f"b{k}": b for k, b in enumerate(self.biases)})
        payload.update(
            num_layers=len(self.weights),
            obs_mean=self.obs_mean,
            obs_std=self.obs_std,
            obs_clip=self.obs_clip,
            state_dim=self.state_dim,
            angle_indices=np.asarray(self.angle_indices, dtype=int),
            energy_feature=self.energy_feature,
            u_min=self.u_min,
            u_max=self.u_max,
            system_name=self.system_name,
            meta=json.dumps(self.meta, default=_json_default),
        )
        np.savez(path, **payload)

    @classmethod
    def load(cls, path) -> "PPOPolicy":
        data = np.load(Path(path), allow_pickle=False)
        L = int(data["num_layers"])
        return cls(
            weights=[data[f"W{k}"] for k in range(L)],
            biases=[data[f"b{k}"] for k in range(L)],
            obs_mean=data["obs_mean"],
            obs_std=data["obs_std"],
            obs_clip=float(data["obs_clip"]),
            state_dim=int(data["state_dim"]),
            angle_indices=tuple(int(i) for i in data["angle_indices"]),
            energy_feature=bool(data["energy_feature"]),
            u_min=data["u_min"],
            u_max=data["u_max"],
            system_name=str(data["system_name"]),
            meta=json.loads(str(data["meta"])),
        )


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return str(obj)


# ------------------------------------------------------------------- training
def _extract_policy(
    model: ActorCritic, obs_rms: RunningMeanStd, system: HamiltonianSystem, ppo_cfg: PPOConfig, meta: dict
) -> PPOPolicy:
    linears = [m for m in model.actor if isinstance(m, nn.Linear)]
    if ppo_cfg.norm_obs:
        mean, std = obs_rms.mean.copy(), np.sqrt(obs_rms.var + 1e-8)
    else:
        mean, std = np.zeros_like(obs_rms.mean), np.ones_like(obs_rms.var)
    return PPOPolicy(
        weights=[lin.weight.detach().cpu().double().numpy().T.copy() for lin in linears],
        biases=[lin.bias.detach().cpu().double().numpy().copy() for lin in linears],
        obs_mean=mean,
        obs_std=std,
        obs_clip=float(ppo_cfg.obs_clip) if ppo_cfg.norm_obs else np.inf,
        state_dim=system.state_dim,
        angle_indices=tuple(system.angle_indices),
        energy_feature=ppo_cfg.energy_feature,
        u_min=system.u_min.copy(),
        u_max=system.u_max.copy(),
        system_name=system.name,
        meta=meta,
    )


def evaluate_in_env(
    policy: PPOPolicy,
    system: HamiltonianSystem,
    target: TargetSet,
    cfg: ExperimentConfig,
    X0: np.ndarray,
    seconds: float,
) -> dict:
    """Deterministic rollouts in the training environment's discretization (sub-step entry checks).

    Cheap proxy of ``simulate_feedback_policy`` for monitoring; returns
    success rate and mean reach time (failures counted as ``seconds``).
    """
    env = ReachEnv(system, target, cfg, PPOConfig(), X0.shape[0], seed=0)
    X = X0.copy()
    reach = np.full(X.shape[0], np.inf)
    active = np.ones(X.shape[0], dtype=bool)
    steps = int(round(seconds / cfg.control_period))
    for k in range(steps):
        idx = np.flatnonzero(active)
        if idx.size == 0:
            break
        Xn, hit = env.integrate(X[idx], policy(X[idx]))
        X[idx] = Xn
        reach[idx[hit]] = (k + 1) * cfg.control_period
        active[idx[hit]] = False
    succ = np.isfinite(reach)
    return {"success_rate": float(succ.mean()), "mean_reach_time": float(np.where(succ, reach, seconds).mean())}


def train_ppo(
    system: HamiltonianSystem,
    target: TargetSet,
    cfg: ExperimentConfig,
    ppo_cfg: PPOConfig,
    seed: int = 0,
    device: str | None = None,
    verbose: bool = False,
) -> tuple[PPOPolicy, dict]:
    """Train PPO on the reach task; returns the deterministic policy and training statistics.

    The stats dict holds ``train_seconds`` (wall clock of the whole training:
    environment simulation, updates and monitoring evaluations),
    ``env_steps`` (control periods simulated), ``device``, ``updates``,
    ``config`` and ``curve`` (one dict per update: env_steps, seconds,
    episodes finished, their mean return / length / success fraction, losses,
    approx KL, std; plus ``eval_success`` / ``eval_time`` when evaluated).
    """
    device = torch.device(device if device is not None else ppo_cfg.device)
    threads = torch.get_num_threads()
    if ppo_cfg.torch_threads:
        torch.set_num_threads(int(ppo_cfg.torch_threads))
    try:
        return _train(system, target, cfg, ppo_cfg, seed, device, verbose)
    finally:
        torch.set_num_threads(threads)


def _train(system, target, cfg, ppo_cfg: PPOConfig, seed, device, verbose):
    t_start = time.perf_counter()
    c = ppo_cfg
    torch.manual_seed(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    seeds = np.random.SeedSequence(seed).spawn(3)
    env = ReachEnv(system, target, cfg, c, c.num_envs, seed=seeds[0])
    eval_rng = np.random.default_rng(seeds[1])
    eval_env = ReachEnv(system, target, cfg, c, 1, seed=eval_rng)
    X_eval = eval_env.sample_states(c.eval_envs) if c.eval_every else None
    m = system.control_dim
    u_center = 0.5 * (system.u_max + system.u_min)
    u_half = 0.5 * (system.u_max - system.u_min)

    obs_dim = env.obs_dim
    model = ActorCritic(obs_dim, m, c).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=c.lr, eps=1e-5)
    obs_rms = RunningMeanStd((obs_dim,))
    ret_rms = RunningMeanStd(())
    disc_return = np.zeros(c.num_envs)

    def norm_obs(F: np.ndarray) -> np.ndarray:
        if not c.norm_obs:
            return F
        return np.clip((F - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -c.obs_clip, c.obs_clip)

    N, S = c.num_envs, c.rollout_steps
    num_updates = max(1, c.total_env_steps // c.batch_size)
    buf_obs = torch.zeros((S, N, obs_dim), device=device)
    buf_act = torch.zeros((S, N, m), device=device)
    buf_logp = torch.zeros((S, N), device=device)
    buf_val = torch.zeros((S, N), device=device)
    rew = np.zeros((S, N))
    dones = np.zeros((S, N))  # episode ended (terminated or truncated) after this step
    raw_obs = env.reset()
    if c.norm_obs:
        obs_rms.update(raw_obs)

    curve: list[dict] = []
    best = None
    env_steps = 0
    for update in range(num_updates):
        if c.anneal_lr:
            optimizer.param_groups[0]["lr"] = c.lr * (1.0 - update / num_updates)
        ep_returns, ep_lens, ep_hits = [], [], []
        raw_batch = np.empty((S, N, obs_dim))
        for step in range(S):
            raw_batch[step] = raw_obs
            obs_t = torch.as_tensor(norm_obs(raw_obs), dtype=torch.float32, device=device)
            with torch.no_grad():
                dist = model.dist(obs_t)
                mean = dist.mean
                a = mean + dist.stddev * torch.randn(mean.shape, generator=gen, device=device)
                logp = dist.log_prob(a).sum(-1)
                v = model.value(obs_t)
            buf_obs[step], buf_act[step], buf_logp[step], buf_val[step] = obs_t, a, logp, v
            a_np = a.cpu().numpy().astype(float)
            U = u_center + u_half * np.clip(a_np, -1.0, 1.0)
            raw_obs, r, term, trunc, info = env.step(U)
            done = term | trunc
            # truncation: bootstrap with V(final state) (normalized with the current statistics)
            if c.norm_reward:
                disc_return = disc_return * c.gamma + r
                ret_rms.update(disc_return)
                disc_return[done] = 0.0
                r_used = r / np.sqrt(ret_rms.var + 1e-8)
            else:
                r_used = r.copy()
            tr = np.flatnonzero(trunc)
            if tr.size:
                fo = torch.as_tensor(norm_obs(info["final_obs"][tr]), dtype=torch.float32, device=device)
                with torch.no_grad():
                    r_used[tr] += c.gamma * model.value(fo).cpu().numpy()
            rew[step] = r_used
            dones[step] = done
            fin = np.flatnonzero(done)
            if fin.size:
                ep_returns.extend(info["ep_return"][fin].tolist())
                ep_lens.extend(info["ep_len"][fin].tolist())
                ep_hits.extend(info["hit"][fin].tolist())
        env_steps += N * S

        # GAE
        with torch.no_grad():
            next_v = model.value(
                torch.as_tensor(norm_obs(raw_obs), dtype=torch.float32, device=device)
            ).cpu().numpy()
        vals = buf_val.cpu().numpy()
        adv = np.zeros((S, N))
        last = np.zeros(N)
        for t in reversed(range(S)):
            nv = next_v if t == S - 1 else vals[t + 1]
            nonterminal = 1.0 - dones[t]
            delta = rew[t] + c.gamma * nv * nonterminal - vals[t]
            last = delta + c.gamma * c.gae_lambda * nonterminal * last
            adv[t] = last
        ret = adv + vals
        if c.norm_obs:
            obs_rms.update(raw_batch.reshape(-1, obs_dim))

        b_obs = buf_obs.reshape(-1, obs_dim)
        b_act = buf_act.reshape(-1, m)
        b_logp = buf_logp.reshape(-1)
        b_adv = torch.as_tensor(adv.reshape(-1), dtype=torch.float32, device=device)
        b_ret = torch.as_tensor(ret.reshape(-1), dtype=torch.float32, device=device)
        b_val = buf_val.reshape(-1)
        B = c.batch_size
        mb = c.minibatch_size
        pg_l = v_l = ent = kl = clipfrac = 0.0
        n_mb = 0
        for epoch in range(c.update_epochs):
            perm = torch.randperm(B, generator=gen, device=device)
            stop = False
            for start in range(0, B, mb):
                idx = perm[start : start + mb]
                dist = model.dist(b_obs[idx])
                new_logp = dist.log_prob(b_act[idx]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                log_ratio = new_logp - b_logp[idx]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    clipfrac += ((ratio - 1.0).abs() > c.clip_coef).float().mean().item()
                a_mb = b_adv[idx]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                pg_loss = torch.max(-a_mb * ratio, -a_mb * ratio.clamp(1 - c.clip_coef, 1 + c.clip_coef)).mean()
                new_v = model.value(b_obs[idx])
                if c.clip_vloss:
                    v_clip = b_val[idx] + (new_v - b_val[idx]).clamp(-c.clip_coef, c.clip_coef)
                    v_loss = 0.5 * torch.max((new_v - b_ret[idx]) ** 2, (v_clip - b_ret[idx]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((new_v - b_ret[idx]) ** 2).mean()
                loss = pg_loss - c.ent_coef * entropy + c.vf_coef * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), c.max_grad_norm)
                optimizer.step()
                pg_l += pg_loss.item()
                v_l += v_loss.item()
                ent += entropy.item()
                kl += approx_kl
                n_mb += 1
                if c.target_kl is not None and approx_kl > c.target_kl:
                    stop = True
                    break
            if stop:
                break

        row = {
            "update": update + 1,
            "env_steps": env_steps,
            "seconds": time.perf_counter() - t_start,
            "episodes": len(ep_returns),
            "ep_return": float(np.mean(ep_returns)) if ep_returns else None,
            "ep_len_seconds": float(np.mean(ep_lens)) * cfg.control_period if ep_lens else None,
            "ep_success": float(np.mean(ep_hits)) if ep_hits else None,
            "pg_loss": pg_l / n_mb,
            "v_loss": v_l / n_mb,
            "entropy": ent / n_mb,
            "approx_kl": kl / n_mb,
            "clipfrac": clipfrac / n_mb,
            "std": float(model.log_std.exp().mean().item()),
        }
        if c.eval_every and ((update + 1) % c.eval_every == 0 or update + 1 == num_updates):
            pol = _extract_policy(model, obs_rms, system, c, {})
            ev = evaluate_in_env(pol, system, target, cfg, X_eval, c.episode_seconds)
            row["eval_success"], row["eval_time"] = ev["success_rate"], ev["mean_reach_time"]
            key = (ev["success_rate"], -ev["mean_reach_time"])
            if c.keep_best and (best is None or key >= best[0]):
                best = (key, pol, update + 1)
        curve.append(row)
        if verbose:
            msg = (
                f"[ppo] upd {update + 1}/{num_updates} steps {env_steps} t {row['seconds']:.0f}s "
                f"ret {row['ep_return']} succ {row['ep_success']} len {row['ep_len_seconds']} "
                f"std {row['std']:.3f} kl {row['approx_kl']:.4f}"
            )
            if "eval_success" in row:
                msg += f" | eval succ {row['eval_success']:.3f} time {row['eval_time']:.2f}"
            print(msg, flush=True)

    final = _extract_policy(model, obs_rms, system, c, {})
    selected_update = num_updates
    if c.keep_best and best is not None:
        final, selected_update = best[1], best[2]
    train_seconds = time.perf_counter() - t_start
    stats = {
        "train_seconds": train_seconds,
        "env_steps": env_steps,
        "env_seconds_simulated": env_steps * cfg.control_period,
        "updates": num_updates,
        "selected_update": selected_update,
        "device": str(device),
        "seed": int(seed),
        "config": c.to_dict(),
        "curve": curve,
    }
    final.meta = {k: v for k, v in stats.items() if k != "curve"}
    return final, stats
