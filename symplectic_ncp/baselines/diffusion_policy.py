"""Diffusion Policy baseline (Chi et al., RSS 2023): conditional denoising diffusion over action chunks.

The policy models p(a_{k:k+Tp} | x_k), the distribution of the next ``Tp``
expert inputs (one per control period) given the current state, with a DDPM
(epsilon prediction, squared-cosine noise schedule, ``train_steps`` noise
levels).  At run time a chunk is sampled with a deterministic DDIM sampler
(eta = 0) on ``inference_steps`` noise levels and its first ``Ta`` inputs
are executed open loop, one per control period (receding horizon; the
closed loop is :func:`symplectic_ncp.simulation.simulate_feedback_policy`,
which accepts the (B, Ta, m) chunks returned by :class:`DiffusionPolicy`).

Implementation choices (Chi et al. leave these to the task or use a CNN
U-Net for images; documented in ``DiffusionPolicyConfig``):

* observation horizon To = 1: the state is Markov, so the condition is the
  current state, encoded like BC (angles -> (sin, cos)) and standardized;
* noise-prediction network: MLP on the flattened noisy chunk with
  ``num_blocks`` residual blocks, each FiLM-conditioned on an embedding of
  (state features, sinusoidal diffusion-step embedding) -- the MLP
  counterpart of Chi et al.'s FiLM-conditioned temporal U-Net, much faster
  on these 1-D input, 2-D state problems;
* actions standardized per input coordinate; the predicted clean chunk is
  clipped to the standardized input bounds at every sampler step (the
  ``clip_sample`` of Chi et al. with the actual input bounds), so samples
  always respect u_min <= u <= u_max;
* AdamW, linear warm-up + cosine learning-rate decay, EMA of the weights
  used for inference (as in Chi et al.);
* sampling noise comes from a seeded CPU ``torch.Generator``: with
  ``noise="stream"`` every query draws fresh noise from the stream (reset
  it with :meth:`DiffusionPolicy.reset` before an evaluation); with
  ``noise="fixed"`` every query starts from the same noise sample, which
  makes the policy a deterministic function of the state.
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from symplectic_ncp.baselines.behavior_cloning import state_features
from symplectic_ncp.experts.demonstration import Demonstration
from symplectic_ncp.systems import HamiltonianSystem
from symplectic_ncp.systems.base import as_batch


@dataclass
class DiffusionPolicyConfig:
    # network
    hidden: int = 256  # width of the residual MLP
    num_blocks: int = 3  # FiLM-conditioned residual blocks
    step_embed_dim: int = 64  # sinusoidal diffusion-step embedding
    cond_dim: int = 128  # embedding of (state features, step embedding)
    angle_features: bool = True  # angles -> (sin, cos), as for BC
    # action chunks (in control periods)
    pred_horizon: int = 16  # Tp: predicted inputs per query (Chi et al.: 16)
    action_horizon: int = 8  # Ta: inputs executed per query (Chi et al.: 8)
    # diffusion
    train_steps: int = 100  # K, noise levels of the DDPM (Chi et al.: 100)
    inference_steps: int = 10  # DDIM steps at run time (Chi et al.: 10 with DDIM)
    clip_sample: bool = True  # clip predicted clean chunks to the input bounds
    noise: str = "stream"  # "stream": fresh seeded noise per query; "fixed": same noise every query
    noise_seed: int = 0
    obs_noise: float = 0.0  # std of Gaussian noise added to the standardized state features in training
    # optimization
    iterations: int = 20000
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-6
    warmup: int = 500
    ema_decay: float = 0.999
    log_every: int = 500
    device: str = "auto"  # "auto": cuda if available, else cpu
    cuda_graph: bool = True  # capture the training step and the sampler in CUDA graphs (launch overhead)


def diffusion_policy_config(system_name: str) -> DiffusionPolicyConfig:
    """Default diffusion-policy hyperparameters for ``system_name``."""
    if system_name == "spring_mass":
        return DiffusionPolicyConfig()
    if system_name == "single_pendulum":
        return DiffusionPolicyConfig()
    raise ValueError(f"unknown system {system_name!r}")


def resolve_device(device: str | None) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# ------------------------------------------------------------------- data
def chunk_dataset(demos: list[Demonstration], pred_horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Observation / action-chunk pairs of all demonstrations.

    Observations are the states at the expert's update instants (every
    control period); the target of the k-th one is the expert's next
    ``pred_horizon`` inputs (k, ..., k + Tp - 1), padded past the end of the
    demonstration with its last input.  Returns X (N, n) and A (N, Tp, m).
    """
    obs, chunks = [], []
    for d in demos:
        X, U = d.control_samples()
        K = U.shape[0]
        idx = np.minimum(np.arange(K)[:, None] + np.arange(pred_horizon)[None, :], K - 1)
        obs.append(X)
        chunks.append(U[idx])
    return np.concatenate(obs), np.concatenate(chunks)


def _safe_std(values: np.ndarray) -> np.ndarray:
    std = values.std(axis=0)
    return np.where(std > 1e-12, std, 1.0)


# --------------------------------------------------------------- schedule
def cosine_alpha_bar(num_steps: int, s: float = 0.008, max_beta: float = 0.999) -> np.ndarray:
    """Cumulative products alpha_bar_k, k = 0..K-1, of the squared-cosine schedule (Nichol & Dhariwal)."""

    def f(t):
        return math.cos((t + s) / (1.0 + s) * math.pi / 2.0) ** 2

    betas = np.array([min(1.0 - f((k + 1) / num_steps) / f(k / num_steps), max_beta) for k in range(num_steps)])
    return np.cumprod(1.0 - betas)


def ddim_timesteps(train_steps: int, inference_steps: int) -> np.ndarray:
    """Decreasing noise levels visited by the DDIM sampler (evenly spaced, ending at 0)."""
    if not 1 <= inference_steps <= train_steps:
        raise ValueError("need 1 <= inference_steps <= train_steps")
    step = train_steps // inference_steps
    return (np.arange(inference_steps) * step)[::-1].copy()


# ---------------------------------------------------------------- network
def sinusoidal_embedding(k: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=k.device, dtype=torch.float32) / max(half - 1, 1))
    args = k.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class FiLMBlock(nn.Module):
    def __init__(self, hidden: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.lin1 = nn.Linear(hidden, hidden)
        self.film = nn.Linear(cond_dim, 2 * hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        self.act = nn.Mish()

    def forward(self, h, c):
        scale, shift = self.film(c).chunk(2, dim=1)
        z = self.act(self.lin1(self.norm(h)) * (1.0 + scale) + shift)
        return h + self.lin2(z)


class NoisePredictor(nn.Module):
    """eps_theta(a_k, k, x): residual MLP with FiLM conditioning on (state, diffusion step)."""

    def __init__(self, obs_dim: int, action_dim: int, cfg: DiffusionPolicyConfig):
        super().__init__()
        self.step_embed_dim = cfg.step_embed_dim
        self.step_mlp = nn.Sequential(
            nn.Linear(cfg.step_embed_dim, cfg.step_embed_dim), nn.Mish(), nn.Linear(cfg.step_embed_dim, cfg.step_embed_dim)
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(obs_dim + cfg.step_embed_dim, cfg.cond_dim), nn.Mish(), nn.Linear(cfg.cond_dim, cfg.cond_dim)
        )
        self.inp = nn.Linear(action_dim, cfg.hidden)
        self.blocks = nn.ModuleList([FiLMBlock(cfg.hidden, cfg.cond_dim) for _ in range(cfg.num_blocks)])
        self.out_norm = nn.LayerNorm(cfg.hidden)
        self.out = nn.Linear(cfg.hidden, action_dim)

    def condition(self, obs, k):
        temb = self.step_mlp(sinusoidal_embedding(k, self.step_embed_dim))
        return self.cond_mlp(torch.cat([obs, temb], dim=1))

    def forward(self, a, k, obs):
        c = self.condition(obs, k)
        h = self.inp(a)
        for block in self.blocks:
            h = block(h, c)
        return self.out(torch.nn.functional.mish(self.out_norm(h)))


# ----------------------------------------------------------------- policy
@dataclass
class DiffusionPolicy:
    """Trained diffusion policy; ``policy(X)`` returns the next ``Ta`` inputs, shape (B, Ta, m)."""

    model: NoisePredictor  # EMA weights
    cfg: DiffusionPolicyConfig
    x_mean: np.ndarray
    x_std: np.ndarray
    u_mean: np.ndarray  # (m,)
    u_std: np.ndarray  # (m,)
    state_dim: int
    control_dim: int
    angle_indices: tuple[int, ...]
    u_min: np.ndarray
    u_max: np.ndarray
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    train_losses: list[float] = field(default_factory=list)

    def __post_init__(self):
        self.device = torch.device(self.device)
        self.model = self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        cfg = self.cfg
        ab = cosine_alpha_bar(cfg.train_steps)
        self._timesteps = ddim_timesteps(cfg.train_steps, cfg.inference_steps)
        prev = np.append(self._timesteps[1:], -1)
        ab_t = ab[self._timesteps]
        ab_prev = np.where(prev >= 0, ab[np.maximum(prev, 0)], 1.0)
        dev = self.device
        self._sqrt_ab = torch.tensor(np.sqrt(ab_t), dtype=torch.float32, device=dev)
        self._sqrt_1mab = torch.tensor(np.sqrt(1.0 - ab_t), dtype=torch.float32, device=dev)
        self._sqrt_ab_prev = torch.tensor(np.sqrt(ab_prev), dtype=torch.float32, device=dev)
        self._sqrt_1mab_prev = torch.tensor(np.sqrt(1.0 - ab_prev), dtype=torch.float32, device=dev)
        Tp, m = cfg.pred_horizon, self.control_dim
        lo = np.tile((np.asarray(self.u_min, float) - self.u_mean) / self.u_std, Tp)
        hi = np.tile((np.asarray(self.u_max, float) - self.u_mean) / self.u_std, Tp)
        self._lo = torch.tensor(lo, dtype=torch.float32, device=dev)
        self._hi = torch.tensor(hi, dtype=torch.float32, device=dev)
        self._x_mean = torch.tensor(self.x_mean, dtype=torch.float32, device=dev)
        self._x_std = torch.tensor(self.x_std, dtype=torch.float32, device=dev)
        self._k = torch.tensor(self._timesteps, dtype=torch.long, device=dev)
        self._graphs: dict = {}
        self.reset()

    @property
    def action_horizon(self) -> int:
        return self.cfg.action_horizon

    def reset(self, seed: int | None = None) -> None:
        """Restart the sampling-noise stream (call before every evaluation for reproducibility)."""
        seed = self.cfg.noise_seed if seed is None else seed
        self._gen = torch.Generator(device="cpu").manual_seed(int(seed))
        self._fixed = torch.randn(self.cfg.pred_horizon * self.control_dim, generator=self._gen)

    def _noise(self, B: int) -> torch.Tensor:
        if self.cfg.noise == "fixed":
            z = self._fixed.expand(B, -1)
        elif self.cfg.noise == "stream":
            z = torch.randn(B, self.cfg.pred_horizon * self.control_dim, generator=self._gen)
        else:
            raise ValueError(f"noise must be 'stream' or 'fixed', got {self.cfg.noise!r}")
        return z.to(self.device)

    def _denoise(self, obs: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """DDIM (eta = 0) from the noise ``a`` to a clean standardized chunk (B, Tp m)."""
        B = a.shape[0]
        for i in range(len(self._timesteps)):
            eps = self.model(a, self._k[i].expand(B), obs)
            x0 = (a - self._sqrt_1mab[i] * eps) / self._sqrt_ab[i]
            if self.cfg.clip_sample:
                x0 = torch.maximum(torch.minimum(x0, self._hi), self._lo)
                eps = (a - self._sqrt_ab[i] * x0) / self._sqrt_1mab[i]
            a = self._sqrt_ab_prev[i] * x0 + self._sqrt_1mab_prev[i] * eps
        return a

    def _graphed_denoise(self, obs: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """``_denoise`` replayed from a CUDA graph captured for the next power-of-two batch size.

        Rows are independent, so the zero padding does not affect the result.
        """
        B = a.shape[0]
        size = max(16, 1 << (B - 1).bit_length())
        if size not in self._graphs:
            s_obs = torch.zeros(size, obs.shape[1], device=self.device)
            s_a = torch.zeros(size, a.shape[1], device=self.device)
            stream = torch.cuda.Stream(self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._denoise(s_obs, s_a)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                s_out = self._denoise(s_obs, s_a)
            self._graphs[size] = (graph, s_obs, s_a, s_out)
        graph, s_obs, s_a, s_out = self._graphs[size]
        s_obs.zero_()
        s_a.zero_()
        s_obs[:B] = obs
        s_a[:B] = a
        graph.replay()
        return s_out[:B]

    @torch.no_grad()
    def sample_chunk(self, X) -> np.ndarray:
        """Full predicted chunk (B, Tp, m) in physical units, clipped to the input bounds."""
        X = as_batch(X, self.state_dim)
        B = X.shape[0]
        Tp, m = self.cfg.pred_horizon, self.control_dim
        if B == 0:
            return np.zeros((0, Tp, m))
        F = state_features(X, self.angle_indices, self.cfg.angle_features)
        obs = (torch.as_tensor(F, dtype=torch.float32, device=self.device) - self._x_mean) / self._x_std
        a = self._noise(B)
        if self.device.type == "cuda" and self.cfg.cuda_graph:
            a = self._graphed_denoise(obs, a)
        else:
            a = self._denoise(obs, a)
        U = a.double().cpu().numpy().reshape(B, Tp, m) * self.u_std + self.u_mean
        return np.clip(U, self.u_min, self.u_max)

    def __call__(self, X) -> np.ndarray:
        """Next ``Ta`` inputs for every state, shape (B, Ta, m), clipped to [u_min, u_max]."""
        return self.sample_chunk(X)[:, : self.cfg.action_horizon]

    # ------------------------------------------------------------ persistence
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "cfg": dataclasses.asdict(self.cfg),
                "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
                "x_mean": self.x_mean, "x_std": self.x_std, "u_mean": self.u_mean, "u_std": self.u_std,
                "state_dim": self.state_dim, "control_dim": self.control_dim,
                "angle_indices": tuple(self.angle_indices), "u_min": self.u_min, "u_max": self.u_max,
                "train_losses": list(self.train_losses),
            },
            path,
        )

    @classmethod
    def load(cls, path, device: str | None = None) -> "DiffusionPolicy":
        data = torch.load(Path(path), map_location="cpu", weights_only=False)
        cfg = DiffusionPolicyConfig(**data["cfg"])
        model = NoisePredictor(len(data["x_mean"]), data["cfg"]["pred_horizon"] * data["control_dim"], cfg)
        model.load_state_dict(data["state_dict"])
        return cls(
            model=model, cfg=cfg, x_mean=data["x_mean"], x_std=data["x_std"], u_mean=data["u_mean"],
            u_std=data["u_std"], state_dim=data["state_dim"], control_dim=data["control_dim"],
            angle_indices=tuple(data["angle_indices"]), u_min=data["u_min"], u_max=data["u_max"],
            device=resolve_device(device if device is not None else cfg.device), train_losses=data["train_losses"],
        )


# --------------------------------------------------------------- training
def train_diffusion_policy(
    system: HamiltonianSystem,
    demos: list[Demonstration],
    dp_cfg: DiffusionPolicyConfig,
    seed: int = 0,
    device: str | None = None,
) -> tuple[DiffusionPolicy, dict]:
    """Fit the denoising network to the demonstrations' action chunks.

    Deterministic given ``seed`` on a given device (weights initialized on
    the CPU, mini-batches / noise levels / noise from a seeded generator on
    the training device).  Returns the policy (EMA weights) and a dict with
    ``train_seconds`` (wall clock incl. data preparation), ``iterations``,
    ``device``, ``final_loss`` (mean denoising MSE over the last
    ``log_every`` iterations), ``num_samples`` and ``loss_history``.
    """
    if not demos:
        raise ValueError("diffusion policy needs at least one demonstration")
    cfg = dp_cfg
    dev = resolve_device(device if device is not None else cfg.device)
    if dev.type == "cuda":  # one-time CUDA context / cuBLAS initialization is not training time
        (torch.ones(8, 8, device=dev) @ torch.ones(8, 8, device=dev)).sum().item()
        torch.cuda.synchronize(dev)
    start = time.perf_counter()

    X, A = chunk_dataset(demos, cfg.pred_horizon)
    angle_indices = tuple(system.angle_indices)
    F = state_features(X, angle_indices, cfg.angle_features)
    x_mean, x_std = F.mean(axis=0), _safe_std(F)
    m = A.shape[2]
    flatU = A.reshape(-1, m)
    u_mean, u_std = flatU.mean(axis=0), _safe_std(flatU)
    obs = torch.tensor((F - x_mean) / x_std, dtype=torch.float32, device=dev)
    act = torch.tensor(((A - u_mean) / u_std).reshape(A.shape[0], -1), dtype=torch.float32, device=dev)
    N, D = act.shape
    ab = torch.tensor(cosine_alpha_bar(cfg.train_steps), dtype=torch.float32, device=dev)
    sqrt_ab, sqrt_1mab = ab.sqrt(), (1.0 - ab).sqrt()

    torch.manual_seed(seed)
    model = NoisePredictor(F.shape[1], D, cfg).to(dev)
    ema = NoisePredictor(F.shape[1], D, cfg).to(dev)
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    params, ema_params = list(model.parameters()), list(ema.parameters())
    use_graph = dev.type == "cuda" and cfg.cuda_graph
    lr = torch.tensor(0.0, device=dev)  # tensor learning rate / EMA decay: updated in place between steps
    decay = torch.tensor(0.0, device=dev)
    extra = {"fused": True, "capturable": True} if dev.type == "cuda" else {}
    opt = torch.optim.AdamW(params, lr=lr if dev.type == "cuda" else cfg.lr, weight_decay=cfg.weight_decay, **extra)
    total, bs = cfg.iterations, cfg.batch_size

    def lr_at(it):  # linear warm-up, then cosine decay to zero
        if it < cfg.warmup:
            return cfg.lr * (it + 1) / cfg.warmup
        return cfg.lr * 0.5 * (1.0 + math.cos(math.pi * (it - cfg.warmup) / max(total - cfg.warmup, 1)))

    # Random numbers are drawn in banks of ``bank`` iterations from a seeded
    # generator on the training device; the step reads row ``ptr`` of the bank,
    # so the same step function runs eagerly or replayed from a CUDA graph.
    bank = 250
    gen = torch.Generator(device=dev).manual_seed(seed)
    bank_idx = torch.empty(bank, bs, dtype=torch.long, device=dev)
    bank_k = torch.empty(bank, bs, dtype=torch.long, device=dev)
    bank_eps = torch.empty(bank, bs, D, device=dev)
    bank_obs = torch.zeros(bank, bs, obs.shape[1], device=dev)  # observation-noise augmentation
    ptr = torch.zeros(1, dtype=torch.long, device=dev)
    window = torch.zeros((), device=dev)

    def refill():
        torch.randint(0, N, (bank, bs), device=dev, generator=gen, out=bank_idx)
        torch.randint(0, cfg.train_steps, (bank, bs), device=dev, generator=gen, out=bank_k)
        bank_eps.normal_(generator=gen)
        if cfg.obs_noise > 0.0:
            bank_obs.normal_(generator=gen).mul_(cfg.obs_noise)
        ptr.zero_()

    def step():
        idx = bank_idx.index_select(0, ptr)[0]
        k = bank_k.index_select(0, ptr)[0]
        eps = bank_eps.index_select(0, ptr)[0]
        cond = obs[idx] + bank_obs.index_select(0, ptr)[0]
        noisy = sqrt_ab[k, None] * act[idx] + sqrt_1mab[k, None] * eps
        loss = torch.mean((model(noisy, k, cond) - eps) ** 2)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=False)
        with torch.no_grad():
            torch._foreach_mul_(ema_params, decay)  # ema <- decay ema + (1 - decay) theta
            torch._foreach_add_(ema_params, torch._foreach_mul(params, 1.0 - decay))
            window.add_(loss.detach())
            ptr.add_(1)

    def prepare(it):
        if it % bank == 0:
            refill()
        if dev.type == "cuda":
            lr.fill_(lr_at(it))
        else:
            opt.param_groups[0]["lr"] = lr_at(it)
        decay.fill_(min(cfg.ema_decay, (1.0 + it) / (10.0 + it)))  # EMA warm-up

    graph = None
    history: list[float] = []
    count = 0
    for it in range(total):
        prepare(it)
        if graph is not None:
            graph.replay()
        elif use_graph and it >= 3:  # capture after a few eager warm-up steps on a side stream
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                step()
            graph.replay()
        elif use_graph:
            stream = torch.cuda.Stream(dev)
            stream.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(stream):
                step()
            torch.cuda.current_stream(dev).wait_stream(stream)
        else:
            step()
        count += 1
        if count == cfg.log_every or it == total - 1:
            history.append(float(window) / count)
            window.zero_()
            count = 0
    del graph
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    train_seconds = time.perf_counter() - start

    policy = DiffusionPolicy(
        model=ema, cfg=cfg, x_mean=x_mean, x_std=x_std, u_mean=u_mean, u_std=u_std,
        state_dim=system.state_dim, control_dim=system.control_dim, angle_indices=angle_indices,
        u_min=system.u_min.copy(), u_max=system.u_max.copy(), device=dev, train_losses=history,
    )
    info = {
        "train_seconds": train_seconds,
        "iterations": total,
        "device": str(dev),
        "final_loss": history[-1] if history else float("nan"),
        "num_samples": int(N),
        "loss_history": history,
    }
    return policy, info
