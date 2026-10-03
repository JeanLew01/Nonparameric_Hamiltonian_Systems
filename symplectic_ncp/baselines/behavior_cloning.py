"""Vanilla behavior cloning baseline (paper, Section IV).

"The BC policy is a three-layer multilayer perceptron with hidden sizes
(24, 24, 16), trained with Adam on the mean-squared imitation loss using
learning rate 1.2e-3, weight decay 5e-4, and 40 epochs."

Choices the paper does not state (documented here and in ``BCConfig``):
ReLU activations; inputs and targets standardized with statistics of the
training set; angles encoded as (sin, cos) when ``cfg.angle_features``;
mini-batch size ``cfg.batch_size``; PyTorch default initialization; Adam's
``weight_decay`` is the coupled L2 penalty of ``torch.optim.Adam``; training
pairs are the expert's (state, input) samples at its own update instants
(:meth:`Demonstration.control_samples`); outputs clipped to ``[u_min, u_max]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from symplectic_ncp.config import BCConfig
from symplectic_ncp.experts.demonstration import Demonstration
from symplectic_ncp.systems import HamiltonianSystem
from symplectic_ncp.systems.base import as_batch


def state_features(X: np.ndarray, angle_indices: tuple[int, ...], angle_features: bool) -> np.ndarray:
    """Raw MLP input: each angle coordinate becomes (sin, cos) if ``angle_features``, others are kept."""
    if not angle_features or not angle_indices:
        return np.asarray(X, dtype=float)
    cols = []
    for i in range(X.shape[1]):
        if i in angle_indices:
            cols += [np.sin(X[:, i]), np.cos(X[:, i])]
        else:
            cols.append(X[:, i])
    return np.stack(cols, axis=1)


def build_mlp(in_dim: int, hidden_sizes: tuple[int, ...], out_dim: int) -> nn.Sequential:
    """MLP with ReLU hidden layers ``hidden_sizes`` and a linear output layer."""
    layers: list[nn.Module] = []
    prev = in_dim
    for width in hidden_sizes:
        layers += [nn.Linear(prev, width), nn.ReLU()]
        prev = width
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def _safe_std(values: np.ndarray) -> np.ndarray:
    std = values.std(axis=0)
    return np.where(std > 1e-12, std, 1.0)


@dataclass
class BehaviorCloningPolicy:
    """Trained BC policy u = clip(MLP(x)) with a numpy forward pass.

    ``weights[k]`` has shape (in_k, out_k) so that a layer is ``h @ W + b``.
    """

    weights: list[np.ndarray]
    biases: list[np.ndarray]
    x_mean: np.ndarray
    x_std: np.ndarray
    u_mean: np.ndarray
    u_std: np.ndarray
    state_dim: int
    angle_indices: tuple[int, ...]
    angle_features: bool
    u_min: np.ndarray
    u_max: np.ndarray
    train_losses: list[float] = field(default_factory=list)  # mean standardized MSE per epoch

    def __post_init__(self):
        # Fold the standardizations into the first and last layers: the
        # forward pass is then a plain MLP on the raw features.
        self.weights = [np.asarray(W, dtype=float) for W in self.weights]
        self.biases = [np.asarray(b, dtype=float) for b in self.biases]
        fused_W, fused_b = list(self.weights), list(self.biases)
        fused_b[0] = fused_b[0] - (self.x_mean / self.x_std) @ fused_W[0]
        fused_W[0] = fused_W[0] / self.x_std[:, None]
        fused_b[-1] = fused_b[-1] * self.u_std + self.u_mean
        fused_W[-1] = fused_W[-1] * self.u_std[None, :]
        self._layers = list(zip(fused_W, fused_b))

    @property
    def control_dim(self) -> int:
        return self.u_mean.shape[0]

    def __call__(self, X) -> np.ndarray:
        """U = clip(pi_BC(X)), shape (B, m)."""
        X = as_batch(X, self.state_dim)
        h = state_features(X, self.angle_indices, self.angle_features)
        last = len(self._layers) - 1
        for k, (W, b) in enumerate(self._layers):
            h = h @ W + b
            if k < last:
                np.maximum(h, 0.0, out=h)
        return np.clip(h, self.u_min, self.u_max)

    def to_torch(self) -> nn.Sequential:
        """Rebuild the (standardized-space) torch MLP from the stored weights."""
        hidden = tuple(W.shape[1] for W in self.weights[:-1])
        model = build_mlp(self.weights[0].shape[0], hidden, self.weights[-1].shape[1]).double()
        linears = [m for m in model if isinstance(m, nn.Linear)]
        with torch.no_grad():
            for lin, W, b in zip(linears, self.weights, self.biases):
                lin.weight.copy_(torch.from_numpy(W.T))
                lin.bias.copy_(torch.from_numpy(b))
        return model

    # ------------------------------------------------------------ persistence
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {f"W{k}": W for k, W in enumerate(self.weights)}
        payload.update({f"b{k}": b for k, b in enumerate(self.biases)})
        payload.update(
            num_layers=len(self.weights),
            x_mean=self.x_mean,
            x_std=self.x_std,
            u_mean=self.u_mean,
            u_std=self.u_std,
            state_dim=self.state_dim,
            angle_indices=np.asarray(self.angle_indices, dtype=int),
            angle_features=self.angle_features,
            u_min=self.u_min,
            u_max=self.u_max,
            train_losses=np.asarray(self.train_losses, dtype=float),
        )
        np.savez(path, **payload)

    @classmethod
    def load(cls, path) -> "BehaviorCloningPolicy":
        data = np.load(Path(path))
        L = int(data["num_layers"])
        return cls(
            weights=[data[f"W{k}"] for k in range(L)],
            biases=[data[f"b{k}"] for k in range(L)],
            x_mean=data["x_mean"],
            x_std=data["x_std"],
            u_mean=data["u_mean"],
            u_std=data["u_std"],
            state_dim=int(data["state_dim"]),
            angle_indices=tuple(int(i) for i in data["angle_indices"]),
            angle_features=bool(data["angle_features"]),
            u_min=data["u_min"],
            u_max=data["u_max"],
            train_losses=[float(v) for v in data["train_losses"]],
        )


def imitation_dataset(demos: list[Demonstration], sample_grid: str = "control") -> tuple[np.ndarray, np.ndarray]:
    """Stack the (state, input) pairs of all demonstrations.

    ``sample_grid="control"``: pairs at the expert's update instants (every
    control period); ``"simulation"``: pairs at every simulation step.
    """
    if sample_grid == "control":
        pairs = [d.control_samples() for d in demos]
    elif sample_grid == "simulation":
        pairs = [(d.states[:-1], d.controls) for d in demos]
    else:
        raise ValueError(f"sample_grid must be 'control' or 'simulation', got {sample_grid!r}")
    return np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs])


def train_behavior_cloning(
    system: HamiltonianSystem, demos: list[Demonstration], cfg: BCConfig, seed: int = 0
) -> BehaviorCloningPolicy:
    """Fit the BC MLP to the demonstrations by MSE imitation (deterministic given ``seed``)."""
    if not demos:
        raise ValueError("behavior cloning needs at least one demonstration")
    X, U = imitation_dataset(demos, cfg.sample_grid)
    angle_indices = tuple(system.angle_indices)
    F = state_features(X, angle_indices, cfg.angle_features)
    x_mean, x_std = F.mean(axis=0), _safe_std(F)
    u_mean, u_std = U.mean(axis=0), _safe_std(U)
    inputs = torch.from_numpy((F - x_mean) / x_std).float()
    targets = torch.from_numpy((U - u_mean) / u_std).float()

    threads = torch.get_num_threads()
    torch.set_num_threads(1)  # single-threaded CPU kernels: bitwise reproducible
    try:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = build_mlp(F.shape[1], tuple(cfg.hidden_sizes), U.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        loss_fn = nn.MSELoss()
        n = inputs.shape[0]
        losses = []
        for _ in range(cfg.epochs):
            order = torch.from_numpy(rng.permutation(n))
            total = 0.0
            for start in range(0, n, cfg.batch_size):
                idx = order[start : start + cfg.batch_size]
                optimizer.zero_grad()
                loss = loss_fn(model(inputs[idx]), targets[idx])
                loss.backward()
                optimizer.step()
                total += loss.item() * idx.shape[0]
            losses.append(total / n)
    finally:
        torch.set_num_threads(threads)

    linears = [m for m in model if isinstance(m, nn.Linear)]
    return BehaviorCloningPolicy(
        weights=[lin.weight.detach().double().numpy().T.copy() for lin in linears],
        biases=[lin.bias.detach().double().numpy().copy() for lin in linears],
        x_mean=x_mean,
        x_std=x_std,
        u_mean=u_mean,
        u_std=u_std,
        state_dim=system.state_dim,
        angle_indices=angle_indices,
        angle_features=cfg.angle_features,
        u_min=system.u_min.copy(),
        u_max=system.u_max.copy(),
        train_losses=losses,
    )
