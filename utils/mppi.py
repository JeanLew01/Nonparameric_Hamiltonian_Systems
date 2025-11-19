import jax
import jax.numpy as jnp
from functools import partial
class MPPI:
    def __init__(self, dyn_fn, dt=0.01, horizon=60, n_samples=2000,
                 Q=None, QT=None, R=None, x_ref=None,
                 noise_scale=jnp.array([4.0, 4.0]), temperature=1.0, seed=0):
        self.dyn = dyn_fn
        self.dt = dt
        self.N = horizon
        self.n_samples = n_samples
        self.Q = Q
        self.QT = QT
        self.R = R
        self.x_ref = x_ref
        self.noise_scale = noise_scale
        self.temperature = temperature
        self.U = jnp.zeros((self.N, R.shape[0]))
        self.rng_key = jax.random.PRNGKey(seed)

    def set_sequence(self, U0):
        self.U = U0

    def _state_error(self, x):
        th1, th1d, th2, th2d = x
        r1, r1d, r2, r2d = self.x_ref
        def wrap(a): return (a + jnp.pi) % (2*jnp.pi) - jnp.pi
        return jnp.array([wrap(th1 - r1), th1d - r1d, wrap(th2 - r2), th2d - r2d],
                         dtype=jnp.float32)

    def _rollout_cost(self, x0, U_seq):
        def scan_rollout(carry, u):
            c, x = carry
            e = self._state_error(x)
            stage = e @ self.Q @ e + u @ self.R @ u
            x_next = self.dyn(x, u, self.dt)
            return (c + stage, x_next), None
        (c, xT), _ = jax.lax.scan(scan_rollout, (0.0, x0), U_seq)
        eT = self._state_error(xT)
        return c + eT @ self.QT @ eT, xT

    @partial(jax.jit, static_argnums=0)   # <-- make `self` static
    def _mppi_step(self, x0, U, rng_key):
        keys = jax.random.split(rng_key, self.n_samples)
        noises = jax.vmap(lambda k: jax.random.normal(k, (self.N, U.shape[1])))(keys) * self.noise_scale
        U_samples = U + noises
        costs, _ = jax.vmap(lambda U_seq: self._rollout_cost(x0, U_seq))(U_samples)
        cmin = jnp.min(costs)
        w = jnp.exp((cmin - costs) * self.temperature)
        wsum = jnp.sum(w) + 1e-8
        U_star = jnp.sum(U_samples * w[:, None, None], axis=0) / wsum
        U_new = jnp.roll(U_star, -1, axis=0).at[-1].set(U_star[-1])
        return U_star[0], U_new

    def step(self, x):
        self.rng_key, sub = jax.random.split(self.rng_key)
        u0, U_new = self._mppi_step(x, self.U, sub)
        self.U = U_new
        return u0