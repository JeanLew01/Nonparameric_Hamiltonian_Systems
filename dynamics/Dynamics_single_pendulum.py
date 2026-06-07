import jax
import jax.numpy as jnp

import gymnasium as gym
from gymnasium import spaces

import numpy as np
import casadi as cs

g = 9.81
m = 1.0
l = 2.0
I = m * l**2

u_min = jnp.array([-20.0], dtype=jnp.float32)
u_max = jnp.array([+20.0], dtype=jnp.float32)


def angle_wrap(x):
    """Wrap angle to (-pi, pi]."""
    return (x + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def state_error(x, x_ref):
    """Angle error wrapped, momentum error linear."""
    theta, p = x
    r1, r2 = x_ref
    return jnp.array(
        [angle_wrap(theta - r1), p - r2],
        dtype=jnp.float32,
    )


def single_pendulum_pdot(x, u):
    theta, _p = x
    (tau,) = u
    return -m * g * l * jnp.sin(theta) + tau


def f_continuous(x, u):
    """State derivative in canonical coordinates [theta, p]."""
    theta, p = x
    theta_dot = p / I
    p_dot = single_pendulum_pdot(x, u)
    return jnp.array([theta_dot, p_dot], dtype=jnp.float32)


def rk4_step(x, u, h):
    """RK4 integrator."""
    k1 = f_continuous(x, u)
    k2 = f_continuous(x + 0.5 * h * k1, u)
    k3 = f_continuous(x + 0.5 * h * k2, u)
    k4 = f_continuous(x + h * k3, u)
    return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@jax.jit
def jax_dynamics(x, u, dt=0.01):
    """JIT-ed single pendulum dynamics step with input clipping."""
    u = jnp.clip(u, u_min, u_max)
    return rk4_step(x, u, dt)


def wrap_angle_np(theta: float) -> float:
    return ((theta + np.pi) % (2 * np.pi)) - np.pi


class SinglePendulumEnv(gym.Env):
    """
    Lossless single pendulum environment in canonical coordinates.

    - Observation: [theta, p] (theta in radians, momentum in Nms)
    - Action: [tau] (Nm), clipped to [u_min, u_max]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dt=0.02,
        episode_seconds=8.0,
        Q=np.diag([5.0, 1.0]),
        R=np.array([[1e-2]], dtype=float),
        QT=np.diag([10.0, 3.0]),
        x_ref=np.array([np.pi, 0.0], dtype=float),
        u_min=np.array([-20.0], dtype=float),
        u_max=np.array([+20.0], dtype=float),
        success_tol=np.array([0.05, 0.05], dtype=float),
        default_x0=np.array([0.0, 0.0], dtype=float),
        reset_noise_std=np.array([0.02, 0.02], dtype=float),
        seed: int | None = None,
        success_eps: float = 0.2,
        terminal_bonus: float = 50.0,
    ):
        super().__init__()

        self.dt = float(dt)
        self.T = float(episode_seconds)
        self.horizon_steps = int(np.round(self.T / self.dt))

        self.Q = np.array(Q, dtype=np.float32)
        self.R = np.array(R, dtype=np.float32)
        self.QT = np.array(QT, dtype=np.float32)
        self.x_ref = np.array(x_ref, dtype=np.float32)

        self.u_min = np.array(u_min, dtype=np.float32)
        self.u_max = np.array(u_max, dtype=np.float32)
        self.success_tol = np.array(success_tol, dtype=np.float32)
        self.default_x0 = np.array(default_x0, dtype=np.float32)
        self.reset_noise_std = np.array(reset_noise_std, dtype=np.float32)

        self.nx = 2
        self.nu = 1

        high_x = np.array([np.pi, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high_x, high=high_x, dtype=np.float32)
        self.action_space = spaces.Box(
            low=self.u_min, high=self.u_max, shape=(1,), dtype=np.float32
        )

        self._build_casadi_dynamics()
        self._build_casadi_integrator()

        self._x: np.ndarray | None = None
        self._k: int = 0

        self._np_random = None
        self.seed(seed)

        self.success_eps = float(success_eps)
        self.terminal_bonus = float(terminal_bonus)

    def _build_casadi_dynamics(self):
        g_local = 9.81
        m_local = 1.0
        l_local = 2.0
        I_local = m_local * l_local**2

        theta = cs.SX.sym("theta")
        p = cs.SX.sym("p")
        tau = cs.SX.sym("tau")

        x = cs.vertcat(theta, p)
        u = cs.vertcat(tau)

        theta_dot = p / I_local
        p_dot = -m_local * g_local * l_local * cs.sin(theta) + tau
        xdot = cs.vertcat(theta_dot, p_dot)

        self._f = cs.Function("f", [x, u], [xdot], ["x", "u"], ["xdot"])

    def _build_casadi_integrator(self):
        x = cs.SX.sym("x", self.nx)
        u = cs.SX.sym("u", self.nu)
        xdot = self._f(x, u)
        dae = {"x": x, "p": u, "ode": xdot}
        opts = {"tf": self.dt}
        self._integrator = cs.integrator("integrator", "cvodes", dae, opts)

    def _step_integrate(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        out = self._integrator(x0=x, p=u)
        x_next = np.array(out["xf"]).squeeze().astype(np.float32)
        return x_next

    def seed(self, seed: int | None = None):
        """Set the environment seed."""
        self._np_random = np.random.default_rng(seed)
        return [seed]

    def reset(self, *, seed: int | None = None, options=None):
        """
        Gymnasium reset API: returns (obs, info).
        options keys:
          - "x0": explicit initial state (shape (2,))
          - "random_range": list/tuple of 2 (low, high) pairs
        """
        if seed is not None:
            self.seed(seed)
        if options is None:
            options = {}

        if "x0" in options:
            x0 = np.array(options["x0"], dtype=np.float32)
            if x0.shape != (self.nx,):
                raise ValueError(
                    f"options['x0'] must have shape ({self.nx},), but got {x0.shape}"
                )
        elif "random_range" in options:
            ranges = options["random_range"]
            if not isinstance(ranges, (list, tuple)) or len(ranges) != self.nx:
                raise ValueError(
                    f"options['random_range'] must be a list/tuple of length {self.nx}"
                )
            x0 = np.array([self._np_random.uniform(low, high) for (low, high) in ranges], dtype=np.float32)
        else:
            noise = self._np_random.normal(
                loc=0.0,
                scale=self.reset_noise_std,
                size=self.nx,
            ).astype(np.float32)
            x0 = self.default_x0 + noise

        self._x = x0.copy()
        self._k = 0

        obs = self._x.astype(np.float32)
        info: dict = {}
        return obs, info

    def _state_error_np(self, x: np.ndarray, x_ref: np.ndarray) -> np.ndarray:
        theta, p = x
        r1, r2 = x_ref
        return np.array(
            [wrap_angle_np(theta - r1), p - r2],
            dtype=np.float32,
        )

    def step(self, action: np.ndarray):
        if self._x is None:
            raise RuntimeError("Call reset() before step().")

        u = np.array(action, dtype=np.float32).reshape(self.nu)
        u = np.clip(u, self.u_min, self.u_max)

        x_next = self._step_integrate(self._x, u)
        self._x = x_next
        self._k += 1

        e = self._state_error_np(self._x, self.x_ref)
        running_cost = float(e.T @ self.Q @ e + u.T @ self.R @ u)
        reward = -running_cost

        success = np.all(np.abs(e) <= self.success_tol)
        terminated = bool(success)
        truncated = self._k >= self.horizon_steps

        if terminated or truncated:
            if np.linalg.norm(e, ord=2) < self.success_eps:
                reward += self.terminal_bonus

        obs = self._x.astype(np.float32)
        info = {
            "cost": running_cost,
            "error": e,
            "success": success,
        }
        return obs, reward, terminated, truncated, info
