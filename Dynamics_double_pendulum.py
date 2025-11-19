import jax
import jax.numpy as jnp

import gymnasium as gym
import gymnasium.spaces as spaces

import numpy as np
import casadi as cs

g  = 9.81
m1 = 1.0
m2 = 1.0
l1 = 1.0
l2 = 1.0
lc1 = 0.5
lc2 = 0.5
I1 = 0.2
I2 = 0.2

u_min = jnp.array([-12.0, -12.0], dtype=jnp.float32)
u_max = jnp.array([+12.0, +12.0], dtype=jnp.float32)

def angle_wrap(x):
    """Wrap angle to (-pi, pi]."""
    return (x + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

def state_error(x, x_ref):
    """Angle errors wrapped, rates linear."""
    th1, th1d, th2, th2d = x
    r1, r1d, r2, r2d = x_ref
    return jnp.array(
        [
            angle_wrap(th1 - r1),
            th1d - r1d,
            angle_wrap(th2 - r2),
            th2d - r2d,
        ],
        dtype=jnp.float32,
    )

def two_link_ddq(x, u):
    th1, th1d, th2, th2d = x
    tau1, tau2 = u

    c2 = jnp.cos(th2)
    s2 = jnp.sin(th2)

    d11 = I1 + I2 + m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * c2)
    d12 = I2 + m2 * (lc2**2 + l1 * lc2 * c2)
    d21 = d12
    d22 = I2 + m2 * lc2**2
    D = jnp.array([[d11, d12],
                   [d21, d22]], dtype=jnp.float32)

    h = m2 * l1 * lc2 * s2
    c1 = -2.0 * h * th1d * th2d - h * th2d**2
    c2_term = h * th1d**2
    Cqd = jnp.array([c1, c2_term], dtype=jnp.float32)

    g1 = -((m1 * lc1 + m2 * l1) * g * jnp.sin(th1) + m2 * lc2 * g * jnp.sin(th1 + th2))
    g2 = -(m2 * lc2 * g * jnp.sin(th1 + th2))
    Gv = jnp.array([g1, g2], dtype=jnp.float32)

    tau = jnp.array([tau1, tau2], dtype=jnp.float32)
    rhs = tau - Cqd - Gv
    ddq = jnp.linalg.solve(D, rhs)
    return ddq[0], ddq[1]

def f_continuous(x, u):
    """State derivative: [th1d, th1dd, th2d, th2dd]."""
    th1, th1d, th2, th2d = x
    th1dd, th2dd = two_link_ddq(x, u)
    return jnp.array([th1d, th1dd, th2d, th2dd], dtype=jnp.float32)

def rk4_step(x, u, h):
    """RK4 integrator."""
    k1 = f_continuous(x, u)
    k2 = f_continuous(x + 0.5 * h * k1, u)
    k3 = f_continuous(x + 0.5 * h * k2, u)
    k4 = f_continuous(x + h * k3, u)
    return x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

@jax.jit
def jax_dynamics(x, u, dt=0.01):
    """JIT-ed dynamics step with torque clipping."""
    u = jnp.clip(u, u_min, u_max)
    return rk4_step(x, u, dt)

def wrap_angle_np(a: float) -> float:
    """Wrap angle to (-pi, pi] using NumPy."""
    return float(np.arctan2(np.sin(a), np.cos(a)))

class DoublePendulumEnv(gym.Env):
    """
    Double-pendulum swing-up / stabilization environment, Gymnasium-style.

    - Observation: [th1, th1d, th2, th2d] (angles in radians, rates in rad/s)
    - Action: [tau1, tau2] (Nm), clipped to [u_min, u_max]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dt=0.02,
        episode_seconds=8.0,
        Q=np.diag([60.0, 2.0, 60.0, 2.0]),
        R=np.diag([1e-2, 1e-2]),
        QT=np.diag([140.0, 5.0, 140.0, 5.0]),
        x_ref=np.array([0.0, 0.0, 0.0, 0.0]),
        u_min=np.array([-12.0, -12.0]),
        u_max=np.array([+12.0, +12.0]),
        success_tol=np.array([0.1, 0.01, 0.1, 0.01]),
        default_x0=np.array([np.pi, 0.0, 0.0, 0.0]),
        reset_noise_std=np.array([0.02, 0.02, 0.02, 0.02]),
        seed: int | None = None,
    ):
        super().__init__()

        self.dt = float(dt)
        self.T = float(episode_seconds)
        self.horizon_steps = int(np.round(self.T / self.dt))
    
        # cost matrices (store as float32 for RL friendliness)
        self.Q = np.array(Q, dtype=np.float32)
        self.R = np.array(R, dtype=np.float32)
        self.QT = np.array(QT, dtype=np.float32)

        self.x_ref = np.array(x_ref, dtype=np.float32)

        # torque limits (NumPy side)
        self.u_min = np.array(u_min, dtype=np.float32)
        self.u_max = np.array(u_max, dtype=np.float32)

        self.success_tol = np.array(success_tol, dtype=np.float32)

        self.default_x0 = np.array(default_x0, dtype=np.float32)
        self.reset_noise_std = np.array(reset_noise_std, dtype=np.float32)

        self.nx = 4
        self.nu = 2

        high_x = np.array([np.pi, np.inf, np.pi, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-high_x, high=high_x, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=self.u_min, high=self.u_max, shape=(2,), dtype=np.float32
        )

        self._build_casadi_dynamics()
        self._build_casadi_integrator()

        # Internal state
        self._x: np.ndarray | None = None
        self._k: int = 0

        # Gymnasium RNG (np_random)
        self._np_random = None
        self.seed(seed)

    def _build_casadi_dynamics(self):
        g_local = 9.81
        m1_local = 1.0
        m2_local = 1.0
        l1_local = 1.0
        l2_local = 1.0
        lc1_local = 0.5
        lc2_local = 0.5
        I1_local = 0.2
        I2_local = 0.2

        th1  = cs.SX.sym("th1")
        th1d = cs.SX.sym("th1d")
        th2  = cs.SX.sym("th2")
        th2d = cs.SX.sym("th2d")
        tau1 = cs.SX.sym("tau1")
        tau2 = cs.SX.sym("tau2")

        x = cs.vertcat(th1, th1d, th2, th2d)
        u = cs.vertcat(tau1, tau2)

        c2 = cs.cos(th2)
        s2 = cs.sin(th2)

        d11 = (
            I1_local
            + I2_local
            + m1_local * lc1_local**2
            + m2_local * (l1_local**2 + lc2_local**2 + 2 * l1_local * lc2_local * c2)
        )
        d12 = I2_local + m2_local * (lc2_local**2 + l1_local * lc2_local * c2)
        d21 = d12
        d22 = I2_local + m2_local * lc2_local**2
        D = cs.vertcat(
            cs.hcat([d11, d12]),
            cs.hcat([d21, d22]),
        )

        h = m2_local * l1_local * lc2_local * s2
        c1 = -2.0 * h * th1d * th2d - h * th2d**2
        c2_term = h * th1d**2
        Cqd = cs.vertcat(c1, c2_term)

        g1 = -(
            (m1_local * lc1_local + m2_local * l1_local) * g_local * cs.sin(th1)
            + m2_local * lc2_local * g_local * cs.sin(th1 + th2)
        )
        g2 = -(m2_local * lc2_local * g_local * cs.sin(th1 + th2))
        Gv = cs.vertcat(g1, g2)

        rhs = u - Cqd - Gv
        ddq = cs.solve(D, rhs)  # [th1dd, th2dd]

        xdot = cs.vertcat(th1d, ddq[0], th2d, ddq[1])

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

    # ------------------------------------------------------------------
    # Seeding / reset / step
    # ------------------------------------------------------------------
    def seed(self, seed: int | None = None):
        """Set the environment seed."""
        # Gymnasium uses np_random; keep a simple wrapper here.
        self._np_random = np.random.default_rng(seed)
        return [seed]

    def reset(self, *, seed: int | None = None, options=None):
        """
        Gymnasium reset API: returns (obs, info).
        options keys:
          - "x0": explicit initial state (shape (4,))
          - "random_range": list/tuple of 4 (low, high) pairs
        Otherwise: default_x0 + Normal(0, reset_noise_std).
        """
        if seed is not None:
            self.seed(seed)
        if options is None:
            options = {}

        # Choose initial state x0
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
            x0 = np.array(
                [self._np_random.uniform(low, high) for (low, high) in ranges],
                dtype=np.float32,
            )
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
        """NumPy version of state error with angle wrap."""
        th1, th1d, th2, th2d = x
        r1, r1d, r2, r2d = x_ref
        return np.array(
            [
                wrap_angle_np(th1 - r1),
                th1d - r1d,
                wrap_angle_np(th2 - r2),
                th2d - r2d,
            ],
            dtype=np.float32,
        )

    def step(self, action: np.ndarray):
        """
        Gymnasium step API:
          obs, reward, terminated, truncated, info
        """
        if self._x is None:
            raise RuntimeError("Call reset() before step().")

        # Ensure action is np.float32 and clipped
        u = np.array(action, dtype=np.float32).reshape(self.nu)
        u = np.clip(u, self.u_min, self.u_max)

        # Integrate dynamics
        x_next = self._step_integrate(self._x, u)
        self._x = x_next
        self._k += 1

        # Quadratic running cost
        e = self._state_error_np(self._x, self.x_ref)
        running_cost = float(e.T @ self.Q @ e + u.T @ self.R @ u)
        reward = -running_cost  # reward = -cost

        # Termination condition: near target (within success_tol)
        success = np.all(np.abs(e) <= self.success_tol)
        terminated = bool(success)

        # Truncation condition: horizon reached
        truncated = self._k >= self.horizon_steps

        obs = self._x.astype(np.float32)
        info = {
            "cost": running_cost,
            "error": e,
            "success": success,
        }

        return obs, reward, terminated, truncated, info