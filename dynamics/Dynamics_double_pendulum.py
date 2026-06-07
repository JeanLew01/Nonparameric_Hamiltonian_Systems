import jax
import jax.numpy as jnp

import gymnasium as gym
from gymnasium import spaces

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
    """Angle errors wrapped, momentum errors linear."""
    th1, th2, p1, p2 = x
    r1, r2, rp1, rp2 = x_ref
    return jnp.array(
        [
            angle_wrap(th1 - r1),
            angle_wrap(th2 - r2),
            p1 - rp1,
            p2 - rp2,
        ],
        dtype=jnp.float32,
    )

def mass_matrix(q):
    """Absolute-angle mass matrix in the manuscript's Hamiltonian model."""
    th1, th2 = q
    delta = th1 - th2
    coupling = m2 * l1 * l2 * jnp.cos(delta)
    return jnp.array(
        [
            [(m1 + m2) * l1**2, coupling],
            [coupling, m2 * l2**2],
        ],
        dtype=jnp.float32,
    )

def f_continuous(x, u):
    """State derivative in canonical coordinates [theta1, theta2, p1, p2]."""
    th1, th2, p1, p2 = x
    tau1, tau2 = u

    q = jnp.array([th1, th2], dtype=jnp.float32)
    p = jnp.array([p1, p2], dtype=jnp.float32)
    M = mass_matrix(q)
    q_dot = jnp.linalg.solve(M, p)
    q1_dot, q2_dot = q_dot

    delta = th1 - th2
    s_delta = jnp.sin(delta)
    coupling_grad = m2 * l1 * l2 * s_delta

    dH_dth1 = coupling_grad * q1_dot * q2_dot + (m1 + m2) * g * l1 * jnp.sin(th1)
    dH_dth2 = -coupling_grad * q1_dot * q2_dot + m2 * g * l2 * jnp.sin(th2)

    p_dot = jnp.array([tau1 - dH_dth1, tau2 - dH_dth2], dtype=jnp.float32)
    return jnp.array([q1_dot, q2_dot, p_dot[0], p_dot[1]], dtype=jnp.float32)

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

def wrap_angle_np(theta: float) -> float:
    return ((theta + np.pi) % (2 * np.pi)) - np.pi

class DoublePendulumEnv(gym.Env):
    """
    Double pendulum environment in canonical Hamiltonian coordinates.

    - Observation: [th1, th2, p1, p2]
    - Action: [tau1, tau2] (Nm), clipped to [u_min, u_max]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        dt=0.02,
        episode_seconds=8.0,
        Q=np.diag([60.0, 60.0, 2.0, 2.0]),
        R=np.diag([1e-2, 1e-2]),
        QT=np.diag([140.0, 140.0, 5.0, 5.0]),
        x_ref=np.array([np.pi, np.pi, 0.0, 0.0]),
        u_min=np.array([-12.0, -12.0]),
        u_max=np.array([+12.0, +12.0]),
        success_tol=np.array([0.1, 0.1, 0.01, 0.01]),
        default_x0=np.array([0.0, 0.0, 0.0, 0.0]),
        reset_noise_std=np.array([0.02, 0.02, 0.02, 0.02]),
        seed: int | None = None,
        # ===== 新增：用于 reward shaping 的 “成功半径” 和 bonus =====
        success_eps: float = 0.2,
        terminal_bonus: float = 50.0,
        # ==========================================================
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

        high_x = np.array([np.pi, np.pi, np.inf, np.inf], dtype=np.float32)
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

        # Gymnasium RNG
        self._np_random = None
        self.seed(seed)

        # ===== 新增：保存 success 半径和 bonus =====
        self.success_eps = float(success_eps)
        self.terminal_bonus = float(terminal_bonus)
        # ==========================================

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

        th1 = cs.SX.sym("th1")
        th2 = cs.SX.sym("th2")
        p1 = cs.SX.sym("p1")
        p2 = cs.SX.sym("p2")
        tau1 = cs.SX.sym("tau1")
        tau2 = cs.SX.sym("tau2")

        x = cs.vertcat(th1, th2, p1, p2)
        u = cs.vertcat(tau1, tau2)

        delta = th1 - th2
        coupling = m2_local * l1_local * l2_local * cs.cos(delta)
        d11 = (m1_local + m2_local) * l1_local**2
        d12 = coupling
        d21 = coupling
        d22 = m2_local * l2_local**2
        D = cs.vertcat(
            cs.hcat([d11, d12]),
            cs.hcat([d21, d22]),
        )

        p = cs.vertcat(p1, p2)
        D_inv = cs.inv(D)
        V = -(m1_local + m2_local) * g_local * l1_local * cs.cos(th1) - m2_local * g_local * l2_local * cs.cos(th2)
        H = 0.5 * cs.mtimes([p.T, D_inv, p]) + V
        gradH = cs.gradient(H, x)
        J = cs.vertcat(
            cs.hcat([cs.SX.zeros(2, 2), cs.SX.eye(2)]),
            cs.hcat([-cs.SX.eye(2), cs.SX.zeros(2, 2)]),
        )
        Gmat = cs.vertcat(cs.SX.zeros(2, 2), cs.SX.eye(2))
        xdot = J @ gradH + Gmat @ u

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
        th1, th2, p1, p2 = x
        r1, r2, rp1, rp2 = x_ref
        return np.array(
            [
                wrap_angle_np(th1 - r1),
                wrap_angle_np(th2 - r2),
                p1 - rp1,
                p2 - rp2,
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

        # ==== 新增：成功终止或截断且进入 success_eps 球 -> 给 bonus ====
        if terminated or truncated:
            if np.linalg.norm(e, ord=2) < self.success_eps:
                reward += self.terminal_bonus
        # ===========================================================

        obs = self._x.astype(np.float32)
        info = {
            "cost": running_cost,
            "error": e,
            "success": success,
        }

        return obs, reward, terminated, truncated, info
