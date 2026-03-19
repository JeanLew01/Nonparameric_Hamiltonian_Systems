from __future__ import annotations

from dataclasses import dataclass, field

import casadi as cs
import do_mpc
import numpy as np

__all__ = [
    "DoublePendulumParams",
    "build_double_pendulum_mpc",
    "SinglePendulumParams",
    "build_single_pendulum_mpc",
    "SpringMassParams",
    "build_spring_mass_mpc",
    "rollout_mpc",
]


def _angle_wrap(x):
    return cs.atan2(cs.sin(x), cs.cos(x))


def _as_1d_array(value) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1)


def _as_column_array(value) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(-1, 1)


def _as_scalar(value) -> float:
    return float(np.asarray(value, dtype=float).reshape(-1)[0])


def _simulator_t_step(simulator) -> float:
    if hasattr(simulator, "t_step"):
        return _as_scalar(simulator.t_step)
    if hasattr(simulator, "settings") and hasattr(simulator.settings, "t_step"):
        return _as_scalar(simulator.settings.t_step)
    if hasattr(simulator, "_settings") and hasattr(simulator._settings, "t_step"):
        return _as_scalar(simulator._settings.t_step)
    raise AttributeError("Could not determine simulator time step.")


def _reset_history_if_available(obj) -> None:
    if hasattr(obj, "reset_history"):
        try:
            obj.reset_history()
        except TypeError:
            pass


@dataclass
class DoublePendulumParams:
    g: float = 9.81
    m1: float = 1.0
    m2: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    lc1: float = 0.5
    lc2: float = 0.5
    I1: float = 0.2
    I2: float = 0.2
    u_min: np.ndarray = field(
        default_factory=lambda: np.array([-12.0, -12.0], dtype=float)
    )
    u_max: np.ndarray = field(
        default_factory=lambda: np.array([12.0, 12.0], dtype=float)
    )
    Q: np.ndarray = field(
        default_factory=lambda: np.diag([60.0, 2.0, 60.0, 2.0]).astype(float)
    )
    QT: np.ndarray = field(
        default_factory=lambda: np.diag([140.0, 5.0, 140.0, 5.0]).astype(float)
    )
    R: np.ndarray = field(
        default_factory=lambda: np.diag([1e-2, 1e-2]).astype(float)
    )
    x_ref: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
    )
    dt: float = 0.02
    n_horizon: int = 120
    collocation_type: str = "radau"

    def __post_init__(self) -> None:
        self.u_min = _as_1d_array(self.u_min)
        self.u_max = _as_1d_array(self.u_max)
        self.Q = np.asarray(self.Q, dtype=float)
        self.QT = np.asarray(self.QT, dtype=float)
        self.R = np.asarray(self.R, dtype=float)
        self.x_ref = _as_1d_array(self.x_ref)
        self.dt = float(self.dt)
        self.n_horizon = int(self.n_horizon)


@dataclass
class SinglePendulumParams:
    g: float = 9.81
    m: float = 1.0
    l: float = 2.0
    I: float | None = None
    damping: float = 0.02
    u_min: np.ndarray = field(default_factory=lambda: np.array([-10.0], dtype=float))
    u_max: np.ndarray = field(default_factory=lambda: np.array([10.0], dtype=float))
    Q: np.ndarray = field(default_factory=lambda: np.diag([5.0, 1.0]).astype(float))
    QT: np.ndarray = field(default_factory=lambda: np.diag([10.0, 3.0]).astype(float))
    R: np.ndarray = field(default_factory=lambda: np.array([[1e-2]], dtype=float))
    x_ref: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0], dtype=float))
    dt: float = 0.02
    n_horizon: int = 120
    collocation_type: str = "radau"

    def __post_init__(self) -> None:
        if self.I is None:
            self.I = self.m * self.l**2
        self.u_min = _as_1d_array(self.u_min)
        self.u_max = _as_1d_array(self.u_max)
        self.Q = np.asarray(self.Q, dtype=float)
        self.QT = np.asarray(self.QT, dtype=float)
        self.R = np.asarray(self.R, dtype=float).reshape(1, 1)
        self.x_ref = _as_1d_array(self.x_ref)
        self.dt = float(self.dt)
        self.n_horizon = int(self.n_horizon)


@dataclass
class SpringMassParams:
    k_spring: float = 1.0
    m: float = 1.0
    damping: float = 0.05
    u_min: np.ndarray = field(default_factory=lambda: np.array([-50.0], dtype=float))
    u_max: np.ndarray = field(default_factory=lambda: np.array([50.0], dtype=float))
    Q: np.ndarray = field(default_factory=lambda: np.diag([10.0, 1.0]).astype(float))
    QT: np.ndarray = field(default_factory=lambda: np.diag([20.0, 2.0]).astype(float))
    R: np.ndarray = field(default_factory=lambda: np.array([[1e-2]], dtype=float))
    x_ref: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0], dtype=float))
    dt: float = 0.02
    n_horizon: int = 120
    collocation_type: str = "radau"

    def __post_init__(self) -> None:
        self.u_min = _as_1d_array(self.u_min)
        self.u_max = _as_1d_array(self.u_max)
        self.Q = np.asarray(self.Q, dtype=float)
        self.QT = np.asarray(self.QT, dtype=float)
        self.R = np.asarray(self.R, dtype=float).reshape(1, 1)
        self.x_ref = _as_1d_array(self.x_ref)
        self.dt = float(self.dt)
        self.n_horizon = int(self.n_horizon)


def build_double_pendulum_mpc(cfg: DoublePendulumParams):
    model = do_mpc.model.Model("continuous")

    th1 = model.set_variable("_x", "th1")
    th1d = model.set_variable("_x", "th1d")
    th2 = model.set_variable("_x", "th2")
    th2d = model.set_variable("_x", "th2d")

    tau1 = model.set_variable("_u", "tau1")
    tau2 = model.set_variable("_u", "tau2")

    c2 = cs.cos(th2)
    s2 = cs.sin(th2)

    d11 = (
        cfg.I1
        + cfg.I2
        + cfg.m1 * cfg.lc1**2
        + cfg.m2 * (cfg.l1**2 + cfg.lc2**2 + 2.0 * cfg.l1 * cfg.lc2 * c2)
    )
    d12 = cfg.I2 + cfg.m2 * (cfg.lc2**2 + cfg.l1 * cfg.lc2 * c2)
    d22 = cfg.I2 + cfg.m2 * cfg.lc2**2
    D = cs.vertcat(
        cs.hcat([d11, d12]),
        cs.hcat([d12, d22]),
    )

    h = cfg.m2 * cfg.l1 * cfg.lc2 * s2
    Cqd = cs.vertcat(
        -2.0 * h * th1d * th2d - h * th2d**2,
        h * th1d**2,
    )

    Gv = cs.vertcat(
        -(cfg.m1 * cfg.lc1 + cfg.m2 * cfg.l1) * cfg.g * cs.sin(th1)
        - cfg.m2 * cfg.lc2 * cfg.g * cs.sin(th1 + th2),
        -cfg.m2 * cfg.lc2 * cfg.g * cs.sin(th1 + th2),
    )

    ddq = cs.solve(D, cs.vertcat(tau1, tau2) - Cqd - Gv)

    model.set_rhs("th1", th1d)
    model.set_rhs("th1d", ddq[0])
    model.set_rhs("th2", th2d)
    model.set_rhs("th2d", ddq[1])
    model.setup()

    mpc = do_mpc.controller.MPC(model)
    mpc.set_param(
        n_horizon=cfg.n_horizon,
        t_step=cfg.dt,
        state_discretization="collocation",
        collocation_type=cfg.collocation_type,
        n_robust=0,
        store_full_solution=True,
    )

    e_vec = cs.vertcat(
        _angle_wrap(th1 - cfg.x_ref[0]),
        th1d - cfg.x_ref[1],
        _angle_wrap(th2 - cfg.x_ref[2]),
        th2d - cfg.x_ref[3],
    )
    u_vec = cs.vertcat(tau1, tau2)
    lterm = cs.mtimes([e_vec.T, cfg.Q, e_vec]) + cs.mtimes([u_vec.T, cfg.R, u_vec])
    mterm = cs.mtimes([e_vec.T, cfg.QT, e_vec])

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(tau1=1e-4, tau2=1e-4)
    mpc.bounds["lower", "_u", "tau1"] = cfg.u_min[0]
    mpc.bounds["upper", "_u", "tau1"] = cfg.u_max[0]
    mpc.bounds["lower", "_u", "tau2"] = cfg.u_min[1]
    mpc.bounds["upper", "_u", "tau2"] = cfg.u_max[1]
    if hasattr(mpc, "settings") and hasattr(mpc.settings, "supress_ipopt_output"):
        mpc.settings.supress_ipopt_output()
    mpc.setup()

    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step=cfg.dt)
    simulator.setup()

    return model, mpc, simulator


def build_single_pendulum_mpc(cfg: SinglePendulumParams):
    model = do_mpc.model.Model("continuous")

    theta = model.set_variable("_x", "theta")
    theta_dot = model.set_variable("_x", "theta_dot")
    tau = model.set_variable("_u", "tau")

    theta_dd = (
        tau - cfg.m * cfg.g * cfg.l * cs.sin(theta) - cfg.damping * theta_dot
    ) / cfg.I

    model.set_rhs("theta", theta_dot)
    model.set_rhs("theta_dot", theta_dd)
    model.setup()

    mpc = do_mpc.controller.MPC(model)
    mpc.set_param(
        n_horizon=cfg.n_horizon,
        t_step=cfg.dt,
        state_discretization="collocation",
        collocation_type=cfg.collocation_type,
        n_robust=0,
        store_full_solution=True,
    )

    e_vec = cs.vertcat(
        _angle_wrap(theta - cfg.x_ref[0]),
        theta_dot - cfg.x_ref[1],
    )
    u_vec = cs.vertcat(tau)
    lterm = cs.mtimes([e_vec.T, cfg.Q, e_vec]) + cs.mtimes([u_vec.T, cfg.R, u_vec])
    mterm = cs.mtimes([e_vec.T, cfg.QT, e_vec])

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(tau=1e-4)
    mpc.bounds["lower", "_u", "tau"] = cfg.u_min[0]
    mpc.bounds["upper", "_u", "tau"] = cfg.u_max[0]
    if hasattr(mpc, "settings") and hasattr(mpc.settings, "supress_ipopt_output"):
        mpc.settings.supress_ipopt_output()
    mpc.setup()

    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step=cfg.dt)
    simulator.setup()

    return model, mpc, simulator


def build_spring_mass_mpc(cfg: SpringMassParams):
    model = do_mpc.model.Model("continuous")

    x_pos = model.set_variable("_x", "x_pos")
    x_dot = model.set_variable("_x", "x_dot")
    force = model.set_variable("_u", "force")

    x_ddot = (force - cfg.k_spring * x_pos - cfg.damping * x_dot) / cfg.m

    model.set_rhs("x_pos", x_dot)
    model.set_rhs("x_dot", x_ddot)
    model.setup()

    mpc = do_mpc.controller.MPC(model)
    mpc.set_param(
        n_horizon=cfg.n_horizon,
        t_step=cfg.dt,
        state_discretization="collocation",
        collocation_type=cfg.collocation_type,
        n_robust=0,
        store_full_solution=True,
    )

    e_vec = cs.vertcat(
        x_pos - cfg.x_ref[0],
        x_dot - cfg.x_ref[1],
    )
    u_vec = cs.vertcat(force)
    lterm = cs.mtimes([e_vec.T, cfg.Q, e_vec]) + cs.mtimes([u_vec.T, cfg.R, u_vec])
    mterm = cs.mtimes([e_vec.T, cfg.QT, e_vec])

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(force=1e-4)
    mpc.bounds["lower", "_u", "force"] = cfg.u_min[0]
    mpc.bounds["upper", "_u", "force"] = cfg.u_max[0]
    if hasattr(mpc, "settings") and hasattr(mpc.settings, "supress_ipopt_output"):
        mpc.settings.supress_ipopt_output()
    mpc.setup()

    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step=cfg.dt)
    simulator.setup()

    return model, mpc, simulator


def rollout_mpc(mpc, simulator, x0, sim_time=10.0, save_U=True, save_prefix=""):
    dt = _simulator_t_step(simulator)
    num_steps = int(sim_time / dt)

    x = _as_1d_array(x0)
    mpc.x0 = x.copy()
    simulator.x0 = x.copy()
    _reset_history_if_available(mpc)
    _reset_history_if_available(simulator)
    mpc.set_initial_guess()

    X = [x.copy()]
    U = []
    T = [0.0]

    for _ in range(num_steps):
        u = _as_1d_array(mpc.make_step(x))
        x = _as_1d_array(simulator.make_step(_as_column_array(u)))

        U.append(u.copy())
        X.append(x.copy())
        T.append(T[-1] + dt)

    X = np.asarray(X, dtype=float)
    U = np.asarray(U, dtype=float)
    T = np.asarray(T, dtype=float)

    if save_U:
        base = f"{save_prefix}_" if save_prefix else ""
        np.save(base + "U.npy", U)
        np.savetxt(base + "U.csv", U, delimiter=",")

    return T, X, U
