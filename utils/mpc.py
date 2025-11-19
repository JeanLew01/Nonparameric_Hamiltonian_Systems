import numpy as np
import casadi as cs
import do_mpc

__all__ = [
    "DoublePendulumParams",
    "build_double_pendulum_mpc",
    "rollout_mpc",
    "CartPoleParams",
    "build_cart_pole_mpc",
    "rollout_cart_pole_mpc",
]

class DoublePendulumParams:
    def __init__(self,
                 g=9.81,
                 m1=1.0, m2=1.0,
                 l1=1.0, l2=1.0,
                 lc1=0.5, lc2=0.5,
                 I1=0.2, I2=0.2,
                 u_min=np.array([-12.0, -12.0], dtype=float),
                 u_max=np.array([+12.0, +12.0], dtype=float),
                 Q=np.diag([60.0, 2.0, 60.0, 2.0]),
                 QT=np.diag([140.0, 5.0, 140.0, 5.0]),
                 R=np.diag([1e-2, 1e-2]),
                 x_ref=np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
                 dt=0.02,
                 n_horizon=120,
                 collocation_type='radau'):
        self.g = g
        self.m1, self.m2 = m1, m2
        self.l1, self.l2 = l1, l2
        self.lc1, self.lc2 = lc1, lc2
        self.I1, self.I2 = I1, I2
        self.u_min = np.array(u_min, dtype=float)
        self.u_max = np.array(u_max, dtype=float)
        self.Q  = np.array(Q,  dtype=float)
        self.QT = np.array(QT, dtype=float)
        self.R  = np.array(R,  dtype=float)
        self.x_ref = np.array(x_ref, dtype=float)
        self.dt = float(dt)
        self.n_horizon = int(n_horizon)
        self.collocation_type = collocation_type

def _angle_wrap(x):
    return cs.atan2(cs.sin(x), cs.cos(x))

def build_double_pendulum_mpc(cfg: DoublePendulumParams):
    """
    Build (model, mpc, simulator) for the double inverted pendulum with do-mpc.
    Matches your original equations and cost setup.
    """
    # ----- model -----
    model = do_mpc.model.Model('continuous')

    th1  = model.set_variable('_x', 'th1')
    th1d = model.set_variable('_x', 'th1d')
    th2  = model.set_variable('_x', 'th2')
    th2d = model.set_variable('_x', 'th2d')

    tau1 = model.set_variable('_u', 'tau1')
    tau2 = model.set_variable('_u', 'tau2')

    c2 = cs.cos(th2)
    s2 = cs.sin(th2)

    d11 = cfg.I1 + cfg.I2 + cfg.m1*cfg.lc1**2 + cfg.m2*(cfg.l1**2 + cfg.lc2**2 + 2*cfg.l1*cfg.lc2*c2)
    d12 = cfg.I2 + cfg.m2*(cfg.lc2**2 + cfg.l1*cfg.lc2*c2)
    d21 = d12
    d22 = cfg.I2 + cfg.m2*cfg.lc2**2
    D = cs.vertcat(
        cs.hcat([d11, d12]),
        cs.hcat([d21, d22])
    )

    h = cfg.m2*cfg.l1*cfg.lc2*s2
    c1 = -2.0*h*th1d*th2d - h*th2d**2
    c2_term = h*th1d**2
    Cqd = cs.vertcat(c1, c2_term)

    g1 = -(cfg.m1*cfg.lc1 + cfg.m2*cfg.l1)*cfg.g*cs.sin(th1) - cfg.m2*cfg.lc2*cfg.g*cs.sin(th1 + th2)
    g2 = -cfg.m2*cfg.lc2*cfg.g*cs.sin(th1 + th2)
    Gv = cs.vertcat(g1, g2)

    tau = cs.vertcat(tau1, tau2)
    rhs = tau - Cqd - Gv
    ddq = cs.solve(D, rhs)

    model.set_rhs('th1',  th1d)
    model.set_rhs('th1d', ddq[0])
    model.set_rhs('th2',  th2d)
    model.set_rhs('th2d', ddq[1])

    model.setup()

    # ----- mpc -----
    mpc = do_mpc.controller.MPC(model)
    setup_mpc = {
        'n_horizon': cfg.n_horizon,
        't_step': cfg.dt,
        'state_discretization': 'collocation',
        'collocation_type': cfg.collocation_type,
        'n_robust': 0,
        'store_full_solution': True,
    }
    mpc.set_param(**setup_mpc)

    # cost
    eth1  = _angle_wrap(th1  - cfg.x_ref[0])
    eth1d =            (th1d - cfg.x_ref[1])
    eth2  = _angle_wrap(th2  - cfg.x_ref[2])
    eth2d =            (th2d - cfg.x_ref[3])
    e_vec = cs.vertcat(eth1, eth1d, eth2, eth2d)
    u_vec = cs.vertcat(tau1, tau2)

    lterm = cs.mtimes([e_vec.T, cfg.Q,  e_vec]) + cs.mtimes([u_vec.T, cfg.R, u_vec])
    mterm = cs.mtimes([e_vec.T, cfg.QT, e_vec])

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(tau1=1e-4, tau2=1e-4)

    # input bounds
    mpc.bounds['lower','_u','tau1'] = cfg.u_min[0]
    mpc.bounds['upper','_u','tau1'] = cfg.u_max[0]
    mpc.bounds['lower','_u','tau2'] = cfg.u_min[1]
    mpc.bounds['upper','_u','tau2'] = cfg.u_max[1]

    mpc.setup()

    # ----- simulator -----
    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step=cfg.dt)
    simulator.setup()

    return model, mpc, simulator

def rollout_mpc(mpc, simulator, x0, sim_time=10.0, save_U=True, save_prefix=""):
    """
    Closed-loop rollout. Returns (T, X, U). Optionally saves U.npy/U.csv.
    """
    dt = simulator.t_step
    N  = int(sim_time / dt)

    X = np.zeros((N+1, 4), dtype=float)
    U = np.zeros((N,   2), dtype=float)
    T = np.zeros(N+1, dtype=float)

    X[0] = np.array(x0, dtype=float)
    mpc.x0 = np.array(x0, dtype=float)
    simulator.x0 = np.array(x0, dtype=float)
    mpc.set_initial_guess()

    x = np.array(x0, dtype=float)
    for k in range(N):
        u = mpc.make_step(x)
        x = simulator.make_step(u)

        X[k+1] = np.squeeze(x)
        U[k]   = np.squeeze(u)
        T[k+1] = T[k] + dt

    if save_U:
        base = (save_prefix + "_" if save_prefix else "")
        np.save(base + "U.npy", U)
        np.savetxt(base + "U.csv", U, delimiter=",")
    return T, X, U

class CartPoleParams:
    def __init__(self,
                 g=9.81,
                 mc=1.0,           # cart mass
                 mp=0.1,           # pole mass
                 l=0.5,            # CoM distance from pivot
                 Ip=0.0,           # pole inertia about its CoM (0 => point mass)
                 u_min=-20.0,
                 u_max=+20.0,
                 Q=np.diag([5.0, 1.0, 60.0, 2.0]),   # [x, xd, th, thd]
                 QT=np.diag([10.0, 2.0, 140.0, 5.0]),
                 R=np.array([[1e-2]]),
                 x_ref=np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
                 dt=0.02,
                 n_horizon=120,
                 collocation_type='radau'):
        self.g = float(g)
        self.mc = float(mc)
        self.mp = float(mp)
        self.l  = float(l)
        self.Ip = float(Ip)
        self.u_min = float(u_min)
        self.u_max = float(u_max)
        self.Q  = np.array(Q,  dtype=float)
        self.QT = np.array(QT, dtype=float)
        self.R  = np.array(R,  dtype=float).reshape(1, 1)
        self.x_ref = np.array(x_ref, dtype=float)
        self.dt = float(dt)
        self.n_horizon = int(n_horizon)
        self.collocation_type = collocation_type

# same angle wrapper you already have
def _angle_wrap(x):
    return cs.atan2(cs.sin(x), cs.cos(x))

# ---------------------------
# Build do-mpc model/controller/simulator
# ---------------------------
def build_cart_pole_mpc(cfg: CartPoleParams):
    """
    Build (model, mpc, simulator) for a cart-pole.
    States: x, xd, th, thd. Input: u (cart force).
    th=0 is upright.
    """
    # ----- model -----
    model = do_mpc.model.Model('continuous')

    x   = model.set_variable('_x', 'x')     # cart position
    xd  = model.set_variable('_x', 'xd')    # cart velocity
    th  = model.set_variable('_x', 'th')    # pole angle (0=upright)
    thd = model.set_variable('_x', 'thd')   # pole angular rate

    u = model.set_variable('_u', 'u')       # cart force

    c = cs.cos(th)
    s = cs.sin(th)

    # Inertia matrix D(q) with q=[x, th]
    D11 = cfg.mc + cfg.mp
    D12 = cfg.mp * cfg.l * c
    D21 = D12
    D22 = cfg.Ip + cfg.mp * cfg.l**2
    D = cs.vertcat(
        cs.hcat([D11, D12]),
        cs.hcat([D21, D22])
    )

    # C(q,qd)qd
    Cqd1 = - cfg.mp * cfg.l * s * thd**2
    Cqd2 = 0.0
    Cqd = cs.vertcat(Cqd1, Cqd2)

    # Gravity
    Gv = cs.vertcat(0.0, cfg.mp * cfg.g * cfg.l * s)

    # Input mapping Bu = [u; 0]
    Bu = cs.vertcat(u, 0.0)

    # Accelerations [xdd; thdd]
    rhs = Bu - Cqd - Gv
    qdd = cs.solve(D, rhs)

    # ---- correct RHS (set each state once) ----
    model.set_rhs('x',   xd)
    model.set_rhs('xd',  qdd[0])
    model.set_rhs('th',  thd)     # only once
    model.set_rhs('thd', qdd[1])

    model.setup()

    # ----- MPC -----
    mpc = do_mpc.controller.MPC(model)
    setup_mpc = {
        'n_horizon': cfg.n_horizon,
        't_step': cfg.dt,
        'state_discretization': 'collocation',
        'collocation_type': cfg.collocation_type,
        'n_robust': 0,
        'store_full_solution': True,
    }
    mpc.set_param(**setup_mpc)

    # Cost
    ex   = (x  - cfg.x_ref[0])
    exd  = (xd - cfg.x_ref[1])
    eth  = _angle_wrap(th  - cfg.x_ref[2])
    ethd = (thd - cfg.x_ref[3])

    e_vec = cs.vertcat(ex, exd, eth, ethd)
    u_vec = cs.vertcat(u)

    lterm = cs.mtimes([e_vec.T, cfg.Q,  e_vec]) + cs.mtimes([u_vec.T, cfg.R, u_vec])
    mterm = cs.mtimes([e_vec.T, cfg.QT, e_vec])

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u=1e-4)

    # Input bounds
    mpc.bounds['lower','_u','u'] = cfg.u_min
    mpc.bounds['upper','_u','u'] = cfg.u_max

    mpc.setup()

    # ----- simulator -----
    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step=cfg.dt)
    simulator.setup()

    return model, mpc, simulator


def rollout_cart_pole_mpc(mpc, simulator, x0, sim_time=10.0, save_U=True, save_prefix=""):
    """
    Closed-loop rollout for the cart-pole.
    States order: [x, xd, th, thd]
    Returns (T, X, U). Saves U.npy/CSV if requested.
    """
    dt = simulator.t_step
    N  = int(sim_time / dt)

    X = np.zeros((N+1, 4), dtype=float)
    U = np.zeros((N,   1), dtype=float)
    T = np.zeros(N+1, dtype=float)

    X[0] = np.array(x0, dtype=float)
    mpc.x0 = np.array(x0, dtype=float)
    simulator.x0 = np.array(x0, dtype=float)
    mpc.set_initial_guess()

    x = np.array(x0, dtype=float)
    for k in range(N):
        u = mpc.make_step(x)      # shape (1,1)
        x = simulator.make_step(u)

        X[k+1] = np.squeeze(x)
        U[k]   = np.squeeze(u)
        T[k+1] = T[k] + dt

    if save_U:
        base = (save_prefix + "_" if save_prefix else "")
        np.save(base + "U.npy", U)
        np.savetxt(base + "U.csv", U, delimiter=",")
    return T, X, U