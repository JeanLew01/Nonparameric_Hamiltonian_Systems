import casadi as cs
import numpy as np
import do_mpc

g = 9.81
m1 = 1.0
m2 = 1.0
l1 = 1.0
l2 = 1.0
lc1 = 0.5
lc2 = 0.5
I1 = 0.2
I2 = 0.2

u_min = np.array([-12.0, -12.0])
u_max = np.array([+12.0, +12.0])

x_ref = np.array([0.0, 0.0, 0.0, 0.0])  # desired state: upright position

Q = np.diag([60.0, 2.0, 60.0, 2.0])
QT = np.diag([140.0, 5.0, 140.0, 5.0])
R = np.diag([1e-2, 1e-2])

def angle_wrap(x):
    return cs.atan2(cs.sin(x), cs.cos(x))

model = do_mpc.model.Model('continuous')

th1 = model.set_variable('_x', 'th1')
th1d = model.set_variable('_x', 'th1d')
th2 = model.set_variable('_x', 'th2')
th2d = model.set_variable('_x', 'th2d')

tau1 = model.set_variable('_u', 'tau1')
tau2 = model.set_variable('_u', 'tau2')

c2 = cs.cos(th2)
s2 = cs.sin(th2)

d11 = I1 + I2 + m1*lc1**2 + m2*(l1**2 + lc2**2 + 2*l1*lc2*c2)
d12 = I2 + m2*(lc2**2 + l1*lc2*c2)
d21 = d12
d22 = I2 + m2*lc2**2
D = cs.vertcat(
    cs.hcat([d11, d12]),
    cs.hcat([d21, d22])
)

h = m2*l1*lc2*s2
c1 = -2.0*h*th1d*th2d - h*th2d**2
c2_term = h*th1d**2
Cqd = cs.vertcat(c1, c2_term)

g1 = (m1*lc1 + m2*l1)*g*cs.sin(th1) + m2*lc2*g*cs.sin(th1 + th2)
g2 = m2*lc2*g*cs.sin(th1 + th2)
Gv = cs.vertcat(g1, g2)

tau = cs.vertcat(tau1, tau2)
rhs = tau - Cqd - Gv
ddq = cs.solve(D, rhs)

# ODEs
model.set_rhs('th1', th1d)
model.set_rhs('th1d', ddq[0])
model.set_rhs('th2', th2d)
model.set_rhs('th2d', ddq[1])

model.setup()

mpc = do_mpc.controller.MPC(model)

dt = 0.02
setup_mpc = {
    'n_horizon': 120,
    't_step': dt,
    'state_discretization': 'collocation',
    'collocation_type': 'radau',
    'n_robust': 0,
    'store_full_solution': True,
}

mpc.set_param(**setup_mpc)

eth1 = angle_wrap(th1 - x_ref[0])
eth1d = (th1d - x_ref[1])
eth2 = angle_wrap(th2 - x_ref[2])
eth2d = (th2d - x_ref[3])

e_vec = cs.vertcat(eth1, eth1d, eth2, eth2d)
u_vec = cs.vertcat(tau1, tau2)

lterm = cs.mtimes([e_vec.T, QT, e_vec]) + cs.mtimes([u_vec.T, R, u_vec])
mterm = cs.mtimes([e_vec.T, Q, e_vec])

mpc.set_objective(mterm=mterm, lterm=lterm)
mpc.set_rterm(tau1=1e-4, tau2=1e-4)

mpc.bounds['lower','_u','tau1'] = u_min[0]
mpc.bounds['upper','_u','tau1'] = u_max[0]
mpc.bounds['lower','_u','tau2'] = u_min[1]
mpc.bounds['upper','_u','tau2'] = u_max[1]

mpc.setup()

simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step = dt)
simulator.setup()

x0 = np.array([np.pi, 0.0, np.pi, 0.0])
mpc.x0 = x0
simulator.x0 = x0
mpc.set_initial_guess()

X = np.zeros((N+1, 4))
U = np.zeros((N, 2))
T = np.zeros(N+1)
X[0,:] = x0

x = x0.copy()
for k in range(N):
    u = mpc.make_step(x)
    x = simulator.make_step(u)

    X[k+1,:] = x.squeeze()
    U[k,:]   = u.squeeze()
    T[k+1]   = T[k] + dt