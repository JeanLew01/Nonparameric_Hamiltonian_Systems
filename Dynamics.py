import jax
import jax.numpy as jnp

g  = 9.81
m1 = 1.0
m2 = 1.0
l1 = 1.0
l2 = 1.0
lc1 = 0.5
lc2 = 0.5
I1 = 0.2
I2 = 0.2

# torque limits
u_min = jnp.array([-12.0, -12.0])
u_max = jnp.array([+12.0, +12.0])

def angle_wrap(x):
    """Wrap angle to (-pi, pi]."""
    return (x + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

def state_error(x, x_ref):
    """Angle errors wrapped, rates linear."""
    th1, th1d, th2, th2d = x
    r1, r1d, r2, r2d = x_ref
    return jnp.array([
        angle_wrap(th1 - r1),
        th1d - r1d,
        angle_wrap(th2 - r2),
        th2d - r2d
    ], dtype=jnp.float32)

def two_link_ddq(x, u):
    """Compute joint accelerations (Spong form)."""
    th1, th1d, th2, th2d = x
    tau1, tau2 = u

    c2 = jnp.cos(th2)
    s2 = jnp.sin(th2)

    d11 = I1 + I2 + m1*lc1**2 + m2*(l1**2 + lc2**2 + 2*l1*lc2*c2)
    d12 = I2 + m2*(lc2**2 + l1*lc2*c2)
    d21 = d12
    d22 = I2 + m2*lc2**2
    D = jnp.array([[d11, d12],
                   [d21, d22]], dtype=jnp.float32)

    h = m2*l1*lc2*s2
    c1 = -2.0*h*th1d*th2d - h*th2d**2
    c2_term = h*th1d**2
    Cqd = jnp.array([c1, c2_term], dtype=jnp.float32)

    # Gravity
    g1 = (m1*lc1 + m2*l1)*g*jnp.sin(th1) + m2*lc2*g*jnp.sin(th1 + th2)
    g2 = m2*lc2*g*jnp.sin(th1 + th2)
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
    """RK4 integrator for stability."""
    k1 = f_continuous(x, u)
    k2 = f_continuous(x + 0.5*h*k1, u)
    k3 = f_continuous(x + 0.5*h*k2, u)
    k4 = f_continuous(x + h*k3, u)
    return x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

@jax.jit
def jax_dynamics(x, u, dt=0.01):
    """JIT-ed dynamics step with torque clipping."""
    u = jnp.clip(u, u_min, u_max)
    return rk4_step(x, u, dt)