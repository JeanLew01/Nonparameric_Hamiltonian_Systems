# pip install --upgrade jax jaxlib  # if needed

from dataclasses import dataclass
from typing import Tuple
import jax
import jax.numpy as jnp

@dataclass
class Params:
    m1: float = 1.0
    m2: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    g:  float = 9.81

# ---- Dynamics (θ1, θ2, ω1, ω2) ----
@jax.jit
def double_pendulum_dynamics(state: jnp.ndarray, p: Params) -> jnp.ndarray:
    """
    state = [theta1, theta2, omega1, omega2]
    returns dstate/dt in the same order.
    """
    θ1, θ2, ω1, ω2 = state
    m1, m2, l1, l2, g = p.m1, p.m2, p.l1, p.l2, p.g

    Δ = θ2 - θ1
    sinΔ = jnp.sin(Δ)
    cosΔ = jnp.cos(Δ)

    denom = (2*m1 + m2 - m2 * jnp.cos(2*Δ))

    # θ1'' (omega1dot)
    num1 = (-g * (2*m1 + m2) * jnp.sin(θ1)
            - m2 * g * jnp.sin(θ1 - 2*θ2)
            - 2 * sinΔ * m2 * (ω2**2 * l2 + ω1**2 * l1 * cosΔ))
    ω1dot = num1 / (l1 * denom)

    # θ2'' (omega2dot)
    num2 = (2 * sinΔ * (ω1**2 * l1 * (m1 + m2)
                        + g * (m1 + m2) * jnp.cos(θ1)
                        + ω2**2 * l2 * m2 * cosΔ))
    ω2dot = num2 / (l2 * denom)

    return jnp.array([ω1, ω2, ω1dot, ω2dot])

# ---- Fixed-step RK4 integrator (JAX-friendly) ----
@jax.jit
def rk4_step(f, state: jnp.ndarray, dt: float, p: Params) -> jnp.ndarray:
    k1 = f(state, p)
    k2 = f(state + 0.5*dt*k1, p)
    k3 = f(state + 0.5*dt*k2, p)
    k4 = f(state + dt*k3, p)
    return state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

@jax.jit
def simulate(f, y0: jnp.ndarray, p: Params, dt: float, steps: int) -> jnp.ndarray:
    """
    Returns trajectory with shape [steps+1, 4].
    """
    def body(y, _):
        y_next = rk4_step(f, y, dt, p)
        return y_next, y_next
    yT, ys = jax.lax.scan(body, y0, None, length=steps)
    return jnp.vstack([y0, ys])

# ---- Energy (useful for sanity checks) ----
@jax.jit
def energies(state: jnp.ndarray, p: Params) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    θ1, θ2, ω1, ω2 = state
    m1, m2, l1, l2, g = p.m1, p.m2, p.l1, p.l2, p.g

    # Cartesian positions (pivot at origin, down is +y convention can be chosen;
    # here we use y up positive -> potential = m g y, with y measured upward)
    x1 =  l1 * jnp.sin(θ1)
    y1 = -l1 * jnp.cos(θ1)
    x2 =  x1 + l2 * jnp.sin(θ2)
    y2 =  y1 - l2 * jnp.cos(θ2)

    # Velocities
    vx1 =  l1 * ω1 * jnp.cos(θ1)
    vy1 =  l1 * ω1 * jnp.sin(θ1)
    vx2 =  vx1 + l2 * ω2 * jnp.cos(θ2)
    vy2 =  vy1 + l2 * ω2 * jnp.sin(θ2)

    T = 0.5*m1*(vx1**2 + vy1**2) + 0.5*m2*(vx2**2 + vy2**2)
    V = m1*g*y1 + m2*g*y2
    E = T + V
    return T, V, E

# ---- Example usage ----
if __name__ == "__main__":
    p = Params(m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81)
    # initial state: θ1, θ2, ω1, ω2
    y0 = jnp.array([1.2, -0.5, 0.0, 0.0])
    dt = 0.005
    steps = 4000

    traj = simulate(double_pendulum_dynamics, y0, p, dt, steps)
    # Example: compute energies over the trajectory
    T, V, E = jax.vmap(lambda s: energies(s, p))(traj)
