import jax
import jax.numpy as jnp

def _linearize_discrete(dyn_fn, x0, u0, dt):
    f = lambda x, u: dyn_fn(x, u, dt)
    A = jax.jacfwd(f, argnums=0)(x0, u0)
    B = jax.jacfwd(f, argnums=1)(x0, u0)
    return A, B

def _dlqr(A, B, Q, R, max_iters=200, tol=1e-8):
    P = Q
    for _ in range(max_iters):
        BT_P = B.T @ P
        S = R + BT_P @ B
        K = jnp.linalg.solve(S, BT_P @ A)
        Acl = A - B @ K
        P_next = Acl.T @ P @ Acl + K.T @ R @ K + Q
        if jnp.max(jnp.abs(P_next - P)) < tol:
            P = P_next
            break
        P = P_next
    BT_P = B.T @ P
    S = R + BT_P @ B
    K = jnp.linalg.solve(S, BT_P @ A)
    return K, P

class SimpleLQRMPC:
    def __init__(self, dyn_fn, dt,
                 x_ref, Q, R,
                 u_ref=None,
                 u_min=None, u_max=None):
        self.dyn = dyn_fn
        self.dt = dt
        self.x_ref = jnp.array(x_ref, dtype=jnp.float32)
        self.u_ref = jnp.zeros((2,), dtype=jnp.float32) if u_ref is None else jnp.array(u_ref, dtype=jnp.float32)
        self.Q = Q
        self.R = R
        self.u_min = u_min
        self.u_max = u_max

        A, B = _linearize_discrete(self.dyn, self.x_ref, self.u_ref, self.dt)
        self.K, self.P = _dlqr(A, B, self.Q, self.R)

    def step(self, x):
        x = jnp.array(x, dtype=jnp.float32)
        u = self.u_ref - self.K @ (x - self.x_ref)
        if self.u_min is not None or self.u_max is not None:
            u_min = self.u_min if self.u_min is not None else -jnp.inf*jnp.ones_like(u)
            u_max = self.u_max if self.u_max is not None else +jnp.inf*jnp.ones_like(u)
            u = jnp.clip(u, u_min, u_max)
        return u