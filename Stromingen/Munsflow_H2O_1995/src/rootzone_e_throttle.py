# %%
import numpy as np
from scipy.special import lambertw
import matplotlib.pyplot as plt
from itertools import cycle
from scipy.integrate import solve_ivp

# %% Exact solution for lambda = 0.5, using the Lambert function W(..)

def x_explicit_lambda_05(t, p, r, x0, t0=0.0):
    """
    Explicit solution for dx/dt = p - r * sqrt(x).
    Accepts scalar or numpy array t.
    Handles p > 0 via Lambert W, and p == 0 via separable formula.
    """
    t = np.asarray(t, dtype=float)
    
    # trivial constant solution if r == 0 -> x = x0 + p*(t-t0)  (but here r>0 usually)
    if np.isclose(p, 0.0):
        # separable solution: sqrt{x}(t) = sqrt{x0} - (r/2)*(t-t0)
        s = np.sqrt(x0) - 0.5 * r * (t - t0)
        s = np.maximum(s, 0.0)   # if hit zero, floor at zero
        return s**2

    # general p > 0 case (Lambert W formula)
    sqrt_x0 = np.sqrt(x0)
    y0 = p - r * sqrt_x0
    
    # Build A(t) = (y0/p) * exp(-y0/p) * exp(- (r^2/(2p))*(t-t0))
    A = (y0 / p) * np.exp(-y0 / p) * np.exp(-(r**2 / (2.0 * p)) * (t - t0))
    arg = -A
    
    # lambertw returns complex arrays in general; for our regime principal branch real part suffices
    W = lambertw(arg, k=0)
    W = np.real_if_close(W, tol=1000)   # convert small imag parts to real
    z = -W.real
    u = (p * (1.0 - z)) / r
    x = (u ** 2)
    
    # numerical safety: ensure non-negative
    x = np.maximum(x, 0.0)
    return x

# %% Numerical solution for any lambda.

def implicit_euler_nr_step(xn, dt, p, r, lam, tol=1e-12, maxit=10, xmin=1e-16):
    """Return x given dx/dt = p - e * x ** lambda
    with 0 < lambda < 1 determining the throttle curve.
    
    Implicit Euler step solved by Newton (scalar).
    Returns x_{n+1} >= 0.
    """
    # Initial guess: explicit Euler predictor (safer than xn)
    xk = max(xn + dt*(p - r * xn**lam), xmin)
    for i in range(maxit):
        # protect power on tiny x
        xk_safe = max(xk, xmin)
        G = xk - xn - dt*(p - r * (xk_safe**lam))
        if abs(G) < tol:
            break
        
        Gp = 1.0 + dt * r * lam * (xk_safe**(lam - 1.0))
        # Newton step
        dx = - G / Gp
        x_new = xk + dx
        
        # Damping if sign problem / negativity
        if x_new < 0.0:
            # backtrack: reduce step until non-negative (and small)
            alpha = 0.5
            while alpha > 1e-6:
                x_try = xk + alpha * dx
                if x_try >= 0.0:
                    x_new = x_try
                    break
                alpha *= 0.5
            else:
                x_new = 0.0
        xk = x_new
        if abs(dx) < tol * max(1.0, abs(xk)):
            break
    return max(xk, 0.0)


def implicit_euler_step_lambda_half(x0, dt, p, r):
    """Return x given p, r, x0 and dt for lambda=0.5
    
    Very cheap and fast iterative numerical solution method.
    
    Support arrays.
    
    $$u_n = \sqrt{x_0}$$
    $$A = u_n^2 + p dt $$
    $$disk = (r dt)^2 + 4 A $$
    $$u_{n+1} =\frac 1 2(-r dt + \sqrt{disk})
    
    $$u_{n+1} =\frac{1}{2}(-r dt + \sqrt{(r dt)^2 + 4 (u_n^2 + p dt)})$$
    
    
    Parameters
    ----------
    x0: float
        initial guess or value for S/Smax
    dt: float
        time step
    p: float
        p = P /Smax with Smax is storage capacity of te root zone
    r: float
        r = E /Smax
    
    Returns
    -------
        x: float
        S(dt)/Smax
    """
    # supports arrays
    u_n = np.sqrt(x0)    
    D = (r * dt) ** 2 + 4.0 * (u_n ** 2 + p * dt)
    u_np1 = (-r * dt + np.sqrt(D)) / 2.0
    return np.maximum(u_np1 ** 2, 0.0)


# %%
import numpy as np

def implicit_step_general(x0, dt, p, r, lam, tol=1e-12, maxit=10, xmin=1e-16):
    """
    Implicit Euler step for dx/dt = p - r * x^lam over one timestep dt.
    Uses closed-form update for lam == 0.5, otherwise uses Newton (vectorized).
    
    Parameters
    ----------
    x0 : scalar or array-like
        current x (>=0). Can be vector of cell values.
    dt : scalar or array-like
        timestep (broadcastable to x0).
    p, r : scalar or array-like
        parameters (broadcastable).
    lam : scalar
        exponent (positive). Use lam == 0.5 for closed form.
    tol : float
        Newton tolerance (absolute residual).
    maxit : int
        maximum Newton iterations.
    xmin : float
        small floor for arguments to power expressions to avoid overflow when lam < 1.
    
    Returns
    -------
    x_new : ndarray or scalar
        updated x after one implicit Euler step (same shape as broadcasted inputs).
    """
    # Make arrays for broadcasting
    x0 = np.asarray(x0, dtype=float)
    dt = np.asarray(dt, dtype=float)
    p  = np.asarray(p, dtype=float)
    r  = np.asarray(r, dtype=float)
    
    # Broadcast shapes
    try:
        # This forces numpy to broadcast inputs to a common shape, or raise ValueError
        shape = np.broadcast(x0, dt, p, r).shape
    except ValueError:
        raise ValueError("x0, dt, p, r must be broadcastable to a common shape")
    
    # Work on arrays
    x0_b = np.broadcast_to(x0, shape).astype(float)
    dt_b = np.broadcast_to(dt, shape).astype(float)
    p_b  = np.broadcast_to(p,  shape).astype(float)
    r_b  = np.broadcast_to(r,  shape).astype(float)
    
    if lam == 0.5:
        # exact closed-form implicit Euler for lambda = 1/2
        u_n = np.sqrt(np.maximum(x0_b, 0.0))
        A = u_n*u_n + dt_b * p_b
        D = (dt_b * r_b)**2 + 4.0 * A
        u_np1 = (-dt_b * r_b + np.sqrt(D)) * 0.5
        x_np1 = u_np1**2
        return x_np1 if x_np1.shape != () else float(x_np1)
    
    # General lam: vectorized Newton on G(x) = x - x0 - dt*(p - r*x^lam) = 0
    # Initial predictor: explicit Euler (clipped to >= xmin)
    xk = np.maximum(x0_b + dt_b * (p_b - r_b * np.maximum(x0_b, xmin)**lam), xmin)
    
    # Newton iterations (vectorized)
    for _ in range(maxit):
        xk_safe = np.maximum(xk, xmin)            # avoid x**(lam-1) blowing up when lam<1
        G = xk - x0_b - dt_b * (p_b - r_b * xk_safe**lam)
        # convergence mask
        mask_conv = np.abs(G) < tol
        if mask_conv.all():
            break
        # derivative G' = 1 + dt * r * lam * x^{lam-1}
        Gp = 1.0 + dt_b * r_b * lam * (xk_safe**(lam - 1.0))
        dx = - G / Gp
        x_new = xk + dx
        # Prevent negative iterates: damp dx where needed
        neg_mask = x_new < 0.0
        if neg_mask.any():
            # simple halving damping until non-negative (vectorized)
            # this loop will rarely iterate more than a few times
            alpha = 1.0
            x_try = x_new.copy()
            # perform up to 10 halvings (safe fallback)
            for _d in range(10):
                bad = (x_try < 0.0) & neg_mask
                if not bad.any():
                    break
                alpha *= 0.5
                x_try[bad] = xk[bad] + alpha * dx[bad]
            x_new = np.maximum(x_try, 0.0)
        xk = x_new
        # global stop if max change small
        if np.max(np.abs(dx)) < tol * np.maximum(1.0, np.max(np.abs(xk))):
            break
    
    xk = np.maximum(xk, 0.0)
    return xk if xk.shape != () else float(xk)




# %% Solve using Runge-Kutta integration

# Problem: dx/dt = p - r * x**lam
def f(t, x, p, r, lam):
    return p - r * x**lam

# Example parameters
p, r = 0.0, 0.25
x0 = 0.4
t_span = (0.0, 1.5)

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_title(r'Comparison: $\dot x = p - r x^\lambda$, $\lambda=0.5$')
ax.set(xlabel='t', ylabel='x(t)')
ax.grid(True)

clrs = cycle('rbgkmcy')
for lam in [1.0, 0.75, 0.5, 0.25, 0.1]:
    clr = next(clrs)
    

    # integrate with RK45 and BDF
    sol_rk = solve_ivp(lambda t, x: f(t, x, p, r, lam), t_span, [x0], method='RK45', rtol=1e-8, atol=1e-10, dense_output=True)

    # compare on a common time grid
    N = 50
    t_plot = np.linspace(t_span[0], t_span[1], N)

    x_rk = sol_rk.sol(t_plot)[0]

    plt.plot(t_plot, x_rk, 'o', color=clr, mfc='none', label=fr'RK45, $\lambda$={lam}')
ax.legend()
plt.show()

# %%

# Using the exact solution for lambda = 0.5, using the Lambert function




# %%
p, x0 = 0.1, 0.9
t = np.linspace(0, 30)

fig, ax = plt.subplots(figsize=(8,6))
ax.set_title(r'x=S/Smax  implicit euler method')
ax.set(xlabel='t', ylabel='S/Smax')
ax.grid(True)

for lam in [0.1]:
    clrs = cycle('rbgkmcy')
    for r in [0., 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]:
        xt = np.zeros_like(t)
        for i, dt in enumerate(t):
            xt[i] = implicit_euler_step(x0, dt, p, r, lam)
        ax.plot(t, xt, label=f"p={p}, r={r}")
ax.legend()

# %%
