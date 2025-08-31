import numpy as np
from scipy.special import lambertw
import matplotlib.pyplot as plt

def x_solution(t, p, e, x0, t0=0.0):
    """
    Explicit solution for dx/dt = p - e*sqrt(x), x(t0)=x0.
    Accepts scalar or numpy array t.
    """
    t = np.asarray(t)
    u0 = np.sqrt(x0)
    y0 = p - e * u0
    # A(t) = (y0/p)*exp(-y0/p) * exp(- (e^2/(2p))*(t-t0) )
    A = (y0 / p) * np.exp(-y0 / p) * np.exp(-(e**2 / (2.0 * p)) * (t - t0))
    arg = -A   # argument for W
    w = lambertw(arg)            # principal branch; returns complex type in general
    w = np.real_if_close(w)      # convert to real if imaginary part is negligible
    z = -w.real                   # z = -W(arg)
    u = (p * (1.0 - z)) / e
    return (u ** 2)

# Example: verify transient approaches steady-state
p, e = 2.0, 1.0
x0, t0 = 0.1, 0.0
t = np.linspace(0, 10, 400)
x_t = x_solution(t, p, e, x0, t0)

plt.plot(t, x_t, label="x(t)")
plt.axhline((p/e)**2, color="red", ls="--", label="steady state (p/e)^2")
plt.xlabel("t")
plt.ylabel("x(t)")
plt.legend()
plt.show()
