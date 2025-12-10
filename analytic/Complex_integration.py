# Complex integration
"""Some exercises in complex integration"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# %% [markdown]
# 
# $$
# \intop_C f(z)dz = \intop_C(u + iv)(dx + i dy)
# $$
# 
# if we separate the integral into its real and imaginary parts as follows
# 
# $$
# \intop_C f(z)dz = \intop_C (u dx + v dy) + i \intop_C (v dx + u dy)
# 
# $$
# %% 
# $$
# \intop_{1+1}^{3 + 3i} z dz

# $$
# %%
def f(z):
    return z
dydx = 1
I1 = quad(lambda z: f(z).real - f(z).imag * dydx, 1, 3)
I2 = quad(lambda z: f(z).imag + f(z).real * dydx, 1, 3)
print(f"result = {I1 + 1j * I2}")



# %%
print(2 * 3)
# %%
