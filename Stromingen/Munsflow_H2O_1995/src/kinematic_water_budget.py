# %% [markdown]

# # Kinematic water budget

# The aim is to demonstrate the water budget of a percolating wave of moisture

# %% --- imports

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import cycle
from scipy.integrate import quad, simpson

import etc

dirs = etc.Dirs(os.getcwd())
# dirs.src = str(dirs.src)

#if dirs.src not in sys.path:
#    sys.path.insert(0, str(dirs.src))

from src.NL_soils import Soil # noqa

# %%  --- load the Dutch soils and import table 4.

wbook = os.path.join(dirs.data, 'NL_VG_soilprops.xlsx')
if not os.path.isfile(wbook):
    raise FileNotFoundError(f"Can't find file `{wbook}`")

Soil.load_soils(wbook=wbook)

# %%

# --- choose a soil
soil = Soil('O05')
q = 0.2 # cm/d
theta = soil.theta_fr_K(q)
theta_fc = soil.theta_fc()
Vtail  = soil.dK_dtheta(theta)
Vfront = (soil.K_fr_theta(theta) - soil.K_fr_theta(theta_fc)) / (theta - theta_fc)

print(f"q={q}, theta={theta:.3f}, theta_fc={theta_fc:.3f}, Vtail={Vtail:.3f}, Vfront={Vfront:.3f}")

# --- choose a time since the infiltration at z=0 has stopped
t = 50 # d
xt = Vtail * t
print(f"t={t:.3g} d, xtail={xt:.1f} cm")

# --- compute the moisture content of the tail and the lack of it
x = np.linspace(0, xt, 100) + 1e-20



vavg = x / t # velocity of the points of the tail
theta_t = soil.theta_fr_V(vavg)
theta_t[0] = theta_fc # prevent NaN

ax =etc.newfig("Tail", "x cm", "theta")
ax.plot(x, theta_t, label=f"tail, t={t:.1f}")
ax.legend()

# --- Verify the graph of the tail from the perspective of theta
thetas = np.linspace(theta_fc, theta, 20)
Vs = soil.dK_dtheta(thetas)

dtheta = 1e-4
Vs2 = (soil.K_fr_theta(thetas + 0.5 * dtheta) - soil.K_fr_theta(thetas - 0.5 * dtheta)) / dtheta

ax.plot(Vs  * t, thetas, 'o', label=f"theta perspective t={t}")
ax.plot(Vs2 * t, thetas, 'o', label=f"theta perspective t={t} from K")

ax.legend()

# --- compute volume in the tail and in the air above the tail
Vol_tail = simpson(theta_t - theta_fc, x=x)
Vol_air  = simpson(theta   - theta_t,  x=x)
Vol_sum1  = t * Vtail * (theta - theta_fc)

# --- compute volume by progressing front must equal Vair of the tail
# --- compute the lack of volume at the front due to its reduced velocity
Vol_lost = t * (Vtail - Vfront) * (theta - theta_fc)
Vol_prog = t *  Vfront          * (theta - theta_fc)
Vol_sum2 = t *  Vtail           * (theta - theta_fc)

print(f"Vol_tail = {Vol_tail:.1f} cm, Vol_air   = {Vol_air:.1f} cm, Vol_sum={Vol_sum1:.1f} cm")
print(f"Vol_lost = {Vol_lost:.1f} cm, Vol_front = {Vol_prog:.1f} cm, Vol_sum={Vol_sum2:.1f} cm")

# %%
