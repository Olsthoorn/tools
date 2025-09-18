# %% [markdown]

# # Kinematic water budget

# The aim is to demonstrate the water budget of a percolating wave of moisture.
# The aim succeeds.

# %% --- imports

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import cycle
from scipy.integrate import simpson

import etc

dirs = etc.Dirs(os.getcwd())
sys.path.insert(0, "")

from src.NL_soils import Soil # noqa

# %%  --- load the Dutch soils and import table 4.

wbook = os.path.join(dirs.data, 'NL_VG_soilprops.xlsx')
if not os.path.isfile(wbook):
    raise FileNotFoundError(f"Can't find file `{wbook}`")

Soil.load_soils(wbook=wbook)

# %% --- functions (plot the profile)

def plot_profile(t_tail=None, t_inf=None, soil=None, q=None, ax=None, color=None, lw=None):    
    """Plot the block-wave of moisture.
    
    Easy to plot for single t_inf and serveral t_tail to show the development of the
    front and the tail.
    
    Parameters
    ---------
    t_inf: float
        Duration of the infiltration phase
    t_tail: float
        Time after infiltration ceased
    soil: Soil object
        current soil used in this simulation
    q_inf: float [cm/d]
        infiltration flux
    ax: axes object
        axes to plot on
    """
    # --- relevant theta's
    theta = soil.theta_fr_K(q)
    theta_fc = soil.theta_fc()
    
    # --- relevant velocties (of moisture points)
    vtail = soil.dK_dtheta(theta)
    vfront = (soil.K_fr_theta(theta) - soil.K_fr_theta(theta_fc)) / (theta - theta_fc)
    
    # --- relevant displacement of block
    xf = vfront * (t_tail + t_inf)  # Front of the infiltrated block
    xt = vtail  *  t_tail           # Rear of the infiltrated block
    xe = vtail  * (t_tail + t_inf)  # head when t_front were t_tail
    
    # --- the entire tail    
    x = np.linspace(0, xt, 50)            # tail points
    v_avg = x / t_tail                # velocity of tail points
    theta_t = soil.theta_fr_V(v_avg)  # theta of tail points

    # --- Verify the water budget

    # --- Total volume of water infiltrated
    s1 = f"Total infiltrated: {t_inf * q:.2f} cm, "

    # --- Compute total volume present within the profile:
    Vol = simpson(theta_t - theta_fc, x=x) + (xf - xt) * (theta - theta_fc)
    
    # --- Issue total volume in profile.
    s2 = f"Volume in profile={Vol:.2f} cm at t={t_inf + t_tail:.1f} d"
    print(s1 + s2)
    
    # --- Show the profile, tail first, block and front next
    ax.plot(x, theta_t, '-', color=color, lw=lw,
            label=fr"$t$={t_tail + t_inf:.1f} d, $t_{{inf}}$={t_inf:.1f} d, $q$={q:.2} cm/d")
    ax.plot([xt, xf, xf, xe], [theta, theta, theta_fc, theta_fc], '-', color=color, lw=lw)
    
# %% -- Choose one soil to check the water budget with

soil = Soil('O05')                # Coarse sand
q = 0.2 # cm/d                    # Constant infiltration rate cm/d
t_inf, t_tail = 60, 10 # d        # Infiltration duration and time since infiltration stopped
theta = soil.theta_fr_K(q)        # Theta pertaining to q
theta_fc = soil.theta_fc()        # Field capacity for this soil
theta_s  = soil.theta_s           # porosity

# --- Moisture point velocity pertaining to q = same as vtail
vtail  = soil.dK_dtheta(theta)

# --- Sharp front velocity
vfront = (soil.K_fr_theta(theta) - soil.K_fr_theta(theta_fc)) / (theta - theta_fc)

# --- Verify consistency of implementation of soil relations
q_     = soil.K_fr_theta(soil.theta_fr_K(q))          # should yield q
vtail_ = soil.dK_dtheta(soil.theta_fr_V(vtail))       # should yield vtail
print("--- Verify consistency of implementation of soil formulas")
print(f"q     = {q:.3f} cm/d --> q_out  = {q_:.3f} cm/d")
print(f"vtail = {vtail:.3f} cm   --> vtail_ = {vtail_:.3f} cm")
print("---")

# --- show the profiles
t_inf = 20.
print(f"q={q}, theta={theta:.3f}, theta_fc={theta_fc:.3f}, vtail={vtail:.3f}, vfront={vfront:.3f}")

title = (f"Situation after infitration. Soil = {soil.code} = '{soil.props['Omschrijving']}', " +
        fr"$\theta_s$={theta_s:.3g}, $\theta$={theta:.3g}, $\theta_{{fc}}$={theta_fc:.3g}")
ax = etc.newfig(title, r"$x$ [cm]", r"--- $\theta$ ---")
fig = ax.figure

clrs = cycle('brgkmcy')
lws  = cycle([4, 2, 1, .5])

# --- time at which the tail hits the moisture front
t_last = t_inf * vfront / (vtail - vfront)

for t_tail in [1e-8, 4.5, t_last]:
    clr = next(clrs)
    lw  = next(lws)
    plot_profile(t_tail=t_tail, t_inf=t_inf, soil=soil, q=q, ax=ax, color=clr, lw=lw)    
ax.legend()
# fig.savefig(os.path.join(dirs.images, "infiltrated_block_movement.png"))


# %% --- construct a moisture tail at t=t_tail after stopping the infiltration

q = 0.2 # cm/d
t_inf,  t_tail = 20., 4.5
theta = soil.theta_fr_K(q)        # Theta pertaining to q
vtail = soil.dK_dtheta(theta)
xt = vtail * t_tail                    # Progress of the first tail point over time t
print(f"t={t_tail:.3g} d, xtail={xt:.1f} cm")

# --- compute the moisture content of the tail (and also the air volume above the tail)
x = np.linspace(0, xt, 100) + 1e-20   # Choose a series of tail points with 0 < x < vtail * t

vavg = x / t_tail                 # Actuall velocity of the points of the tail
theta_t = soil.theta_fr_V(vavg)   # theta of these points theta(v_actual)
theta_t[0] = theta_fc             # prevent NaN

# --- show the tail
ax =etc.newfig("Tail", "x cm", "theta")
ax.plot(x, theta_t, label=f"tail, t={t_tail:.1f}")
ax.legend()

#  --- Verify the graph of the tail from the perspective of theta
thetas = np.linspace(theta_fc, theta, 20) # Choose some theta values
vs = soil.dK_dtheta(thetas)             # The moisture wave velocity pertaining to these theta value

# --- Plot the position of the points with the given theta values
# --- These should fall on the previously plotted tail-line
ax.plot(vs  * t_tail, thetas, 'o', label=f"theta perspective t={t_tail}")
ax.legend()

# --- Plot the positio of the remaining infiltration block
x_front = (t_inf + t_tail) * vfront
x_tail  = t_tail * vtail
ax.plot([x_tail, x_tail, x_front, x_front], [theta_fc, theta, theta, theta_fc], label="infiltrate" )

# ---  Budget verification

# --- Verfiy the water budget of the tail
# --- compute volume in the tail and in the air above the tail
Vol_tail = simpson(theta_t - theta_fc, x=x)   # Water volume below the tail
Vol_air  = simpson(theta   - theta_t,  x=x)   # Air volume above the tail
Vol_sum1  = t_tail * vtail * (theta - theta_fc)    # Sum of air and water volume in tail zone

# --- compute water budget at the (sharp)
# --- compute the lack of volume at the front caused by its reduced velocity
Vol_lost = t_tail * (vtail - vfront) * (theta - theta_fc)

# --- Volume of the progressing front (has V=vfront)
Vol_prog = t_tail *  vfront          * (theta - theta_fc)

# --- Vol_lost (from theretical head) must equal V_tail
# --- Vol_prog must equal V_air
print(f"Vol_lost = {Vol_lost:.2f} should equal V_tail = {Vol_tail:.2f}")
print(f"Vol_prog = {Vol_prog:.2f} should equal V_air  = {Vol_air :.2f}")

# --- The sum of the latter two volumes must equal the total volume infiltrated
Vol_sum2 = t_tail *  vtail           * (theta - theta_fc)

print(f"Vol_tail = {Vol_tail:.2f} cm, Vol_air   = {Vol_air :.2f} cm, Vol_sum={Vol_sum1:.2f} cm")
print(f"Vol_lost = {Vol_lost:.2f} cm, Vol_front = {Vol_prog:.2f} cm, Vol_sum={Vol_sum2:.2f} cm")

# %%
plt.show()
