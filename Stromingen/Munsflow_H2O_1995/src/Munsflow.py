# %% [markdown]
# 
# # Munsflow (flow pasasing the unsaturated percolation zone)

# Notebook to run the "Munsflow". Munsflow is an application that simulates percolation through thick unsaturaed zones applying the linearized advection-diffusion equation. The unsaturated flow equations being strongly non-linear, this can only work in situaions where the moisture content is largely constant (varies little), so that the conductivity and the diffusivity may be considered constant and equal to the average situation. It is believed by many to be the case in thick unsaturated percolation zones like in the Veluwe and other high-elevation sandy areas. Thick percolation zones may start with thicknesses beyond say 5 m, in which the travel time for water leaving the rootzone is in th order of weeks or months. The Munsflow idea is use of the linearized AD equation, which then allows defining a unique impulse response, step respons and block response, with which one can compute the downrd flux at any depth of the percolation zone by convolution.

# %%

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import lfilter
import etc

dirs = etc.Dirs(os.getcwd())
sys.path.insert(0, os.getcwd())

from src import NL_soils as nls
from src.rootz_rch_model import RchLam

# %%
wbook = os.path.join(dirs.data, 'NL_VG_soilprops.xlsx')
nls.Soil.load_soils(wbook)
sand = nls.Soil('O05')

# %% Importing the recharge data

# Get the meteodata in a pd.DataFrame
meteo_csv = os.path.join(dirs.data, "DeBilt.csv")
os.path.isfile(meteo_csv)
deBilt = pd.read_csv(meteo_csv, header=0, parse_dates=True, index_col=0)
meteo = deBilt
meteo = deBilt.loc[deBilt.index >= np.datetime64("1985-01-01"), :]

rchSim = RchLam(Smax_I=2.5, Smax_R=100, lam=0.25)
rch = rchSim.simulate(meteo)

q_avg_cm = rch['RCH'].mean() / 10. # cm/d

tau = np.asarray(meteo.index - meteo.index[0]) / np.timedelta64(1, 'D')
tau[0] = 1e-6

# %%

title=f"Block respsonse and Impulse reponse for soil={sand.code}, [{sand.props['Omschrijving']}]"
ax = etc.newfig(title, "time [d]", "Block response of q [sm/d]")

zs = [500., 1000., 2000.]
BR, IR, SR, q = dict(), dict(), dict(), dict()

title=f"Step respsonse for soil={sand.code}, [{sand.props['Omschrijving']}]"
ax = etc.newfig(title, "time [d]", "Step response of (q = 1.0 [cm/d])")

tau_i = dict()
for i, z in enumerate(zs):    
    SR[i] = sand.SR_erfc(z, tau, q_avg_cm)    
    mask = SR[i] < 0.999
    tau_i[i] = tau[mask]
    SR[i] = SR[i][mask]

    ax.plot(tau_i[i], SR[i], label=f"z={z} cm, soil={sand.code}")

title=f"Impulse and block response for soil={sand.code}, [{sand.props['Omschrijving']}]"
ax = etc.newfig(title, "time [d]", "Block response of (q = 1, dt = 1 D) [cm/d]")

for i, z in enumerate(zs):
    # 
    BR[i] = sand.BR(sand.SR_erfc, z, tau_i[i], q_avg_cm)
    IR[i] = sand.IR(z, tau_i[i], q_avg_cm)

    ax.plot(tau_i[i], BR[i], '-',  label=f"Block response:   z={z} cm, soil={sand.code}")
    ax.plot(tau_i[i], IR[i], '--', label=f"Impulse response: z={z} cm, soil={sand.code}")
    
ax.legend()
    

ax.legend()
# %%
#  simulation of recharge
title=f"q root_zone and recharge at depth z for [{sand.props['Omschrijving']}]"
ax = etc.newfig(title, "time [d]", "recharge q [cm/d]")

#ax.plot(meteo.index, rch['RCH'] / 10., label = "From root zone [cm/d]")
for i, z in enumerate(zs):
    if i < 2:
        continue
    q[i] = lfilter(BR[i], 1., rch['RCH'] / 10.)
    ax.plot(meteo.index, q[i], label=f"z={z}")

ax.set_xlim(np.datetime64("1990-01-01"), np.datetime64("1995-01-01"))
ax.legend()

plt.show()

# Balance

for i, z in enumerate(zs):    
    print(f"z = {z} cm, q = {q[i].sum():.3g} cm, rch = {rch['RCH'].sum() / 10.: .3g} cm")

# %%
