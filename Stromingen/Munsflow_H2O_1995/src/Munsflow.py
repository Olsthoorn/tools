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
print(f"Project directory (cwd): {dirs.home} ")

if "" not in sys.path:
    sys.path.insert(0, "")

sys.path = sorted(sys.path)
sys.path.insert(1, str(dirs.home))

from src.NL_soils import Soil # noqa
from src.rootz_rch_model import get_deBilt_recharge # noqa


# --- update figure settings
plt.rcParams.update({
    'font.size': 15,
    'figure.figsize': (10, 6),
    'axes.grid': True,
    'grid.alpha': 0.5,
    'lines.linewidth': 2,
    'lines.markersize': 5
})

mmpcm = 10 # mm per cm


def simulate_munsflow(meteo, soil, z, q_avg_cm):
    """Simulate Munsflow using 'RCH' column in meteo.
    
    Parameters
    ----------
    meteo: pd.DataFrame with meteo data having column 'RCH'
        Meteo data in mm/d.
    z: float
        depth of the water table.
    q_avg: float
        average long-term moisture content in profile
    
    Returns
    -------
    pd.Series
        q values at water table, index is that of meteo
    """
    tau = (meteo.index - meteo.index[0]) / np.timedelta64(1, 'D')
    BR = soil.BR(soil.SR_erfc, z, tau, q_avg_cm)
    q_cmpd = lfilter(BR, 1., meteo['RCH'].values / mmpcm)
    return pd.Series(q_cmpd, index=meteo.index)

# %%
if __name__ == '__main__':

    # %% --- load the Soil properties ---
    

    
    soil = Soil('O05')

    #  --- Importing the recharge data
    meteo = get_deBilt_recharge(Smax_I=0.5, Smax_R=100, lam=0.25,
                                datespan=("2020-01-01", None))

    q_avg_cm = 2 * meteo['RCH'].mean() / 10. # cm/d

    tau = np.asarray(meteo.index - meteo.index[0]) / np.timedelta64(1, 'D')
    tau[0] = 1e-6


    # %% === Compute and show the step, block and impulse responses ====

    title=f"Block respsonse and Impulse reponse for soil={soil.code}, [{soil.props['Omschrijving']}]"
    ax = etc.newfig(title, "time [d]", "Block response of q [sm/d]")

    zs = [250, 500., 1000., 2000.]
    BR, IR, SR, q = dict(), dict(), dict(), dict()

    title=f"Step respsonse for soil={soil.code}, [{soil.props['Omschrijving']}]"
    ax = etc.newfig(title, "time [d]", "Step response of (q = 1.0 [cm/d])")

    tau_i = dict()
    for i, z in enumerate(zs):    
        SR[i] = soil.SR_erfc(z, tau, q_avg_cm)    
        mask = SR[i] < 0.999
        tau_i[i] = tau[mask]
        SR[i] = SR[i][mask]

        ax.plot(tau_i[i], SR[i], label=f"z={z} cm, soil={soil.code}")

    title=f"Impulse and block response for soil={soil.code}, [{soil.props['Omschrijving']}]"
    ax = etc.newfig(title, "time [d]", "Block response of (q = 1, dt = 1 D) [cm/d]")

    for i, z in enumerate(zs):
        # 
        BR[i] = soil.BR(soil.SR_erfc, z, tau_i[i], q_avg_cm)
        IR[i] = soil.IR(z, tau_i[i], q_avg_cm)

        ax.plot(tau_i[i], BR[i], '-',  label=f"Block response:   z={z} cm, soil={soil.code}")
        ax.plot(tau_i[i], IR[i], '--', label=f"Impulse response: z={z} cm, soil={soil.code}")
        
    ax.legend()

    # %% === Simulation of recharge ===
    title=fr"$q$ from root_zone and recharge at depth z for [{soil.props['Omschrijving'].capitalize()}]"

    ax = etc.newfig(title, "time [d]", r"recharge $q$ [cm/d]")

    #ax.plot(meteo.index, meteo['RCH'] / 10., label = "From root zone [cm/d]")
    q = dict()

    for i, z in enumerate(zs):
        print(f"Running for i={i} ...")
        q[i] = lfilter(BR[i], 1., meteo['RCH'].values / mmpcm)
        ax.plot(meteo.index, q[i], label=f"z={z:.0f} cm")
        print(f"z = {z} cm, q = {q[i].sum():.3g} cm, rch = {meteo['RCH'].sum() / 10.: .3g} cm")
    print("... done.")

    # ax.set_xlim(np.datetime64("1990-01-01"), np.datetime64("1995-01-01"))
    ax.legend(loc='best')
    ax.grid(True)
    plt.show()

    # --- Balance

    for i, z in enumerate(zs):    
        print(f"z = {z} cm, q = {q[i].sum():.3g} cm, rch = {meteo['RCH'].sum() / 10.: .3g} cm")
        
        

    # %%

    qSeries = simulate_munsflow(meteo=meteo, soil=soil, z=2000, q_avg_cm=0.1)

    title = (f"Simulated recharge. Soil = {soil.code}, {soil.props['Omschrijving'].capitalize()}, z={z:.0f} cm, q_avg={q_avg_cm} cm/d")

    ax = etc.newfig(title, "time [d]", "q at water table [cm/d]")
    ax.plot(qSeries.index, qSeries.values, label=f"recharge at z={z:.0f} cm")

# %%