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
from scipy.integrate import quad, simpson
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
    'figure.titlesize': 15,
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
                                datespan=("1992-01-01", "2012-01-01"))

    q_avg_cm = 1 * meteo['RCH'].mean() / 10. # cm/d

    tau = np.asarray(meteo.index - meteo.index[0]) / np.timedelta64(1, 'D')
    tau[0] = 1e-6

    # %% --- show the recharge for De Bilt

    ax = etc.newfig("De Bilt, outflow root zone, $S_I$=0.5 mm, $S_R$=100 mmm, $\lambda$=0.25", "time", "mm/d")
    mask = np.logical_and(meteo.index >= np.datetime64("1990-01-01"), meteo.index <= np.datetime64("2005-01-01"))
    ax.plot(meteo.index[mask], meteo.loc[mask, 'RCH'], label=r'$q$ from root zone')
    ax.grid(True)
    ax.legend()
    
    ax.figure.savefig(os.path.join(dirs.home, '../Kinematic_wave/LyX', "q_fr_root_zone.png"))


    # %% === Compute and show the step, block and impulse responses ====

    # --- specify the cases
    zs = [250, 500., 1000., 2000.]
    BR, IR, SR, q = dict(), dict(), dict(), dict()

    q_avg_cm = 0.1

    title=f"Step respsonse for soil={soil.code}, [{soil.props['Omschrijving']}], q_avg={q_avg_cm} cm/d"
    ax = etc.newfig(title, "time [d]", "SR", figsize=(10, 5), fontsize=15)

    tau_i = dict()
    for i, z in enumerate(zs):    
        SR[i] = soil.SR_erfc(z, tau, q_avg_cm)    
        mask = SR[i] < 0.999
        tau_i[i] = tau[mask]
        SR[i] = SR[i][mask]

        ax.plot(tau_i[i], SR[i], label=f"z={z:.0f} cm")
    ax.grid(True)
    ax.legend()
    
    ax.figure.savefig(os.path.join(dirs.home, '../Kinematic_wave/LyX', f'SR_munsflow_{q_avg_cm * 10:.0f}mmpd'))

    title=f"Impulse and block response for soil={soil.code}, [{soil.props['Omschrijving']}, q_avg = {q_avg_cm} cm/d]"
    ax = etc.newfig(title, "time [d]", "IR and BR for (dt=1 d)", figsize=(10, 5), fontsize=15)

    for i, z in enumerate(zs):
        # 
        BR[i] = soil.BR(soil.SR_erfc, z, tau_i[i], q_avg_cm)
        IR[i] = soil.IR(z, tau_i[i], q_avg_cm)
        
        # Check volume below IR
        yBr = BR[i]
        volBr = simpson(BR[i], x=tau_i[i])
        yIr = IR[i]
        volIr = simpson(IR[i], x=tau_i[i])
        print(f"{i}, volIr={volIr:.5f}, volBr={volBr:.5f}")
        

        ax.plot(tau_i[i], BR[i], '-',  label=f"BR:   z={z:.0f} cm")
        ax.plot(tau_i[i], IR[i], '--', label=f"IR: z={z:.0f} cm")
        
    ax.grid(True)
    ax.legend(fontsize=12)
    
    ax.figure.savefig(os.path.join(dirs.home, '../Kinematic_wave/LyX', f'BR_munsflow_{q_avg_cm * 10:.0f}mmpd'))
    
    
    # %% --- Voor verchillende q_avg
    zs = [1000., 1000., 1000., 1000.]
    BR, IR, SR, q = dict(), dict(), dict(), dict()

    q_avg_cms = [0.1, 0.2, 0.3, 0.5]

    title=f"Step respsonse for soil={soil.code}, [{soil.props['Omschrijving']}], afh van q_avg"
    ax = etc.newfig(title, "time [d]", "SR", figsize=(10, 5), fontsize=15)

    tau_i = dict()
    for i, (z, qavg) in enumerate(zip(zs, q_avg_cms)):    
        SR[i] = soil.SR_erfc(z, tau, qavg)    
        mask = SR[i] < 0.999
        tau_i[i] = tau[mask]
        SR[i] = SR[i][mask]

        ax.plot(tau_i[i], SR[i], label=f"z={z:.0f} cm, q_avg={qavg:.1f} cm/d")
    ax.grid(True)
    ax.legend()
    
    ax.figure.savefig(os.path.join(dirs.home, '../Kinematic_wave/LyX', 'SR_munsflow_afh_q'))


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