# %% [markdown]
# # US_soils
# 
# This module implements the different functions that mathematically describe
# the unsaturated zone using the Brooks and Corey (1966) relations.
# Alternative mathematical formulation (BC) using the Van Genughten Mualem (1980) relations
# Have been used in NL_soils to implement the properties of the Dutch Staraingreeks series
# of soils.
#
# Many of the formulas and much of the theory of the unsaturated zone can be found
# in the book Charbeneau (2000).
#
# @TO 2025-09-01
# """

# %%
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import cycle
from importlib import reload
from pathlib import Path

import etc
reload(sys.modules['etc'])

from .soil_base import SoilBase  #noqa

wbook_folder = os.path.join(Path(__file__).resolve().parent.parent, 'data')

# %% Define soil properties (from Charbeneau (2000), yet without their uncertainties.
class Soil(SoilBase):

    def __init__(self, soil_code: str)-> None:
        """Return soil object.

        Parameters
        ----------
        soil_code: str
            code of one of the Dutch soils 'O01'..'O18'  or 'B01'..'B18'

        Usage
        -----
        Soil.load_soils(wbook)
        zand_boven = Soil('O01')
        zand_onder = Soil('B01')
        """
        SoilBase.__init__(self, soil_code)

        # Any additional self.props parameters go here
        # self.props['el'] = 0.5
        # ...
        self.code = soil_code


    @classmethod
    def load_soils(cls, wbook):
        """Load US soils table of Brooks and Corey parameters once into class attribute.

        The parameters are from:
        Charmbneau (2000), table 4.4.1, p199, Originally published by Carsen & Parish (1980)
        The parameters are Brooks & Corey parameters and their standard deviations.

        parameters
        ----------
        wbook: str | path
            Excel workbook with the table of Charbeneau (dimensions have been converted to cm and cm/d)
            Look in project-->data for US_BC_soilprops.xlsx


        >>>import Soils
        >>>Soils.load_soils(wbook) # look at ../data/US_BC_soilprops.xlsx
        print(Soils.data)
        """
        df = pd.read_excel(wbook, sheet_name='Sheet1',
                            usecols='A:M',
                            dtype={'A':str, 'B':str, 'C':str, 'D':float, 'E':float, 'F':float, 'G':float, 'H':float, 'I':float,
                                   'J':float, 'K':float, 'L':float, 'M':float},
                            index_col='code',
                            header=0,
                            skiprows=[0, 2])

        cls.data = df

        # The dimensions of these data are
        cls.dimensions={
            'Main cat':'',
            'Soil Texture': '',
            'Ks': 'cm/d', 'sigma_Ks': 'cm/d',
            'theta_s':'', 'sigma_theta_s': '',
            'theta_r':'', 'sigma_theta_r': '',
            'psi_b':'cm', 'sigma_psi_b': 'cm',
            'lambda':'', 'sigma_lambda': '',
            }

        extra_in_df   = set(df.columns) - set(cls.dimensions.keys())
        missing_in_df = set(cls.dimensions.keys()) - set(df.columns)

        if missing_in_df or extra_in_df:
            raise ValueError(
                f"Mismatch between dimensions and DataFrame columns. "
                f"Missing in DataFrame: {missing_in_df}, Extra in DataFrame: {extra_in_df}"
            )

    def S_fr_psi(self, psi: float | np.ndarray)-> float | np.ndarray:
        """Return S(psi)"""
        psi_b, lambda_ = self.props['psi_b'], self.props['lambda']
        return (psi_b / psi) ** lambda_

    def psi_fr_S(self, S: float | np.ndarray)-> float | np.ndarray:
            """Return psi(S)"""
            return self.props['psi_b'] * S ** (-1 / self.props['lambda'])

    def dS_dpsi(self, psi: float | np.ndarray)-> np.ndarray:
        """Return dS_dpsi"""
        lambda_, psi_b= self.props['lambda'], self.props['psi_b']
        return - lambda_ / psi_b * (psi / psi_b) ** (-1 -lambda_)

    def dpsi_dS(self, S: float | np.ndarray)-> np.ndarray:
        """dpsi/dS"""
        psi_b, lambda_ = self.props['psi_b'], self.props['lambda']
        return -psi_b / lambda_ * S ** (-1 - 1 /lambda_)

    def K_fr_S(self, S: float | np.ndarray)-> float | np.ndarray:
        """Return K(S)"""
        Ks, lambda_ = self.props['Ks'], self.props['lambda']
        return Ks * S ** (3 + 2  / lambda_)

    def dK_dS(self, S: float | np.ndarray)-> float | np.ndarray:
        """Return dK/dS"""
        epsilon = 3 + 2  / self.props['lambda']
        return self.props['Ks'] * epsilon * S ** (epsilon - 1)


# %% --- load the Soil properties ---
wbook = os.path.join(wbook_folder, 'US_BC_soilprops.xlsx')
Soil.load_soils(wbook)


# %%
if __name__ == '__main__':
    # wbook = os.path.join(wbook_folder, 'US_BC_soilprops.xlsx')
    # Soil.load_soils(wbook) # load once
    Soil.pretty_data()

    sand_us = Soil('US09')
    loam_u3 = Soil('US04')

    soil_codes = ['US09', 'US04', 'US12']


    # %% Show psi-theta for all US soils.

    title = r"$\psi(\theta)$ and $\psi(\theta(\psi))$ for US soils using Brooks and Corey relations"
    ax = etc.newfig(title, r'$\theta$', r'$\psi$ [cm]', yscale='log')

    pF_fc, pF_wp = 2.5, 4.2

    clrs = cycle('brgkmc')
    for soil_code in soil_codes:
        soil = Soil(soil_code)
        soil_nm = soil.props['Soil Texture']
        psi = soil.psispace()
        theta = soil.theta_fr_psi(psi)
        clr = next(clrs)
        ax.plot(theta, soil.psi_fr_theta(theta), '-', color=clr, label=f'{soil_nm}')
        ax.plot(soil.theta_fr_psi(psi), psi,'.', color=clr)

        ax.plot(soil.theta_fc(pF=pF_fc), 10 ** pF_fc, 's', mfc=clr, label='field capacity')
        ax.plot(soil.theta_wp(pF=pF_wp), 10 ** pF_wp, 'o', mfc=clr, label='wilting point')

    ax.legend(loc='upper right')
    #plt.show()


    # %% Show dtheta/dpsi for all US soils.

    title = r"$\psi(\theta)$ and $\psi(\theta(\psi))$ for US soils using Brooks and Corey relations"
    ax = etc.newfig(title, r'$\psi$', r'-d$\theta$/d$\psi$ [cm]',
                    xscale='log', yscale='log')

    clrs = cycle('brgkmc')
    for soil_code in soil_codes:
        soil = Soil(soil_code)
        soil_nm = soil.props['Soil Texture']
        psi = soil.psispace()
        theta = soil.theta_fr_psi(psi)
        clr = next(clrs)
        ax.plot(psi, -soil.dtheta_dpsi(psi), '-', color=clr, label=f'{soil_nm}')
        ax.plot(psi, -1 / soil.dpsi_dtheta(theta),'.', color=clr)

    ax.legend(loc='upper right')
    #plt.show()


    # %% Show K(theta) K(theta(psi)) K(psi) K(psi(theta))

    title = r"$K(\theta), K(\theta(\psi)), K(\psi), K(\psi(\theta))$,"
    ax = etc.newfig(title, r'$\theta$', r'$K$ [cm/d]', yscale='log')

    clrs = cycle('brgkmc')
    for soil_code in soil_codes:
        soil = Soil(soil_code)
        soil_nm = soil.props['Soil Texture']
        psi = soil.psispace()
        theta = soil.theta_fr_psi(psi)

        clr = next(clrs)
        ax.plot(theta, soil.K_fr_theta(theta), '-', color=clr,
                 label=fr'$K_\theta(\theta)$ {soil_nm}')
        ax.plot(theta, soil.K_fr_theta(soil.theta_fr_psi(psi)), '.', color=clr,
                 label=fr'$K_\theta(\theta(\psi))$ {soil_nm}')
        ax.plot(theta, soil.K_fr_psi(psi), 'x', color=clr,
                 label=fr'$K_\psi(\psi)$ {soil_nm}')
        ax.plot(theta, soil.K_fr_psi(soil.psi_fr_theta(theta)), 'o', mfc='none', ms=10, color=clr,
                label=fr'$K_\psi(\psi(\theta))$ {soil_nm}')

    ax.legend(loc='lower right')
    # plt.show()


    # %% Show K(theta) K(theta(psi)) K(psi) K(psi(theta))

    title = r"$K(\theta)$, d$K(\theta)$/d$(\psi)$" #, +
             # " from BC-relations")
    ax = etc.newfig(title, r'$\theta$ [-]', r'$K$ and d$K$/d$\theta$', yscale='log')

    clrs = cycle('brgkmc')
    for soil_code in soil_codes:
        soil = Soil(soil_code)
        soil_nm = soil.props['Soil Texture']
        psi = soil.psispace()
        theta = soil.theta_fr_psi(psi)

        clr = next(clrs)
        ax.plot(theta, soil.K_fr_theta(theta), '-', color=clr, label=fr'$K$, {soil_nm}')
        ax.plot(theta, soil.dK_dtheta(theta),'--', color=clr, label=fr'd$K$/d$\theta$, {soil_nm}')

    ax.legend(loc='lower right')
    #plt.show()


    # %% Show K(theta) K(theta(psi)) K(psi) K(psi(theta))

    title = r"$K(\theta)$, d$K(\theta)$/d$(\psi)$" #, +
             # " from BC-relations")
    ax = etc.newfig(title, r'$\theta$ [-]', r'$K$ and d$K$/d$\theta$', yscale='log')

    clrs = cycle('brgkmc')
    for soil_code in soil_codes:
        soil = Soil(soil_code)
        soil_nm = soil.props['Soil Texture']
        psi = soil.psispace()
        theta = soil.theta_fr_psi(psi)

        clr = next(clrs)
        ax.plot(theta, soil.K_fr_theta(theta), '-', color=clr, label=fr'$K$, {soil_nm}')
        ax.plot(theta, soil.dK_dtheta(theta),'--', color=clr, label=fr'd$K$/d$\theta$, {soil_nm}')

    ax.legend(loc='lower right')
    #plt.show()


    # %% Show K(psi) dK(psi)/d(psi)

    title = r"-d$K_\psi(\psi)$/d$(\psi)$" #, +
             # " from BC-relations")
    ax = etc.newfig(title, r'$\psi$ [-]', r'-d$K_\psi(\psi)$/d$\psi$', xscale='log', yscale='log')

    clrs = cycle('brgkmc')
    for soil_code in soil_codes:
        soil = Soil(soil_code)
        soil_nm = soil.props['Soil Texture']
        psi = soil.psispace()

        clr = next(clrs)
        ax.plot(psi, soil.K_fr_psi(psi), '-', color=clr, label=fr'$K_\psi$, {soil_nm}')
        ax.plot(psi, -soil.dK_dpsi(psi),'--', color=clr, label=fr'd$K_\psi$/d$\psi$, {soil_nm}')

    ax.legend(loc='lower right')
    #plt.show()


    # %% -dPsi/dTheta
    title = r"-d$\psi$/d$\theta$ for Brooks and Corey"
    ax = etc.newfig(title, r'$\theta$', r'-d$\psi$/d$\theta$ (notice the minus sign)',
                    yscale='log')

    clrs = cycle('brgkmc')
    for soil_code in soil_codes:
        clr = next(clrs)
        soil = Soil(soil_code)
        soil_nm = soil.props['Soil Texture']
        psi = soil.psispace()
        theta = soil.theta_fr_psi(psi)
        ax.plot(theta, -soil.dpsi_dtheta(theta), '-', color=clr, label=f'K, {soil_nm}')

    ax.legend(loc='upper right')
    #plt.show()

    # %% Effect of parameters of the different soil types
    title = "Psi for Brooks and Corey lambda={lambda_:.3f}"
    ax = etc.newfig(title, 'US soil code', 'paramter value', yscale='log')

    df = Soil.data

    ax.plot(df.index, df['Ks'], lw=2, ls='-',  label=r'$K_s$ [cm/d]')
    ax.plot(df.index, df['theta_s'],  ls='--', label=r'$\theta_s$ [-]')
    ax.plot(df.index, df['theta_r'],  ls='--', label=r'$\theta_r$ [-]')
    ax.plot(df.index, df['psi_b'],    ls='-',  label=r'$\psi_b$ [cm]')
    ax.plot(df.index, df['lambda'],   ls='-.', label=r'$\lambda$ [-]')

    ax.legend()

    for i, soil_code in enumerate(df.index):
        xi = i
        ax.text((i + 0.5) / (len(df)), 0.4, df.loc[soil_code, 'Soil Texture'], transform=ax.transAxes, rotation=90, fontsize=15, ha='center')

    #plt.show()

    # %% Show the analytic IR

    q_avg = .2 # cm/d
    t = np.linspace(0, 300, 301)[1:]

    soil = Soil("US03")  # Sand
    soil_nm = soil.props['Soil Texture']
    v, D = soil.get_vD(q_avg)

    ttl = f"Soil {soil.code}: {soil_nm}\n"
    title = ttl + f"Impulse Response for q_avg={q_avg:.3g} m/d, v={v:.3g} cm/d, D={D:.3g} m2/d"
    ax = etc.newfig(title, 't', 'unit response')

    clrs = cycle('rbgkmc')
    for z in [150, 300, 500, 750, 1000, 2000, 3000]: # z in cm
        clr = next(clrs)
        ax.plot(t, soil.IR(z, t, q_avg), '-', color=clr, label=f'IR, soil={soil_nm}, z={z:.0f} cm,  method of moments')
        ax.plot(t, soil.IR_PIII(z, t, q_avg), '--', color=clr, label=f'IR_PIII, soil={soil_nm}, z={z:.0f} cm, method of moments')
    ax.legend(loc='best')
    # plt.show()

    # %% Compute the Step Reponse

    q_avg = .2 # cm/d
    t = np.linspace(0, 300, 301)[1:]

    soil = Soil("US03")
    soil_nm = soil.props['Soil Texture']
    ttl = f"Soil {soil.code}: {soil_nm}\n"
    title = ttl + f"Step Response for q_avg={q_avg:.3g} m/d, v={v:.3g}, D={D:.3g} m2/d"
    ax = etc.newfig(title, 't', 'unit response')

    clrs = cycle('rbgkmc')
    for z in [150, 300, 500, 750, 1000, 2000, 3000]:
        clr = next(clrs)
        ax.plot(t, soil.SR_Phi(z, t, q_avg), 'x', color=clr, label=f'{soil_nm}, z={z:.3g} cm, IR_Phi')
        ax.plot(t, soil.SR_erfc(z, t, q_avg), '.', color=clr, label=f'{soil_nm}, z={z:.3g} m, IR_erfc')
        ax.plot(t, soil.SR_PIII(z, t, q_avg), color=clr, label=f'{soil_nm}, z={z:.3g} m, method of moments')

    ax.legend(loc='best')
    # plt.show()

    # %% Compute the  Block  Response

    q_avg = 0.2 # cm/d
    t = np.linspace(0, 300, 301)[1:]

    soil = Soil("US03")  # Sand
    soil_nm = soil.props['Soil Texture']
    v, D = soil.get_vD(q_avg)
    ttl = f"Soil {soil.code}: {soil_nm}\n"
    title = ttl + f"BLock Response for q_avg={q_avg:.3g} m/d, v={v:.3g} cm/d, D={D:.3g} cm2/d"
    ax = etc.newfig(title, 't', 'unit response')

    clrs = cycle('rbgkmc')
    for z in [150, 300, 500, 750, 1000, 2000, 30000]:
        for soil_nm in ['sand']:
            clr = next(clrs)
            ax.plot(t, soil.BR(soil.SR_Phi, z, t, q_avg),  '-', color=clr, label=f'soil_nm, z={z:.0f} cm, BR_phi')
            ax.plot(t, soil.BR(soil.SR_erfc, z, t, q_avg), '.', color=clr, label=f'soil_nm, z={z:.0f} cm, BR_erfc')
            ax.plot(t, soil.BR(soil.SR_PIII, z, t, q_avg), '+', color=clr, label=f'soil_nm, z={z:.0f} cm, BR_mom')

    ax.legend(loc='best')
    # plt.show()


    # %%
    plt.show()

    # %%
