# US soils reads table 4.4.1 from Charbeneau (2000) to get the Brooks and Corey parameters
# for 12 US soils.

# %%
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import cycle
from importlib import reload

import etc
reload(sys.modules['etc'])
dirs = etc.Dirs()

from src.soil_base import SoilBase # noqa

# %% Define soil Van Genughten properties (from Heinen et al (2018) for Dutch soils: Staring Series
class Soil(SoilBase):
    
    data = None
    
    def __init__(self, soil_code: str =None, HBW: bool = True) -> None:
        """Return soil object.
        
        Parameters
        ----------
        soil_code: str
            code of one of the Dutch soils 'O01'..'O18'  or 'B01'..'B18'
        HBW: bool, default True
            use K according to Heinen, Bakker & Wösten (2018)
            if False K according to Van Genughten - Mualem is used with el=0.5
            
        Usage
        -----
        Soil.load_soils(wbook)
        zand_boven = Soil('O01', HBW=False)
        zand_onder = Soil('B01')
        """
        if soil_code is not None:
            SoilBase.__init__(self, soil_code)
        else:
            raise ValueError("Missing positional argument 'soil_code'")
        
        # Additional properties        
        self.props['el'] = 0.5
        self.props['m'] = 1 - 1 / self.props['n']
        self.code = soil_code

        # Set HBW boolean to True of False to choose conductiviy model to be
        # either according to Heinen, Bakker and Wösten (2018) or
        # according to Vn Genughten and Mualem (1980) using "el" l=0.5.
        # Because we obtain all parameters from HBW, HBW = True is default.
        self.HBW = HBW
        
 
    @classmethod
    def load_soils(cls, wbook: str | Path)-> None:
        """Load Dutch soils parameters once into class attribute.
        
        The parameters are from:
        Heinen, M, G.Bakker & J.H.M. Wösten (2018) "Waterretentie- en
        doorlatendheidskarakteristieken van boven- en ondergronde in   
        Nederland: de Staringreeks, Update 2018. WUR rep. 2978, ISSN 1566-7197.
        """
        df = pd.read_excel(wbook, sheet_name='Sheet1',
                            usecols='A:O',
                            dtype={'A':str, 'B':str, 'C':str, 'D':float, 'E':float, 'F':float, 'G':float, 'H':float, 'I':float,
                                   'J':str, 'K':str, 'L':str, 'M':str, 'N':int, 'O':int},
                            index_col='code',
                            header=0,
                            skiprows=[1])
        
        df['theta_r'] = np.fmax(df['theta_r'], 0.01)
        
        cls.data = df
        
        # The dimensions of these data are
        cls.dimensions={
            'Hoofdsoort':'',
            'Omschrijving':'',
            'theta_r':'',
            'theta_s':'',
            'alpha':'1/cm',
            'n':'',
            'lambda':'',
            'Ks':'cm/d',            
            'Leem':'%',
            'Lutum':'%',
            'os':'%',
            'M50':'um',
            'N1':'',
            'N2':''}
        
        extra_in_df   = set(df.columns) - set(cls.dimensions.keys())
        missing_in_df = set(cls.dimensions.keys()) - set(df.columns)
        
        if missing_in_df or extra_in_df:
            raise ValueError(
                f"Mismatch between dimensions and DataFrame columns. "
                f"Missing in DataFrame: {missing_in_df}, Extra in DataFrame: {extra_in_df}"
            )
                        
    def S_fr_psi(self, psi: float | np.ndarray)-> float | np.ndarray:
        """Return S(psi)"""
        alpha, n, m = self.props['alpha'], self.props['n'], self.props['m']        
        return (1 + (alpha * psi) ** n) ** (-m)
    
    def psi_fr_S(self, S: float | np.ndarray)-> float | np.ndarray:
        """Return psi(S)"""
        alpha, n, m = self.props['alpha'], self.props['n'], self.props['m']
        return 1 / alpha * (S ** (-1 / m) - 1) ** (1  / n)
    
    def dS_dpsi(self, psi: float | np.ndarray)-> np.ndarray:
        """Return dS_dpsi"""
        alpha, m, n = self.props['alpha'], self.props['m'], self.props['n']
        apsi = alpha * psi
        return - alpha * m * n * (1 + apsi ** n) ** (-1 - m) * apsi ** (n - 1)
    
    def dpsi_dS(self, S: float | np.ndarray)-> np.ndarray:
        """Return dpsi/dS"""    
        alpha, n, m = self.props['alpha'], self.props['n'], self.props['m']
        return -(1 / (alpha * n * m)) *(S ** (-1/m) - 1) ** (-1 + 1/n) * S ** (-1 - 1/m)  
        #return -(1 / (alpha * n * m)) *(S ** (-1/m) - 1) ** (1/n - 1) * S ** (-1/m - 1)
        
    # Van Genughten and Mualem
    def K_VGM_fr_S(self, S: float | np.ndarray, S_limit: float = 1e-12)-> float | np.ndarray:
        """Return K(S)"""        
        Ks, el, m = self.props['Ks'], self.props['el'], self.props['m']
        S = np.atleast_1d(S).clip(S_limit, 1.0)

        B = 1 - ((1 - S) ** (1/m)) ** m
        K = Ks * S ** el * B ** 2
        return K.item() if K.size == 1 else K
        
    def dK_VGM_dS(self, S: float | np.ndarray, S_limit: float = 1e-22)-> float | np.ndarray:
        """Return dK/dS"""
        Ks, el, m = self.props['Ks'], self.props['el'], self.props['m']        
        S = np.atleast_1d(S).clip(S_limit, 1.0)        

        B = 1 - ((1 - S) ** (1/m)) ** m
        dBdS = (1 - S ** (1 / m)) ** (m - 1) * S ** (-1 + 1 / m)
        dKdS = Ks * (el * S ** (el -1) * B ** 2 + 2 * S ** el * B * dBdS)
        return dKdS.item() if dKdS.size == 1 else dKdS
    
    # Heinen, Bakker and Wösten (2018):
    # def K_HBW_fr_S(self, S: float | np.ndarray, S_limit: float = 1e-12)-> float | np.ndarray:
    #     """Return K(S) according to Heinen, Bakker and Wösten (2018)"""
    #     Ks = self.props['Ks']
    #     n, m, lambda_ =self.props['n'], self.props['m'], self.props['lambda']
        
    #     S = np.atleast_1d(S).clip(S_limit, 1.0)        
    #     K = Ks * S ** (lambda_ + 2) * (S ** (-1) - (S ** (-1 / m) - 1) ** (1 - 1 / n)) ** 2        
    #     return K.item() if K.size == 1 else K

    def K_HBW_fr_S(self, S: float | np.ndarray, S_limit: float = 1e-4)-> float | np.ndarray:
        """Return K(S) according to Heinen, Bakker and Wösten (2018)"""
        Ks = self.props['Ks']
        n, m, lambda_ =self.props['n'], self.props['m'], self.props['lambda']
        
        S = np.atleast_1d(S).clip(S_limit, 1.0)
        K = Ks * S ** (lambda_ + 2) * (S ** (-1) - (S ** (-1 / m) - 1) ** (1 - 1 / n)) ** 2
        return K.item() if K.size == 1 else K
    
    def dK_HBW_dS(self, S: float | np.ndarray, S_limit: float = 1e-12)-> float | np.ndarray:
        """Return K(S) according to Heinen, Bakker and Wösten (2018)"""
        Ks, n, m, lambda_ = self.props['Ks'], self.props['n'], self.props['m'], self.props['lambda']        
        S = np.atleast_1d(S).clip(S_limit, 1.0)

        A = S ** (-1) - (S ** (-1/m) - 1) ** (1-1/n)
        dAdS = -S ** (-2) - (1 - 1/n) * (S ** (-1/m) - 1) ** (-1/n) * (-1/m) * S ** (-1-1/m)
        dKdS = Ks * ((lambda_ + 2) * S ** (lambda_ + 1) * A ** 2 +
                     S ** (lambda_ + 2) * A * dAdS
        )                                                 
        return dKdS.item() if dKdS.size == 1 else dKdS

    
    def K_fr_S(self, S: float | np.ndarray, S_limit: float = 1e-12)-> float | np.ndarray:
        """Return K(S) according HBW or VGM"""
        if self.HBW is True:
            return self.K_HBW_fr_S(S, S_limit=S_limit)
        else:
            return self.K_VGM_fr_S(S, S_limit-S_limit)
        
    def dK_dS(self, S: float | np.ndarray, S_limit: float = 1e-12)-> float | np.ndarray:
        """Return dK_dS according to HBW or VGM"""
        if self.HBW is True:
            return self.dK_HBW_dS(S, S_limit=S_limit)
        else:
            return self.dK_VGM_dS(S, S_limit=S_limit)


# -------------------------------------------
# Minimal tests — only run if you call python soil.py
# -------------------------------------------
def _smoke_test():
    """Quick internal test to check if Soil behaves at all."""
    
    soil_code = 'O01'
    soil = Soil(soil_code)
    
    S = np.linspace(0.1, 1.0, 5)
    K = soil.K_fr_S(S)
    print("K(S) =", K)

    # Test the interpolator
    K_test = [2.0, 5.0, 9.0]
    print("S(K) =", soil.S_fr_K(K_test))

# %%
if __name__ == "__main__":
    wbook = os.path.join(dirs.data, 'NL_VG_soilprops.xlsx')
    Soil.load_soils(wbook) # load once

    dirs = etc.Dirs('~/GRWMODELS/python/tools/Stromingen/Munsflow_H2O_1995')
    
    _smoke_test()


    # %% SGet the soil data from the Excel workbook
    wbook = os.path.join(dirs.data, 'NL_VG_soilprops.xlsx')
    Soil.load_soils(wbook) # load once
    Soil.pretty_data()
    sand_b = Soil("O01")
    sand_o = Soil("B01")

    # %% Show the parametr values parameters
    
    for BO in ['Boverngronden', 'Ondergronden']:
        title = f'K(theta) bij Veldcapaciteit van {BO} van de Staringreeks'
        ax = etc.newfig(title, 'soil_code', 'parameter value', yscale='log')

        soil_codes = [c for c in Soil.data.index if c.startswith(BO[0])]

        idx = soil_codes
        ax.plot(idx, Soil.data.loc[soil_codes, 'Ks'], lw=2, label=r'$K_s$ [cm/d]')
        ax.plot(idx, Soil.data.loc[soil_codes, 'theta_r'], lw=2, label=r'$\theta_r$')
        ax.plot(idx, Soil.data.loc[soil_codes, 'theta_s'], lw=2, label=r'$\theta_s$')        
        ax.plot(idx, Soil.data.loc[soil_codes, 'n'], '--', lw=2, label='n')
        ax.plot(idx, Soil.data.loc[soil_codes, 'lambda'], '-.', lw=2, label=r'$\lambda$ [-]')
        
        for i, code in enumerate(soil_codes):
            soil = Soil(code)

            ax.text((i + 1) / (len(soil_codes) + 1), 0.2,
                    soil.props['Omschrijving'], ha='center', rotation=90, fontsize=15, zorder=5,
                    bbox=dict(facecolor='white',
                            alpha=1,
                            edgecolor='none'),
                    transform=ax.transAxes)
            
        ax.legend(fontsize=15, loc='lower right')

        #ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        plt.xticks(rotation=90, ha="center")
        plt.tight_layout()

    
    # %% Get soil_codes of sands of "Benedengronden"
    
    soil_codes = Soil.data.index[Soil.data.loc[:, 'Hoofdsoort'] == 'Zand']
    soil_codes = [code for code in soil_codes if code.startswith('B')]


    # %% psi(theta)
    
    title = r"$\psi$ [cm] as a function of $\theta$, $\psi(\theta)$, with field capacity and wilting point"
    ax=etc.newfig(title, r'$\theta$', r'pF = log$_{10}(\psi$ [cm])', yscale='linear')
    
    psi = np.logspace(0, 4.2)
        
    clrs = cycle("brgkmcy")
    for code in soil_codes:
        clr = next(clrs)
        soil = Soil(code)
        theta = soil.theta_fr_psi(psi)
        soil_nm = soil.props['Omschrijving']
        
        
        pF  = np.log10(psi)
        pF1 = np.log10(soil.psi_fr_theta(theta))
        ax.plot(theta, pF,  '-', color=clr, label=fr"$\psi(\theta(\psi))$, {soil_nm}")
        ax.plot(theta, pF1, '.', color=clr, label=fr"$\psi(\theta(\psi))$, {soil_nm}")
        
        # Field capacity and wilting point
        pF_fc, pF_wp = 2.5, 4.2
        lbl1 = f"fc[pF={pF_fc}]" if code.endswith('01') else ""
        lbl2 = f"fc[pF={pF_wp}]" if code.endswith('01') else ""
        
        # Plot field_capacity and wilting_point
        ax.plot(soil.theta_fc(pF=pF_fc), pF_fc, 'o', color=clr, label=lbl1)
        ax.plot(soil.theta_wp(pF=pF_wp), pF_wp, 's', color=clr, label=lbl2)
        
    ax.legend(title="Legenda voor sands Bovengronden")


    # %% Show dtheta/dpsi for all sands.
    
    title = r"$\psi(\theta)$ and $\psi(\theta(\psi))$ for US soils using Van Genughten relations"
    ax = etc.newfig(title, r'$\psi$', r'-d$\theta$/d$\psi$ [cm]',
                    xscale='log', yscale='log')    

    psi = np.logspace(0, 4.2)
        
    clrs = cycle('brgkmc')
    for soil_code in soil_codes:
        soil = Soil(soil_code)
        soil_nm = soil.props['Omschrijving']
        theta = soil.theta_fr_psi(psi)     
        clr = next(clrs)
        ax.plot(psi, -soil.dtheta_dpsi(psi), '-', color=clr, label=fr'd$\theta$/d$\psi$   {soil_nm}')
        ax.plot(psi, -1 / soil.dpsi_dtheta(theta),'.', color=clr, label=fr'1 / d$\psi$/d$\theta$ {soil_nm}')

    ax.legend(loc='upper right')


    # %% Show K(theta)

    psi = np.logspace(0, 4.2)
    
    for BO in ['Bovengronden', 'Ondergronden']:
        clrs = cycle('brgkmc')
    
        title=fr"$K$ [cm/d] versus $\theta$ voor {BO}"
        ax = etc.newfig(title, r'$\theta$', r'$K(\theta)$',
                    yscale='log') #, ylim=(1e-7, 1e0))

        soil_codes = [c for c in Soil.data.index if c.startswith(BO[0])]
    
        for i, code in enumerate(soil_codes):
            clr = next(clrs)
            soil = Soil(code)
            soil_nm = soil.props['Omschrijving']
            theta = soil.theta_fr_psi(psi)
            ax.plot(theta, soil.K_fr_theta(theta), '-', color=clr, label=fr"{code}: $K(\theta)$ {soil_nm}")
            
            # # Numerical dKdtheta to check
            # K = soil.K_fr_theta(theta)
            # theta_mid = 0.5 * (theta[:-1] + theta[1:])            
            # dKdth = np.diff(K) / np.diff(theta)
            # ax.plot(theta_mid, dKdth, 'o', color=clr, label=fr"{code}: d$K$/d$\theta),numerical$ {soil_nm}")

            ax.plot(theta, soil.dK_dtheta(theta), '--', color=clr, label=fr"{code}: d$K$/d$\theta)$ {soil_nm}")
        ax.legend(loc='lower right')


    # %% Show the throttle function to throttle ET depending on theta
    
    soil = sand_o

    title = f"Throttle functions(theta) for ET  for soil {soil.props['Omschrijving']}"
    ax = etc.newfig(title, r"$\theta$", r"throttle $\alpha(\theta)$")
    
    
    theta = np.linspace(soil.props['theta_r'], soil.props['theta_s'], 200)
    
    # ax.plot(theta, soil.alpha_wet(theta), label=r'alpha_wett')
    ax.plot(theta, soil.feddes_alpha(theta), label='feddes_alpha')
    ax.plot(theta, soil.smooth_alpha(theta), label='smooth_alpha')
    beta = 1.0
    for w in [0.1, 0.25, 0.5, 0.75, 1.0]:
        ax.plot(theta, soil.alpha_dry_blend(theta, beta=beta, w=w), '-.',
            label=f'alpha_dry_blend, beta={beta}, w={w}')
    for beta in [1.0]:
           ax.plot(theta, soil.mualem_alpha(theta, beta=beta), '.-', label=f'mualem_alpha, beta={beta}')
    ax.legend()
    # plt.show()

    # Add the responsen

    # %% Compute the Impulse Response

    q_avg = .2 # cm/d
    t = np.linspace(0, 750, 751)[1:]

    soil = Soil("O01")  # Sand
    soil_nm = soil.props['Omschrijving']
    v, D = soil.get_vD(q_avg)

    ttl = f"Soil {soil.code}: {soil_nm}\n"
    title = ttl + f"Impulse Response for q_avg={q_avg:.3g} m/d, v={v:.3g}, D={D:.3g} m2/d"
    ax = etc.newfig(title, 't', 'unit response')

    clrs = cycle('rbgkmc')
    for z in [150, 300, 500, 750, 1000, 2000, 3000]: # z in cm
        clr = next(clrs)
        ax.plot(t, soil.IR(z, t, q_avg), '-', color=clr, label=f'IR, soil={soil_nm}, z={z:.0f} cm')
        ax.plot(t, soil.IR_PIII(z, t, q_avg), '--', color=clr, label=f'IR_PIII, soil={soil_nm}, z={z:.0f} cm')
    ax.legend(loc='best')
    # plt.show()

    # %% Compute the analytic Step Response
    
    q_avg = .2 # cm/d
    t = np.linspace(0, 750, 751)[1:]

    # soil = Soil("B05")  # Sand
    soil_nm = soil.props['Omschrijving']

    ttl = f"Soil {soil.code}: {soil_nm}\n"
    title = ttl + f"Step Response for q_avg={q_avg:.3g} m/d, v={v:.3g}, D={D:.3g} m2/d"
    ax = etc.newfig(title, 't', 'unit response')
    
    clrs = cycle('rbgkmc')
    for z in [150, 300, 500, 750, 1000, 2000, 3000]: # z in cm
        clr = next(clrs)
        ax.plot(t, soil.SR_Phi(z, t, q_avg), 'x', label=f'{soil_nm}, z={z:.0f} cm, IR_Phi')
        ax.plot(t, soil.SR_erfc(z, t, q_avg), '.', label=f'{soil_nm}, z={z:.0f} cm, IR_erfc')
        ax.plot(t, soil.SR_PIII(z, t, q_avg), label=f'{soil_nm}, z={z:.0f} cm, method of moments')
            
    ax.legend(loc='best')
    # plt.show()
    
    # %% Compute the Block Response

    q_avg = 0.2 # cm/d
    t = np.linspace(0, 750, 751)[1:]

    #soil = Soil("B05")  # Sand
    soil_nm = soil.props['Omschrijving']
    v, D = soil.get_vD(q_avg)

    ttl = f"Soil {soil.code}: {soil_nm}\n"
    title = ttl + f"Block Response for q_avg={q_avg:.3g} m/d, v={v:.3g}, D={D:.3g} m2/d"
    ax = etc.newfig(title, 't', 'unit response')

    clrs = cycle('rbgkmc')
    for z in [150, 300, 500, 750, 1000, 2000, 3000]: # z in cm
        clr = next(clrs)
        for soil_nm in ['sand']:
            ax.plot(t, soil.BR(soil.SR_Phi, z, t, q_avg),  '-', label=f'soil_nm, z={z:.0f} cm, BR_phi')
            ax.plot(t, soil.BR(soil.SR_erfc, z, t, q_avg), '.', label=f'soil_nm, z={z:.0f} cm, BR_erfc')                         
            ax.plot(t, soil.BR(soil.SR_PIII, z, t, q_avg), '+', label=f'soil_nm, z={z:.0f} cm, BR_mom')
            
    ax.legend(loc='best')
    # plt.show()
    
    # %% Show ratio  dK(theta)/dtheta / K(theta)

    psi = np.logspace(0, 4.2)

    ax = etc.newfig("V/q = (dK(theta)/theta) / K(theta) ",
                    "S",
                    "V/q = (dK(theta)/dtheta) / K(theta)",
                    yscale='log')
    for code in soil_codes:
        soil = Soil(code)
        soil_nm = soil.props['Omschrijving']

        S = soil.S_fr_psi(psi)
        theta = soil.theta_fr_S(S)

        ax.plot(S, soil.dK_dtheta(theta) / soil.K_fr_theta(theta),
            label=f"soil {soil.code}, {soil.props['Omschrijving']}")
    
    ax.legend()


    # %% Show theta_fr_V(v)
    
    # The distance travelled from the release time of particles is their speed, which is
    # always constant, in fact even at a sharp front, but that it hold for the point that
    # at time t has just reached the shock front.
    
    dtype = np.dtype([('t', '<f8'), ('tst', '<f8'), ('z', '<f8'), ('theta', '<f8'), ('v', '<f8')])
    
    N = 20
    profile = np.zeros(N, dtype=dtype).copy()
    prf = profile
    
    Zmax, dzmin, dzmax = 2000., 0.1, 1.
    dz = np.linspace(dzmin, dzmax, 20)
    
    profile['z'] = Zmax * np.cumsum(dz) / np.sum(dz)
    profile['tst']= np.linspace(0, 101, 20)
    profile['theta'] = soil.theta_fr_psi(300)
    profile['v'] = soil.K_fr_theta(profile['theta'])
    
    v_avg = np.zeros(N)
    
    times = np.linspace(0, 100, 101)
    
    ax = etc.newfig("Updating thetas", "z [m]", "theta [-]")
    
    for t, dt in zip(times[1:], np.diff(times)):
        mask = t > prf['tst']
        if np.all(mask) is False:            
            continue
        
        v_avg[mask] = prf['z'][mask] / (t - prf['tst'][mask])
        prf['theta'][mask] = soil.theta_fr_V(v_avg[mask])
        prf['t'] = t
        ax.plot(prf['z'], prf['theta'], label=f"t={t:.3f} d")
        
        prf['z'] += prf['v'] * dt
        
        ax.plot(prf['z'], prf['theta'], label=f"t={t}")
    ax.legend()
    
    def f(frame, args=(times, profile)):
        t = times[frame]
        mask = t > profile['tst1']
        if np.all(mask) is False:
            return line
        else:
            v_avg[mask] = profile['z'][mask] / (t - profile['tst1'][mask])
            profile['theta'][mask] = soil.theta_fr_V(v_avg[mask])
            profile['t'] = t
            return line


    # %%
    plt.show()
