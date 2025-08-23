# Van Genughten soils using the data of Wösten et al.

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import cycle
import etc

import warnings
warnings.filterwarnings("error")

# %%
class Soil():
    
    data = None
    
    def __init__(self, soil_code=None):
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
        
        if Soil.data is None:
            raise RuntimeError("Load soils first with Soil.load(filename)")
        
        try:            
            self.props = Soil.data.loc[soil_code].copy()            
        except (ValueError, IndexError):
            raise IndexError(f"Soil code {soil_code} unknown, not in dutch_soils")
        self.props['el'] = 0.5
        self.props['m'] = 1 - 1 / self.props['n']
        self.code = soil_code        

    @classmethod
    def load_soils(cls, wbook):
        """Load Dutch soils parameters once into class attribute.
        
        The parameters are from:
        Heinen, M, G.Bakker & J.H.M. Wösten (2018) "Waterretentie- en
        doorlatendheidskarakteristieken van boven- en ondergronde in   
        Nederland: de Staringreeks, Update 2018. WUR rep. 2978, ISSN 1566-7197.
        """
        df = pd.read_excel(wbook, sheet_name='Sheet1',
                            usecols='A:O',
                            dtype={'A':str, 'B':str, 'C':float, 'D':float, 'E':float, 'F':float, 'G':float, 'H':float, 'I':str,
                                   'J':str, 'K':str, 'L':str, 'M':str, 'N':int, 'O':int},
                            index_col='code',
                            header=0,
                            skiprows=[1])
        
        cls.data = df
        
        # The dimensions of these data are
        cls.dimensions={
            'Hoofdsoort':'',
            'theta_r':'',
            'theta_s':'',
            'alpha':'1/cm',
            'n':'',
            'lambda':'',
            'Ks':'cm/d',
            'Omschrijving':'',
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
            
    @classmethod
    def pretty_data(cls):
        if cls.data is None:
            return "No soil  data loaded."
        print(cls.data.to_string()) # nicely formatted DataFrame
        
    def __str__(self):
        """Print soil properties in a readable format, starting with Hoofdsoort and Omschrijving."""
        # Order keys: start with hoofdsoort and Omschrijving
        keys = ['Hoofdsoort', 'Omschrijving'] + [k for k in self.props.index if k not in ('Hoofdsoort', 'Omschrijving')]
        
        lines = [f"Grondsoort code: {self.code}"]
        for k in keys:
            val = self.props[k]
            unit = Soil.dimensions.get(k, '')
            lines.append(f"{k:12} {unit:6}: {val}")
        return "\n".join(lines)

    def __repr__(self):
        return f"Soil('{self.props['code']}')"

    def S(self, theta, eps=1e-12): 
        r"""Return $S(\theta)$"""       
        theta_s, theta_r = self.props['theta_s'], self.props['theta_r']        
        return (np.clip(theta, theta_r + eps, theta_s - eps) - theta_r) / (theta_s - theta_r)
    
    def S_fr_psi(self, psi):
        """Return S(psi)"""
        alpha, n, m = self.props['alpha'], self.props['n'], self.props['m']        
        return (1 + (alpha * psi) ** n) ** (-m)
    
    def psi_fr_theta(self, theta):
        """Return $psi(theta)$"""
        alpha, n, m = self.props['alpha'], self.props['n'], self.props['m']
        S_ = self.S(theta)
        return (1 / alpha) * (S_ ** (-1 / m) - 1) ** (1 / n)
    
    def theta_fr_S(self, S, eps=1e-12):
        """Return theta(S)"""
        theta_s, theta_r = self.props['theta_s'], self.props['theta_r']
        return theta_r + (theta_s - theta_r) * np.clip(S, eps, 1 - eps)
    
    def theta_fr_psi(self, psi):
        """Return theta(psi)"""
        alpha, n, m = self.props['alpha'], self.props['n'], self.props['m']
        S_ = (1 + (alpha * psi) ** n) ** (-m)
        return self.theta_fr_S(S_)
    
    def B(self, theta):
        """Return B(\theta)=1 - (1 - S^{1/m})^m$"""
        m = self.props['m']
        S_ = self.S(theta)
        return 1 - (1 - S_ ** (1/m)) ** m
    
    def K(self, theta):
        """Return K(theta)"""
        Ks, el = self.props['Ks'], self.props['el']
        S_ = self.S(theta)
        return Ks * S_ ** el * self.B(theta) ** 2
    
    def dK_dtheta(self, theta):
        """Return dK/dtheta"""
        pr = self.props
        Ks, theta_s, theta_r = pr['Ks'], pr['theta_s'], pr['theta_r']
        el, m =  pr['el'], pr['m']
        S_ = self.S(theta)
        B_ = self.B(theta)
        
        # dB/dS with safe evaluation
        dB_dS = (1 - S_ ** (1/m)) ** (m - 1) * S_ ** (1 / m - 1) 
        
        return Ks / (theta_s - theta_r) * (
            el * S_ ** (el - 1) * B_ ** 2 + S_ ** el * 2 * B_ * dB_dS
            )
        
    def dpsi_dtheta(self, theta):
        """Return dpsi/dtheta."""
        pr = self.props
        theta_s, theta_r = pr['theta_s'], pr['theta_r']
        alpha, n, m = pr['alpha'], pr['n'], pr['m']
        S_ = self.S(theta)
        
        F = - 1 / (alpha * n * m * (theta_s - theta_r))
        return F * S_ ** (-1 / m -1) * (S_ ** (-1 / m) - 1) ** (1 / n - 1)
    
    def theta_fc(self, pF=2.5):
        """Return theta at field capacity (where pF = 2.0 or 2.5). (vdMolen (1973))"""        
        psi_fc = 10 ** pF # cm
        return self.theta_fr_psi(psi_fc)
    
    def theta_wp(self, pF=4.2):
        """Return theta at wilting point (where pF = 5.2) (vdMolen (1973))"""        
        psi_wp = 10 ** pF # cm
        return self.theta_fr_psi(psi_wp)
    
    def available_moisture(self):
        """Return available moisture to plants (between fc and wp)."""
        return self.theta_fc() - self.theta_wp()
    
    def alpha_wet(self, theta, psi1=10.0, psi2=25.0):
        """Return suffocation factor for wet conditions.
        
        Note that the input is theta, but the criteria are with psi1 and psi2.
        
        Parameters
        ----------
        theta: float or np.array
            moisture content
        psi1: float
            suction head below wich ET=0 (total suffocation), alpha=0
        psi2:
            suction head above which alpha=1 (no suffocation)
        """
        theta = np.atleast_1d(theta)
        psi = self.psi_fr_theta(theta)
        alpha = np.zeros_like(psi)
        alpha[psi < psi1] = 0.0
        alpha[psi > psi2] = 1.0
        rng = (psi >= psi1) & (psi <= psi2)
        alpha[rng] = (psi[rng] - psi1) / (psi2 - psi1)
        if len(alpha) == 1:
            return np.item(alpha)
        else:
            return alpha

    def feddes_alpha(self, theta, psi1=10, psi2=25, psi3=400, psi_w=16000):
        """
        Compute Feddes reduction factor alpha(ψ) for given water content(s).
        Input is θ, converted internally to ψ using van Genuchten–Mualem relations.

        Feddes stress function (example parameters in cm suction head):
            psi < psi1        → 0.0 (saturation, anoxia)
            psi1 < psi < psi2 → linear increase to 1.0 (wet side)
            psi2 ≤ psi ≤ psi3 → 1.0 (optimal)
            psi3 < psi < psi_w→ linear decrease to 0.0 (dry side)
            psi ≥ psi_w       → 0.0 (permanent wilting)

        Parameters
        ----------
        theta : float or array_like
            Volumetric moisture content.
        psi1, psi2, psi3, psi_w : float
            Suction head thresholds (cm).

        Returns
        -------
        alpha : float or ndarray
            Stress reduction factor in [0, 1], same shape as input.
        """
        theta = np.atleast_1d(theta)
        psi = self.psi_fr_theta(theta)  # convert θ → ψ
        alpha = np.zeros_like(psi, dtype=float)

        # Wet side: increase from 0 → 1
        mask_wet = (psi1 < psi) & (psi < psi2)
        alpha[mask_wet] = (psi[mask_wet] - psi1) / (psi2 - psi1)

        # Optimal zone: α = 1
        mask_opt = (psi2 <= psi) & (psi <= psi3)
        alpha[mask_opt] = 1.0

        # Dry side: decrease from 1 → 0
        mask_dry = (psi3 < psi) & (psi < psi_w)
        alpha[mask_dry] = (psi_w - psi[mask_dry]) / (psi_w - psi3)

        # Return scalar if input was scalar
        if alpha.size == 1:
            return alpha.item()
        return alpha

        
    def smooth_alpha(self, theta, s=0.2, pF50=3.6, psi2=25):
        """Return smooth ET throttling factor.

        Parameters
        ----------
        theta : float | array_like
            Volumetric moisture content.
        s : float, default=0.2
            Controls steepness of the dry-side logistic decline (0.15 < s < 0.30).
        pF50 : float, default=3.6
            pF at which alpha = 0.5 (typically 3.5 < pF50 < 3.8).
        psi2 : float, default=25
            Suction head [cm] at upper bound of wet side (Feddes psi2).

        Returns
        -------
        alpha : float or ndarray
            Reduction factor between 0 and 1.
        """
        theta = np.atleast_1d(theta)
        psi   = self.psi_fr_theta(theta)

        # pF definition
        pF = np.log10(psi)
        pF2 = np.log10(psi2)
        pFw = 4.2   # wilting at pF ~ 4.2

        # Start with wet-side reduction (Feddes-style)
        alpha = self.alpha_wet(theta, psi2=psi2)

        # Set to zero beyond wilting
        alpha[pF > pFw] = 0.0

        # Logistic decline on dry side
        rng = (pF >= pF2) & (pF < pFw)
        alpha[rng] = 1.0 / (1.0 + np.exp((pF[rng] - pF50) / s))

        # Return scalar if input was scalar
        return np.item(alpha) if alpha.size == 1 else alpha
            
    
    # def mualem_alpha(self, theta, beta=1.0, psi2=25):
    #     """Return smooth ET throttling factor acc Mualem.

    #     Parameters
    #     ----------
    #     theta: float, ndarray    
    #         moisture content        
    #     beta: tunes curvature
    #         0.7 < beta < 1.3, beta > 1 steeper, beta < 1 gentler
    #     """
    #     th_fc, th_wp = self.theta_fc(), self.theta_wp()
    #     th_suff = self.theta_fr_psi(psi2)
        
    #     theta = np.atleast_1d(theta)
    #     alpha = np.ones_like(theta)
                
    #     K_fc = max(self.K(th_fc), 1e-30)

    #     rng = (theta > th_wp) & (theta <= th_fc)
    #     alpha[rng] = (self.K(theta[rng]) / K_fc) ** beta

    #     alpha  *= self.alpha_wet(theta, psi2=psi2)

    #     # Return scalar if input was scalar
    #     return np.item(alpha) if alpha.size == 1 else alpha

    def mualem_alpha(self, theta, beta=1.0, psi2=25, psi_fc=300, psi_w=16000):
        """Return smooth ET throttling factor acc Mualem.

        Parameters
        ----------
        theta: float, ndarray    
            moisture content        
        beta: tunes curvature
            0.7 < beta < 1.3, beta > 1 steeper, beta < 1 gentler
        """        
        theta = np.atleast_1d(theta)
        psi = self.psi_fr_theta(theta)
        
        alpha = np.ones_like(psi)
                
        K_fc = max(self.K(self.theta_fr_psi(psi_fc)), 1e-30)

        rng = (psi > psi_fc) & (psi < psi_w)
        alpha[rng] = (self.K(self.theta_fr_psi(psi[rng])) / K_fc) ** beta
        alpha[psi >= psi_w] = 0

        alpha  *= self.alpha_wet(theta, psi2=psi2)

        # Return scalar if input was scalar
        return np.item(alpha) if alpha.size == 1 else alpha

    def alpha_dry_blend(self, theta, beta=1.0, w=0.6,
                        psi1=10, psi2=25, psi3=400, psi_w=16000, eps=1e-12):
        """
        Dry-side throttle: geometric blend of Mualem and Feddes.
        w=1 → pure Mualem, w=0 → pure Feddes.
        """
        psi_fc, psi_w = 300, 16000
        theta = np.atleast_1d(theta)
        th_fc, th_wp = self.theta_fc(), self.theta_wp()

        # masks
        dry = (theta > th_wp) & (theta < th_fc)

        a_mualem = self.mualem_alpha(theta, beta=beta, psi2=psi2, psi_fc=psi_fc, psi_w=psi_w)        
        a_feddes = self.feddes_alpha(theta, psi1=psi1, psi2=psi2, psi3=psi3, psi_w=psi_w)
        a_m = np.clip(a_mualem, eps, 1.0)
        a_f = np.clip(a_feddes, eps, 1.0)

        alpha = np.exp(w * np.log(a_m) +  (1 - w) * np.log(a_f))
        
        # enforce bounds
        alpha = np.clip(alpha, 0.0, 1.0)
        return alpha.item() if alpha.size == 1 else alpha


# %%
if __name__ == '__main__':
    dirs = etc.Dirs('~/GRWMODELS/python/tools/Stromingen/Munsflow_H2O_1995')

    # %%
    wbook = os.path.join(dirs.data, 'VG_soilprops.xlsx')
    Soil.load_soils(wbook) # load once
    Soil.pretty_data()
    sand_b = Soil("O01")
    sand_o = Soil("B01")

    # %% psi(theta)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set(title=r"$\psi(\theta)$", xlabel=r'$\theta$', ylabel=r'$\psi$ [cm]')
    ax.set_yscale('log')
    ax.grid(True)

    clrs = cycle("brgkmcy")
    for code in Soil.data.index:
        clr = next(clrs)
        soil = Soil(code)
        if soil.props['Hoofdsoort'] != 'Zand':
            continue
        theta = np.linspace(soil.props['theta_r'], soil.props['theta_s'])[1:-1]
        label=f"soil[{code}]:  {soil.props['Omschrijving']}" if code.endswith('01') else ""
        ax.plot(theta, soil.psi_fr_theta(theta), color=clr, label=label)
        
        pF_fc, pF_wp = 2.5, 4.2
        lbl1 = f"fc[pF={pF_fc}]" if code.endswith('01') else ""
        lbl2 = f"fc[pF={pF_wp}]" if code.endswith('01') else ""
        ax.plot(soil.theta_fc(pF=pF_fc), 10 ** pF_fc, 'o', color=clr, label=lbl1)
        ax.plot(soil.theta_wp(pF=pF_wp), 10 ** pF_wp, 's', color=clr, label=lbl2)
    ax.legend(title="Legenda voor alleen B01 en O01")


    # %% psi(theta)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set(title="retention curves for sand_b", xlabel='theta', ylabel='pF [log(cm)]')
    ax.set_yscale('log')
    ax.grid(which='both')

    theta = np.linspace(sand_b.props['theta_r'], sand_b.props['theta_s'], 100)
    psi = np.logspace(0, 4.2, 100)

    ax.plot(sand_b.theta_fr_psi(psi), psi, label="theta fr psi")
    ax.plot(theta, sand_b.psi_fr_theta(theta), '.', label="psi fr theta")
    ax.legend()


    # %% K(theta)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set(title=r"$K$ versus $\theta$", xlabel=r'$\theta$', ylabel=r'$K(\theta)$')
    ax.set_yscale('log')
    ax.grid(True)

    for code in Soil.data.index:
        soil = Soil(code)
        if soil.props['Hoofdsoort'].lower() != 'zand':
            continue
        theta = np.linspace(soil.props['theta_r'], soil.props['theta_s'])[1:-1]
        ax.plot(theta, soil.K(theta), label=f"soil[{code}]: {soil.props['Omschrijving']}")
    ax.legend()


    # %% K(theta)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set(title=r"$dK/d\theta$", xlabel=r'$\theta$', ylabel=r'$dK/d\theta$')
    ax.set_yscale('log')
    ax.set_yscale('linear')
    ax.grid(True)

    theta = np.linspace(sand_b.props['theta_r'], sand_b.props['theta_s'], 100)[1:-1]

    d = 1e-3
    dkdtheta_num = (sand_b.K(theta + d) - sand_b.K(theta - d)) / (2 * d)

    ax.plot(theta, sand_b.dK_dtheta(theta), label="analytical")
    ax.plot(theta, dkdtheta_num, '.', label="numeric")
    ax.legend()


    # %% Show field capacities and wilting points
    for BO in ['Boverngronden', 'Ondergronden']:
        title = f'Veldcapaciteit en Verwelkingspunt van {BO} van de Staringreeks'
        xlabel = 'soil code'
        ylabel = r'$\theta$'
        ax = etc.newfig(title, xlabel, ylabel)

        fcs, wps, codes = [], [], []
        for code in [c for c in Soil.data.index if c.startswith(BO[0])]:
            soil = Soil(code)
            codes.append(code)
            fcs.append(soil.theta_fc())
            wps.append(soil.theta_wp())
            
            i = int(code[1:])
            ax.text(int(code[1:]) - 1, 0.15,
                    soil.props['Omschrijving'], ha='center', rotation=90, fontsize=15, zorder=5,
                    bbox=dict(facecolor='white',
                            alpha=1,
                            edgecolor='none'))

        ax.plot(codes, np.array(fcs), 'b', label='Veldcapaciteit')
        ax.plot(codes, np.array(wps), 'r', label='Verwelkingspunt')
        ax.legend(fontsize=15, loc='lower right')

        #ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        plt.xticks(rotation=90, ha="center")
        plt.tight_layout()


    # %% K(theta) bij veldcapaciteit (pF=2.5)
    for BO in ['Boverngronden', 'Ondergronden']:
        title = f'K(theta) bij Veldcapaciteit van {BO} van de Staringreeks'
        xlabel = 'soil code'
        ylabel = 'K [cm/d]$'
        ax = etc.newfig(title, xlabel, ylabel)

        fcs, wps, kvalues, codes = [], [], [], []
        for code in [c for c in Soil.data.index if c.startswith(BO[0])]:
            soil = Soil(code)
            codes.append(code)
            K = soil.K(soil.theta_fc(pF=2.5))
            kvalues.append(K)
            
            
            i = int(code[1:])
            ax.text(int(code[1:]) - 1, 0.01,
                    soil.props['Omschrijving'], ha='center', rotation=90, fontsize=15, zorder=5,
                    bbox=dict(facecolor='white',
                            alpha=1,
                            edgecolor='none'))

        ax.plot(codes, np.array(kvalues), 'b', label='k bij velccapaciteit')    
        ax.legend(fontsize=15, loc='lower right')

        #ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        plt.xticks(rotation=90, ha="center")
        plt.tight_layout()

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
    plt.show()
    
# %%
