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
