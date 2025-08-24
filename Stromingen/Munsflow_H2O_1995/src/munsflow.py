# Munsflow, See Zwamborn (1995), Zwamborn e.a. (1995)
"""Munsflow implements linear percolation through thick unsaturated zones
by using a transfer function derived from the linearized PDE for transport
through the unsaturated zone. It was supposed by Kees Maas around 1994 and
implemented by Marette Zwamborn (1995) in her MSc and also used by Gehrels (1999) in his PhD. KIWA used it in several projects in which thicker unsaturated percolation zones
mattered.

The script below (this module) implements the different functions mathematically
describing the unsaturated zone and percolation through it using the Brooks and Corey (1966) mathematical formuation (BC). Van Genughten's (1980) formulation can also be used
but, while being somewhat more comples, does not necessarily offer netter results.

Many of the formulas and much of the theory of the unsaturated zone can be found
in the book Charbeneau (2000).
"""

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import gamma as Gamma, erfc
from scipy.stats import gamma
import pandas as pd
from itertools import cycle
import etc

# %%
dirs =     dirs = etc.Dirs('~/GRWMODELS/python/tools/Stromingen/Munsflow_H2O_1995')

# %% Define soil properties (from Charbeneau (2000), yet without their uncertainties.
class Soil():
    
    def __init__(self):
        pass
    
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
                            header=18,
                            skiprows=[1])
        
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

    @classmethod
    def pretty_data(cls):
        if cls.data is None:
            return "No soil  data loaded."
        print(cls.data.to_string()) # nicely formatted DataFrame
        
    def __str__(self):
        """Print soil properties in a readable format, starting with Main cat and Soil Texture."""
        # Order keys: start with hoofdsoort and Omschrijving
        keys = ['Main cat', 'Soil Texture'] + [k for k in self.props.index if k not in ('Main cat', 'Soil Texture')]
        
        lines = [f"Soil Texture code: {self.code}"]
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
        psi_b, lambda_ = self.props['psi_b'], self.props['lambda']
        return (psi_b / psi) ** lambda_
    
    def psi_fr_theta(self, theta):
        """Return $psi(theta)$"""
        psi_b, lambda_ = self.props['psi_b'], self.props['lambda']        
        S_ = self.S(theta)
        return psi_b * S_ ** (-1 / lambda_)
    
    def theta_fr_S(self, S, eps=1e-12):
        """Return theta(S)"""
        theta_s, theta_r = self.props['theta_s'], self.props['theta_r']
        return theta_r + (theta_s - theta_r) * np.clip(S, eps, 1 - eps)
    
    def theta_fr_psi(self, psi):
        """Return theta(psi)"""
        psi_b, lambda_ = self.props['psi_b'], self.props['lambda']
        S_ = (psi_b / psi) ** lambda_
        return self.theta_fr_S(S_)
        
    def K(self, theta):
        """Return K(theta)"""
        Ks, lambda_ = self.props['Ks'], self.props['lambda']
        S_ = self.S(theta)
        return Ks * S_ ** (3 + 2  / lambda_)
    
    def dK_dtheta(self, theta):
        """Return dK/dtheta"""
        pr = self.props
        Ks, theta_s, theta_r, lambda_ = pr['Ks'], pr['theta_s'], pr['theta_r'], pr['lambda']
        S_ = self.S(theta)
        dS_dtheta = 1 / (theta_s - theta_r)        
        return Ks * S_ ** (3 + 2  / lambda_ - 1) * dS_dtheta
        
    def dpsi_dtheta(self, theta):
        """Return dpsi/dtheta."""
        pr = self.props        
        psi_b, lambda_ = pr['psi_b'], pr['lambdan']
        S_ = self.S(theta)        
        return psi_b / lambda_ * S_ ** (1 / lambda_ - 1)
        
    
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
        
        # masks
        # dry = (theta > th_wp) & (theta < th_fc)

        a_mualem = self.mualem_alpha(theta, beta=beta, psi2=psi2, psi_fc=psi_fc, psi_w=psi_w)        
        a_feddes = self.feddes_alpha(theta, psi1=psi1, psi2=psi2, psi3=psi3, psi_w=psi_w)
        a_m = np.clip(a_mualem, eps, 1.0)
        a_f = np.clip(a_feddes, eps, 1.0)

        alpha = np.exp(w * np.log(a_m) +  (1 - w) * np.log(a_f))
        
        # enforce bounds
        alpha = np.clip(alpha, 0.0, 1.0)
        return alpha.item() if alpha.size == 1 else alpha



# %%

def get_vD_BC(q_avg):
    """Return parameters v and D for the linearized IR and SR.
    
    Using the BC relations we can alway determin thee parameters h0 and h1
    and k0 and k1 of the linearized dh/dtheta and dk/dtheta at some theta_avg.
    
    Parameters
    ----------
    q_avg:  float
        the average percolation rate.
    
    Returns
    -------
    v: the velocity of the theta wave.
    D: the Diffusivity of the theta wave.
    """
    global pr_BC
    theta_s, theta_r = pr_BC['theta_s'], pr_BC['theta_r']
    epsilon = 3 + 2 / pr_BC['lambda_'] 
     
    # k0 is the average hydraulic conductivity (beloging to average theta)
    k0 = q_avg
    
    # average theta beloning to the average conductivity
    theta_avg = theta_r + (theta_s - theta_r) * (k0 / pr_BC['K_s']) ** (1 / epsilon)
    
    # derivatives of pressure head and conductivity with respecte to theta at theta_avg   
    h1 = -dpsi_dtheta_BC(theta_avg)
    k1 = +dK_dtheta_BC(theta_avg)
    
    v = k1      # kinematic wave velocity
    D = k0 * h1 # lienarized diffusivaty
    return v, D
    

def IR(z, tau, q_avg=None):
    """Return impulse response of q_avg with the linearized convection dispersion equation for flux.
    
    $$ \frac{partial q}{partial t} = -v \frac{\partial q}{\partial z} + D \frac {\partial^2 q}{\partial z^2}$$
    
    Parameters
    ----------
    z: ndarray
        depth below the root zone
    tau: ndarray
        time
    v: float [L/T]
        vertical front velocity (first derivative of conductivity at $\theta = \theta_{avg}$)
    D: diffusivity, float [L^2/T]
        equals k0 h1, where k0 is conductivity at theta_avg and h1 is the first
        derivative ($dh/theta$) at $theta=theta_{avg}$
    q_avg: average recharge (passage through the percolation zone) [L/T]
    """
    v, D = get_vD_BC(q_avg)  
    arg = np.clip((z -  v * tau) ** 2 / (2 * D * tau), 0, 50)
    return z / np.sqrt(4 * np.pi * D * tau ** 3) * np.exp(-arg)


def SR_Phi_BC(z, tau, q_avg, clip=50):
    """Return step response of q_avg with the linearized convection dispersion equation for flux.
    
    The step response is here computed using the exact soution with
    the cumulative normal distribution.
    """    
    v, D = get_vD_BC(q_avg)
    arg1 = np.clip( (z - v * tau) / np.sqrt(2 * D * tau), -clip, clip)
    arg2 = np.clip(-(z + v * tau) / np.sqrt(2 * D * tau), -clip, clip)
    phi1 = norm.cdf(arg1)
    phi2 = norm.cdf(arg2)
    factor = np.exp(v * z / D)
    return 1 - (phi1 - factor * phi2)

def SR_erfc_BC(z, t, q_avg, clip=50):
    """Return step response of q with the linearized convection dispersion equation for flux.
    Maas (1994) eq. 3.25
    
    The step response is here computed using the exact solution with erfc's.
    """    
    v, D = get_vD_BC(q_avg)
    sqrt4Dt = np.sqrt(4.0 * D * t)
    A = np.clip((z - v * t) / sqrt4Dt, -clip, clip)
    B = np.clip((z + v * t) / sqrt4Dt, -clip, clip)
    return 0.5 * (erfc(A)  + np.exp(v * z / D) * erfc(B))


def soil_props_BC(K_s=None, theta_r=None, theta_s=None, lambda_=3.7, psi_b=46.0):
    r"""Store and return the relevant unsaturated soil properties.
    
    This function is trivial, not needed.
    
    Parameters
    ----------
    K_s: float [L/T]
        saturated hydraulic conductivity.
    theta_r: float [-]
        residual moisture content (at k = 0, field capacity)
    theta_s: float [-]
        saturated moisture content
    lambda_: float [-]
        Brooks and Corey parameter in $\Theta = (\psi / \psi_b) ^\lambda$ 
    psi_b: flat [cm]
        Air entry pressure (pressure head at theta=theta_s (top capillary zone))
    """
    return {'K_s': K_s, 'theta_r': theta_r, 'theta_s': theta_s, 'lambda_': lambda_, 'psi_b': psi_b}

def theta_from_psi_BC(psi):
    """Return theta given the matrix suction head head using BC relations.
    
    Returns theta for the given soil pr_BC in gobal pr_BC, using BC.
    """
    global pr_BC
    theta_r, theta_s, psi_b = pr_BC['theta_r'], pr_BC['theta_s'], pr_BC['psi_b']
    theta = theta_r + (theta_s - theta_r) * (psi_b / psi) ** pr_BC['lambda_']
    theta[psi < pr_BC['psi_b']] = pr_BC['theta_s']
    return theta


def psi_from_theta_BC(theta):
    """Return matrix suction head from theta using BC relations.
    
    Returns
    -------
    Matrix pressure suction head psi given theta, using BC relations.
    """
    global pr_BC
    theta_r, theta_s, psi_b = pr_BC['theta_r'], pr_BC['theta_s'], pr_BC['psi_b']
    return psi_b / ((theta - theta_r) / (theta_s - theta_r)) ** pr_BC['lambda_']
    
def dpsi_dtheta_BC(theta):
    """Return dPsi/dtheta using BC relations."""
    global pr_BC
    theta_r, theta_s, psi_b, lam = pr_BC['theta_r'], pr_BC['theta_s'], pr_BC['psi_b'], pr_BC['lambda_']
    return -psi_b / (lam * (theta_s - theta_r)) * (
        (theta - theta_r) / (theta_s - theta_r)) ** (- 1 / lam - 1)


def K_BC(theta=None):
    """Return unsaturated hydraulic conductivity according to BC relations."""
    global pr_BC
    epsilon = 3 + 2 / pr_BC['lambda_']
    K_s, theta_r, theta_s = pr_BC['K_s'], pr_BC['theta_r'], pr_BC['theta_s']
    return K_s * ((theta - theta_r) / (theta_s - theta_r)) ** epsilon

def dK_dtheta_BC(theta=None):
    """Return the first derivative with respect to theta of the hydralic conductivity according to Brooks and Corey.
    """
    global pr_BC
    epsilon = 3 + 2 / pr_BC['lambda_']
    K_s, theta_r, theta_s = pr_BC['K_s'], pr_BC['theta_r'], pr_BC['theta_s']
    return K_s / (theta_s - theta_r) * ((theta - theta_r) / (theta_s - theta_r)) ** (epsilon - 1)

def theta_from_psi_vG(psi):
    """Return theta given psi according to van Genughten."""
    global pr_vG
    theta_s, theta_r, a, n = pr_vG['theta_s'], pr_vG['theta_r'], pr_vG['alpha'], pr_vG['n']
    m = 1 - 1 / n
    theta = theta_r + (theta_s - theta_r) / (1 + (a * psi) ** n) ** m
    return theta
    
def fc_vG():
    """Return field capacity (theta at psi=100 cm - theta_r)."""
    global pr_vG
    theta_r = pr_vG['theta_r']
    psi_fc = 100 # cm
    return theta_from_psi_vG(psi_fc) - theta_r

def Kh_vG(psi):
    """Return hydraulic conductivity as function of psi, using vG."""
    global pr_vG
    Ks, a, n, lambda_ = pr_vG['Ks'], pr_vG['alpha'], pr_vG['n'], pr_vG['lambda']
    m = 1 - 1 / n   
    f = (1 + a * psi ** n) ** m
    
    return Ks * (f - a * psi ** (n - 1)) ** 2 / f ** (lambda_ + 2)

def PIII_BC(z, tau, q_avg):
    """Return Peason III distribution based on the moments.
    
    Parameters
    ----------
    z: ndarray or float
        depth below the root zone.
    tau: ndarray or float
        time.
    q_avg: float
        mean percolation rate.
    """
    v, D = get_vD_BC(q_avg)
    
    z = -z
    
    M1 = - z / v                   # first moment
    M2 = - 2 * z * D / v ** 3      # second central moment
    M3 = -12 * z * D ** 2 / v ** 5 # third central moment
    
    a = 2 * M2 / M3
    b = M1 - 2 * M2 ** 2 / M3
    n = 4 * M2 ** 3 / M3 ** 2
    
    # check
    # aa = v ** 2 / (3 * D)
    # bb = - z / (3 * v)
    # nn = 2 * z * v / (9 * D)    
    # print(a, aa)
    # print(b, bb)
    # print(n, nn)
    
    pIII = np.zeros_like(tau)
    pIII[tau >= b] = (a ** n * (tau[tau >= b] - b) ** (n - 1) / Gamma(n) 
                 * np.exp(-a * (tau[tau >= b] - b)))
    return pIII

def IR_PIII_BC(z, tau, q_avg):
    return PIII_BC(z, tau, q_avg) # equivalent)


def SR_PIII_BC(z, tau, q_avg):
    """Pierson III als step response
    
    The integration of IR_PIII_BC is known and is astandard function,
    which is used here.
    """
    v, D = get_vD_BC(q_avg)
    
    z = -z
    
    M1 = - z / v                   # first moment
    M2 = - 2 * z * D / v ** 3      # second central moment
    M3 = -12 * z * D ** 2 / v ** 5 # third central moment
    
    a = 2 * M2 / M3
    b = M1 - 2 * M2 ** 2 / M3
    n = 4 * M2 ** 3 / M3 ** 2
    
    F = np.zeros_like(tau)
    F[t >= b] = gamma.cdf(tau[tau >= b] - b, a=n, scale=1/a)
    return F
    
def BR(sr_func, z, tau, q_avg):
    """Return the block response using step response function sr_func.
    
    Parameters
    ----------
    sr_func: function
        step response function (must take z, t, q_avg)
    z: float [m] or ndarray
        depth
    t: ndarray (starting with 0) [d]
        time
    q_avg: float [m/d]
        average recharge
    """
    BR = np.hstack((0., sr_func(z, tau[1:], q_avg)))
    BR[1:] -= BR[:-1]
    return BR

# %%
    
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("K and dKdtheta for Brooks and Corey")
ax.set_xlabel('theta [-]')
ax.set_ylabel('K and dKdtheta')
ax.grid(True)
    
clrs = cycle('brgkmc')
for soil_nm in ['sand', 'sandy loam', 'loam', 'silt']:
    clr = next(clrs)
    pr_BC = soil_props_BC(**soils.loc[soil_nm])
    theta = np.linspace(pr_BC['theta_r'], pr_BC['theta_s'])
    ax.plot(theta, K_BC(theta), '-', color=clr, label=f'K, {soil_nm}')
    ax.plot(theta, dK_dtheta_BC(theta),'--', color=clr, label=f'dK/dtheta, {soil_nm}')

ax.legend(loc='upper left')
#plt.show()

# %% -dPsi/dTheta
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("- dpsi/dtheta for Brooks and Corey")
ax.set_xlabel('theta [-]')
ax.set_ylabel('-dPsi/dtheta')
ax.grid(True)
ax.set_yscale('log')

clrs = cycle('brgkmc')
for soil_nm in ['sand', 'sandy loam', 'loam', 'silt']:
    clr = next(clrs)
    pr_BC = soil_props_BC(**soils.loc[soil_nm])
    theta = np.linspace(pr_BC['theta_r'], pr_BC['theta_s'])
    ax.plot(theta, -dpsi_dtheta_BC(theta), '-', color=clr, label=f'K, {soil_nm}')

ax.legend(loc='upper right')
#plt.show()

# %% Effect of parameters of the different soil types
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("Psi for Brooks and Corey lambda={lambda_:.3f}")
ax.set_xlabel('S [-]')
ax.set_ylabel('pF')    
ax.grid(True)
ax.set_xlim(0, 1.2)
ax.set_ylim(0, 4.5)

pF =np.linspace(-2, 4.2, 101)
psi = 10 ** pF / 100. # To meters

for soil_nm in ['sand', 'loamy sand', 'sandy loam', 'loam', 'silt']:
    pr_BC = soil_props_BC(**soils.loc[soil_nm])
    theta = theta_from_psi_BC(psi)
    S = theta / pr_BC['theta_s']
    ax.plot(S, pF, label=r"{}, n={:.3f}, $\theta_r$={:.3f}, $\psi_b$={:.3f} m, $\lambda$={:.3f}".format(
        soil_nm, pr_BC['theta_s'], pr_BC['theta_r'], pr_BC['psi_b'], pr_BC['lambda_']))

ax.legend()
#plt.show()

# %% Chabeneau (2000) fig. 4.4.3, showing the test-type equal to the one used by charbeneau in fig. 4.4.3
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title("Psi for Brooks and Corey lambda={lambda_:.3f}, see Charbeneau (2000) fig. 4.4.3")
ax.set_xlabel('S [-]')
ax.set_ylabel('negative pressure head psi [m]')    
ax.grid(True)
ax.set_xlim(0, 1.2)
ax.set_ylim(0, 2.00)

pr_BC['psi_b']  = 46.
pF =np.linspace(-2, 4.2, 101)
psi = 10 ** pF / 100. # To meters
for soil_nm in ['sand', 'sandy loam', 'loam']:
    pr_BC = soil_props_BC(**soils.loc[soil_nm])
    theta = theta_from_psi_BC(psi)
    S = theta / pr_BC['theta_s']
    ax.plot(S, psi, label=r"{}, n={:.3f}, $\theta_r$={:.3f}, $\psi_b$={:.3f} m, $\lambda$={:.3f}".format(
        soil_nm, pr_BC['theta_s'], pr_BC['theta_r'], pr_BC['psi_b'], pr_BC['lambda_']))
ax.legend()
#plt.show()

# %% Using sand, we will compute the analytic IR

q_avg = 0.07 # m/d
t = np.linspace(0, 200, 201)[1:]

v, D = get_vD_BC(q_avg)

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title(f"IR voor q_avg={q_avg:.3g} m/d, v={v:.3g}, D={D:.3g} m2/d")
ax.set_xlabel("t")
ax.set_ylabel("unit response")
ax.grid(True)

for z in [5, 10, 20, 30]:

    for soil_nm in ['sand']:
        pr_BC = soil_props_BC(**soils.loc[soil_nm])        
        ax.plot(t, IR_PIII_BC(z, t, q_avg), label=f'soil_nm, z={z:.3g} moments')
ax.legend(loc='best')
# plt.show()

# %% Using sand, we will compute the analytic SR

q_avg = 0.07 # m/d
t = np.linspace(0, 200, 201)[1:]

v, D = get_vD_BC(q_avg)

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title(f"SR voor q_avg={q_avg:.3g} m/d, v={v:.3g}, D={D:.3g} m2/d")
ax.set_xlabel("t")
ax.set_ylabel("unit response")
ax.grid(True)

for z in [5, 10, 20, 30]:

    for soil_nm in ['sand']:
        pr_BC = soil_props_BC(**soils.loc[soil_nm])                            
        ax.plot(t, SR_Phi_BC(z, t, q_avg), 'x', label=f'soil_nm, z={z:.3g} m, IR_Phi')
        ax.plot(t, SR_erfc_BC(z, t, q_avg), '.', label=f'soil_nm, z={z:.3g} m, IR_erfc')              
        ax.plot(t, SR_PIII_BC(z, t, q_avg), label=f'soil_nm, z={z:.3g} m, momenten')
        
ax.legend(loc='best')
plt.show()

# %%
# %% Using sand, we will compute the analytic SR

q_avg = 0.07 # m/d
t = np.linspace(0, 200, 201)[1:]

v, D = get_vD_BC(q_avg)

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title(f"SR voor q_avg={q_avg:.3g} m/d, v={v:.3g}, D={D:.3g} m2/d")
ax.set_xlabel("t")
ax.set_ylabel("block response")
ax.grid(True)

for z in [5, 10, 20, 30]:

    for soil_nm in ['sand']:
        pr_BC = soil_props_BC(**soils.loc[soil_nm])
        ax.plot(t, BR(SR_Phi_BC, z, t, q_avg),  '-', label=f'soil_nm, z={z:.3g} m, BR_phi')
        ax.plot(t, BR(SR_erfc_BC, z, t, q_avg), '.', label=f'soil_nm, z={z:.3g} m, BR_erfc')                         
        ax.plot(t, BR(SR_PIII_BC, z, t, q_avg), '+', label=f'soil_nm, z={z:.3g} m, BR_mom')
        
ax.legend(loc='best')
plt.show()


# %%
