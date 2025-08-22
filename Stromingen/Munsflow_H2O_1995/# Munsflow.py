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
from pathlib import Path

# %%
class Dirs():
    def __init__(self):
        self.home = '/Users/Theo/GRWMODELS/python/tools/Stromingen/Munsflow_H2O_1995'
        self.docs = os.path.join(self.home, 'docs')
        self.images = os.path.join(self.home, 'images')
        self.data = os.path.join(self.home, 'data')

dirs = Dirs()
# %% Define soil properties (from Charbeneau (2000), yet without their uncertainties.

# turn typed array into a pd.DataFrame
soils_dtype = np.dtype([('soil', 'U15'), ('theta_s', '<f8'), ('theta_r', '<f8'), ('K_s', '<f8'), ('psi_b', '<f8'), ('lambda_', '<f8')])

soils = np.zeros(10, dtype=soils_dtype)
soils_ = np.array([('clay'           , 0.38, 0.068, 0.048 , 1.25 , 0.09),
                  ('clay loam'      , 0.41, 0.095, 0.062 , 0.53 , 0.31),
                  ('loam'           , 0.43, 0.078, 0.25  , 0.28 , 0.56),
                  ('loamy sand'     , 0.41, 0.057, 3.5   , 0.081, 1.28),
                  ('silt'           , 0.46, 0.034, 0.060 , 0.62 , 0.37),
                  ('silty loam'     , 0.45, 0.067, 0.11  , 0.50 , 0.41),
                  ('silty clay loam', 0.36, 0.070, 0.0048, 2.0  , 0.09),
                  ('silty clay'     , 0.43, 0.089, 0.017 , 1.0  , 0.23),
                  ('sand'           , 0.43, 0.045, 7.1   , 0.069, 1.68),
                  ('sandy clay'     , 0.38, 0.100, 0.029 , 0.37 , 0.23),
                  ('sandy clay loam', 0.39, 0.100, 0.31  , 0.17 , 0.48),
                  ('sandy loam'     , 0.41, 0.065, 1.1   , 0.13 , 0.89),
                  ('test'           , 0.43, 0.045, 7.1   , 0.41 , 3.7),
                  ], dtype=soils_dtype)

soils = pd.DataFrame(soils_, index=soils_['soil']).drop('soil', axis=1)


# %% Read the Van Genughten soil properties for the 18 Dutch soils

# The dimensions of these data are
vG_dims={'theta_r':'L3/L3', 'theta_s':'L3/L3', 'alpha':'1/cm', 'n':'-', 'lambda':'-', 'Ks':'cm/d'}

wbook = os.path.join(dirs.docs, 'VG_soilprops.xlsx')
props = pd.read_excel(wbook, sheet_name='props',
              usecols='A:I',
              dtype={'a':str, 'b':str, 'c':float, 'd':float, 'e':float, 'f':float, 'g':float, 'h':float, 'i':str},
              header=0,
              index_col=1,
              skiprows=[1])

# %%

def get_vD_BC(q_avg):
    """Return parameters v and D for the linearized IR and SR.
    
    Using the BC relations we can alway determin the parameters h0 and h1
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
