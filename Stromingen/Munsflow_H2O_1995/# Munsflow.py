# Munsflow

# Munsflow, zie Zwamborn (1995), Zwamborn e.a. (1995)

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import gamma as Gamma, erfc
from scipy.stats import gamma
import pandas as pd
from itertools import cycle

# %% Impulse response

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

# %%

def get_vD(q_avg):
    """Return parameters for the IR and SR.
    """
    global props
    theta_s, theta_r = props['theta_s'], props['theta_r']
    epsilon = 3 + 2 / props['lambda_'] 
     
    k0 = q_avg
    theta_avg = theta_r + (theta_s - theta_r) * (k0 / props['K_s']) ** (1 / epsilon)    
    h1 = -dpsi_dtheta(theta_avg)
    k1 = +dK_BC_dtheta(theta_avg)
    v = k1    
    D = k0 * h1
    return v, D
    

def IR(z, t, q_avg=None):
    """Return impuls response of q with the linearized convection dispersion equation for flux.
    
    $$ \frac{partial q}{partial t} = -v \frac{\partial q}{\partial z} + D \frac {\partial^2 q}{\partial z^2}$$
    
    Parameters
    ----------
    z: ndarray
        depth below the root zone
    t: ndarray
        time
    v: float [L/T]
        vertical front velocity (first derivative of conductivity at $\theta = \theta_{avg}$)
    D: diffusivity, float [L^2/T]
        equals k0 h1, where k0 is conductivity as average moisture content and h1 is the first
        derivative ($dh/theta$) at $theta-theta_{avg}$
    q_avg: average recharge (passage through the percolation zone) [L/T]
    """
    v, D = get_vD(q_avg)  
    arg = np.clip((z -  v * t) ** 2 / (2 * D * t), 0, 50)
    return z / np.sqrt(4 * np.pi * D * t ** 3) * np.exp(-arg)


def SR_Phi(z, t, q_avg, clip=50):
    """Return step response of q with the linearized convection dispersion equation for flux.
    """    
    v, D = get_vD(q_avg)
    arg1 = np.clip( (z - v * t) / np.sqrt(2 * D * t), -clip, clip)
    arg2 = np.clip(-(z + v * t) / np.sqrt(2 * D * t), -clip, clip)
    phi1 = norm.cdf(arg1)
    phi2 = norm.cdf(arg2)
    factor = np.exp(v * z / D)
    return 1 - (phi1 - factor * phi2)

def SR_erfc(z, t, q_avg, clip=50):
    """Return step response of q with the linearized convection dispersion equation for flux.
    Maas (1994) eq. 3.25
    """    
    v, D = get_vD(q_avg)
    sqrt4Dt = np.sqrt(4.0 * D * t)
    A = np.clip((z - v * t) / sqrt4Dt, -clip, clip)
    B = np.clip((z + v * t) / sqrt4Dt, -clip, clip)
    return 0.5 * (erfc(A)  + np.exp(v * z / D) * erfc(B))


def soil_props(K_s=None, theta_r=None, theta_s=None, lambda_=3.7, psi_b=46.0):
    r"""Store and return the relavant unsaturated soil properties.
    
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

def theta_from_psi(psi):
    """Return pressure head from theta.
    """
    global props
    theta_r, theta_s, psi_b = props['theta_r'], props['theta_s'], props['psi_b']
    theta = theta_r + (theta_s - theta_r) * (psi_b / psi) ** props['lambda_']
    theta[psi < props['psi_b']] = props['theta_s']
    return theta

def psi_from_theta(theta):
    """Return pressure head from theta.
    """
    global props
    theta_r, theta_s, psi_b = props['theta_r'], props['theta_s'], props['psi_b']
    return psi_b / ((theta - theta_r) / (theta_s - theta_r)) ** props['lambda_']
    
def dpsi_dtheta(theta):
    global props
    theta_r, theta_s, psi_b, lam = props['theta_r'], props['theta_s'], props['psi_b'], props['lambda_']
    return -psi_b / (lam * (theta_s - theta_r)) * (
        (theta - theta_r) / (theta_s - theta_r)) ** (- 1 / lam - 1)


def K_BC(theta=None):
    """Return unsaturated hydraulic conductivity according to Brooks and Corey."""
    global props
    epsilon = 3 + 2 / props['lambda_']
    K_s, theta_r, theta_s = props['K_s'], props['theta_r'], props['theta_s']
    return K_s * ((theta - theta_r) / (theta_s - theta_r)) ** epsilon

def dK_BC_dtheta(theta=None):
    """Return the first derivative with respect to theta of the hydralic conductivity according to Brooks and Corey.
    """
    global props
    epsilon = 3 + 2 / props['lambda_']
    K_s, theta_r, theta_s = props['K_s'], props['theta_r'], props['theta_s']
    return K_s / (theta_s - theta_r) * ((theta - theta_r) / (theta_s - theta_r)) ** (epsilon - 1)


def PIII(z, t, q_avg):
    """Return Peason III distribution based on the moments.
    """
    v, D = get_vD(q_avg)
    
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
    
    F = np.zeros_like(t)
    F[t >= b] = a ** n * (t[t >= b] - b) ** (n - 1) / gamma(n) * np.exp(-a * (t[t >= b] - b))
    return F

def IR_PIII(z, t, q_avg):
    """Return Peason III distribution based on the moments.
    """
    v, D = get_vD(q_avg)
    
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
    
    F = np.zeros_like(t)
    F[t >= b] = a ** n * (t[t >= b] - b) ** (n - 1) / Gamma(n) * np.exp(-a * (t[t >= b] - b))
    return F

def SR_PIII(z, t, q_avg):
    """Pierson III als step response"""
    v, D = get_vD(q_avg)
    
    z = -z
    
    M1 = - z / v                   # first moment
    M2 = - 2 * z * D / v ** 3      # second central moment
    M3 = -12 * z * D ** 2 / v ** 5 # third central moment
    
    a = 2 * M2 / M3
    b = M1 - 2 * M2 ** 2 / M3
    n = 4 * M2 ** 3 / M3 ** 2
    
    F = np.zeros_like(t)
    F[t >= b] = gamma.cdf(t[t >= b] - b, a=n, scale=1/a)
    return F
    
def BR(sr_func, z, t, q_avg):
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
    BR = np.hstack((0., sr_func(z, t[1:], q_avg)))
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
    props = soil_props(**soils.loc[soil_nm])
    theta = np.linspace(props['theta_r'], props['theta_s'])
    ax.plot(theta, K_BC(theta), '-', color=clr, label=f'K, {soil_nm}')
    ax.plot(theta, dK_BC_dtheta(theta),'--', color=clr, label=f'dK/dtheta, {soil_nm}')

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
    props = soil_props(**soils.loc[soil_nm])
    theta = np.linspace(props['theta_r'], props['theta_s'])
    ax.plot(theta, -dpsi_dtheta(theta), '-', color=clr, label=f'K, {soil_nm}')

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
    props = soil_props(**soils.loc[soil_nm])
    theta = theta_from_psi(psi)
    S = theta / props['theta_s']
    ax.plot(S, pF, label=r"{}, n={:.3f}, $\theta_r$={:.3f}, $\psi_b$={:.3f} m, $\lambda$={:.3f}".format(
        soil_nm, props['theta_s'], props['theta_r'], props['psi_b'], props['lambda_']))

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

props['psi_b']  = 46.
pF =np.linspace(-2, 4.2, 101)
psi = 10 ** pF / 100. # To meters
for soil_nm in ['sand', 'sandy loam', 'loam']:
    props = soil_props(**soils.loc[soil_nm])
    theta = theta_from_psi(psi)
    S = theta / props['theta_s']
    ax.plot(S, psi, label=r"{}, n={:.3f}, $\theta_r$={:.3f}, $\psi_b$={:.3f} m, $\lambda$={:.3f}".format(
        soil_nm, props['theta_s'], props['theta_r'], props['psi_b'], props['lambda_']))
ax.legend()
#plt.show()

# %% Using sand, we will compute the analytic IR

q_avg = 0.07 # m/d
t = np.linspace(0, 200, 201)[1:]

v, D = get_vD(q_avg)

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title(f"IR voor q_avg={q_avg:.3g} m/d, v={v:.3g}, D={D:.3g} m2/d")
ax.set_xlabel("t")
ax.set_ylabel("unit response")
ax.grid(True)

for z in [5, 10, 20, 30]:

    for soil_nm in ['sand']:
        props = soil_props(**soils.loc[soil_nm])        
        ax.plot(t, IR_PIII(z, t, q_avg), label=f'soil_nm, z={z:.3g} moments')
ax.legend(loc='best')
# plt.show()

# %% Using sand, we will compute the analytic SR

q_avg = 0.07 # m/d
t = np.linspace(0, 200, 201)[1:]

v, D = get_vD(q_avg)

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title(f"SR voor q_avg={q_avg:.3g} m/d, v={v:.3g}, D={D:.3g} m2/d")
ax.set_xlabel("t")
ax.set_ylabel("unit response")
ax.grid(True)

for z in [5, 10, 20, 30]:

    for soil_nm in ['sand']:
        props = soil_props(**soils.loc[soil_nm])                            
        ax.plot(t, SR_Phi(z, t, q_avg), 'x', label=f'soil_nm, z={z:.3g} m, IR_Phi')
        ax.plot(t, SR_erfc(z, t, q_avg), '.', label=f'soil_nm, z={z:.3g} m, IR_erfc')              
        ax.plot(t, SR_PIII(z, t, q_avg), label=f'soil_nm, z={z:.3g} m, momenten')
        
ax.legend(loc='best')
plt.show()

# %%
# %% Using sand, we will compute the analytic SR

q_avg = 0.07 # m/d
t = np.linspace(0, 200, 201)[1:]

v, D = get_vD(q_avg)

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_title(f"SR voor q_avg={q_avg:.3g} m/d, v={v:.3g}, D={D:.3g} m2/d")
ax.set_xlabel("t")
ax.set_ylabel("block response")
ax.grid(True)

for z in [5, 10, 20, 30]:

    for soil_nm in ['sand']:
        props = soil_props(**soils.loc[soil_nm])
        ax.plot(t, BR(SR_Phi, z, t, q_avg),  '-', label=f'soil_nm, z={z:.3g} m, BR_phi')
        ax.plot(t, BR(SR_erfc, z, t, q_avg), '.', label=f'soil_nm, z={z:.3g} m, BR_erfc')                         
        ax.plot(t, BR(SR_PIII, z, t, q_avg), '+', label=f'soil_nm, z={z:.3g} m, BR_mom')
        
ax.legend(loc='best')
plt.show()


# %%
