"""Compute recharge using KNMI data.

This module computes recharge from KNMI weather data, using three different methods:
1. Makkink method: uses the Makkink evapotranspiration without considering interception and root zone storage.
2. Bin method: evaporation is only cut off when the root zone becomes empty.
3. Earth method: evaporation is pinched according to the root storage filling fraction, as described in the PhD thesis of JC Gehrels (1999).
It also computes the recharge for a given period in the year, allowing comparison of recharge amounts between successive years.

It is used to analyze the recharge in the Hilversum area, particularly for the Peilbuis14 piezometer.

This module requires the `tools` and `etc` modules for directory management and color cycling, respectively.

It is designed to be run as a script, with the main functionality encapsulated in the `recharge` function and the `rch_yr_period` function for yearly recharge analysis.

It also includes a section to visualize the results using matplotlib, showing the recharge and groundwater levels over time.

It uses hilv_proxy data, which is a cleaned version of the De Bilt weather data, stored in a CSV file.

It stores the results in a specified directory for further analysis or reporting.
"""

# %%
import os
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import etc
from itertools import cycle 


dirs = etc.Dirs()
os.chdir(dirs.python)
print("os.getcwd()\n", os.getcwd())

# %% Recharge function

class RechargeBase(ABC):

    month_names = 'jan feb mar apr may jun jul aug sep okt nov dec'.split()
    
    def __init__(self, Smax_I: float = 2.5, Smax_R: float = 2.5)-> float:
        self.Smax_I = Smax_I
        self.SMax_R = Smax_R
        self.S_I = 0.
        self.S_R = 0.
        
    def interception(self, P: float, E: float)-> float:
        """Accept P, E0 and return q, E1
        
        Parameters
        ----------
        P: float
            Precipitation
        E0: float
            Potential evaporation

        Returns                
        -------
        q: float
            downward outflow of inerception (fall-thorugh)
        E1: float
            Left over of E0 after evaporation of intercepted water
        """
        S0, Smax = self.S_I, self.Smax_I
        dt = self.dt
        S = S0 + (P- E) * dt
        if S > Smax:                            
            q = (S - Smax) / dt
            S = self.Smax    
        elif S < 0:
            S = 0.
            q = 0
        else:
            q = 0
        E1 = P - (S - S0) / dt - q
        self.S_I = S
        return q, E1

        
    # May be overwritten
    @abstractmethod
    def root_zone(self, P1: float, E1: float)-> float:
        """Accept P1, E1 and return Sto"""
        pass
    
    def check_data(self, PE: pd.DataFrame)-> np.ndarray:
        """Verify input before simulation."""
           
        # Checks for data consistency

        # Required columns present?
        missing_columns = {'RH', 'EV24'} - set(pd.columns)
        if missing_columns:
            raise KeyError("Missing columns {missing_columns}")
        
        # index in np.datetime64 time stamp format?
        if not isinstance(PE.index[0], np.datetime64):
            raise TypeError("Index of wheather series should be np.datetime64")

        # Data in mm/d?
        if PE['EV24'].mean() > 0.01: # 1 cm/d
            raise ValueError("Date probably not in m/d.")
       
       # Initialyze the storage
        self.S_I = self.Smax_I
        self.S_R = self.Smax_R / 2.
        
        # Get the time step length in days
        self.dt = np.diff(PE.index) / np.timedelta64(1, 'D')

        PE = PE.loc[:, ['RH']].to_records()
        
        dtype = np.dtype([('t', np.datatime64), ('RH', float), ('EV24', float), ('STO', float),
                          ('IC', float), ('EA', float), 'RCH', float]) 
       
        Out = np.zeros(len(PE), dtype=dtype)
        Out['t']  = PE.index
        Out['RH'] = PE['RH'].values
        Out['EV24'] = PE['EV24'].values
        return Out

    
    def simulate(self, PE: pd.DataFrame)-> np.ndarray:
        """Return result of the simulation.
        
        PE is a pandas DataFrame with recharge in column ['RH']
        and evaporation in column ['EV24']. This data generally
        comes from downloading from the KNMI weather service.
        
        Convert the index column to np.datetime64.
        Convert the columns 'RH' and 'EV24' to m/d.
                            
        We assume E0 = EV24 / 0.8

        # Te following columns will be added or replaced:
        PE['STO'] = 0. # Current root-zone zone storage [mm]
        PE['IC' ] = 0. # Actual interception [mm/d]
        PE['EA' ] = 0. # Actual evapotranspiration [mm/d]
        PE['RCH'] = 0. # Actual recharge [mm/d]
        """
       
        # Out is empty recarray with the proper fields
        Out = self.check_data(PE)
            
        for pe in Out: # dt = datetime
            
            P0 = pe['RH']
            E0 = pe['EV24'] / 0.8 # Open water evap. during assumed 12h daytime

            P1, E1 = self.interception(P0, E0)
            rch, ea = self.root_zone(  P1, E1)
            
            pe['IC']  = pe['RH'] - P1
            pe['STO'] = self.S_R
            pe['EA']  = ea
            pe['RCH'] = rch
               
        Out = np.DataFrame(Out)
        return Out

class RchMak(RechargeBase):
        """Compute recharge without interception and rootzone.
        Given E0  =(Emak / 0.8) as input Emx = 0.8 E0
    """
    
        def __init__(self, Smax_I: float = 2.5, Smax_R: float = 2.5)-> float:
            super().init(Smax_I, Smax_R)
            
        def interception(self, P: float, E0: float)-> float:
            """Return fall through and left over of E0 after interception"""
            # Just return the intput, no interception used
            Emak = 0.8 * E0
            return P, Emak
            
        def root_zone(self, P: float, E: float)-> float:
            """Accept P, E and return q and EA"""  
            # Just return the intput, no root zone used
            return P, E

    
class RchBin(RechargeBase):
    """Compute recharge with interception and root_zone storage assuming
    that the evapotranspiration is not throttled with S/Sm..
    """

    
    def __init__(self, Smax_I: float = 2.5, Smax_R: float = 2.5)-> float:
            super().init(Smax_I, Smax_R)
            
    def root_zone(self, P: float, E: float)-> float:
        """Accept P, E and return Sto
        
        Parameters
        ----------
        P: float
            "recharge" (output from the interceptiion model)
        E: float
            Potential evaporation (left over from interception model)

        Returns                
        -------
        q: float
            downward outflow of root_zone
        EA: float
            Actual evaporation from root zone
        """
        S0, Smax = self.S_R, self.Smax_R          
        dt = self.dt
        S = S0 + (P- E) * dt
        if S > Smax:                            
            q = (S - Smax) / dt
            S = self.Smax    
        elif S < 0:
            S = 0.
            q = 0
        else:
            q = 0
        Ea = P - (S - S0) / dt - q
        self.S_R = S
        return q, Ea

class RchEarth(RechargeBase):
    """Compute recharge with interception and root_zone storage assuming
    that the evapotranspiration from th eroot zone is reduced by factor S/Smax.
                
    Decay of storage in root zone (during 12 h).
    exp(-2 Emax / Sto * 0.5 dtau) = exp(- Emax / Sto * tau), answer is the same for 12 and 24 h
    decay = np.exp(-(Emax / STOmax) * dtau) # Storage decay by evap. during daytime            
    pe['EA']  = pe['STO'] * (1 - decay) # Actual crop evap.
    """
    
    def __init__(self, Smax_I: float = 2.5, Smax_R: float = 2.5)-> float:    
        super().__init__(Smax_I, Smax_R)
        
        
    def root_zone(self, P, E):
        """Decay of storage in root zone (during 12 h).
        
        exp(-2 Emax / Sto * 0.5 dtau) = exp(- Emax / Sto * tau)
        
        answer is the same for 12 and 24 h
        
        decay = np.exp(-(Emax / STOmax) * dtau)
        
        Storage decay by evap. during daytime            
        
        pe['EA']  = pe['STO'] * (1 - decay) # Actual crop evap.
        """
        S0, Smax = self.Smax_R
        S = Smax * (P / E+ (S0 / Smax - P / E ) * np.exp(- E / Smax * self.dt))
        if S >Smax:
            q = (S - Smax) / self.dt
            S = Smax
        else:
            q = 0.
        Ea = P - (S - S0) / self.dt - q
        self.S_R = S
        return q, Ea


class RchRoot(RechargeBase):
    """Compute recharge with interception and root_zone storage assuming
    that the evapotranspiration is reduced by factor sqrt(S/Smax).
                
    Decay of storage in root zone (during 12 h).
    exp(-2 Emax / Sto * 0.5 dtau) = exp(- Emax / Sto * tau), answer is the same for 12 and 24 h
    decay = np.exp(-(Emax / STOmax) * dtau) # Storage decay by evap. during daytime            
    pe['EA']  = pe['STO'] * (1 - decay) # Actual crop evap.
    """
    
    def __init__(self, Smax_I: float = 2.5, Smax_R: float = 2.5)-> float:    
        super().__init__(Smax_I, Smax_R)
        
    def newS(p, e, dt, S0, Sm):
        x0 = np.sqrt(S0 / Sm)
        z0 = 1 - (e/p) * np.sqrt(x0)
        arg = z0 * e ** z0 * np.exp(e ** 2 /(2 * p) * dt)
        z = scipy.special.lambertw(arg)
        x = ((p - (1 - z)) / 2) ** 2
        
        
    def root_zone(self, P, E):
        """Decay of storage in root zone (during 12 h).
        
        exp(-2 Emax / Sto * 0.5 dtau) = exp(- Emax / Sto * tau)
        
        answer is the same for 12 and 24 h
        
        decay = np.exp(-(Emax / STOmax) * dtau)
        
        Storage decay by evap. during daytime            
        
        pe['EA']  = pe['STO'] * (1 - decay) # Actual crop evap.
        """
        S0, Smax = self.Smax_R
        def z(t) = scipy.special.lambertW(z0 * e ** z0 * np.exp(e ** 2 / (2 * p) * dt))
        x = (p - (1 - z(t)) / e) ** 2
        S = Smax * (P / E+ (S0 / Smax - P / E ) * np.exp(- E / Smax * self.dt))
        if S >Smax:
            q = (S - Smax) / self.dt
            S = Smax
        else:
            q = 0.
        Ea = P - (S - S0) / self.dt - q
        self.S_R = S
        return q, Ea




def rch_yr_period(PE, period=(10, 7), verbose=False):
    """Return recharge during given period in year, for subsequent years in the database.
    
    It aims to compare recharge amounts between successive years and find extreme years.
    
    Parameters
    ----------
    PE: pd.DataFrame with recharge, datetime index and field ['RCH'] [mm/d]
        The recharge
    period: tuple of two ints.
        start month and and month in next year.
    verbose: bool
        print results
        
    Returns
    -------
    rec_array with dtype=[('start', np.datetime64) ('end', np.datetime64), ('rch, float)]
    """
    years = np.arange(PE.index[0].year, PE.index[-1].year)

    dtype=np.dtype([('start', np.datetime64(None, 'D')),
                    ('end', np.datetime64(None, 'D')),
                    ('rch', float)])
    
    r_period = np.zeros(len(years), dtype=dtype)
        
    for i, yr1 in enumerate(years):
        yr2 = yr1 if period[1] > period[0] else yr1 + 1
        start = np.datetime64(f"{yr1}-{period[0]:02d}-01", 'D')
        end   = np.datetime64(f"{yr2}-{period[1]:02d}-01", 'D')
        idx = PE.index[np.logical_and(PE.index >= start, PE.index <= end)]
        
        r_period[i] = (start, end, np.asarray(PE.loc[idx, ['RCH']], dtype=float).sum())
    
    if verbose:
        print("Jaarlijks neerslagoverschot in gegeven periode [mm]")
        print(r_period)
        
    return r_period


if __name__ == "__main__":
    # %% Compute and show recharge computed from De Bilt data with three different methods
    # This is the main part of the script, which computes recharge and visualizes it.
    # It is the same as in the main block, but without the `if __name__ == '__main__':` guard.

    Smax_R  = 100 # mm
    Smax_I  = 2.5
    
    # Get De Bilt weather data (previously stored as a cleaned csv file)
    stn = 'HilvProxy' # 'De Bilt260'
    
    PE = pd.read_csv(os.path.join(dirs.weer, stn + ".csv"), index_col=0, parse_dates=True)


    # Compute recharge by each of the three methods
    rchMakkink = RchMak(Smax_I, Smax_R)
    rchBin     = RchBin(Smax_I, Smax_R)
    rchEarth   = RchEarth(Smax_R, Smax_I)

    rch_mak = rchMakkink.simulate(PE)
    rch_bin = rchBin.simulate(PE)
    rch_ear = rchEarth.simulate(PE)

    # Show the results for a given period
    # Add the measured head in Peilbuis14 near Monnikenberg 
    start, end = np.datetime64("2020-01-01"), np.datetime64("2024-08-31")
    index = PE.index[np.logical_and(PE.index >= start, PE.index <= end)]

    fig, axs = plt.subplots(4, 1, sharex=True, sharey=False, gridspec_kw={'hspace': 0.3})
    fig.set_size_inches(8, 11.5)
    fig.suptitle(f'\n\nStation {stn}, STOmax = {Smax_R} mm, ICmax = {Smax_I} mm/d')

    methods = ['Makkink', 'Bin', 'Earth']
    for i, method, rch , df in zip(range(3), methods, [rch_mak, rch_bin, rch_ear]):
        RH, EV24, EA, STO, RCH, IC = df.loc[index, ['RH', 'EV24', 'EA', 'STO', 'RCH', 'IC']].mean()

        title = (f"\nRecharge, method = {method}\n" + 
                f"P={RH:.2f}, EM={EV24:.2f} EA={EA:.2f} RCH={RCH:.2f}, IC={IC:.2f} all mm/d, STO={STO:.1f} mm ")

        axs[i].set_title(title)
        axs[i].set_ylabel("mm/d")
        axs[i].grid()
        axs[i].plot(df.index, df['RCH'], label=f"N={RCH:.2f} mm/d")
        axs[i].plot(df.index, df['STO'], label=f"STO={STO:.1f} mm")
        axs[i].plot(df.index, df['IC'] , label=f"Ic={IC:.2f} mm/d")
        axs[i].legend()

    # Add the registered logged head data for Peilbuis 14

    # Load previously pickled head data for all availabel Hilversum piezometers
    head_data = tools.load_object(os.path.join(dirs.grw, 'head_data'))

    pz_name = 'Peilbuis14'  
    pz = head_data.piez[pz_name]['log']

    # pz.to_csv(os.path.join(dirs.data,'peilbuis14.csv'))

    # Plot head Peilbuis14 on last axes
    axs[-1].set_title(f"Grondwaterstandsverloop {pz_name}")
    axs[-1].set_ylabel("m +NAP")
    axs[-1].grid()
    axs[-1].plot(pz.index, pz['LoggerHead'], label=pz_name)
    axs[-1].legend()

    # Limit view of axes to desired period
    axs[-1].set_xlim(start, end)

    print('Recharge according to Makkink method:\n', rch_mak.loc[index].mean(), '\n')
    print('Recharge according to BIN method:\n', rch_bin.loc[index].mean(), '\n')
    print('Recharge according to EARTH method:\n', rch_ear.loc[index].mean(), '\n')

    fig.savefig(os.path.join(dirs.lyx, f'rch_{stn}.png'))

    plt.show()

    print('Recharge according to Makkink method:\n', rch_mak.loc[index].mean(), '\n')
    print('Recharge according to BIN method:\n', rch_bin.loc[index].mean(), '\n')
    print('Recharge according to EARTH method:\n', rch_ear.loc[index].mean(), '\n')
    

    
# %%

def f(x, p, r, lam):
    return p - r * x**lam

def f_derivatives_at_x(x, p, r, lam):
    # returns f, f1, f2  where
    # f = f(x), f1 = f'(x), f2 = f''(x)
    f0 = p - r * x**lam
    f1 = - r * lam * x**(lam - 1)
    f2 = - r * lam * (lam - 1) * x**(lam - 2)
    return f0, f1, f2

def taylor_step3(x0, h, p, r, lam, xmin=1e-6):
    """
    third-order Taylor step for dx/dt = p - r x^lambda
    x0: current x (must be > xmin)
    h: step
    returns x1 (approx x(t0+h))
    """
    if x0 <= xmin:
        raise ValueError("x0 too small for safe Taylor expansion; increase xmin or use implicit method")
    f0, f1, f2 = f_derivatives_at_x(x0, p, r, lam)
    # time derivatives along solution:
    x1p = f0                              # x'
    x2p = f1 * f0                         # x''
    x3p = f2 * f0**2 + (f1**2) * f0       # x''' (as derived earlier)
    x1 = x0 + h*x1p + 0.5*h**2 * x2p + (1.0/6.0)*h**3 * x3p
    return x1

# Example usage:
p, r, lam = 0.5, 1.2, 0.5
x0 = 0.2
h = 0.01
x1 = taylor_step3(x0, h, p, r, lam, xmin=1e-4)
print(x1)

# %%
# Example usage:
p, r, lam = 0.0, 1.2, 0.5
x0 = 0.5
t = np.linspace(0, 1)

fig, ax = plt.subplots()
for lam in [1, 0.7, 0.5, 0.3, 0.1]:
    x1 = taylor_step3(x0, t, p, r, lam, xmin=1e-4)
    ax.plot(t, x1, label=f"lambda = {lam}")
ax.legend()

# %%
# Example usage, copute the derivative = p - r * x ** lambda:
p, r, lam = 0.0, 1.2, 0.5
x0 = 0.5
t = np.linspace(0,1)
dt = 0.001

fig, ax = plt.subplots()
ax.set_title("$dx/dt = p - e x^\lambda$ and numerically $x(t+dt) - x(t))/ dt$")
ax.set_xlabel("t")
ax.set_ylabel('dx/dt')
ax.grid(True)

clrs = cycle('rbgkmcy')
for lam in [1, 0.7, 0.5, 0.3, 0.1]:
    clr = next(clrs)
    x1 = taylor_step3(x0, t,      p, r, lam, xmin=1e-4)
    x2 = taylor_step3(x0, t + dt, p, r, lam, xmin=1e-4)
    
    ax.plot(t, p - r * x1 ** lam, '-', color=clr, label=fr"$dx/dt = p - e x^\lambda$, $\lambda={lam}$")
    ax.plot(t, (x2 - x1) / dt,'-.', color=clr,  label=fr"$(x(t + dt) - x(t) / dt, \lambda={lam}$" )
ax.legend()


# %%
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from scipy.integrate import solve_ivp

# Problem: dx/dt = p - r * x**lam
def f(t, x, p, r, lam):
    return p - r * x**lam

# Derivatives of f wrt x at point x (needed for Taylor time-derivs)
def f_x_derivs(x, p, r, lam):
    # returns f, f1, f2
    f0 = p - r * x**lam
    f1 = - r * lam * x**(lam - 1)
    f2 = - r * lam * (lam - 1) * x**(lam - 2)
    return f0, f1, f2

def taylor_step3_single(x0, h, p, r, lam, xmin=1e-12):
    if x0 <= xmin:
        raise ValueError("x0 too small for Taylor expansion")
    f0, f1, f2 = f_x_derivs(x0, p, r, lam)
    x1p = f0                    # x'
    x2p = f1 * f0               # x''
    x3p = f2 * f0**2 + (f1**2) * f0  # x'''
    x_new = x0 + h*x1p + 0.5*h**2 * x2p + (1.0/6.0)*h**3 * x3p
    return x_new

def integrate_taylor3(x0, t_span, n_steps, p, r, lam):
    t0, t1 = t_span
    ts = np.linspace(t0, t1, n_steps+1)
    xs = np.empty_like(ts)
    xs[0] = x0
    for i in range(n_steps):
        h = ts[i+1] - ts[i]
        xs[i+1] = taylor_step3_single(xs[i], h, p, r, lam)
    return ts, xs

# Example parameters
p, r = 0.0, 0.25
x0 = 0.4
t_span = (0.0, 1.5)

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_title(r'Comparison: $\dot x = p - r x^\lambda$, $\lambda=0.5$')
ax.set(xlabel='t', ylabel='x(t)')
ax.grid(True)

clrs = cycle('rbgkmcy')
for lam in [1.0, 0.75, 0.5, 0.25, 0.1]:
    clr = next(clrs)
    
    # integrate with Taylor (many small steps)
    # ts_taylor, xs_taylor = integrate_taylor3(x0, t_span, n_steps=1000, p=p, r=r, lam=lam)

    # integrate with RK45 and BDF
    sol_rk = solve_ivp(lambda t, x: f(t, x, p, r, lam), t_span, [x0], method='RK45', rtol=1e-8, atol=1e-10, dense_output=True)
    # sol_bdf = solve_ivp(lambda t, x: f(t, x, p, r, lam), t_span, [x0], method='BDF', rtol=1e-8, atol=1e-10, dense_output=True)

    # compare on a common time grid
    N = 50
    t_plot = np.linspace(t_span[0], t_span[1], N)

    x_rk = sol_rk.sol(t_plot)[0]
    # x_bdf = sol_bdf.sol(t_plot)[0]
    x_taylor_interp = np.interp(t_plot, ts_taylor, xs_taylor)

    plt.plot(t_plot, x_rk, 'o', color=clr, mfc='none', label=fr'RK45, $\lambda$={lam}')
    # plt.plot(t_plot, x_bdf, '--', color=clr, label=fr'BDF, $\lambda$={lam}')
    # plt.plot(t_plot, x_taylor_interp, '.', color=clr, label=fr'Taylor3 ({N} steps), $\lambda$={lam}')
ax.legend()
plt.show()

# %%

import numpy as np
from scipy.special import lambertw

def x_explicit(t, p, r, x0, t0=0.0):
    """
    Explicit solution for dx/dt = p - r * sqrt(x).
    Accepts scalar or numpy array t.
    Handles p > 0 via Lambert W, and p == 0 via separable formula.
    """
    t = np.asarray(t, dtype=float)
    # trivial constant solution if r == 0 -> x = x0 + p*(t-t0)  (but here r>0 usually)
    if np.isclose(p, 0.0):
        # separable solution: sqrt{x}(t) = sqrt{x0} - (r/2)*(t-t0)
        s = np.sqrt(x0) - 0.5 * r * (t - t0)
        s = np.maximum(s, 0.0)   # if hit zero, floor at zero
        return s**2

    # general p > 0 case (Lambert W formula)
    sqrt_x0 = np.sqrt(x0)
    y0 = p - r * sqrt_x0
    # Build A(t) = (y0/p) * exp(-y0/p) * exp(- (r^2/(2p))*(t-t0))
    A = (y0 / p) * np.exp(-y0 / p) * np.exp(-(r**2 / (2.0 * p)) * (t - t0))
    arg = -A
    # lambertw returns complex arrays in general; for our regime principal branch real part suffices
    W = lambertw(arg, k=0)
    W = np.real_if_close(W, tol=1000)   # convert small imag parts to real
    z = -W.real
    u = (p * (1.0 - z)) / r
    x = (u ** 2)
    # numerical safety: ensure non-negative
    x = np.maximum(x, 0.0)
    return x

# %%
p, x0 = 1.0, 0.9
t = np.linspace(0, 10)

fig, ax = plt.subplots(figsize=(8,6))
ax.set_title(r'x=S/Smax for different p and r, function with power=0.5')
ax.set(xlabel='t', ylabel='S/Smax')
ax.grid(True)

for r in [0., 0.1, 0.2, 0.3, 0.5]:
    x = x_explicit(t, p, r, x0, t0=0.0)
    ax.plot(t, x, label=f"p={p}, r={r}")
ax.legend()

# %%

def implicit_euler_step(xn, dt, p, r, lam, tol=1e-12, maxit=10):
    # Solve G(x) = x - xn - dt*(p - r*x**lam) = 0
    x = xn  # initial guess
    for _ in range(maxit):
        G = x - xn - dt*(p - r * x**lam)
        if abs(G) < tol:
            break
        Gp = 1 + dt * r * lam * x**(lam - 1)   # derivative of G
        dx = - G / Gp
        x += dx
        if abs(dx) < tol:
            break
    return max(x, 0.0)



# %%
p, x0 = 0.1, 0.9
t = np.linspace(0, 30)

fig, ax = plt.subplots(figsize=(8,6))
ax.set_title(r'x=S/Smax  implicit euler method')
ax.set(xlabel='t', ylabel='S/Smax')
ax.grid(True)

for lam in [0.1]:
    clrs = cycle('rbgkmcy')
    for r in [0., 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]:
        xt = np.zeros_like(t)
        for i, dt in enumerate(t):
            xt[i] = implicit_euler_step(x0, dt, p, r, lam)
        ax.plot(t, xt, label=f"p={p}, r={r}")
ax.legend()

# %%
