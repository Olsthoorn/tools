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
import sys

sys.path.insert(0, os.getcwd())

from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
import etc
from itertools import cycle
from typing import Any

# Project directory namespace
dirs = etc.Dirs(os.getcwd())

# Verify that project directory is the current working directory
# Start VScode from the project directory to realize this.
print("os.getcwd()\n", os.getcwd())


# %% Recharge function

class RechargeBase(ABC):

    month_names = 'jan feb mar apr may jun jul aug sep okt nov dec'.split()
    
    def __init__(self, Smax_I: float = 2.5, Smax_R: float = 100., **_: Any)-> None:
        self.Smax_I = Smax_I
        self.Smax_R = Smax_R
        self.S_I = 0.
        self.S_R = 0.
            
    def dt_hit_interception(self, P: float, E: float)->float:
        """Return S, t, point where S hits top or bottom of  reservoir."""
        Smin, Smax, Dt = 0., self.Smax_I, self.dt
        
        S0 = self.S_I # current interception storage
        
        # New guessed reservoir contents update
        S1 = S0 + (P - E) * Dt

        # Get S, dt
        if np.isclose(P, E): # Nothing changes
            return S1, Dt
        elif P > E:          # Reservoir fills up and may overflow at Dt < dt
            dt = (Smax  - S0) / (P - E)
            if dt <= Dt:     # Moment when reservoir becamse full.
                return Smax, dt
            else:
                return S1, Dt
        elif P < E:          # Reservoir is emptied
            dt = (Smin - S0) / (P - E) # Moment when reservoir became empty
            if dt <= Dt:
                return Smin, dt
            else:            # Reservoir neither overflows not empties
                return S1, Dt 
        else:
            raise ValueError("Didn't expect to get here in t_hit")
     
     
    def interception(self, P0: float, E0: float)-> float:
        """Accept P0, E0 and return q, E1
        
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
            Evaporaton from interception resrvoir
        """
        Dt = self.dt
        S0 = self.S_I
                
        S, dt1 = self.dt_hit_interception(P0, E0)
        
        self.S_I = S
        
        dt1 = min(dt1, Dt)
        dt2 = max(Dt - dt1, 0)

        if np.isclose(P0, E0):
            q = 0.
            E1 = E0
            
        if P0 > E0:
            q = (P0 - E0) * dt2 / Dt
            E1 = P0 - (S - S0 ) / Dt - q
            
        elif P0 < E0:
            q = 0.
            E1 = P0 - (S - S0 ) / Dt - q
            
        B = P0 - E1 - (S - S0) / Dt - q
            
        return q, E1, B

        
    # May be overwritten
    @abstractmethod
    def S_rootzone(self, P: float, E: float, Dt:float)-> float:
        """Compute the root zone filling at the dt = t - t0."""
        pass
    
    @abstractmethod    
    def dt_hit_root_zone(self, P: float, E: float, Dt: float)->float:
        """Return S, t, point where S hits top or bottom of  root-zone reservoir."""
        pass
    
    
    def root_zone(self, P: float, E: float)-> float:
        """Accept P, E and return q, E1
        
        Parameters
        ----------
        P: float
            Precipitation
        E: float
            Potential evaporation after interception

        Returns                
        -------
        q: float
            downward outflow from root zone
        E1: float
            Evapo-transpiration from root zonen reservoir
        """
        Dt = self.dt
        S0 = self.S_R
                
        S, dt1 = self.dt_hit_root_zone(P, E, Dt)
        
        self.S_R = S # The new value at Dt
        
        dt2 = Dt - dt1 # Always >= 0 after dt_hit_root_zone
        
        if np.isclose(dt2, 0):
            q = 0   
        elif P >= E:            
            q = (P - E) * dt2 / Dt
        else:
            q = 0
        
        E1 = P - (S - S0 ) / Dt - q         
                
        # Budget:
        B = P - E1 - (S - S0) - q 
        
        return q, E1, B

    
    def check_data(self, PE: pd.DataFrame)-> np.ndarray:
        """Verify input before simulation and prepare Out.
        
        Parameters
        ----------
        PE: pd.DataFrame with fields ['RH', 'EV24']  in mm/d
            Meteo input, daily values of precipition and E acc. to Makkink.
        
        Returns
        -------
        Out: np.ndarray with given dtype
            See fields in dtype definition.
            The index of PE is included in the field 't'
        """
           
        # Checks for data consistency

        # Required columns present?
        missing_columns = {'RH', 'EV24'} - set(PE.columns)
        if missing_columns:
            raise KeyError("Missing columns {missing_columns}")
        
        # index in np.datetime64 time stamp format?
        if not isinstance(PE.index[0], (pd.Timestamp, np.datetime64)):
            raise TypeError("Index of wheather series should be np.datetime64")

        # Data in mm/d?
        # if PE['EV24'].mean() > 0.01: # 1 cm/d
        #     raise ValueError("Date probably not in m/d.")        
        if PE['EV24'].mean() < 0.01:
            raise ValueError("Date probably not in mm/d.")        
       
       # Initialyze the storage
        self.S_I = self.Smax_I
        self.S_R = self.Smax_R / 2.
        
        # Get the time step length in days
        self.dt = (np.diff(PE.index) / np.timedelta64(1, 'D'))[0]

        PE = PE.loc[:, ['RH', 'EV24']].to_records()
        
        # Don't need the index, its kept
        dtype = np.dtype([('t', 'datetime64[ns]'), ('RH', float), ('EV24', float), ('STO', float),
                          ('IC', float), ('EA', float), ('RCH', float), ('BI', float), ('BR', float)]) 
       
        # Prepare the output array with specified dtype, that also holds the intput and timestamps
        Out = np.zeros(len(PE), dtype=dtype)  
        Out['t']  = PE.index      
        Out['RH'] = PE['RH']
        Out['EV24'] = PE['EV24']
        return Out

    
    def simulate(self, PE: pd.DataFrame)-> np.ndarray:
        """Return result of the simulation.
        
        PE is a pandas DataFrame with recharge in column ['RH']
        and evaporation in column ['EV24']. This data generally
        comes from downloading from the KNMI weather service.
        
        Convert the index column to np.datetime64.
        Convert the columns 'RH' and 'EV24' to m/d.
                            
        We assume E0 = EV24 / 0.8 ??

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
            E0 = pe['EV24']    # Open water evap. during assumed 12h daytime (/0.8?)
            if P0 > 20:
                pass
            P1, E1, BI = self.interception(P0, E0)
            rch, Ea, BR = self.root_zone(  P1, E0 - E1)
            
            if Ea > 15:
                pass
            
            pe['IC']  = pe['RH'] - P1  # intercepted
            pe['STO'] = self.S_R       # Current storage level
            pe['EA']  = Ea             # Actual expo-transpiration
            pe['RCH'] = rch            # Recharge (to percolation zone)
            pe['BI'] = BI              # Budget Interception
            pe['BR'] = BR              # Budget root zone
               
        Out = pd.DataFrame(data=Out, index=Out['t']).drop(['t'], axis=1)
        return Out

# %% Recharge according to P - Emakkink
class RchMak(RechargeBase):
        """Compute recharge without interception and rootzone.
        Given E0  =(Emak / 0.8) as input Emx = 0.8 E0 ??
        The recharge must be P - 0.8 E0 ??
        Where E0 = E_makkink/0.8. ??
    """
        def __init__(self, Smax_I: float = 2.5, Smax_R: float = 100., **_: Any)-> None:
            super().__init__(Smax_I, Smax_R)
            
        def dt_hit_root_zone(self, P: float, E: float, Dt: float)->float:
            """Return Smax, dt, does nothing in this model."""
            return self.Smax_R, Dt

        def S_rootzone(self, P: float, E: float, Dt: float)->float:
            """Return the rootzone reservoir filling. Does nothing in this model."""
            S = self.S_R
            return S
            
        def interception(self, P: float, E0: float)-> float:
            """Return fall through and left over of E0 after interception"""
            # Just return the intput, no interception used            
            return P, E0
            
# %% Recharge with interception and a root-zone bin without E throttling until empty
class RchBin(RechargeBase):
    """Compute recharge with interception and root_zone storage assuming
    that the evapotranspiration is not throttled.
    """
    def __init__(self, Smax_I: float = 2.5, Smax_R: float = 100., **_: Any)-> None:
            super().__init__(Smax_I, Smax_R)
            
    def S_rootzone(self, P: float, E: float, Dt: float)->float:
        """Return rootzone filling after arbirtrary time Dt"""      
        return self.S_R + (P- E) * Dt
    
    def dt_hit_root_zone(self, P: float, E: float, Dt: float)->float:
        """Return S, t, point where S hits top or bottom of  root-zone reservoir."""
        Smin, Smax= 0., self.Smax_R
        
        S0 = self.S_R # current interception storage
        
        # New guessed reservoir contents update
        S1 = self.S_rootzone(P, E, Dt)

        # Get S, dt
        if np.isclose(P, E): # Nothing changes
            return S1, Dt
        elif P > E:          # Reservoir fills up and may overflow at Dt < dt
            dt = (Smax  - S0) / (P - E)
            if dt <= Dt:     # Moment when reservoir becamse full.
                return Smax, dt
            else:
                return S1, Dt
        elif P < E:          # Reservoir is emptied
            dt = (Smin - S0) / (P - E) # Moment when reservoir became empty
            if dt <= Dt:
                return Smin, dt
            else:            # Reservoir neither overflows not empties
                return S1, Dt 
        else:
            raise ValueError("Didn't expect to get here in t_hit")


# %% Recharge with interception and linear throttling of E
class RchEarth(RechargeBase):
    """Compute recharge with interception and root_zone storage assuming
    that the evapotranspiration from th eroot zone is reduced by factor S/Smax.
                
    Decay of storage in root zone (during 12 h).
    exp(-2 Emax / Sto * 0.5 dtau) = exp(- Emax / Sto * tau), answer is the same for 12 and 24 h
    decay = np.exp(-(Emax / STOmax) * dtau) # Storage decay by evap. during daytime            
    pe['EA']  = pe['STO'] * (1 - decay) # Actual crop evap.
    """
    
    def __init__(self, Smax_I: float = 2.5, Smax_R: float = 100., **_: Any)-> None:    
        super().__init__(Smax_I, Smax_R)
        return
        
    def S_rootzone(self, P: float, E: float, Dt:float)-> float:
        """Compute the reservoir filling after arabitrary time Dt"""
        
        S0, Smax = self.S_R, self.Smax_R
        p, r, x0 = P / Smax, E / Smax, S0 /Smax
        
        if np.isclose(E, 0):
            x = x0 + p * Dt
        else:            
            x = p / r - (p / r - x0) * np.exp(- r * Dt)
        return Smax * x
    
    def dt_hit_root_zone(self, P: float, E: float, Dt:float)->float:
        """Return S, t, point where S hits top or bottom of  root-zone reservoir."""
        S0, Smax = self.S_R, self.Smax_R
        
        S1 = self.S_rootzone(P, E, Dt)
        
        # Get S, dt
        if np.isclose(P, E): # Nothing changes
            return S1, Dt
        elif np.isclose(E, 0):
            dt = (Smax - S0) / P
            if dt < Dt:
                return Smax, dt
            else:
                return S1, Dt
        elif P < E:
            return S1, Dt
        else:
            dt = Smax / E * np.log((P - E * S0 / Smax) / (P - E))
            if dt < Dt:
                return Smax, dt
            else:
                return S1, Dt


# %% Recharge with throttlling of E according to root S/Smax
    
class RchLam(RechargeBase):
    """Compute recharge with interception and root_zone storage assuming
    that the evapotranspiration is reduced by factor (S/Smax)^lambda.
    
    Lambda is set by hand in function self.dt_hit_root_zone).
    Make sure lambda is not much smaller than 0.5, because the
    stiffness of the solution near the empty reservoir make the
    solver fail.

    lambda = 1 yields the same reults as the RchEarth object.
    """
    def __init__(self, Smax_I: float = 2.5, Smax_R: float = 100., lam: float = 0.5, **_: Any)-> None:    
        super().__init__(Smax_I, Smax_R)
        
        self.lam = lam
        
        self.event_y_eq_1.terminal = True
        self.event_y_eq_0.terminal = True
        
        self.event_y_eq_1.direction = +1
        self.event_y_eq_0.direction = -1
        
        self.events = [self.event_y_eq_1, self.event_y_eq_0]
                
    @staticmethod
    def rhs(t, x, p, r, lam):
        return p - r * (x ** lam)

    eps = 1e-6

    @staticmethod
    def event_y_eq_1(t, y, *_):
        return y[0] - (1.0 - RchLam.eps)
    
    @staticmethod
    def event_y_eq_0(t, y, *_):
        return y[0] - RchLam.eps
    
            
    def S_rootzone(self, P: float, E:float, Dt: float)->  float:        
        """Return relative filling of reservoir using exact method.        
        """
        return 0.

    def dt_hit_root_zone(self, P: float, E: float, Dt: float)->float:        
        """Return t when reservoir hits a target filling x0.
        """
        S0, Smax = self.S_R, self.Smax_R        
        p, r, x0 = P / Smax, E / Smax, S0 / Smax
                
        small_tol = 0.1 / Smax
        xbase, xtop = small_tol, 1 - small_tol
        

        xacc = p - r * x0 ** self.lam
        if x0 > xtop:
            if xacc >= 0:
                return Smax, 0               
        elif x0 < xbase:
            if xacc <= 0:
                return 0., Dt
            else:
                x0 = xbase
        
        sol = solve_ivp(self.rhs, [0, Dt], [x0], method='RK45', args=(p,r,self.lam),
                events=self.events, dense_output=True)
        
        if sol.success:
            if sol.t_events[0]:                
                x1 = sol.y_events[0].ravel()[-1]
                dt = sol.t_events[0].ravel()[-1]
            else:
                x1 = sol.y.ravel()[-1]
                dt = sol.t.ravel()[-1]
            return Smax * x1, dt                
        else:
            raise RuntimeError("Runge Kutta did not finish successfully exit status {soi.status}")

    @staticmethod        
    def show_throttling_E_by_lam(lam=[0.1, 0.25, 0.5, 1.0]):
        """Show the throttling of E by lambda: E = E0 (S/Smax) ** lambda
        
        lam = float | iterable of float
        """
        
        x = np.linspace(0., 1.0, 200)
        lambdas = np.atleast_1d(np.asarray(lam))
        
        ax = etc.newfig(r"Efect of throttling E by $\lambda$: ($E = E_0 \cdot x^\lambda$)",
                        r"$x = S/S_{max}$",
                        r"$x^\lambda$")
        
        for lam in lambdas:
            ax.plot(x, x ** lam, label=fr"\ambda = {lam:3g}")
            
        ax.legend(loc="lower right")
        return ax
        
        
    def show_effect_of_lam(self, x0=[0.5], PE=((2, 1)), lam=0.5, t=1.0, ax=None):
        """Show x(t) for different p and r and given lambda

        Parameters
        ----------
        t: float
            Duration of simulation (1 = 1 day) which will be from 0 to t
        x0: ones-dimensional with len equal to that of PE:
            initial relative filling of the root zone reservoir, eg 0.5.
        PE: listlike of 2-tuples with (P, E) in mm/d
            Given tupbles of precipiation and potential evaporation
        lam: float
            power in dS/dt = P/Smax - E/Smax (S/Smax) ** lam, the velocity of reservoir filling
            
        rchSimulator = RchLam(deBilt, lam=0.3)      
        """
        PE = np.atleast_2d(np.asarray(PE))
        pr = PE / self.Smax_R
        if np.isscalar(x0):
            x0 = x0 * np.ones(len(pr))
        if len(x0) != len(pr):
            raise ValueError("Len of x0 must equal len of 2d sequence PE.")
        
        sol = solve_ivp(self.rhs, [0, t], x0, method='RK45', args=(pr[:, 0], pr[:, 1], lam),
                        events=self.event_y_eq_0)
        
        if ax is None:
            ax = etc.newfig(
            fr"Show effect for $\lambda$ of different $P$ and $E$, [Smax={self.Smax_R} mm]",
                            "time [d]",
                            r"$x=S/S_{max}$")
        
        if True: # sol.success:
            events = len(sol.t_events[0]) > 0
            clrs = cycle("rbgkmcy")
            for i, ((p, r), (P, E)) in enumerate(zip(pr, PE)):
                clr = next(clrs)
                ax.plot(sol.t, sol.y[i], '-.', color=clr,
                        label=fr"$\lambda$={lam:.1f}, P={P:6.3g} mm/d , E={E:6.3g} mm/d")
                if events:
                    ax.plot(sol.t_events, t.y_events[i], 'o', color=clr, mfc='none')
        else:
            raise RuntimeError("Runge Kutta did not finish successfully exit status {soi.status}")
        ax.legend(loc="upper left")
        return ax
    
def change_meteo(meteo):
    """Meteo is a pd.DataFrame with fields 'RH' and 'EV24', in mm/d.
    
    For exploration purposes, to easier see that happens.
    
    """
    E = cycle([0., 1., 0., 2., 0., 3., 0., 4., 0., 5])
    P = cycle([0., 0., 2., 2., 0., 0., 4., 4., 0., 0.])
    i1 = 0
    meteo = meteo.copy()
    for _ in range(1000):
        i2 = i1 + 30
        if i2 > len(meteo):
            break
        idx = meteo.index
        meteo.loc[idx[i1]:idx[i2], 'RH']   = next(P)
        meteo.loc[idx[i1]:idx[i2], 'EV24'] = next(E)
        i1 = i2
    return meteo
    
    

    return meteo
# %% Recharge over a period in the year
def rch_yr_period(PE: pd.DataFrame, period: tuple=(10, 7), verbose: bool =False)->pd.DataFrame:
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

# %%
if __name__ == "__main__":

    # Get the meteodata in a pd.DataFrame
    meteo_csv = os.path.join(dirs.data, "DeBilt.csv")
    os.path.isfile(meteo_csv)
    deBilt = pd.read_csv(meteo_csv, header=0, parse_dates=True, index_col=0)
    deBilt_short = deBilt.loc[deBilt.index >= np.datetime64("2020-01-01"), :]
    
    # deBilt_short = change_meteo(deBilt_short)
    
    # Storage capacity
    Smax_I, Smax_R = 1.5, 100 # mm
    
    date_span = (np.datetime64("2020-01-01"), np.datetime64("2021-03-31"))
    date_span = (deBilt_short.index[0], deBilt_short.index[-1])
    
    clrs = cycle('rbgkmcy')
    
    title = "Recharge computed by different methods"
    idx = ((deBilt_short.index >= date_span[0]) &
                               (deBilt_short.index <= date_span[1]))

    labels = ['Makkink', 'bin', 'earth', 'lambda'][1:]
    rchClasses = [RchMak, RchBin, RchEarth, RchLam][1:]
    
    if True:
        ax1, ax2, ax3 = etc.newfigs(('EV24 and EA', 'P and RCH', 'STO'),
                'time', ('mm/d', 'mm/d', 'mm' ), figsize=(12, 10))

        lam = 0.25
        
        for rchClass, label in zip(rchClasses, labels):
            rch_simulator = rchClass(Smax_I=Smax_I, Smax_R=Smax_R, lam=lam)
            rch = rch_simulator.simulate(deBilt_short)
            
            clr = next(clrs)
            ax2.plot(rch.index[idx], rch['RH'][idx], '-',  lw=0.25, color=next(clrs),  label=label + ' P')
            ax1.plot(rch.index[idx], rch['EV24'][idx], '-',lw=0.25, color=next(clrs),  label=label + ' EV24')
            ax1.plot(rch.index[idx], rch['EA' ][idx], '-', lw=0.75, color=next(clrs),  label=label + ' EA')
            ax2.plot(rch.index[idx], rch['RCH'][idx], '-', lw=0.75, color=next(clrs),  label=label + 'RCH')
            ax3.plot(rch.index[idx], rch['STO'][idx], '-', lw=0.75, color=next(clrs),  label=label + ' STO')        
            print(f'Recharge according to method {label}:\n', rch.loc[idx].mean(), '\n')

        for ax in [ax1, ax2, ax3]:
            ax.legend(loc='upper right')    
        
        # fig.savefig(os.path.join(dirs.images, f'rch_{stn}.png'))

        plt.show()
        
    if False:
        rchlam = RchLam()
        
        rchlam.show_throttling_E_by_lam(lam=[0.1, 0.25, 0.5, 1.0])
        
        PE = [(0, 0), (1, 0), (0, 1), (1, 1), (4, 3.5)]
        
        ax = None
        for lam in [0.1, 0.25, 0.5, 1.0]:
            ax = rchlam.show_effect_of_lam(x0=0.5, PE=PE, lam=lam, t=250.0, ax=ax)
        
        plt.show()

# %%
