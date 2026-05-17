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

print("sys.executable =", sys.executable)
print("sys.path =")
for p in sys.path:
    print(p)
print()

from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd
from tools.etc import etc, descr, logo
from itertools import cycle
from typing import Any
from pathlib import Path

class Dirs():
    """Namespace project directories."""
    def __init__(self, home=None):
        if home is None:
            home = '/Users/Theo/Development/python/hydro_tools/tools/root_zone'
        self.home = home
        self.data = os.path.join(self.home, 'data')
        self.images = os.path.join(self.home, 'images')
        
dirs = Dirs()


# %% Recharge function

class RechargeBase(ABC):

    month_names = 'jan feb mar apr may jun jul aug sep okt nov dec'.split()
    
    def __init__(self, Smax_I: float=None, Smax_R: float=None, Scrit_R: float=None, **_: Any)-> None:
        """Sets reservoir capacities [mm]. The Scrit_R is only used in the
        Earth model. It's the root zone capacity below which evaporation
        will be linearly reduced. It's set to np.Inf to exclude its use.
        
        TO 2026-06-14
        """
        self.Smax_I = Smax_I # --- Interception reservoir capacity [mm]
        self.Smax_R = Smax_R # --- Root_zone reservoir capacity [mm]
        self.Scrit_R = Scrit_R # --- Root_zone critical capacity [mm] in Earthmodel 
        self.S_I = 0. # --- Interception reservoir initial capacity [mm]
        self.S_R = 0. # --- Root zone reservoir initial capacity [mm]

     
    def interception(self, S: float, P: float, E: float, Dt: float)-> float:
        """Accept P, E and return S, q, EA
        
        Parameters
        ----------
        S: float
            Filling of the interception reservoir [mm]
        P: float
            Precipitation [mm/d]
        E: float
            Potential evaporation [mm/d]
        Dt: float
            Time step

        Returns                
        -------
        S: float
            filling of reservoir at end of time step
        q: float
            downward outflow of inerception (fall-thruogh)
            averaged over the time step.
        EA: float
            Actual evaporaton from interception reservoir
        """
        # --- Current interception storage level        
        S0, Smin, Smax, q = S, 0., self.Smax_I, 0.
        Dt = self.Dt
        
        def EA(P, S, S0, q, Dt):
            # --- Actual evporation from interception reservoir
            return np.round(P - (S - S0) / Dt - q, 4)
        
        if np.isclose(Smin, Smax): # --- No interception basin
            # --- pass on S0, P and E
            return S0, P, 0.
                    
        if np.isclose(P, E): # --- Nothing happens
            return S0, q, E            
                    
        if P > E: # --- Fills
            dt = (Smax - S0) / (P - E)
            if dt < Dt: # -- Full
                dt2 = Dt - dt
                q = (P - E) * dt2 / Dt
                return Smax, q, EA(P, Smax, S0, q, Dt)
            else: # --- Stuck in the middle
                S += (P - E) * Dt
                return S, q, EA(P, S,  S0, q, Dt)
            
        elif P < E: # --- Empties
            dt = (Smin- S0) / (P - E)
            if dt < Dt: # --- Empty
                return Smin, q, EA(P, Smin, S0, q, Dt)
            else:
                S += (P - E) * Dt
                return S, q, EA(P, S, S0, q, Dt)
        else:
            raise ValueError("Should never have gotten here.")

        
    # May be overwritten
    @abstractmethod
    def root_zone(self, S: float, P: float, E: float, Dt:float)-> tuple:
        """Compute the root zone filling at the dt = t - t0.
        
        Returns
        -------
        S, q, EA
        
        """
        pass
            
    
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
           
        # --- Required columns present?
        missing_columns = {'RH', 'EV24'} - set(PE.columns)
        if missing_columns:
            raise KeyError("Missing columns {missing_columns}")
        
        # --- Is index in np.datetime64 of pd.Timestamp format?
        if not isinstance(PE.index[0], (pd.Timestamp, np.datetime64)):
            raise TypeError("Index of wheather series should be np.datetime64")

        # --- Are data in mm/d?
        #     raise ValueError("Date probably not in m/d.")        
        if PE['EV24'].mean() < 0.01:
            raise ValueError("Date probably not in mm/d.")        
       
        # --- Initialyze the interception and root zone storage
        if self.Smax_I is not None:
           self.S_I = self.Smax_I / 2        
        if self.Smax_R is not None:
            self.S_R = self.Smax_R / 2.
        
        # --- Get the time step length in days
        self.Dt = (np.diff(PE.index) / np.timedelta64(1, 'D'))[0]

        # --- Convert pd.DataFrame to numpy record array
        PE = PE.loc[:, ['RH', 'EV24']].to_records()
        
        # --- Don't need the index, its kept
        dtype = np.dtype([('t', 'datetime64[ns]'), ('Dt', float),
                          ('RH', float), ('EV24', float), ('STO', float),
                          ('IC', float), ('EA', float), ('RCH', float), ('BI', float), ('BR', float)]) 
       
        # --- Prepare the output array with a dtype
        #     that also holds the intput and the timestamps
        Out = np.zeros(len(PE), dtype=dtype)  
        Out['t']      = PE.index
        Out['Dt'][1:] = np.diff(Out['t']) / np.timedelta64(1, 'D')
        Out['Dt'][0]  = Out['Dt'][1]
        Out['RH']     = PE['RH']
        Out['EV24']   = PE['EV24']
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
        PE['Dt']  = time step length in D from PE.index
        PE['STO'] = 0. # Current root-zone zone storage [mm]
        PE['IC' ] = 0. # Actual interception [mm/d]
        PE['EA' ] = 0. # Actual evapotranspiration [mm/d]
        PE['RCH'] = 0. # Actual recharge [mm/d]
        """
        # --- Out is empty recarray with the proper dtype
        Out = self.check_data(PE)        
        
        # --- Makkink case
        if self.Smax_I is None or self.Smax_R is None:            
            Out['EA'] = Out['EV24']
            Out['RCH'] = Out['RH'] - Out['EV24']
            return pd.DataFrame(data=Out, index=Out['t']).drop(['t'], axis=1)            
        
        # --- Start values for simulation
        SI0 = self.Smax_I / 2
        SR0 = self.Smax_R / 2
            
        # --- Run of each record of the record-array
        for pe in Out: # dt = datetime
            
            # --- Pecipiation and Makkink E from pd.DataFrame
            P0 = pe['RH']
            E0 = pe['EV24']    # Open water evap. during assumed 12h daytime (/0.8?)
            Dt = pe['Dt']
            
            # --- Compute interception and rootzone changes during time step
            SI, PI,  EI = self.interception(SI0, P0, E0,      Dt)
            SR, rch, ER = self.root_zone(   SR0, PI, E0 - EI, Dt)
                        
            # --- Water budgets
            BI = P0 - EI - (SI - SI0) / Dt - PI
            BR = PI - ER - (SR - SR0) / Dt - rch
            
            SI0, SR0 = SI, SR
            
            
            # --- Store in DataFrame
            pe['IC']  = (P0 - PI) * Dt  # [mm] intercepted
            pe['STO'] = SR + SI         # [mm] Current total storage at end of time step
            pe['EA']  = (EI + ER) * Dt  # [mm] Actual evapo-transpiration
            pe['RCH'] = rch * Dt        # [mm] Recharge (to percolation zone)
            pe['BI'] = BI * Dt          # [mm] Budget Interception
            pe['BR'] = BR * Dt          # [mm] Budget root zone
               
        Out = pd.DataFrame(data=Out, index=Out['t']).drop(['t'], axis=1)
        return Out

# %% Recharge according to P - Emakkink
class RchMak(RechargeBase):
    """Compute recharge without interception and rootzone.
    Given E0  =(Emak / 0.8) as input Emx = 0.8 E0 ??
    The recharge must be P - 0.8 E0 ??
    Where E0 = E_makkink/0.8. ??
    """
    def __init__(self, **_):
        super().__init__()
        self.name = 'RchMak'
    
    def root_zone(self, S: float, P: float, E: float, Dt: float)->tuple:
        """Return the rootzone reservoir filling. Does nothing in this model."""  
        q = 0.          
        return S, q, E
            
    def interception(self, S:float, P: float, E: float)-> tuple:
        """Return fall through and left over of E0 after interception"""
        # Just return the intput, no interception used
        P1 = 0.           
        return S, P1, E
            
# %% Recharge with interception and a root-zone bin without E throttling until empty
class RchBin(RechargeBase):
    """Compute recharge with interception and root_zone storage assuming
    that the evapotranspiration is not throttled.
    """
    def __init__(self, Smax_I, Smax_R, **_):
        super().__init__(Smax_I=Smax_I, Smax_R=Smax_R)
        self.name = 'RchBin'

    def root_zone(self, S: float, P: float, E: float, Dt: float)-> tuple:
        """Accept P, E and return q, E1
        
        Parameters
        ----------
        S: float [mm]
            Fill of reservoir
        P: float [mm/d]
            Overflow from interception
        E: float [mm/d]
            Potential ET after interception
        Dt: float [d]
            Length of time step

        Returns                
        -------
        q: float [mm/d] average over Dt
            downward outflow from root zone
        EA: float [mm/d]
            E from root zone reservoir
        """                
        def EA(P, S, S0, Dt, q):
            # --- Actual evapotranspiration over the entire time step
            return np.round(P - (S - S0) / Dt - q, 4)
        
        S0, Smax, Smin, q = S, self.Smax_R, 0., 0.
        
        if np.isclose(P, E):            
            return S, q, EA(P, S, S0, Dt, q)
        
        if P > E: # --- Fills
            dt = (Smax - S) / (P - E)
            if dt < Dt: # --- Full
                dt2 = Dt - dt
                q = (P - E) * dt2 / Dt                
                return Smax, q, EA(P, Smax, S0, Dt, q)
            else:
                S += (P - E)  * Dt
                return S, q, EA(P, S, S0, Dt, q)
            
        elif P < E: # --- Empties
            dt = (Smin - S) / (P - E)
            if dt < Dt: # --- Empty
                return Smin, q, EA(P, Smin, S0, Dt, q)
            else:
                S += (P - E) * Dt
                return S, q, EA(P, S, S0, Dt, q)

        else:
            raise ValueError("Should never have gotten here.")



# %% Recharge with interception and linear throttling of E
class RchEarth(RechargeBase):
    """Compute recharge with interception and root_zone storage.
    
    The RchEarth model assumes E is not restraint as long as SR>Scrit_R
    and linearly restraint when SR< Scrit_R.
    """
    def __init__(self, Smax_I, Smax_R, Scrit_R, **_):
        super().__init__(Smax_I=Smax_I, Smax_R=Smax_R, Scrit_R=Scrit_R)
        self.name = 'RchEarth'
        
        self.Scrit_R = max(0, min(self.Scrit_R, self.Smax_R))
                
    def rootzone_top(self, S: float, P: float, E: float, Dt:float)-> float:
        """Compute S, q and E when SR>Scrit_R.
        
        Report dt, to signal if Scrit was hit during the time step.
        """
        Smax, Smin, q = self.Smax_R, self.Scrit_R, 0.
        
        if np.isclose(Smin, Smax):
            return S, P, 0.
        
        if np.isclose(P, E):
            return S, q, E
        
        if P > E: # --- Fills        
            dt = (Smax - S) / (P - E)
            if dt < Dt: # --- Full
                dt2 = Dt - dt
                q = (P - E) * dt2  / Dt                
                return Smax, q, Dt
            else:
                S += (P - E)  * Dt
                return S, q, Dt
            
        elif P < E: # --- Empties
            dt = (Smin - S) / (P - E)
            if dt < Dt: # --- (S=Scrit)                                
                return Smin, q, dt
            else:
                S += (P - E) * Dt
                return S, q, Dt

        else:
            raise ValueError("Should never have gotten here.")
    

    def rootzone_bot(self, S: float, P: float, E: float, Dt:float)-> float:
        """Compute S, q and E when S < Scrit.
        
        Report dt to signal if S hits Scrit during the time step.
        
        q is always zero when S<=Scrit.
        """        
        S0, Smax, q = S, self.Scrit_R, 0.
        
        if np.isclose(P, E) or np.isclose(Smax, 0): # --- Nothing happens
            return S, Dt
        
        if np.isclose(E, 0): # --- P > 0, E == 0 --> Linear
            dt = (Smax - S0) / P
            if dt < Dt:
                return Smax, dt
            else:
                S += P * Dt
                return S, Dt
        
        # --- System characteristic time
        T = Smax / E
              
        if P > E: # --- Fills
            dt = T * np.log((S/Smax - P/E)/(1 - P/E))
            if dt < Dt: # --- Full                                    
                return Smax, dt
            
        # --- All other cases must succeed        
        S = Smax * (P/E + (S0/Smax - P/E) * np.exp(-Dt / T))
        return S, Dt
    
    
    def root_zone(self, S: float, P: float, E: float, Dt:float)-> float:
        
        S0, Scrit, Smax, q = S, self.Scrit_R, self.Smax_R, 0.
        
        def EA(P, S, S0, Dt, q):
            # --- Actual evapotranspiration over the entire time step
            return np.round(P - (S - S0) / Dt - q, 4)
        
        if np.isclose(P, E):
            return S, q, E        
        if Scrit <= S <= Smax:
            S, q, dt = self.rootzone_top(S0, P, E, Dt)
            if P > E:
                pass  # --- Done: works as linear model         
            elif dt < Dt:
                # --- Continue bottom reservoir for the remaining time
                S, dt = self.rootzone_bot(S, P, E, Dt-dt)
                                    
        else:
            S, dt = self.rootzone_bot(S0, P, E, Dt)
            if dt < Dt:
                # --- Storage passes top of bottom root zone
                #     Continue in upper (linear) root zone for the remaining time.
                S, q, _ = self.rootzone_top(S, P, E, Dt - dt)
                q *= (Dt - dt) / Dt
                                
        return S, q, EA(P, S, S0, Dt, q)
    

# %% Recharge with throttlling of E according to root S/Smax
    
class RchLam(RechargeBase):
    """Compute recharge with interception and root_zone storage assuming
    that the evapotranspiration is reduced by factor (S/Smax)^lambda.
    
    Lambda values can be anything between at least 0.25 and 1. It may be that
    for still smaller lambda, the RK45 integrator may fail.
    
    This is the most general method. 

    lambda = 0 yields the same results as the RchBin  object
    lambda = 1 yields the same reults as the RchEarth object.
    """
    def __init__(self, Smax_I: float = 2.5, Smax_R: float = 100., lam: float = 0.5, **_: Any)-> None:    
        super().__init__(Smax_I=Smax_I, Smax_R=Smax_R)        
        self.name = 'RchLam'        
        self.lam = lam
        
        self.event_y_eq_1.terminal = True
        self.event_y_eq_0.terminal = True
        
        self.event_y_eq_1.direction = +1 # --- Smax passed from below
        self.event_y_eq_0.direction = -1 # --- Smin passed from above
        
        # --- Events to check along the way during integration
        self.events = [self.event_y_eq_1, self.event_y_eq_0]
                
    @staticmethod
    def rhs(t, y, p, r, lam):
        """Return p - r y^lam."""
        return p - r * (np.clip(y, 0, None) ** lam)

    eps = 1e-6

    @staticmethod
    def event_y_eq_1(t, y, *_):
        """Return y-1, to track overshoot."""
        return y[0] - (1.0 - RchLam.eps)
    
    @staticmethod
    def event_y_eq_0(t, y, *_):
        """Return y to track undershoot."""        
        return y[0] - RchLam.eps
    

    def root_zone(self, S: float, P: float, E: float, Dt: float)->float:        
        """Return S, q, E.
        """
        # --- Note q is the recharge in mm/d
        S0, Smax, q = S, self.Smax_R, 0.

        # --- Nothing happens
        if np.isclose(P, E):
            return S0, q, E
        
        # --- Use dimensionless variables
        p, r, y = P / Smax, E / Smax, S0 / Smax
                
        # --- Protect against overreach
        small_tol = 0.001 / Smax
        ybase, ytop = small_tol, 1 - small_tol
        
        # ---- dy/dt
        yacc = p - r * y ** self.lam
        if y > ytop: # --- Full
            if yacc >= 0:
                q = (P - E) * Dt
                return Smax, q, E               
        elif y < ybase: # --- Empty
            if yacc <= 0:
                E = P
                return 0., q, E
            else:
                y = ybase
        
        # --- scipy's initial value ODE solver, using method RK45
        sol = solve_ivp(self.rhs, [0, Dt], [y], method='RK45', args=(p,r,self.lam),
                events=self.events, dense_output=True)
        
        # --- Handle results
        if sol.success:
            if len(sol.t_events[0]) > 0:
                # --- Get last time and event value.
                y1 = sol.y_events[0].ravel()[-1]
                dt = sol.t_events[0].ravel()[-1]
            else:
                # --- Get last time and value of the integration.
                y1 = sol.y.ravel()[-1]
                dt = sol.t.ravel()[-1]
            if dt < Dt: # --- Overflow for the remainder of the time step
                if P > E:
                    q = (P - E) * (Dt - dt)/Dt
                else:
                    q = 0
            S = Smax * y1
            EA = P - (S - S0) / Dt - q
            return S, q, EA
        else:
            raise RuntimeError("Runge Kutta did not finish successfully exit status {soi.status}")

    @staticmethod        
    def show_throttling_E_by_lam(lam=[0.1, 0.25, 0.5, 1.0]):
        """Show the throttling of E by lambda: E = E0 (S/Smax) ** lambda.

        This just show the function (S/Smax) ** lambda for 0<S<Smax.
        
        lam = float | iterable of float
        """
        
        y = np.linspace(0., 1.0, 200)
        lambdas = np.atleast_1d(np.asarray(lam))
        
        ax = etc.newfig(r"Effect of throttling E by $\lambda$: ($E = E_0 \cdot y^\lambda$)",
                        r"$y = S/S_{max}$",
                        r"$y^\lambda$")
        
        for lam in lambdas:
            ax.plot(y, y ** lam, label=fr"\lambda = {lam:3g}")
            
        ax.legend(loc="lower right")        
        return ax
        
        
    def show_effect_of_lam(self, x0=[0.5], PE=((2, 1)), lam=0.5, t=1.0, ax=None):
        """Show y(t)=S(t)/Smaxfor different p and r and given lambda

        This is just to show the course of the storage under constant P and E.

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
                            r"$y=S/S_{max}$")
        
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
    
def change_meteo(meteo: pd.DataFrame)-> pd.DataFrame:
    """Meteo is a pd.DataFrame with fields 'RH' and 'EV24', in mm/d.
    
    For exploration purposes, one can alter the P and E of the given
    pd.DataFrame. This way it will be easier to figure out what happens
    internally when one gets strange results.
    
    P and E are changed every 30 days and cycled (repeated).
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

# %%
# --- Get weather data for De Bilt
def get_deBilt_recharge(Smax_I=0.5, Smax_R=100, lam=0.25, datespan=("2020-01-01", None)):
    """Return leak from rootzone for de Bilt within datespan.
    
    Parameters
    ----------
    Smax_I: float
        Capacity of interception reservoir [mm].
    Smax_R: float
        Capacity of root zone reservoir [mm].
    datespan: 2-tuple of datetime strings
        start and end date for simulation.
    lam: float  0.1 < lam <= 1
        Power in reduction of evaporation from root zone
        Ea = E (S/Smax)^lam
        
    Returns
    -------
    pd.DataFrame with fields in deBilt with field 'RCH' added.
    """
    meteo_csv = os.path.join(dirs.data, "DeBilt.csv")
    os.path.isfile(meteo_csv)
    deBilt = pd.read_csv(meteo_csv, header=0, parse_dates=True, index_col=0)

    # --- Shorten the series for convenience
    if datespan[0]:
        deBilt = deBilt.loc[deBilt.index >= np.datetime64(datespan[0])]
    if datespan[1]:
        deBilt = deBilt.loc[deBilt.index <= np.datetime64(datespan[1])]

    # --- Get the recharge simulator
    # rch_simulator = RchEarth(Smax_I=Smax_I, Smax_R=Smax_R, lam=None)
    rch_simulator = RchLam(Smax_I=Smax_I, Smax_R=Smax_R, lam=lam)

    # --- Compute the recharge, see field 'RCH'
    return rch_simulator.simulate(deBilt)

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
    # Demonstration of the use of the rootzone model to convert Precipitation
    # and Makkink evaporation into a recharge series.
    # Makkink is used as E0, no changes made.
    # All data are in mm/d.
    # The decline (throttling) of E with the decline of the relative storage
    # may also be regarded as a crop resduction in some sense. Given all
    # uncertainties, trying to be more sophisticated does not make much sense.
    # TO 2025-09-05

    # Get the meteodata in a pd.DataFrame
    meteo_csv = os.path.join(dirs.data, "DeBilt.csv")
    os.path.isfile(meteo_csv)
    deBilt = pd.read_csv(meteo_csv, header=0, parse_dates=True, index_col=0)
    deBilt_short = deBilt.loc[deBilt.index >= np.datetime64("2020-01-01"), :]
    
    # deBilt_short = change_meteo(deBilt_short)
    
    date_span = (np.datetime64("2020-01-01"), np.datetime64("2021-03-31"))
    date_span = (deBilt_short.index[0], deBilt_short.index[-1])
    
    clrs = cycle('rbgkmcy')
    
    title = "Recharge computed by different methods"
    idx = ((deBilt_short.index >= date_span[0]) &
                               (deBilt_short.index <= date_span[1]))

    labels = ['Makkink', 'bin', 'earth', 'lambda']
    rchClasses = [RchMak, RchBin, RchEarth, RchLam]
    linewidths = [0.25, 0.50, 0.75, 1.0]
    
    
    # --- Loop control, chooses which models
    i1, i2 = 1, 4 # len(labels) + 1
    
    # --- Storage capacity
    Smax_I, Smax_R, Scrit_R = 1.5, 100, 50. # mm

    # --- Power lam model
    lam = 0.25

    if True:        
        ax1, ax2, ax3 = etc.newfigs(('EV24 and EA', 'P and RCH', 'STO'),
                'time', ('mm/d', 'mm/d', 'mm' ), figsize=(12, 10))
                
        ax1.plot(deBilt_short.index, deBilt_short['EV24'], 'k.', ms=2, label='EV24')
        ax2.plot(deBilt_short.index, deBilt_short['RH']  , 'k.', ms=2, label='Precip')
        
        for rchClass, label, lw in zip(rchClasses[i1:i2], labels[i1:i2], linewidths[i1:i2]):
                
            print(f"=== Running model = {label} ===")
                
            rch_simulator = rchClass(Smax_I=Smax_I, Smax_R=Smax_R,
                                     Scrit_R=Scrit_R, lam=lam)
            rch = rch_simulator.simulate(deBilt_short)
            
            clr = next(clrs)
            ax1.plot(rch.index[idx], rch['EA' ][idx], '-', lw=lw, color=clr,  label=label + ' EA')
            ax2.plot(rch.index[idx], rch['RCH'][idx], '-', lw=lw, color=clr,  label=label + 'RCH')
            ax3.plot(rch.index[idx], rch['STO'][idx], '-', lw=lw, color=clr,  label=label + ' STO')        
            print(f'Recharge according to method {label}:\n', rch.loc[idx].mean(), '\n')

        for ax in [ax1, ax2, ax3]:
            ax.legend(loc='upper right') 
            
        fig = plt.gcf()
        logo(fig, os.path.basename(os.__file__))
        fig.suptitle(("Different recharge models" + "\n" +
                      rf"Smax_I={Smax_I} mm, Smax_R={Smax_R} mm, Scrit_R={Scrit_R} mm, $\lambda$={lam}")
        ) 
        
        fig.savefig(os.path.join(dirs.images, 'rch_dBilt.pdf'))
             


    if False:
        rchlam = RchLam()
        
        rchlam.show_throttling_E_by_lam(lam=[0.1, 0.25, 0.5, 1.0])
        fig = plt.gcf()
        logo(fig, os.path.basename(os.__file__))
        fig.savefig(os.path.join(dirs.images, "throttling_by_lambda.pdf"))


        PE = [(0, 0), (1, 0), (0, 1), (1, 1), (4, 3.5)]
        
        ax = None
        for lam in [0.1, 0.25, 0.5, 1.0]:
            ax = rchlam.show_effect_of_lam(x0=0.5, PE=PE, lam=lam, t=250.0, ax=ax)
        logo(fig, os.path.basename(os.__file__))
        fig.savefig(os.path.join(dirs.images, "effect_of_lam_constant_P_E.pdf"))

        
        
    plt.show()

# %%
