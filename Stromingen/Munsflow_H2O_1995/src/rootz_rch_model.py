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
    that the evapotranspiration is reduced linearly according to the
    remainder of the storaga.
                
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
    
    