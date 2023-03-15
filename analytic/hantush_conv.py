#!/usr/bin/env python
# -*- coding: utf-8 -*-
#%%

tools = '/Users/Theo/GRWMODELS/python/tools/'

import sys
import os

if not tools in sys.path:
    sys.path.insert(0, tools)

import numpy as np
import matplotlib.pyplot as plt 
from scipy import signal
from scipy.integrate import quad
from etc import newfig
import pandas as pd

#%%
def Wh(u, rho):
    """Return Hantush well function by fast and accuratelyusing scipy's quad.
    
    This turns out to be a very accurate yet fast impementation, about as fast
    as the exp1 function form scipy.special.
    
    Parameters
    ----------
    u np.ndarray
        argument of Wh and Theis's W (=exp1)
    rho: float
        r / Lambda
        
    Returns
    -------
    Wh, err: tuple
        Wh together with its accuracy for all u.

    """
    def whkernel(y, rho):
        """Return kernel of Wh function, with rho=r/lambda."""
        return np.exp(-y - (rho/2) ** 2 / y ) / y    

    def whquad(u, rho):
        """Return Wh(u,rho) function, that integrates from u to infinity using quad."""
        return quad(whkernel, u, np.inf, args=(rho))

    # Turn whquad into a regular numpy function accepting arrays
    wh = np.frompyfunc(whquad, 2, 2) # 2 inputs and two outputs h and err
    
    return wh(u, rho)

class Hantush:
    """"Class implementing Hantush objects. 
    
    Such an object is situated as a point and computes the drawdown
    accoring to Hantush at an other point given a time series
    input for the extraction. 
    """
    def __init__(self, name='name?', kD=None, S=None, cstar=None, xw=0.0, yw=0.0):
        """Return a Hantush object. 
        
        Parammeters
        -----------
        name: str
            name of this Hantush object
        kD: float
            aquifer transmissivity
        S: float
            storage coefficient
        cstar: float or 2 floats (above and below)
            effective resistance felt by aquifer cstar = c[0]c[1]/(c[0] + c[1])
        t: time series for output
        xw: float
            x coordinate of well
        yw: float
            y coordinate of well
        """
        if np.isscalar(cstar):
            pass
        else:
            cT, cB = cstar[0], cstar[-1]
            cstar = cT * cB / (cT + cB)
            
        self.kD, self.S, self.cstar = kD, S, cstar
        self.L = np.sqrt(kD * cstar)
        self.xw = xw
        self.yw = yw
        return
        
    def ddn_simple(self, Q=None, xp=None, yp=None, t=None):
        """Return head change time series due to simple Q starting at t=0 (sign same as Q).
        
        Hence, a negative Q, meaning extration, causes a negative head change.
        Because we only comopute head changes for superposition, we don't need hTB here.
        
        Parameters
        ----------
        Q: float 
            constant extraction from first time. 
        x, y: floats,
            location of observation point.
        t: ndarray of floats or np.datetime64
            times at which drawdown is required.
        """
        try:
            tau = (t - t[0]) / np.timedelta64(1, 'D')
        except:
            tau = t - t[0]
            
        r = np.sqrt((self.xw - xp) ** 2 + (self.yw - yp) ** 2)
        rho = r / self.L
        u = r ** 2 * self.S / (4 * self.kD * tau)
        return Q  / (4 * np.pi * self.kD) * Wh(u, rho)[0]
        
        
    def ddn_conv(self, Qseries=None, xp=None, yp=None):
        """Compute (by convolution) the drawdown at x, y as a time series.
        
        Because we only comopute head changes for superposition, we don't need hTB here.
        
         Parameters
        -----------
        Qseries: pd.DataFrame with field Q and index np.datetime64.
            Flow Q, extraction negative, injections positive (Q and ddn same sign)
        xp, yp: floats 
            location of observation point 
        """
        assert isinstance(Qseries, pd.Series), "Qseries must be a pd.DataFrame"
        assert Qseries.index.values.dtype == np.dtype('datetime64[ns]'),\
                        "index of the Qseries must be of type np.datetime64."
        
        dtime = np.array(Qseries.index)
        tau = (dtime - dtime[0]) / np.timedelta64(1, 'D')
        
        blk_resp = self.BR(tau, xp=xp, yp=yp)
        dh = signal.lfilter(blk_resp, 1.0, Qseries) # head change
        ddn = pd.DataFrame(index=Qseries.index, columns=['s', 'q'])
        ddn['s'] = dh
        ddn['q'] = ddn['s'] / self.cstar # change of local injection (same sign as dh and Q)
        return ddn
        
    def BR(self, tau, xp, yp):
        """Return the block Hantush respons 
        Parameters
        -----------
        tau: ndarray of floats
            Uniform time series, starting at zero.
        xp, yp: floats
            Location for which the block response is computed (the object knows where the well is).
        """
        assert tau[0] == 0, "tau must start with zero!"
        assert np.isclose(np.diff(np.diff(tau)).sum(), 0.), "Time series must be equal distance."
        Dt = tau[1] - tau[0]
        
        r = np.sqrt((self.xw - xp) ** 2 + (self.yw - yp) ** 2)
        rho = r / self.L
        u = r ** 2 * self.S / (4 * self.kD * tau[1:])  # skip first, because division by zero
        self.step_resp = np.hstack((0, 1  / (4 * np.pi * self.kD) * Wh(u, rho)[0])) 
        self.blk_resp = np.hstack((0, np.diff(self.step_resp))) # first must be zero see formaula.
        return self.blk_resp


class HantushCollection:
    """Collection of Hantush objects with same properties but different locations."""
    
    def __init__(self, name='name?', kD=None, S=None, cstar=None, locations=None):
        """"Return an instantiated collection of Hantush objects.
        
        Parameters
        ----------
        name: str
            name of this collection
        kD: float
            Transmissivity.
        S: seuence of floats.
            Storage coefficient of each aqufer.
        cstar: float or 2-tuple of floats
            Effective resistance combining that of the overlaying and underlaying aquitard.
            if cstar is a 2-tuple, then cstar = (cT * cB / (cT + cB))
        locations: pd.DataFrame
            the locations of the Hantush objects (i.e. the Hantush wells.
        """
        self.name = name
        
        assert np.isscalar(cstar) or len(cstar) == 2, "Cstar must be a float or a 2-tuple of floats."
        if len(cstar) == 2:
            cT, cB = cstar
            cstar = cT * cB / (cT + cB)
        
        self.hantDict = dict()
        for name in locations.index:
            xw, yw = locations.loc[name][['xc', 'yc']]
            self.hantDict[name] = Hantush(name, kD=kD, S=S, cstar=cstar, xw=xw, yw=yw)
        return
    
    def ddn_conv(self, QseriesDf=None, xp=None, yp=None):
        """Return the combined drawdown by the Hantush collection at point xp, yp.
        
        Parameters
        ----------
        QseriesColl: a pd.DataFrame
            the flow for each of the hantush objects. The columns correspond to the Hantush object names.
            They all share the same index.
        xp, yp: 2 floats
            location of point for which the drawdown is computed.
        """
        ddn = pd.DataFrame(index=QseriesDf.index, columns=['s', 'q'])
        ddn['q'] = 0.0
        ddn['s'] = 0.0
        for name in QseriesDf.columns:
            han = self.hantDict[name]
            deltaDdn = han.ddn_conv(Qseries=QseriesDf[name], xp=xp, yp=yp)
            ddn['s'] += deltaDdn['s']
            ddn['q'] += deltaDdn['q']
        return ddn


def get_testQ(datetime, Qlist=None):
    """Return pd.DataFrame with test values for Qseries.
    
    Parameters
    ----------
    datetime: pd.index or an array with np.datetime64 as dtype
        simulation times
    Qlist: sequence
        A series of Q values to be used to set Qt
        QL will be used equidistantly spread over the entire time.
    """
    if not Qlist:
        Qlist =[0, 400, 200, 100, 700, 200, 0, 200, 800, 0, 200, 0, 500, 0,
                200, 500, 200, 200, 900, 900, 0,   0,   0,]

    t0 = datetime[0]
    NQ = len(datetime) // len(Qlist)
    
    Qt = np.zeros(len(datetime))

    for i, Qi in enumerate(Qlist):
        Qt[datetime > t0 + i * NQ * np.timedelta64(1, 'D')] = Qi
    return pd.DataFrame(index=datetime, data=Qt, columns=['Q'])


#%% 
if __name__ == '__main__':
    scen = 2
    
    name='sterrebos'
    xw, yw, xp, yp = 250., 250., 0., 0.
    kD = 200 # m2.day
    c = (500, 5000) #d
    S = 0.1

    han = Hantush(name=name, kD=kD, S=S, cstar=c, xw=xw, yw=yw)
    
# %%
    if scen == 0:
        Q = -1200.
        t = np.logspace(-2, 3, 51)
        ddn = han.ddn_simple(Q=Q, xp=xp, yp=yp, t=t)
        
        ax=newfig("Hantush well drawdown Q={:.0f} m3/d".format(Q), "time [d]", "dd", xscale='linear')
        ax.plot(t, ddn)

# %% lfilter total period
    if scen == 1:
        t0 = np.datetime64("2016-01-01")
        t1 = np.datetime64("2022-01-08")
        datetime = np.arange(t0, t1, np.timedelta64(1, 'D'))    
        Q1 = get_testQ(datetime)
    
        # Both Qseries as Q_sq contain the same pd.DataFrame with the results.
        Qseries = pd.DataFrame(index=datetime, data=-Q1, columns=['Q'])
        Q_sq = han.ddn_conv(Qseries=Qseries, xp=xp, yp=yp)
        
        ax=newfig("Hantush well by convolution", "time", "Q [m3/d]", invert_yaxis=False)           
        ax.plot(Q_sq.index, Q_sq.Q, label='Q in m3/d')
        
        ax=newfig("Hantush well  by convolution", "time", "s = h-h0 [m]", invert_yaxis=False)
        ax.plot(Q_sq.index, Q_sq.s, label="head change")
        
        ax=newfig("Hantush well by convolution", "time", "q [m/d]", invert_yaxis=False)
        ax.plot(Q_sq.index, Q_sq.q, label="head change")

# %% lfilter for shorter period
    if scen == 2:
        t0 = np.datetime64("2016-01-01")
        t1 = np.datetime64("2016-04-01")
        datetime = np.arange(t0, t1, np.timedelta64(1, 'D'))
        Q2 = get_testQ(datetime, Qlist=[0, 600, 0, 300])
    
        Qseries = pd.DataFrame(index=datetime, data=-Q2, columns=['Q'])
        Q_sq = han.ddn_conv(Qseries=Qseries, xp=xp, yp=yp)
        
        ax=newfig("Hantush well Q/Q.mean and ddn/ddn.mean by convolution", "time", "Q/Q.mean() and ddn/ddn.mean()[m]", invert_yaxis=False)
        ax.plot(Q_sq.index, Q_sq.Q  / np.mean(Q_sq.Q)  ,  '.', label="Q/W.mean()")
        ax.plot(Q_sq.index, Q_sq.s  / np.mean(Q_sq.s),  '-', label="s/s.mean()")
        ax.plot(Q_sq.index, Q_sq.q  / np.mean(Q_sq.q),  'x', label="q/q.mean()")
        ax.legend()
        
# %% lfilter and other filter use from scipy.lsignal.lfilter documentation
    if scen == 3:
        rng = np.random.default_rng()
        t = np.linspace(-1, 1, 201)
        x = (np.sin(2*np.pi*0.75*t*(1-t) + 2.1) +\
                    0.1*np.sin(2*np.pi*1.25*t + 1) +\
                     0.18*np.cos(2*np.pi*3.85*t))
        xn = x + rng.standard_normal(len(t)) * 0.08
        b, a = signal.butter(3, 0.05)
        zi = signal.lfilter_zi(b, a)
        z, _ = signal.lfilter(b, a, xn, zi=zi*xn[0])
        z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
        y = signal.filtfilt(b, a, xn)
        plt.figure
        plt.plot(t, xn, 'b', alpha=0.75)
        plt.plot(t, z, 'r--', t, z2, 'r', t, y, 'k')
        plt.legend(('noisy signal', 'lfilter, once', 'lfilter, twice',
                                        'filtfilt'), loc='best')
        plt.grid(True)
        plt.show()
        
    plt.show()
    # %%
    
