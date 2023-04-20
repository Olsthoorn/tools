#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
#    Multilayer analytic single-point model with vertical resistance at top
#    and bottom for connection to outside world.
#    
#    The aquifer is a stack of n + 1 aquitards interlaced with n aquifers, defined
#    only by their storage coefficient, since there is no horizontal flow.
#    The aquitards are defined only by their resistance.
#    There is a prescribed head above the top aquitard, and below the bottom aquitard.
#    
#    With n layers, there are n + 1 aquitards and n + 2 heads, i.e. n
#    for the aquifers plus one above the stack and one below it.
#
# Used to simulate drawdown associated with Groningen A7 tunnel construction 2022

# The Hantush well drawdown function
# The Hantush well drawdown convoluting the extraction input to deal
# with time-variable extacctioin
#
# Analytic single point multi-layer drawdown model developed, also with convolution
# ColumnMdlAnal   multilayer analytical model
# Column1LAnal    single layer analytical model
# ColumnMultiNum  multi-layer one cell numerical model to verify the analytical
#
# Several scenarios at the end of this file,
# TO 20230416

tools = '/Users/Theo/GRWMODELS/python/tools/'

import os
import sys
if not tools in sys.path:
    sys.path.insert(0, tools)

from signal import signal
import numpy as np
import matplotlib.pyplot as plt 
import scipy.linalg as lalg
import scipy.signal as signal
from scipy.integrate import quad
from etc import newfig


#%%
def Wh(u, rho):
    """Return Hantush well function by integration using scipy functionality.
    
    This turns out to be a very accurate yet fast impementation, about as fast
    as the exp1 function form scipy.special.
    
    In fact we define three functions and finally compute the desired answer
    with the last one. The three functions are nicely packages in the overall
    W_theis1 function.

    """
    def whkernel(y, rho): return np.exp(-y - (rho/2) ** 2 / y ) / y    

    def whquad(u, rho): return quad(whkernel, u, np.inf, args=(rho))

    wh = np.frompyfunc(whquad, 2, 2) # 2 inputs and tow outputs h and err
    
    return wh(u, rho)

class Hantush:
    """"Class implementing Hantush objects. 
    
    Such an object is situated as a point and computes the drawdown
    accoring to Hantush at an other point given a time-series
    input for the extraction by the object. 
    """
    def __init__(self, kD=None, S=None, c=None, xw=0.0, yw=0.0, rw=0.25):
        """Return a Hantush object. 
        
        Parammeters
        -----------
        kD: float
            aquifer transmissivity
        S: float
            storage coefficient
        c: float
            effective resistance felt by aquifer
        t: time series for output
        xw, yw, rw: floats
            location of well and well radius
        """
        
        self.kD, self.S, self.c = kD, S, c
        self.L = np.sqrt(kD * c)
        self.xw = xw
        self.yw = yw
        self.rw = rw
        return
        
    def ddn_simple(self, Q=None, xp=None, yp=None, t=None):
        """Return drawdown time series due to simple extraction Q starting at t=0 (no convolution).
        
        Parameters
        ----------
        Q: float 
            constant extraction from first time. 
        xp, yp: floats,
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
    
        
    def ddn_conv(self, Qandt=None, x=None, y=None, t=None):
        """Compute the drawdown by convolution at a point x, y as a time series.
        
        Parameters
        ----------
        Qandt: pd.DataFrame
         DataFrame with index dtype np.datetime64 and columns 'Q' and 't'
        x, y: floats 
         location of observation point 
        t: ndarray 
         times at which drawdown is desired
        """
        assert Qandt.dtype == np.datetime64,\
            "Qandt must be pd.DataFrame with index dtype np.datetime64 not{}".format(Qandt.dtype)
        
        tau = (Qandt.index - Qandt.index[0]) / np.timedelta64(1, 'D')

        blk_resp = self.BR(tau)
        self.ddn = signal.lfilter(blk_resp, 1.0, Qandt['Q'])
        
    def BR(self, t=None, xp=None, yp=None):
        """Return block response of the Hantush object.
        
        Parameters
        ----------
        t: ndarray of floats or np.datetime64
         time series
        xp, yp: floats
         location of observation point    
        """
        tau = t - t[0]
        r = np.sqrt((self.xw - xp) ** 1 + (self.yw - yp) ** 2)
        rho = r / self.L
        u = r ** 2 * self.S / (4 * self.kD * tau)
        self.step_resp = 1  / (4 * np.pi * self.kD) * Wh(u, rho)
        self.blk_resp = np.diff(self.step_resp)
        return self.blk_resp


#%% 
class ColumnMdlAnal:
    """Multilayer analytic single-point model with vertical resistance at top and bottom for connection to outside world.
    
    The aquifer is a stack of n + 1 aquitards interlaced with n aquifers, defined only by their storage coefficient, since there is no horizontal flow.
    The aquitards are defined only by their resistance.
    There is a prescribed head above the top aquitard, and below the bottom aquitard.
    
    With n layers, there are n + 1 aquitards and n + 2 heads, i.e. n for the aquifers plus one above the stack and one below it.
    """
    def __init__(self, name=None, S=None, c=None, x=0, y=0):
        """
        Parameters
        ----------
        name: str
            key or name of model, may refer to observation well.
        S : sequence
            Storage coefficients, one for each aquifer
        c : sequence
            Aquitard resistances, len(c) must be len(S) + 1
        x, y: float 
            Location of the point model
        """
        assert len(c) == len(S) + 1, "len(c)={} must be len(S) + 1 = {}".format(len(s), len(S) + 1)

        self.S = np.array(S)
        self.Sm1 = np.diag(1/self.S, k=0)
        self.c = np.array(c)
        self.N = len(S)        

        self.name = name
        self.x = x
        self.y = y

        self.A, self.E, self.V = self.sysmat()
        self.Vm1 = lalg.inv(self.V)
        self.Am1 = lalg.inv(self.A)
        self.Em1 = lalg.inv(self.E)
        return
    
    def sysmat(self):
        """Return system matrix  and related arrays and vectors.
        
        Returns
        -------
        A: ndarray(N, N)
        E: ndarray(N), eigen values of A
        V: ndarray(N, N): right eigen values of A
        "
        """
        dL = 1 / (self.S * self.c[:-1])
        dR = 1 / (self.S * self.c[ 1:])        
        A  = -np.diag(dL[1:], k=-1) -np.diag(dR[:-1], k=+1) + np.diag(dL + dR)
        E, V = lalg.eig(A)
        return A, np.diag(E), V
        
    def sim(self, h0=None, hTB=(None, None), q=None, datetime=None):
        """Simulate over time using h0 as start values.
        
        Parameters
        ----------
        h0: np.array of length N
            Initial head in the model.
        hTB: sequence of 2 or an array of (2, nt)
            The head above the top of the model and below the bottom of the model.
        q: np.array of size (N, nt)
            The rechage in all layers [m/d]
        datetime: np.array of dattime64 or floats [d]
            The datetimes for which to simulate. first time is that for the initial head
            so h[:, 0] = h0
        """
        Nt = len(datetime)
        h0, hTB, q = self.siminputcheck(h0, hTB, q, datetime)
        
        qTB = np.zeros_like(q) # q has shape (self.N, len(datetime))
        qTB[ 0] = hTB[ 0] / self.c[ 0]
        qTB[-1] = hTB[-1] / self.c[-1]
        
        I = np.eye(self.N)
        h = h0 * np.ones((1, len(datetime))) # h0 has shape(self.N, 1)
        
        # tau will be floats
        try: # np.datetime64 object array
            tau =(datetime - datetime[0]) / np.timedelta64(1, 'D')
        except: # Then floats
            tau = datetime - datetime[0]
        
        dt = np.diff(tau)
        assert np.isclose(np.diff(dt).sum(), 0.), "All time steps must be the same."
        Dt = dt[0]
        
        expm = lalg.expm(-self.E * Dt)
        X = self.V @ self.Em1 @ expm @ self.E @ self.Vm1
        Y = self.Am1 @ self.Sm1 - self.V @ self.Em1 @ expm @ self.Vm1 @ self.Sm1
        for i in range(Nt - 1):            
            h[:, i + 1] = X @ h[:, i] + Y @ (q[:, i] + qTB[:, i])
        return h
            
    def siminputcheck(self, h0=None, hTB=None, q=None, datetime=None):
        """Check the input for the simulation.
        
        h0 must be given as a N array. 
        hTB and q may be given as a 2-tuple or 2-list or an array of 2, Nt
        t must be an n-sequence of arbitrary length Nt > 1
        """
        self.Nt = len(datetime)
        assert self.Nt > 1, "lent(t) must be larger than 1"
        
        h0 = np.array(h0)
        assert h0.size == self.N, f"h0 must be a vector of length N = {self.N}, not {len(h0)}"
        h0 = h0.reshape(self.N, 1)
        
        # Checking htop and hbottom:
        hTB = np.array(hTB)
        if hTB.ndim == 1:
            hTB = hTB[:, np.newaxis]
        assert len(hTB) == 2, "Use squence of len (2) or ndarray =of size(2, Nt)."

        # extend hTB over all datetimes
        if hTB.shape[1] < self.Nt:
            hTB = hTB * np.ones((1, self.Nt))

        assert hTB.shape == (2, self.Nt),\
            f"hT.shape must be ({2},{self.Nt}), not ({hTB.shape[0]},{hTB.shape[1]})]"

        # Check q
        q = np.array(q)
        if q.ndim == 1:
            q = q[:, np.newaxis]
        if q.shape[1] < self.Nt:
            q = q * np.ones((1, self.Nt))
            
        assert q.shape == (self.N, self.Nt),\
            f"q.shape must be ({self.N}, {self.Nt}), not ({q.shape[0]},{q.shape[1]})"
            
        return h0, hTB, q
    
    def hinf(self, hTB=None):        
        qTB = np.zeros(self.N)
        qTB[ 0] = hTB[0] / self.c[0]
        qTB[-1] = hTB[1] / self.c[-1]
        
        return self.Am1 @ self.Sm1 @ qTB
 
 # %% Numeriek kolommodel
 
class Col1LayAnalMdl:
    """Single layer numeric single point model with resistance at top and bottom for connection to outside world.
    
    There is one aquifer with a resistent layer on top and one at the bottom.
    On th etop and at the bottom the head is prescribed.
    Influx (exflux) can be specified.    
    """
     
    def __init__(self, name=None, S=None, c=None, x=-0, y=0):
        """Initialize and return object.
        
        Parameters
        ----------
        name : str
            Name of the model.
        S: float
            Aquifer storage coefficient. 
        c: sequence of 2 floats 
            Resistance of layer on top and of layer at bottom of the aquifer.
        x, y: 2 floats 
            Location of the model.
        """
        self.name = name
        
        assert len(c) == 2, f"len(c)={len(c)} must be 2."

        self.S = S
        self.c = c
        self.cStar = c[0] * c[-1] / (c[0] + c[-1])
        self.T = S * self.cStar
        self.x = x
        self.y = y

        self.eps = 2/3.  # implicitness
        return

    def sim(self, h0=None, hTB=(None, None), q=None, datetime=None):
        """Simulate over time using h0 as start values.
        
        Parameters
        ----------
        h0: float
            Initial head in the model.
        hTB: sequence of 2 or an array of (2, nt)
            The head above the top of the model and below the bottom of the model.
        q: np.array of size (1, nt)
            The recharge in all layers [m/d]
        datetime: np.array of dattime64 or floats [d]
            The datetimes for which to simulate. first time is that for the initial head
            so h[0] = h0
        """
        Nt = len(datetime)
        
        hTB = np.array(hTB)
        if hTB.ndim == 1:
            hTB = hTB[:, np.newaxis] * np.ones(len(datetime))
            
        hStar = (self.c[-1] * hTB[0] + self.c[0] * hTB[1]) / (self.c[0] + self.c[-1])
        
        if np.isscalar(q):
            q = q * np.ones(len(datetime))
        assert len(q) == Nt, f"Shape of q must {Nt} or q must be a scalar."

        h = h0 * np.ones(len(datetime)) # h0 has shape(self.N, 1)
        
        # tau will be floats
        try: # np.datetime64 object array
            tau =(datetime - datetime[0]) / np.timedelta64(1, 'D')
        except: # Then floats
            tau = datetime - datetime[0]
        
        dt = np.diff(tau)
        assert np.isclose(np.diff(dt).sum(), 0.), "All time steps must be the same."
        
        Dt = dt[0]
        exp = np.exp(-Dt / self.T)
        hinf = hStar + q * self.cStar
        for i in range(Nt - 1):
            h[i+1] = h[i] * exp + hinf[i] * (1 - exp)
        return h
     
 
class ColumnMdlNum:
    """Multilayer numeric single point model with resistance at top and bottom for connection to outside world.
    
    The aquifer is a stack of n + 1 aquitards with interlaced n aquifers, defined by their storage coefficient.
    The aquitards are defined by their resistance.
    There is a given head above the top and below the bottom aquitard.
    
    With n layers, there are n + 1 aquitards and n + 2 heads, i.e. n for the aquifers plus one above the stack and one below it.
    
    """
    
    def __init__(self, name=None, S=None, c=None, x=0, y=0):
        """
        Parameters
        ----------
        name: str
            key or name of model, may refer to observation well.
        S : sequence
            Storage coefficients, one for each aquifer
        c : sequence
            Aquitard resistances, len(c) must be len(S) + 1
        x, y: float 
            Location of the point model
        """
        assert len(c) == len(S) + 1, "len(c)={} must be len(S) + 1 = {}".format(len(s), len(S) + 1)

        self.S = np.array(S)
        self.Sm1 = np.diag(1/self.S, k=0)
        self.c = np.array(c)
        self.N = len(S)        

        self.name = name
        self.x = x
        self.y = y

        self.A, self.E, self.V = self.sysmat()
        self.Vm1 = lalg.inv(self.V)
        self.Am1 = lalg.inv(self.A)
        self.eps = 2/3.  # implicitness
        return
    
    def sysmat(self):
        """Return system matrix  and related arrays and vectors.
        
        Returns
        -------
        A: ndarray(N, N)
        E: ndarray(N), eigen values of A
        V: ndarray(N, N): right eigen values of A
        "
        """
        dL = 1 / (self.S * self.c[:-1]) # left sub diagonal
        dR = 1 / (self.S * self.c[ 1:])  # richt sub diagonal       
        A  = -np.diag(dL[1:], k=-1) -np.diag(dR[:-1], k=+1) + np.diag(dL + dR)
        E, V = lalg.eig(A)
        return A, np.diag(E), V
    
    
    def sim(self, h0=None, hTB=(None, None), q=None, datetime=None):
        """Simulate over time using h0 as start values.
        
        Parameters
        ----------
        h0: np.array of length N
            Initial head in the model.
        hTB: sequence of 2 or an array of (2, nt)
            The head above the top of the model and below the bottom of the model.
        q: np.array of size (N, nt)
            The rechage in all layers [m/d]
        datetime: np.array of dattime64 or floats [d]
            The datetimes for which to simulate. first time is that for the initial head
            so h[:, 0] = h0
        """
        h0, hTB, q = self.siminputcheck(h0, hTB, q, datetime)
        
        qTB = np.zeros_like(q) # q has shape (self.N, len(datetime))
        qTB[ 0] = hTB[ 0] / self.c[ 0]
        qTB[-1] = hTB[-1] / self.c[-1]
        
        I = np.eye(self.N)
        h = h0 * np.ones((1, len(datetime))) # h0 has shape(self.N, 1)
        
        # tau will be floats
        try: # np.datetime64 object array
            tau =(datetime - datetime[0]) / np.timedelta64(1, 'D')
        except: # Then floats
            tau = datetime - datetime[0]
        
        dt = np.diff(tau)
        assert np.isclose(np.diff(dt).sum(), 0.), "All time steps must be the same."
        
        Dt = dt[0]
        Z = lalg.inv(I + self.A * self.eps  * Dt)
        X = Z @ (I - self.A * (1 - self.eps) * Dt)
        Y = Z @ self.Sm1 * Dt
        for i, _ in enumerate(dt):
            h[:, i+1] = X @ h[:, i] + Y @ (q[:, i] + qTB[:, i])
        return h
            
    def siminputcheck(self, h0=None, hTB=None, q=None, datetime=None):
        """Check the input for the simulation.
        
        h0: ndarray (N,)
            Initial head vector
        hTB: 2-sequence or ndarray of (2, len(datettime)
            head at top and at bottom
            Will be converted to ndarray (2, len(datetime))
        q: N-sequence or ndarray of (N, len(datetime))
            influx for each of the layers.
            Will be converted to an ndarray (1, len(datetime))            
        datetime: ndarray of np.datetime64 or ndarray of floats
            simulation times including (date)time of initial head.
            datetime is used but not altered.
        """
        self.Nt = len(datetime)
        assert self.Nt > 1, "lent(t) must be larger than 1"
        
        h0 = np.array(h0)
        assert h0.size == self.N, f"h0 must be a vector of length N = {self.N}, not {len(h0)}"
        h0 = h0.reshape(self.N, 1)
        
        # Checking htop and hbottom:
        hTB = np.array(hTB)
        if hTB.ndim == 1:
            hTB = hTB[:, np.newaxis]
        assert len(hTB) == 2, "Use squence of len (2) or ndarray =of size(2, Nt)."

        # extend hTB over all datetimes
        if hTB.shape[1] < self.Nt:
            hTB = hTB * np.ones((1, self.Nt))

        assert hTB.shape == (2, self.Nt),\
            f"hT.shape must be ({2},{self.Nt}), not ({hTB.shape[0]},{hTB.shape[1]})]"

        # Check q
        q = np.array(q)
        if q.ndim == 1:
            q = q[:, np.newaxis]
        if q.shape[1] < self.Nt:
            q = q * np.ones((1, self.Nt))
            
        assert q.shape == (self.N, self.Nt),\
            f"q.shape must be ({self.N}, {self.Nt}), not ({q.shape[0]},{q.shape[1]})"
            
        return h0, hTB, q
    
    def dhdt(self, h0=None, hTB=(None, None), q=None, datetime=None):
        """Simulate dhdt as s function of input with constant h0 and return the data.
        
        Parameters
        ----------
        h0: np.array of length N
            Initial head in the model.
        hTB: sequence of 2 or an array of (2, nt)
            The head above the top of the model and below the bottom of the model.
        q: np.array of size (N, nt)
            The rechage in all layers [m/d]
        datetime: np.array of dattime64 or floats [d]
            The datetimes for which to simulate. first time is that for the initial head
            so h[:, 0] = h0
        """
        h0, hTB, q = self.siminputcheck(h0, hTB, q, datetime)
        
        qTB = np.zeros_like(q) # q has shape (self.N, len(datetime))
        qTB[ 0] = hTB[ 0] / self.c[ 0]
        qTB[-1] = hTB[-1] / self.c[-1]
        
        I = np.eye(self.N)
        h = h0 * np.ones((1, len(datetime))) # h0 has shape(self.N, 1)
        
        # tau will be floats
        try: # np.datetime64 object array
            tau =(datetime - datetime[0]) / np.timedelta64(1, 'D')
        except: # Then floats
            tau = datetime - datetime[0]
        
        dt = np.diff(tau)
        assert np.isclose(np.diff(dt).sum(), 0.), "All time steps must be the same."
        
        Dt = dt[0]
        Z = lalg.inv(I + self.A * self.eps  * Dt)
        X = Z @ (I - self.A * (1 - self.eps) * Dt)
        Y = Z @ self.Sm1 * Dt
        
        dhdt = np.zeros_like(h)
        
        for i, _ in enumerate(dt):
            #h[:, i+1] = X @ h[:, i] + Y @ (q[:, i] + qTB[:, i])
            dhdt[:, i + 1] = - self.A @ h[:, i] + self.Sm1 @ (q[:, i] + qTB[:, i]) 
        return dhdt

    def hinf(self, hTB=None):        
        qTB = np.zeros((self.N, 1))
        qTB[ 0, 0] = hTB[ 0] / self.c[ 0]
        qTB[-1, 0] = hTB[-1] / self.c[-1]        
        return self.Am1 @ self.Sm1 @ qTB
    
    def test1(self, h0=None, tau=None):
        ha = h0[:, np.newaxis] * np.ones((1, len(tau)))
        hb = h0[:, np.newaxis] * np.ones((1, len(tau)))
        for i, t in enumerate(tau):
            ha[:, i] = self.Am1 @ lalg.expm(self.E * t) @ self.A @ h0
            hb[:, i] = lalg.expm(self.E * t) @ h0
        return ha, hb

    def test2(self, h0=None, tau=None):
        ha = h0[:, np.newaxis] * np.ones((1, len(tau)))
        hb = h0[:, np.newaxis] * np.ones((1, len(tau)))
        for i, t in enumerate(tau):
            ha[:, i] = self.V @ lalg.expm(self.E * t) @ self.E @ self.Vm1 @ h0
            hb[:, i] = lalg.expm(self.E * t) @ self.V @ self.E @ self.Vm1 @ h0
        return ha, hb

def checkLeakageHantush(kD=900, S=0.2, c=500, r=np.logspace(1, 2, 11), Q=1200, t=1, verbose=True):
    """Verify computation of local leakage Hantush.
    
    q1 = vertical leakage based on Hatnush drawdown
    q2 = vertical leakage based on deriveative of Wh(u, rho)
    
    Parameters
    ----------
    kD: float 
        transmissivity
    S: float 
        storage coefficient 
    c: float
        resitance of aquitard 
    r: ndarray of floats 
        distance from well 
    Q: float 
        extraction by well 
    t: float 
        time for which to compute and shown leakage
    verbose: boolean
        if True plot
    """
    
    R = r[:]
    U = R ** 2 * S / (4 * np.pi * kD)
    L = np.sqrt(kD * c)
    Rho = R / L
    s  = np.zeros_like(R)
    q2 = np.zeros_like(R)
    for i, (u, r, rho) in enumerate(zip(U, R, Rho)):
        s[i] =  Q / (4 * np.pi * kD) * Wh(u, rho)[0]
        q2[i] = Q / (2 * np.pi * r) * np.exp((-u - rho ** 2  / (4 * u)) / u) * (
        rho ** 2 / u ** 2 / 2 - rho ** 2 / (r * 2 * u) - 1)
    
    q1 = s / c
    
    if verbose:
        ttl = f"Q = {Q:.0f} m2/d, kD={kD:.0f} m2/d, S={S:.3f}, c={c:.0f} d, t={t:.2f} d, "        
        ax = newfig("Compare lekage Hantush, " + ttl, "r", "leakage m/d")
        ax.plot(R, q1, label="q based on Hantush drawdown")
        ax.plot(R, q2, label="q based on derivative of W(u, rho)")
        ax.legend()
    return q1, q2

#%% 
if __name__ == '__main__':
    scen = 1
    name='sterrebos'
    kD = [50, 200, 2000] # m2.day
    Q = 1200
    xw, yw = 250., 250.,
    xm, ym = 0., 0.
    c = [500, 500, 500, 500] # Keileem, ....
    S = [0.001, 0.001, 0.001] # Freatisch, Peelo, ...

    t0 = np.datetime64("2022-01-01T00:00")
    
    if scen == 0:
        c = [500, 50, 50, 500000] # Keileem, ....
        S = [0.01, 0.01, 0.01] # Freatisch, Peelo, ...
        
        # Make a time series for the head at the top and bottom boundaries
        Nt = 501
        h0 = np.array([2.5, 2.5, 2.5])
        hTB = (1, -1)  # Head in top and bottom
        hTB = (10, -0.)
        hTB = (5., 5.)
        hTB = (2., -2.)
        hTB2 = (2, -2.)
        hTB = (0., 0.)
        hTB = np.array(hTB)[:, np.newaxis] * np.ones(Nt)
        hTB[:, Nt//2:] =np.array(hTB2)[:, np.newaxis] * np.ones(hTB.shape[1] - Nt//2) 

        # Simulation time
        tau = 1.0 * np.arange(Nt)
        datetime = t0 + tau * np.timedelta64(1, 'D')
        
        # Injection flow (recharge and from other sources, prescribed)
        q = np.zeros((len(S), len(tau)))

    elif scen == 1:
        c = [500, 500, 500, 500] # Keileem, ....
        S = [0.1, 0.1, 0.1] # Freatisch, Peelo, ...
        Nt = 721 # d
        h0 = np.array([1, 0.1, -1])
        hTB = (0., 0.)
        tau = 1.0 * np.arange(Nt)
        datetime = t0 + tau * np.timedelta64(1, 'D')
        q = np.zeros((len(S), len(tau)))
        q[ 0, tau > 180] = +0.002
        q[ 0, tau > 360] = -0.002
        q[ 0, tau > 540] = +0.000
        q[-1, tau > 640] = -0.005

    if scen >= 100:
        han = Hantush(kD=kD[-1], S=S[-1], c=c[-1], x=xw, y=yw)
        ddn = han.ddn_simple(Q=Q, x=xm, y=ym, t=datetime)
        
        ax=newfig(f"Hantush well drawdown Q={Q:.0f} m3/d", "time [d]", "dd", xscale='log', yinvert=True)
        tau = (datetime - datetime[0]) / np.timedelta64(1, 'D')
        ax.plot(tau, ddn)
    
    # %% Column model =============
    
    # Simulate the analytic multi layer column model
    cmdlanal = ColumnMdlAnal(name=name, S=S, c=c, x=xm, y=xm)     # Instantiate the model
    hanal = cmdlanal.sim(h0=h0, hTB=hTB, q=q, datetime=datetime)  # Simulate
     
     # Simulate the numeric multilayer column model
    cmdlnum = ColumnMdlNum(name=name, S=S, c=c, x=xm, y=xm)      # Instantiate the model
    hnum = cmdlnum.sim(h0=h0, hTB=hTB, q=q, datetime=datetime)   # Simulate

    # Simulate the single layer analytical model
    # Make sure the intput is compatible with that of the multilayer models
    cmdl1 = Col1LayAnalMdl(name=name, S=np.array(S).sum(), c=(c[0], c[-1]), x=xm, y=xm) # Instantiate
    h1L = cmdl1.sim(h0=h0[0], hTB=hTB, q=0., datetime=datetime) # Simulate

    clr = ['r', 'b', 'g', 'orange']

    # Plot the result
    ax = newfig(f"Columun model {name} h0={h0}, hTB={hTB}", "datetime", "head [m]")
    ax.plot(datetime, hanal[0], color=clr[0], label='hanal[0]') # First layer
    ax.plot(datetime, hanal[1], color=clr[1], label='hanal[1]') # Second layer
    ax.plot(datetime, hanal[2], color=clr[2], label='hanal[2]') # Third layer

    ax.plot(datetime, hnum[0], '.', color=clr[0], label='hanal[0]') # First layer
    ax.plot(datetime, hnum[1], '.', color=clr[1], label='hanal[1]') # Second layer
    ax.plot(datetime, hnum[2], '.', color=clr[2], label='hanal[2]') # Third layer
    
    ax.plot(datetime,h1L, lw=2, label='h1L') # The only layer of this model

    ax.plot(datetime, hTB[0], 'k-',  label='hTB[0]') # Top boundary head
    ax.plot(datetime, hTB[1], 'k--', label='hTB[1]') # Bottom boundary head
    
    ax.legend()
 
    if scen > 100:   # not tested
        dhdt = cmdlnum.dhdt(h0=h0, hTB=hTB, q=q, datetime=datetime)
        ax = newfig(f"Columun model {name} dhdt, h0={h0}, hTB={hTB}", "datetime", "dhdt [m/d]")
        ax.plot(datetime, dhdt[0], label='dhdt[0]')
        ax.plot(datetime, dhdt[1], label='dhdt[1]')
        ax.plot(datetime, dhdt[2], label='dhdt[2]')

        Sc = (cmdlnum.S[0] * cmdlnum.c[0], cmdlnum.S[-1] * cmdlnum.c[-1])

        ax.plot(datetime, hTB[0] / Sc[0], 'k-',  label='dhd1_hTB[0]')
        ax.plot(datetime, hTB[1] / Sc[1], 'k--', label='dhdt_hTB[1]')
    
        ax.legend()
     
    if scen > 100:       
        checkLeakageHantush()
    
    plt.show()
    # %%
    
