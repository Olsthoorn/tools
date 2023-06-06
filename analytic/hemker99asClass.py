"""Implemtation of
   
   Hemker 1999 Transient well flow in vertically heterogeneous aquifers JoH 225 (1999)1-18

   This is still the implemnentation of Hemker-Maas 1087, which needs to be
   converted to Hemker(1999)
   
   TO 2023-04-10 2023-05-13
"""

import os
import sys
tools = os.path.abspath('..')
if not tools in sys.path:
    sys.path.insert(0, tools)

import warnings
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.linalg as la
import scipy.special as sp
from hantush_convolution import Wh
from fdm import Grid
from fdm.fdm3t import fdm3t
from analytic import hantush_convolution
from etc import newfig, color_cycler, line_cycler
import ttim


def stehfest_coefs(N=None):
    """Return the N Stehfest coefficients"""
    v = np.zeros(N, dtype=float)
    for i in range(1, N + 1):
        j1, j2 = int((i + 1) / 2), int(min(i, N / 2))
        for k in range(j1, j2+1):
            v[i - 1] += k ** (N / 2) * sp.factorial(2 * k) / (
                sp.factorial(N / 2 - k) * sp.factorial(k) *
                sp.factorial(k - 1) * sp.factorial(i - k) *
                sp.factorial(2 * k - i)
            )
        v[i - 1] *= (-1) ** (i + N/2)
    return v

vStehfest = stehfest_coefs(N=16)

def sysmat(p=None, kD=None, c=None, S=None,
           top=None, bot=None):           
    """Return the system matrix of the multyaquifer system.
    
    Parameters
    ----------
    p:  float
        laplace variable
    kD: np.ndarray [m2/d]
        vector of n layer transmissivities determins number of layers.
    c:  np.ndarray [d]
        vector of n+1 interlayer vertical hydraulic resistances.
    S:  np.ndarrray [-]
        vector of storage coefficients for the aquifers
    top: str
        top of system is 'conf' or 'semi'
    botclosed: bool
        bottom of system is 'conf' or  'semi'
    """
    n = len(kD)
    if top == 'conf' and len(c) < n:
        c = np.hstack((np.inf, c))
    if bot == 'conf' and len(c) < n + 1:
        c = np.hstack((c, np.inf))
        
    aSt = np.ones_like(c)
    bSt = np.ones_like(c)
    B = - np.diag(aSt[ :-1] / c[ :-1] + aSt[1:] / c[1:], k=0)\
        + np.diag(bSt[1:-1] / c[1:-1], k=+1)\
        + np.diag(bSt[1:-1] / c[1:-1], k=-1)
    if top == 'conf':
        B[0, 0]   = -1. / c[1]
    if bot == 'conf':        
        B[-1, -1] = -1. / c[-2]
    
    if S is None: # steady
        A = np.diag(1 / kD) @ (-B)
    else:
        A = np.diag(1 / kD) @ (p * np.diag(S) - B)
    return A

def sysmat_steady(kD=None, c=None, top=None, bot=None):
    """Return the system matrix of the multyaquifer system.

    Parameters
    ----------
    kD: np.ndarray [m2/d]
        vector of n layer transmissivities determins number of layers.
    c:  np.ndarray [d]
        vector of n+1 interlayer vertical hydraulic resistances.
    top: str
        'conf' or 'semi'
    bot: str
        'conf' or 'semi'
    """
    return sysmat(p=0, kD=kD, c=c, S=np.zeros_like(kD), top=top, bot=bot)

def multRadial(rs=None, Q=None, kD=None, c=None, top=None, bot=None):
    """Return steady state multiaquifer radial flow to well screens with spresribed flows.
    
    Parameters
    ----------
    rs: np.ndarray
        distances for which to compute the drawdown
    Q:  np.ndarray [m3/d]
        vector of layer extractions.
    kD: np.ndarray [m2/d]
        vector of n layer transmissivities determines number of layers.
    c:  np.ndarray [d]
        vector of n+1 interlayer vertical hydraulic resistances.
    topclosed: bool
        top of system is closed (impervious)
    botclosed: bool
        bottom of system is closed (impervious)
    """
    assert top in ['semi', 'conf'], "top must be 'conf' or 'semi'"
    assert bot in ['semi', 'conf'], "bot must be 'conf' or 'semi'"
    
    A   = sysmat_steady(kD=kD, c=c, top=top, bot=bot)
    Tp05   = np.diag(np.sqrt(kD))
    Tm05   = np.diag(1. / np.sqrt(kD))
    
    D   = Tp05 @ A @ Tm05 # makes the A symmetric
    W, R = la.eig(D)
    V   = Tm05 @ R
    Vm1 = la.inv(V)
    
    Tm1 = np.diag(1 / kD)
    
    s = np.zeros((len(kD), len(rs)))
    for ir, r in enumerate(rs):
        K0r  = np.diag(sp.k0(r * np.sqrt(W.real)))
        s[:, ir] = (V @ K0r @ Vm1 @ Tm1 @ Q[:, np.newaxis] / (2 * np.pi)).flatten()
    return s

def sLaplace(p=None, r=None, rw=None, rc=None,
        screens=None, Q=None, kD=None, c=None, S=None, top=None, bot=None):
    """Return the laplace transformed drawdown for t and r scalars.
    
    The solution is the Laplace domain is

    s_(r, p) = 1 / (2 pi p) K0(r sqrt(A(p))) T^(-1) q
    
    With A the system matrix (see sysmat).

    p: float [1/d]
        Laplace variable
    r: float [m]
        Distance from the well center
    rw: float
        well radius
    rc: float
        radius of storage (casing at top in which water table fluctuates)
    Q: float
        total extraction from the aquifers
    screens: np.ndarray
        Vector telling which aquifers are screened
    kD: np.ndarray
        Vector of transmissivities of the aquifers [m2/d]
    c:  np.ndarray
        Vector of the vertical hydraulic resistances of the aquitards [d]
    S:  np.ndarray
        Vector of  the storage coefficients of the aquifers
    r:  float
        Distance at which the drawdown is to be computed
    """
    A   = sysmat(p=p, kD=kD, c=c, S=S, top=top, bot=bot)
    Tp05   = np.diag(np.sqrt(kD))
    Tm05   = np.diag(1. / np.sqrt(kD))
    
    T = np.diag(kD)
    Ti = kD * screens # vector of floats
    Tw = np.sum(Ti)
    One = np.ones((np.sum(screens > 0), 1))
    
    C1 = Q / (2 * np.pi * Tw) # float, constant
    C2 = rc ** 2 / (2 * Tw)   # float, constant
    
    D   = Tp05 @ A @ Tm05
    W, R = la.eig(D)
    V   = Tm05 @ R
    Vm1 = la.inv(V)
    
    K0K1r = np.diag(sp.k0(r * np.sqrt(W.real))/
                    (rw * np.sqrt(W.real) * sp.k1(rw * np.sqrt(W.real))))
    
    E = np.diag(screens)[:, screens != 0]
    assert np.all(E.T @ screens[:, np.newaxis] == np.ones(len(screens[screens > 0]))
                  ), "E.T @ screens must be all ones"
        
    U   = E.T @ V @ (np.eye(len(kD)) + p * C2 * K0K1r) @ Vm1 @ E
    Um1 = la.inv(U)

    s_ = Q / (2 * np.pi * p * Tw) * V @ K0K1r @ V.T @ T @ E @ Um1 @ One
    
    return s_.flatten() # Laplace drawdown s(p)

def assert_input(z0=None, D=None, kr=None, kz=None, c=None, Ss=None, top='conf', bot='conf', **kw):
    """Verify variable except t and r and update kw.
    
    Parameters in kw
    ----------
    z0: float
        top of flow system (top of top aquitard)
    D: np.ndarray [m] of length n
        thickness of aquifers
    kr: np.ndarray [m/d] on flength n
        horizontal conductivity of aquifers
    kz: np.ndarray [m/d] on flength n + 1
        vertical conductivity of aquifers
    c:  np.ndarray [d] of length n-1
        vertical hydraulic resistances
    Ss:  np.ndarray [m2/d] of length n
        specific storage coefficients of aquifers
    top: str
        'conf' or 'semi'
    bot: 'str'
        'conf or 'semi'
    """           
    for name, var in zip(['kr', 'kz', 'D', 'c', 'Ss'],
                         [kr, kz, D, c, Ss]):
        assert isinstance(var, np.ndarray), f"{name} not an np.ndarray"

    assert top == 'semi' or top == 'conf', "top must be 'semi' or 'conf'"
    assert bot == 'semi' or bot == 'conf', "bot must be 'semi' or 'conf'"
    
    nlay = len(D)
    nc   = nlay + 1 if top=='semi' and bot=='semi' else nlay if top=='semi' or bot=='semi' else nlay - 1
    assert len(c) == nc, 'len(c) must equal {}, not {}'.format(nc, len(c))

    for name , v in zip(['D', 'kr', 'kz', 'c', 'Ss'],
                    [D, kr, kz, c, Ss]):
        assert np.all(v >= 0.), f"all {name} must be >= 0."
           
    return

def backtransform(t, r, rw=None, rc=None, Q=None, kD=None,
                  c=None, S=None, screens=None, top=None, bot=None):
    """Return s(t, r) after backtransforming from Laplace solution.

    Parameters
    ----------
    t:  float or np.ndarray [d]
        tinme or times
    r:  float or np.ndarray [m]
        distances to well center
    rw: float [m]
        well radius
    rc: float [m]
        radius of casing or storage part of well in which water table fluctuates
    Q:  float [m3/d]
        Total extracton from well (over all screened aquifers)
    kD: np.ndarray [m2/d] on flength n
        transmissivity
    c:  np.ndarray [d] of length n+1
        vertical hydraulic resistances 
    S:  np.ndarray [m2/d] of length n
        storage coefficients
    screens: np.ndarray [-] length n
       screened aquifers indicated by 1 else 0
    top: str
        'conf' or 'semi'
    tot: str
        'conf' or 'semi'
    """
    s = np.zeros_like(kD)
    for v, i in zip(vStehfest, range(1, len(vStehfest) + 1)):
        p = i * np.log(2.) / t
        s += v * sLaplace(p=p, r=r, rw=rw, rc=rc, Q=Q,
                        kD=kD, c=c, S=S, screens=screens, top=top, bot=bot)
    s *= np.log(2.) / t
    return s.flatten()

def solution(t=None, r=None, Q=None, D=None, kr=None, kz=None, c=None, Ss=None, screens=None, rw=None, rc=None, top=None, bot=None, **kw):
    """Return the multilayer transient solution Maas Hemker 1987
    
    t:  float or np.ndarray [d]
        Time at which the dradown is to be computed
    r:  float or np.ndarray [m]
        Vector of distances from the well center at which drawdown is computed
    kw: dict
     other required and superfluous arguments
        
    Notice
    ------
    Either t or r must be a scalar.
        
    Returns
    -------
    s[:, t] for given r if r is scalar
    or
    s[:, r] for given t if t is scalar
    or
    s[:, 0] for both r and t scalars
    """
    assert_input(D=D, kr=kr, kz=kz, Ss=Ss, c=c, screens=screens, top=top, bot=bot, **kw)

    if np.isscalar(t): t = np.array([t])
    if np.isscalar(r): r = np.array([r])
    if r[0] == 0.: r=r[r:]
    shape = (len(t), len(D), len(r))

    # Add ctop aquitard if given separately
    if 'cTop' in kw and kw['cTop'] is not None: # cTop
        c[0] = kw['cTop']

    # Add the vertical resistance of the layers to c
    Ca = D / kz    
    caq = 0.5 * (Ca[:-1] + 0.5 * Ca[1:])
    nlay  = len(D)
    if top == 'semi':
        c[1:nlay + 1] += caq
    else:
        c[ :nlay    ] += caq
        
    s = np.zeros(shape)
    for it, t_ in enumerate(t):
        for ir, r_ in enumerate(r):
            s[it, :, ir] = backtransform(t=t_, r=r_, rw=rw, rc=rc, Q=Q, kD=kr * D, c=c, S=Ss * D, screens=screens, top=top, bot=bot)
    return s # Drawdown s[it, il, ir]

class Hemker1999:
    """Analytic solution of multilayer axial flow with well storage  by Kick Hemker (1999)
    
    Using a class allows keeping fixed data in memort and make computation faster by
    preventing needless recalculations, which is releant when requiring time series
    possibily combined with solution for many points.
    """
    
    def __init__(self, z0=0., D=None, kr=None, kz=None, c=None, Ss=None,
                 top=None, bot=None, **_):
        
        self.z0 = z0
        
        self.assert_input(D=D, kr=kr, kz=kz, c=c, Ss=Ss, top=top, bot=bot)
        
        return
        
    def simulate_well(self, t=None, r=None, Q=None, rw=None, rc=None, screens=None):
        """Compute the drawdown for all t and r.
        
        The resulting array will be of size s[nt, nlay, nr]
        
        Parameters
        ----------
        t: int or np.ndarray of times
            simulation times
        r: float or np.ndarray of distances
            distance from well center
        Q: float
            well extraction
        rw: float
            well radius
        rc: float
            equivalent caisson radius for well storage
        screens: np.ndarray [-] length len(D)
           screened aquifers indicated by 1 else 0
        """
        self.rw, self.rc = rw, rc
        
        assert isinstance(screens, np.ndarray) and len(screens) == len(self.D), "screens must be np.ndarray of length self.D=={}".format(len(self.D))
        screens[screens != 0] = 1
        
        E = np.diag(screens)[:, screens != 0]
        assert np.all(E.T @ screens[:, np.newaxis] == np.ones(len(screens[screens > 0]))
                      ), "E.T @ screens must be all ones"
        self.screens, self.E = screens, E
        
        self.Ti = self.kD * screens # vector of floats
        self.Tw = np.sum(self.Ti)
        self.One = np.ones((np.sum(screens > 0), 1))
        
        t = np.array([t])
        r = np.array([r])
        t, r = t.flatten(), r.flatten()
        assert isinstance(t, np.ndarray), "t must be a ndarray or a scalar !"
        assert isinstance(r, np.ndarray), "r must be a ndarray or a scalar !"
        
        shape = len(t), self.nlay, len(r)
        
        s = np.zeros(shape)
        for it, t_ in enumerate(t):
            for ir, r_ in enumerate(r):
                s[it, :, ir] = self.backtransform(t=t_, r=r_, Q=Q)
        return s

    def assert_input(self, D=None, kr=None, kz=None, c=None,
                 Ss=None, top=None, bot=None):
        """Check and set self parameters after verifying length of variables.

        Parameters
        ----------
        D: np.ndarray [m] of length n
            thickness of aquifers
        kr: np.ndarray [m/d] on flength n
            horizontal conductivity of aquifers
        kz: np.ndarray [m/d] on flength n + 1
            vertical conductivity of aquifers
        c:  np.ndarray [d] of length n-1
            vertical hydraulic resistances 
        Ss:  np.ndarray [m2/d] of length n
            specific storage coefficients of aquifers
        top: str
            'conf' or 'semi'
        bot: str
            'conf' or 'semi'
        """
        for name, var in zip(['kr', 'kz', 'D', 'Ss'], [kr, kz, D, Ss]):
            assert isinstance(var, np.ndarray), "{} not an np.ndarray".format(name)
        
        nlay = len(D)
        
        assert len(kr) == nlay, "len(kr): ({}) != number of layers ({}) !".format(len(kr), nlay)
        assert len(kz) == nlay, "len(kz): ({}) != number of layers ({}) !".format(len(kz), nlay)
        
        if Ss is not None:
            assert isinstance(Ss, np.ndarray), "Ss not an np.ndarray !"
            assert len(Ss) == nlay, "len(Ss) ({}) != numbr of layers  ({})".format(len(Ss), nlay)
        
        if c is None:
            c = np.zeros(nlay + 1)
        else:
            assert isinstance(c, np.ndarray), "c not an np.ndarray !"

        c = np.zeros(nlay + 1) if c is None else\
            np.hstack((0, c, 0)) if len(c) == nlay - 1 else\
                np.hstack((c, 0)) if len(c) == nlay else c
    
        c[:-1] += 0.5 * (D / kz)
        c[1: ] += 0.5 * (D / kz)
        
        if top == 'conf': c[ 0] = np.inf
        if bot == 'conf': c[-1] = np.inf

        for name , v in zip(['D', 'kr', 'kz', 'c', 'Ss'], [D, kr, kz, c, Ss]):
            assert np.all(v >= 0.), "all {} must be >= 0.".format(name)
                   
        # Below variables are added to self:
        self.nlay = nlay
        self.z = np.hstack((self.z0, self.z0 -  np.cumsum(D)))

        self.D, self.kr, self.kz, self.Ss = D, kr, kz, Ss
        
        self.kD, self.c, self.S = D * kr, c, D * Ss
        self.top, self.bot = top, bot
        
        # Used in analytical method laplace()
        self.T    = np.diag(self.kD)
        self.S    = np.diag(self.S)
        self.Tp05 = np.diag(np.sqrt(self.kD))
        self.Tm05 = np.diag(1 / np.sqrt(self.kD))
        self.Tm1  = np.diag(1 / self.kD)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning) # Division by zero ok
            self.Sm1  = np.diag(1 / self.S)
            
        # This is done in simulate because depends on e, the well
        #self.Ti = self.kD * screens # vector of floats
        #self.Tw = np.sum(self.Ti)
        #self.One = np.ones((np.sum(screens > 0), 1))
        return
    
    def tau2t(self, tau=None, r=None):
        """Return t when  dimensionless time tau is given.
        
        where tau = 1/u = 4 kD t  / (r **2 S)
        
        Parameters
        ----------
        tau: np.ndarray
            values of 4 kD t / (r ** 2 * S) # tau = 1/u, input for Theis
        r: float
            distance from well that is used to calculate t.
        """
        t = r ** 2  * self.S[1:].sum() / (4 * self.kD.sum()) * tau
        return t
    
    
    def sysmat(self, p=None): 
        """Return the system matrix of the multyaquifer system.

        Parameters
        ----------
        p:  float of None
            laplace variable or None for the steady state system matrix
        """
        if hasattr(self, 'B'):
            pass
        else:
            c = self.c
            aSt = np.ones_like(c)
            bSt = np.ones_like(c)
            B = - np.diag(aSt[ :-1] / c[ :-1] + aSt[1:] / c[1:], k=0)\
                + np.diag(bSt[1:-1] / c[1:-1], k=+1)\
                + np.diag(bSt[1:-1] / c[1:-1], k=-1)
            if self.top == 'conf': # Don't need c[0] to be np.inf
                B[0, 0]   = -1. / c[1]
            if self.bot == 'conf': # Don't need c[-1] to be np.inf
                B[-1, -1] = -1. / c[-2]                
            self.B = B

        if p is None: # steady
            A = self.Tm1 @ (-self.B)
        else:
            A = self.Tm1 @ (p * self.S - self.B)
        return A

    def sLaplace(self, p=None, r=None, Q=None):
        """Return the laplace transformed drawdown for t and r scalars.

        The solution is the Laplace domain is

        s_(r, p) = 1 / (2 pi p) K0(r sqrt(A(p))) T^(-1) q

        With A the system matrix (see sysmat).

        p: float [1/d]
            Laplace variable
        r: float [m]
            Distance from the well center
        Q: float
            total extraction from the aquifers
        """
        A    = self.sysmat(p=p)  # depends on p

        # C1 = Q / (2 * np.pi * self.Tw)      # float, constant
        C2 = self.rc ** 2 / (2 * self.Tw)   # float, constant

        D    = self.Tp05 @ A @ self.Tm05 # depends on p
        W, R = la.eig(D)                 # depends on p
        V    = self.Tm05 @ R             # depends on p
        Vm1  = la.inv(V)                 # depends on p

        # K0K1r = np.diag(sp.k0(r * np.sqrt(W.real)) / self.rw * np.sqrt(W.real) * sp.k1(self.rw * np.sqrt(W.real)))
        K0K1r = np.diag(sp.k0(r * np.sqrt(W.real)))
        
        U   = self.E.T @ V @ (np.eye(self.nlay) + p * C2 * K0K1r) @ Vm1 @ self.E
        Um1 = la.inv(U)

        s_ = Q / (2 * np.pi * p * self.Tw) * V @ K0K1r @ V.T @ self.T @ self.E @ Um1 @ self.One

        return s_.flatten() # Laplace drawdown s(p)

    def backtransform(self, t, r, Q=None):
        """Return s(t, r) after backtransforming from Laplace solution.

        Parameters
        ----------
        t:  float [d]
            tinme
        r:  float [m]
            distances to well center
        rw: float [m]
            well radius
        rc: float [m]
            radius of casing or storage part of well in which water table fluctuates
        Q:  float [m3/d]
            Total extracton from well (over all screened aquifers)
        """
        s = np.zeros(self.nlay)
        for v, i in zip(vStehfest, range(1, len(vStehfest) + 1)):
            p = i * np.log(2.) / t
            s += v * self.sLaplace(p=p, r=r, Q=Q)
        s *= np.log(2.) / t
        return s.flatten()

class FdmAquifMdl:
    """Setup a numerical axially symmetric Hemker aquifer system that
    can simulate a well in a multi-layer aqufier system with well-bore
    storage according to Hemker(1999) Journal of Hydrology.
    
    A single extraction is given, while the well can have multiple
    screens. The head in all the screens will be the same.
    
    The well has a radicus rw, but there is also a caisson radius,
    in which the water level fluctuates and causes well bore storage.
    The area of this caisson is assumed to be circular with radius rc.
    An adequate rc can be chosen to match the actual area.
    
    This class uses the finite dfiference model fdm3t in tools.fdm.fdm3t
    module.
    
    The model should work with exactly the same inputs as the
    analytical Hemker solution. This way, the numerical and analytical
    solutions can be easily comopared.
    
    To achieve a central column of radius rw, the distance array r is adapted to 
    
    r = np.array([0, rw, r[r> rw]])
    
    To implement a multi screened well with a single head and single total extractiion, we set all cells of the well column to get kz = np.inf, and then kh=0 at the unscreenee parts 
    and kh=np.inf in the screened parts.
    
    First set up the model, and then create a well, then simulate the model with the well
    finally extract the data from the results.
    """
    def __init__(self, z0=None, d=None, D=None,
                 c=None, kr=None, kz=None, Ss=None, top='conf', bot='conf', **kw):
        """Return numerical axial FDM model object.
        Parameters
        ----------
        zo: float
            elevation of top of the system
        d: None
            all semi-confined layers including top and bottom have resistance but no thickness
        D: np.array
            thickness of aquifer (layers)
        kr: np.array
            horizontal hydraulic conductivities of the layers
        kz: np.array
            vertical hydraulic conductivities of the layers
        c: np.ndaray
            vertical resistances between the layers and the possible semi-pervious top and bottom.
            len(c) == nlay if top == 'semi' or bot == 'semi'
            len(c) == nlay + 1 if top == 'semi' and bot == 'semi'
        Ss: np.ndarray
            Specific storage of the layers
        top, bot: str
            top == 'conf' or 'semi', same for bot.
        """
        assert_input(z0=z0, D=D, kr=kr, kz=kz, c=c, Ss=Ss, top=top, bot=bot)
        
        assert d is None, "d must be None"
        
        self.top, self.bot = top, bot
        self.z0, self.d, self.D = z0, d, D
        self.kr, self.kz, self.c, self.Ss = kr, kz, c, Ss
        
        self.Nlay = len(self.D)
        
        self.Nc = self.Nlay - 1 + (top == 'semi') + (bot == 'semi')
        assert self.Nc == len(self.c), f"len(c) != nc={self.Nc}"
        assert self.Nc == len(c), f"len(c) != nc={self.Nc}, not top=='{top}' and bot='{bot}'"
        
        # Top and bottoms of the layers (no aquitards. Semi-perv. top and bottom have thickness zero)
        self.z = self.z0 - np.hstack((0, np.cumsum(self.D)))
                      
    def simulate_well(self, t=None, r=None, Q=None, rw=None, rc=None, screens=None):
        """Simulate a multi-screen well.
        
        Parameters
        ----------
        t: np.ndarray
            the simulation times
        r: np.ndarray
            coordinates of vertical grid lines
        Q: float
            injection flow, negative for extraction
        rw: float
            well radius
        rc: float
            well caisson radius for well-bore storage
        screens: ndarray of bool
            indicates which layers are screened        
        """
        assert isinstance(screens, np.ndarray), "screens must be an 1D ndarray"
        assert len(screens) == len(self.D), "len(screens) != self.gr.nlay!"
        screens[screens != 0] == 1

        r = np.hstack((0., rw, r[r > rw]))        
        gr = Grid(r, None, self.z, axial=True)
        E = screens[:, np.newaxis, np.newaxis] * np.ones((1, 1, gr.nx), dtype=int)
    
        # Cells that are screen or casing cells
        screen = np.logical_and(gr.XM < rw, E)
        casing = np.logical_and(gr.XM < rw, np.logical_not(E))
        
        kr = gr.const(self.kr)
        kz = gr.const(self.kz)
        Ss = gr.const(self.Ss)
                
        GHB = None
        if self.top =='semi':
            cells = self.gr.const(False); cells[0][1:] == True
            cond = (self.gr.Area / (self.c[0] + 0.5 * self.D[0] / self.kz[0]))[1:]
            GHB = gr.GHB(cells, h=0, cond=cond)
            self.c = self.c[1:]
        if self.bot == 'semi':
            cells = gr.const(False); cells[-1][1:] == True
            cond = (gr.Area / (self.c[-1] + 0.5 * self.D[-1] / self.kz[-1]))[1:]
            ghb = gr.GHB(cells, h=0, cond=cond)
            GHB = ghb if GHB is None else np.vstack((GHB, ghb))
            self.c = self.c[:-1]
        
        c  = gr.const(self.c)
        
        kr[screen] = 1e+6
        kr[casing] = 1e-6 
        kz[screen] = 1e+6
        kz[casing] = 1e+6 

        Ss[:, 0, 0] = 1.0 * (rc / rw) ** 2 / self.D.sum()

        kDscreen = screens * self.kr * self.D
        kDwell   = np.sum(kDscreen)
        
        IBOUND = gr.const(1, dtype=int)

        assert np.isscalar(Q), 'Q must be a scalar'

        HI = gr.const(0.)           # Initial heads
        FQ = gr.const(0.)      # Fixed flows
        FQ[:, 0, 0] = Q * kDscreen / kDwell
        # Run the finite difference model
        out = fdm3t(gr=gr, t=t, kxyz=(kr, kr, kz), Ss=Ss, c=c, GHB=GHB,
                FQ=FQ, HI=HI, IBOUND=IBOUND)
        return out

def hem99fdm(t=None, r=None, Q=None,
             z0=None, D=None, kr=None, kz=None, c=None, Ss=None, screens=None, rw=None, rc=None,
             top=None, bot=None, **_):
    """Setup and run an axially symmetric multilayer model with a multi-screen well in its center.
    
    The idea is to setup an axially symmetric finite difference model with minnimal input
    that will yield results with the same input as an analytic model with the same input.
    
    When calling all input can be provide as a dict, which is partly unpacjked in the head of this function
    leaving extra, essential and non-essential parameters in the remaining **kw.
    
    The well is implemented as the center colum with r between 0 and rw.
    
    To achieve a central column of radius rw, the distance array r is adapted to 
    
    r = np.array([0, rw, r[r> rw]])
    
    To implement a multi screened well with a single head and single total extractiion, we set
    all cells of the well column to get kz = np.inf, and then kh=0 at the unscreenee parts 
    and kh=np.inf in the screened parts.

    The outputs can be readily interpolated to get them for specific times, distances and and model layers or even z values.
    This interpolation and plotting is done using the function showPhi.
        
    Parameters
    ----------
    t: np.array
        simulation times
    r: np.array
        distances from center of well. It will be adapted to [0, rw, r[r>rw]]
    Q: float
        well injection rate (extraction negative)
    z0: flaot
       elevation of top of model
    D : np.ndarray
        Thickness of the model layers
    kw: np.ndarray
        horizontal conductivity of the model layers 
    kz: np.ndarray
        vertical conductivity of the model layers
    c : np.ndaray
        vertical resistances between the layers
        len(c) == len(D) - 1 if top='conf' and bot == 'conf'
        len(c) == len(D) + 1 if top='semi' and bot == 'semi'
        len(c) == len(D) if top == 'semi' or bot == 'semi' but not both
    Ss: np.ndarray
        Specific storage coefficients of the layers
    Screens: np.ndarray of ones and zeros
        Indicates by 1 which layers are screenen and by 0 unscreened
    rw: float
        well radius
    rc: float
        radius of well bore storage, used to calculation Ss in top well cell.
    top: 'semi' or 'conf'
        boundary above top layer
    bot: 'semi' or 'conf'
        boundary below bottom layer
    """
    assert_input(z0=z0, D=D, kr=kr, kz=kz, c=c, Ss=Ss, top=top, bot=bot)
        
    r = np.hstack((0., rw, r[r > rw]))
    z = np.hstack((z0, z0 -  np.cumsum(D)))
    gr = Grid(r, None, z, axial=True)
    
    IBOUND = gr.const(1, dtype=int) # No fixed heads
        
    # Well screen array
    assert isinstance(screens, np.ndarray) and len(screens) == len(D), f"screens must be ndaray of lenght {len(D)} containing only ones and zeros"
    E = screens[:, np.newaxis, np.newaxis] * np.ones((1, gr.nx), dtype=int)
    
    # Distribute the extraction of the screended parts of the well (useful, but necesary) because
    # kz=np.inf in the entire well and
    # kh=0 in cased parts and kh=np.inf in screened parts of the well.
    kDscreen = screens * kr * D
    kDwell   = np.sum(kDscreen)
    
    # Cells that are screen or casing cells
    screen = np.logical_and(gr.XM < rw,     E)
    casing = np.logical_and(gr.XM < rw, np.logical_not(E))
    
    # No horizontal flow in casing, vertical flow in screen and casing
    kr = gr.const(kr); kr[screen] = 1e+6; kr[casing]=1e-6 # inside well
    kz = gr.const(kz); kz[screen] = 1e+6; kz[casing]=1e+6 # inside well
    
    # Resistance between model layers is added in assert_input(kw)
    GHB = None
    if top == 'semi':
        cells = gr.const(0, dtype=bool)
        cells[0, :, 1:] = True
        cond = gr.Area / (c[0] + 0.5 * gr.DZ[0] / kz[0])
        GHB = gr.GHB(cells, h=0, cond=cond[1:])
    if bot == 'semi':
        cells = gr.const(0, dtyp=bool)
        cells[-1, : 1:] = True
        cond = gr.Area / (c[-1]+ 0.5 * gr.DZ[-1] / kz[-1])
        ghb = gr.GHB(cells, h=0, cond=cond[1:])
        GHB = ghb if GHB is None else np.vstack((ghb, GHB))
    
    c = gr.const(c)
    
    # Storage and well bore storage
    Ss = gr.const(Ss[:, np.newaxis, np.newaxis])
    Ss[:, 0, 0] = (rc / rw) ** 2 / np.sum(D) # Well bore storage divided over entire aquifer
    
    # Initial heads and fixed flows
    HI = gr.const(0.) # Initial heads
    FQ = gr.const(0.) # Fixed flows
    
    # Extraction of multi-screened well
    assert np.isscalar(Q)
    FQ[:, 0, 0] = Q * kDscreen / kDwell # Distribute Q over the whole screened part of the well.
    
    # Run the finite difference model
    out = fdm3t(gr=gr, t=t, kxyz=(kr, kr, kz), Ss=Ss, c=c, GHB=GHB,
                FQ=FQ, HI=HI, IBOUND=IBOUND)
    return out

def interpolate(fdm3t, t=None, r=None, z=None, IL=None, method='linear'):
    """Return head for given times, distances and layers.
    
    For the interpolation to work, fdm3t['gr'].shape must be (any, 1, >1)
    
    Most useful is to leave z and IL both None, which ensures all layers are in the result.
    Further leave either t or r None, which yields an array with 
    to select all times for specific r or one with all r for specific times
    
    Parameters
    ----------
    fdm3t: dictionary
        output of fdm.fdrm3t.fdm3t
        Through fdm3t['t] and fdm3t['gr'].xm all t and r of the numerically  computed results are known;
        t and r are different, they are the times and distances for which presentation output is desired.
        If t is None, fdm3t['t] is used.
        If r is None, fdm3t['r].xm is used.
        If z is None the layers in IL are used.
        If z and IL are none, all layers are used.
    t: sequence or scalar or None
        times at which output is desired or None for all times
    r: sequence
        distances at which output is desired or None for all distances
    z: sequence of scalar or None
        z-values for which output is desired or None for specified IL
    Il: sequence if ints or scalar or None
        layers for which output is desired of none for specified (interpolated) zs is use
    method: 'linear', 'cubic' or 'spline'
        interpolation method
        
    returns
    -------
    PhiI: np.ndarray
        Interpolated heads dimension (Nt, Nz, Nr)
        where Nt corresponds to fdm3t['t] if t is None and to t if not.
        Where Nz correspoinds to fdm3t['gr']['nz'] if both z and IL are None
        or to z if z z is not None or to IL if IL is not None
    """
    assert (IL is None or z is None) and not (
                (IL is not None) and (z is not None)
            ), 'Either `IL` or `z` must be none !'
    assert (not (t is None and r is None)
            and not (
                (t is not None and r is not None))
            ), 'Either `t` or `r` must be none!'

    gr = fdm3t['gr']

    # Option 1: if time is not specified, all times is implied
    if t is None:
        t =fdm3t['t']
    else:
        t = np.array([t]) if np.isscalar(t) else np.array(t)
        
    if r is None:
        r = gr.xm
    else:
        r = np.array([r]) if np.isscalar(r) else np.array(r)
    
    if z is None:
        if IL is None:
            IL = np.arange(gr.nlay, dtype=int)
        else:
            IL = np.array([IL], dtype=int) if np.isscalar(IL) else np.array(IL, dtype=int)
            assert np.all(np.logical_and(IL >= -1, IL < gr.nlay))
    else:
        z = np.array([z]) if np.isscalar(z) else np.array(z)
        assert np.all(np.logical_and(z <= gr.zm[0], z >= gr.zm[-1])),\
            "All z must be {:.4g}>=z>={:.4g} m".format(gr.z[0], gr.z[-1])
        
    Phi = fdm3t['Phi'][:, :, 0, :] # sqeeze y

        
    points = fdm3t['t'], -gr.zm, gr.xm
    interp = scipy.interpolate.RegularGridInterpolator(
            points, Phi, method=method,
            bounds_error=True, fill_value=np.nan)
    
    zeta = -z if IL is None else -gr.zm[IL]
    T, Z, R = np.meshgrid(t, zeta, r, indexing='ij')
    PhiI = interp(np.vstack((T.ravel(), Z.ravel(), R.ravel())).T).reshape(T.shape)
    
    return PhiI
    
def showPhi(fdm3t=None, t=None, r=None, z=None, IL=None, method=None, 
              xlim=None, ylim=None, xscale=None, yscale=None, show=True, **kw):
    """Return head for given times, distances and layers.
    
    Parameters
    ----------
    fdm3t: dictionary
        output of fdm.fdrm3t.fdm3t
        Through fdm3t['t] and fdm3t['gr'].xm all t and r of the numerically  computed results are known;
        t and r are different, they are the times and distances for which presentation output is desired.
        If t is None, fdm3t['t] is used.
        If r is None, fdm3t['r].xm is used.
        If z is None the layers in IL are used.
        If z and IL are none, all layers are used.
    t: sequence or scalar or None
        times at which output is desired or None for all times, i.e. for fdm3t['t]
    r: sequence
        distances at which output is desired or None for all distances
    z: sequence of scalar or None
        z-values for which output is desired or None for specified IL
    Il: sequence if ints or scalar or None
        layers for which output is desired of none for specified (interpolated) zs is use
    method: 'linear', 'cubic' or 'spline'
        interpolation method
    show: bool
        if False only return PhiI
        if True, interpolate and show and return Phi and ax
    
    Options:
    if t is None and z is not None:
        (z, r) combinations for all t
        option = 1
    elif t is None and IL is not None
        (il, r) combinations for all t
        option = 2
    elif r is None and z is not None:
        (t, z) combinations for all r
        option = 3
    elif r is None and IL is not None:
        (t, il) combinations for all r
        option = 4
    else:
        raise ValueError illegal combination
    """
    
    PhiI = interpolate(fdm3t, t=t, r=r, z=z, IL=IL, method=method)
    
    if showPhi == False:
        return PhiI
    
    # Option 1: if time is not specified, all times is implied
    if t is None:
        t =fdm3t['t']
        if z is not None:
            option = 1 # Zs
        else:
            option = 2 # IL
            if IL is None:
                IL = np.arange(fdm3t['gr'].nlay, dtype=int)
    else: # t not None --> r None)
        r = fdm3t['gr'].xm
        if z is not None:
            option = 3
        else:
            option = 4
                    
    t = np.array([t]) if np.isscalar(t) else np.array(t)
    r = np.array([r]) if np.isscalar(r) else np.array(r)
    z = np.array([z]) if np.isscalar(z) else np.array(z)

    graphOpts = {
        1: {'title': '(z, r) comb. for all t',   'xlabel': 'time [d]', 'ylabel': 'h [m]',
            'label': 'z={:.4g} m, r={:4g} m'},
        2: {'title': '(lay, r) comb. for all t', 'xlabel': 'time [d]', 'ylabel': 'h [m]',
            'label': 'layer={}, r={:4g} m'},
        3: {'title': '(z, t) comb. for all r',   'xlabel': 'r [m]', 'ylabel': 'h [m]',
            'label': 't={:.4g} d, z={:4g} m'},
        4: {'title': '(lay, t) comb. for all r', 'xlabel': 'r [m]', 'ylabel': 'h [m]',
            'label': 't={:4g} d, layer={}'},
    }
    
    o = graphOpts[option]
    ax = newfig(o['title'], o['xlabel'], o['ylabel'], xlim=xlim, ylim=ylim,
                xscale=xscale, yscale=yscale)

    if option == 1:
        for iz, z_ in enumerate(z):
            for ir, r_ in enumerate(r):
                ax.plot(t[1:], PhiI[1:, iz, ir], label=o['label']
                        .format(z_, r_))
    if option == 2:
        for il, iL_ in enumerate(IL):
            for ir, r_ in enumerate(r):
                ax.plot(t[1:], PhiI[1:, il, ir], label=o['label']
                        .format(iL_,r_))                    
    if option == 3:
        for it, t_ in enumerate(t):
            for iz, z_ in enumerate(z):
                ax.plot(r[1:], PhiI[it, iz, 1:], label=o['label']
                        .format(t_, z_))
    if option == 4:
        for it, t_ in enumerate(t):
            for il, iL_ in enumerate(IL):
                ax.plot(r[1:], PhiI[it, iL_, 1:], label=o['label']
                        .format(t_, iL_))                    

    ax.legend(loc='best')
    return PhiI, ax


def ttimz(z0=0., d=None, D=None, top='conf'):
    """Return top and bottof of aquifers including top aquitard if topclosed==False.
    
    Parameters
    ----------
    z0: float, optional
        top elevation of the layer system
    d: sequence of floats or None
        thicknesses of the aquitards (zeros allowed) or None instead.
    D: sequence of floats
        thicknesses of the aquifers (zeros not allowed)
    top: 'conf' or 'semi'
        boundary condition above top layer
    'bot': 'conf' or 'semi'
        boundary condtion below the bottom layer
        Has no effect in ttim, 'bot' is always 'conf'.
    Returns
    -------
    vector of the tops and bottoms of the aquifers.
    In case len(d) == len(D) - 1, the top is confined (topclosed=True)
    In case len(d) == len(D) the top is semi-confined. The top d is then
    the thickness of the top aquitard.
    """
    if d is not None:
        if top == 'conf':
            assert len(d) == len(D) - 1
        elif top == 'conf':
            assert len(d) == len(D)
        else:
            raise ValueError("top must be 'conf' or 'semi'")
    else:
        d = np.zeros(len(D))
        
    if len(d) == len(D) - 1:
        d = np.hstack((0., d))
        
    assert np.isscalar(z0), 'z0 (top of model elevation) must be a scalar.'
    
    z = z0 - np.hstack((0., np.cumsum(np.array([z for t in zip(d, D) for z in t]))))
    if top == 'conf':
        z = z[1:]
    return z

def ttimc(c=None, kz=None, D=None, top='conf'):        
    """Return proper c-arrray for ttim ModelMaq.
    
    Parameters
    ----------
    c: None, or scalar or sequence
        The aquitard resistances including top semi-confined layer
        if topclosed==False.
    kz: np.ndarray
        vertical aquifer conductances
    D: np.ndarray
        aquifer thicknesses
    top: 'conf' or 'semi'
        boundary type above top layer
    """
    assert top == 'semi' or top == 'conf', "'top' must be 'semi' or 'conf'"
    
    if top == 'semi':
        assert len(c) == len(D),\
            "top' == 'semi' len(c) must equal len(D) else len(c) must equal len(D) - 1"
      
    caq = np.zeros(len(D)) if kz is None else D / kz
    caq = 0.5 * (caq[:-1] + caq[1:])
    
    if top == 'semi':
        c[1:] += caq
    else:
        c += caq
        c += caq
    return c


class ttimmdl:
    
    def __init__(self, z0=0., d=None, D=None, kr=None, kz=None, c=None, Ss=None, top=None, bot=None,
           tmin=1e-5, tmax=1e3, M=10, **_):
        """Use ttim to simulate analytically.

        ttim.ModelMaq(
            kaq: Any = 1, z: Any = [1, 0],
            c: Any = [],
            Saq: Any = [0.001],
            Sll: Any = [0], poraq: Any = [0.3],
            porll: Any = [0.3],
            topboundary: str = 'conf' | 'semi',
            phreatictop: bool = False,
            tmin: int = 1,
            tmax: int = 10,
            tstart: int = 0,
            M: int = 10, # number of coefficiens in Laplace back transformation
        timmlmodel: Any | None = None) -> None
        """
        phreatictop=False
        self.D = D

        c = ttimc(c=c, D=D, kz=kz, top=top)
        z = ttimz(z0=z0, d=d, D=D, top=top)

        self.mdl = ttim.ModelMaq(kaq=kr, z=z, Saq=Ss, c=c, 
                        phreatictop=phreatictop, tmin=tmin, tmax=tmax, M=M)


    def simulate_well(self, t=None, r=None, Q=None, screens=None, rw=None, rc=None):
        """Return heads of simulated well at x,y = 0.0.
        
        Parameters
        ----------
        t: float or np.ndarray
            simulation times
        r: float or np.ndarray
            distances from the well center
        Q: float
            total well extraction
        screens: np.ndarray of zeros and ones
            ones indicate which layers are screened
        rw: float
            well radius
        rc: float
            equivalent caisson radius, to compute wellbore storage
        """
        screens = np.arange(len(self.D), dtype=int)[screens > 0]
        
        _ = ttim.Well(self.mdl, xw=0., yw=0., rw=rw, rc=rc,
                          tsandQ=[(0, -Q)], layers=screens)
        self.mdl.solve()

        if np.isscalar(r):
            return self.mdl.head(x=r, y=0, t=t)
        else:
            return self.mdl.headalongline(x=r, y=0, t=t)

def test0(kw):
    """Compare numeric and analytical results
    for parameters in cases['test0']
    """
    
    for name in ['D', 'kr', 'kz', 'c', 'Ss', 'screens', 'rw', 'rc', 'top', 'bot']:
        if name not in kw:
            raise ValueError(f"{name} not in input (kw)")
    
    t = np.logspace(-1, 4, 91)
    r = [1, 10, 100]
    Q = 1000.
    rw, rc, screens = kw['rw'], kw['rc'], kw['screens']
    
    # Use analytic class implementation =================================================
    hem1 = Hemker1999(**kw)
    sclass  = hem1.simulate_well(t=t, r=r, Q=Q, rw=rw, rc=rc, screens=screens)
    
    # Using analytic function implementation ============================================
    sfunc  = solution(t=t, r=r, Q=Q, **kw)
    
    # Using ttim  =============================================================
    tim1 = ttimmdl(M=len(vStehfest), tmin=t[0], tmax=t[-1], **kw)
    sttim = tim1.simulate_well(t=t, r=r, Q=Q, rw=rw, rc=rc, screens=screens)

    # Fdm as function implementation =========================================
    R = np.logspace(-1, 6, 71)
    out = hem99fdm(t=t, r=R, Q=Q, **kw)
    PhiFdmFunc = interpolate(out, t=None, r=r)

    ## Fdm as class ============================================================
    fdmmdl = FdmAquifMdl(**kw)
    out = fdmmdl.simulate_well(t=t, r=R, Q=Q, rw=rw, rc=rc, screens=screens)
    PhiFdmClass  = interpolate(out, t=None, r=r)
    #
    ax = newfig(kw['title'], 'time [d]', 's [m]', xscale='log')
    cc = color_cycler()
    for ir, r in enumerate(r):
        clr = next(cc)
        for iL in range(len(kw['D'])):
            ax.plot(t, sclass[:, iL, ir], '-', color=clr, label=f'scl, iL={iL}, r = {r:.3g} m')
            ax.plot(t, sfunc[ :, iL, ir], '+', color=clr, label=f'sfu, iL={iL}, r = {r:.3g} m')
            ax.plot(t, sttim[ iL, :, ir], 'x', color=clr, label=f'stt, iL={iL}, r = {r:.3g} m')            
            ax.plot(t, PhiFdmFunc[:, iL, ir], 'o', color=clr, label=f'sfdmf, il={iL}, r = {r:.3g} m')
            ax.plot(t, PhiFdmClass[:, iL, ir], '.', color=clr, label=f'fdmc, il={iL}, r = {r:.3g} m')
            ax.plot(t, PhiFdmClass[:, iL, ir], '.', color='y')
    ax.legend(loc='lower right')

def test1(kw):
    """Simulate Theis in a 4L numerical model and visualize it with showPhi in different four ways.
    
    Simulates a Theis case and shows the different options to present it using showPhi.
    
    For the parameters see the cases dict for this example.
    """
    # Numerical (model has 4 layers)
    
    D  = kw['D'].sum()
    kD = (kw['kr'] * kw['D']).sum()
    S  = (kw['Ss'] * kw['D']).sum()
    kw['Q'] = 4 * np.pi * kD
    
    ax = newfig('Theis in multilayer model', r'$\tau = 4 kD / (r^2 S) t$', r'$s_D = s 4 pi kD / Q$', xscale='log', yscale='log',
                xlim=(1e-3, 1e3), ylim=(1e-6, 1e1))
    
    ax.plot(kw['tau'], sp.exp1(1/kw['tau']),  'kx', lw=1, label='Theis')
    
    hem = Hemker1999(**kw)
    
    # Analytic for r values of r but all times
    rs = [5, 50, 500]
    lc = line_cycler()
    for r_ in rs:
        ls = next(lc)
        t = r_ ** 2 * S / (4 * kD) * kw['tau']
        kw['t'] = t
        
        sa  = hem.simulate(t=t,r=r_, Q=kw['Q'])
        out = hem99fdm(**kw)
        
        PhiI = interpolate(out, t=None, r=r_, z=None, IL=None)
        cc = color_cycler()
        for iL in [0, -1]:
            clr = next(cc)
            ax.plot(kw['tau'][1:], PhiI[1:, iL, :], ls, color=clr,
                    label='numeric:  r = {} m, iL = {}'.format(r_, iL))
            ax.plot(kw['tau'], sa[:, iL, :], '+', color=clr,
                    label='analytic: r = {} m, iL = {}'.format(r_, iL))
        
        # Same but now for all layers
        PhiI = interpolate(out, t=None, r=r_, z=None, IL=None)
        cc = color_cycler()
        for iL in range(out['gr'].nlay):
            clr = next(cc)
            ax.plot(kw['tau'][1:], PhiI[1:, iL, 0], '--',  color=clr, label='numeric:  r = {:.4g} m, layer {}'.format(r_, iL))
            ax.plot(kw['tau'], sa[  :, iL, 0], '.', color=clr, label='analytic: r = {:.4g} m, layer {}'.format(r_, iL))
    ax.legend(loc='lower right')
    
    # Get the data for only three times and all r
    ts = np.array([1. , 3., 10.])
    ts[ts < out['t'][0]] = out['t'][0]
    ts[ts > out['t'][-1]] = out['t'][-1]
    zs = [-5, -15, -25, -35]
    z = -15.
    PhiIA = interpolate(out, t=ts, r=None, z=z,    IL=None)
    PhiIB = interpolate(out, t=ts, r=None, z=None, IL=None)
    
    ax = newfig('Theis in layers versus r/D', r'$r/D$', r'$s 4 \pi kD / Q$', xscale='log')
    for it, t in enumerate(ts):
        r = out['gr'].xm[1:]
        u = r ** 2 * S / (4 * kD * t)
        ax.plot(r / D, sp.exp1(u), '.', label='Theis, t={:.4g}'.format(t))
        ax.plot(r / D, PhiIA[it, 0, 1:], '-', label='t={:.4g} d, z={:.4g} m'.format(t, z))
        for il in range(out['gr'].nlay):
            ax.plot(r / D, PhiIB[it, il, 1:], '--', label='t={:.4g} d, il={} m'.format(t, il))
    ax.legend(loc='lower right')
            

def boulton_analytical(kw):
    """Return plot of Boulton's problem analytic implementation as class."""
    
    TAUMIN = 0.1 # below which Stehfest breaks (tested experimentally, N Stehfest = 16, is best)
    
    assert_input(kw)
  
    r_ = kw['r_']    
  
    kD = (kw['kr'] * kw['D'])[-1]
    Q  = 4 * np.pi * kD
    kw['Q'] = Q
    
    S  = kw['Ss'] * kw['D']
    Sy, SA = S[0], S[1]
    
    tauA = kw['tau']
    tauB = tauA * SA / (Sy + SA)
    
    kw['t'] = r_  ** 2 * SA * tauA / (4 * kD) 
    
    # rho = r_ over B values used by Hemker (1999)
    rhos = np.array([0.01, 0.1, 0.2, 0.4, 0.6,
                         0.8, 1.0, 1.5, 2.0, 2.5, 3.0])
    rhos = np.array([0.01, 1.0, 1.5, 3.])
    
    # xlim, ylim = None, None # reveal Stehfest breaks down for small tau
    xlim, ylim = (0.8 * TAUMIN, 1e9), (1e-4, 1e2)
    
    ax = newfig(kw['title'] + r'N-Stehfest={}, $T=r^2 S_A/(4 kD)={:.4g} d$'.
                format(len(vStehfest), r_ ** 2 * SA / (3 * kD)),
                r"$\tau = 4 kD t / (r^2 S)$",
                r"$(4 \pi kD) / Q s$",
                xscale='log', yscale='log',
                xlim=xlim, ylim=ylim)
    ax.plot(tauA, scipy.special.exp1(1/tauA), 'r-',
            label='Theis for S=SA')
    ax.plot(tauA, scipy.special.exp1(1/tauB), 'b-',
            label='Theis for S=(Sy + SA)')
    
    cc = color_cycler()
    kw['r'] = np.array([r_])
    for rho in rhos:
        color = next(cc)
        c = np.zeros(len(kw['D']) - 1)
        c[0]  = (r_ / rho) ** 2 / kD
        kw['c'] = c
        kw['r'] = [r_]
            
        # Using the function
        sb  = solution(**kw)
        
        # Using the class
        h99obj = Hemker1999(**kw)        
        t = kw['t']
        tau = kw['tau']
        t = h99obj.tau2t(r=r_, tau=tau)
    
        sa  = h99obj.simulate(t, r_, Q)
        
        # Using fdm
        fdmmdl = FdmAquifMdl(**kw)
        fdmwell = FdmWell(fdmmdl, **kw)
        out = fdmwell.simulate(t=kw['t'], Q=kw['Q'])
        hfdm  = out['Phi'][:, 0, :]

        # using ttim simulation
        httim = ttmmdl(**kw)
        
        lbl = f'r/B = {rho:.4g}'
        lc = line_cycler()
        for il in range(h99obj.nlay):
            ls = next(lc)
            ax.plot(tau[tau > TAUMIN], ls, sa[:, il, 0][tau > TAUMIN], 
                    color=color, label='sa ' + lbl)
            ax.plot(tau[tau > TAUMIN], ls, sb[:, il, 0][tau > TAUMIN],
                    color=color, label='sb ' + lbl)
            ax.plot(tau[tau > TAUMIN], ls, hfdm[tau > TAUMIN, il, 0],
                    color=color, label='fdm ' + lbl)
            ax.plot(tau[tau > TAUMIN], ls, httim[il][tau > TAUMIN],
                    color=color, abel='ttm ' + lbl)
        
    ax.legend(loc='lower right')
    print("Done, test succeeded !")
    return ax

def h99_F07_Szeleky(kw):
    """Simulate figure 07 in Hemker (1999), case Sceleky."""
    # TODO: Not yet right
    rs = [0.1, 10.]
    kD = np.sum(kw['kr'] * kw['D'])
    S  = np.sum(kw['Ss'] * kw['D'])
    tau = kw['tau']
    
    ax = newfig(kw['name'], r'$tau = 4 kD / (r^2 S) t$', r'$s [m]$',
        xlim=(1e-2, 1e6), ylim=(1e-3, 1e1), xscale='log', yscale='log')
    
    for ir, r_ in enumerate(rs):
        kw['t'] = tau * r_ ** 2 * S / (4 * kD)
        kw['rc'] = kw['rw']
        
        out = hem99fdm(**kw)
        PhiI = interpolate(out, t=None, r=r_, z=None, IL=None)
            
        for il in [3]:
            ax.plot(tau, PhiI[:, il, 0], '-', label='rc={:.4g} r = {:.4g} m, layer = {}'.format(kw['rc'],r_, il))
        
        kw['rc'] = 0.0    
        out = hem99fdm(**kw)
        PhiI = interpolate(out, t=None, r=r_, z=None, IL=None)
    
        for il in [3]:
            ax.plot(tau, PhiI[:, il, 0], '--', label='rc={:.4g} r = {:.4g} m, layer = {}'.format(kw['rc'],r_, il))


    
        #sa = solution(r=kw['r'], **kw)
    
    ax.legend(loc='lower right')
    return ax
  
def h99_f11(kw):
    """Simulate and return figure 11 in Hemker(1999)."""
    
    etop, ebot = kw['screens']
    D  = np.sum(kw['D'])
    kD = np.sum(kw['kr'] * kw['D'])
    S  = np.sum(kw['D'] * kw['Ss'])
    kw['Q'] = 4 * np.pi * kD
    
    r_ = 0.2 * D
    tau = np.logspace(-3, 5, 81) # A is aquifer (layer 1)
    
    kw['t'] = r_  ** 2 * S * kw['tau'] / (4 * kD)
    kw['screens'] = etop
    out = hem99fdm(**kw)
    PhiItop = interpolate(out, t=None, r=[r_], z=None, IL=None)

    kw['screens'] = ebot
    out = hem99fdm(**kw)
    PhiIbot = interpolate(out, t=None, r=[r_], z=None, IL=None)
    
    title = ('Drawdown responses to a partially penetrating well\n' +
             'in the lowest part (solid) or the uppermost part (dashed)\n' +
             'of a five-layer Ss- heterogeneous aquifer.')
    ax = newfig(title=title, xlabel=r'$\tau = 4 kD t / (r^2 S)$', ylabel=r'$s 4 \pi kD / Q$',
                xscale='log', yscale='log', xlim=(1e-2, 1e4), ylim=(1e-3, 1e2))
    
    cc = color_cycler()
    for il in range(PhiItop.shape[1]):
        color = next(cc)                  
        ax.plot(tau[:], PhiItop[:, il, 0], '-' , color=color, marker='x', label='top, layer {}'.format(il + 1))
        ax.plot(tau[:], PhiIbot[:, il, 0], '--', color=color, marker='+', label='bot, layer {}'.format(il + 1))
    
    ax.legend(loc='lower right')
    return ax

def Hantush(kw):
    """Simulate Hantush with the same input as for boulton (1963).
    
    The only difference being this situation and Boulton is
    that with Hantush, the head in layer 0 is fixed and with Boulton it is not.
    """
    
    r_ = 500.    
    # Using a fixed r_ makes that we use the same times
    # for each r_ over B, but requires adapting B and, hence,
    # adapting c, so that we have the desirede values of r_ over B

    # r_ over B values used by Hemker (1999)
    r_B = np.array([0.01, 0.1, 0.2, 0.4, 0.6,
                         0.8, 1.0, 1.5, 2.0, 2.5, 3.0])
    
    kD = np.sum(kw['kr'][1:] * kw['D'][1:])        
    Sy = kw['D'][0] * kw['Ss'][0]
    SA = np.sum(kw['D'][1:] * kw['Ss'][1:])
    kw['c'] = np.zeros(len(kw['D']) - 1) # adapted in loop below
    kw['Q'] = 4 * np.pi * kD
    
    # tau used by Hemker, tauA is tau for the aquifer
    tauA = kw['tau'] # A is aquifer (layer 1)
    tauB = tauA * SA / Sy       
    kw['t'] = r_  ** 2 * SA * tauA / (4 * kD) 
    
    title ='{}, type curves for r/B from 0.01 to 3'\
                .format(kw['name'])
    xlabel = r'$\tau = 4 kD t /(r^2 S_2)$'
    ylabel = r'$\sigma = 4 \pi kD s / Q$' 
    ax = newfig(title, xlabel, ylabel,
                ylim=(1e-3, 1e2), xlim=(1e-1, 1e9),
                xscale='log', yscale='log')
    
    # The two Theis curves
    ax.plot(tauA, scipy.special.exp1(1/tauA), 'r', lw=3, label='Theis for SA')
    ax.plot(tauA, scipy.special.exp1(1/tauB), 'b', lw=3, label='Theis for Sy + SA')
            
    cc = color_cycler()
    for rho in r_B:
        color = next(cc)
        
        # Adapt c to to get the right r/B            
        B = r_ / rho
        kw['c'][0] = np.array([B ** 2 / kD])
        
        hem = Hemker1999(**kw)
        sa = hem.simulate(t=kw['t'], r=rho * B, Q=kw['Q'])
        
        out = hem99fdm(**kw) # using correct c
            
        # Interpolate numerical to set of times layers and distances:
        PhiI = interpolate(out, t=None, r=rho * B, z=None, IL=None, method='linear')

        assert kw['topclosed'] == False, 'topclosed must be False for numerical Hantush!'
        ax.plot(tauA, hantush_conv.Wh(1/tauA, rho)[0], '-', color=color,
                label='Wh(tau, {:.4g})'.format(rho))

        # Plot Theis for S = Sy (specific yield)Show the drawdown in the top and first layer for this r/B
        for il in [0, 1]:            
            if rho in [0.01, 1.0, 1.5, 3.]:
                labelN = 'num: r/B = {:.4g}'.format(rho)
                labelA = 'ana: r/B = {:.4g}'.format(rho)
                lw = 2.
            else:
                labelN = '_'
                labelA = '_'
                color = 'k'
                lw = 0.5
            ax.plot(tauA[:], PhiI[:, il, 0], '+', color=color, lw=lw, label=labelN)
            ax.plot(tauA[:], sa[  :, il, :], 'x', color=color, lw=lw, label=labelA)
    
    ax.legend(loc='lower right') 
    return ax   

def h99_F2_Boulton(kw):
    """Simulalte boulton (1963) delayed yield or just Hantush
    
    The only difference being that with Hantush, the head in layer 0 is fixed.
    
    Boulton = Hantush with topclosed == True
    """    
    r_ = 500.
    # Using a fixed r_ makes that we use the same times
    # for each r_ over B, but requires adapting B and, hence,
    # adapting c, so that we have the desirede values of r_ over B
    
    # r_ over B values used by Hemker (1999)
    r_B = np.array([0.01, 0.1, 0.2, 0.4, 0.6,
                         0.8, 1.0, 1.5, 2.0, 2.5, 3.0])

    kD = np.sum(kw['kr'][1:] * kw['D'][1:])        
    Sy = kw['D'][0] * kw['Ss'][0]
    SA = np.sum(kw['D'][1:] * kw['Ss'][1:])
    kw['c'] = np.zeros(len(kw['D']) - 1) # adapted in loop below
    kw['Q'] = 4 * np.pi * kD

    # tau used by Hemker, tauA is tau for the aquifer
    tauA = kw['tau'] # A is aquifer (layer 1)
           
    kw['t'] = r_  ** 2 * SA * tauA / (4 * kD) 
    
    # tau values for the aquifer with Sy instead of SA
    tauB = tauA * SA / Sy
    
    title ='{}, type curves for r/B from 0.01 to 3'\
                .format(kw['name'])
    xlabel = r'$\tau = 4 kD t /(r^2 S_2)$'
    ylabel = r'$\sigma = 4 \pi kD s / Q$' 
    
    ax = newfig(title, xlabel, ylabel,
                ylim=(1e-3, 1e2), xlim=(1e-1, 1e9),
                xscale='log', yscale='log')
    
    # The two Theis curves (hor axis both tauA (tau for aquifer storage (elastic storage)))
    ax.plot(tauA, scipy.special.exp1(1/tauA), 'r', lw=3, label='Theis for SA')
    ax.plot(tauA, scipy.special.exp1(1/tauB), 'b', lw=3, label='Theis for Sy + SA')
            
    cc = color_cycler()
    rNum = kw['r']
    for rho in r_B:
        color = next(cc)
        
        # Adapt c to to get the right r/B            
        B = r_ / rho
        kw['c'][0] = np.array([B ** 2 / kD])
        kw['r'] = rNum
        out = hem99fdm(**kw) # using correct c
        
        # Analytisch:
        kw['r'] = rho * B
        
        hem = Hemker1999(**kw)
        Q = 4 * np.pi * hem.kD.sum()
        t = hem.tau2t(r=r_, tau=kw['tau'])
        
        sa = hem.simulate(t=kw['t'], r=rho * B, Q=kw['Q'])
    
        # Interpolate numerical to set of times layers and distances:
        PhiI = interpolate(out, t=None, r=rho * B, z=None, IL=None, method='linear')

        assert kw['topclosed'] == True, 'topclosed must be true for Boulton'
        
        for il in [0, 1]:
            if rho in [0.01, 1.0, 1.5, 3.]:
                labelN = 'num: r/B = {:.4g}'.format(rho)
                labelA = 'ana: r/B = {:.4g}'.format(rho)
                lw = 2.
            else:
                labelN = '_'
                labelA = '_'
                color = 'k'
                lw = 0.5
            ax.plot(tauA[:], PhiI[:, il, 0], '-', color=color, lw=lw, label=labelN)
            ax.plot(tauA[:], sa[:,il], 'x', color=color, lw=lw, label=labelA)
    
    ax.legend(loc='lower right') 
    return ax   
        
def h99_F3(kw):
    """Simulate case of Moench with vertical resistance within aquifer and on top."""
    # The problem is essentially the same as Boulton's, however
    # the delayed yield is also due to vertical anisotropy within the aquifer.
    
    tau = kw['tau']
    D  = np.sum(kw['D'][1:])
    cA = np.sum(kw['D'][1:] / kw['kz'][1:])
    kD = np.sum(kw['D'][1:] * kw['kr'][1:])
    Sy = kw['D'][0] * kw['Ss'][0]
    SA = 1e-3 * Sy
    kw['Ss'][1:] = SA / D
    kw['Q'] = 4 * np.pi * kD

    r_ = kw['rd'] * D # m, fixed

    ax1 = newfig(kw['title'] + '\n' +r'kD={:.4g} cA={:.4g}, Sy={:.4g}, SA={:.4g}, r/D = {:.4g} m'.format(
        kD, cA, Sy, SA, r_ / D), r'$\tau = 4 kD t /(r^2 S_2)$', r'$\sigma = 4 \pi kD s / Q$',
         ylim=(1e-2, 1e1), xlim=(1e-1, 1e5),
         xscale='log', yscale='log')
    
    ax2 = newfig(kw['title'] + '\n' +r'kD={:.4g}, cA={:.4g}, Sy={:.4g}, SA={:.4g}, r/D = {:.4g} m'.format(
        kD, cA, Sy, SA, r_ / D), r'$\tau = 4 kD t /(r^2 S_2)$', r'$\sigma = 4 \pi kD s / Q$',
         ylim=(1e-2, 1e1), xlim=(1e-1, 1e5),
         xscale='log', yscale='log')

    ax1.plot(tau,         scipy.special.exp1(1/tau), 'b--', lw=1, label='Theis for SA')
    ax1.plot(tau * Sy/SA, scipy.special.exp1(1/tau), 'r--', lw=1, label='Theis for Sy + SA')
    ax2.plot(tau,         scipy.special.exp1(1/tau), 'b--', lw=1, label='Theis for SA')
    ax2.plot(tau * Sy/SA, scipy.special.exp1(1/tau), 'r--', lw=1, label='Theis for Sy + SA')
    
    for gamma in [1., 10., 100.]:
        cT = cA / gamma
        kw['c'] = np.zeros(len(kw['D']) - 1); kw['c'][0] = cT
        B = np.sqrt(kD * cT)
         
        kw['t'] = r_  ** 2 * SA * tau / (4 * kD)
        
        # Numemrical
        out = hem99fdm(**kw)
        PhiI = interpolate(out, t=None, r=r_, z=None, IL=None)
        ax1.plot(tau, PhiI[:,  1, 0], '-', label=r'numerical: lay=  1, $\gamma$={:.4g}, r/B={:.4g}'.format(gamma, r_/B))
        ax2.plot(tau, PhiI[:, 20, 0], '-', label=r'numerical: lay= 20, $\gamma$={:.4g}, r/B={:.4g}'.format(gamma, r_/B))
        
        # Analytical
        hem = Hemker1999(**kw)
        sa = hem.simulate(t=kw['t'], r=r_, Q=kw['Q'])
        ax1.plot(tau, sa[:,  1, 0], 'x', label=r'analytical: lay= 1, $\gamma$={:.4g}, r/B={:.4g}'.format(gamma, r_/B))
        ax2.plot(tau, sa[:, 20, 0], 'x', label=r'analytical: lay 20, $\gamma$={:.4g}, r/B={:.4g}'.format(gamma, r_/B))
        
    ax1.legend(loc='lower right')
    ax2.legend(loc='lower right')
    return
    
def h99_F6(kw):
    """Simlate fig 6 in Hemker (1999) well storage with partially penetrating aquifer."""
    # Papadopoulos and Cooper (1967)
    assert kw['topclosed'] == False, 'topclosed must be False for {}'\
        .format('H99_F6')
    
    RRw = [1., 5., 20., 100., 1000.]  # R/Rw
    RRw = [1., 1000.]
    RD = [0.001, 0.1, 0.5]           # R/D
    
    tau = np.logspace(-2, 6, 81) # A is aquifer (layer 1)
    D  = np.sum(kw['D'])
    kD = np.sum(kw['kr'] * kw['D'])
    S  = np.sum(kw['Ss'] * kw['D'])
    Q  = 4 * np.pi * kD
    
    kw['Q'] = Q
    
    title =kw['name']
    xlabel = r'$\tau = 4 kD t /(r^2 S)$'
    ylabel = r'$\sigma = 4 \pi kD s / Q$' 
    ax = newfig(title, xlabel, ylabel,
                ylim=(1e-3, 1e2), xlim=(1e-1, 1e6),
                xscale='log', yscale='log')
    
    lc = line_cycler()
    for rd in RD:
        cc = color_cycler()
        ls = next(lc)
        r = rd * D
        for rrw in RRw:
            color=next(cc)        
            rw = r / rrw    
            kw['rw'], kw['rc'] = rw, rw
            kw['t'] = r  ** 2 * S * tau / (4 * kD)
            out = hem99fdm(**kw)
            PhiI = interpolate(out, t=None, r= r, z=None, IL=None)
            ax.plot(tau, PhiI[:, 1, 0], ls, color=color,
                    label='numerical: r/rw={:.4g}, r/D = {:.4g}, r={:.4g} m, rw={:.4g} m'.format(rrw, rd, r, rw))

            hem = Hemker1999(**kw)
            sa = hem.simulate(t=kw['t'], r=r, Q=Q)
            ax.plot(tau, sa[:, 1, 0], '+', color=color,
                    label='analytic:  r/rw={:.4g}, r/D = {:.4g}, r={:.4g} m'.format(rrw, rd, r))
    
    ax.legend(loc='lower right')
    return ax
        
def boulton_well_storage(kw):
    """Simulate Boulton's solution for well storage."""
    
    assert kw['topclosed'] == True, 'topclosed must be true for Boulton'
    B  = 500.
    
    kD = kw['kr'][1] * kw['D'][1]        
    Sy = (kw['D'] * kw['Ss']).sum()
    SA = kw['D'][1] * kw['Ss'][1]
    Q  = 4 * np.pi * kD; kw['Q'] = Q
    
    kw['c'] = np.array([B ** 2 / kD])
    
    tauA = kw['tau']   # tau = 4 kD t  / (r ** 2 * S)
    tauB = kw['tau'] * SA / Sy
    
    title ='{}'.format(kw['title'])
    xlabel = r'$\tau = 4 kD t /(r^2 S_A)$'
    ylabel = r'$\sigma = 4 \pi kD s / Q$' 
    ax = newfig(title, xlabel, ylabel,
                ylim=(1e-3, 1e2), xlim=(1e-2, 1e6),
                xscale='log', yscale='log')
    
    ax.plot(tauA, scipy.special.exp1(1/tauA), '-', color='k', lw=2, label='Theis for SA')
    ax.plot(tauA, scipy.special.exp1(1/tauB), '-', color='grey', lw=2, label='Theis for Sy')
    
    #r_B = np.array([0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0])
    r_B = np.array([0.1, 0.3, 1.0, 3.0]) * 0.1
    rrw = np.array([1, 3, 10, 30])
    
    for rc in rrw * kw['rw']:
        kw['rc'] = rc
        cc = color_cycler()
        for rho in r_B:
            color = next(cc)
            
            r_ = B * rho
            
            kw['t'] = r_  ** 2 * SA * tauA / (4 * kD) 
            
            out = hem99fdm(**kw)
        
            PhiI = interpolate(out, t=None, r= r_, z=None, IL=None)
            
            for il in [0, 1]:
                ls = '--' if il == 0 else '-'
                ax.plot(tauA, PhiI[:, il, 0], ls, color=color, label='il={} r_={:.4g}'.format(il, r_))
        
    ax.legend(loc='lower right')
    return ax

cases ={ # Numbers refer to Hemker Maas (1987 figs 2 and 3)
    'test0': {
        'comment': """Test comparing analytic and numeric results
        """,
        'title': """Compare both analytical implementations.
        They are the same, but Stehfest fails at large times.""",
        'tau': None,
        'z0': 0.,
        'rw': 0.1,
        'rc': 0.1,
        'D' : np.array([10., 10., 10., 10.]),
        'kr' : np.array([1e-6, 10., 10., 10.]),        
        'kz': np.array([ 10., 10., 10., 10.]),
        'Ss': np.array([1., 1., 1., 1.,]) * 1e-6,
        'c' : np.array([1., 0., 0.,]),
        'screens' : np.array([0, 1, 1, 1]),
        'top': 'conf',
        'bot': 'conf',
        },
    'test1': {
        'comment': """Simulate Theis in a 4L numerical model with fully penetrating well and visualize it with showPhi in different four ways.
        """,
        'title': 'Theis numerically 4L model visualized in four different ways with showPhi',
        'tau': np.logspace(-1, 3., 41),
        'r': np.hstack((0., np.logspace(-3., 4., 71))), # r --> [0, rw, rPVC ...
        'z0': 0.,
        'rw': 1e-3,
        'rc': 0.,
        'D' : np.array([10., 10., 10., 10.]),
        'kr' :np.array([10., 10., 10., 10.]),        
        'kz': np.array([ 1.,  1.,  1.,  1.]),
        'Ss': np.array([ 1.,  1.,  1.,  1.,]) * 1e-5,
        'c' : np.array([ 0.,  0.,  0.,]),
        'screens' : np.array([ 1,   1,   1,   1]),
        'topclosed': True,
        'botclosed': True,
        'label': 'Test input',
        },
    'boulton_analytical': {
        'comment': """Test2 compares the analytic solution computed using
        functions and the analytic solution computed using the class.
        It applies both implementations on the Boulton example. The test
        reveils there's no difference in the outcomes.
        It shows that for low values of tau, Stehfest back transformation breaks down.
        """,
        'title': 'Boulton analytically using a two-layer model.',
        't' : None,
        'tau': np.logspace(-2, 9, 121),
        'r' : np.hstack((0., np.logspace(-1, 6, 141))),
        'r_': 5000.,
        'z0': 0.,
        'rw': 0.01,
        'rc': 0.01,
        'Q' : None,
        'D' : np.array([1., 10.]),
        'kr': np.array([1e-6, 1e+0]),
        'kz': np.array([1e+6, 1e+6]),
        'Ss': np.array([1e-1, 1e-6]),
        'c' : None,
        'screens':  np.array([0, 1]),
        'topclosed': True,
        'botclosed': True,
        'label': 'Boulton (1963)',
        },
    'Hantush': {
        'name': 'Hantush using one layer',
        'comment': """Hantush (1955) type curves.
        """,
        't' : None,
        'tau': np.logspace(-2, 9, 121),
        'r' : np.hstack((0., np.logspace(-1, 6, 141))),
        'z0': 0.,
        'rw': 0.01,
        'rc': 0.01,
        'Q' : 4 * np.pi * 10,  # Q = 4 pi kD
        'D' : np.array([1., 10.]),
        'kr': np.array([1e-6, 1e+0]),
        'kz': np.array([1e+6, 1e+6]),
        'Ss': np.array([1e-1, 1e-6]),
        'c' : np.array([9e+3]),
        'screens':  np.array([0., 1]),
        'topclosed': False,
        'botclosed': True,
        'label': 'Hantush (1955)',
    },
    'Boulton': {
        'name': 'Bouton (1963/64) Delayed yield',
        'comment': """Boulton's delayed yield model (1963) can be
        repro- duced by a two-layer (two-aquifers-one-aquitard) confined
        system. The top layer only represents a reservoir that produces
        the delayed drainage. Horizontal flow is impeded here by a
        near-zero radial conductivity. The ratio of the time-dependent
        drainage and the quantity of water remaining to be drained (the
        delay-index 1/alpha) is defined by cSy9=c2S1) in the two-layer
        model, and so the drainage factor B in Boulton's model is
        analogous to a leakage factor defined as sqrt(k2 D2 c2) in the
        twolayer model.
        In Fig. 2 dimensionless drawdown 4 pi kD / Q is plotted versus
        dimensionless time, expressed here as 4 kD t(r^2 S2) for
        different values of r/B. The resulting set of curves are
        identical to the Boulton type curves (1963). Further
        curve-fitting revealed that in a similar graph of Kruseman
        and de Ridder (1979) the intermediate time segment of the
        r/B = 3.0-curve levels at too low a value, 6e-2 instead of 7e-2. Fig. 2 also shows the calculated response of the top layer which, when multiplied with Sy (S1, specific yield), represents the volume of water drained in Boulton\'s model. The straight-line segments with a 1/1 slope prove that, until
        t = cSy and within a distance of 3B, drainage is approximately
        linear with time. The accurate reproduction of Boulton's results
        verifies all main components of the multilayer calculations
        except the finite difference approxima- tion, since vertical
        flow in the saturated zone is neglected.
        """,
        't' : None,
        'tau': np.logspace(-2, 9, 121),
        'r' : np.hstack((0., np.logspace(-1, 6, 141))),
        'z0': 0.,
        'rw': 0.01,
        'rc': 0.01,
        'Q' : None,
        'D' : np.array([1., 10.]),
        'kr': np.array([1e-6, 1e+0]),
        'kz': np.array([1e+6, 1e+6]),
        'Ss': np.array([1e-1, 1e-6]),
        'c' : None,
        'screens':  np.array([0, 1]),
        'topclosed': True,
        'botclosed': True,
        'label': 'Boulton (1963)',
        },
    'H99 F3 Moench': {
        'title': 'Moench 1995/96 using 1 drainage layer and 20 sublayers',
        'comment': """A more realistic model of unconfined well flow is
        obtained when time-dependent drainage from above the water table
        (Boulton's 1963 model) and vertical flow within the saturated
        zone (Neuman's 1974 model) are both considered (Narasimhan and Zhu, 1993). Fig. 3 compares the theoretical responses of an
        analytical solution developed by Moench (1995, 1996) with the
        multilayer results for three models with a decreasing effect of
        delayed yield and for a water-table piezometer and another
        piezometer located at the base of the aquifer. The dimensionless
        parameter gamma is introduced by Moench and may be defined in
        terms of multilayer model properties as the ratio of two
        resistances: the vertical flow resistance in the aquifer D/k2
        and the delayed drainage resistance (c2). The multilayer model is set up as 20 sublayers of the same thickness for the aquifer and an additional top layer for delayed drainage, as described
        for Boulton's model.
        The response curves at the water table coincide almost
        perfectly, while the deep piezometer shows drawdowns that are
        slightly too small, increase with g (by decreasing c2) and are
        especially obvious during intermediate time. The errors of less
        than 5% are mainly caused by the conditions in the lowest
        sublayer: the multilayer approximation finds an average drawdown
        for the deepest sublayer, while the analytical solution is
        computed for the true base of the aquifer.
        """,     
        't' : None,
        'tau': np.logspace(-2, 5, 71),
        'r' : np.hstack((0., np.logspace(-1, 6, 141))),
        'rd': 10, # r/D
        'z0': 1.0,
        'rw': 0.1,
        'rc': 0.0,
        'D' : np.ones(21),
        'kr': np.hstack((1e-6, np.ones(20))),
        'kz': np.hstack((1e+6, np.ones(20) * 1e-2)),
        'Ss': np.hstack((1e-1, np.ones(20) * 1e-4)), 
        'screens':  np.hstack((0, np.ones(20, dtype=int))),
        'topclosed': True,
        'botclosed': True,
        'label': 'Moench (1995/95)',
        },
    'H99 F4 Moench': {
        'title': 'Moench 1995/96 using 1 drainage layer and 20 sublayers, partially penetrating well',
        'comment': """A more realistic model of unconfined well flow is
        obtained when time-dependent drainage from above the water table
        (Boulton's 1963 model) and vertical flow within the saturated
        zone (Neuman's 1974 model) are both considered (Narasimhan and Zhu, 1993). Fig. 3 compares the theoretical responses of an
        analytical solution developed by Moench (1995, 1996) with the
        multilayer results for three models with a decreasing effect of
        delayed yield and for a water-table piezometer and another
        piezometer located at the base of the aquifer. The dimensionless
        parameter gamma is introduced by Moench and may be defined in
        terms of multilayer model properties as the ratio of two
        resistances: the vertical flow resistance in the aquifer D/k2
        and the delayed drainage resistance (c2). The multilayer model is set up as 20 sublayers of the same thickness for the aquifer and an additional top layer for delayed drainage, as described
        for Boulton's model.
        The response curves at the water table coincide almost
        perfectly, while the deep piezometer shows drawdowns that are
        slightly too small, increase with g (by decreasing c2) and are
        especially obvious during intermediate time. The errors of less
        than 5% are mainly caused by the conditions in the lowest
        sublayer: the multilayer approximation finds an average drawdown
        for the deepest sublayer, while the analytical solution is
        computed for the true base of the aquifer.
        """,     
        't' : None,
        'tau': np.logspace(-2, 5, 71),
        'r' : np.hstack((0., np.logspace(-3, 6, 141))),
        'rd': 1., # r/D
        'z0': 1.0,
        'rw': 0.1,
        'rc': 0.0,
        'D' : np.ones(21),
        'kr': np.hstack((1e-6, np.ones(20))),
        'kz': np.hstack((1e+6, np.ones(20) * 1e-2)),
        'Ss': np.hstack((1e-1, np.ones(20) * 1e-4)), 
        'screens':  np.hstack((0, np.ones(5, dtype=int), np.zeros(15, dtype=int))), # partially penetrating well
        'topclosed': True,
        'botclosed': True,
        'label': 'Moench (1995/95)',
        },
    'Boulton Well Bore Storage': {
        'title': 'Boulton Well Bore Storage with Delayed Yield',
        't' : None,
        'tau': np.logspace(-2, 6, 81),
        'r' : np.hstack((0., np.logspace(-2, 8, 101))),
        'z0': 1.,
        'rw': 0.128,
        'rc': 0.128,
        'Q' :  None,
        'D' : np.array([ 0.01, 10,]),
        'kr': np.array([    1, 10]),
        'kz': np.array([   10, 10]),
        'Ss': np.array([   10, 1e-4]),
        'c' : np.array([100]),
        'screens':  np.array([0, 1]), # Upper or Lower Layer is screened
        'topclosed': True,
        'botclosed': True,
        'label': 'Boulton (19??)',
        },
    'H99 F4': {
        'name': 'Partially penetrating well with vertically varying Ss', 
        't' : None,
        'tau': np.logspace(-2, 7, 91),
        'r' : np.hstack((0., np.logspace(-1, 6, 141))),
        'z0': 0.,
        'rw': 0.128,
        'rc': 0.128,
        'Q' : 4 * np.pi * 10,  # Q = 4 pi kD
        'D' : np.array([10., 10., 10., 10., 10.]),
        'kr': np.array([10., 10., 10., 10., 10.]),
        'kz': np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        'Ss': np.array([7., 5., 3.5, 2.5, 2.]) * 1e-5,
        'c' : None,
        'screens':  np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]]), # Upper or Lower Layer is screened
        'topclosed': True,
        'botclosed': True,
        'label': 'Boulton (19??)',
        },
    'H99 F6 well bore storage ppw': {
        'name': """Wellbore storage: P&S (1967), B&S (1976).
        Stehfest fails for small tau and small r in the top layer.""",
        'comment':"""To verify multi- layer results, a more complicated solution for flow to a
        partially penetrating well with wellbore storage, recharged by a no-drawdown top boundary
        (Boulton and Streltsova, 1976) was used. The plotted family of type curves (Fig. 6) are
        dimensionless drawdowns in the second layer of a five-layer model with storativity S = 1e-3 at dimensionless distances of 0.001, 0.1 and 0.5, discharged by a well of various diameters, screened in the upper three layers. 
        """,
        't' : None,
        'tau': np.logspace(-2, 5, 71),
        'r' : np.hstack((0., np.logspace(-5, 4, 91))),
        'z0': 1.0,
        'rw': 0.01,
        'rc': 0.01,
        'Q' : None,
        'D' : np.ones(5),
        'kr': np.ones(5),
        'kz': np.ones(5) * 1e-1,
        'Ss': np.ones(5) * 0.2e-3, 
        'c' : None, # depends on gamma
        'screens':  np.array([1, 1, 1, 0, 0], dtype=int),
        'epsilon' : 1.,
        'topclosed': False,
        'botclosed': True,
        'label': 'Papadopoulos & Cooper (1967)',
    },
    'H99_F7 Szeleky': {
        'name': """Székely (1995) Heterogeneous aquifer, PP well.
        The effect of well-bore storage is shown. It's large for rc=0.1 and
        r=0.1 and vertually absent for r=10 m.""",
        'comment':
        """Szekely\'s (1995) example of an analytical solution for a partially penetrating well
        is used to test the hybrid analytical-numerical method under hetero- geneous conditions.
        The conductivity and the specific storage of the upper half of the confined aquifer
        (Kr = 10, Kz = 1 m/d, Ss = 1e-4 /m), are a factor ten higher than in the lower half.
        The pumping wel (Q = 1000 m3/d, rw = 0.1 m) is screened in the uppermost quarter of a 40 m
        thick aquifer, while the piezometer is located in the lowest quarter at r = 10 m, Fig. 7
        shows the drawdown as a function of tD for the pumping well and the piezometer for three
        cases: the analytical solution and multilayer solutions with and without wellbore storage.
        Only four sublayers were used for the multilayer model, but the correspondence with the
        analytical solution is so close, that the individual curves are hard to distinguish,
        except for the early-time segment of the piezometer. The effect of the wellbore storage,
        although prominent in the well, has practically disappeared at the base of the aquifer.
        """,
        't' : None,
        'tau': np.logspace(-3, 6, 91),
        'r' : np.hstack((0., np.logspace(-0.1, 6, 71))),
        'z0': 0.,
        'rw': 0.1,
        'rc': 0.1,
        'Q' : 1.0e3,
        'D' : np.array([10., 10., 10., 10.]),
        'kr': np.array([10., 10.,  1.,  1.]),
        'kz': np.array([10., 10.,  1.,  1.]) * 1e-1,
        'Ss': np.array([10., 10.,  1.,  1.]) * 1e-5,            
        'c' : None,
        'screens' : np.array([1, 0, 0, 0]),
        'topclosed': True,
        'botclosed': True,
        'label': 'Székely (95)',
        },
    'H99_F11': {
        'name': 'PPW, heterogeneous specific storage Ss',
        'comment': 
            """The aquifer is confined, K-homogeneous, anisotropic (Kr/Kz =10) and consists of five
            sublayers of equal thickness, with Ss-values that decrease with depth proportionally as
            14:10:7:5:4. Ratios are based on a shallow, 50 m thick aquifer with a storativity of 0.002
            and estimated Ss-values for the sublayers of 7., 5., 3.5, 2.5 and 2. x 1e-5.
            """,
        'tau': np.logspace(-3, 5, 81),
        'r' : np.hstack((0., np.logspace(-1, 6, 141))),
        'z0': 0.,
        'rw': 0.05,
        'rc': 0.05,
        'D' : np.array([10., 10., 10., 10., 10.]),
        'kr': np.array([10., 10., 10., 10., 10.]),
        'kz': np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        'Ss': np.array([7., 5., 3.5, 2.5, 2.]) * 1e-5, # [/m]        
        'screens' :  np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]]), # Upper or lower Layer is screened        
        'topclosed': True,
        'botclosed': True,
        'label': 'Hemker (1999) fig 11',
        },
}
 
if __name__ == "__main__":
      
    test0(cases['test0']) # ok
    #test1(cases['test1']) # ok
    #boulton_analytical(cases['boulton_analytical']) # ok
    #Hantush(cases['Hantush']) # ok
    #h99_F2_Boulton(cases['Boulton']) # ok
    #h99_F3(cases['H99 F3 Moench']) # ok
    #h99_F3(cases['H99 F4 Moench']) # ok
    #h99_F6(cases['H99 F6 well bore storage ppw']) # not yet ok, but looks a bit like in the paper
    #boulton_well_storage(cases['Boulton Well Bore Storage']) # ok
    #h99_f11(cases['H99_F11']) # ok note that tau uses 4 kD / (r^2 S) t while Hemker kD / (r^2 S) t
    #h99_F07_Szeleky(cases['H99_F7 Szeleky']) #ok
    
    plt.show()
