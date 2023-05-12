"""Implemtation of
   
   Hemker 1999 Transient well flow in vertically heterogeneous aquifers JoH 225 (1999)1-18

   This is still the implemnentation of Hemker-Maas 1087, which needs to be
   converted to Hemker(1999)
   
   TO 2023-04-10
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.special as sp
from hantush_conv import Wh
from fdm import Grid
from fdm.fdm3t import fdm3t
from analytic import hantush_conv
from etc import newfig, color_cycler, line_cycler
import scipy


def stehfest_coefs(N=10):
    """Return the N Stehfest coefficients"""
    v = np.zeros(N)
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

vStehfest = stehfest_coefs(N=10)

def sysmat(p=None, kD=None, c=None, S=None,
           topclosed=False, botclosed=True, **kw):           
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
    topclosed: bool
        top of system is closed (impervious)
    botclosed: bool
        bottom of system is closed (impervious)
    """
    n = len(kD)
    if topclosed and len(c) < n:
        c = np.hstack((np.inf, c))
    if botclosed and len(c) < n + 1:
        c = np.hstack((c, np.inf))
        
    aSt = np.ones_like(c)
    bSt = np.ones_like(c)
    B = - np.diag(aSt[ :-1] / c[ :-1] + aSt[1:] / c[1:], k=0)\
        + np.diag(bSt[1:-1] / c[1:-1], k=+1)\
        + np.diag(bSt[1:-1] / c[1:-1], k=-1)
    if topclosed:
        B[0, 0]   = -1. / c[1]
    if botclosed:        
        B[-1, -1] = -1. / c[-2]
    
    if S is None: # steady
        A = np.diag(1 / kD) @ (-B)
    else:
        A = np.diag(1 / kD) @ (p * np.diag(S) - B)
    return A

def sysmat_steady(**kw):
    """Return the system matrix of the multyaquifer system.

    Parameters
    ----------
    kD: np.ndarray [m2/d]
        vector of n layer transmissivities determins number of layers.
    c:  np.ndarray [d]
        vector of n+1 interlayer vertical hydraulic resistances.
    topclosed: bool
        top of system is closed (impervious)
    botclosed: bool
        bottom of system is closed (impervious)
    """
    kw.update(p=0, S=None)
    return sysmat(**kw)

def multRadial(rs=None, Q=None, kD=None, c=None, topclosed=False, botclosed=True, **kw):
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
    A   = sysmat_steady(kD=kD, c=c, **kw)
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
        e=None, Q=None, kD=None, c=None, S=None, **kw):
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
    e: np.ndarray
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
    A   = sysmat(p=p, kD=kD, c=c, S=S, **kw)
    Tp05   = np.diag(np.sqrt(kD))
    Tm05   = np.diag(1. / np.sqrt(kD))
    
    T = np.diag(kD)
    Ti = kD * e # vector of floats
    Tw = np.sum(Ti)
    One = np.ones((np.sum(e > 0), 1))
    
    C1 = Q / (2 * np.pi * Tw) # float, constant
    C2 = rc ** 2 / (2 * Tw)   # float, constant
    
    D   = Tp05 @ A @ Tm05
    W, R = la.eig(D)
    V   = Tm05 @ R
    Vm1 = la.inv(V)
    
    K0K1r = np.diag(sp.k0(r * np.sqrt(W.real))/
                    (rw * np.sqrt(W.real) * sp.k1(rw * np.sqrt(W.real))))
    
    E = np.diag(e)[:, e != 0]
    assert np.all(E.T @ e[:, np.newaxis] == np.ones(len(e[e > 0]))
                  ), "E.T @ e must be all ones"
        
    U   = E.T @ V @ (np.eye(len(kD)) + p * C2 * K0K1r) @ Vm1 @ E
    Um1 = la.inv(U)

    s_ = Q / (2 * np.pi * p * Tw) * V @ K0K1r @ V.T @ T @ E @ Um1 @ One
    
    return s_.flatten() # Laplace drawdown s(p)

def assert_input(t=None, r=None, z0=None, D=None, kr=None, kz=None, c=None,
                 Ss=None, e=None, **kw):
    """Return r, t after verifying length of variables.
    
    Parameters
    ----------
    t:  float or np.ndarray [d]
        time or times, analytical solution
    r:  float or np.ndarray [m]
        distances to well center, analytical solution
    z0: float
        top of flow system (top of top aquitard)
    rw: float [m]
        well radius
    rc: float [m]
        radius of casing or storage part of well in which water table fluctuates
    D: np.ndarray [m] of length n
        thickness of aquifers
    k: np.ndarray [m/d] on flength n
        horizontal conductivity of aquifers
    kz: np.ndarray [m/d] on flength n + 1
        vertical conductivity of aquifers
    c:  np.ndarray [d] of length n-1
        vertical hydraulic resistances 
    Ss:  np.ndarray [m2/d] of length n
        specific storage coefficients of aquifers
    e: np.ndarray [-] length n
       screened aquifers indicated by 1 else 0
    """
    assert r is not None, "r must not be None"
    
    if np.isscalar(r): r = [r]
    if r is not None: r = np.array(r)
        
    if c is None:
        c = np.zeros_like(D)[:-1]
    
    for name, var in zip(['kr', 'kz', 'D', 'c', 'Ss', 'e'],
                         [kr, kz, D, c, Ss, e]):
        assert isinstance(var, np.ndarray), "{} not an np.ndarray".format(name)
        kw[name] = var
    
    assert len(D) == len(kr),  "Len(D) {} != len(k) {}".format(len(D), len(kr))
    assert len(c) == len(D) - 1 , "Len(c)={} != len(D) - 1 = {}".format(len(c), len(D) - 1)
    
    for name , v in zip(['D', 'kr', 'kz', 'c', 'Ss'],
                    [D, kr, kz, c, Ss]):
        assert np.all(v >= 0.), "all {} must be >= 0.".format(name)
    
    assert np.all(np.array([ee == 0 or ee == 1 for ee in e], dtype=bool)),\
            'All values in e must be 0 or 1'
    
    kw['z0'] = z0
    kw['z' ] = np.hstack((z0, z0 -  np.cumsum(D)))
    kw['kD'] = D * kr
    kw['C']  = 0.5 * (D / kz)[:-1] + c + 0.5 * (D / kz)[1:]
    kw['S']  = D * Ss
    
    kD = np.sum(kw['kD'][1:])
    
    # Simulation time and Q
    if kw['Q'] is None:
        kw['Q'] = 4 * np.pi * kD
    
    if t is None:
        t = kw['Q'] / (4 * np.pi * kD) * kw['tau']
    elif np.isscalar(t):
        t = np.array([t])
    else:
        assert isinstance(t, np.ndarray), \
            't must be np.ndarray as this point, not {}'.format(type(t))
        
    assert isinstance(t, np.ndarray) and isinstance(r, np.ndarray),\
        "t and r must both be vectors, at this point."
        
    kw['shape'] = (len(t), len(D), len(r))

    return t, r, kw

def backtransform(t, r, rw=None, rc=None, Q=None, kD=None,
                  c=None, S=None, e=None, **kwargs):
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
    e: np.ndarray [-] length n
       screened aquifers indicated by 1 else 0
    """
    s = np.zeros_like(kD)
    for v, i in zip(vStehfest, range(1, len(vStehfest) + 1)):
        p = i * np.log(2.) / t
        s += v * sLaplace(p=p, r=r, rw=rw, rc=rc, Q=Q,
                        kD=kD, c=c, S=S, e=e, **kwargs)
    s *= np.log(2.) / t
    return s.flatten()

def solution(t=None, r=None, **kw):
    """Return the multilayer transient solution Maas Hemker 1987
    
    ts:  float or np.ndarray [d]
        Time at which the dradown is to be computed
    rs:  float or np.ndarray [m]
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
    t, r, kw = assert_input(t=t, r=r, **kw)
    
    n = len(kw['D'])
    c = kw['c']

    kw['c'] = np.hstack((0, c, 0))\
        if len(c) == n - 1 else np.hstack((c, 0))\
            if len(c) == n else c
    
    kw['c'][:-1] += 0.5 * (kw['D'] / kw['kz'])
    kw['c'][1: ] += 0.5 * (kw['D'] / kw['kz'])
    
    
    s = np.zeros(kw['shape'])
    for it, t_ in enumerate(t):
        for ir, r_ in enumerate(r):
            s[it, :, ir] = backtransform(t=t_, r=r_, **kw)
    return s # Drawdown s[it, il, ir]

class Hemker1999:
    """Analytic solution of multilayer axial flow with well storage  by Kick Hemker (1999)
    
    Using a class allows keeping fixed data in memort and make computation faster by
    preventing needless recalculations, which is releant when requiring time series
    possibily combined with solution for many points.
    """
    
    def __init__(self, z0=0., rw=None, rc=None, D=None, kr=None, kz=None, c=None, Ss=None, e=None,
                 topclosed=True, botclosed=True, **kw):
        
        self.z0 = z0
        self.rw = rw
        self.rc = rc
        
        self.assert_input(D=D, kr=kr, kz=kz, c=c, Ss=Ss, e=e, topclosed=topclosed, botclosed=botclosed)
        
        return
        
    def simulate(self, t=None, r=None, Q=None):
        """Compute the drawdown for all t and r.
        
        The resulting array will be of size s[nt, nlay, nr]
        """
        if np.isscalar(t): t = np.array([t])
        if np.isscalar(r): r = np.array([r])
        t, r = t.flatten(), r.flatten()
        assert isinstance(t, np.ndarray), "t must be a ndarray or a scalar !"
        assert isinstance(r, np.ndarray), "r must be a ndarray or a scalar !"
        
        shape = len(t), self.nlay, len(r)
        
        s = np.zeros(shape)
        for it, t_ in enumerate(t):
            for ir, r_ in enumerate(r):
                s[it, :, ir] = self.backtransform(t=t_, r=r_, Q=Q)
        return s

    def assert_input(self, D=None, kr=None, kz=None, c=None, e=None,
                 Ss=None, topclosed=True, botclosed=False):
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
        e: np.ndarray [-] length n
           screened aquifers indicated by 1 else 0
        topclosed: bool
            True of aquifer system is closed on top.
            Steady state may need topclosed = False.
        botclosed: bool
            True of aquifer system is closed at bottom.
            Steady stage might need botclosed = False
        """
        for name, var in zip(['kr', 'kz', 'D', 'Ss', 'e'], [kr, kz, D, Ss, e]):
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
        
        if topclosed: c[ 0] = np.inf
        if botclosed: c[-1] = np.inf

        for name , v in zip(['D', 'kr', 'kz', 'c', 'Ss'], [D, kr, kz, c, Ss]):
            assert np.all(v >= 0.), "all {} must be >= 0.".format(name)

        assert np.all(np.array([ee == 0 or ee == 1 for ee in e], dtype=bool)),\
                   'All values in e must be 0 or 1'
                   
        E = np.diag(e)[:, e != 0]
        assert np.all(E.T @ e[:, np.newaxis] == np.ones(len(e[e > 0]))
                      ), "E.T @ e must be all ones"

        # Below variables are added to self:
        self.nlay = nlay
        self.z = np.hstack((self.z0, self.z0 -  np.cumsum(D)))

        self.D, self.kr, self.kz, self.Ss = D, kr, kz, Ss
        self.e, self.E = e, E
        self.kD, self.c, self.S = D * kr, c, D * Ss
        self.topclosed, self.botclosed = topclosed, botclosed
        
        # Used in analytical method laplace()
        self.T    = np.diag(self.kD)
        self.S    = np.diag(self.S)
        self.Tp05 = np.diag(np.sqrt(self.kD))
        self.Tm05 = np.diag(1 / np.sqrt(self.kD))
        self.Tm1  = np.diag(1 / self.kD)
        self.Sm1  = np.diag(1 / self.S)
        
        self.Ti = self.kD * e # vector of floats
        self.Tw = np.sum(self.Ti)
        self.One = np.ones((np.sum(e > 0), 1))
        return
    
    def tau2t(self, tau=None, r=None):
        """Return t when  dimensioinless tau is given.
        
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
            if self.topclosed: # Don't need c[0] to be np.inf
                B[0, 0]   = -1. / c[1]
            if self.botclosed: # Don't need c[-1] to be np.inf
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

        K0K1r = np.diag(sp.k0(r * np.sqrt(W.real))/
                        (self.rw * np.sqrt(W.real) * sp.k1(self.rw * np.sqrt(W.real))))

        U   = self.E.T @ V @ (np.eye(self.nlay) + p * C2 * K0K1r) @ Vm1 @ self.E
        Um1 = la.inv(U)

        s_ = Q / (2 * np.pi * p * self.Tw) * V @ K0K1r @ V.T @ self.T @ self.E @ Um1 @ self.One

        return s_.flatten() # Laplace drawdown s(p)

    def backtransform(self, t, r, Q=None, **kwargs):
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
            s += v * self.sLaplace(p=p, r=r, Q=Q, **kwargs)
        s *= np.log(2.) / t
        return s.flatten()

def hemk99numerically(t=None, r=None, z=None, rw=None, rc=None, topclosed=True, botclosed=True, **kw):
    """Check Hemker(1999) numerically.
    
    Setup and run an axially symmetric multilayer model with a multi-screen well in its center.
    
    The idea is to setup a model that will yield results with the same input as an analytic model.
    
    The well is implemented the center colum with r between 0 and rw in which all cells get kv = np.inf
    and in which the unscreend cells get kh=0 and the screend cells also kv=np.inf. Instead of np.inf
    we may use a high value.
    
    The distance array rc is adapted to make sure it is r = np.array([0, rw, r[r> rw]])
    
    The outpus can be interpolated to get them for specific times, r and model layers or even z.
    
    This is done with showPhi, which may also plot them.
    
    
    Parameters
    ----------
    t: np.array
        simulation times
    r: np.array
        distances from center of well. It will be adapted to [0, rw, r[r>rw]]
    z: np.array
       elevation of layer planes (tops and bottoms in one array)
    rw: float
        well radius
    rc: float
        radius of well bore storage, used to calculation Ss in top well cell.
    topclosed, botclosed: bool
        if False then IBOUND becomes -1 instead of 1
    """
    t, r, kw = assert_input(t=t, r=r, **kw)
        
    # Make sure rw is r[1] and we include rw + dr
    r = np.hstack((0., rw, rw, r[r > rw]))
    
    gr = Grid(r, None, kw['z'], axial=True)

    # Well screen array
    e = kw['e'][:, np.newaxis, np.newaxis] #  * np.ones((1, gr.nx))
    E = e * np.ones((1, gr.nx), dtype=int)
    
    # Cells that are screen or casing cells
    screen = np.logical_and(gr.XM < rw,     E)
    casing = np.logical_and(gr.XM < rw, np.logical_not(E))
    
    # No fixed heads
    IBOUND = gr.const(1, dtype=int)
    if not topclosed:
        IBOUND[0,  :, 1:] = -1
    if not botclosed:
        IBOUND[-1, :, 1:] = -1
    
    # No horizontal flow in casing, vertical flow in screen and casing
    kr = gr.const(kw['kr']); kr[screen] = 1e+6; kr[casing]=1e-6 # inside well
    kz = gr.const(kw['kz']); kz[screen] = 1e+6; kz[casing]=1e+6 # inside well
    
    # Resistance between model layers
    c  = gr.const(kw['c'][:, np.newaxis, np.newaxis])
    c[:, :, gr.xm < rw] = 0. # No resistance in well
    
    Ss = gr.const(kw['Ss'][:, np.newaxis, np.newaxis])
    Ss[0, 0, 0] = (rc / rw) ** 2 / gr.DZ[0, 0, 0] # Well bore storage (top well cell)
    
    HI = gr.const(0.) # Initial heads
    FQ = gr.const(0.) # Fixed flows
    
    # Boundary conditions, extraction from screens proportional to kDscreen / kDwell
    assert np.isscalar(kw['Q']), "Q must be a scalar see kw['Q']"
    
    # Distribute the extraction of the screended parts of the well.
    # Note that this is strictly not necessary due to setting kv=np.inf in the entire well
    # and kh=0 in cased portion and np.inf in screened ones
    kDscreen = kw['e'] * kw['kr'] * kw['D']
    kDwell   = np.sum(kDscreen)
    FQ[:, 0, 0] = kw['Q'] * kDscreen / kDwell
    
    # Run the finite difference model
    out = fdm3t(gr=gr, t=t, kxyz=(kr, kr, kz), Ss=Ss, c=c,
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

    # Option 1: if time is not specified, all times is implied
    if t is None:
        t =fdm3t['t']
    if r is None:
        r = fdm3t['gr']['xm']
    if z is None and IL is None:
        IL = np.arange(fdm3t['gr'].nlay, dtype=int)
                    
    t = np.array([t]) if np.isscalar(t) else np.array(t)
    r = np.array([r]) if np.isscalar(r) else np.array(r)
    z = np.array([z]) if np.isscalar(z) else np.array(z)

    Phi = fdm3t['Phi'][:, :, 0, :] # sqeeze y

    if IL is None:
        V = -fdm3t['gr'].zm
        v = -z
    else:
        np.arange(fdm3t['gr'].nz)
        v = IL
        
    points = t, V, r
    interp = scipy.interpolate.RegularGridInterpolator(
            points, Phi, method=method,
            bounds_error=True, fill_value=np.nan)
    Z, T, R = np.meshgrid(v, t, r)
    PhiI = interp(np.vstack((T.ravel(), Z.ravel(), R.ravel())).T).reshape(T.shape)
    
    return PhiI
    

def showPhi(t=None, r=None, z=None, IL=None, method=None, fdm3t=None,
              xlim=None, ylim=None, xscale=None, yscale=None, show=True, **kw):
    """Return head for given times, distances and layers.
    
    Parameters
    ----------
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
    fdm3t: dictionary
        output of fdm.fdrm3t.fdm3t
    show: bool
        Weather to plot or only return PhiI
    
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

def test0(kw):
    """Simulate test0."""
    
    # Numerical (model has 4 layers)
    out = hemk99numerically(**kw)
    
    t = out['t']
    
    # Analytic for r values of r but all times
    rs = [5, 50, 500]
    
    # Show numerical results and return the axis for all times, three rs values and two z-values
    PhiI, ax = showPhi(t=None, r=rs, z=[-5, -35], IL=None,
                    method='linear', fdm3t=out,
                    xlim=None, ylim=None, xscale=None, yscale=None)
    
    # Same but now for all 3 layers
    IL = [0, 1, 2, 3]
    PhiI, ax = showPhi(t=None, r=rs, z=None, IL=IL,
                        method='linear', fdm3t=out,
                        xlim=None, ylim=None, xscale=None, yscale=None)
    
    kw['r'] = rs
    sa  = solution(**kw)
    for il in IL:
        for ir, r_ in enumerate(rs):
            label='layer={}, r={:0.4g} m'.format(il, r_)
            ax.plot(out['t'], sa[:, il, ir], '--', label=label)
    ax.legend(loc='lower right')
    
    # Get the data for only three times and all r
    PhiI, ax = showPhi(t=[1, 3, 10], r=None, z=[-5, -15, -25, -35], IL=None,
                    method='linear', fdm3t=out,
                    xlim=None, ylim=None, xscale='log', yscale=None)
    
    # Choose a set of times and layers for output
    ts = [1., 3., 10.]
    IL = [0, 1, 2, 3]
    
    # Select the numerical data for these times and layers
    PhiI, ax = showPhi(t=ts, r=None, z=None, IL=IL,
                    method='linear', fdm3t=out,
                    xlim=None, ylim=None, xscale='log', yscale=None)
    
    # Run analytical model
    kw['t'] = ts
    sa  = solution(**kw)
    
    for it, t_ in enumerate(ts):
        for il in IL:                
            label='layer={}, t={:0.4g} d'.format(il, t_)
            ax.plot(out['gr'].xm, sa[it, il, :], '--', label=label)
    ax.legend(loc='lower right')

def test1(kw):
    """Simultate using analytic class Hemker1999"""
    
    # TODO needs testing

    test1 = Hemker1999(**kw)
    sa  = test1.simulate(kw['t'], kw['r'], kw['Q'])
    # Using the function
    sb  = solution(**kw)
    
    assert np.all(sa == sb), "Test failed: not all sa == sb !"
    print("Done, test succeeded, all sa == sb !")

def test2(kw):
    """Return plot comparing analytic implentation as function and as class."""
    
    r_ = 500
    kw['r'] = r_
            
    kD = (kw['kr'] * kw['D']).sum()
    Q  = 4 * np.pi * kD
    kw['Q'] = Q
    S  = kw['Ss'] * kw['D']
    Sy, SA = S[0], S[1]
    tauA = kw['tau']
    tauB = tauA * SA / (Sy + SA)
    kw['t'] = r_  ** 2 * SA * tauA / (4 * kD) 
    
    # rho = r_ over B values used by Hemker (1999)
    r_B = np.array([0.01, 0.1, 0.2, 0.4, 0.6,
                         0.8, 1.0, 1.5, 2.0, 2.5, 3.0])
    xlim, ylim = None, None
    # xlim, ylim = (1e-1, 1e9), (1e-3, 1e2)
    
    ax = newfig("Test 2 analytic for Boulton case",
                r"$\tau = r^2 S / (4 kD t)$",
                r"$(4 \pi kD) / Q s$",
                xscale='log', yscale='log',
                xlim=xlim, ylim=ylim)
    ax.plot(tauA, scipy.special.exp1(1/tauA), 'r-',
            label='Theis for S=SA')
    ax.plot(tauA, scipy.special.exp1(1/tauB), 'b-',
            label='Theis for S=(Sy + SA)')
    
    cc = color_cycler()
    for rho in r_B:
        color = next(cc)
        c  = (r_ / rho) ** 2 / kD
        kw['c'] = np.array([c])
            
        # Using the function
        sb  = solution(**kw)
        # Using the class
        h99obj = Hemker1999(**kw)        
        t = kw['t']
        t = h99obj.tau2t(r=r_, tau=kw['tau'])
    
        sa  = h99obj.simulate(t, r_, Q)
    
        assert np.all(np.isclose(sa.ravel() / sb.ravel(), 1, atol = 0.0001)), "Test failed: not all sa == sb !"
        
        if rho in [0.01, 1.0, 1.5, 3.]:
            lbl = 'r/B = {:.4g}'.format(rho)
            lw = 2.
        else:
            lbl = ''
            color = 'k'
            lw = 0.25                
        for il in range(h99obj.nlay):
            marker = 'x' if il == 0 else '+'
            if lbl:
                label = lbl + ', layer={}'.format(il)
            else:
                label = ''
            ax.plot(kw['tau'], sa[:, il, 0], 
                    marker=marker, color=color, lw=0.5,
                    label=label)
            ax.plot(kw['tau'], sb[:, il, 0], '-',
                    color=color, lw=0.5)
        
    ax.legend(loc='lower right')
    print("Done, test succeeded !")
    return ax

def h99_F07_Szeleky(kw):
    """Simulate figure 07 in Hemker (1999), case Sceleky."""
    # TODO: Not yet right
    rs = [0.1, 1., 10.]
    tau = np.logspace(-3, 6, 91)
    kD = np.sum(kw['kr'] * kw['D'])
    S  = np.sum(kw['Ss'] * kw['D'])
    
    ax = newfig(kw['name'], r'$t_D$', r'$s [m]$',
        xlim=(1e-2, 1e6), ylim=(1e-3, 1e1), xscale='log', yscale='log')
    for ir, r_ in enumerate(rs):
        kw['t'] = tau * r_ ** 2 * S / (4 * kD)
        out = hemk99numerically(**kw)
        kw['r'] = rs
        sa = solution(r=kw['r'], **kw)
        PhiI = showPhi(t=kw['t'], r=rs, z=None, IL=None,
                    method='linear', fdm3t=out, show=False)
        ax.plot(tau, kw['Q'] / (4 * np.pi * kD) * scipy.special.exp1(1/tau), '--',
            label='Theis')
    
        for il in [0, 3]:
            ax.plot(tau, PhiI[:, il, 0], label='r = {:.4g} m, layer = {}'.format(r_, il))
    
    ax.legend(loc='lower right')
    return ax
  
def h99_F08(kw):
    """Simulate fig 8 in Hemker (1999):"""
    k = 1.0
    ax = newfig("Hemker 99 (fig 8)", r's_D', 'z', xlim=(10, 0), ylim=(-24, 0))
    for kkacc in [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1.e6]:
        kw['kr'][:8]  = k / kkacc
        kw['kr'][8:16]= k
        kw['kr'][16:] = k / kkacc
        kw['kz'] = kw['kr']
        D  = np.sum(kw['D'])
        kD = np.sum(kw['kr'] * kw['D'])
        S  = np.sum(kw['Ss'] * np.sum(kw['D']))
        
        kDscreens = kw['kr'] * kw['D'] * kw['e']
        kw['Q'] = kw['kr'] * kw['D'] * kw['e']
        r_ = 0.5 * (kw['r'][1] + kw['r'][2])  # 0.2 * D
        
        tauA = np.logspace(-3, 5, 81) # A is aquifer (layer 1)
        kw['t'] = r_  ** 2 * S * tauA / (4 * kD)
        out = hemk99numerically(**kw)
        PhiI = showPhi(t=None, r=[r_], z=None, IL=None,
                method='linear', fdm3t=out,
                xlim=None, ylim=None, xscale=None, yscale=None, show=False)
        
        ax.plot(PhiI[-1, :, 0], out['gr'].zm, label='k/k\' = {:.4g}'.format(kkacc))
    ax.legend(loc='lower right')
    return ax

def h99_F11(kw):
    """Simulate and return figure 11 in Hemker(1999)."""        
    etop, ebot = kw['e']
    kw['e'] = etop
    D  = np.sum(kw['D'])
    kD = np.sum(kw['kr'] * kw['D'])
    S  = np.sum(kw['D'] * kw['Ss'])
    kw['Q'] = 4 * np.pi * kD
    r_ = 0.2 * D
    tauA = np.logspace(-3, 5, 81) # A is aquifer (layer 1)
    
    kw['t'] = r_  ** 2 * S * tauA / (4 * kD)
    kw['e'] = etop
    out = hemk99numerically(**kw)
    PhiItop = showPhi(t=None, r=[r_], z=None, IL=[0, 1, 2, 3, 4],
            method='linear', fdm3t=out,
            xlim=None, ylim=None, xscale=None, yscale=None, show=False)
    kw['e'] = ebot
    out = hemk99numerically(**kw)
    PhiIbot = showPhi(t=None, r=[r_], z=None, IL=[0, 1, 2, 3, 4],
            method='linear', fdm3t=out,
            xlim=None, ylim=None, xscale=None, yscale=None, show=False)
    
    title = ('Drawdown responses to a partially penetrating well\n' +
             'in the lowest part (solid) or the uppermost part (dashed)\n' +
             'of a five-layer Ss- heterogeneous aquifer.')
    ax = newfig(title=title, xlabel=r'$t_D$', ylabel=r'$S_D$',
                xscale='log', yscale='log', xlim=(1e-2, 1e4), ylim=(1e-3, 1e2))
    
    cc = color_cycler()
    for il in [0, 1, 2, 3, 4]:
        sigmaTop = 4 * np.pi * kD / kw['Q'] * PhiItop[:, il]
        sigmaBot = 4 * np.pi * kD / kw['Q'] * PhiIbot[:, il]
        color = next(cc)                  
        ax.plot(tauA[1:], sigmaTop[1:], '-' , color=color, marker='x', label='top, layer {}'.format(il + 1))
        ax.plot(tauA[1:], sigmaBot[1:], '--', color=color, marker='+', label='bot, layer {}'.format(il + 1))
    
    ax.legend(loc='lower right')
    return ax

def h99_f11(kw):
    """Simulate and return figure 11 in Hemker(1999)."""
    
    # TODO: Still the model is too late compared to fig 11. in Hemker 99
    etop, ebot = kw['e']
    kw['e'] = etop
    D  = np.sum(kw['D'])
    kD = np.sum(kw['kr'] * kw['D'])
    S  = np.sum(kw['D'] * kw['Ss'])
    kw['Q'] = 4 * np.pi * kD
    r_ = 0.2 * D
    tauA = np.logspace(-3, 5, 81) # A is aquifer (layer 1)
    
    kw['t'] = r_  ** 2 * S * tauA / (4 * kD)
    kw['e'] = etop
    out = hemk99numerically(**kw)
    PhiItop = showPhi(t=None, r=[r_], z=None, IL=[0, 1, 2, 3, 4],
            method='linear', fdm3t=out,
            xlim=None, ylim=None, xscale=None, yscale=None, show=False)
    kw['e'] = ebot
    out = hemk99numerically(**kw)
    PhiIbot = showPhi(t=None, r=[r_], z=None, IL=[0, 1, 2, 3, 4],
            method='linear', fdm3t=out,
            xlim=None, ylim=None, xscale=None, yscale=None, show=False)
    
    title = ('Drawdown responses to a partially penetrating well\n' +
             'in the lowest part (solid) or the uppermost part (dashed)\n' +
             'of a five-layer Ss- heterogeneous aquifer.')
    ax = newfig(title=title, xlabel=r'$t_D$', ylabel=r'$S_D$',
                xscale='log', yscale='log', xlim=(1e-2, 1e4), ylim=(1e-3, 1e2))
    
    cc = color_cycler()
    for il in [0, 1, 2, 3, 4]:
        sigmaTop = 4 * np.pi * kD / kw['Q'] * PhiItop[:, il]
        sigmaBot = 4 * np.pi * kD / kw['Q'] * PhiIbot[:, il]
        color = next(cc)                  
        ax.plot(tauA[1:], sigmaTop[1:], '-' , color=color, marker='x', label='top, layer {}'.format(il + 1))
        ax.plot(tauA[1:], sigmaBot[1:], '--', color=color, marker='+', label='bot, layer {}'.format(il + 1))
    
    ax.legend(loc='lower right')
    return ax

def h99_F2_Boulton_or_Hantush(kw):
    """Simulalte boulton (1963) delayed yield or just Hantush
    
    The only difference being that with Hantush, the head in layer 0 is fixed.
    
    """
    # Hantush = Boulton with topclosed == False
    kw = cases[case]
    
    r_ = 500.
    
    hem = Hemker1999(**kw)
    Q = 4 * np.pi * hem.kD.sum()
    t = hem.tau2t(r=r_, tau=kw['tau'])
    
    
    r = kw['r'][kw['r'] > hem.rw] 
    sa = hem.simulate(t=t, r=r, Q=Q)
    
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
    tauB = tauA * SA / Sy  # =  4 * kD * kw['t'] / (r_ ** 2 * Sy)
    title ='{}, type curves for r/B from 0.01 to 3'\
                .format(kw['name'])
    xlabel = r'$\tau = 4 kD t /(r^2 S_2)$'
    ylabel = r'$\sigma = 4 \pi kD s / Q$' 
    ax = newfig(title, xlabel, ylabel,
                ylim=(1e-3, 1e2), xlim=(1e-1, 1e9),
                xscale='log', yscale='log')
    # The two Theis curves (note both for tauA)
    ax.plot(tauA, scipy.special.exp1(1/tauA), 'r', lw=3, label='Theis for SA')
    
    # TauA on x-axis and exp1(1/tauB) on y-axis
    ax.plot(tauA, scipy.special.exp1(1/tauB), 'b', lw=3, label='Theis for Sy + SA')
            
    cc = color_cycler()
    rNum = kw['r']
    for rho in r_B:
        color = next(cc)
        
        # Adapt c to to get the right r/B            
        B = r_ / rho
        kw['c'][0] = np.array([B ** 2 / kD])
        kw['r'] = rNum
        out = hemk99numerically(**kw) # using correct c
        
        # Analytisch:
        kw['r'] = rho * B
        sa = solution(**kw)
        
        hem = Hemker1999(**kw)
        Q = 4 * np.pi * hem.kD.sum()
        t = hem.tau2t(r=r_, tau=kw['tau'])
        r = rNum[rNum > hem.rw] 
        sh = hem.simulate(t=t, r=r, Q=Q)
    
        # Interpolate numerical to set of times layers and distances:
        PhiI = showPhi(t=None, r= rho * B, z=None, IL=[0, 1],
                        method='linear', fdm3t=out,
                        xlim=None, ylim=None, xscale='log', yscale='log', show=False)
        if case == 'Hantush':
            assert kw['topclosed'] == False, 'topclosed must be False for numerical Hantush!'
            ax.plot(tauA, hantush_conv.Wh(1/tauA, rho)[0], color=color, marker='x',
                    label='Wh(tau, {:.4g})'.format(rho))
        else:
            assert kw['topclosed'] == True, 'topclosed must be true for Boulton'
        # Plot Theis for S = Sy (specific yield)Show the drawdown in the top and first layer for this r/B
        for il in [0, 1]:
            sigma = 4 * np.pi * kD / kw['Q'] * PhiI[:, il, 0]
            if rho in [0.01, 1.0, 1.5, 3.]:
                labelN = 'num: r/B = {:.4g}'.format(rho)
                labelA = 'ana: r/B = {:.4g}'.format(rho)
                lw = 2.
            else:
                labelN = '_'
                labelA = '_'
                color = 'k'
                lw = 0.5
            ax.plot(tauA[1:], sigma[1:], '-', color=color, lw=lw, label=labelN)
            ax.plot(tauA[1:], sa[1:,il], 'x', color=color, lw=lw, label=labelA)
    
    ax.legend(loc='lower right') 
    return ax   
        
def h99_F3(kw):
    """Simulate case of Moench with vertical resistance within aquifer and on top."""
    # The problem is essentially the same as Boulton's, however
    # the delayed yield is due to vertical anisotropy.
    kw = cases[case]
    
    tauA = np.logspace(-2, 5, 71) # A is aquifer (layer 1)
    D  = np.sum(kw['D'][1:])
    C  = np.sum(kw['D'][1:] / kw['kz'][1:])
    kD = np.sum(kw['D'][1:] * kw['kr'][1:])
    Sy = kw['D'][0] * kw['Ss'][0]
    SA = 1e-3 * Sy
    kw['Ss'][1:] = SA / D
    
    r_ = D * np.sqrt(kw['kr'][1] / kw['k'][1])
    
    sigma = SA / Sy
    beta = kw['kz'][1] * r_ ** 2  / (kw['kr'][1] * D ** 2)
    
    kw['Q'] = 4 * np.pi * kD
    kw['t'] = r_  ** 2 * SA * tauA / (kD)   # / (4 kD)
    tauB = kD * kw['t'] / (r_ ** 2 * Sy)
    title =r'{}, type curves for values of gamma. $\sigma$ = {:.4g}, $\beta$={:.4g}'\
                .format(kw['name'], sigma, beta)
    xlabel = r'$\tau = 4 kD t /(r^2 S_2)$'
    ylabel = r'$\sigma = 4 \pi kD s / Q$' 
    ax = newfig(title, xlabel, ylabel,
                ylim=(1e-2, 1e1), xlim=(1e-1, 1e5),
                xscale='log', yscale='log')
    ax.plot(tauA / 4, scipy.special.exp1(1/tauA), 'r', lw=3, label='Theis for SA')
    ax.plot(tauA / 4, scipy.special.exp1(1/tauB), 'b', lw=3, label='Theis for Sy + SA')
    
    # Gamma is (D/k)/c
    gammas = np.array([1.0, 10., 100.])
    
    cc = color_cycler()        
    for gamma in gammas:
        color = next(cc)
        
        # D/k = C
        # gamma = (C * Sy) / (c[0] * SA)
        c = np.zeros(len(kw['D']) - 1)
        c[0] = C * Sy / SA / gamma
        
        kw['c'] = c
        
        B = np.sqrt(kD * (c[0] + C))
        out = hemk99numerically(**kw)
    
        PhiI = showPhi(t=None, r= r_, z=None, IL=None,
                        method='linear', fdm3t=out,
                        xlim=None, ylim=None, xscale='log', yscale='log', show=False)
        assert kw['topclosed'] == True, 'topclosed must be true for Boulton'
        ll = line_cycler()
        for il in [0, 1, -1]: # top and bottom layer of the aquifer
            ls = next(ll)
            s = 4 * np.pi * kD / kw['Q'] * PhiI[:, il, 0]
            label = r'$\gamma$={:.4g}, layer={}'.format(gamma, il)
            ax.plot(tauA[1:], s[1:], color=color, ls=ls, label=label)
    
    ax.legend(loc='lower right')
    return ax   
    
def h99_F6(kw):
    """Simlate fig 6 in Hemker (1999) well storage with partially penetrating aquifer."""
    # Papadopoulos and Cooper (1967)
    kw = cases[case]
    assert kw['topclosed'] == False, 'topclosed must be False for {}'\
        .format(case)
    
    RcRw = [1., 5., 20., 100., 1000.]
    RRD = [0.001, 0.1, 0.5] 
    
    tau = np.logspace(-2, 6, 81) # A is aquifer (layer 1)
    D  = np.sum(kw['D'][1:])
    kD = np.sum(kw['kr'][1:] * kw['D'][1:])
    S  = np.sum(kw['Ss'][1:] * kw['D'][1:])
    Sy = kw['Ss'][0] * kw['D'][0]
    Q  = 4 * np.pi * kD
    
    kw['Q'] = Q
    
    title =kw['name']
    xlabel = r'$\tau = 4 kD t /(r^2 S)$'
    ylabel = r'$\sigma = 4 \pi kD s / Q$' 
    ax = newfig(title, xlabel, ylabel,
                ylim=(1e-3, 1e2), xlim=(1e-1, 1e6),
                xscale='log', yscale='log')
    cc = color_cycler()
    
    for rcrw in RcRw:
        kw['rc'] = kw['rw'] * rcrw
        
        color = next(cc)
        for rrd in RRD:
            color = next(cc)
            
            r_ = rrd * D
            kw['t'] = r_  ** 2 * S * tau / (4 * kD) 
            out = hemk99numerically(**kw)
    
            PhiI = showPhi(t=None, r= [r_], z=None, IL=None,
                            method='linear', fdm3t=out,
                            show=False)
            sigma = 4 * np.pi * kD / kw['Q'] * PhiI[:, 2, 0]                
            ax.plot(tau[1:], sigma[1:], color=color,
                    label='rc/rw={:.4g}, r/D = {:.4g}, r={:.4g} m'
                    .format(rcrw, rrd, r_))
    
    ax.legend(loc='lower right')
    return ax
        
def boulton_well_storage(kw):
    """Simulate Boulton's solution for well storage."""
    kw = cases[case]
    
    r_ = 500.
    tauA = np.logspace(-2, 6, 81) # A is aquifer (layer 1)
    kD = kw['kr'][1] * kw['D'][1]        
    Sy = kw['D'][0] * kw['Ss'][0]
    SA = kw['D'][1] * kw['Ss'][1]
    
    kw['t'] = r_  ** 2 * SA * tauA / (4 * kD) 
    tauB = 4 * kD * kw['t'] / (r_ ** 2 * Sy)
    title ='{}, type curves and time-dependent drainage for r/B from 0.01 to 3'\
                .format(kw['name'])
    xlabel = r'$\tau = 4 kD t /(r^2 S_2)$'
    ylabel = r'$\sigma = 4 \pi kD s / Q$' 
    ax = newfig(title, xlabel, ylabel,
                ylim=(1e-3, 1e2), xlim=(1e-1, 1e6),
                xscale='log', yscale='log')
    ax.plot(tauA, scipy.special.exp1(1/tauA), 'r', lw=3, label='Theis for SA')
    
    r_B = np.array([0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0])
    
    cc = color_cycler()
    
    for rho in r_B:
        
        color = next(cc)
        
        B = 10.; r_ = B * rho
        kw['c'][0] = np.array([B ** 2 / kD])
        
        out = hemk99numerically(**kw)
    
        PhiI = showPhi(t=None, r= rho * B, z=None, IL=[0, 1],
                        method='linear', fdm3t=out,
                        xlim=None, ylim=None, xscale='log', yscale='log', show=False)
        if case == 'Hantush':
            assert kw['topclosed'] == False, 'topclosed must be False for Hantush!'
            ax.plot(tauA, hantush_conv.Wh(1/tauA, rho)[0], color=color, marker='x',
                    label='Wh(tau, {:.4g})'.format(rho))
        else:
            assert kw['topclosed'] == True, 'topclosed must be true for Boulton'
        for il in [0, 1]:
            sigma = 4 * np.pi * kD / kw['Q'] * PhiI[:, il, 0]
            if rho in [0.01, 1.0, 1.5, 3.]:
                label = 'r/B = {}'.format(rho)
                lw = 2.
            else:
                label = '_'
                color = 'k'
                lw = 0.5
            ax.plot(tauA[1:], sigma[1:], color=color, lw=lw, label=label)
    
    ax.legend(loc='lower right')
    return ax


cases ={ # Numbers refer to Hemker Maas (1987 figs 2 and 3)
    'test0': {
        'name': 'Test input and plotting', # (4 pi kD / Q) t
        't': None,
        'tau': np.logspace(-3, 1., 41),
        'r': np.hstack((0., np.logspace(-1., 4., 101))), # [0, rw, rPVC ...
        'z0': 0.,
        'rw': 0.1,
        'rc': 10.,
        'Q' : 1.0e+3,
        'D' : np.array([10., 10., 10., 10.]),
        'kr' : np.array([10., 10., 1., 1.]),        
        'kz': np.array([ 1., 1., 0.1, 0.1]),
        'Ss': np.array([10., 10., 1., 1.,]) * 1e-5,
        'c' : np.array([0., 0., 0.,]),
        'e' : np.array([1, 1, 1, 0]),
        'topclosed': True,
        'botclosed': True,
        'label': 'Test input',
        },
    'test1': {
        'comment': """Test to see that the analytic solution computed
        with the implemented functions yields the same results as the
        one implemented in a class. The test succeeds.
        """,
        'name': 'Test input and plotting', # (4 pi kD / Q) t
        't':  np.array([1., 3., 10.]),
        'tau': None,
        'r': np.array([1., 10., 100.]), 
        'z0': 0.,
        'rw': 0.1,
        'rc': 10.,
        'Q' : 1.0e+3,
        'D' : np.array([10., 10., 10., 10.]),
        'kr' : np.array([10., 10., 1., 1.]),        
        'kz': np.array([ 1., 1., 0.1, 0.1]),
        'Ss': np.array([10., 10., 1., 1.,]) * 1e-5,
        'c' : np.array([0., 0., 0.,]),
        'e' : np.array([1, 1, 1, 0]),
        'topclosed': True,
        'botclosed': True,
        'label': 'Test input',
        },
    'test2': {
        'comment': """Test2 compares the analytic solution computed using
        functions and the analytic solution computed using the class.
        It applies both implementations on the Boulton example. The test
        reveils there's no difference in the outcomes.
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
        'e':  np.array([0, 1]),
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
        'e':  np.array([0., 1]),
        'topclosed': False,
        'botclosed': True,
        'label': 'Hantush (1955)',
    },
    'H99 F2 Boulton': {
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
        'e':  np.array([0, 1]),
        'topclosed': True,
        'botclosed': True,
        'label': 'Boulton (1963)',
        },
    'H99 F3 Moench': {
        'name': 'Moench 1995/96 using 1 drainage layer and 20 sublayers',
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
        'z0': 1.0,
        'rw': 0.01,
        'rc': 0.01,
        'Q' : 4 * np.pi / 20.,  # Q = 4 pi kD
        'D' : np.ones(21),
        'kr': np.hstack((1e-6, np.ones(20))),
        'kz': np.hstack((1e+6, np.ones(20) * 1e-1)),
        'Ss': np.hstack((1e-1, np.ones(20) * 1e-4)), 
        'c' : None, # depends on gamma
        'e':  np.hstack((0, np.ones(20, dtype=int))),
        'topclosed': True,
        'botclosed': True,
        'label': 'Moench (1995/95)',
        },
    'Boulton Well Bore Storage': {
        'name': 'Boulton Well Bore Storage and Delayed Yield',
        't' : None,
        'tau': np.logspace(-2, 5, 71),
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
        'e':  np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]]), # Upper or Lower Layer is screened
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
        'e':  np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]]), # Upper or Lower Layer is screened
        'topclosed': True,
        'botclosed': True,
        'label': 'Boulton (19??)',
        },
    'H99 F6 well bore storage ppw': {
        'name': 'Wellbore storage: P&S (1967), B&S (1976)',
        'comment':"""To verify multi- layer results, a more complicated solution for flow to a
        partially penetrating well with wellbore storage, recharged by a no-drawdown top boundary
        (Boulton and Streltsova, 1976) was used. The plotted family of type curves (Fig. 6) are
        dimensionless drawdowns in the second layer of a five-layer model with storativity S = 1e-3 at dimensionless distances of 0.001, 0.1 and 0.5, discharged by a well of various diameters, screened in the upper three layers. 
        """,
        't' : None,
        'tau': np.logspace(-2, 5, 71),
        'r' : np.hstack((0., np.logspace(-1, 4, 101))),
        'z0': 1.0,
        'rw': 0.01,
        'rc': 0.01,
        'Q' : None,
        'D' : np.ones(6),
        'kr': np.hstack((1e-6, np.ones(5))),
        'kz': np.hstack((1e+6, np.ones(5) * 1e-1)),
        'Ss': np.hstack((1e-1, np.ones(5) * 0.2e-3)), 
        'c' : None, # depends on gamma
        'e':  np.array([0, 1, 1, 1, 0, 0], dtype=int),
        'topclosed': False,
        'botclosed': True,
        'label': 'Papadopoulos & Cooper (1967)',
    },
    'H99_F7 Szeleky': {
        'name': 'Székely (1995) Heterogeneous aquifer, PP well',
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
        'r' : np.hstack((0., np.logspace(-1, 6, 141))),
        'z0': 0.,
        'rw': 0.1,
        'rc': 0.001,
        'Q' : 1.0e3,
        'D' : np.array([10., 10., 10., 10.]),
        'kr': np.array([10., 10., 1., 1.]),
        'kz': np.array([10., 10., 1., 1.]) * 1e-1,
        'Ss': np.array([10., 10., 1., 1.]) * 1e-5,            
        'c' : None,
        'e' : np.array([1, 0, 0, 0]),
        'topclosed': True,
        'botclosed': True,
        'label': 'Székely (95)',
        },
    'H99_F8': {
        'name': 'PPW, heterogeneous conductivity (Maas 87)',
        'comment': 
            """Three layers of equal thickness, bounded by a no- drawdown top and a no-flow base,
            are discharged by a well in the middle part. Steady-state drawdown profiles are computed
            for various conductivity contrasts between the screened middle layer and the other parts
            of the aquifer.
            Fog 8 shows the results of a 24-sublayer model, which are very similar to the analytical
            solution, although each profile is simply drawn by connecting the calculated drawdown
            points in the middle of each sublayer.
            """,
        't' : None,
        'r' : np.hstack((0., np.logspace(-1, 6, 141))),
        'z0': 0.,
        'rw': 0.1,
        'rc': 0.001,
        'Q' : 4 * np.pi * 10,  # Q = 4 pi kD
        'D' : np.ones(24) * 1e0,
        'kr': np.ones(24) * 1e0,
        'kz': np.ones(24) * 1e0,
        'Ss': np.ones(24) * 1e-20,
        'c' : None,
        'e':  np.hstack((np.zeros(8), np.ones(8), np.zeros(8))),
        'topclosed': False,
        'botclosed': True,
        'label': 'Hemker (1999) fig 08',
        },
    'H99_F11': {
        'name': 'PPW, heterogeneous specific storage Ss',
        'comment': 
            """The aquifer is confined, K-homogeneous, anisotropic (Kr/Kz =10) and consists of five
            sublayers of equal thickness, with Ss-values that decrease with depth proportionally as
            14:10:7:5:4. Ratios are based on a shallow, 50 m thick aquifer with a storativity of 0.002
            and estimated Ss-values for the sublayers of 7., 5., 3.5, 2.5 and 2. x 1e-5.
            """,
        't' : None,
        'tau': np.logspace(-2, 8, 81),
        'r' : np.hstack((0., np.logspace(-1, 6, 141))),
        'z0': 0.,
        'rw': 0.1,
        'rc': 0.001,
        'Q' : 4 * np.pi * 10,  # Q = 4 pi kD
        'D' : np.array([10., 10., 10., 10., 10.]),
        'kr': np.array([10., 10., 10., 10., 10.]),
        'kz': np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        'Ss': np.array([7., 5., 3.5, 2.5, 2.]) * 1e-5, # [/m]
        'c' : None,
        'e':  np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]]), # Upper or Lower Layer is screened
        'topclosed': True,
        'botclosed': True,
        'label': 'Hemker (1999) fig 11',
        },
    'Vennebulten': {
        'name': 'Kruse & De Ridder (1994, p105) Delayed yield',
        't' : np.logspace(0, 5, 101) /(24 * 60), # d
        'r' : np.hstack((0., np.logspace(-1, 4, 41))),
        'z0': 0.,
        'rw': 0.1,
        'rc': 0.1,
        'rp': [10., 30., 90., 280.],
        'Q' : 873,  # m3/d
        'D' : np.array([10., 11.]),
        'kr': np.array([0.4, 135.]),
        'kz': np.array([0.04, 13.5]),
        'Ss': np.array([5e-3 / 10., 5e-4 / 11.]),
        'c' : np.array([0.]),
        'e':  np.array([0., 1]),
        'topclosed': True,
        'botclosed': True,
        'label': 'Vennebulten (1966)',
        },
}
 
if __name__ == "__main__":
      
    test0(cases['test0'])
    #test1(cases['test1'])
    #test2(cases['test2'])
    #boulton_well_storage(cases['Boulton Well Bore Storage'])
    #h99_F08(cases['H99_F08'])
    #h99_F07_Szeleky(cases['H99_F07 Szeleky'])
    #h99_F6(cases['H99 F6 well bore storage ppw'])
    #h99_F2_Boulton_or_Hantush(cases['H99 F2 Boulton'])
    #h99_F3(cases['H99 F3 Moench'])

    plt.show()
