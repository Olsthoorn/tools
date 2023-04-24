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
from etc import newfig, color_cycler
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
    Ti = kD * E # vector of floats
    Tw = np.sum(Ti)
    One = np.ones((len(kD), 1))
    
    C1 = Q / (2 * np.pi * Tw) # float, constant
    C2 = rc ** 2 / (2 * Tw)   # float, constant
    
    D   = Tp05 @ A @ Tm05
    W, R = la.eig(D)
    V   = Tm05 @ R
    Vm1 = la.inv(V)
    
    K0K1r = np.diag(sp.k0(r * np.sqrt(W.real))/
                    (rw * np.sqrt(W.real) * sp.k1(rw * np.sqrt(W.real))))
    
    E = np.diag(e)[:, e != 0]
    assert np.all(E.T @ e[:, np.newaxis] == np.ones((len(e), 1))
                  ), "E.T @ e must be all ones"
        
    U   = E.T @ V @ (np.eye(len(kD)) + p * C2 * K0K1r) @ Vm1 @ E
    Um1 = la.inv(U)

    s_ = Q / (2 * np.pi * p * Tw) @ V @ K0K1r @ V.T @ T @ E @ Um1 @ One
    
    return s_.flatten() # Laplace drawdown s(p)


def assert_input(ts=None, rs=None, z0=None, D=None, kr=None, kz=None, c=None,
                 Ss=None, e=None, **kwargs):
    """Return r, t after verifying length of variables.
    
    Parameters
    ----------
    ts: float or np.ndarray [d]
        tine or times
    rs: float or np.ndarray [m]
        distances to well center
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
    assert not (isinstance(ts, np.ndarray) and isinstance(rs, np.ndarray)),\
        "ts and rs must not both be vectors, at least one must be a scalar."
    
    t = ts if np.isscalar(ts) else None
    r = rs if np.isscalar(rs) else None
    
    
    for name, var in zip(['kr', 'kz', 'D', 'c', 'Ss', 'e'],
                         [kr, kz, D, c, Ss, e]):
        assert isinstance(var, np.ndarray), "{} not an np.ndarray".format(name)
        kwargs[name] = var
     
    assert len(D) == len(kr),  "Len(D) {} != len(k) {}".format(len(D), len(kr))
    assert len(c) == len(D) - 1 , "Len(c)={} != len(D) - 1 = {}".format(len(c), len(D) - 1)
    
    for name , v in zip(['D', 'kr', 'kz', 'c', 'Ss'],
                    [D, kr, kz, c, Ss]):
        assert np.all(v >= 0.), "all {} must be >= 0.".format(name)
    
    assert np.all(np.array([ee == 0 or ee == 1 for ee in e], dtype=bool)),\
            'All values in e must be 0 or 1'
    
    kwargs['z0'] = z0
    kwargs['z' ] = np.hstack((z0, z0 -  np.cumsum(D)))
    kwargs['kD'] = D * kr
    kwargs['C']  = 0.5 * (D / kz)[:-1] + kw['c'] + 0.5 * (D / kz)[1:]
    kwargs['S']  = D * Ss
        
    return t, r, kwargs

def backtransform(t, r, rw=None, rc=None, Q=None, kD=None,
                  c=None, S=None, e=None, **kwargs):
    """Return s(t, r) after backtransforming from Laplace solution.

    Parameters
    ----------
    ts: float or np.ndarray [d]
        tine or times
    rs: float or np.ndarray [m]
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


def solution(ts=None, rs=None, rw=None, rc=None,
             Q=None, kD=None, c=None, S=None, e=None, **kwargs):
    """Return the multilayer transient solution Maas Hemker 1987
    
    ts: float or np.ndarray [d]
        Time at which the dradown is to be computed
    rs: float or np.ndarray [m]
        Vector of distances from the well center at which drawdown is computed
    rw: float [m]
        well radius
    rc: float [m]
        radius of casing or storage part of well in which water table fluctuates
    Q: float
        Vector of extractions from the aquifers
    kD: np.ndarray
        Vector of transmissivities of the aquifers [m2/d]
    c:  np.ndarray
        Vector of the vertical hydraulic resistances of the aquitards [d]
    S:  np.ndarray
        Vector of  the storage coefficients of the aquifers
    e: np.ndarray [-] length n
       screened aquifers indicated by 1 else 0
        
    Notice
    ------
    Either ts or rs must be a scalar.
        
    Returns
    -------
    s[:, ts] for given r if rs is scalar
    or
    s[:, rs] for given t if ts is scalar
    or
    s[:, 0] for both rs and ts scalars
    """

    t, r = assert_input(ts, rs, Q=Q, kD=kD, c=c, S=S, e=e)
    
    if isinstance(rs, np.ndarray):        
        s = np.zeros((len(kD), len(rs)))
        for ir, r in enumerate(rs):
            s[:, ir] = backtransform(t, r, rw=rw, rc=rc, Q=Q,
                                     kD=kD, c=c, S=S, e=e, **kw)
    elif isinstance(ts, np.ndarray):
        s = np.zeros((len(kD), len(ts)))
        for it, t in enumerate(ts):
            s[:, it] = backtransform(t, r, rw=rw, rc=rc, Q=Q, kD=kD, c=c, S=S, e=e, **kw)
    else:
        s = np.zeros((len(kD), 1))
        s[:, 0] = backtransform(t, r, rw=rw, rc=rc, Q=Q, kD=kD, c=c, S=S, e=e, **kw)
        
    return s # Drawdown s[:, it] or s[:, ir] or s[:, 0]


cases ={ # Numbers refer to Hemker Maas (1987 figs 2 and 3)
    'test0': {'name': 'Test input and plotting',
        't': np.logspace(-3., 1., 40),
        'r': np.hstack((0., np.logspace(-1., 4., 100))), # [0, rw, rPVC ...
        'z0': 0.,
        'rw': 0.1,
        'rc': 10.,
        'Q' : 1.0e+3,
        'D' : np.array([10., 10., 10., 10.]),
        'kr' : np.array([10., 10., 1., 1.]),        
        'kz': np.array([ 1., 1., 0.1, 0.1]),
        'Ss': np.array([10., 10., 1., 1.,]) * 1e-5,
        'c' : np.array([0., 0., 0.,]),
        'e' : np.array([1, 0, 0, 0]),
        'topclosed': True,
        'botclosed': True,
        'label': 'Test input',
        },

    'Hantush': {'name': 'Hantush using one layer', # Hantush (single layer, does it work with one aquifer?)
        't' : None,
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
    'Boulton': {'name': 'Bouton (1963) Delayed yield',
        't' : None,
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
        'topclosed': True,
        'botclosed': True,
        'label': 'Boulton (1963)',
        },
    
    'Vennebulten': {'name': 'Kruse & De Ridder (1994, p105) Delayed yield',
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

    'Moench': {'name': 'Moench using 20 sublaters and 1 drainage layer on top',       
        't' : None,
        'r' : np.hstack((0., np.logspace(-1, 6, 141))),
        'z0': 1.0,
        'rw': 0.01,
        'rc': 0.01,
        'Q' : 4 * np.pi / 20.,  # Q = 4 pi kD
        'D' : np.ones(21),
        'kr': np.hstack((1e-6, np.ones(20))),
        'kz': np.hstack((1e+6, np.ones(20))),
        'Ss': np.hstack((1e-1, np.ones(20) * 1e-6)), 
        'c' : np.hstack((9000, np.zeros(19))),
        'e':  np.hstack((0, np.ones(20, dtype=int))),
        'topclosed': True,
        'botclosed': True,
        'label': 'Moench (1995/95)',
        },
'PPW': { 'name': 'Partially penetrating well', 
        'rw': 0.5,
        'rc': 0.5,
        'Q' : 1.2e+3,
        'kD': np.hstack((1e-5, np.ones(20) * 150)),
        'S' : np.hstack((0.1,  np.ones(20) * 1e-3)),
        'c' : np.hstack((100., np.ones(21) * 100.)),
        'topclosed': True,
        'botclosed': True,
        'label': 'Moench (93, 94) (20 layers)',
    },
    'Wellbore storage': { 'name': 'Papadopoulos & Cooper (67)', 
        'rw': 0.5,
        'rc': 0.5,
        'Q' : 1.2e+3,
        'kD': np.hstack((1e-5, np.ones(4) * 150)),
        'S' : np.hstack((0.001,  np.ones(4) * 1e-3)),
        'c' : np.hstack((100., np.ones(5) * 100.)),
        'e' : [1, 1, 1, 0, 0],
        'rd': [0.001, 0.1, 0.5], # Dimensionless disance r/D
        'topclosed': False,
        'botclosed': True,
        'label': 'Moench (93, 94) (20 layers)',
    },
    'Heterogenneous': {'name': 'Székely (95) Heterogeneous aquifer, PP well',
        'rw': 0.5,
        'rc': 0.5,
        'Q' : 1.2e3,
        'kD': np.array([1., 1., 1., 1.]) * 1e2,
        'S' : np.array([1., 1., 1., 1.]) * 1e4,            
        'c' : np.array([1e+7, 1e+1, 1e+7, 1e7, 1e7]),
        'e' : [1, 0, 0, 0],
        'topclosed': False,
        'botclosed': False,
        'label': 'Székely (95)',
        },
}
    
def hemk99numerically(t=None, z=None, r=None, rw=None, rc=None, topclosed=True, **kw):
    """Check Hemker(1999) numerically.
    
    Run axially multilayer model. To deal with multiple screens, and uniform
    head inside the well, rPVC and rOut are included in the distance array.
    Then rW < r < rPVC is considered PVC in the unscreend layers. And
    rPVC < r < rOut is considered just out of the well representing the
    head just outside the well casing both where the well is screend and where
    it is unscreend. Just in plotting for all r, only the heads for r > r[2] == rPVC
    are shown by showPhi.
    
    
    Parameter
    ----------
    t: simulation times
        times
    z: None, sequence of array
       elevation of layer planes (tops and bottoms in one array)
    r: np.array of floats
        distances from well center
    rw: float
        well radius; must be same as r[1]
    rc: float
        radius of well bore storage, used to calculation Ss in top well cell.
    """
    AND, NOT = np.logical_and, np.logical_not
    _, _, kw = assert_input(**kw)
        
    # Make sure rw is r[1] and we include rw + dr
    dr = 0.005
    r = np.hstack((r[0], rw, rw + dr, r[r > rw + dr]))
    
    gr = Grid(r, None, kw['z'], axial=True)

    # Well screen array
    e = kw['e'][:, np.newaxis, np.newaxis] #  * np.ones((1, gr.nx))
    E = e * np.ones((1, gr.nx))
    
    screen = AND(gr.XM < rw,     E)
    casing = AND(gr.XM < rw, NOT(E))
    
    # No fixed heads
    IBOUND = gr.const(1, dtype=int)
    if not topclosed:
        IBOUND[0,  :, 1:] = -1
    
    # No horizontal flow in casing, vertical flow in screen and casing
    kr = gr.const(kw['kr']); kr[screen] = 1e+6; kr[casing]=1e-6 # inside well
    kz = gr.const(kw['kz']); kz[screen] = 1e+6; kz[casing]=1e+6 # inside well
    
    c  = gr.const(kw['c'])
    c[:, :, gr.xm < rw] = 0. # No resistance in well
    
    Ss = gr.const(kw['Ss'])
    Ss[0, 0, 0] = (rc / rw) ** 2 / gr.DZ[0, 0, 0] # Well bore storage (top well cell)
    
    HI = gr.const(0.)
    FQ = gr.const(0.)
    
    # Boundary conditions, extraction from screens proportional to kDscreen / kDwell
    kD = kw['kr'] * kw['D']
    FQ[e.ravel() != 0, 0, 0] = kw['Q'] * kD[e.ravel() != 0] / np.sum(kD[e.ravel() != 0])
           
    return fdm3t(gr=gr, t=t, kxyz=(kr, kr, kz), Ss=Ss, c=c,
                FQ=FQ, HI=HI, IBOUND=IBOUND)
    
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

    assert (not (IL is None and z is None)
            and not (
                (t is not None) and (r is not None))
            ), 'Either `IL` or `z` must be none !'
    
    assert (not (t is None and r is None)
            and not (
                (t is not None and r is not None))
            ), 'Either `t` or `r` must be none!'

    # Option 1: if time is not specified, all times is implied
    if t is None:
        t =fdm3t['t']
        if z is not None:
            option = 1 # Zs
        else:
            option = 2 # IL
    else: # t not None --> r None)
        r = fdm3t['gr'].xm
        if z is not None:
            option = 3
        else:
            option = 4
                    
    t = np.array([t]) if np.isscalar(t) else np.array(t)
    r = np.array([r]) if np.isscalar(r) else np.array(r)
    z = np.array([z]) if np.isscalar(z) else np.array(z)

    Phi = fdm3t['Phi'][:, :, 0, :] # sqeeze y

    if option in [1, 3]:
        points = fdm3t['t'], -fdm3t['gr'].zm, fdm3t['gr'].xm
        interp = scipy.interpolate.RegularGridInterpolator(
            points, Phi, method=method,
            bounds_error=True, fill_value=np.nan)
        Z, T, R = np.meshgrid(-z, t, r)
        PhiI = interp(np.vstack((T.ravel(), Z.ravel(), R.ravel())).T).reshape(T.shape)
    else:
        points = fdm3t['t'], np.arange(fdm3t['gr'].nz), fdm3t['gr'].xm
        interp = scipy.interpolate.RegularGridInterpolator(
            points, Phi, method=method,
            bounds_error=True, fill_value=np.nan)
        L, T, R = np.meshgrid(IL, t, r)
        PhiI = interp(np.vstack((T.ravel(), L.ravel(), R.ravel())).T).reshape(T.shape)

    if show == False:
        return PhiI

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
    
if __name__ == "__main__":
      
    ts =np.linspace(0, 50, 11)
    rs = np.array([1.0, 3.0, 10., 30., 100., 300., 1000., 3000., 10000.])
 
    case = 'Moench'
    
    if case == 'test0':
        kw = cases[case]

        out = hemk99numerically(**kw)

        PhiI, ax = showPhi(t=None, r=[5, 50, 500], z=[-5, -35], IL=None,
                        method='linear', fdm3t=out,
                        xlim=None, ylim=None, xscale=None, yscale=None)

        PhiI, ax = showPhi(t=None, r=[5, 50, 500], z=None, IL=[0, 1, 2],
                            method='linear', fdm3t=out,
                            xlim=None, ylim=None, xscale=None, yscale=None)

        PhiI, ax = showPhi(t=[1, 3, 10], r=None, z=[-5, -20, -35], IL=None,
                        method='linear', fdm3t=out,
                        xlim=None, ylim=None, xscale='log', yscale=None)

        PhiI, ax = showPhi(t=[1, 3, 10], r=None, z=None, IL=[0, 1, 2],
                        method='linear', fdm3t=out,
                        xlim=None, ylim=None, xscale='log', yscale=None)
        
    if case == 'Boulton' or case=='Hantush':
        # Hantush = Boulton with topclosed == False
        kw = cases[case]
        
        r_ = 500.

        tauA = np.logspace(-2, 9, 111) # A is aquifer (layer 1)

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
                    ylim=(1e-3, 1e2), xlim=(1e-1, 1e9),
                    xscale='log', yscale='log')

        ax.plot(tauA, scipy.special.exp1(1/tauA), 'r', lw=3, label='Theis for SA')
        ax.plot(tauA, scipy.special.exp1(1/tauB), 'b', lw=3, label='Theis for Sy + SA')
        
        r_B = np.array([0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0])
        
        cc = color_cycler()
        
        for rho in r_B:
            
            color = next(cc)
            
            B = r_ / rho
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
        
        
    if case == 'Moench':
        kw = cases[case]
        
        r_ = 500.

        tauA = np.logspace(-2, 5, 71) # A is aquifer (layer 1)

        kD = np.sum(kw['kr'][1:] * kw['D'][1:])
        Sy = kw['D'][0] * kw['Ss'][0]
        SA = np.sum(kw['D'][1:] * kw['Ss'][1:])
        
        kw['t'] = r_  ** 2 * SA * tauA / (4 * kD) 

        tauB = 4 * kD * kw['t'] / (r_ ** 2 * Sy)

        title ='{}, type curves and time-dependent drainage for values of gamma'\
                    .format(kw['name'])
        xlabel = r'$\tau = 4 kD t /(r^2 S_2)$'
        ylabel = r'$\sigma = 4 \pi kD s / Q$' 
        ax = newfig(title, xlabel, ylabel,
                    ylim=(1e-2, 1e1), xlim=(1e-1, 1e5),
                    xscale='log', yscale='log')

        ax.plot(tauA, scipy.special.exp1(1/tauA), 'r', lw=3, label='Theis for SA')
        ax.plot(tauA, scipy.special.exp1(1/tauB), 'b', lw=3, label='Theis for Sy + SA')
        
        gammas = np.array([1.0, 10., 100.])
        
        cc = color_cycler()
        
        for gamma in gammas:
            
            c2 = np.sum(kw['D'][1:] / kw['kz'][1:])
            color = next(cc)
            
            kw['c'][0] = gamma * c2
            
            out = hemk99numerically(**kw)
        
            PhiI = showPhi(t=None, r= r_, z=None, IL=[1, 20],
                            method='linear', fdm3t=out,
                            xlim=None, ylim=None, xscale='log', yscale='log', show=False)

            assert kw['topclosed'] == True, 'topclosed must be true for Boulton'

            for il in [1, -1]:
                sigma = 4 * np.pi * kD / kw['Q'] * PhiI[:, il, 0]
                label = 'gamma = {}'.format(gamma)
                lw = 2.
                ax.plot(tauA[1:], sigma[1:], color=color, lw=lw, label=label)
        
        ax.legend(loc='lower right')
        
        
        
    
    if case == 'Vennebulten':
        kw = cases[case]

        out = hemk99numerically(**kw)

        kD = kw['kr'][1] * kw['D'][1]
        B = np.sqrt(kD * kw['c'][0])
        c = kw['c'][0]
        
        PhiI, ax = showPhi(t=None, r=[90.], z=None, IL=[1],
                        method='linear', fdm3t=out,
                        xlim=None, ylim=None, xscale='log', yscale='log')
        
        ax.set_title('{}, kD={:.4g} m2/d, kz={}, Sa = {}, Sy = {}'\
            .format(kw['name'], kD, kw['kz'][0],
                    kw['Ss'][1] * kw['D'][1],
                    kw['Ss'][0] * kw['D'][0]))
    
    plt.show()
    
    print('Done')
    
    raise SystemExit
    


    shn = solution(ts=ts, rs=rs, **kw)
    
    if case == 'Hant': # Hantush
        t  = ts
        Q  = cases[case]['Q' ][0]
        kD = cases[case]['kD'][0]
        S  = cases[case]['S' ][0]
        c  = cases[case]['c' ][0]
        u = rs ** 2 * S / (4 * kD * t)
        L = np.sqrt(kD * c)

        sHantu = np.zeros(len(rs))
        for ir, r in enumerate(rs):
            u = r ** 2 * S / (4 * kD * t)
            sHantu[ir] = Q / (4 * np.pi * kD) * Wh(u, r/L)[0]
        sTheis = Q / (4 * np.pi * kD) * sp.exp1(rs ** 2 * S / (4 * kD * t))
    
        fig,ax = plt.subplots()
        ax.set_title(kw['name'] + "drawdown at t={:.4g} d".format(t))
        ax.set_xlabel('r [m]')
        ax.set_ylabel('s [m]')
        ax.set_xscale('log')
        ax.grid()
        for iL, s_ in enumerate(shn):
            ax.plot(rs, s_, label='Layer {}'.format(iL))

        ax.plot(rs, sHantu, 'o', label='Hantush t={:.2f} d'.format(t))
        ax.plot(rs, sTheis, 'x', label='Theis')
        
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.legend()
        print('Done transient!')
    
        # Steady state
        s = multRadial(rs=rs, **kw)
        
        fig,ax = plt.subplots()
        ax.set_title(kw['name'] + "drawdown steady")
        ax.set_xlabel('r [m]')
        ax.set_ylabel('s [m]')
        ax.set_xscale('log')
        ax.grid()
        for iL, s_ in enumerate(s):
            ax.plot(rs, s_, label='Layer {}'.format(iL))

        s1 = Q / (2 * np.pi * kD) * sp.k0(rs / np.sqrt(kD * c))

        ax.plot(rs, s1, 'o', label='Steady 1 layer')
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.legend()
        print('Done steady !')
        
    # As a function of time for given distance (ts is ndarray, r=scalar)
    ts, rs = 1.e-3, np.logspace(1, 4, 16)
    
    case='3'
    kw = cases[case]
    shn = solution(ts=ts, rs=rs, **kw)
    for iL in range(len(kw['kD'])):
        plt.plot(rs, shn[iL], label=f'layer {iL}')
    #plt.xscale('log')
    #plt.yscale('log')
    plt.grid()
    plt.legend()
    
    
    
    plt.show()
