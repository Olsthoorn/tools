"""Implementation of:
   
   Maas and Hemker 1987 Unsteady flow to wells in layered and fissured aquifer
   systems JoH 90 (1987)231-249

    A the end, the implementation is tested using the the graph in the
    publication and reproducing it both analytically and nuemrically

   TO 2023-03-03
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.special as sp
from hantush_conv import Wh
import warnings

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

def sysmat(p=None, kD=None, c=None, S=None, St=None,
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
    St: np.ndarray [1/m] or None
        vector of the storage coefficients of the aquitards
    topclosed: bool
        top of system is closed (impervious)
    botclosed: bool
        bottom of system is closed (impervious)
    """
    b = p * c * St
    if np.any(b > 700):
        b[b > 700] = 700 # prevent overflow of sinh
    aSt = np.ones_like(c) if St is None else b / np.tanh(b)
    bSt = np.ones_like(c) if St is None else b / np.sinh(b)
    B = - np.diag(aSt[ :-1] / c[ :-1] + aSt[1:] / c[1:], k=0)\
        + np.diag(bSt[1:-1] / c[1:-1], k=+1)\
        + np.diag(bSt[1:-1] / c[1:-1], k=-1)
    if topclosed:
        b00 = 1. if St is None else np.sqrt(p * c[ 1] * St[ 0]) / np.tanh(np.sqrt(p * c[ 1] * St[ 0]))
        B[0, 0]   = -b00 / c[1]
    if botclosed:        
        bnn = 1. if St is None else np.sqrt(p * c[-2] * St[-1]) / np.tanh(np.sqrt(p * c[-2] * St[-1]))
        B[-1, -1] = -bnn / c[-2]
    
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
    kw.update(p=0, S=None, St=None)
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
  
def sLaplace(p=None, r=None,
        Q=None, kD=None, c=None, S=None, St=None, **kw):
    """Return the laplace transformed drawdown for t and r scalars.
    
    The solution is the Laplace domain is

    s_(r, p) = 1 / (2 pi p) K0(r sqrt(A(p))) T^(-1) q
    
    With A the system matrix (see sysmat).

    p: float [1/d]
        Laplace variable
    r: float [m]
        Distance from the well center
    Q: float
        Vector of extractions from the aquifers
    kD: np.ndarray
        Vector of transmissivities of the aquifers [m2/d]
    c:  np.ndarray
        Vector of the vertical hydraulic resistances of the aquitards [d]
    S:  np.ndarray
        Vector of  the storage coefficients of the aquifers
    St: np.ndarray or None
        Vector of storage coefficients of the aquitards
    r:  float
        Distance at which the drawdown is to be computed
    """
    A   = sysmat(p=p, kD=kD, c=c, S=S, St=St, **kw)
    Tp05   = np.diag(np.sqrt(kD))
    Tm05   = np.diag(1. / np.sqrt(kD))
    
    Tm1 = np.diag(1 / kD)
    
    D   = Tp05 @ A @ Tm05
    W, R = la.eig(D)
    V   = Tm05 @ R
    Vm1 = la.inv(V)
    
    K0r  = np.diag(sp.k0(r * np.sqrt(W.real)))
    
    s_ = V @ K0r @ Vm1 @ Tm1 @ Q[:, np.newaxis] / (2 * np.pi * p)
    
    return s_.flatten() # Laplace drawdown s(p)


def assert_input(ts=None, rs=None, Q=None, kD=None, c=None, S=None, St=None):
    
    assert not (isinstance(ts, np.ndarray) and isinstance(rs, np.ndarray)),\
        "ts and rs must not both be vectors, at least one must be a scalar."
    
    t = ts if np.isscalar(ts) else None
    r = rs if np.isscalar(rs) else None
    
    for name, var in zip(['kD', 'c', 'S', 'St'],
                         [kD, c, S, St]):
        if name == 'St':
            if St is None:
                continue # skip
        assert isinstance(var, np.ndarray), "{} not an np.ndarray".format(name)
        
    assert len(c) == len(kD) + 1, "Len(c) != len(kD) + 1"
    
    assert np.all(c > 0.), "all c must be > 0."
    assert np.all(kD > 0.), "all kD must be > 0."
    
    for name, var in zip(['Q', 'kD', 'S'], [Q, kD, S]):
        assert len(var) == len(kD), "len {} != len(kD)".format(name)
    
    for name, var in zip(['c', 'St'], [c, St]):
        if name == 'St':
            if St is None:
                continue
            else:
                St[St < 1e-20] = 1e-20 # prevent division by zero
        assert len(var) == len(c), "len {} != len(c)".format(name)
    return t, r

def backtransform(t, r, Q=None, kD=None, c=None, S=None, St=None, **kw):
    """Return s(t, r) after backtransforming from Laplace solution."""
    s = np.zeros_like(kD)
    for v, i in zip(vStehfest, range(1, len(vStehfest) + 1)):
        p = i * np.log(2.) / t
        s += v * sLaplace(p=p, r=r, Q=Q,
                        kD=kD, c=c, S=S, St=St, **kw)
    s *= np.log(2.) / t
    return s.flatten()


def solution(ts=None, rs=None,
             Q=None, kD=None, c=None, S=None, St=None, **kw):
    """Return the multilayer transient solution Maas Hemker 1987
    
    ts: float or np.ndarray [d]
        Time at which the dradown is to be computed
    rs: float or np.ndarray [m]
        Vector of distances from the well center at which drawdown is computed
    Q: float
        Vector of extractions from the aquifers
    kD: np.ndarray
        Vector of transmissivities of the aquifers [m2/d]
    c:  np.ndarray
        Vector of the vertical hydraulic resistances of the aquitards [d]
    S:  np.ndarray
        Vector of  the storage coefficients of the aquifers
    St: np.ndarray or None
        Vector of storage coefficients of the aquitards
        
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

    t, r = assert_input(ts, rs, Q=Q, kD=kD, c=c, S=S, St=St)
    
    if isinstance(rs, np.ndarray):        
        s = np.zeros((len(kD), len(rs)))
        for ir, r in enumerate(rs):
            s[:, ir] = backtransform(t, r, Q=Q, kD=kD, c=c, S=S, St=St, **kw)
    elif isinstance(ts, np.ndarray):
        s = np.zeros((len(kD), len(ts)))
        for it, t in enumerate(ts):
            s[:, it] = backtransform(t, r, Q=Q, kD=kD, c=c, S=S, St=St, **kw)
    else:
        s = np.zeros((len(kD), 1))
        s[:, 0] = backtransform(t, r, Q=Q, kD=kD, c=c, S=S, St=St, **kw)
        
    return s # Drawdown s[:, it] or s[:, ir] or s[:, 0]

cases ={ # Numbers refer to Hemker Maas (1987 figs 2 and 3)
    'Mine': {'name': 'My personal test',
        'Q' : np.array([1e+2, 0e+0, 0e+0]),
        'kD': np.array([1e+2, 1e+2, 1e+2]) * 1.5,
        'S' : np.array([1e-3, 1e-3, 1e-3]),
        'c' : np.array([1e+5, 1e+2, 1e+2, 1e+2]) * 2.5,
        'St': np.array([1e-3, 1e-3, 1e-3, 1e-3]),
        'topclosed': False,
        'botclosed': False,
        'label': 'Test1',
        },
    'Hant1': {'name': 'Hantush using one layer', # Hantush (single layer, does it work with one aquifer?)
        'Q' : np.array([1e+2]),
        'kD': np.array([1e+2]),
        'S' : np.array([1e-1]),
        'c' : np.array([1e+3, 1e+7]), # Artificially closed using high c at bottom
        'St': None,
        'topclosed': False,
        'botclosed': False, # This should also work
        'label': 'Hantush',
        },
    'Hant2': {'name': 'Hantush using 2 layers', # Hantush using two layers with low intermediate resistance
        'Q' : np.array([1e+2, 0e+0]),
        'kD': np.array([1e+2, 1e+2]) * 1.5,
        'S' : np.array([1e-3, 1e-3]),
        'c' : np.array([1e+2, 1e+1, 1e+7]),
        'St': np.array([0e-0, 0e-0, 0e-0]),
        'topclosed': False,
        'botclosed': False,
        'label': 'Hantush',
        },
    '1A': { 'name': 'Fig 1A', 
        'Q':  np.array([0., 0., 1., 0., 0.]) * 1e+2,
        'kD': np.array([1., 1., 1., 1., 1.]) * 1e+2,
        'S' : np.array([1., 1., 1., 1., 1.]) * 1e-3,
        'c' : np.array([1., 1., 1., 1., 1., 1.]) * 1e+2,
        'St': np.array([1., 1., 1., 1., 1., 1.]) * 1e-3,
        'topclosed': False,
        'botclosed': False,
    },
    '1B': { 'name': 'Fig 1B', 
        'Q':  np.array([0., 0., 1., 0., 0.]) * 1e+2,
        'kD': np.array([1., 1., 1., 1., 1.]) * 1e+2,
        'S' : np.array([1., 1., 1., 1., 1.]) * 1e-3,
        'c' : np.array([1., 1., 1., 1., 1., 1.]) * 1e+2,
        'St': np.array([1., 1., 1., 1., 1., 1.]) * 1e-3,
        'topclosed': True,
        'botclosed': True,
    },
    '2A': {'name': 'One aquitard system Fig. 2 A',
        'Q' : np.array([1., 1.]) * 4 * np.pi * 1e+2,
        'kD': np.array([1., 1.]) * 1e2,
        'S' : np.array([1., 1.]) * 1e4,            
        'c' : np.array([1e+7, 1e+1, 1e+7]),
        'St': np.array([0., 1., 0.]) * 1.6 * 1e-3,
        'topclosed': False,
        'botclosed': False,
        'label': 'Curve A',
        },
    '2A1': {'name': 'Three-aquitard system Fig. 2 A1',
        'Q':  np.array([1., 1.]) * 4 * np.pi * 1e+4,
        'kD': np.array([1., 1.]) * 1e+2,
        'S':  np.array([1., 1.]) * 1e-4,
        'Dt': np.array([1., 1., 1.]) * 1e+1,
        'c':  np.array([1e12, 100., 1e12]),
        'Sst': np.array([1., 1., 1.]) * 1.6e-3,
        'topclosed': False,
        'botclosed': False,
        'label': 'A1'
        },
    '2A2': {'name': 'Three-aquitard system Fig. 2 A2',
        'Q' : np.array([0., 1.]) * 4 * np.pi * 1e+2,
        'kD': np.array([1., 1.]) * 1e+2,
        'S' : np.array([1., 1.]) * 1e-4,            
        'c' : np.array([1., 1., 1.]) * 1e+2,
        'St': np.array([0., 1., 0.]) * 1.6e-3,
        'topclosed': True,
        'botclosed': True,
        'label': 'A2',
        },
    '2B': {'name': 'Three-aquitard system Fig. 2 B',
        'Q' : np.array([0., 1.]) * 4 * np.pi * 1e+2,
        'kD': np.array([1., 1.]) * 1e+2,
        'S' : np.array([1., 1.]) * 1e-4,            
        'c' : np.array([1., 1., 1.]) * 1e+2,
        'St': np.array([1., 1., 1.]) * 1.6e-3,
        'topclosed': True,
        'botclosed': True,
        'label': 'B',
        },
    '2C': {'name': 'Three-aquitard system Fig. 2 C',
        'Q' : np.array([0., 1.]) * 4 * np.pi * 1e+2,
        'kD': np.array([1., 1.]) * 1e+2,
        'S' : np.array([1., 1.]) * 1e-4,            
        'c' : np.array([1., 1., 1.]) * 1e+2,
        'St': np.array([1., 1., 1.]) * 1.6e3,
        'topclosed': False,
        'botclosed': True,
        'label': 'C',
        },
    '2D': {'name': 'Three-aquitard system Fig. 2 D',
        'Q' : np.array([0., 1.]) * 4 * np.pi * 1e+2,
        'kD': np.array([1., 1.]) * 1e+2,
        'S' : np.array([1., 1.]) * 1e-4,            
        'c' : np.array([1., 1., 1.]) * 1e+2,
        'St': np.array([1., 1., 1.]) * 1.6e3,
        'topclosed': False,
        'botclosed': False,
        'label': 'D'
        },
    '3': {'name': 'Hemker-Maas (1987) Fig. 3',
        'Q' : np.array([ 0., 1. , 0. , 0.])      * 1e+4,
        'kD': np.array([ 2., 1.5, 0.5, 2.])      * 1e+3,
        'S' : np.array([10., 4. , 1. , 3.])      * 1e-4,
        'c' : np.array([ 1., 1.5, 1. , 4., 20.]) * 1e+3,
        'St': np.array([30., 5. , 3. , 2., 10.]) * 1e-4,
        'topclosed': False,
        'botclosed': False,
        'label': 'Fig 3'
        },
    '4A': {'name': 'Hemker-Maas (1987) Fig. 4a',
        'Q' : np.array([0e+0, 1e+0, 0e+0]),
        'kD': np.array([1e-3, 1e+0, 1e-3]),
        'S' : np.array([1e-1, 1e-4, 1e-3]),
        'c' : np.array([1e12, 1e+2, 1e+0, 1e12]),
        'St': np.array([0., 0., 0., 0.]),
        'topclosed': False,
        'botclosed': False,
        'label': 'A'
        },
    '4B': {'name': 'Hemker-Maas (1987) Fig. 4b',
        'Q' : np.array([0e+0, 1e+0]),
        'kD': np.array([1e-3, 1e+0]),
        'S' : np.array([1e-1, 1e-4]),
        'c' : np.array([1e12, 1e+2, 1e+0]),
        'St': np.array([0., 0., 0.]),
        'topclosed': False,
        'botclosed': True,
        'label': 'B'
        },
    '4C': {'name': 'Hemker-Maas (1987) Fig. 4c',
        'Q' : np.array([1.]),
        'kD': np.array([1.]),
        'S' : np.array([1e-4]),
        'c' : np.array([1e+2, 1e+0]),
        'St': np.array([0., 0.]),
        'topclosed': True,
        'botclosed': True,
        'label': 'C'
        }
}
    

if __name__ == "__main__":

    ts =50
    rs = np.array([1.0, 3.0, 10., 30., 100., 300., 1000., 3000., 10000.])
    
    case = '3'
    kw = cases[case]
    
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
