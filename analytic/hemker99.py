# Hemker 1999 Transient well flow in vertically heterogeneous aquifers JoH 225 (1999)1-18

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.special as sp

## % First define the system matrix
def sysmat(kD=None, c=None):
    """Return the system matrix of the multyaquifer system.
    
    Parameters
    ----------
    kD: np.ndarray [m2/d]
        vector of n layer transmissivities determins number of layers.
    c:  np.ndarray [d]
        vector of n+1 interlayer vertical hydraulic resistances.
    """
    assert isinstance(kD, np.ndarray), "kD not an np.ndarray."
    assert isinstance(c, np.ndarray), "c not an np.ndarray."
    assert len(c) == len(kD) + 1, "Len(kD) != len(c) + 1"
    
    A = np.diag(1 / (c[:-1] * kD ) + 1 / (c[1:] * kD), k=0) -\
        np.diag(1 / (c[1:-1] * kD[1:]),k=-1) -\
        np.diag(1 / (c[:-2]  * kD[:-1]), k=+1)
    return A

def slaplace(Q, e, kD, c, rs, rw, rc):
    """Return the laplace transformed drawdown.
    Q: float
        Total extraction from the well
    e: (logical) vector indicating which layers are screened
    kD: np.ndarray
        Sublayer transmissivities [m2/d]
    c:  np.ndarray
        top, bottom and interlayer vertical resistances [d]
    rs: np.ndarray
        distances at which the drawdown is to be computed
    rc: float
        radius of well casing
    rw: float
        radius of well bore
    """
    A   = sysmat(kD, c)
    T   = np.diag(kD)
    Tw  = np.sum(e * kD) # Total transmissivity at well screen
    E   = np.diag(kD) # Todo Check
    one = np.ones_like(kD)
    D   = np.sqrt(T) @ A @ la.inv(np.sqrt(T))
    W, R = la.eig(D)
    V    = la.inv(T) @ R
    Vm1  = la.inv(V)
    K0r  = np.diag(sp.k0(r  * np.sqrt(W)))
    rwsqrtw = rw * np.sqrt(W)   
    K1rw = np.diag(rwsqrtw * sp.k1(rwsqrtw))
    C1 = Q / (2 * np.pi * Tw)
    C2 = rc ** 2  / (2 * Tw)
    g = V @ np.diag(1 / rk1) @ Vm1 @ ((C1 / p) @ e - p * C2 * sw)
    slaplace = V @ K0r @ Vm1 @ g
    
    
    rk1 = np.diag(rwsqrtw * sp.k1(rwsqrtw / (rwsqrtw * sp.k1(rwsqrtw) +
                p * rc ** 2 / (2 * Tw) * sp.k0(rwsqrtw))))
    Um1 = V @ rk1 @ Vm1
    one = np.ones_like(kD)[:, np.newaxis]
    sqrtw = np.sqrt(W)
    for r in rs:
        V @ np.diag(sp.k0(r * sqrtw) / (rw * sqrtw * sp.k1(rw * sqrtw))) @ VT @\
            T @ E @ la.inv(U) @ one
    
V = ...

    
if __name__ == "__main__":
    
    kD = np.array([100., 200., 300.])
    c  = np.array([150., 250., 350., 450.])
    A = sysmat(kD, c)
    
    print(A)

