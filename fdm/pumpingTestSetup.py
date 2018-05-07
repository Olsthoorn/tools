# -*- coding: utf-8 -*-
"""
Spyder Editor

Axial model setup for Python-implemented transient fdm3t model

Also exercise in asynchronus (parallel) simulation (future).

This script can be used as a wrapper for efficient pumping test analysis
and its parameter optimization

TO 20180524, Montereggliano, Italy
"""

from fdm import fdm3t, mfgrid
import numpy as np
#import asyncio

      
def grid(rW=None, zGr=None, xGr=None):
    if xGr is None:
        xGr = np.hstack((0., np.logspace(np.log10(rW), 4, 51))) # make sure xGr[-1] is large enough
    yGr = [-0.5, 0.5]
    return mfgrid.Grid(xGr, yGr, zGr, axial=True)
    
def k_s(gr, c=None, kD=None, S=None):
    # gnerate Kh, Kv, Ss
    Kh, Kv = gr.ckD2k(c, kD)
    Ss     = gr.s2ss(S)
    return Kh, Kv, Ss
    
def get_FQ(gr, z, Q, kh):
    Iw, Q = gr.well(x=0., y=0., z=z, Q=Q, kh=kh)
    if len(Iw) == 0:
        raise 'No cells found to put the well. Check zWell.'
    FQ = gr.const(0.)
    FQ.ravel()[Iw] = Q
    return FQ, Iw

def main(zGr, c, kD, S, t, rWell, zWell, Q):
    '''Run axial fdm model and return results.
    parameters
    ----------
        zGr : ndarray(nLay)
            z values of layer interfaces
        c : ndarray(naquitard)
            resistances [d] of aquitards
        kD : ndarray(naquifer)
            tranmissivities of aquifers
        S : ndarray(naquitard + naquifer)
            strorage coefficients of all aquitards and aquifers in order
        t : ndarray(nt)
            times when output is wanted
        rWell : float
            well radius
        zWell : tuple of 2 floats
            top and bottom of well screen
        Q : float
            if neg. extraction from screen of positive injection
    '''
    gr = grid(rWell, zGr) # use default x and y coordinates
    kh, kv, ss = k_s(gr, c=c, kD=kD, S=S)
    FQ, Iw = get_FQ(gr, z=zWell, Q=Q, kh=kh)
    HI = gr.const(0.)
    IBOUND = gr.const(1, dtype=int)
    IBOUND[0] = -1

    kh.ravel()[Iw] = 1000. # hard wired, gravel pack    
    if np.all(IBOUND[0]==-1): kv[0] /= 2.0 # top layer resistance with centered heads
    
    kwargs ={'gr': gr, 't' : t, 'kxyz': (kh, kh, kv), 'Ss': ss,
          'FQ': FQ, 'HI': HI, 'IBOUND': IBOUND, 'epsilon': 1.0}
    
    return fdm3t.fdm3t(**kwargs)

def Jphi(piezoms):
    '''return contribution of obj function by head differences
    '''
    dJ = 0.
    for piez in piezoms:
        dJ += np.sum(piez(gr, out) ** 2)
    return dJ
    
def Jpriori(params):
    '''return condtributin to obj. function by a prirori paramter values
    '''
    dJ = 0.
    for par in params:
        dJ += np.sum(par(gr, out) ** 2)
    return dJ

if __name__ == "__main__":
    
    # just specify the essential parameters and run main
    zGr = [-40, -35, -25, -20, -10, -7.5, 0]
    c   = [100, 250, 400.]
    kD  = [350, 450, 550]
    S   = [0, 1e-3, 0, 1e-3, 0, 1e-3]
    t      = np.logspace(-3, 1, 41)
    rWell, zWell, Q = 0.25, (-20., -40.), -1200.
    out = main(zGr, c, kD, S, t, rWell, zWell, Q)
    
    
    J = Jphi(piezoms) + L * Jparams
    
    
