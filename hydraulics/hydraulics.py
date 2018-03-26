#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 10:04:11 2018

@author: Theo
"""

import numpy as np

def lam_DW(Re, k, Dh, eps=1e-6, verbose=False):
    '''
    REturn Darcy-Weisbach lambda to compute flow in channels (Huistman, 1969, p69)
    
    solves $$ \frac 1 \lambda = - 2 log\left(frac 1 {0.4 Re \sqrt {\lambda}} +
                                \frac k {3.7 D_H} \right) $$
    
    Newton-Rpahson is applied to solve $ \lambda $
    
    parameters
    ----------
        Re : float
            Reynolds number [-], v Dh / nu
        k : float
            wall roughness [m]
        Dh : float
            Hydraulic diameter. (=4 hydraulic radius)            
    '''
    
    a, b = 1./ (0.4 * Re), k / (3.7 * Dh)
    
    yyacc = lambda x : (x + 2 * np.log(a * x + b) / np.log(10.) / 
                        (1 + 2 * a / np.log(10) / (a * x + b)))
    
    # iteration starting with x=1.
    
    x = 1.0
    for i in range(100):
        dx = yyacc(x)
        x -= dx
        if verbose:
            print('{} {:12.4g} {:12.4g} {:12.4g}'.format(i, dx, x, 1/x**2))
        if abs(dx) < eps:
            return 1 / x**2
  
def lam_channel(w=None, h=None, k=None, v=None):
    g = 9.81
    nu = 1.79e-6 # m2/s
    F = h * (w + h)
    Omega = w + np.sqrt(2) * h
    Dh = 4 * F / Omega
    Re = v * Dh / nu
    lam = lam_DW(Re, k, Dh)
    I = lam/Dh * v**2 / (2 * g)
    Q = v * F
    return lam, I * 1000., Q
      

g  = 9.81
nu = 1.79e-6

F      = 6.31
Omega  = np.array([6.62,  6.62, 12.6, 12.6])
k      = np.array([0.003, 0.08, 0.08, 0.08])
v      = np.array([0.792, 0.792, 1.0, 1.5])

Dh = 4 * F / Omega
Re = v * Dh / nu

for re, kk, dh, vv in zip(Re, k, Dh, v):
    lam = lam_DW(re, kk, dh)

    I = lam * vv**2 / (2* g) / dh
    print('lam {:10.4g}, I {:10.4g}'.format(lam_DW(re, kk, dh), I*1000))


print('\nlam {:10.4g} I {:10.4g} Q {:10.4g}'.format(*lam_channel(w=15., h=0.80, k=0.08, v=1.0)))



    
    
    