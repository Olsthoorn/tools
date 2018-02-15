#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 14:39:04 2018

Conversie van RD naar WGS en omgekeerd

http://home.solcon.nl/pvanmanen/Download/Transformatieformules.pdf

Schreutelkamp, F.H. en G.L. Strang van Hees (2001)


@author: Theo
"""
import numpy as np

# The first value is for zone 31, the second for zone 32
# Zone 31 is west of 6 degrees and zone 32 east of 6 degrees.

E0 = np.array([ 663304.11,  252878.65])
N0 = np.array([5780984.54, 5784453.44])
A0 = E0
B0 = N0
A1 = np.array([99947.539, 99919.783])
B1 = np.array([3290.106,  -4982.166])
A2 = np.array([20.008,      -30.208])
B2 = np.array([1.310,         3.016])
A3 = np.array([2.041,         2.035])
B3 = np.array([0.203,        -0.309])
A4 = np.array([0.001,        -0.002])
B4 = np.array([0.000,         0.001])

X0 = np.array([155000., 155000.])
Y0 = np.array([463000., 463000.])
C0 = X0
D0 = Y0
C1 = np.array([99944.187, 99832.079])
D1 = np.array([-3289.996, 4977.793])
C2 = np.array([-20.039,     30.280])
D2 = np.array([0.668,        1.514])
C3 = np.array([-2.042,      -2.034])
D3 = np.array([0.066,       -0.099])
C4 = np.array([0.001,       -0.001])
D4 = np.array([0.000,        0.000])

#%% From UTM 2 RD ===============================

def rd2utm(X, Y):

    if isinstance(X, np.ndarray):
        Iz = np.zeros_like(X, dtype=int)
        Iz[X>197000] = 1  # Zone 31 or zone 32
        dx = (X - X0[Iz]) * 1e-5
        dy = (Y - Y0[Iz]) * 1e-5
    else:
        Iz = 0 if X < 197000. else 1
        dx = (X - X0[Iz]) * 1e-5
        dy = (Y - Y0[Iz]) * 1e-5

    E = E0[Iz] +\
        A1[Iz] * dx -\
        B1[Iz] * dy +\
        A2[Iz] * (dx ** 2 - dy ** 2) -\
        B2[Iz] * (2 * dx * dy) +\
        A3[Iz] * (dx ** 3 - 3 * dx * dy ** 2) -\
        B3[Iz] * (3 * dx ** 2 * dy - dy ** 3) +\
        A4[Iz] * (dx ** 4 - 6 * dx ** 2 * dy ** 2 + dy **4) -\
        B4[Iz] * (4 * dx ** 3 * dy - 4 * dy **3 * dx)

    N = N0[Iz] +\
        B1[Iz] * dx +\
        A1[Iz] * dy +\
        B2[Iz] * (dx **2 - dy ** 2) +\
        A2[Iz] * (2 * dx * dy) +\
        B3[Iz] * (dx**3 - 3 * dx * dy ** 2) +\
        A3[Iz] * (3 * dx ** 2 * dy - dy ** 3) +\
        B4[Iz] * (dx ** 4 - 6 * dx **2 * dy ** 2 + dy ** 4) +\
        A4[Iz] * (4 * dx ** 3 * dy - 4 * dy ** 3 * dx)

    return E, N

# from WGS to UTM ====================================
def utm2rd(E, N):

    if isinstance(E, np.ndarray):
        Iz = np.zeros_like(E, dtype=0) # zone 31 or zone 32
        Iz[E>300000.] = 0 # Zone 31 (west of Apeldoorn)
        dE = (E - E0[Iz]) * 1e-5
        dN = (N - N0[Iz]) * 1e-5
    else:
        Iz = 0 if E>500000. else 1
        dE = (E - E0[Iz]) * 1e-5
        dN = (N - N0[Iz]) * 1e-5

    X = X0[Iz] +\
        C1[Iz] * dE -\
        D1[Iz] * dN +\
        C2[Iz] * (dE ** 2 - dN ** 2) -\
        D2[Iz] * (2 * dE * dN) +\
        C3[Iz] * (dE ** 3 -3 * dE * dN ** 2) -\
        D3[Iz] * (3 * dE **2 * dN - dN ** 3) +\
        C4[Iz] * (dE ** 4 - 6 * dE **2 * dN ** 2 + dN ** 4) -\
        D4[Iz] * (4 * dE **3 * dN - 4 * dN ** 3 * dE)

    Y = Y0[Iz] +\
        D1[Iz] * dE +\
        C1[Iz] * dN +\
        D2[Iz] * (dE ** 2 - dN ** 2) +\
        C2[Iz] * (2 * dE * dN) +\
        D3[Iz] * (dE ** 3 - 3 * dE * dN ** 2) +\
        C3[Iz] * (3 * dE ** 2 * dN - dN ** 3) +\
        D4[Iz] * (dE ** 4 - 6 * dE ** 2 * dN ** 2 + dN ** 4) +\
        C4[Iz] * (4 * dE ** 3 * dN - 4 * dN ** 3 * dE)

    return X, Y

#%% from RD to WGS ====================================
phi0 = 52.15517440 # northing
lam0 = 5.38720621  # easting

# ............ PK, QK,  Kpq,     PL, QL,  Lpq
F = np.array([(0, 1, 3235.65389, 1, 0, 5260.52916),
             (2, 0,  -32.58297, 1, 1,  105.94684),
             (0, 2,   -0.24750, 1, 2,    2.45656),
             (2, 1,   -0.84978, 3, 0,   -0.81885),
             (0, 3,   -0.06550, 1, 3,    0.05594),
             (2, 2,   -0.01709, 3, 1,   -0.05607),
             (1, 0,   -0.00738, 0, 1,    0.01199),
             (4, 0,    0.00530, 3, 2,   -0.00256),
             (2, 3,   -0.00039, 1, 4,    0.00128),
             (4, 1,    0.00033, 0, 2,    0.00022),
             (1, 1,   -0.00012, 2, 0,   -0.00022),
             (0, 0,    0.0,     5, 0,    0.00026)])

def rd2wgs(X, Y):

    x0 = X0[0]
    y0 = Y0[0]

    PK  = np.array(F[:,0], dtype=int)
    QK  = np.array(F[:,1], dtype=int)
    Kpq = np.array(F[:,2])
    PL  = np.array(F[:,3], dtype=int)
    QL  = np.array(F[:,4], dtype=int)
    Lpq = np.array(F[:,5])


    dx = (X - x0) * 1e-5
    dy = (Y - y0) * 1e-5

    phi = 0. * Y  # so that X and phi are of the same type
    lam = 0. * X  # so that Y and lam are of the same type

    for pk, qk, kpq, pl, ql, lpq in zip(PK, QK, Kpq, PL, QL, Lpq):
        phi += kpq * dx ** pk * dy ** qk
        lam += lpq * dx ** pl * dy ** ql

    return lam0 + lam/3600., phi0 + phi/3600.

#%% From WGS to RD ===================================

G = np.array([(0, 1, 190094.945, 1, 0, 309056.544),
             (1, 1, -11832.228, 0, 2,   3638.893),
             (2, 1,   -114.221, 2, 0,     73.077),
             (0, 3,    -32.391, 1, 2,   -157.984),
             (1, 0,     -0.705, 3, 0,     59.788),
             (3, 1,     -2.340, 0, 1,      0.433),
             (1, 3,     -0.608, 2, 2,     -6.439),
             (0, 2,     -0.008, 1, 1,     -0.032),
             (2, 3,      0.148, 0, 4,      0.092),
             (0, 0,      0.0,   1, 4,     -0.054)])

def wgs2rd(E, N):

    phi0 = 52.15517440
    lam0 = 5.38720621

    dphi = 0.36 * (N - phi0)
    dlam = 0.36 * (E - lam0)

    if isinstance(dphi, np.ndarray):
        dphi = np.array(dphi)
        dlam = np.array(dlam)

    PR  = np.array(G[:,0], dtype=int)
    QR  = np.array(G[:,1], dtype=int)
    Rpq = G[:,2]
    PS  = np.array(G[:,3], dtype=int)
    QS  = np.array(G[:,4], dtype=int)
    Spq = G[:,5]

    X = X0[0]
    Y = Y0[0]

    for pr, qr, rpq, ps, qs, spq in zip(PR, QR, Rpq, PS, QS, Spq):
        X += rpq * dphi ** pr * dlam ** qr
        Y += spq * dphi ** ps * dlam ** qs
    return X, Y

def toMdl(xyW, georef):
    '''Return model coordinates.

    parameters
    ----------
        xyW: ndarray [n, 2]
            array of world coordinates
        georef : (xm0, ym0, xw0, yw0, alfa)
            xm0, ym0 is rotation point in model coordinates
            yw0, yw1 is rotation point in world coordinates
            alfa is the rotation of the model counter-clockwise in degrees.
    '''

    xm0, ym0, xw0, yw0, alfa = georef

    xyM0  = np.array([[xm0], [ym0]])
    xyW0  = np.array([[xw0], [yw0]])
    alpha = alfa * np.pi/180.

    if np.ndim(xyW) == 1:
        xyW = xyW[:, np.newaxis]
    else:
        xyW = xyW.T

    dxyW = xyW - xyW0

    rot = [(np.cos(alpha), np.sin(alpha)), (-np.sin(alpha), np.cos(alpha))]

    return (np.dot(rot, dxyW) + xyM0).T


def toWld(xyM, georef):
    '''Return world coordinates.

     parameters
     ----------
        xyM: ndarray [n, 2]
            array of model coordinates
        georef : (xm0, ym0, xw0, yw0, alfa)
            xm0, ym0 is rotation point in model coordinates
            yw0, yw1 is rotation point in world coordinates
            alfa is the rotation of the model counter-clockwise in degrees.
    '''

    xm0, ym0, xw0, yw0, alfa = georef

    xyM0  = np.array([[xm0], [ym0]])
    xyW0  = np.array([[xw0], [yw0]])
    alpha = alfa * np.pi/180.

    if np.ndim(xyM) == 1:
        xyM = xyM[:, np.newaxis]
    else:
        xyM = xyM.T

    dxyM = xyM - xyM0

    rot = [(np.cos(alpha), -np.sin(alpha)), (np.sin(alpha), np.cos(alpha))]

    return (np.dot(rot, dxyM) + xyW0).T


if __name__ == '__main__':

    #from coords import wgs2rd

    pnts = np.array([[  3.812747,  51.334768],
                   [  3.827913,  51.335037],
                   [  3.815888,  51.323154],
                   [  3.828413,  51.321776],
                   [  3.818668,  51.326969],
                   [  3.820776,  51.332901]])

    XY0 = np.vstack(wgs2rd(pnts[:,0], pnts[:,1])).T
    XY1 = np.vstack(wgs2rd(pnts[:,0], pnts[:,1])).T
    print(XY0)
    print(XY1)

    # Westertoren
    wtwgs0 = 4.88352559, 52.37453253
    wtrd0  = 120700.723, 487525.502
    wtutm0 = 628217.312, 5804365.552

    print("Rondje coordinatentransformatie Westertoren")
    print('WGS :', wtwgs0, 'Startwaarde in wgs.')

    wtrd = wgs2rd(*wtwgs0)
    print('RD  ', wtrd, 'Van wgs naar rd.')

    wtutm = rd2utm(*wtrd)
    print('UTM: ', wtutm, 'Van rd naar utm.')

    wtrd1  = utm2rd(*wtutm)
    print('RD  : ', wtrd1, 'Van utm naar rd')

    wtwgs1 = rd2wgs(*wtrd1)
    print('wgs " ', wtwgs1, 'Van rd turug naar wgs.')

    wtrd2 = wgs2rd(*wtwgs1)
    print('RD " ', wtrd2, 'Van wgs terug naar wgs (verificatie in m.')



    # Martinitoren utm zone 32
    mtwgs0 = 6.56820053, 53.21938317
    mtrd0  = 233883.131, 582065.167
    mtutm0 = 337643.235, 5899435.841

    print("Rondje coordinatentransformatie Martinitoren")
    print('WGS :', mtwgs0, 'Startwaarde in wgs.')

    mtrd = wgs2rd(*mtwgs0)
    print('RD  ', mtrd, 'Van wgs naar rd.')

    mtutm = rd2utm(*mtrd)
    print('UTM: ', mtutm, 'Van rd naar utm.')

    mtrd1  = utm2rd(*mtutm)
    print('RD  : ', mtrd1, 'Van utm naar rd')

    mtwgs1 = rd2wgs(*mtrd1)
    print('wgs " ', mtwgs1, 'Van rd turug naar wgs.')

    mtrd2 = wgs2rd(*mtwgs1)
    print('RD " ', mtrd2, 'Van wgs terug naar wgs (verificatie in m.')


    #%% world to model and model to world coordinates

    import coords

    JkFb = (183642., 338243.)
    JkGl = (183350., 337477.)
    JkKb = (182872., 336280.)
    JkHh = (182458., 335511.)

    fxpnts = np.array([JkFb, JkGl, JkKb, JkHh])

    xy0 = np.array(JkHh[:])

    alfa = -24.4 # degrees counter clockwise

    xyM = coords.toMdl(fxpnts, xy0, alfa)
    xyW = coords.toWld(xyM, xy0, alfa)
    print("fxpnts")
    print(fxpnts)
    print("xyM")
    print(xyM)
    print("xyW")
    print(xyW)


