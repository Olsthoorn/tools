import os
import sys

tools = '/Users/Theo/GRWMODELS/python/tools'

if not tools in sys.path:
    sys.path.insert(1, tools)

import kml

import numpy as np

def wgs2rd(E,N, verbose=False):
    '''returns Dutch rd coordinates (x,y) from GoogleEarth WGS84 coordainates (E,N)

    (original fortran code obtained from Peter Vermeulen 2009)

    TO 091123, 171123

    '''

    E = np.asarray(E, dtype=float)
    N = np.asarray(N, dtype=float)

    if not np.all(E.shape == N.shape):
        raise ValueError('E.shape must equal N.shape')

    shape = E.shape

    E = E.ravel()
    N = N.ravel()

    deltax = 100.0   # step size to compute derivative
    deltay = 100.0   # same in y direction

    # initial values for X and Y = Amersfoort
    X = 155000 * np.ones_like(E)
    Y = 463000 * np.ones_like(N)

    if verbose: print('while', end='')
    i=0
    while True:
        i += 1
        if verbose: print('.', end='')
        if i%50 == 0: print(i)

        # All equivalent to Amersfoort
        E0, N0 = rd2wgs(X, Y)
        E1, N1 = rd2wgs(X + deltax, Y + deltay)

        dX =  (E - E0) / (E1 - E0) * deltax
        dY =  (N - N0) / (N1 - N0) * deltay

        X += dX
        Y += dY

        if verbose:
            print('E0 = ', ('{:12.0f}'*len(E)).format(*E0))
            print('N0 = ', ('{:12.0f}'*len(E)).format(*N0))
            print('E1 = ', ('{:12.0f}'*len(E)).format(*E1))
            print('N1 = ', ('{:12.0f}'*len(E)).format(*N1))
            print('dX = ', ('{:12.0f}'*len(E)).format(*dX))
            print('dY = ', ('{:12.0f}'*len(E)).format(*dY))
            print('X  = ', ('{:12.0f}'*len(E)).format(*X))
            print('Y  = ', ('{:12.0f}'*len(E)).format(*Y))
            print()

        # 1 m accuracy criterion
        if np.all(np.abs(dX) < deltax/100) and \
           np.all(np.abs(dY) < deltay/100):
            if verbose:
                print('dX = ', ('{:12.0f}'*len(E)).format(*dX))
                print('dY = ', ('{:12.0f}'*len(E)).format(*dY))
            break
        else:
            pass

    if verbose: print(i, ' iterations')
    if len(X) == 1:
        return float(X), float(Y)
    else:
        return X.reshape(shape),Y.reshape(shape)



def rd2wgs(x,y):
    '''returns WGS84 (E, N) = (lon, lat) when given Dutch rd-coords (x, y)

       fortran 90 routine received from Peter Vermeulen Deltares
       converted to Matlab by TO 090916 121007

    SEE ALSO: wgs2rd, kmlpath kmlpath2rd getDinoXSec

    TO 090916, 171225
    '''

    x = np.asarray(x)
    y = np.asarray(y)

    if not np.all(x.shape == y.shape):
        raise ValueError('x.shape must equal y.shape')

    shape = x.shape

    phibes, lambes = rd2bessel(x, y)
    phiwgs, lamwgs = bessel2wgs84(phibes, lambes)
    lon, lat       = lamwgs, phiwgs

    if isinstance(lon, float):
        return lon, lat
    else:
        return lon.reshape(shape), lat.reshape(shape)


def rd2bessel(x, y):
    '''returns bessel coordinates (phi, lam) when given Dutch rd coordinates (x, y)
    '''

    x0 = 1.55e5
    y0 = 4.63e5
    k  = 0.9999079
    bigr = 6382644.571
    m  = 0.003773953832
    n  = 1.00047585668
    e  = 0.08169683122

    lambda0 = np.pi * .029931327161111111
    b0      = np.pi * .28956165138333334

    # r, while preventing division by zero
    eps = 1e-100
    r = np.fmax(np.sqrt((x-x0)**2 + (y - y0)**2), eps)

    sa = (x - x0) / r
    ca = (y - y0) / r

    psi = np.arctan2(r, k * 2 * bigr) * 2

    cpsi = np.cos(psi);
    spsi = np.sin(psi);

    sb = ca * np.cos(b0) * spsi + np.sin(b0) * cpsi
    cb = np.sqrt(1.0 - sb ** 2)
    b  = np.arccos(cb)
    sdl = sa * spsi / cb
    dl = np.arcsin(sdl)
    lamb = dl / n + lambda0
    w = np.log(np.tan(b / 2.0 + np.pi / 4.0))
    q = (w - m) / n
    phiprime = np.arctan(np.exp(q)) * 2 - np.pi / 2

    dq = e / 2 * np.log((e * np.sin(phiprime) + 1) / (1 - e * np.sin(phiprime)))
    phi = np.arctan(np.exp(q + dq)) * 2 - np.pi / 2
    phiprime = phi

    lamb = lamb / np.pi * 180
    phi  = phi  / np.pi * 180

    return phi,lamb


def bessel2wgs84(phibes, lambes):
    ''' returns WGS84 coordinates (E, N) when given Bessel (phi, lam)
    '''

    a =  52.0
    b =   5.0
    c = -96.862
    d =  11.714
    e =   0.125
    f =   1e-5
    g =   0.329
    h =  37.902
    i =  14.667

    dphi   = phibes - a
    dlam   = lambes - b
    phicor = (c - dphi * d - dlam * e) * f
    lamcor = (dphi * g - h - dlam * i) * f
    phiwgs = phibes + phicor
    lamwgs = lambes + lamcor

    return phiwgs,lamwgs



if __name__ == '__main__': # self test

    # Amersfoort en andere punten
    x = np.array([155000, 244000,  93000,  98000, 177000])
    y = np.array([463000, 601000, 464000, 471000, 439000])

    print('\n\nInitial x, y: ')
    print('x  = ', ('{:12.0f}'*len(x)).format(*x))
    print('y  = ', ('{:12.0f}'*len(y)).format(*y))

    lon, lat = rd2wgs(x, y)
    print('lon= ', ('{:12.3f}'*len(x)).format(*lon))
    print('lat= ', ('{:12.3f}'*len(y)).format(*lat))

    X, Y = wgs2rd(lon, lat)

    print('final x, y: ')
    print('X  = ', ('{:12.0f}'*len(X)).format(*X))
    print('Y  = ', ('{:12.0f}'*len(Y)).format(*Y))

    print('\nCheck Amersfoort:')
    x, y   = 155000, 463000  # Amersfoort
    e, n   = rd2wgs(x, y)
    x1, y1 = wgs2rd(e, n)    # should return Amersfoort
    print('x={:.0f}, y={:.0f} --> e={:.4f}, n={:.4f} --> x={:.0f}, y={:.0f}'
          .format(x, y, e, n, float(x1), float(y1)))



    # convert kml file from Google Earth to rd coordinates
    kmlfile = os.path.join(tools, 'coords/data/Nederland.kml')

    map = kml.Patches(kmlfile)
    map.plot(color='brown')

    kml.nederland(color='orange')


