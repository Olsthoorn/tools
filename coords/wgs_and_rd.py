import numpy as np


def wgs2rd(E, N, verbose=False):
    """WGS2RD computes Dutch national coords (xRD,yRD) from GoogleEarth WGS coords (E,N)
    
    Example:
       xRD, yRD = wgs2rd(E, N, verbose=False)
        
    SEE ALSO: rd2wgs rd2gw gw2rd getDinoXSec kmlpath kmlpath2rd
    
    TO 091123, 220208  (original fortran code obtained from Peter Vermeulen 2009)
    """

    assert E.shape == N.shape, "E.shape ({}) != N.shape ({})".format(str(E.shape), str(N.shape)
    shape = E.shape

    deltax = 100   # step size to compute derivative
    deltay = 100   # same in y direction

    X = 155000 * np.ones(E.size)
    Y = 463000 * np.ones(E.size)
    
    dXY = np.zeros((2, E.size)) 

    while True:
        E0, N0 = rd2wgs(X, Y)
        
        dEdx, dNdx = rd2wgs(X + deltax, Y)
        dEdx = (dEdx - E0) / deltax
        dNdx = (dNdx - N0) / deltax
        
        dEdy, dNdy = rd2wgs(X, Y + deltay)
        dEdy = (dEdy - E0) / deltay
        dNdy = (dNdy - N0) / deltay

        for i in range(len(E.size)):
            dXY[ :,i] = [[dEdx[i], dEdy[i]], [dNdx[i], dNdy[i]]] \ [[E[i] - E0[i]], [N[i] - N0[i]]]
        
        E1, N1 = rd2wgs(X + DXY.T[0], Y + DXY.T[1])
        
        if verbose:
            print('DE={%12.4f} DN={%12.4f}  DX={%12.4g} DY={%12.4g}'.format(E - E1, N - N1, DX, DY))
        
        if np.all(np.abs(DX) < deltax) and np.all(np.abs(DY) < deltay):
            break
            
    return (X+DX).reshape(shape), (Y+DY).reshape(shape)



def rd2wgs(x=None, y=None): # returns [lamwgs,phiwgs,LL]
    """Return WGS Lat(Easting) Long(Northing) RD2WGS from Dutch rd coordinates.

    USAGE
    -----
    long, lat, LL] = rd2wgs(x, y)
    
    Origin:
        fortran 90 routine received from Peter Vermeulen
        converted to Matlab by TO 090916

    TO 090916
    """
    if x is None or y is None:
        lamwgs, phiwgs, LL=selftest()
        return lamgw,  phiwgs, LL
    
    phibes, lambes = rd2bessel(x, y)
    phiwgs, lamwgs = bessel2wgs84(phibes, lambes)
    
    N=len(x(:))
    
    LL = []
    for i=1:len(phibes):
        LL.append('N {%.7g}, E {%.7g}'.format(phiwgs[i],lamwgs[i]))
        
    return lamwgs, phiwgs, LL
                
def rd2bessel(x, y):
    """Convert xy to Bessel,"""
    
    x0 = 1.55e5
    y0 = 4.63e5
    k  =.9999079
    bigr = 6382644.571
    m = .003773953832
    n = 1.00047585668
    e = .08169683122
    
    lambda0 = np.pi * .029931327161111111
    b0 = np.pi * .28956165138333334
    d__1 = x - x0
    d__2 = y - y0
    r = np.sqrt(d__1 * d__1 + d__2 * d__2)
    
    sa = (x - x0) / r
    ca = (y - y0) / r

    sa( r==0 )=0.0
    ca( r==0 )=0.0

    psi = np.atan2(r, k * 2 * bigr) * 2
    cpsi = np.cos(psi)
    spsi = np.sin(psi)
    sb = ca * np.cos(b0) * spsi + np.sin(b0) * cpsi
    d__1 = sb
    cb = np.sqrt(1.0 - d__1  * d__1)
    b = np.acos(cb)
    sdl = sa * spsi / cb
    dl = np.asin(sdl)
    lamb = dl / n + lambda0
    w = np.log(np.tan(b / 2 + np.pi / 4))
    q = (w - m) / n
    phiprime = np.atan(np.exp(q)) * 2 - np.pi / 2

    for _ in range(4): # presumably iterates 4 times
        dq = e / 2 * np.log((e * np.sin(phiprime) + 1) / (1 - e * np.sin(phiprime)))
        phi = np.atan(np.exp(q + dq)) * 2 - np.pi / 2
        phiprime = phi;

    lamb = lamb / np.pi * 180
    phi  = phi  / np.pi * 180
    
    return phi,lamb
        
def bessel2wgs84(phibes, lambes):
    """Convert Bessel2 WGS84.""""
    a = 52.0
    b = 5.0
    c = -96.862
    d = 11.714
    e = 0.125
    f = 1e-5
    g = 0.329
    h = 37.902
    i = 14.667;
    
    dphi = phibes - a
    dlam = lambes - b
    phicor = (c - dphi * d - dlam * e) * f
    lamcor = (dphi * g - h - dlam * i) * f
    phiwgs = phibes + phicor
    lamwgs = lambes + lamcor
    return phiwgs,lamwgs
            
def selftest():
    """Amersfoort en andere punten."""
    x = np.array([155000, 244000, 93000, 98000, 177000])
    y = np.array([463000, 601000, 464000, 471000, 439000])
    long, lat, LL = rd2wgs(x,y)
    return long, lat, LL
    

