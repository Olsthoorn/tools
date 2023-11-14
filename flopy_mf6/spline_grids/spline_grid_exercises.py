#!/usr/bin/env python3

# Gerating mf6.VDIS grids with regular rows and columns using both affine
# transformation in quadraingular blocks and splines.
# The total number of cells along the row or column can be specified as well as
# the total number of cells along rows or columns of each block. This is true
# for the affine grid as it is for the spline grie.

# The grid will be generatd by hand as well as by functions for implementation in
# a gridv grid object.

# TO 20220325

# %%

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import flopy

from scipy.interpolate import splprep, splev

tools = '/Users/Theo/GRWMODELS/python/tools/'
sys.path.append(tools)

from flopy_mf6.homogenCoords import Q2US

rv = lambda Z: Z.ravel()

# %%

def showvec(dxy, ls='b-', label=""):
    """Plot the vector dxy with given liinestyle."""
    plt.plot([0., dxy[0]], [0., dxy[1]], ls, label=label)

def Mrotate(alpha, degrees=False):
    """Return 2D coordinatesaffine rotation matrix"""
    if degrees:
        alpha *= np.pi / 180.
    return np.array([[np.cos(alpha), -np.sin(alpha), 0.],
                     [np.sin(alpha),  np.cos(alpha), 0.],
                     [0., 0., 1.]])

def Mstretch(Lx, Ly):
    """Return 2D affine stretch matrix"""
    return np.array([[ Lx,  0., 0.],
                     [  0., Ly, 0.],
                     [  0., 0., 1.]])

def Mshear(Sx, Sy):
    """Return 2D affine shearing matrix"""
    return np.array([[  1.,  Sx, 0.],
                     [  Sy,  1., 0.],
                     [  0., 0., 1.]])


def Mmove(dx, dy):
    """Return 2D affine transpose matrix"""
    return np.array([[  1.,  0., dx],
                     [  0.,  1., dy],
                     [  0., 0., 1.]])

class Grobj:
    """Graphics object (bunch of points that may be manipulated and shown."""
    def __init__(self, points):
        points = np.array(points)
        assert points.ndim == 2 and points.shape[1] == 2, 'points must be a sequence of (n, 2) of floats'
        self.points = np.hstack(((points, np.ones((points.shape[0], 1)))))
        
    def rotate(self, alpha, degrees=False):
        M = Mrotate(alpha, degrees=degrees)
        self.points = (M @ self.points.T).T
        return self

    def stretch(self, Lx, Ly):
        M = Mstretch(Lx, Ly)
        self.points = (M @ self.points.T).T
        return self

    def shear(self, Sx, Sy):
        M = Mshear(Sx, Sy)
        self.points = (M @ self.points.T).T
        return self

    def move(self, dx, dy):
        M = Mmove(dx, dy)
        self.points = (M @ self.points.T).T
        return self
    
    def plot(self, fmt, **kwargs):
        ax =plt.gca()
        ax.plot(self.points[:, 0], self.points[:, 1], fmt, **kwargs)


# %%

def points_in_cells(X, Y, p):
    """Return the grid cells the point is in.
    
    The algorithm checks if each point is to the left of all for 4 cell edges.
    This function is fully vectorized by putting info per cell on rows and per point in column.
    
    Paameters
    ---------
    X, Y: ndarrays (nrow + 1, ncol + 1)
        Vertex coordinates of grid (y decreasing)
    p: 2-vector (point)
        Point to check.
        
    Returns
    -------
    array of cell numbers telling in which cell each given point lies.
    """
    ll = lambda X: X[ 1:, :-1]
    lr = lambda X: X[ 1:,  1:]
    ur = lambda X: X[:-1,  1:]
    ul = lambda X: X[:-1, :-1]
    rv = lambda X: X.ravel()[:, np.newaxis] # Use column vector for the sides
    
    p = np.array(p)
    if p.shape == (1, 2):
        p = p[np.newaxis, :]
        
    # edges of all cells (bottom, right, top, left) All column vectors
    dxb, dyb = rv(lr(X) - ll(X)), rv(lr(Y) - ll(Y))
    dxr, dyr = rv(ur(X) - lr(X)), rv(ur(Y) - lr(Y))
    dxt, dyt = rv(ul(X) - ur(X)), rv(ul(Y) - ur(Y))
    dxl, dyl = rv(ll(X) - ul(X)), rv(ll(Y) - ul(Y))
    
    xp, yp = p[:, 0][np.newaxis, :], p[:, 1][np.newaxis, :] # Both row vectors
    # vectors of point to the 4 cell corners (ncells, npoints)
    # using full broadcasting (np, nrow * ncol)
    dxpll, dypll = xp - rv(ll(X)), yp - rv(ll(Y)) # (ncells, npoints) by broad casting
    dxplr, dyplr = xp - rv(lr(X)), yp - rv(lr(Y))
    dxpur, dypur = xp - rv(ur(X)), yp - rv(ur(Y))
    dxpul, dypul = xp - rv(ul(X)), yp - rv(ul(Y))
        
    incells = np.zeros_like(dxpll, dtype=int)
    # point left of eeach of the 4 cell edges?
    incells += (dxb * dypll - dyb * dxpll) > 0
    incells += (dxr * dyplr - dyr * dxplr) > 0
    incells += (dxt * dypur - dyt * dxpur) > 0
    incells += (dxl * dypul - dyl * dxpul) > 0
    
    isin = np.where(incells == 4, True, False)
    Max = [np.argmax(f) for f in isin.T] # max for each column
    Min = [np.argmin(f) for f in isin.T] # Min for each column
    incell = [mx if mx != mn else -1 for mx, mn in zip(Max, Min)] # if no 4 in column, use -1, the point outside grid
    return incell # Id's of the cells where points are in


def interp_spline(ctr, n=50):
    """Return a B-spline through the control points.
    
    Parameters
    ----------
    ctr: ndarray [m, 2]
        the control points.
    n: int or ndarray of ints
        int: number of cells in total.
        ndarray of ints: number of cellsper pair of ctr points
        
    Returns 
    -------
    out: ndaray (n, 2)
        points of the spline
    """
    #from scipy.interpolate import splprep, splev

    try:
        tck, u = splprep(ctr.T, k=3, s=0)
    except:
        tck, u = splprep(ctr.T, k=1, s=0)

    if np.isscalar(n):
        U = np.linspace(0, 1, n + 1)
    else:
        n  = np.array(n)
        assert len(n) == len(u) - 1,\
            f"len(n) = {len(n)} must equal len(ctr) - 1 = {len(ctr)}"
        
        U = np.array([0.])
        for u0, u1, ni in zip(u[:-1], u[1:], n):
            U = np.append(U, np.linspace(u0, u1, ni + 1)[1:])
    return splev(U,tck) 


def get_spline_grid(Xp, Yp, nx, ny):
    """Return splined gridpoints.
    
    Parameters
    ----------
    Xp: ndarray of size (Nx, Ny)
        x-coordinates of control points outlining the grid coarsely.
    Yp: ndarray of size (Nx, Ny)
        y-coordinates of control points outlining the grid coarsely.
    nx: int or ndarray of ints
        The total number of grid cells along the rows.
        if ndarray of ints, number of cells per grid block along rows.
        len(nx) must be Xp.shape[0] - 1
    ny: int or ndarray of ints
        If int: total number of grid cells along columns.
        if ndarray of ints: number of grid cells between to control point rows.
        len(ny) must be Xy.shape[0] - 1
    
    Returns
    -------
    X, Y: ndarrays of size (sum(ny) + 1, sum(nx) + 1) of grid coordinates
    """ 

    # Check input:
    assert np.all(Xp.shape == Yp.shape), f"Xp.shape={Xp.shape} <> Yp.shape{Yp.shape}"
    if np.isscalar(nx):
        nx_total = nx
    else:
        nx_total = np.sum(nx)
        assert Xp.shape[1] == len(nx) + 1,\
          f"len(nx_={len(nx)} array must be len Xp.shape[0] - 1={Xp.shape[1] - 1}"
    if np.isscalar(ny):
        ny_total = ny
    else:
        ny_total = np.sum(ny)
        assert Xp.shape[0] == len(ny) + 1,\
          f"len(nx_={len(ny)} array must be Xp.shape[0] - 1={Xp.shape[0] - 1}"

    # Generate points along the splines throug X[j], Y[j]
    Xh = np.zeros((Xp.shape[0], nx_total + 1))
    Yh = np.zeros((Yp.shape[0], nx_total + 1))

    for j, (xp, yp) in enumerate(zip(Xp, Yp)):
        ctr = np.vstack((xp, yp)).T
        x, y = interp_spline(ctr, nx)
        Xh[j, :], Yh[j, :] = x, y

    # Generate points through each Xh[:, i], Yh[:, i]
    X = np.zeros((ny_total + 1, nx_total + 1))
    Y = np.zeros((ny_total + 1, nx_total + 1))
    for i, (xp, yp) in enumerate(zip(Xh.T, Yh.T)):
        ctr = np.vstack((xp, yp)).T
        x, y = interp_spline(ctr, ny)
        X[:, i] = x 
        Y[:, i] = y 
    return X, Y # Return coordinates of final grid

# %%

def get_vertcells_from_diamond(xd=None, yd=None, nrow=None, ncol=None):
    """Return vertices and cell2D in diamond shape for mf6 disv package.
    
    Parameters
    ----------
    xp, yp:  2 ndarrays or squences of each 4 points
        corner points of diamond, anti-clock wise
    now, ncol: 2 ints
        number of rows and number of columns of grid.
        
    Returns
    -------
    vertices: list of list, each record [iv, xv, yv] (int, float, float)
        definition of cell corners in horizontal plan
    cell2D: list of list, each record [ic, xc, yc, nvert, verts]
        definiction of cells in horizontal plane
    
    Examples
    ----------
    >>> xd = np.array([0, 25, 30, 5])
    >>> yd = np.array([0, 15, 35, 40])
    >>> vertices, cell2D = get_vertcells_from_diamond(xp, yp, nrow=nrow, ncol=ncol)
    """
    assert len(xd) == 4 and len(yd) == 4, "len(xp) and len(yp) must both be 4."
    xd, yd = np.array(xd), np.array(yd)
    x = np.array([xd[1] - xd[0], yd[1] - yd[0], 0.])
    y = np.array([xd[3] - xd[0], yd[3] - yd[0], 0.])
    assert np.cross(x, y)[-1] > 0, "xp and yp must be arranged anti-clockwise."
    
    ctr = lambda X: 0.25 * (X[:-1, :-1] + X[:-1, 1:] + X[1:, 1:] + X[1:, :-1])
    rv = lambda X: X.ravel()

    # ======== Generate NODES (cell vertices) in diamond first ====================
    # Relative coordinates to refer to points inside the diamond. -0.5<= u,v <= 0.5
    u = np.linspace(-0.5, 0.5, ncol + 1)
    v = np.linspace(-0.5, 0.5, nrow + 1)
    U, V = np.meshgrid(u, v)

    # Array of relative coordinates within the diamond.
    UV = np.vstack((+rv((U - 0.5) * (V - 0.5)),
                    -rv((U + 0.5) * (V - 0.5)),
                    +rv((U + 0.5) * (V + 0.5)),
                    -rv((U - 0.5) * (V + 0.5)))).T

    # Cell vertices inside the bounding diamond
    Iv = np.arange((nrow + 1) * (ncol + 1)).reshape((nrow + 1), ncol + 1)
    Xv = (UV @ xd).reshape((nrow + 1, ncol + 1))
    Yv = (UV @ yd).reshape((nrow + 1, ncol + 1))

    vertices = [[iv, xv, yv] for iv, xv, yv in zip(rv(Iv), rv(Xv), rv(Yv))]

    vertices_ra = np.zeros(len(rv(Iv)), np.dtype([('x', float), ('y', float)]))
    vertices_ra['x'] = rv(Xv)
    vertices_ra['y'] = rv(Yv)

    
    # ======= Get cell centers and cell vertices next =============================
    Ic = np.arange(nrow * ncol).reshape((nrow, ncol))
    Xc, Yc = ctr(Xv), ctr(Yv)
    # Vertex index of cell corners, clockwise
    Iverts = np.vstack((rv(Iv[:-1, :-1]), rv(Iv[1:, :-1]), rv(Iv[1:, 1:]), rv(Iv[:-1, 1:]))).T
    cell2D = [[ic, xc, yc, 4, verts]
                 for ic, xc, yc, verts in zip(rv(Ic), rv(Xc), rv(Yc), rv(Iverts))]
    
    cell2D_ra = np.zeros(len(rv(Ic)), dtype=[
        ('x', 'float'), ('y', float), ('nvert', float), ('verts', 'int', 4)])
    
    cell2D_ra['x'] = Xc.ravel()
    cell2D_ra['y'] = Yc.ravel()
    cell2D_ra['nvert']= 4
    cell2D_ra['verts'] = Iverts
                        
    return vertices, cell2D, vertices_ra, cell2D_ra

# nrow, ncol = 6, 7
# xd = np.array([0, 25, 30, 5])
# yd = np.array([0, 15, 35, 40])
# vertices, cell2D, vertices_ra, cell2D_ra = get_vertcells_from_diamond(xd, yd, nrow=nrow, ncol=ncol)

# %% grid of a grid of quadraingles for Modflow mf6

def newfig(title="", xlabel="", ylabel="", xlim=None, ylim=None, aspect=None):
    _, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    if aspect: ax.set_aspect(aspect)
    ax.grid(True)
    return ax

def show_grid(title, vertices_ra, cell2D_ra):
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.plot(vertices_ra['xv'], vertices_ra['yv'], '.k')
    plt.plot(cell2D_ra['xc'], cell2D_ra['yc'], 'rx')

    for verts in cell2D_ra['verts']:    
        plt.plot(vertices_ra['xv'][verts], vertices_ra['yv'][verts], 'b')
    return plt.gca()

def xdyd_getter(Xp, Yp):
    """Return next xd, yd.
    
    Parameters
    ----------
    Xp, Yp: ndarray (Ny, Nx):
        Coordinates of quadraingles from which the grid
        is constructed.
        
    Returns
    -------
    xd, yd: ndarray of length 4.
        Clockwise coordinates of corners of next quadrangle.
    """
    Xp, Yp = np.array(Xp), np.array(Yp)
    assert np.all(Xp.shape == Yp.shape) and len(Xp.shape) == 2,\
        "Xp and Yp must be 2D and of the same shape."
    ny, nx = Xp.shape
    for j in range(ny - 1):
        for i in range(nx - 1):
            xd = np.array([Xp[j, i], Xp[j, i+1], Xp[j+1, i+1], Xp[j+1, i]])
            yd = np.array([Yp[j, i], Yp[j, i+1], Yp[j+1, i+1], Yp[j+1, i]])
            yield xd, yd
    return

def subgrid(xd, yd, nx, ny):
    """Return coordinates of subgrid in quandrangle given by the 4 points xd, yd.
    
    Parameters
    ----------
    xd, yd: ndarray (4)
        coordinates of the 4 corners of the quandrangle outlining this subgrid.
    nx, ny: 2 ints
        number of cells (not vertices) within subgrid in x and y direction.
    """
    rv = lambda Z: Z.ravel()
    
    u = np.linspace(-0.5, 0.5, nx + 1)
    v = np.linspace(-0.5, 0.5, ny + 1)
    U, V = np.meshgrid(u, v)
    
        # Array of relative coordinates within the diamond.
    UV = np.vstack((+rv((U - 0.5) * (V - 0.5)),
                    -rv((U + 0.5) * (V - 0.5)),
                    +rv((U + 0.5) * (V + 0.5)),
                    -rv((U - 0.5) * (V + 0.5)))).T

    # Cell vertices inside the bounding diamond
    Iv = np.arange((ny + 1) * (nx + 1)).reshape((ny + 1), nx + 1)
    Xv = (UV @ xd).reshape((ny + 1, nx + 1))
    Yv = (UV @ yd).reshape((ny + 1, nx + 1))
    return Xv, Yv

def quadrangle_grid(Xp, Yp, Nx, Ny):
    """Return a structured grid defined by xc, Yc, nx and ny.
    
    Parameters
    ----------
    Xp: np.array (Ny, Nx)
        Array of the x doords of the corner points of the quadrangles making up the grid.
    Yp: np.array (Ny, Nx)
        Array of the y coords of the corner ponts of the quadrangles making up the grid.
    Nx: seq of len(Xp.dims[1]) of ints.
        the number of cells along x-direction for each quadrangle.
    Ny: sed of len(Yp.dims[1]) of ints
        the number of cells along y-direction for each quadrangle.
    
    Returns
    -------
    vertices: list of lists [iv, xv, yv] defining the vertices
        The vertices of the grid.
    cell2D: list of list defining the cells [ic, xc, yc, nvert, vertIds]
        The cells.
    vertices_ra: np.recarray defining the verties
        same as vertices but as a recarray instead of a list.
    cell2D_ra: np.recarray or strutured array
        same as cell2D but as a recarray instead of a list.
    
    Examples
    --------
    >>> Xp = np.array([ [  0., 15., 30., 40., 65],
    >>>                 [ -5., 10., 25., 45., 70],
    >>>                 [ -3., 13., 27., 41., 65],
    >>>                 [-10., 11., 23., 37., 67]])
    >>> Yp = np.array([ [50., 45., 55., 50., 52.],
    >>>                 [37., 35., 40., 36., 35],
    >>>                 [18., 22., 19., 22., 20],
    >>>                 [ 5., -3.,  1., 5.,  4]])
    >>> Nx = np.array([5, 3, 12, 5])
    >>> Ny = np.array([6, 4, 10])
    >>> vertices, cell2D, vertices_ra, cell2D_ra = quadrangle_grid(Xp, Yp, Nx, Ny)
    >>> 
    >>> plt.plot(vertices_ra['xv'], vertices_ra['yv'], '.k')
    >>> plt.plot(cell2D_ra['xc'], cell2D_ra['yc'], 'rx')
    >>> 
    >>> for verts in cell2D_ra['verts']:    
    >>>     plt.plot(vertices_ra['xv'][verts], vertices_ra['yv'][verts], 'b')
    >>> plt.show()
    """
    rv = lambda Z: Z.ravel()
    ctr = lambda Z: 0.25 * (Z[:-1, :-1] + Z[:-1, 1:] + Z[1:, 1:] + Z[1:, :-1])
    
    Xp, Yp = np.array(Xp), np.array(Yp)
    assert len(Xp.shape) == 2 and np.all(Xp.shape == Yp.shape),\
        "Xp and Yp must be 2D arrays of quadrangle corner coordinates."
    Nx, Ny = np.array(Nx, dtype=int), np.array(Ny, dtype=int)
    assert len(Nx) == Xp.shape[1] - 1, "Nx must be sequence of ints of len Xp.shape[1] - 1"
    assert len(Ny) == Xp.shape[0] - 1, "Ny must be sequence of ints of len Xp.shape[0] - 1"
    
    nrow, ncol = np.sum(Ny), np.sum(Nx)
    X = np.zeros((nrow + 1, ncol + 1))
    Y = np.zeros((nrow + 1, ncol + 1))
    Ix = np.hstack((0, np.cumsum(Nx)))
    Iy = np.hstack((0, np.cumsum(Ny)))
    
    nxdyd = xdyd_getter(Xp, Yp)

    for j1, j2, ny in zip(Iy[:-1], Iy[1:], Ny):
        for i1, i2, nx in zip(Ix[:-1], Ix[1:], Nx):
            xd, yd = next(nxdyd)
            xv, yv = subgrid(xd, yd, nx, ny)
            X[j1:j2 + 1, i1:i2 + 1] = xv
            Y[j1:j2 + 1, i1:i2 + 1] = yv
            
    Xc, Yc = ctr(X), ctr(Y)
    Ic = np.arange(nrow * ncol, dtype=int).reshape((nrow, ncol))
    Iv = np.arange((nrow + 1) * (ncol + 1), dtype=int).reshape((nrow + 1, ncol + 1))
    Icorners = np.vstack((rv(Iv[:-1, :-1]), rv(Iv[:-1, 1:]), rv(Iv[1:, 1:]), rv(Iv[1:, :-1]))).T
    
    vertices = [[iv, xc, yc] for iv, xc, yc in zip(rv(Iv), rv(X), rv(Y))]
    cell2D = [[ic, xc, yc, 4, corners] for ic, xc, yc, corners in
                                zip(rv(Ic), rv(Xc), rv(Yc), Icorners)]
    
    vertices_ra = np.zeros(len(rv(Iv)), dtype=[('xv', '<f8'), ('yv', '<f8')])
    vertices_ra['xv'] = rv(X)
    vertices_ra['yv'] = rv(Y)
    cell2D_ra = np.zeros(len(rv(Ic)), dtype=[
        ('xc', '<f8'), ('yc', '<f8'), ('nvert', '<i4'), ('verts', 'i4', 4)])
    cell2D_ra['xc'] = rv(Xc)
    cell2D_ra['yc'] = rv(Yc)
    cell2D_ra['nvert'] = 4
    cell2D_ra['verts'] = Icorners
    return vertices, cell2D, vertices_ra, cell2D_ra

# %% specify the number of cells per major block defined by Xp and Yp

ul = lambda Z: Z[:-1, :-1]
ur = lambda Z: Z[:-1, 1: ]
ll = lambda Z: Z[1:, :-1]
lr = lambda Z: Z[1:, 1:]

def get_areas(Xv, Yv, all=False):
    """Return the area of the cells.
    
    The cell area is compute by the cross product of the ll corner of the cell +
    the cross project of the upper left corner of the cell (divided by 2).
    
    Parameters
    ----------
    Xv: ndarray (ny + 1, nx + 1)
        x-coords of grid vertices.
    Yv: ndarray (ny +1, nx + 1)
        y-coords of grid vertices.
    all: bool
        if True All A, All and Aur are all returned. 
        if False, only A is returned. 
    
    Returns
    -------
    cell area: ndarray (ny + 1, nx + 1)
        area of the cells.
        If all is True, then also the All and Aur are retured, i.e. the area of the
        lower left and upper right triangles of the cells, We need that to compute
        cell centers.
    """
    Vll1 = np.vstack((rv(lr(Xv) - ll(Xv)),
                      rv(lr(Yv) - ll(Yv)),
                    )).T
    Vll2 = np.vstack((rv(ul(Xv) - ll(Xv)),
                      rv(ul(Yv) - ll(Yv)),
                    )).T
    All = 0.5 * np.array([np.cross(vll1, vll2)
                          for vll1, vll2 in zip(Vll1, Vll2)])
    
    Vur1 = np.vstack((rv(ul(Xv) - ur(Xv)),
                      rv(ul(Yv) - ur(Yv)),
                    )).T
    Vur2 = np.vstack((rv(lr(Xv) - ur(Xv)),
                      rv(lr(Yv) - ur(Yv)),
                    )).T

    Aur = 0.5 * np.array([np.cross(vur1, vur2)
                          for vur1, vur2 in zip(Vur1, Vur2)])
    
    if all:
        return All + Aur, All, Aur
    else:
        return All + Aur

# %%

def get_centers(Xv, Yv):
    """Return the coordinates of the cell centers.
    
    This is the area weighted sum of the center of the lower left triangle
    and of the upper right triangle of the cell,
    
    Parameters
    ----------
    Xv: ndarray (ny + 1, nx + 1)
        x-coords of grid vertices.
    Yv: ndarray (ny +1, nx + 1)
        y-coords of grid vertices.
    """
     
    Cll = np.vstack((rv(ll(Xv) + lr(Xv) + ul(Xv)) / 3.,
                      rv(ll(Yv) + lr(Yv) + ul(Yv)) / 3.))
    Cur = np.vstack((rv(ur(Xv) + ul(Xv) + lr(Xv)) / 3.,
                      rv(ur(Yv) + ul(Yv) + lr(Yv)) / 3.))
     
    A, All, Aur = get_areas(Xv, Yv, all=True)

    Xc = (Cll[0] * All + Cur[0] * Aur) / A
    Yc = (Cll[1] * All + Cur[1] * Aur) / A
    return Xc, Yc

def get_vertices_cell2D(Xv, Yv):
    """Return both the vertex list and the cell2D list required by mf6 vdis.
    
    Parameters
    ----------
    Xv: ndarray (nrow +1, ncol + 1)
        x coordinates of grid vertices.
    Yv: ndarray (nrow +1, ncol + 1)
        x coordinates of grid vertices.
    
    Returns
    -------
    vertices: list of list or tuples with each line is (iv, xv, yv)
        The vertices as a list of lists.
    cell2D: list of list or tuples with each line is (ic, xc, yc, nvert, vertices)
        The cells definition as required by mf6 vdis
    """
    
    # Vertex numbers as an array of Xv.shape
    Iv = np.arange(np.prod(Xv.shape), dtype=int).reshape(Xv.shape)
    
    # Vertex numbers of cell corners (clockwise)
    Ivertices = np.vstack((rv(ul(Iv)), rv(ur(Iv)), rv(lr(Iv)), rv(ll(Iv)))).T
    
    # Cell numbers (flattened)
    Icells = np.arange((Xv.shape[0] -1) * (Xv.shape[1] - 1), dtype=int)
    
    Xc, Yc = get_centers(Xv, Yv)
    
    nvert = 4
    cell2D = [(ic, xc, yc, nvert, verts)
                for ic, xc, yc, verts in zip(Icells, Xc, Yc, Ivertices)]
    vertices = [(iv, xv, yv) for iv, xv, yv in zip(rv(Iv), rv(Xv), rv(Yv))]
    return vertices, cell2D
    
    
#%% One quadrangle model
                
if __name__ == '__main__':
    # Generating verious grids using  the affine and spline methods
    title = "Affine grid within one quadrilateral"
    Xp = np.array([[0, 25],
                   [5, 30]])
    Yp = np.array([[35, 40],
                   [0, 15]])
    Nx, Ny =[5], [6]
    vertices, cell2D, vertices_ra, cell2D_ra = quadrangle_grid(Xp, Yp, Nx, Ny)
    show_grid(title, vertices_ra, cell2D_ra)


    # %% one-cell model grid
    title = "Grid consisging of a single quadrilateral"
    Xp = np.array([[0, 25],
                [5, 30]])
    Yp = np.array([[35, 40],
                [0, 15]])
    Nx, Ny =[1], [1]
    vertices, cell2D, vertices_ra, cell2D_ra = quadrangle_grid(Xp, Yp, Nx, Ny)

    show_grid(title, vertices_ra, cell2D_ra)
    
# %% Axially symmetric

    title = 'An axially symmetri grid'
    r, a = np.logspace(-1, 3, 31), np.pi / 18

    Xp = np.array([r * np.cos(-a / 2),
                r * np.cos(+a / 2)])
    Yp = np.array([r * np.sin(-a / 2),
                r * np.sin(+a / 2)])
    Nx = np.ones(len(r) -1, dtype=int)
    Ny = [1]
    vertices, cell2D, vertices_ra, cell2D_ra = quadrangle_grid(Xp, Yp, Nx, Ny)

    show_grid(title, vertices_ra, cell2D_ra)

# %% Axially symmetric

    title = 'Another axially symmetric grid'
    r = np.logspace(-1, 3, 31)[np.newaxis, :]
    a = np.linspace(0., np.pi, 19)[:, np.newaxis]
    Xp = np.cos(a) @ r
    Yp = np.sin(a) @ r

    Nx = np.ones(Xp.shape[-1] -1, dtype=int)
    Ny = np.ones(Yp.shape[ 0] -1, dtype=int)

    vertices, cell2D, vertices_ra, cell2D_ra = quadrangle_grid(Xp, Yp, Nx, Ny)

    ax = show_grid(title, vertices_ra, cell2D_ra)
    ax.set_aspect(1.0)

# %% Axially symmetric

    asrow = lambda x: x[np.newaxis, :]
    ascol = lambda x: x[:, np.newaxis]
    
    title =  'A more complicated grid'
    ncol = 11
    R = np.linspace(100, 400, ncol + 1)
    r = asrow(R)
    a = ascol(np.linspace(0., np.pi/2, ncol))
    Xp1 = np.cos(a) @ r
    Yp1 = np.sin(a) @ r

    R = np.linspace(100, 400, ncol + 1)[::-1]
    r = asrow(R)
    a = ascol(np.arange(1.5 * np.pi, np.pi/2, -0.05 * np.pi))
    Xp2 = np.cos(a) @ r + 0
    Yp2 = np.sin(a) @ r + 500

    R = 500 + np.linspace(100, 400, ncol + 1)[::-1]
    x = asrow(R)
    y = ascol(np.arange(0., 440., 40.))
    Xp3, Yp3 = y @ np.ones_like(x), np.ones_like(y) @ x

    Xp = np.vstack((Xp1, Xp2, Xp3))
    Yp = np.vstack((Yp1, Yp2, Yp3))
    Nx = np.ones(Xp.shape[-1] -1, dtype=int)
    Ny = np.ones(Yp.shape[ 0] -1, dtype=int)

    vertices, cell2D, vertices_ra, cell2D_ra = quadrangle_grid(Xp, Yp, Nx, Ny)

    ax = show_grid(title, vertices_ra, cell2D_ra)
    ax.set_aspect(1.0)

# %% Using supports to generate the grid 

    "A complete grid form basic quadrilaterals with specified number of cells"
    Xp = np.array([ [  0., 15., 30., 40., 65],
                    [ -5., 10., 25., 45., 70],
                    [ -3., 13., 27., 41., 65],
                    [-10., 11., 23., 37., 67]])
    Yp = np.array([ [50., 45., 55., 50., 52.],
                    [37., 35., 40., 36., 35],
                    [18., 22., 19., 22., 20],
                    [ 5., -3.,  1., 5.,  4]])
    Nx = np.array([5, 3, 12, 5])
    Ny = np.array([6, 4, 10])
    vertices, cell2D, vertices_ra, cell2D_ra = quadrangle_grid(Xp, Yp, Nx, Ny)
    
    show_grid(title, vertices_ra, cell2D_ra)


    title = 'A grid  based on splines through the overal course netwerk of Quadrilaterals'
    ax = newfig(title, "x", "y")
    n = 50
    ax.plot(Xp, Yp, 'bo')
    for xp, yp in zip(Xp, Yp):
        ctr = np.vstack((xp, yp)).T
        x, y = interp_spline(ctr, n)
        ax.plot(x, y, 'b')
    for xp, yp in zip(Xp.T, Yp.T):
        ctr = np.vstack((xp, yp)).T
        x, y = interp_spline(ctr, n)
        ax.plot(x, y, 'r')
    plt.show()
    
# %%  # Computing the spline grid ourselves, no function is used

    ax = newfig("Spline grid", "x", "y")
    ax.plot(Xp, Yp, 'bo')
    n = 25

    # First file Xh and Yh with n point
    Xh = np.zeros((Xp.shape[0], n + 1))
    Yh = np.zeros((Yp.shape[0], n + 1))

    ax.plot(Xp, Yp, 'bo')
    for j, (xp, yp) in enumerate(zip(Xp, Yp)):
        ctr = np.vstack((xp, yp)).T
        x, y = interp_spline(ctr, n)
        ax.plot(x, y, 'b.-')
        Xh[j] = x 
        Yh[j] = y

    # Final grid points
    X = np.zeros((n + 1, n + 1))
    Y = np.zeros((n + 1, n + 1))

    for i, (xp, yp) in enumerate(zip(Xh.T, Yh.T)):
        ctr = np.vstack((xp, yp)).T
        x, y = interp_spline(ctr, n)
        ax.plot(x, y, 'r.-')
        X[:, i] = x
        Y[:, i] = y
        
    for x, y in zip(X, Y):
        ax.plot(x, y, 'b')
    for x, y in zip(X.T, Y.T):
        ax.plot(x, y, 'r')
        
    plt.show()
        
# %% Using the get_spline_grid function and computing the cell area and center
    ax = newfig("", "x", "y")

    nx = [10, 20, 5, 10]
    ny = [5, 15, 20] 

    Xv, Yv = get_spline_grid(Xp, Yp, nx, ny)

    A = get_areas(Xv, Yv, all=False)
    Xc, Yc = get_centers(Xv, Yv)
                            
    Atotal = np.sum(A)
    ax.set_title(f"Spline grid with computed cell centers. Total area = {Atotal:.0f} m2")

    # The course-grid points
    ax.plot(Xp, Yp, 'bx')

    # The vertices
    for x, y in zip(Xv, Yv):
        ax.plot(x, y, 'b') # 'Horizontal' lines (along rows)
    for x, y in zip(Xv.T, Yv.T):
        ax.plot(x, y, 'r') # 'Vertical lines (along columns)
    ax.plot(Xc, Yc, 'g.') # Plot centers

    # For mf6 VDIS        
    vertices, cell2D = get_vertices_cell2D(Xv, Yv)

    # Find in which cells these pointw are
    # Generate a set of random points.
    points = np.vstack(((np.random.rand(10) - 0.5) * 60. + 30.0,
                (np.random.rand(10) - 0.5) * 50. + 25.0)).T
    # Get the cells in which each point lies or -1 if outside
    Ic = points_in_cells(Xv, Yv, points)

    # Plot the point and use ic as label
    for point, ic in zip(points, Ic):
        ax.plot(point[0], point[1], 'yo', label="{}".format(ic))
        print(point, ic) # Print the point
        
    ax.legend()
    plt.show()

    # For getting the uv from xy within cells use homogeneous coordinates

    # Get the uv for the points that are in cells ic
    uv = np.zeros_like(points, dtype=float)
    for i, (ic, point) in enumerate(zip(Ic, points)):
        verts = cell2D[ic][-1]
        xq = Xv.ravel()[verts]
        yq = Yv.ravel()[verts]
        uv[i] = Q2US(xq, yq, point)
        print("Ic = {}, (x,y) =({:8.2f},{:8.2f}), (u,v) =({:5.2f},{:5.2f})".format(ic, *point, *uv[i]))


# %% Trying out the affine arrays

    dxy = np.array([5., 5., 1.])
    showvec(dxy, 'r-o', 'original vector')

    alf = 45
    dxy1 = Mrotate(alf, degrees=True) @ dxy
    showvec(dxy1, 'g-o', 'rotated {}'.format(alf))

    Lx, Ly = .5, 2
    dxy2 = Mstretch(0.5, 2) @ dxy
    showvec(dxy2, 'm-o', 'stretched {} {}'.format(Lx, Ly))

    Sx, Sy = 1.0, 2.0
    dxy3 = Mshear(Sx, Sy) @ dxy
    showvec(dxy3, 'c-o', 'sheared {} {}'.format(Sx, Sy))

    dx, dy = 3.0, 1.0
    dxy4 = Mmove(dx, dy) @ dxy
    showvec(dxy4, 'k-o', 'moved {} {}'.format(dx, dy))

    plt.gca().set_aspect(1.0)
    plt.grid('True')
    plt.legend()
    plt.show()

    points = np.array([[3., 6.0, 7.0, 2.0, 3.0], [1.0, 1.5, 5.5, 4.7, 1.0]]).T
    r, a = 1.0, np.linspace(0, 2 * np.pi, 25)
    points = np.vstack(((2.0 + r * np.cos(a), 2.0 + r * np.sin(a)))).T
    #points = np.array([[3., 6.0, 6.0, 3.0, 3.0], [3.0, 3.0, 6.0, 6.0, 3.0]]).T
    #points = np.array([[-2., 2.0, 2.0, -2.0, -2.0], [-2.0, -2.0, 2.0, 2.0, -2.0]]).T
    gobj =Grobj(points)

    reset_points = True

    title = "Some tranformations (rel. to origin). Reset_points = {}".format(reset_points)
    gobj.plot('r-.', label='original quadrilateral')

    gobj.rotate(alf, degrees=True); gobj.plot('g-.', label='rotate {}'.format(alf))

    if reset_points: gobj =Grobj(points)
    gobj.stretch(Lx, Ly); gobj.plot('m-.', label='stretch {} {}'.format(Lx, Ly))

    if reset_points: gobj =Grobj(points)
    gobj.shear(Sx, Sy); gobj.plot('c-.', label='shear {} {}'.format(Sx, Sy))

    if reset_points: gobj =Grobj(points)
    gobj.move(dx, dy); gobj.plot('k-.', label='move {} {}'.format(dx, dy))

    ax = plt.gca()
    ax.set_title(title)
    ax.plot(0.0, 0.0, 'ro', ms=8, label='origin')
    ax.grid()
    ax.set_aspect(1.0)
    ax.legend()
    plt.show()


    # %%
    title = 'Chaining the movements.'
    gobj =Grobj(points)
    gobj = gobj.rotate(alf, degrees=True).stretch(Lx, Ly).shear(Sx, Sy).move(dx, dy)
    gobj.plot('k', label='final')
    ax = plt.gca()
    ax.set_title(title)
    plt.grid()
    plt.show()
# %%
