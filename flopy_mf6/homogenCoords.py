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


import scipy.linalg as la

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

def affine_subgrid(xd, yd, nx, ny):
    """Return coordinates of subgrid in quandrangle given by the 4 points xd, yd.
    
    Parameters
    ----------
    xd, yd: ndarray (4)
        coordinates of the 4 corners of the quandrangle outlining this subgrid.
    nx, ny: 2 ints
        number of cells (not vertices) within subgrid in x and y direction.
        
    >>> xq, yq = [-3.  7.  9.  1.] , [-5. -7.  3.  6.]
    >>> Xv, Yv = affine_subgrid(xq, yq, nx=10, ny=10)
    >>> XY = np.vstack(((Xv.ravel(), Yv.ravel()))).T
    >>> plt.plot(np.append(xq, xq[0]), np.append(yq, yq[0]), '-',
            Xv.ravel(), Yv.ravel(), 'b.', label='Points in quadrilateral.')
    >>> plt.show()
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


def US2Q_Q2US_arrays(x, y):
    """Return arrays to transform from unit square to quadrilateral given by 4 corners x, y and  vice versa.
    
    Credits:
    Paul Heckbert (1999) Projective Mappings for Image Warping
    Excerpted from pages 17-21 of Fundamentals of Texture Mapping and Image Warping, Paul Heckbert, Master’s thesis, UCB/CSD 89/516, CS Division, U.C. Berkeley, June 1989, http://www.cs.cmu.edu/ ̃ph, with corrections and a change to column vector notation by Paul Heckbert.
    
    The unit square as coordinates [[0,0], [1,0], [1,1], [0,1]] the quadrilateral is arbitrary but counter clockwise.
    
    Parameters
    ----------
    x, y npdarray of (4,)
        coordinates of quaddrilateral.
        
        
    >>> u = np.array([0, 1, 1, 0])
    >>> v = np.array([0, 0, 1, 1])
    >>> x = np.array([-3, 7, 9, 1])
    >>> y = np.array([-5, -7, 3, 6])
    
    >>> # US2Q = matrix to transform from Unit square to Qaudrilateral
    >>> # Q2US = matrix to transform from quaerilateral to unit square
    >>> US2Q, Q2US = unitsquare2quadrilateral(x, y)

    >>> # Case 1 unit square to quadrilateral
    >>> print("Case 1: unit square to quadrilateral")
    >>> xx, yy = np.zeros(4), np.zeros(4)
    >>> for i, (u_, v_) in enumerate(zip(u, v)):
    >>>     x_, y_, w_ = US2Q @ np.array([u_, v_, 1.])
    >>>     xx[i] = x_ / w_
    >>>     yy[i] = y_ / w_
    >>> print(xx, '\n', yy)

    >>> print("Case 2: quadrilatreal to unit square")
    >>> uu, vv = np.zeros(4), np.zeros(4)
    >>> for i, (x_, y_) in enumerate(zip(x, y)):
    >>>     u_, v_, q_ = Q2US @ np.array([x_, y_, 1.])
    >>>     uu[i] = np.round(u_/q_, 5)
    >>>     vv[i] = np.round(v_/q_, 5)
    >>> print(uu, '\n', vv)

    """
    sx = x[0] - x[1] + x[2] - x[3]
    sy = y[0] - y[1] + y[2] - y[3]
    dx1, dx2 = x[1] - x[2], x[3] - x[2]
    dy1, dy2 = y[1] - y[2], y[3] - y[2]
    
    # Matrix coefficients:
    g = la.det([[ sx, dx2], [ sy, dy2]]) /  \
        la.det([[dx1, dx2], [dy1, dy2]])
    h = la.det([[dx1, sx ], [dy1,  sy]]) /  \
        la.det([[dx1, dx2], [dy1, dy2]])
        
    a = x[1] - x[0] + g * x[1] 
    b = x[3] - x[0] + h * x[3]
    c = x[0]
    d = y[1] - y[0] + g * y[1]
    e = y[3] - y[0] + h * y[3]
    f = y[0]
    i = 1
    # Transform matrix from unit square to quadrilateral:
    Mus2q    = np.array([[a, b, c],[d, e, f], [g, h, i]]) 
    
    # Transform matrix from quadrilateral to unit square ( = Adjoint Matrix)
    # (can be used to back-transform, like inverse, but is always stable).
    Mq2us = np.array([[e * i - f * h, c * h - b * i, b * f - c * e],
                    [f * g - d * i, a * i - c * g, c * d - a * f],
                    [d * h - e * g, b * g - a * h, a * e - b * d]])
    
    # True inverse, may be singular, we don't need it, we just use the adjoint matrix
    #Mm1 = Madj/ la.det(M) # Inverse matrix
    
    return Mus2q, Mq2us #, Mm1

def Q2US(xq, yq, points):
    """Return point wihin quadrilateral in uv coordinates of unit square.
    
    Parameters
    ----------
    xq, yq: ndarray (4,)
        coordinates of 4 corner points of quadrilaterial that is to be converted to unit square
    points: ndarray (n, 2)
        points within the quadrilateral that are to be converted to points within the unit square.
    
    Returns
    -------
    uv: ndarray (n, 2)
        coordinates within the unit square.
        
    >>> # A mesh in Q to unit square
    >>> Xv, Yv = affine_subgrid(xq, yq, nx=10, ny=10)
    >>> XY = np.vstack(((Xv.ravel(), Yv.ravel()))).T
    >>> plt.plot(np.append(xq, xq[0]), np.append(yq, yq[0]), '-',
    ...         Xv.ravel(), Yv.ravel(), 'b.', label='Points in quadrilateral.')
    >>> plt.show()
    >>> uv0 = Q2US(xq, yq, np.vstack(((xq, yq))).T)
    >>> u0, v0 = uv0[:, 0], uv0[:, 1]
    >>> plt.plot(np.append(u0, u0[0]), np.append(v0, v0[0]), 'r-')
    >>> uv = Q2US(xq, yq, XY)
    >>> plt.plot(*uv.T, 'r.', label='Points in unit square.')
    >>> plt.show()

    """
    _, Mq2us = US2Q_Q2US_arrays(xq, yq)
    
    points = np.array(points, dtype=float)
    if points.ndim == 1: points = points[np.newaxis, :]
    
    uv = np.zeros_like(points, dtype=float)
    for i, (xp, yp) in enumerate(points):
        u, v, q =  Mq2us @ np.array([xp, yp, 1.])
        uv[i, :] = u / q - 0.5, v / q - 0.5 # subtract 0.5, 0.5 to zero-centered unit square
    return uv

def US2Q(xq, yq, points):
    """Return points in uv coordinates to that within the quadrilateral.
    
    Parameters
    ----------
    xq, yq: ndarray (4,)
        coordinates of the 4 corners of the quadrilateral (contour clockwise)
    points: ndarray (n, 2)
        uv coordinates of points to be converted from unit square to quarilateral
        The unit square is (-0.5, -0.5) to (0.5, 0.5)
        
    Returns
    -------
    xy: ndarray (n, 2)
        coordinates within quadrilateral.
        
    >>> # A complete spriral from unit square to Q
    >>> xq, yq = [-3.  7.  9.  1.] , [-5. -7.  3.  6.]
    >>> L = np.linspace(0, 5, 251)
    >>> u = L / 10 * np.cos((2 * np.pi) * L)
    >>> v = L / 10 * np.sin((2 * np.pi) * L)
    >>> uv = np.vstack(((u, v))).T
    >>> plt.plot([-0.5, 0.5, 0.5, -0.5, -0.5], [-0.5, -0.5, 0.5, 0.5, -0.5], 'r-', label='Unit square')
    >>> plt.plot(*uv.T,  'r.', label='Points to be transformed to quadrilateral.')
    >>> plt.show()
    >>> xy = US2Q(xq, yq, uv)
    >>> plt.plot(np.append(xq, xq[0]), np.append(yq, yq[0]), label='Quadrilateral')
    >>> plt.plot(*xy.T, 'bp', label='Points transfromed from unit square')
    >>> plt.show()
    """
    Mus2q, _ = US2Q_Q2US_arrays(xq, yq)
    
    points = np.array(points, dtype=float)
    if points.ndim == 1: points = points[np.newaxis, :]
    
    xy = np.zeros_like(points, dtype=float)
    for i, (up, vp) in enumerate(points):
        x, y, m =  Mus2q @ np.array([up + 0.5, vp + 0.5, 1.]) # add (0.5, 0.5) to zero-centered unit square.
        xy[i, :] = x / m, y / m
    return xy

if __name__ == '__main__':
    u = np.array([0, 1, 1, 0])
    v = np.array([0, 0, 1, 1])
    xq = np.array([-3, 7, 9, 1])
    yq = np.array([-5, -7, 3, 6])
        
    # US2Q = matrix to transform from Unit square to Qaudrilateral
    # Q2US = matrix to transform from quaerilateral to unit square
    Mus2q, Mq2us = US2Q_Q2US_arrays(xq, yq)

    # Case 1 unit square to quadrilateral
    print("Case 1: unit square to quadrilateral")
    xx, yy = np.zeros(4), np.zeros(4)
    for i, (u_, v_) in enumerate(zip(u, v)):
        x_, y_, w_ = Mus2q @ np.array([u_, v_, 1.])
        xx[i] = x_ / w_
        yy[i] = y_ / w_
    print(xx, '\n', yy)

    print("Case 2: quadrilatreal to unit square")
    uu, vv = np.zeros(4), np.zeros(4)
    for i, (x_, y_) in enumerate(zip(xq, yq)):
        u_, v_, q_ = Mq2us @ np.array([x_, y_, 1.])
        uu[i] = np.round(u_/q_, 5)
        vv[i] = np.round(v_/q_, 5)
    print(uu, '\n', vv)
    
   
    # Apply from US2Q. Just a few points from unit square to Q
    ax = newfig("Few points in unit square", "u", "v")
    uv = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5],
          [-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]])
    ax.plot([-0.5, 0.5, 0.5, -0.5, -0.5], [-0.5, -0.5, 0.5, 0.5, -0.5], 'r-', label='Unit square')
    ax.plot(*uv.T, 'r.', label='points to be transformed to quadrilateral.')
    ax = newfig("Transformed pointsl", "x", "y")
    ax.plot(np.append(xq, xq[0]), np.append(yq, yq[0]), label='quadrilateral')    
    xy = US2Q(xq, yq, uv)
    ax.plot(*xy.T, 'rp', label='transformed points')
    
    # A complete spriral from unit square to Q
    L = np.linspace(0, 5, 251)
    u = L / 10 * np.cos((2 * np.pi) * L)
    v = L / 10 * np.sin((2 * np.pi) * L)
    uv = np.vstack(((u, v))).T
    ax = newfig("Sprial in unit square to quadrilateral", "u", "u")
    ax.plot([-0.5, 0.5, 0.5, -0.5, -0.5], [-0.5, -0.5, 0.5, 0.5, -0.5], 'r-', label='Unit square')
    ax.plot(*uv.T,  'r.', label='Points to be transformed to quadrilateral.')
    ax.legend()
    ax = newfig("Sprial points transformed from unit square", "x", "y")
    xy = US2Q(xq, yq, uv)
    ax.plot(np.append(xq, xq[0]), np.append(yq, yq[0]), label='Quadrilateral')
    ax.plot(*xy.T, 'bp', label='Points transfromed from unit square')
    ax.legend()

    # A mesh in Q to unit square
    ax = newfig("Points in quadrilateral", "x", "y")    
    Xv, Yv = affine_subgrid(xq, yq, nx=10, ny=10)
    XY = np.vstack(((Xv.ravel(), Yv.ravel()))).T
    ax.plot(np.append(xq, xq[0]), np.append(yq, yq[0]), '-',
            Xv.ravel(), Yv.ravel(), 'b.', label='Points in quadrilateral.')
    ax.legend()
    ax = newfig("From points in quadrilateral to unit square", "u", "v")  
    uv0 = Q2US(xq, yq, np.vstack(((xq, yq))).T)
    u0, v0 = uv0[:, 0], uv0[:, 1]
    ax.plot(np.append(u0, u0[0]), np.append(v0, v0[0]), 'r-')
    uv = Q2US(xq, yq, XY)
    ax.plot(*uv.T, 'r.', label='Points in unit square.')
    ax.legend()
    

# %%
