#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 22:45:09 2017

@author: Theo
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as Polygon


def inpoly(x, y, pgcoords):
    """Returns bool array of shame shape as x telling which grid points are inside polygon
    parameters:
    -----------
        x, y: ndarrays of the same shape or arraylike
              shape of x and y must be the same.
        pgcoords: sequence of coordinate tuples or array of vertices of the
                  shape. (like it comes from a shape file.)
    return:
    -------
    boolean array, True if point x.flatten()[i], y.flatten()[i] inside shape.
        boolean array has the same shape as x
    @TO 20170315
    """
    try:
        isinstance(pgcoords,(list, tuple, np.ndarray))
        len(pgcoords[0])==2
        pgon = Polygon(pgcoords)
    except:
        print('pgcoords must be like [(0, 0), (1, 0), ..] or\n'
            +'an np.ndarray of shape [Np, 2]')
        raise TypeError("Can't create polygon, pgcoords error")

    try:
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        x.shape == y.shape
    except:
        raise TypeError("x and y not np.ndarrays with same shape.")

    if len(x.shape)==1 and len(y.shape)==2:
        X, Y = np.meshgrid(x, y)
    else:
        X = x
        Y = y
    xy = np.vstack((X.ravel(), Y.ravel())).T
    return pgon.contains_points(xy).reshape(X.shape)


if __name__ == '__main__':

    NOT = np.logical_not

    # example use of inpoly
    x    = np.linspace(-10, 10, 21)
    y    = np.linspace(-10, 10, 21)
    X, Y = np.meshgrid(x, y)                       # grid points
    xp   = [(-3, 12), (5, -12), (14, 8), (-3, 12)] # polygon
    xy   = np.array(xp)          # needed for direct plotting

    L = inpoly(X, Y, xp)         # L is boolean array, has shape of X and Y

    fig, ax = plt.subplots()     # show it

    ax.plot(X[NOT(L)], Y[NOT(L)], 'b.', label='outside polygon')
    ax.plot(X[L],      Y[L],      'ro', label='inside  polygon')

    ax.plot(xy[:, 0], xy[:, 1], '-k') # plot the polygon

    ax.set_title('inpoly test and demo')
    ax.legend()