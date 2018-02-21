#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 12:01:56 2017

Tools for reading and writing shapefiles

using the shapefile module

@author: Theo
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.path import Path as Polygon
from collections import OrderedDict
from shapefile import Reader
OR  = np.logical_or
AND = np.logical_and
NOT = np.logical_not

#%% Facilitate cooperation Bob - Theo

#%% directories
python  ='/Users/Theo/GRWMODELS/python'
modules = os.path.join(python, 'modules')

if not modules  in sys.path:
    sys.path.insert(1, "../bob_tools")



#%% Head module in main program

def ticks(xmin, xmax, dx):
    '''return suitable rounded ticks for plotting, for any axis
    parameters
    ----------
    xmin, xmax : floats
        xlim
    dx : float
        desired interval, should be a nice number
    '''
    xmin = np.floor(xmin /dx) * dx
    xmax = np.ceil(xmax /dx) * dx
    n = int((xmax - xmin) / dx)
    return np.linspace(xmin, xmax, n + 1)


def plotshapes(rdr, **kwargs):
    '''plots polygons in rdr as colored lines and filled patches
    kwargs
    ------
    title, xlabel, ylabel: strings
        plot labels
    xlim, ylim : tuple, list or ndarray of length 2 of floats
        plot extension
    xticks, yticks : float
        interval lenght of xticks and yticks
    facecolor : string
        fieldname in shape records holding facecolor names
    edgecolor: string
        fielname in shape records holding edgecolor names
        or a valid color name to be used for all edges
    alpha: string or float between 0 and 1
        fielname in shape records holding alpha (transparency values)
        or float between 0 and 1 for uniform alpha
    zorder: string of float
        fieldname in shape records holding zorder for shape
    grid: bool
        set grid lines
    '''
    if isinstance(rdr, str):
        rdr = Reader(rdr)

    grid = kwargs.pop('grid', None)

    fieldnames = [f[0] for f in rdr.fields[1:]]

    ax = kwargs.pop('ax', None)
    if ax is None:
        fig, ax = plt.subplots()

    kw1 = dict()
    for k in ['title', 'xlabel', 'ylabel', 'xlim', 'ylim', 'xticks', 'yticks']:
        if k in kwargs.keys():
            kw1[k] = kwargs[k]
            if k=='xticks':   # Use xticks=dx in call
                kw1[k] = ticks(*kwargs['xlim'], kwargs[k])
            if k=='yticks':   # Use yticks=dy in call
                kw1[k] = ticks(*kwargs['ylim'], kwargs[k])

    ax.set(**kw1)

    kw2 = dict()
    for k in ['facecolor', 'edgecolor', 'ec', 'fc', 'alpha', 'zorder', 'color']:
        if k in kwargs:
            try:
                kw2[k] = fieldnames.index(k)
            except:
                # no error checking here. Error may arise a plotting of shape !
                kw2[k] = kwargs[k]

    for i, (shp, rec) in enumerate(zip(rdr.shapes(), rdr.records())):
        pth = mpl.path.Path(shp.points)

        for k in kw2:
            try:
                kw2[k] = rec[kw2[k]]
            except:
                pass

        P = mpl.patches.PathPatch(pth, **kw2)
        ax.add_patch(P)

    if grid is not None:
        ax.grid(grid)

    ax.legend(loc='best')
    return ax


def set_array(rdr, var, gr, dtype=float, out=None):
    '''Fill array of size gr.shape with value if value in shape
    parameters
    ----------
    rdr : shapefile.Reader
    var : str
    name of variable to pick from record to fill this array
    gr : bethune_tools.GRID
    dtype: dtype of numpy.ndarray to be returnd (default = float)
    out: [None, ndarray]
        if None a new array is formed else out is returnd and filled in place
    '''

    fldNames = [p[0] for p in rdr.fields][1:]  # first item of each field.

    idx = fldNames.index(var)

    if out is None:
        out = np.zeros(gr.shape, dtype=dtype)

    for sr in rdr.shapeRecords():
        val = sr.record[idx]
        pgon = np.array(sr.shape.points)
        I = inpoly(gr.XM, gr.ZM, pgon)
        out[I] = val

    return out

def plot_array(A, gr, rdr, title='title', ax=None, cmap=None,
               edgecolors='none', alpha=None, xlab="X", ylab="Y", xlim=None, ylim=None, para="Parameter",  svas=None, dpi=200, lloc='best', bba=None, ncol=2, grid=None):
    '''plots cross section of array'''
    if ax is None:
        fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    if not xlim is None:
        ax.set_xlim(xlim)
    if not ylim is None:
        ax.set_ylim(ylim)
        A=A.copy()
    A[np.isnan(A)]=-999.99
    C = plt.pcolormesh(gr.x, gr.z, A[:, 0, :], axes=ax, cmap=cmap, edgecolors=edgecolors,
                   norm=None, vmin=0., alpha=alpha)
    plotshapes(rdr, ax=ax, ncol=2)
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())#, bbox_to_anchor = bba)#scatterpoints= SC, fontsize = FS)
    cbar = plt.colorbar(C)
    cbar.ax.set_ylabel(para)
    if not grid is None:
        ax.grid(True)
    if svas is None:
        ax.legend().draggable()
    if not svas is None:
        ax.legend(bbox_to_anchor=bba, loc=lloc, ncol=ncol)
        plt.savefig(filename=svas, dpi=dpi, bbox_inches='tight')
    return None


def inpoly(x, y, pgcoords):
    """Returns bool array [ny, nx] telling which grid points are inside polygon

    """
    pgcoords = np.array(pgcoords)
    assert pgcoords.shape[1]==2 and pgcoords.ndim==2,\
        "coordinates must be an arra of [n, 2]"
    pgon = Polygon(pgcoords)

    try:
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        shape = x.shape
        x.shape == y.shape
    except:
        raise TypeError("x and y are not np.ndarrays with same shape.")

    xy = np.vstack((x.ravel(), y.ravel())).T
    return pgon.contains_points(xy).reshape(shape)

def __main__():
    shpdir = '/Users/Theo/' + \
        'Instituten-Groepen-Overleggen/HYGEA/Consult/2017/' + \
        'DEME-julianakanaal/REGIS/Limburg - REGIS II v2.2/shapes'
    shpnm = 'steilrandstukken.shp'
    shpnm = 'Steilrand.shp'
    shpnm = 'SteilrandGebieden.dbf'

    shapefileName =os.path.join(shpdir, shpnm)

    rdr   = Reader(shapefileName)

    fldNms = [p[0] for p in rdr.fields][1:]
    print(fldNms)

    kwargs = {'title': os.path.basename(shapefileName),
              'grid': True,
              'xticks': 1000.,
              'yticks': 1000.,
              'xlabel': 'x RD [m]',
              'ylabel': 'y RD [m]',
              'edgecolor': 'y',
              'facecolor' : 'r',
              'alpha': 0.5}

    plotshapes(rdr, **kwargs)

if __name__ == '__main__':
    __main__()

