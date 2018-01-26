#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 00:22:37 2017

Here is where we put any tools needed by the Bethune piping project

@author: Bob
"""
import matplotlib.pyplot as plt
import numpy as np
from inpoly import inpoly
#import shapefile as shp
from matplotlib.path import Path
import matplotlib.patches as patches
#import matplotlib as mpl

def plot_shapes(rdr, ax=None,
                facecolor='color',
                edgecolor='k',
                linewidth=0.1,
                alpha  = 1.0,
                title=None, xlabel="x [m]", ylabel="y [m]",
                xlim=None, ylim=None, **kwargs):
    '''plots shapes in shape file, given shapefile reader
    parameters
    ----------
    rdr : shapefil.Reader
    ax  : matplotlib.Axis
    title : str
    xlabel : str
    ylabel : str
    xlim : tuple
    ylim : tuple
    saveas : str
        figure type like png, jpg, tif
    dpe : int
        dots per inch when saved
    loc : str
        location for legend
    bba : anchor of legend bbox
    ncol : number of column in legend
    '''

    if ax is None:
        fig, ax = plt.subplots()

    if not title is None:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(rdr.bbox[0], rdr.bbox[2])
        ax.set_ylim(rdr.bbox[1], rdr.bbox[3])

    if not xlim is None:
        ax.set_xlim(xlim)

    if not ylim is None:
        ax.set_ylim(ylim)


    fldNames = [p[0] for p in rdr.fields][1:]

    if isinstance(facecolor, str):
        if facecolor in fldNames:
            icf = fldNames.index(facecolor)
        else:
            icf = 0
    elif isinstance(facecolor, dict):
        fc = facecolor
        icf = -1
    else:
        raise ValueError("facecolor must be a string or a dict")

    if isinstance(edgecolor, str):
        if edgecolor in fldNames:
            ice = fldNames.index(edgecolor)
        else:
            ice = 0
    elif isinstance(edgecolor, dict):
        ec = edgecolor
        ice = -1
    else:
        raise ValueError("Edgecolor must be a string or a dict")

    if isinstance(alpha, (int, float)):
        if alpha in fldNames:
            ica = fldNames.index(alpha)
        else:
            ica = 0
    elif isinstance(alpha, dict):
        af = alpha
        ica = -1
    else:
        raise ValueError("Alpha must be a string or a dict")

    id = fldNames.index('id')

    for sr in rdr.shapeRecords():
        if ice > 0:
            edgecolor = sr.record[ice]
        elif ice < 0:
            edgecolor = ec[sr.record[id]]
        if icf > 0:
            facecolor = sr.record[icf]
        elif icf < 0:
            facecolor = fc[sr.record[id]]
        if ica > 0:
            alpha = sr.record[ica]
        elif ica < 0:
            alpha = af[sr.record[id]]

        pth = Path(sr.shape.points)
        P = patches.PathPatch(pth,
                              facecolor=facecolor,
                              edgecolor=edgecolor,
                              alpha=alpha,
                              linewidth=linewidth, **kwargs)
        ax.add_patch(P)

    return ax


def set_array(rdr, var, gr, out=None, row=None, dtype=float):
    '''returs array of size gr.shape with value if value in shape
    row depends whether a cross section or a layer is filled.

    parameters
    ----------
    rdr : shapefile.Reader
    var : str
    name of variable to pick from record to fill this array
    gr : bethune_tools.GRID
    out : None, 2D numpy.ndArray of size [gr.nlay, gr.ncol]
    row : None, float
        row for which cross section should be filled
        out must be [gr.Nz, gr.Nx]
        if None, then layer is filed, size of out [gr.Ny, gr.Nx]
    dtype: dtype
        of numpy.ndarray to be returnd (default = float)
    '''

    fldNames = [p[0] for p in rdr.fields][1:]  # first item of each field.

    idx = fldNames.index(var)

    if out is None:
        if row is None:
            out = np.zeros((gr.ny, gr.nx))
        else:
            out = np.zeros((gr.nz, gr.nx))

    assert len(out.shape) == 2,\
        "ndim of array = {} but must be 2".format(out.ndim)

    if row is None:
        assert np.all(gr.shape[1:] == out.shape),\
            "shape of out ({},{}) does not match (gr.ny={}, gr.nx={})"\
            .format(*out.shape[:1], gr.ny, gr.nx)
    else:
        assert np.all(out.shape == gr.shape[0::2]),\
            "shape of out ({},{}) does not match (gr.nz={}, gr.nx={})"\
            .format(*out.shape[:1], gr.nz, gr.nx)

    for sr in rdr.shapeRecords():
        val = sr.record[idx]
        pgon = np.array(sr.shape.points)
        if row is None:
            I = inpoly(gr.Ym, gr.Xm, pgon)
        else:
            I = inpoly(gr.XM[:, row, :], gr.ZM[:, row, :], pgon)
        out[I] = val

    return out


