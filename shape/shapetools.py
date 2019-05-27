#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 12:01:56 2017

Tools for reading and writing shapefiles

using the shapefile module

@author: Theo
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.path import Path as Polygon
from collections import OrderedDict
import pandas as pd
import shapefile
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)-15s -  %(levelname)s - %(message)s')
logger = logging.getLogger('__name__')

OR  = np.logical_or
AND = np.logical_and
NOT = np.logical_not

dbasefield = {"<class 'int'>": ('N', 16), "<class 'str'>": ('C', 15),
              "<class 'float'>": ('F', 15, 3),
              'datetime64[ns]': ('C', 32), "<class 'bool'>": ('L', 1),
              "<class 'timestamp'>": ('T', 8),
              'int32':('I', 10), 'int64':('I', 16), 'long' : ('I', 16),
              'float32': ('F', 15, 5), 'float64': ('O', 30, 16),
              'object': ('C', 20)}

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


def fldnames(path):
    ''''Return fieldnames in shapefile.

    parameters
    ----------
    path : str
        full path to shapefile

    TO 180521
    '''

    try:
        rdr = shapefile.Reader(path)
    except FileNotFoundError:
        raise "Can't find <{}>".format(path)
    return [f[0] for f in rdr.fields[1:]]


def shapes2dict(path, key=None):
    '''Return contents of shapefile as dict.

    parameters
    ---------
    path : str
        full path to shapefile
    key : str or None
        fld name that will be used as key for dict.
        Note that key must be hasheble to function as dict key.
    returns
    -------
    dict with one of the keys = 'points' which contains the shape
        coordinates of the record.

    TO 180521
    '''

    try:
        rdr = shapefile.Reader(path)
    except FileNotFoundError:
        raise "Can't find <{}>".format(path)

    fld = [f[0] for f in rdr.fields[1:]]

    if key is None:
        idx = 0
    else:
        try:
            idx = fld.index(key)
        except:
            raise LookupError("key <{}>  not in fields of shape".format(key))


    shpdict={} # generate a dict with the borehole properties
    for shp, rec in zip(rdr.shapes(), rdr.records()):
        key = rec[idx]
        try:
            shpdict[key] = {k:v for k, v in zip(fld, rec) if v != key}
        except:
            raise "key <{}> must be hasheble".format(key)
        shpdict[key].update({'points': np.array(shp.points)})

    return shpdict



def dict2shp(mydict, fileName, shapetype=None, xy=('x', 'y'),
             usecols=None, verbose=False):
    '''Save dict to shapefile.

    mydict may be a pd.DataFrame or a dictionary
    in both cases the key will be the name of the individual shapes that
    reside in the shapefile that is generated.

    In case mydict is a pd.DataFrame, each record represents a separate shape with
    the column headers equal the fields of the shape records.
    But with a pd.DataFrame x, and y can only be a single number, so this is
    suitable only for POINT shape, not for POLYLINES or POLYGONS because these
    have more x, and y values each.

    parameters
    ----------
        mydict : a dict of dicts or a list of dicts like
            {'paris': {'pop': 10, 'area': 120, 'x': [0.1, 0.3, ...]], 'y': [0.4, -0.2, ...]},
            'london': {'pop': 8.2, 'area': 150, 'x' ...},
            }
            The keys in mydict will be the names the individual shapes.
            THe keys in the subdict will be field names in the records
            and the x an dy fields as indicated in the xy keyword argument their coordinates.
            These coordinates must by arrays'

            Note: the key sof mydict will be use as extra field 'name' in the shape records.

        fileName: str
            Name of shapefile without extension
        shapetype: any legal type name e.g. 'POINT', 'POINTZ', 'POLYLINE', ...
            Type of shapefile.
            If None, then 'POINT' is used when len(xy)==2 and 'POINTZ' if
            len(xy)==3.
        xy: tuple of 2 or 3 str e.g. xy=('x', 'y' [, 'z'])
            dict keys denoting x, y (and z) coordinates.
        usecols: list of str
            list of str naming the keys to be used in the shapefile
    returns
    -------
        None, shapefile is saved to fname
    '''

    #is mydict a lst of dicts or a dict of dicts?
    assert isinstance(mydict, (dict, list, tuple, pd.DataFrame))

    if  isinstance(mydict[list(mydict.keys())[0]], (tuple, list)):
        # then turn it into a dict of dict with key =__noname__
        old_dict = mydict
        mydict={}
        mydict['noname'] = old_dict

    if isinstance(mydict, dict):
        shape_names = list(mydict.keys())
    elif isinstance(mydict, pd.DataFrame):
        shape_names = list(mydict.keys)
    else:
        raise TypeError(
            'mydict of illegal type {}, must be list, dict or pd.DataFrame'
            .format(type(mydict)))

    if usecols is None: usecols = []

    if shapetype is None:
        if len(xy) == 2:
            shapetype = 'POINT'
        elif len(xy) == 3:
            shapetype = 'POINTZ'
        else:
            raise ValueError(
                "xy must containt 2 or 3 fieldnames containing x, y (and z) shape coordinates")

    shapetype = shapetype.upper()

    legal_types = [k for k in shapefile.__dict__ if k[:2] in ['PO', 'MU']]

    assert shapetype in legal_types, '''
        Shapetype must be `POINT, POINTZ, 'POLYLINE, MULTIPOINT etc acceptable by shapes.
        '''

    wr = shapefile.Writer(eval('shapefile.' + shapetype))

    # generate shapefile fields
    wr.field('name', 'C', 20)   # firrst field will be the name of the shape record

    # add the other fields
    fields = [k for k in mydict[shape_names[0]].keys()]
    rec_keys = usecols if usecols else [k for k in fields if not k in xy]

    for k in rec_keys: # fields of sub dict
        try:
            wr.field(str(k), *dbasefield[str(type(k))])
        except:
            logger.debug("Can't handle field <{}>".format(str(k)))

    if verbose:
        print(', '.join(rec_keys))

    # generate records and points (or parts of polyline)
    for shape_name in shape_names:
        rec = mydict[shape_name]
        wr.record(shape_name, *[rec[k] for k in rec_keys]) # includes name

        if shapetype.startswith('POLY'):
            if shapetype.startswith('POLYG'): # close polyline if necessary
                if np.any(rec[xy][0] != rec[xy][-1]):
                    rec[xy] = np.vstack((rec[xy], rec[xy][:1]))
                wr.poly(parts=[[(x_, y_)
                    for x_, y_ in zip(rec[xy][0], rec[xy][1])]])
            else:
                wr.line(parts=[[(x_, y_)
                    for x_, y_ in zip(rec[xy[0]], rec[xy[1]])]])
        else:
            wr.point(*[rec[k] for k in xy])

    wr.save(fileName)
    if verbose:
        logger.debug('Dict saved to shapefile <{}>'.format(fileName))

    return

def frm2shp(myfrm, fileName, shapetype=None, xy=('x', 'y'),
            usecols=None, verbose=False):
    '''Save pd.Dataframe to shapefile.

    parameters
    ----------
        myfrm : a pandas.DataFrame
            dataframe containing columnns including for coordinates
        fileName: str
            Name of shapefile without extension
        shapetype: 'POINT' or 'POINTZ'
            type of shapefile.
            if None, then `POINT` is used when len xy==2 and `POINTZ` if
            len xy == 3
        xy: tuple of 2 or 3 str
            dict keys denoting x, y (and z) coordinates
        usecols: list of str
            list of str naming the keys to be used in the shapefile

    returns
    -------
        None, shapefile is saved
    '''
    if usecols is None:
        usecols = []

    if shapetype is None:
        if len(xy) == 2:
            shapetype = 'POINT'
        elif len(xy) == 3:
            shapetype = 'POINTZ'
        else:
            raise ValueError("xy must containt 2 or 3 strings")

    legal_types = [k for k in shapefile.__dict__ if k[:2] in ['PO', 'MU']]

    if not shapetype.upper() in legal_types:
            raise TypeError("shapetype must be 'POINT' or 'POINTZ'")

    wr = shapefile.Writer(eval('shapefile.' + shapetype.upper()))

    keys = []

    if verbose:
        print(', '.join([str(myfrm[k].dtype) for k in myfrm]))

    for k in myfrm.columns:
        if not k in xy:
            if not usecols or k in usecols:
                try:
                    wr.field(k, *dbasefield[str(myfrm[k].dtype)])
                    keys.append(k)
                except:
                    logger.debug("Can't handle column <{}>".format(k))


    if len(keys) == 0:
        print('Valid keys are:\n' + ', '.join([str(k) for k in dbasefield]))
        raise TypeError('No valid keys found in dataframe.')

    for i in range(len(myfrm)):
        rec = dict(myfrm.iloc[i][keys])
        wr.record(**rec)
        coords = tuple(myfrm.iloc[i][[k for k in xy]].values)
        wr.point( *coords)

    wr.save(fileName)

    if verbose:
        print('Dataframe saved to shapefile <{}>'.format(fileName))

    return


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
        rdr = shapefile.Reader(rdr)

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

if __name__ == '__main__':

    GIS       = '/Users/Theo/GRWMODELS/python/Juka_model/GIS/shapes/' # change
    shpnm = 'steilrandstukken.shp'
    shpnm = 'Steilrand.shp'
    shpnm = 'SteilrandGebieden.dbf'

    #demebores = shapes2dict(os.path.join(GIS, 'demebores.shp'), key='Hole_ID')

    #%% using dict2shp and frm2shp

    mydict = {'Hello': 'hello', 'length': 3,
              'width': 4.2, 'x' : 180320., 'y': 337231., 'z': 23.3}

    dict2shp(mydict, 'shapefromdict', xy=('x', 'y'), verbose=True)


    dataPath = '/Users/Theo/GRWMODELS/python/Juka_model/data'
    excelFile ='Boorgatgegevens.xlsx'
    #os.path.isfile(os.path.join(dataPath, excelFile))

    usecols = ['Hole ID', 'Easting', 'Northing', 'Elevation', 'Starting Depth',
       'Ending Depth', 'Boordatum',
       'Boormethode', 'Boordiameter', 'Afwerking']

    boreholes = pd.read_excel(os.path.join(dataPath, excelFile),
                  sheet_name='Collars')


    frm2shp(boreholes, 'shapefromframe', xy=('Easting', 'Northing', 'Elevation'),
            shapetype='POINTZ', usecols=[], verbose=True)

    import fdm
    x = np.logspace(-1, 3, 30)
    y = x[:]
    z = [0, -1]
    gr = fdm.Grid(x, y, z)
    dict2shp(gr.asdict(), 'grid', shapetype='POLYLINE', xy=('x', 'y'))




