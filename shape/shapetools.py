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
from matplotlib.path import Path
from matplotlib.transforms import Bbox
import matplotlib.patches as patches
from collections import OrderedDict
import pandas as pd
import shapefile
import datetime
#import geopandas
import logging
import pdb

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

logging.disable(level=logging.DEBUG)

#%% Head module in main program

# matplotlib.path.Path codes
MOVETO, LINETO, CLOSE = 1, 2, 79

shapeNames = {  0: 'NULL',
                1: 'POINT',
                3: 'POLYLINE',
                5: 'POLYGON',
                8: 'MULTIPOINT',
                11: 'POINTZ',
                13: 'POLYLINEZ',
                15: 'POLYGONZ',
                18: 'MULTIPOINTZ',
                21: 'POINTM',
                23: 'POLYLINEM',
                25: 'POLYGONM',
                28: 'MULTIPOINTM',
                31: 'MULTIPATCH'}

shapeTypes = {value: key for key, value in shapeNames.items()}

#%% Shape_df

class Shape_df:
    """Object holding a dict of shapes.

    This is a major object for easy handling of shapes. It can be
    geneated by specifying paths and a pd.DataFrame or be read from a shapefile
    using               shpdf = Shape_df().get(shapefilename)
    The object has path.Path to keep lines and, therefore, may use all of
    path.Path methods. It further has the records as a pd.DataFrame and, so
    all pandas functionality is readily available. The object can be
    written to a shapefile.
    # TODO make sure that the parts are correctly written as well. As I understand
    the description, multiple paths must be written as a list if lists in which
    each list is a list of list with each time a vertec pair

    @TO Nov. 2020
    """

    def __init__(self, data=None, paths=None, xy=None,
                 shapeType=shapefile.POLYLINE):
        """Generate a Shape_df object using a list of paths, and a pd_dataframe.

        The input is a list of paths or lines (n, 2) coordiates and the data is
        the corresponding DataFrame with the properties of each shape.
        The order of the paths and the data matter.
        If paths is None, then xy must be a tuple of two headers that point at
        the x and y coordinates of the record. We then inteprete the data as
        a point shape.

        Parameters
        ----------
        data: pd.DataFrame
            the recrods, holding the data, record per shape and in the same order.
            The index of the DataFrame may be used to recognize each shape.
        paths: list of lines (n, 2) nd.arrays or path.Paths
            line coordinatres coordinates pertaining to each shape.
        xy: Non or a tuple of labels indicating the x and y coordinate.
            xy is only not None for point shapes that have the coordinates in the record it self.
            In that case paths must be None
        shapeType: int (rather use one of shapefile module constants)
            `ARCGIS` shapefile type. See attributes of `shapefile` module.

        Remarks
        -------
            If data is a pd.DataFrame with 'path' in its columns then this
            data is accepted as is and paths are assumed to be matplotlib.Path
            objects.
            If data is a pd.DataFrame without a column called 'path' then 'xy'
            is assumed to be a tuple of column names holding the x and y values
            of points, with one points per record.
        """
        self.shapeType = shapeType

        if data is None:
            if paths is None:
                self.shapeType = shapeType
                return
            else:
                raise ValueError("Data must be a pd.DataFrame is paths is None")
        else:
            assert isinstance(data, pd.DataFrame), "dadta must be a pd.DataFrame"
            self.data = data

            if paths is not None:
                if len(self.data) != len(paths):
                    raise ValueError("len(paths_list)= {} != len(data)= {}.".
                                 format(len(paths), len(self.data)))
                else:
                    self.data['path'] = self.check_paths(paths)
            else:
                if 'path' in data.columns:
                    self.data['path'] = self.check_paths(data['path'].values)
                else:
                    if not isinstance(xy, (tuple, list)):
                        raise ValueError("xy must be a tuple of column names indicating the x and y coordinate of the record.")
                    else:
                        XY = np.vstack((data[xy[0]].values, data[xy[1]].values)).T
                        self.data['path'] = [Path([pnt], codes=[MOVETO]) for pnt in XY]

        self.update_bbox()
        self.set_shape_type()
        return

    def __getitem__(self, key):
        """Return column of the internal DataFrame."""
        return self.data[key]


    def __setitem__(self, key, values):
        """Add column to underlying DataFrame."""
        self.data[key] = values


    def set_shape_type(self):
        """Return shapeType (ESRI shapefile types).

        One shapeType allowed per DataFrame colun 'path'. So only the first
        path is inspected to determine the shapeType of the whole DataFrame.

        The type is determined based on only the codes of the firs path.
        Recognized types are POINT, MULTIPOINT, POLYLINE and POLYCON.
        """
        p = self.data['path'].iloc[0]
        if len(p.codes) == 1:
            self.shapeType = shapefile.POINT
        elif np.all(p.codes) == MOVETO:
            self.shapeType = shapefile.MULTIPOINT
        elif p.codes[-1] == CLOSE:
            self.shapeType = shapefile.POLYGON
        else:
            self.shapeType = shapefile.POLYLINE
        return


    def update_bbox(self):
        """Return bbox for given points/vertices.

        Parameters
        ----------
        path: path.Path object
            the path for which the bbox object is desired.
        """
        self.bbox = Bbox.union([
            Bbox((np.min(p.vertices, axis=0),
                  np.max(p.vertices, axis=0)))
                      for p in self.data['path']])
        return


    def check_paths(self, paths):
        """Check the paths.

        To handle sets of polylines belonging to one shape, proper path codes
        must be used. I.e. 2 for normal line-to points and 1 for move-to ponts, i.e.
        start of a new line. Shapefiles use parts for that.

        """
        MOVETO = 1
        LINETO = 2
        if not np.all([isinstance(path, Path) for path in paths]):
            raise ValueError("The paths_list must be a list or a tuple or path_like coordinates.")
        for p in paths:
            if len(p) == 0:
                continue
            if p.codes is None:
                p.codes = np.ones(len(p.vertices), dtype=int) * MOVETO
                p.codes[1:] = LINETO
        if self.shapeType in [5, 15, 15]: # Check if line must be closed
            for p in paths:
                if len(p) == 0:
                    continue
                if not np.all(p.vertices[0] == p.vertices[-1]):
                    p.vertices = np.vstack((p.vertices, p.vertices[-1][np.newaxis, :]))
                    p.codes = np.hstack((p.codes, LINETO))
        return paths


    def append(self, shape_df=None):
        """Append shape_df to self.

        Parameters
        ----------
        shape_df: shapetools.Shape_df object
            shapes in a Shape_df object.
        """
        if not isinstance(shape_df, Shape_df):
            raise TypeError('shapes must be an objec of type {}'.format(type(self)))

        self.data = pd.merge(self.data, shape_df, how='outer')
        self.update_bbox()
        return


    def dbf_field_type(self, tp, strlen=30, digits=10, decimals=10):
        """Return dbf field header with type and leng and decimals."""
        if tp == bool:
            return ('L', 1, 0)
        elif tp in (int, np.int64):
            return ('N', digits, 0)
        elif tp in (float, np.float64):
            return ('N', digits, decimals )
        elif tp == str:
            return ('C', strlen, 0)
        elif tp in (np.datetime64, pd.Timestamp, datetime.datetime):
            return ('D', 8, 0)
        else:
            raise TypeError('Unknown dtype: {}. May be cast it to a known data type'.format(tp))


    def detect_shapeType(self):
        """Return shapeType and shapeType name, determined from path."""
        def shpcode(p, shape_codes):
            n = min(p.vertices.shape[-1], 4)
            return shape_codes[n - 2]

        shpTypes = []

        paths = self.data['path'].values
        for p in paths:
            if len(p.vertices) == 0:
                shpTypes.append(0)
            elif len(p.vertices) == 1:
                shpTypes.append(shpcode(p, (1, 11, 21)))
            else:
                if np.all(p.vertices[0] == p.vertices[-1]) or p.codes[-1] == CLOSE:
                    shpTypes.append(shpcode(p, (5, 15, 25)))
                else:
                    shpTypes.append(shpcode(p, (3, 13, 23)))

        self.shapeType = np.min(np.array(shpTypes, dtype=int))
        self.shapeTypeName = shapeNames[self.shapeType]
        return


    def save(self, fname=None, shapeType=None):
        """Save the object to a shapefile.

        Parameters
        ----------
        fname: str
            filename without extension
        """
        POLYKEYS = [k for k in shapeTypes.keys() if k.startswith('POLY')]
        MPKEYS   = [k for k in shapeTypes.keys() if k.startswith('MULTIPOINT')]
        PKEYS    = [k for k in shapeTypes.keys() if k.startswith('POINT')]

        self.detect_shapeType()

        wr = shapefile.Writer(
                fname, shapeType=self.shapeType, autobalance=True)

        # Only save the columns of types int, float, str
        # inspect the first rectord
        save_cols = [(col, type(v)) for col, v in zip(self.data.columns, self.data.iloc[0])
                             if isinstance(v, (np.int64, int, float, str))]

        for col, tp in save_cols:
             wr.field(col, *self.dbf_field_type(tp))

        data = self.data[[col_tp[0] for col_tp in save_cols]]

        for idx, p in zip(data.index, self.data['path'].values):
            if self.shapeTypeName in POLYKEYS:
                parts = np.hstack((np.where(p.codes == MOVETO)[0], len(p)))
                lines = []
                for j,k in zip(parts[:-1], parts[1:]):
                    line =[tuple(xy) for xy in p.vertices[j:k]]
                    lines.append(line)
                wr.line(lines)
            elif self.shapeTypeName in MPKEYS:
                wr.point(list(p.vertices))
            elif self.shapeTypeName in PKEYS:
                wr.point(*p.vertices[0])
            else:
                raise ValueError(f"shapeType {self.shapeType} not implemented, use oneof:\n" +
                                 str(shapeTypes))

            # TODO: capture date field perhaps it works whwen sending a datetime to wr
            wr.record(*[v for v in data.loc[idx]])

        print("Shapefile saved to <{}>".format(fname))
        wr.close()
        return


    def get(self, fname=None):
        """Get the object from a shapefile."""
        MOVETO = 1
        LINETO = 2
        PGONKEYS = [k for k in shapeTypes.keys() if k.startswith('POLYGON')]

        with shapefile.Reader(fname) as sf:
            self.shapeType = sf.shapeType
            columns = [f[0] for f in sf.fields[1:]]
            self.data = pd.DataFrame(
                [list(rec) for rec in sf.records()], columns=columns)
            paths = self.check_paths([Path(shp.points) for shp in sf.shapes()])
            # parts tells at which vertex a new fline starts
            # we translate that in the codes associated with the path vertices
            self.parts = [shp.parts for shp in sf.shapes()]
            for part, pth in zip(self.parts, paths):
                codes = LINETO * np.ones(len(pth), dtype=int)
                codes[part] = MOVETO
                if sf.shapeTypeName in PGONKEYS:
                    ends = [p - 1 for p in part + [len(part)]][1:]
                    codes[ends] = CLOSE
                pth.codes = codes
            self.data['path'] = paths
            self.update_bbox()
        return


    def get_id_array(self, X=None, Y=None):
        """Return array of the same shape as X and or Y telling the path in which each ponit lies.

        Values outside all paths will be indicated by -9999

        Parameters
        ----------
        X: nd.array or sequence (n) or (n, 2)
            X-coordinates or points (if shape is n, 2)
        Y: nd.array or sequene (n)
            Y-coordinatea, ignored if shape(X) is (n, 2)

        Returns
        -------
        Array with path id's obtained from stored data index.
        The array shape is the same as that of the input.
        Points outside every path are indicated by -999.
        """
        if Y is None:
            pnts = np.array(X)
            if not pnts.ndim == 2 and pnts.shape[1] == 2:
                raise ValueError("shape of points must be (n, 2)")
            shape = None
        else:
            pnts= (np.array(X), np.array(Y))
            if not pnts[0].shape == pnts[1].shape:
                raise ValueError("Shape of X ({},{}) != shape of Y ({}{})"
                                 .format(*pnts[0], *pnts[1]))
            shape = X.shape
            pnts = np.vstack((pnts[0].ravel(), pnts[1].ravel())).T

        Ids = np.ones(pnts.shape[0], dtype=int) * -999

        # Must reindex:
        self.data.index = np.arange(len(self.data), dtype=int)

        for i, p in zip(self.data.index, self.data['path'].values):
          Ids[p.contains_points(pnts)] = i

        return Ids if shape is None else Ids.reshape(shape)


    def fill_array(self, X=None, Y=None, parameter=None, A=None):
        """Fill the array with the parameter values.

        Parameters
        ----------
        parameter: str
            coluns (key) in self.data
        A: an array of the same shape as X and Y or None.
            A to be filled. If None then X.shape is used and A is initially
            filled with NaNs.
        X array of coordinates
            X-coordaintes (n) points (x,y) pairs.
                if Y is None, then X.shape must be (n,2)
        Y coordinates or None
            if Y is None then X.shape must be (n,2)
        """
        if not parameter in self.data.columns:
            raise ValueError("Parameter <{}> not in columns [{}].".format(
                                    parameter, ' '.join(self.data.columns)))

        line = (Y is None) and (X.ndim == 2) and (X.shape[1] == 2)

        if line:
            if A is None:
                A = np.zeros_like(X[:, 0])
            else:
                if not A.shape == X[:, 0].shape:
                    raise ValueError("A..shape != X[:,0].shape")
            pnts = X
        else:
            if not X.shape == Y.shape:
                raise ValueError("X.shape != Y.shape")
            if A is None:
                A = np.nan * np.zeros(X.shape)
            pnts = np.vstack((X.ravel(), Y.ravel())).T

        for value, path in zip(self.data[parameter].values, self.data['path'].values):
            A.ravel()[path.contains_points(pnts)] = value

        return A

    def get_extents(self, ax=None):
        """Return bbox encompassing all paths in Shape_df."""
        if ax is not None:
            bboxes = [Bbox([ax.get_xlim(), ax.get_ylim()])]
        else:
            bboxes = []

        for path in self.data['path'].values[1:]:
            bboxes.append(path.get_extents())
        return Bbox.union(bboxes)


    def plot(self, ax=None, color_key='color', xlim=None, ylim=None, rounder=None,
             label=None, offset=(0,0), rotation=0, **kw):
        """Plot the shapes, using a field for color or specify fc in kw.

        Parameters
        ----------
        ax: Axes obj
            axes to plot in
        color_key: str
            key in self.data to use for the facecolor of the patch.
        rounder: float or None
            if not None will round xlim and ylim if not given to a larger
            bbox of which the edges are rounded to multiples of rounder.
            Rounder only as effect if xim and or ylim are None
        label: str
            column label of which the string value to plot
            if None, no labels will be plotted.
        offset: 2 tuple of floats
            text offset in user coordinates
        rotation: float
            text rotation in degrees counter clockwise
        kw: addtional files
            options passed on to patches.PathPatch
        """
        #pdb.set_trace()
        options = ['ec', 'fc', 'color', 'alpha']
        keys = [k for k in options if k in self.data.columns]

        if self.shapeType == shapefile.POINT:
            if not 'marker' in kw:
                kw['marker'] = 'o'

        for i, idx in enumerate(self.data.index):
            rec = dict(self.data.iloc[i])

            for key in keys:
                if isinstance(rec[key], str):
                    kw[key] = rec[key]
                elif isinstance(rec[key], float):
                    if not np.isnan(rec[key]):
                        kw[key] = rec[key]
                else:
                    pass

        paths = self.data['path'].values

        if self.shapeType in [5, 15, 25]:
            ax.add_patch(patches.PathPatch(paths[i], **kw))
        if self.shapeType in [3, 13, 13]:
            ax.plot(*paths[i].vertices.T, **kw)
        if self.shapeType in [1, 11, 21, 28]:
            if not 'marker' in kw:
                kw['marker'] = 'o'
            ax.plot(*paths[i].vertices.T, **kw)

            if label:
                for i, p in enumerate(paths):
                    ax.text(*(p.vertices[0] + np.array(offset)), self.data[label].iloc[i], rotation=rotation)

        if xlim is None:
            if rounder:
                xlim = (rounder * np.floor(self.bbox.x0 / rounder),
                        rounder * np.ceil( self.bbox.x1 / rounder))
            else:
                xlim = (self.bbox.x0, self.bbox.x1)
            ax.set_xlim(xlim)
        if ylim is None:
            if rounder:
                ylim = (rounder * np.floor(self.bbox.y0 / rounder),
                        rounder * np.ceil( self.bbox.y1 / rounder))
            else:
                ylim = (self.bbox.y0, self.bbox.y1)
            ax.set_ylim(ylim)
        return ax

def update_fig_limits(bbox, ax=None):
    """Update xlim and ylim based on bbox and existing xlim and ylim.

    Parameters
    ----------
    bbox: transforms.Bbox object
        the bbox
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xlim = min(xlim[0], bbox.x0), max(xlim[1], bbox.x1)
    ylim = min(ylim[0], bbox.y0), max(ylim[1], bbox.y1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return

def ticks(xmin, xmax, dx):
    """Return suitable rounded ticks for plotting, for any axis.

    Parameters
    ----------
    xmin, xmax : floats
        xlim
    dx : float
        desired interval, should be a nice number
    """
    xmin = np.floor(xmin /dx) * dx
    xmax = np.ceil(xmax /dx) * dx
    n = int((xmax - xmin) / dx)
    return np.linspace(xmin, xmax, n + 1)


def fldnames(path):
    """Return fieldnames in shapefile.

    Parameters
    ----------
    path : str
        full path to shapefile

    TO 180521
    """
    try:
        rdr = shapefile.Reader(path)
    except FileNotFoundError:
        raise "Can't find <{}>".format(path)
    return [f[0] for f in rdr.fields[1:]]


def shapes2dict(path, key=None):
    """Return contents of shapefile as dict.

    Parameters
    ----------
    path : str
        full path to shapefile
    key : str or None
        fld name that will be used as key for dict.
        Note that key must be hasheble to function as dict key.

    Returns
    -------
    dict with one of the keys = 'points' which contains the shape
        coordinates of the record.

    TO 180521
    """
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
    """Save dict to shapefile.

    mydict may be a pd.DataFrame or a dictionary
    in both cases the key will be the name of the individual shapes that
    reside in the shapefile that is generated.

    In case mydict is a pd.DataFrame, each record represents a separate shape with
    the column headers equal the fields of the shape records.
    But with a pd.DataFrame x, and y can only be a single number, so this is
    suitable only for POINT shape, not for POLYLINES or POLYGONS because these
    have more x, and y values each.

    Parameters
    ----------
    mydict : a dict of dicts or a list of dicts like
        {'paris': {'pop': 10, 'area': 120, 'x': [0.1, 0.3, ...]], 'y': [0.4, -0.2, ...]},
        'london': {'pop': 8.2, 'area': 150, 'x' ...},
        }
        The keys in mydict will be the names the individual shapes.
        The keys in the subdict will be field names in the records
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

    Returns
    -------
        None, shapefile is saved to fname
    """
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
        shape_names = list(mydict.index)
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
    """Save pd.Dataframe to shapefile.

    Parameters
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

    Returns
    -------
        None, shapefile is saved
    """
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
    """Plot polygons in rdr as colored lines and filled patches.

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
    """
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
    """Fill array of size gr.shape with value if value in shape.

    Parameters
    ----------
    rdr : shapefile.Reader
    var : str
    name of variable to pick from record to fill this array
    gr : bethune_tools.GRID
    dtype: dtype of numpy.ndarray to be returnd (default = float)
    out: [None, ndarray]
        if None a new array is formed else out is returnd and filled in place
    """
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
    """Plot cross section of an array."""
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
    """Return bool array [ny, nx] telling which grid points are inside polygon."""
    pgcoords = np.array(pgcoords)
    assert pgcoords.shape[1]==2 and pgcoords.ndim==2,\
        "coordinates must be an arra of [n, 2]"
    pgon = Path(pgcoords)

    try:
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        shape = x.shape
        x.shape == y.shape
    except:
        raise TypeError("x and y are not np.ndarrays with same shape.")

    xy = np.vstack((x.ravel(), y.ravel())).T
    return pgon.contains_points(xy).reshape(shape)


def inpatch(patch=None, XY=None):
    """Return which points are within the path of the patch.

    Parameters
    ----------
    patch: matplotlib.patches.Patch object
        the patch whose path we need
    XY: (n, 2) np.ndarray
        the points to check whether they are inside the patch's path.
    """
    pth   = patch.get_path()
    transf = patch.get_transform()
    user_coordinates_path = transf.transform_path(pth)
    return user_coordinates_path.contains_points(XY)

#%% point2line

def point2line(point=None, line=None):
    """Return distance from point perpendicular to line.

    Only the first and last coordinates of ppth will be used

    Parameters
    ----------
    point: tuple of 2 or similar
        point from which the distance is to be computed.
    line: path.Path or an nd.array (n, 2)
        path representing the line (two vertices)

    Returns
    -------
    tuple (r, mu, xs, ys)
        r distance to line
        mu relative distance along line. point is wthin line if between 0 and 1
        xs, ys the intersection of the line perpendicular through point
        use my to pick points within the line end points.
    if point or line is None:
        run selftest
    """
    if np.any(point) and np.any(line):
        if isinstance(line, Path):
            x0, y0 = line.vertices[ 0]
            x1, y1 = line.vertices[-1]
        else:
            x0, y0 = line[ 0]
            x1, y1 = line[ 1]
        xp, yp = point
        Lp = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        ex, ey = (x1 - x0) / Lp, (y1 - y0) / Lp
        #M = np.array([[-ey, -ex],[ex, -ey]])
        Minv = np.array([[-ey, ex],[-ex, -ey]])
        rhs = np.array([[x0 - xp], [y0 - yp]])
        lam, mu = (Minv @ rhs).T[0]
        r = np.abs(lam)
        xs, ys = x0 + mu * ex, y0 + mu * ey
        print(r, mu / Lp, xs, ys)
        return r, mu / Lp, xs, ys
    else: # reunt the selftest
        line = np.array([[300, 0], [0, 500]])
        points = np.random.rand(40, 2) * 500
        res = []
        for p in points:
            res.append(point2line(p, line))
        res = np.array(res)
        fig, ax = plt.subplots()
        ax.plot(*line.T, 'k')
        ax.plot(*points.T, 'o')
        for p1, p2 in zip(points, res[:, -2:]):
            ax.plot(*np.array([p1, p2]).T)
        ax.set_aspect(1.)
        plt.show()


#%% point2polyline

def point2polyline(point, polyline):
    """Return shortest distance to polyline.

    The distance is either to the neartst vertex or perpendicular to
    the nearest line-segement, whichever distance is shorter.

    Parameters
    ----------
    point: 2 tuple
        xp, yp coordinate from which the distnace is to be computed.
    polyline: (n,2) array or a path.Path object

    Returns
    -------
    (r, mu, xp, yp,, xs, ys) with xs, ys the intersection

    """
    if isinstance(polyline, Path):
        polyline = polyline.vertices
    else:
        if not isinstance(polyline, np.ndarray) and not polyline.shape[1] == 2:
            raise ValueError("Polyline must be a path or an ndarray of shape (n, 2)")

    # Get this distance to every vertex of the polyline and from that
    # the minimum distance to any of the vertices of the polyling
    xp, yp = point
    x, y = polyline.T
    R = np.sqrt((x - xp) ** 2 + (y - yp) ** 2)
    i = np.argmin(R)
    r, xs, ys, mu  = R[i], x[i], y[i], np.nan

    # next, compute the perpendicular line to every line segment
    # check if the intersection lies between teh two vertices of the segment
    # if so, use that distance instead of the previous
    for i in range(len(polyline) - 1):
        line = polyline[i:i+2]
        r_, mu_, xs_, ys_ = point2line(point, line)
        if mu_ <= 1 and mu_ >= 0 and r_ < r:
            r, mu, xs, ys = r_, mu_, xs_, ys_
    return  r, mu, xp, yp, xs, ys


#%% poin23linnes

def point2lines(point, lines, within=True):
    """Return perpendicular distances to a list of lines or paths.

    Parameters
    ----------
    point : 2-tuple of floats
        The point form which the distance is to be computed.
    lines : list of lines or paths
        The  lines to which the perpendiular distance is to be computed.
        lines can be a list of lists or ndarrays in which each is a list of
        (n, 2) coordinates
    within: bool
        if True, then distance is also computed to the line extended beyond
        its start and end point. Otherwise returns np.nan of outside.

    Returns
    -------
    ndarray of floats shape (n, 4) per line np.array[r, xp, yp, xs, ys, mu]
        The shortest pendicular distance to each line or path.
        if beyond the ends of the line and within is False then
        the lines in the array are np.array([nan, xp, yp, nan, nan, mu])
    """
    xp, yp = point
    rxyxy = np.zeros(len(lines), 6) * np.nan
    for i, line in  enumerate(lines):
        if isinstance(line, Path):
            line = Path.vertices[[0, -1]]
        else:
            line = line[[0, -1]]
        r_, mu_, xp_, yp_, xs_, ys_ = point2line(point, line)
        if (mu_ >= 0 and mu_ <= 1) or within:
            rxyxy[i]= np.array([r_, xp, yp, xs_, ys_, mu_])
        else:
            rxyxy[i] = [np.nan, xp, yp, np.nan, np.nan, mu_]
        return rxyxy

#%% show_p2line

def show_p2line(point, line):
    """Show the instersection of line forom pont perpendicular to other line.

    Parameters
    ----------
    point: two tuple
        x, y of point from which the distance is to be computed.
    line: ndarray of 2x2 or sequence of two points
        line to which the perpendicular distance is to be computed.
    """
    fig, ax = plt.subplots()
    ax.plot(*point, 'ro')
    ax.plot(*line.T, 'b')
    r, xp, yp, xs, ys, mu = point2line(point, line)

    ax.plot([xp, xs], [yp, ys], 'm')
    ax.plot([line[0, 0], xs], [line[0, 1], ys], 'c')
    ax.set_aspect(1.)
    return abs(r)

def show_p2polyline(point, polyline):
    """Show the instersection of line from point to polyline.

    Parameters
    ----------
    point: two tuple
        x, y of point from which the distance is to be computed.
    polyline: ndarray of (n, 2) or Path
        polyline to which the shortest distance is to be computed.
    """
    fig, ax = plt.subplots()
    ax.plot(*point, 'ro')
    if isinstance(polyline, Path):
        ax.plot(*polyline.vertices.T, 'b')
    else:
        ax.plot(*polyline.T, 'b')
    r, xp, yp, xs, ys, mu = point2polyline(point, polyline)
    ax.plot([xp, xs], [yp, ys], 'm')
    ax.set_aspect(1.)
    return abs(r)


#%% main

if __name__ == '__main__':

    GIS       = '/Users/Theo/GRWMODELS/python/Juka_model/GIS/shapes/' # change
    shpnm = 'steilrandstukken.shp'
    shpnm = 'Steilrand.shp'
    shpnm = 'SteilrandGebieden.dbf'

    #demebores = shapes2dict(os.path.join(GIS, 'demebores.shp'), key='Hole_ID')

    #%% using dict2shp and frm2shp

    mydict = {'Hello': 'hello', 'length': 3,
              'width': 4.2, 'x' : 180320., 'y': 337231., 'z': 23.3}

    pdb.set_trace()
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




