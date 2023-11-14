
# -*- coding: utf-8 -*-
"""Definition of Grid class.

A Grid class instance stores a Modflow Grid (rectangular with variable z).
It is used as both a grid container to pass a grid to functions
and as a server to requist numerous grid properties for instance
necessary for plotting. These properties can becomputed on the fly
and need not be storedr.
The Grid class also performs error checking.

Created on Fri Sep 30 04:26:57 2016
Adapted to mf6.VDIS Fri 25 Mar 2022

@author: Theo
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as Polygon
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.axes import Axes
from datetime import datetime
import pandas as pd
from collections import OrderedDict
import scipy.interpolate as ip
from scipy.interpolate import splprep, splev, RectBivariateSpline, interp2d
import pdb

def AND(*args):
    """Return results of multiple and's on array."""
    L = args[0]
    for arg in args:
        L = np.logical_and(L, arg)
    return L

def OR(*args):
    """Return results of multiple or's on array."""
    L = args[0]
    for arg in args:
        L = np.logical_or(L, arg)
    return L

NOT = np.logical_not

intNaN = -999999

def newfig(title, xlabel, ylabel, xlim=None, ylim=None, figsize=(8, 6), fontsize=None):
    """Return ax of new fig."""
    fig,ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if figsize:
        fig.set_size_inches(figsize)
    if fontsize:    
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fontsize)
    ax.grid()
    return ax
    

# interpolate along the first dimension (time or z) for points x, y.
# This can be used if your data is (nt, ny, nx) or if it's (nz, ny, nx)

def show_lines(self, ax=None, co_dict=None,**kwargs):
    """Show muliple lines contained in coordinate dictioinary co_dict.

    Parameters
    ----------
    ax: plt.Axes
        axis or None to generate a fig with axis
    co_dict: dict with model coordinates to be plotted
        coordinates are co_dict[key][:,0], co_dict[key][:,1]

    kwargs ['title', 'xlabel', 'ylabel', 'xlim', 'ylim'] --> ax

    all other kwargs --> ax.plot

    @TO 180608
    """
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_title(kwargs.pop('title', 'Title'))
        ax.set_xlabel(kwargs.pop('xlabel', 'x [m]'))
        ax.set_ylabel(kwargs.pop('ylabel', 'y [m]'))
        ax.grid()
    if 'xlim' in kwargs: ax.set('xlim', kwargs.pop('xlim'))
    if 'ylim' in kwargs: ax.set('ylim', kwargs.pop('ylim'))

    for k in co_dict:
        ax.plot(co_dict[k][:,0], co_dict[k][:,1], label=k)


class StressPeriod:
    """Define stress period object.

    Was used with Juliana Canal project for conveniently handling stress
    periods and manipulating them.
    @TO 2018
    """

    def __init__(self, events_df, tsmult=1.25, dt0=1/24.):
        """Return stress period object.

        Note that only the start time of the last stress period will be used
        as the end time of the simulation. So at least two stress periods
        have to be defined. Doing so, only the first is used and its lengt
        is defined by the start time of the last stress period.

        The number of time steps in each stress period is determined by
        tsmult and dt0. The time-step lengths will be adjusted such that
        tsmult is applied exactly and the first time_step is less or equal to
        dt0.

        Parameters
        ----------
        events_df : pd.DataFram
            must have the following columns:
                ['SP', 'year', 'month', 'day', 'hour,
                                 'nstp', 'tsmult', 'remark']
        tsmult : float
            time step multiplier within stress periods
        dt0 : float
            approximate length of first time step of each stress period.
        """
        # time-unit_conversion from seconds
        self.events = events_df

        # conversion of tiem dimension
        self.ufac =  {'s': 1., 'm': 1/60., 'h': 1/3600. , 'd': 1/86400.,
                    'w': 1/(7 * 86400.)}

        columns = set(self.events.columns)
        required_columns = set(['SP', 'year', 'month', 'day', 'hour',
                                     'steady', 'remark'])
        if not required_columns.issubset(columns):
            missed = [m for m in required_columns.difference(columns)]
            raise Exception('Missing required columns:' +
                            ', '.join(missed)[1:])

        self.events.fillna(method='ffill', inplace=True)

        # Extract stress period information. Note that the last SP is a dummy
        # that does not count, so select [:-1]
        self.SP_numbers = np.asarray(np.unique(self.events['SP']),
                                     dtype=int)

        if any(np.diff(self.SP_numbers)>1):
            print(self.SP_numbers[1:][np.diff(self.SP_numbers)>1])
            raise Exception('Stress periods not consecutive')

        self.nper = len(self.SP_numbers)

        if self.nper < 1:
            raise ValueError('Need at least 2 stress periods.\n' +
                    'The last one only determines the end time of the simulation.')

        self.SP = dict()
        prev_sp = -1
        for i in self.events.index:
            se = self.events.loc[i] # next stress event
            sp = int(se['SP'])

            """ an arbitrary number of duplicates may be used. This implies
            that only the last line with this SP is kept."""

            start = np.datetime64(datetime(year  =int(se['year']),
                                 month =int(se['month']),
                                 day   =int(se['day']),
                                 hour  =int(se['hour']),
                                 minute=0,
                                 second=0), 's')
            if sp > prev_sp:
                # use time and steady of fist event of new sp only (is safe)
                self.SP[sp] = {'start' : start, 'steady': se['steady']}
                if sp > 0:
                    # set the end time of prev. sp, it is the start time of the new one
                    self.SP[prev_sp]['end'] = start
            prev_sp = sp

        self.SP[sp]['end'] = start # last df_event is the end time of the last sp

        # Convert back to pd.DataFrame (now with one entry per stress period)
        self.SP = pd.DataFrame(self.SP).T

        decreasing = self.SP['start'].diff() < pd.Timedelta(0, 'D')
        if np.any(decreasing):
            np.where(decreasing)[0]
            raise ValueError('Start times for stress period(s) {} not increasing !'
                  .format(np.where(decreasing)[0]))

        """ The three lines above moves the start times of the subsequent SP up.
        The original first start_time will end last because it gets the last
        index. However, this index is lost as the last stress period is
        ignored; only its start time is used as the end time of the simulation.
        """

        # Yields perlen in days
        self.SP['perlen'] = np.abs(self.SP['end'] - self.SP['start'])
        perlen_days = np.asarray(self.SP['perlen'], dtype=float) / (1.e9 * 86400)

        """
        Step length multiplier:
            `dt0 = perlen ((tsmult-1) / (tsmult**nstp - 1))`
        Hence:
            nstp =  ln(perlen / dt0 * (tsmult -1) + 1) / ln(tsmult)
        """
        self.SP['nstp']   = np.array(
                np.ceil(
                    np.log( (np.asarray(perlen_days) / dt0) * (tsmult - 1) + 1)\
                        / np.log(tsmult)
                        ), dtype=int)

        self.SP['steady'] = np.asarray(self.SP['steady'], dtype=int)

        # stready or transient
        self.SP['nstp'].loc[self.SP['steady'] == 1] = 1

        self.SP['tsmult'] = tsmult

    def get_perlen(self, asfloat=True, tunit='D'):
        """Return length of stress periods."""
        plen = np.asarray(self.SP['perlen'], dtype='timedelta64[s]')
        if asfloat:
            return np.asarray(plen, dtype=np.float32) * self.ufac[tunit.lower()]
        else:
            return plen

    @property
    def steady(self):
        """Return which stress periods are steady."""
        return np.asarray(self.SP['steady'], dtype=bool)

    @property
    def tsmult(self):
        """Return time step multiplication array."""
        return np.asarray(self.SP['tsmult'], dtype=float)

    @property
    def nstp(self):
        """Return number of time steps for all stress periods."""
        return np.asarray(self.SP['nstp'], dtype=int)

    def get_datetimes(self, sp_only=False, aslists=True, fmt=None):
        """Return times all steps, starting at t=0.

        Rarameters
        ----------
        sp_only : bool
            return only values for the end of each stress period
        aslists : bool
            if True: return as two lists first keys second values
            if False: return as tuples (key, value)
        fmt: None or format
            fmt like in datetime.strfime, results are strings

        Returns
        -------
        an OrderedDict with keys (sp, stpnr)
        """
        _datetimes = OrderedDict()
        for sp in self.SP.index: # self.SP is a DataFrame
            stpNrs  = np.arange(self.SP['nstp'][sp], dtype=int)
            factors = self.SP['tsmult'][sp] ** stpNrs
            dt     = np.cumsum(self.SP['perlen'][sp] * \
                               (factors / np.sum(factors)))
            for it, stpnr in enumerate(stpNrs):
                deltat = self.SP['start'][sp] + dt[it]
                if fmt is None:
                    _datetimes[(stpnr, sp)] = pd.to_datetime(deltat)
                else:
                    _datetimes[(stpnr, sp)] = pd.to_datetime(deltat).strftime(fmt)


        keys = list(_datetimes.keys())
        if sp_only:
            # the keys for the end of each stress period
            kk   = [k for k, j in zip(keys[:-1], keys[1:]) if j[0] > k[0]] + [keys[-1]]
            od = OrderedDict()
            for key in kk:
                od[key] = _datetimes[key]
            _datetimes=od
        if aslists:
           return [list(_datetimes.keys()), list(_datetimes.values())]
        else:
            return _datetimes

    def get_keys(self, sp_only=False):
        """Return entry keys for the stress-period object."""
        return self.get_datetimes(sp_only=sp_only, aslists=True)[0]


    def get_oc(self, what=['save_head', 'print_budget', 'save_budget'], sp_only=False):
        """Return input for OC (output control).

        Parameters
        ----------
        what: list. Optional items in list:
            'save head',
            'save drawdown',
            'save budget',
            'print head',
            'print drawdown',
            'print budget'
        sp_only: bool
            return oc only for stress periods (not for time steps)
        """
        labels=[]
        for w in [w.lower() for w in what]:

            if   w.startswith('save'):  s1 = 'save'
            elif w.startswith('print'): s1 = 'print'
            else:
                raise ValueError("key must start with 'save' or 'print'")

            if   w.endswith('head'):     s2 = 'head'
            elif w.endswith('drawdown'): s2 = 'drawdown'
            elif w.endswith('budget'):   s2 = 'budget'
            else:
                raise ValueError("key must end with 'head', 'drawdown' or 'budget'" )

            labels.append(' '.join([s1, s2]))

        keys = self.get_keys(sp_only=sp_only)

        sp_stp = [(k[1], k[0]) for k  in keys]

        return dict().fromkeys(sp_stp, labels)



    def get_times(self, asfloats=True, tunit='D', sp_only=False, aslists=True):
        """Return datetime of all steps, starting at start time of SP[0].

        Returns
        -------
        OrderedDict with keys (sp, stepnr)
        """
        dt   = self.get_datetimes(sp_only=False, aslists=False)
        keys = list(dt.keys())
        _times = OrderedDict()
        start = self.SP['start'][0]
        for k in keys:
            _times[k]=np.timedelta64(dt[k] - start,  's')

        if asfloats:
            for k in keys:
                _times[k]=np.float32(_times[k]) * self.ufac[tunit.lower()] # to days

        if sp_only:
            # the keys for the end of each stress period
            kk = [k for k, j in zip(keys[:-1], keys[1:]) if j[0] > k[0]] + [keys[-1]]
            od = OrderedDict()
            for k in kk:
                od[k] = _times[k]
            _times = od

        if aslists:
           return [list(_times.keys()), list(_times.values())]
        else:
            return _times

    def get_steplen(self, asfloats=True, tunit='D',  aslists=True):
        """Return steplen all steps.

        Returns
        -------
        OrderedDict with keys (sp, stepnr)
        """
        _dt      = self.get_datetimes(aslists=False, sp_only=False)
        _steplen = OrderedDict()
        keys = list(_dt.keys())

        _steplen[keys[0]] = np.timedelta64(_dt[keys[0]] - self.SP['start'][0], 's')

        for k0, k1 in zip(keys[:-1], keys[1:]):
            _steplen[k1] = np.timedelta64(_dt[k1] - _dt[k0], 's')

        if asfloats:
            for k in keys:
                _steplen[k] = np.float32(_steplen[k]) * self.ufac[tunit.lower()]

        if aslists:
            return [list(_steplen.keys()),list( _steplen.values())]
        else:
            return _steplen


    def show(self, ax=None, co_dict=None, **kwargs):
        """Show the lekvakken.

        Parameters
        ----------
        ax: plt.Axis
            axis or None to generate a fig with axis
        co_dict: dict with model coordinates to be plotted
            coordinates are co_dict[key][:,0], co_dict[key][:,1]
        """
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_title(kwargs.pop('title', 'Title'))
            ax.set_xlabel(kwargs.pop('xlabel', 'x [m]'))
            ax.set_ylabel(kwargs.pop('ylabel', 'y [m]'))
        if 'xlim' in kwargs: ax.set_xlim('xlim')
        if 'ylim' in kwargs: ax.set_ylim('ylim')

        for i in self.events.index:
            se = self.events.loc[i]
            ax.plot([se.x1, se.x2, se.x2, se.x1, se.x1],
                    [se.y2, se.y2, se.y1, se.y1, se.y2], **kwargs)
        if co_dict is not None:
            for k in co_dict:
                ax.plot(co_dict[k][:,0], co_dict[k][:,1], label=k)


def cleanU(U, iu):
    """Clean up relative coordinates U, iu.

    Clean up relative coordiantes U, iu such that
    U lies between 0 and 1 (within a cell) and iu is
    the cell number. The point is inside the model
    if the cell number >=0 and <nx or likwise
    ny and nz for the other two directions.
    """
    U     = np.array(U,  dtype=float)
    iu    = np.array(iu, dtype=int)
    shift = np.array(U,  dtype=int)  # integer part of U
    shift[U < 0] -= 1
    U  -=  shift  # now U is between 0 and 1
    iu +=  shift  # iu is update pointing at cell where point is
    return U, iu


def index(xp, x, left=-999, right=-999):
    """Return index for points xp in grid x.

    Parameters
    ----------
    xp : arraylike
        points to be interpolated
    x : arraylike
        grid points
    left : float
        value for values left of left-most x
    right : float
        value for values right of right-most x

    Returns
    -------
    I : arraylike
     index of cells of xp
    """
    xp = np.array(xp)
    x  = np.array(x)
    if np.any(np.diff(x)<0):
        x  = -x
        xp = - xp
    assert np.all(np.diff(x) > 0), "x is not fully increasing or decreasing"

    I = np.array(np.interp(xp, x, np.arange(len(x)), left, right),
                 dtype=int)
    return np.fmin(I, len(x))

# %%

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

def check_clockwise(X, Y):
    """Return True if clockwise, else False.
    
    Parameteres
    -----------
    Xp, Yp: ndarrays (n, m)
        Coordinates of blocks of grid.
    """
    dx1, dy1 = X[0, 1] - X[0, 0], Y[0, 1] - Y[0, 0]
    dx2, dy2 = X[1, 0] - X[0, 0], Y[1, 0] - Y[0, 0]
    z = np.cross([dx1, dy1], [dx2, dy2])
    return z < 0
    

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

def IAJA(self, icelltype):
    """Return pointer arrays IA and JA for active cells in regular grid.
    
    Pointers for dealing with conecttion (array elements) in CRS sparge storage.
    Computed here for a regular grid with icelltype==0 indicating inactive cells.
    
    Parameters
    ----------
    JA: ndarray of int (NJA)
        JA points to the cells connected to cell IA 
    IA: ndarray of int (NOD)
        IA the pointS to the actual cell in JA
    """
    if icelltype == 1:
        icelltype = self.const(1, dtype=int)
        
    Iact = self.NOD; Iact[icelltype==0] = -1
    
    minus1 = self.const(-1, dtype=int)
    
    T = minus1.copy(); T[ 1:,   :,   :] = Iact[:-1,   :,   :]
    N = minus1.copy(); N[  :,  1:,   :] = Iact[  :, :-1,   :]
    W = minus1.copy(); W[  :,   :,  1:] = Iact[  :,   :, :-1]
    E = minus1.copy(); E[  :,   :, :-1] = Iact[  :,   :,  1:]
    S = minus1.copy(); S[  :, :-1,   :] = Iact[  :,  1:,   :]
    B = minus1.copy(); B[:-1,   :,   :] = Iact[ 1:,   : ,  :]
    
    JA = np.vstack((self.NOD.ravel(), T.ravel(), N.ravel(), W.ravel(), E.ravel(), S.ravel(), B.ravel())).T    
    IA = np.hstack((0, np.cumsum([np.sum(ja != -1) for ja in JA])))
    JA = JA[JA != -1] 
    return IA, JA

def get_vertcells_from_diamond(xd=None, yd=None, nrow=None, ncol=None):
    """Return vertices and cell2d in diamond shape for mf6 disv package.
    
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
    cell2d: list of list, each record [ic, xc, yc, nvert, verts]
        definiction of cells in horizontal plane
    
    Examples
    ----------
    >>> xd = np.array([0, 25, 30, 5])
    >>> yd = np.array([0, 15, 35, 40])
    >>> vertices, cell2d = get_vertcells_from_diamond(xp, yp, nrow=nrow, ncol=ncol)
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
    cell2d = [[ic, xc, yc, 4, np.array(verts, dtype=int)]
                 for ic, xc, yc, verts in zip(rv(Ic), rv(Xc), rv(Yc), rv(Iverts))]
    
    cell2d_ra = np.zeros(len(rv(Ic)), dtype=[
        ('x', 'float'), ('y', float), ('nvert', float), ('verts', 'int', 4)])
    
    cell2d_ra['x'] = Xc.ravel()
    cell2d_ra['y'] = Yc.ravel()
    cell2d_ra['nvert']= 4
    cell2d_ra['verts'] = Iverts
                        
    return vertices, cell2d, vertices_ra, cell2d_ra

# nrow, ncol = 6, 7
# xd = np.array([0, 25, 30, 5])
# yd = np.array([0, 15, 35, 40])
# vertices, cell2d, vertices_ra, cell2d_ra = get_vertcells_from_diamond(xd, yd, nrow=nrow, ncol=ncol)

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

def show_grid(vertices_ra, cell2d_ra):
    plt.plot(vertices_ra['xv'], vertices_ra['yv'], '.k')
    plt.plot(cell2d_ra['xc'], cell2d_ra['yc'], 'rx')

    for verts in cell2d_ra['verts']:    
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

def quadrangle_grid(Xp, Yp, Nx, Ny, XY=False):
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
    XY: bool
        if True return X, Y node  vertices arrays
        if True return vertices, cell2d, vertices_ra, cell2d_ra
    
    Returns
    -------
    vertices: list of lists [iv, xv, yv] defining the vertices
        The vertices of the grid.
    cell2d: list of list defining the cells [ic, xc, yc, nvert, vertIds]
        The cells.
    vertices_ra: np.recarray defining the verties
        same as vertices but as a recarray instead of a list.
    cell2d_ra: np.recarray or strutured array
        same as cell2d but as a recarray instead of a list.
    
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
    >>> vertices, cell2d, vertices_ra, cell2d_ra = quadrangle_grid(Xp, Yp, Nx, Ny)
    >>> 
    >>> plt.plot(vertices_ra['xv'], vertices_ra['yv'], '.k')
    >>> plt.plot(cell2d_ra['xc'], cell2d_ra['yc'], 'rx')
    >>> 
    >>> for verts in cell2d_ra['verts']:    
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
            
    if XY:
        return X, Y

    Xc, Yc = ctr(X), ctr(Y)
    Ic = np.arange(nrow * ncol, dtype=int).reshape((nrow, ncol))
    Iv = np.arange((nrow + 1) * (ncol + 1), dtype=int).reshape((nrow + 1, ncol + 1))
    Icorners = np.vstack((rv(Iv[:-1, :-1]), rv(Iv[:-1, 1:]), rv(Iv[1:, 1:]), rv(Iv[1:, :-1]))).T
    
    vertices = [[iv, xc, yc] for iv, xc, yc in zip(rv(Iv), rv(X), rv(Y))]
    cell2d = [[ic, xc, yc, 4, *corners] for ic, xc, yc, corners in
                                zip(rv(Ic), rv(Xc), rv(Yc), Icorners)]
    
    vertices_ra = np.zeros(len(rv(Iv)), dtype=[('xv', '<f8'), ('yv', '<f8')])
    vertices_ra['xv'] = rv(X)
    vertices_ra['yv'] = rv(Y)
    cell2d_ra = np.zeros(len(rv(Ic)), dtype=[
        ('xc', '<f8'), ('yc', '<f8'), ('nvert', '<i4'), ('verts', 'i4', 4)])
    cell2d_ra['xc'] = rv(Xc)
    cell2d_ra['yc'] = rv(Yc)
    cell2d_ra['nvert'] = 4
    cell2d_ra['verts'] = Icorners
    return vertices, cell2d, vertices_ra, cell2d_ra

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
    rv = lambda Z: Z.ravel()
    
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
    rv = lambda Z: Z.ravel()
     
    Cll = np.vstack((rv(ll(Xv) + lr(Xv) + ul(Xv)) / 3.,
                      rv(ll(Yv) + lr(Yv) + ul(Yv)) / 3.))
    Cur = np.vstack((rv(ur(Xv) + ul(Xv) + lr(Xv)) / 3.,
                      rv(ur(Yv) + ul(Yv) + lr(Yv)) / 3.))
     
    A, All, Aur = get_areas(Xv, Yv, all=True) # A=Area, All area of ll half, Aur = area of ur half

    Xc = (Cll[0] * All + Cur[0] * Aur) / A
    Yc = (Cll[1] * All + Cur[1] * Aur) / A
    return Xc, Yc

# %%
def get_vertices_cell2d(Xv, Yv):
    """Return both the vertex list and the cell2d list required by mf6 vdis.
    
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
    cell2d: list of list or tuples with each line is (ic, xc, yc, nvert, vertices)
        The cells definition as required by mf6 vdis
    """
    rv = lambda Z: Z.ravel()
    
    # Vertex numbers as an array of Xv.shape
    Iv = np.arange(np.prod(Xv.shape), dtype=int).reshape(Xv.shape)
    
    # Vertex numbers of cell corners (clockwise)
    Ivertices = np.vstack((rv(ul(Iv)), rv(ur(Iv)), rv(lr(Iv)), rv(ll(Iv)))).T
    
    # Cell numbers (flattened)
    Icells = np.arange((Xv.shape[0] -1) * (Xv.shape[1] - 1), dtype=int)
    
    Xc, Yc = get_centers(Xv, Yv)
    
    nvert = 4
    cell2d = [(ic, xc, yc, nvert, *verts)
                for ic, xc, yc, verts in zip(Icells, Xc, Yc, Ivertices)]
    vertices = [(iv, xv, yv) for iv, xv, yv in zip(rv(Iv), rv(Xv), rv(Yv))]
    return vertices, cell2d
    
def points_in_cells(X, Y, p):
    """Return the grid cells the point is in.
    
    The algorithm checks if each point is to the left of all for 4 cell edges.
    This function is fully vectorized by putting info per cell on rows and per point in column.
    
    Parameters
    ----------
    X, Y: ndarrays (nrow + 1, ncol + 1)
        Vertex coordinates of grid (y decreasing)
    p: 2-vector (point)
        Point to check.
        
    Returns
    -------
    array of cell numbers telling in which cell each given point lies,
    or -1 if point lies outside model grid.
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


class GridVdis:
    """Define a class to hold 3D finite difference unstructured grid with
        structured rows and columns like old modflow but with X, Y vertices
        more or less arbitrary.

    Grid instances are 3D finite difference grids having many properties
    that can be computed when requested. It can also plot itself.

    Attributes
    ----------
    `axial` : bool
        If true interpret grid as axially symmetric, ignore y-values
    `min_dz` : float
        Mininum layer thickness to enforce
    'spline' : bool
        if True generate splines through control points.
        if Ffalse generate straight lines (quadraingles) through control points.

    Methods
    -------
    `plot(linespec)`
        Plot the grid

    Examples
    --------
    gr = GridVdis(X, Y, z, nx=100, ny=10, axial=False)
    gr.plot('r-')

    Obtaining grid property values:
    gr.nx, gr.ny, gr.nz, gr.shape, gr.xm, gr.ym, gr.zm etc. etc.

    Note that Z is the array of all surface elevations starting at the top of the model
    and continuing to its bottom, that is including all boundaries between
    the bottom of confining beds and underlying model layers.

    The shape of the model must be equal to that of the model cells, excluding
    the confininb beds.
    The tops and bottom should reference whos of the model layers and or
    those of the confining beds.


    @TO 160930
    """
    
    def __init__(self, Xp, Yp, z, min_dz=0.001, tol=1e-5, ncellsx=None, ncellsy=None,
                 georef=None, spline=True):
        """Construct grid object.

        Parameters
        ----------
        `Xp` : ndarray 2D
            x-coordinates of grid control points.
        `Yp` : arraylike
            y-coordinates of the gridd control points.
            Make sure Y is descending to prevent an upsidedown grid.
        `z` : arraylike
            z-coordinates of the grid cells' tops and bottoms
            z may be a vector, implying all z-planes are horizontal. The
            vector holds tops and bottoms of these layers. The length
            of z, therefore is one more than the number of layers. `z`
            as a vector will be made unique and descending.

            z may be a full 3D array holding tops and bottoms of a cells.
            The length of z in the 3rd dimension is one more than the
            number of layers.
            z will be made unique and descending in 3rd dimension.

            Notice
            ------
            z must inlcude a top and a bottom of every layer (mf6 vdis does not have confining beds).
        `min_dz` : float
            Mininum layer thickness to be enforced (top down)
        `tol` : float
            Difference in x and y coordinates above with coordinates are
            considered distinct. Coordinates will be rounded to tol before
            applying uninque to them
        ncellsx: int or sequence of ints
            Total number of cells along rows if int.
            Total number of cells along rows between each pair of control points.
        ncellsy: int of sequence of ints
            Total number of cells along columns if int.
            Total number of cells along columns between each pair of control points.
        'georef` : tuple
            (x0, x0, xw, yw, angle) relating world and model coordinates
        'spline' : bool:
            if True the grid will be made of splines through control points.
            if False teh grid will be made of affine quadrangles through control points.
            
        Returns
        -------
        GridVdis object
        
        >>> gr = GridVdis(Xp, Yp, z, ncellsx=50, ncellsy=50, min_dz=0.001, georef=None, spline=True)  # Grid is based on splines through the points defiend by Xp, and Yp
        >>> gr = GridVdis(Xp, Yp, z, ncellsx=50, ncellsy=50, min_dz=0.001, georef=None, spline=False) # Grid based on quadrilaterals given by Xp, Yp
        """
        self.AND = np.logical_and
        self.NOT = np.logical_not

        self.intNaN = -9999

        self._min_dz = min_dz
        self._tol = tol
        self._digits = int(abs(np.log10(tol)))
        
        if len(Xp.shape) == 1 and len(Yp.shape) == 1:
            nx = len(Xp) - 1
            ny = len(Yp) - 1
            Xp = Xp[np.newaxis, :] * np.ones((ny + 1, 1))
            Yp = Yp[:, np.newaxis] * np.ones((1, nx + 1))
        if ncellsx is None:
            ncellsx = np.ones(Xp.shape[1] - 1)
        if ncellsy is None:
            ncellsy = np.ones(Yp.shape[0] - 1)
            
        assert check_clockwise(Xp, Yp), ("Xp, Yp cells must be clockwise," +
                    "i.e. like when Xp to the right and Yp downward (or rotated)")
            
        if spline:
            self._Xv,  self._Yv = get_spline_grid(Xp, Yp, ncellsx, ncellsy)
        else:
            self._Xv, self._Yv = quadrangle_grid(Xp, Yp, ncellsx, ncellsy, XY=True)
            
        self._Xc, self._Yc = get_centers(self._Xv, self._Yv)
        self._Area = get_areas(self._Xv, self._Yv, all=False)
        self.vertices, self.cell2d = get_vertices_cell2d(self._Xv, self._Yv)

        self.nx = self._Xv.shape[1] - 1
        self.ny = self._Xv.shape[0] - 1

        z = np.array(z)
        # Check shape of z:
        #   It may be a vector for grids having horizontal layers
        #   or a 3D array where the layers correspond with tops and bottoms at the cell centers.
        if z.ndim == 1: # z is given as a vector (tops and bottoms of layers)
            self._full = False
            
            # unique and put diecton upward increasing
            self._z = np.array(np.unique(z)[::-1], dtype=float)
            
            # enforce min_dz
            dz = np.fmax(np.abs(np.diff(self._z)), self._min_dz)
            
            # Then redefine self._z using _z[0] and dz
            self.nz = len(dz)
            self._z = self._z[0] * np.ones(self.nz + 1)
            self._z[1:] -= np.cumsum(dz)
            self._z = self._z[:, np.newaxis, np.newaxis] # finally, make z_ 3D
            
        elif len(z.shape) == 3: # z is given as full 3D grid
            if self.ny != z.shape[1] or self.nx != z.shape[2]:
                print("\n\
                    The number of rows and columsn of Z is ({0},{1}).\n\
                    But expected was (ny, nx) = (nrow, ncol)=({3}, {4}) !\n\
                    Maybe the length of x and or y have been changed by\n\
                    __init__ to eliminate coordinates that are less\n\
                    than tol={2} apart.\n\
                    \n\
                    Remedy:\n\
                     1. Check that the difference between the sorted x coordinates\n\
                        is larger than the default or specified tolerance={2}.\n\
                     2. If so, then check the same for the y coordinates.\n\
                     3. If also correct, verify the size of your Z array.\n\
                        It must have (ny=nrows={3}, nx=ncol={4}).".
                        format(z.shape[0],z.shape[1],self._tol,
                               self.ny, self.nx))
                raise ValueError("See printed message for details")
            else:
                self._full = True
                self._Z = np.array(z, dtype=float)
                self.nz = self._Z.shape[0] - 1
                # make sure Z runs downward
                if np.all(self._Z[0] < self._Z[-1]): # @TO 210629 np.all
                    self._Z = self._Z[::-1]

                # make sure the Z[i] - Z[i+1] >= self._min_dz (min layer thickness)
                for iz in range(self.nz):
                    dz = self._Z[iz] - self._Z[iz + 1] # must be positive
                    dz[dz <= self._min_dz] = self._min_dz
                    self._Z[iz + 1] = self._Z[iz] - dz

        else: # illegal shape
            s ="""\n\
                    z.shape = {}, but expected was a 1D vector\n\
                    or 3D array (nz+1, {}, {})""". \
                          format(z.shape, self.ny, self.nx)
            raise ValueError(s)

        # Cells per layer
        self.ncpl = self.nx * self.ny
        # Verti es per layer
        self.nvert = np.prod(self._Xv.shape)
        self.ncol, self.nrow, self.nlay = self.nx, self.ny, self.nz

        self._shape = (self.nz, self.ny, self.nx)
        self._shape_LIc = (self.nz, self.ncpl)

        # Check if input wass consistent
        assert self.nz > 0, "nlay = {} must be >1".format(self.nz)

        if georef is None:
            georef = np.array([0., 0., 0., 0., 0.])
        else:
            georef = np.array(georef)
            assert len(georef)==5,"georef must be a sequence or array having\
                    values [xm0, ym0, xw0, yw0, angle]"
        self.georef_ = {'xm0': georef[0], 'ym0': georef[1],
                       'xw0': georef[2], 'yw0': georef[3],
                       'alfa': georef[4],
                       'cos': np.cos(georef[4] * np.pi/180.),
                       'sin': np.sin(georef[4] * np.pi/180.)}

        if not isinstance(self._min_dz, float) or self._min_dz <= 0.:
            raise ValueError(
                """min_dz must be a postive float.

                Remedy:
                Use min_dz=value explicitly when calling mfgrid.Grid.
                """)

    @property
    def georef(self):
        """Return the grid's georef = (xm0, ym0, xw0, yw0, alfa)."""
        return (self.georef_['xm0'], self.georef_['ym0'],
                self.georef_['xw0'], self.georef_['yw0'],
                self.georef_['alfa'])

    @property
    def full(self):
        """Return boolean, indicating if full grid is stored.

        If True, the grid stored is a full grid or one in which the z coordinates
        are the same througout the grid (horionzontal layers only).
        """
        return self._full # prevent ref to original

    @property
    def Xv(self):
        """Return x-coordinates of model nodes (not world), always 3D."""
        return self._Xv

    @property
    def Yv(self):
        """Return x-coordinates of model nodes (not world), always 3D."""
        return self._Yv

    @property
    def XV(self):
        """Return x-coordinates of model vertices (not world), always 3D."""
        return self.Xv * np.ones((self.nz, self.ny + 1, self.nx + 1))

    @property
    def YV(self):
        """Return y-coordinates of model vertices (not world) always 3D."""
        return self.Yv * np.ones((self.nz, self.ny + 1, self.nx + 1))

    @property
    def extent(self):
        """Return extent of model grid (Xv.min(), Xv.max(), Yv.min() Yv.max()"""
        return self.Xv.min(), self.Xv.max(), self.Yv.min(), self.Yv.max()

    @property
    def zmean(self):
        """Return average cell top and bottom elevatons [1, 1, nz+1]."""
        if self._full:
            return (self._Z).mean(axis=2).mean(axis=1)
        else:
            return self._Z[:, 0, 0]

    @property
    def Z(self):
        """Return cell top and bottom elevation [nz+1, ny, nx]."""
        if self._full:
            return self._Z.copy()# prevent ref. to original
        else:
            return self._z * np.ones((self.nlay + 1, self.nrow, self.ncol))

    @property
    def shape(self):
        """Return shape of the grid."""
        return self._shape
    @property
    def shape_LIc(self):
        """Return shape of div grid (Layer, layerNd)"""
        return self._shape_LIc

    @property
    def nod(self):
        """Return number of cells in the grid."""
        return np.prod(self._shape)

    @property
    def NOD(self):
        """Return cell numbers in the grid."""
        return np.arange(self.nod).reshape(self.shape)

    def todisv(self, A):
        """Return array A reshaped from LRC to LIcell."""
        assert A.ndim in [2, 3], "A.ndiim must be 2 or 3 not {}.".format(A.ndim)
        if A.ndim == 2:
            return A.ravel()
        else:
            return A.reshape((self.nlay, self.ncpl))

    def todis(self, A):
        """Return array A reshaped from LRC to LIcell."""
        assert A.ndim in [1, 2], "A.ndim must be 1, or 2 to convert to dis, not {}".format(A.ndim)
        if A.ndim == 1:
            return A.reshape((self.nrow, self.ncol))
        else:
            return A.reshape((self.nlay, self.nrow, self.ncol))


    def I2LIc(self, Imask, astuples=None, aslist=None):
        """Return ndarray [L, Icell] indices generated from global indices or boolean array I.

        Parameters
        ----------
        Imask : ndarray of int or bool
            if dtype is int, then I is the global cell index

            if dtype is bool, then I is a zone array of shape
            [ny, nx] or [nz, ny, nx]

        astuples: bool or None
            return as ((l, icell), (l, icell), ...)
        aslist: bool or None:
            return as [[l, icell], [l, icell], ...]
            where l is layer number and icell the cell number within the layer.
        """
        if Imask.dtype == bool:
            if Imask.ndim == 1:
                I = self.NOD.ravel()[Imask]
            elif Imask.ndim == 2:
                I = self.NOD[0][Imask]
            else:
                I = self.NOD[Imask]
        else:
            I = Imask

        I = np.array(I, dtype=int)
        L = I // self.ncpl
        Ic = np.array(I - L * self.ncpl)
        if astuples:
            return tuple((l, ic) for l, ic in zip(L.ravel(), Ic.ravel()))
        elif aslist:
            return [[l, ic] for l, ic, in zip(L.ravel(), Ic.ravel())]
        else:
            return np.vstack((L.ravel(), Ic.ravel())).T # ok, an array np.array([[L, Ic], [L, Ic]])
        
    def LIcell(self, Imask, astuples=None, aslist=None):
        """Return LIc from global index of Imask.
        
        The name was used in several files, but illogic.
        It is replaced by I2Ic see definition above.
        """
        return self.I2LIc(Imask, astuples=astuples, aslist=aslist)
        
        
    def LRC2LIcell(self, LRC):
        """Return LIcell instead of LRC, keeping type of LRC."""
        LIcell = [(il, ir * self.nrow + ic) for il, ir, ic in LRC]
        if isinstance(LRC, np.ndarray):
            return np.array(LIcell, dtype=int)
        else:
            return LIcell
        

    def LRC(self, Imask, astuples=None, aslist=None):
        """Return ndarray [L R C] indices generated from global indices or boolean array I.

        Parameters
        ----------
        Imask : ndarray of int or bool
            if dtype is int, then I is global index

            if dtype is bool, then I is a zone array of shape
            [ny, nx] or [nz, ny, nx]

        astuples: bool or None
            return as ((l, r, c), (l, r, c), ...)
        aslist: bool or None:
            return as [[l, r, c], [l, r, c], ...]
        """
        if Imask.dtype == bool:
            if Imask.ndim == 1:
                I = self.NOD.ravel()[Imask]
            elif Imask.ndim == 2:
                I = self.NOD[0][Imask]
            else:
                I = self.NOD[Imask]
        else:
            I = Imask

        I = np.array(I, dtype=int)
        L = np.array(I / self.ncpl, dtype=int)
        R = np.array((I - L * self.ncpl) / self.ncol, dtype=int)
        C = I - L * self.ncpl - R * self.ncol
        if astuples:
            return tuple((l, r, c) for l, r, c in zip(L, R, C))
        elif aslist:
            return [[l, r, c] for l, r, c, in zip(L, R, C)]
        else:
            return np.vstack((L, R, C)).T

    def LRC_zone(self, zone):
        """Return ndarray [L R C] indices generated from zone array zone.

        Parameters
        ----------
        zone : ndarray of dtype bool
            zonearray, can be o shape (ny, nx) or (nz, ny, nx)
        """
        if zone.ndim == 2:
            return self.LRC(self.NOD[0][zone])
        else:
            return self.LRC(self.NOD[zone])


    def lrc(self, x, y, z=None, Ilay=None):
        """Return zero-based LRC indices (iL,iR, iC) of points x, y, z.

        Points must be given as x, y, z or as x, y, iLay.
        The shape of x, y, and z or iLay must be the same.
        If z is None then iLay is used.
        If iLay is also None, then iLay is all zeros.

        Parameters
        ----------
        x : ndarray
            x-coordinates
        y : ndarray
            y-coordinates
        z : ndarray | None
            z-coordinates
        iLay : ndarray | None
            layer indices

        Returns
        -------
        [iL, iR, iC]
        indices outside the extents of the model are < 0 (-999)

        @TO 171105
        """
        if np.isscalar(x): x = [x]
        if np.isscalar(y): y = [y]
        if np.isscalar(z): z = [z]
        if np.isscalar(Ilay): Ilay = [Ilay]

        x = np.array(x)
        y = np.array(y)

        if z    is not None: z    = np.array(z)
        if Ilay is not None: Ilay = np.array(Ilay, dtype=int)

        assert np.all(x.shape == y.shape), "x.shape must equal y.shape"

        if z is not None:
            assert np.all(x.shape == z.shape), "x.shape must equal z.shape"
        else:
            if Ilay is not None:
                assert np.all(x.shape == Ilay.shape), "x.shape must equal iLay.shape"
            else:
                Ilay = np.zeros_like(x, dtype=int)

        incells = points_in_cells(self.Xv, self.Yv, np.vstack(((x, y))).T)
        
        Irow, Icol = Ic2rc(self, incells, astuples=False).T
        
        if z is not None:
            Ilay = np.zeros_like(x, dtype=int)
            for i, (_z, ix, iy) in enumerate(zip(z, Icol, Irow)):
                Ilay[i] = index(_z, self.Z[:, iy, ix])

        return np.vstack((np.asarray(Ilay, dtype=int),
                          np.asarray(Irow, dtype=int),
                          np.asarray(Icol, dtype=int))).T

    def I(self, LRC):
        """Return global index given LRC (zero based)."""
        LRC = np.array(LRC)
        return LRC[:,0] * self.ny * self.nx + LRC[:, 1] * self.nx + LRC[:, 2]

    def cell_pairs(self, polyline, open=False):
        """Return cell pairs left and right of polygon contour like for HB package.

        Parameters
        ----------
        gr: fdm.mfgrid.Grid
            grid object
        polygon: list of coordinate pairs or a XY array [n, 2]
            contour coordinates
        open: True|False
            use open=True for an open polyline instead of a closed polygon

        Returns
        -------
        Cell pairs: array (n, 2) of int
            array of cell pairs froming the HFB 
            
        >>> x = np.linspace(-100., 100., 21)
        >>> y = np.linspace(-100., 100., 21)
        >>> z = [0, -10, -20]
        >>>
        >>> gr = Grid(x, y, z)
        >>>
        >>> polygon = np.array([(23, 15), (45, 50.), (10., 81.), (-5., 78), (-61., 51.), (-31., 11.),
        >>>            (-6., -4.), (-42., -20.), (-50., -63.), (-7., -95.),
        >>>            (31., -80.), (60., -71.), (81., -31.), (5., -63.),
        >>>            (25., -15.), (95., 40.), (23, 15)])
        >>>
        >>> pairs = cell_pairs(gr, polygon)
        >>>
        >>>
        >>> fig, ax = plt.subplots()
        >>> ax.set_title('Node pairs for the hor. flow-barrier package of Modflow')
        >>> ax.set_xlabel('x [m]')
        >>> ax.set_ylabel('y [m]')
        >>> gr.plot_grid(world=False, ax=ax)
        >>> ax.plot(polygon[:,0], polygon[:,1])
        >>> ax.plot(gr.Xm.ravel()[pairs[:,0]], gr.Ym.ravel()[pairs[:,0]], '.r', label='column 1')
        >>> ax.plot(gr.Xm.ravel()[pairs[:,1]], gr.Ym.ravel()[pairs[:,1]], '.b', label='column 2')
        >>> for pair in pairs:
        >>>     ax.plot(gr.Xm.ravel()[pair], gr.Ym.ravel()[pair], 'k-')
        """
        A = 1e8 # np.Inf does not work far outside th3 grid

        if open==True:
            polygon_west = np.vstack(( np.array([-A, polyline[0][1]]),
                                       polyline,
                                       np.array([-A, polyline[-1][1]])))
            polygon_south = np.vstack((np.array([polyline[0][0], -A]),
                                       polyline,
                                       np.array([polyline[-1][0], -A])))
            In1 = self.inpoly(polygon_west)
            mask_west  = np.hstack((In1, np.zeros((self.ny, 1), dtype=bool)))
            In2 = self.inpoly(polygon_south)
            mask_south = np.vstack((np.zeros((1, self.nx), dtype=bool), In2))
        else:
            if not np.all(polyline[0] == polyline[-1]):
                polyline = np.vstack((polyline, polyline[-1:, :]))
            In = self.inpoly(polygon)
            mask_west  = np.hstack((In, np.zeros((self.ny, 1), dtype=bool)))
            mask_south = np.vstack((np.zeros((1, self.nx), dtype=bool), In))

        west  = np.abs(np.diff(mask_west , axis=1)) == 1
        south = np.abs(np.diff(mask_south, axis=0)) == 1

        east = np.hstack((np.zeros((self.ny, 1), dtype=bool), west[:,:-1]))
        north= np.vstack((south[1:,:], np.zeros((1, self.nx), dtype=bool)))

        pairs = np.array([np.hstack((self.NOD[0][west], self.NOD[0][north])),
                          np.hstack((self.NOD[0][east], self.NOD[0][south]))],
                          dtype=int).T
        return pairs


    def lines2HFB(self, lines, open=True, layers=0, cs=np.nan ):
        """Return HFB input records for many barrier lines at once.

        Parameters
        ----------
        lines : list of line. A line is a list of barrier coordinates
            barrier coordinates are [(xMdl, yMdl), (  ,), ( , )]
        layers: int or list of ints
            layer number for each barrier, if scalar, all layers are the same.
        cs : float or list of floats
            hydraulic resistance of each barrier [T].
            If scalar all barriers get the same resistance.

        @TO 180707
        """
        if not isinstance(lines[0], np.ndarray):
            if not isinstance(lines[0][0], float):
                raise ValueError("Lines must be a list of lines")

        if isinstance(layers, int):
            layer = layers
            layers = [layer for i in range(len(lines))]
        if isinstance(cs, (int, float)):
            c  = cs
            cs = [float(c) for i in range(len(lines))]
        if not (len(lines) == len(layers) and len(lines) == len(cs)):
            raise ValueError(
              'Nr of layers ({}) or of c values ({}) does not equal number of lines ({}).'
                               .format(len(layers), len(cs), len(lines)))

        HFB = []
        for line, layer, c in zip(lines, layers, cs):
            Hfb = self.line2HFB(line, open=open, layer=layer, c=c)
            for hfb in Hfb:
                HFB.append(hfb)

        return HFB


    def line2HFB(self, polyline, open=True, layer=0, c=np.nan):
        """Return HFB input given a polyline (mdl coordinates.

        Parameters
        ----------
        polyline : list of coordinate tuples
            polyline that defines the horizontal flow barrier path (mdl coordinates)
        open : bool
            if False, polyline is a polygon (it will then be closed)
        layer : int
            layer number in which the polyline lies.
        c : float
            HFB resistance [T], it is d/k. HFB uses 1/c.

        Returns
        -------
        list of tuples of the form that HFB needs
        [(L, R1, C1, R2, C2, c), (....)]

        @TO 180607
        """
        pairs = self.cell_pairs(polyline, open=open)

        return self.cell_pairs2HFB(pairs, c=c)


    def cell_pairs2HFB(self, cell_pairs, c=np.nan):
        """Return HFB input as  a list of tuples.

        Parameters
        ----------
        cell_pairs : np.array([nlines, 2]) of model node numbers
            cell_pairs obtained from Grid.cell_pairs()
        c : float
            resistance [T] to use for this HFB

        Returns
        -------
        list of tuples of the form that HFB needs
        [(L, R1, C1, R2, C2, c), (....)]
        """
        LRC1 = self.I2LRC(cell_pairs[:, 0])
        LRC2 = self.I2LRC(cell_pairs[:, 1])
        R = np.ones_like(LRC1[:,0], dtype=float) / c
        return [(*lrc1rc2, r) for lrc1rc2, r in zip(np.hstack((LRC1, LRC2[:, 1:])), R)]


    def I2Icell(self, I, astuples=True):
        """Return L, CellId from global Index. 
        
        Parameters
        ----------
        I: array of ints (n)
            global cell indices
        astuples: bool
            if True return the resulting [L, CellId] as tuples
            else return them as an array of ints
        """
        LRC = self.I2LRC(I, astuples=astuples)
        
        Icell = np.arange(self.nrow * self.col, dtype=int).reshape((
                                    self.row, self.nrow))
        
        out = [(l, Icell[r, c]) for l, r, c in LRC]
        if astuples:
            return out
        else:
            return np.array(out, dtype=int)
        
    def Ic2rc(self, Ic, astuples=True):
        """Return (IR, IC) from array of nodes Ic.

        Parameters
        ----------
        Ic : list of layer node numbers
            The node numbers to be converted to RC
        astuples: bool
            if True --> rc are tuples else LRC is array of ints

        Returns
        -------
        list of tuples or np.array[n, 2] of ints

        """
        Ic = np.asarray(Ic, dtype=int).ravel()

        Nlay = self.ny * self.nx

        IRow = Ic // self.nx
        ICol = Ic - IRow * self.nx
        
        IRow[Ic < 0] = -1
        ICol[Ic < 0] = -1

        out = [(r, c) for r, c in zip(IRow, ICol)]

        if astuples:
            return out
        else:
            return np.array(out, dtype=int)


    def I2LRC(self, I, astuples=True):
        """Return (IL, IR, IC) from array of nodes I.

        Parameters
        ----------
        I : list of global node numbers
            The node numbers to be converted to LRC
        astuples: bool
            if True --> lrc are tuples else LRC is array of int

        Returns
        -------
        list of tuples or np.array[n, 3] of ints

        """
        I = np.asarray(I, dtype=int).ravel()

        Nlay = self.ny * self.nx

        IL = I // Nlay
        IR = (I - IL * Nlay) // self.nx
        IC = I - IL * Nlay - IR * self.nx
        
        IL[I < 0] = -1
        IR[I < 0] = -1
        IC[I < 0] = -1

        out = [(i, r, c) for i, r, c in zip(IL, IR, IC)]

        if astuples:
            return out
        else:
            return np.array(out, dtype=int)


    def interp2(self, hds=None, x=None, y=None, z=None, world=False):
        """Return interpolated hds at Xc, Yc using the heads in layer defined by z.

        Point x, y is generally not the cell center. So interpololate the head at
        x, y using he heads in the surrounding cells such that the point
        x, y is in a rectanble formed by 2 by 2 cells in the plane.

        The 2 x 2 set of cells is lookedup between which mids the point lies.
        Then a bilinear interpolation is done to get the head.

        Parameters
        ----------
        hds: hds from flopy a 4D np.ndarray
            computed heads
        x, y, z : floats
            considered ponts(s)
        world: bool
            true of x, y, z in world coordinates

        Returns
        -------
        heads: np.ndarray of shape [nt, nz, 1]
        """
        if world:
            x, y = self.world2model(x, y)

        LRC = self.lrc(x, y, z)[0]
        L, R1, C1 = LRC[0], LRC[1], LRC[2]

        C2 = C1 + 1 if x > self.xm[C1] else C1 - 1
        R2 = R1 + 1 if y < self.ym[R1] else R1 - 1

        if C2 < C1: C1, C2 = C2, C1
        if R2 < R1: R1, R2 = R2, R1

        C2 = C2 if C2 < self.nx else C1
        C1 = C1 if C1 >= 0    else C2
        R2 = R2 if R2 < self.ny else R1
        R1 = R1 if R1 >= 0    else R2

        hsub = hds[:, L, [R1, R2], :][:, :, [C1, C2]]

        u = 0 if C1 == C2 else 2 * (x - self.xm[C1]) / (self.xm[C2] - self.xm[C1]) - 1
        v = 0 if R1 == R2 else 2 * (y - self.ym[R1]) / (self.ym[R2] - self.ym[R1]) - 1

        h = np.zeros_like(hds[:, 0, 0, 0])

        for ir, b in zip([0, 1], [-1, +1]):
            for ic, a in zip([0, 1], [-1, +1]):
                h += hsub[:, ir, ic] * (a + u) * (b + v) * a * b
                #print(a - u, b - v, (a - u) * (b - v))  a * b

        return h / 4


    def interp(self, points, Z=None, world=True, **kwargs):
        """Return interpolated Z values at points.

        Parameters
        ----------
        points: np.ndarray of ponts like so np.array([[x1m y1], [x2, y2], ...])
            Array of points to interpolate between.
        Z : array of shape [n, ny, nx] to be interpolated
            Array to be interpolated, default is self.Z
        world: bool
            true if points are in world coordinates
        kwargs: additional kwargs

        Returns
        -------
        [s, zp, I: 3 np arrays
                s has shape [n, 3] where columns are s, x, y
                zp has shape [n, len(Z)]
                I has shape (n,) = indices of cornerpoints

        @TO 20180919
        """
        points = np.asarray(points)
        if world:
            points = np.vstack(self.world2model(*points.T)).T

        P1, P2 = points[:-1], points[1:]

        x0, y0 = P1[0]

        S  = np.array([]) # the points and interpolated points
        SP = np.array([]) # the points themselves (piezom locations in the profile)

        for (x1, y1), (x2, y2) in zip(P1, P2):
            dx, dy = x2 - x1, y2 - y1
            L = np.sqrt(dx ** 2 + dy ** 2)

            if len(SP) == 0: SP = np.array([[0, x0, y0]]) # initialize
            if len(S)  == 0: S  = np.array([[0, x0, y0]])

            SP = np.vstack((SP, np.array([[SP[-1][0] + L, x2, y2]])))

            lam1 = (self.x - x1) / dx
            lam2 = (self.y - y1) / dy
            lam = np.hstack((0, lam1, lam2, 1.0))

            # only within 0 1 , sort and remove doubles
            lam = np.unique(lam[np.logical_and(lam >= 0., lam <=  1.)])

            if len(lam) == 0:
                continue # two piezoms in same borehole

            # use [startpoint, points_in_cells, endpoint]
            lamb = np.hstack((lam[0], 0.5 * (lam[:-1] + lam[1:]), lam[-1]))

            s = np.vstack((lamb * L, x1 + lamb * dx, y1 + lamb * dy)).T

            if len(s)  == 0:
                continue

            s[:, 0] += S[-1][0]
            S = np.vstack((S, s))

        if Z is None: Z = self.Z

        if Z.ndim != 3:
            raise ValueError('Z must be a 3D array')

        z = np.zeros((len(S), len(Z)))
        for i, zeta in enumerate(Z):
            #f1 = interp2d(self.Xm, self.Ym[::-1], Z[i, ::-1,  :].T)
            #A = f1(self.xm, self.ym)
            f = RectBivariateSpline(self.xm, self.ym[::-1],
                    Z[i, ::-1,  :].T, kx=1, ky=1, s=0)
            x, y = S[:, 1], S[:, 2]
            z[:, i] = f(x, y, grid=False)

        I = np.unique(S.T[0], return_index=True)

        return S[I[1]], z[I[1]], SP


    def interpxy(self, Z, points, iz=0, **kwargs):
        """Return values by interpolation at points.

        Performs linear interpolation over the first dimension of a 3D array.
        The last two dimenstions as y, x with points = np.array([nPoints, 2]),
        according to new values from a 2D array new_x.

        So if Z is [nz, ny, nx] you get the interpolated values as [nz, np].

        If Z is [nt, nz, ny, nx] then nz is squeezed out by using iz.
        The results is an array of [nt np]

        If Z is  [ny, nx] it is fist turned into [1, ny, nx] and the result
        is a vector nx

        Parameters
        ----------
        Z : 2D or a 3-D or 4-D ndarray (double type)
            Array containing the z values to interpolate.
            That is Z=([[nt ]nz,] ny, nx)
        points : 2-D ndarray (double type)
            Array containing the points at which we want to interpolate Z
            xp = points[:,0] and yp = points[:, 1]
        iz : int
            layer number.
            In case Z is 4D, this layer number is first
            squeezed out like Z = Z[:, ilay, :, :]
        kwargs : additional keyword arguments
            additional kwargs are passed on to interpolator.
            See for these arguments `scipy.interpolate.RectBivariateSPline`

        Results
        -------
            array of shape(len(Z), len(points)) with the interpolated values for
            each z.
            Notice that Z may also be time. It's just the first dimension of
            the 3D array that is interpolated on x, y.

            If your array is 4D, i.e. (time, z, y, x), then squeeze out the
            z is squeezed out first. (Interpreted as layer.)
        """
        fill_value = -999.

        if Z.ndim == 2:
            Z = Z[np.newaxis, : ,:] # makes it work also if Z.shape=(ny, nx)
        elif Z.ndim == 4:
            Z = Z[:, iz, :, :] # squeeze out the layer

        assert Z.shape[-2:] == self._shape[-2:],\
            'Z must be 3D of shape (nz,{},{})'.format(self.ny, self.nx)
        assert Z.ndim == 3, 'Z.ndim must be 3.'

        points = np.array(points)
        assert points.shape[-1] == 2, 'Points must be of shape (npoints, 2)'

        xM, yM = self.world2model(points[:,0], points[:,1])

        result = np.zeros((len(Z), len(points)))
        for iz_, A in enumerate(Z):
            m = np.min(A)
            A[np.isnan(A)] = fill_value # prevent nan

            f = interp2d(self.xm, self.ym[::-1], A[::-1], fill_value=fill_value)
            result[iz_, :] = [f(x, y) for x, y in zip(xM, yM)]
        result[result < m] = np.nan
        return result


    @property
    def dz(self):
        """Return vector [nz] of average layer thicknesses."""
        return np.abs(np.diff(self._Z, axis=0).mean(axis=2).mean(axis=1))

    @property
    def Dx(self):
        """Return 2D array [ny, nx] of column widths."""
        X = 0.5 * (self._Xv[:-1, :] + self._Xv[1:, :])
        Y = 0.5 * (self._Yv[:-1, :] + self._Yv[1:, :])
        Dx = np.sqrt((X[:, 1:] - X[:, :-1]) ** 2 + (Y[:, 1:] - Y[:, :-1]) ** 2)
        return Dx

    @property
    def Dy(self):
        """Return 2D array [ny, nx] of row widths."""
        X = 0.5 * (self._Xv[:, :-1] + self._Xv[:, 1:])
        Y = 0.5 * (self._Yv[:, :-1] + self._Yv[:, 1:])
        Dy = np.sqrt((X[1:, :] - X[:-1, :]) ** 2 + (Y[1:, :] - Y[:-1, :]) ** 2)
        return Dy

    @property
    def DX(self):
        """Return 3D gid [ny, nx, nz] of column."""
        return np.ones(self.shape) * self.Dx[np.newaxis, :, :]

    @property
    def DY(self):
        """Return 3D grid [ny, nx, nz] of row widths."""
        return self.Dy[np.newaxis, :, :] * np.ones(self.shape)

    @property
    def DZ(self):
        """Return 3D grid [ny, nx, nz] of Layer thicnesses."""
        if self._full == False:
            return np.abs(np.diff(self._Z, axis=0) * np.ones(self.shape))
        else:
            return np.abs(np.diff(self._Z, axis=0))

    @property
    def Area(self):
        """Return area of the cells as a 2D grid [ny, nx]."""
        return self._Area

    @property
    def area(self):
        """Return total area of grid (scalar), see Grid.Area for cell areas."""
        return np.sum(self._Area)

    @property
    def Volume(self):
        """Return volume of cells as a 3D grid [ny, nx, nz]."""
        return self._Area.reshape((self.nrow, self.ncol))[np.newaxis, :, :] * self.DZ

    @property
    def volume(self):
        """Return total volume of grid (scalar), see Grid.Volume for cell volumes."""
        return self.Volume.sum()

    @property
    def Xm(self):
        """Return grid column center coordinates vector."""
        return self.Xc.reshape((self.nrow, self.ncol))

    @property
    def Ym(self):
        """Return grid row center coordinates vector."""
        return self.Yc.reshape((self.nrow, self.ncol))
    
    @property
    def XM(self):
        """Return column center coordinates as a 3D grid [nz, ny, nx]."""
        return self.Xm[np.newaxis, :, :] * np.ones(self.shape)

    @property
    def YM(self):
        """Return row center coordinates as a 3D grid [nz, ny, nx]."""
        return self.Yc[np.newaxis, :, :] * np.ones(self.shape)

    @property
    def ZM(self):
        """Cell center coordinates as a 3D grid [nz, ny, nx]."""
        return 0.5 * (self.Z[:-1, :, :] + self.Z[1:, :, :])

    @property
    def Xc(self):
        """Return Xc cell centers as 1d ndarray.
        """
        return self._Xc

    @property
    def Yc(self):
        """Return Yc cell centers as a 1d ndarray.
        """
        return self._Yc

    @property
    def ZC(self):
        """Return Z of cell centers, same as ZM (nlay, nrow, ncol).
        """
        return self.ZM

    @property
    def XC(self):
        """Return full3D Xc cell centers.
        """
        return self.XM

    @property
    def YC(self):
        """Return full 3S Yc cell centers.
        """
        return self.YM

    @property
    def Xp(self):
        """Return Xp grid coords except first and last as 2D vertical grid [nz+1, nx-1].

        Convenience property for plotting stream lines.
        """
        return self._Xv[:, 1:-1]

    @property
    def Zp(self, row=0):
        """Return Zp grid coords in z-x cross section [nz+1, nx-1].

        Except first and last as 2D vertical grid.

        Averaging over adjacent columns.

        Convenience property for plotting stream lines.
        """
        _Zp = self._Z[:, row, :]
        if _Zp.shape[-1] > 1:
            return  0.5 * (_Zp[:, :-1] + _Zp[:, 1:])
        else:
            return  _Zp * np.ones((1, self.nx - 1))

    @property
    def ZT(self):
        """Return elevation of the top of the cells as a 3D grid [nz, ny, nx]."""
        return self.Z[:-1]

    @property
    def ZB(self):
        """Return elevation of the top of the cells as a 3D grid [nz, ny, nx]."""
        return self.Z[1:]

    @property
    def top(self):
        """Return grid top as wanted by mf6 package disv."""
        return self.ZT[0].ravel()

    @property
    def botm(self):
        """Return botm as wanted by mf6 package disv."""
        return self.ZB.reshape(self.nz, self.ncpl)

    @property
    def Zgr(self):
        """Return Z array of all grid points.

        This array has shape (ny+1, nx+1, nz+1) instead of (nz-1, ny, nx)
        of the Z array. The elevation of the internal grid points is computed
        as the aveage of that of the 4 surrounding cells given in Z.

        See Also
        --------
        Plot_grid3d where it is used to plot a 3D wire frame of the grid.

        Notice
        ------
        Zgr is not stored to save memory; it is always computed
        from Z when it is used.
        """
        Zgr = np.zeros((self.nz + 1, self.ny + 1, self.nx + 1))
        msk = np.zeros_like(Zgr)
        Zgr[:, :-1, :-1] += self.Z; msk[:, :-1, :-1] += 1
        Zgr[:,  1:, :-1] += self.Z; msk[:,  1:, :-1] += 1
        Zgr[:, :-1,  1:] += self.Z; msk[:, :-1,  1:] += 1
        Zgr[:,  1:,  1:] += self.Z; msk[:,  1:,  1:] += 1
        return Zgr / msk


    @property
    def Xw(self):
        """Return Xw of nodes in world coordinates."""
        return self.georef_['xw0']\
                + (self.Xv - self.georef_['xm0']) * self.georef_['cos']\
                - (self.Yv - self.georef_['ym0']) * self.georef_['sin']
    @property
    def Yw(self):
        """Return Yw of nodes in world coordinates."""
        return self.georef_['yw0']\
                + (self.Xv - self.georef_['xm0']) * self.georef_['sin']\
                + (self.Yv - self.georef_['ym0']) * self.georef_['cos']

    @property
    def Xmw(self):
        """Return XM of cell centers in world coordinates."""
        return self.georef_['xw0']\
                + (self.Xm - self.georef_['xm0']) * self.georef_['cos']\
                - (self.Ym - self.georef_['ym0']) * self.georef_['sin']
    @property
    def Ymw(self):
        """Return Ym of cell centers in world coordinates."""
        return self.georef_['yw0']\
                + (self.Xm - self.georef_['xm0']) * self.georef_['sin']\
                + (self.Ym - self.georef_['ym0']) * self.georef_['cos']

    @property
    def XMw(self):
        """Return XM of cell centers in world coordinates."""
        return self.georef_['xw0']\
                + (self.XM - self.georef_['xm0']) * self.georef_['cos']\
                - (self.YM - self.georef_['ym0']) * self.georef_['sin']
    @property
    def YMw(self):
        """Return YM of cell centers in world coordinates."""
        return self.georef_['yw0']\
                + (self.XM - self.georef_['xm0']) * self.georef_['sin']\
                + (self.YM - self.georef_['ym0']) * self.georef_['cos']

    """Methods"""


    def world2model(self, xw, yw):
        """Return model coordinates when world coordinates are given.

        Parameters
        ----------
        xw : numpy.ndarray
            world coordinates
        yw : numpy.ndarray
            world coordinates

        Returns
        -------
        x : numpy.ndarray
            model coordinates
        y : numpy.ndarray
            model coordinates
        """
        x = self.georef_['xm0']\
            + (xw - self.georef_['xw0']) * self.georef_['cos']\
            + (yw - self.georef_['yw0']) * self.georef_['sin']
        y = self.georef_['ym0']\
            - (xw - self.georef_['xw0']) * self.georef_['sin']\
            + (yw - self.georef_['yw0']) * self.georef_['cos']
        return x, y


    def m2world(self, x, y):
        """Return world coordinates when model coordinates are given.

        Parameters
        ----------
        x : numpy.ndarray
            model coordinatesa
        y : numpy.ndarray
            model coordinates

        Returns
        -------
        xw : numpy.ndarray
            world coordinates
        yw : numpy.ndarray
            world coordinates
        """
        xw = self.georef_['xw0']\
            + (x - self.georef_['xm0']) * self.georef_['cos']\
            - (y - self.georef_['ym0']) * self.georef_['sin']
        yw = self.georef_['yw0']\
            + (x - self.georef_['xm0']) * self.georef_['sin']\
            + (y - self.georef_['ym0']) * self.georef_['cos']
        return xw, yw


    def ixyz(self, xw, yw, zp, order=None, world=True, left=intNaN, right=intNaN):
        """Return indices ix, iy,, iz of cell where points (xp, yp, zp) are in.

        Notice: all three inputs are required because z varies per (ix,iy)
        It may be simplified when gr.full = False (uniform layers)

        The indices always resulting in values beteen 0 and the number of cells
        in the respective row, column and layer minus 1. These are legal
        indices. To remove points outside the model, fiter them on their
        coordinates.

        Parameters
        ----------
            xp : np.ndarray of x-grid coordinates
            yp : np.ndarray of y-grid coordinates
            zp : np.ndarray of z-grid coordinates
            order : order of returned indices tuple
                None  : [ I ] global array index
                'CRL' = (Col, Row Layer), same order as input (xp, yp, zp)
                'RCL' = (Row, Col, Layer)
                'LRC' = (Layer, Row, Col) (as wanted by Modflow packaeges)
            world : [True]|False
                if True (default) then world coordinates else model coordinates

        Returns
        -------
        typle of cell indices, see descirption under 'order'
        In the tuple are:
        ix : ndarray of int
            always between and including 0 and nx-1
        iy : ndarray of int
            always between and including 0 and ny-1
        iz : ndarray of int
            always between and including 0 and nz-1

        @TO 161031
        """
        xw = np.array(xw, ndmin=1, dtype=float)
        yw = np.array(yw, ndmin=1, dtype=float)
        zp = np.array(zp, ndmin=1, dtype=float)

        if world:
            xp, yp = self.world2model(xw, yw)
        else:
            xp, yp = xw, yw

        if not (np.all(xp.shape == yp.shape) and np.all(xp.shape == zp.shape)):
            print("Shapes of xp={}, yp={} and zp={} must match".\
                  format(xp.shape, yp.shape, zp.shape))
            raise ValueError("shapes don't match")

        ix = self.ix(xp)
        iy = self.iy(yp)
        iz = np.zeros(ix.shape, dtype=int) + intNaN

        Lu = ix >= 0
        Lv = iy >= 0

        L = AND(Lu, Lv)
        if self.full == False:
            iz[L] = index( -zp[L], -self.z, left, right)
        else:
            for i in range(len(zp)):
                if L[i] == True:
                    z = self._Z[:, iy[i], ix[i]]
                    if AND(zp[i] <= z[0], zp[i] >= z[-1]):
                        iz[i] = index(-zp[i], -z, left, right)
        iz = np.fmin(iz, self.nz - 1)
        if order is None:
            return (self.nx * self.ny * iz + self.nx * iy + ix)
        elif order=='CRL':
            return np.vstack((ix, iy, iz)).T
        elif order == 'LRC':
            return np.vstack((iz, iy, ix)).T
        elif order == 'RCL':
            return np.vstack((iy, ix, iz)).T
        else:
            print("Use 'RCL', 'LRC' or 'RCL' for order.")
            raise ValueError("Unrecognied input, use RCL, LRC or RCL")


    def norm_grid(self):
        """Generate a normalized grid from current grid."""
        up = np.arange(self.nx + 1, dtype=float)
        vp = np.arange(self.ny + 1, dtype=float)
        wp = np.arange(self.nz + 1, dtype=float)
        return Grid(up, vp, wp)


    def outer(self, xy0, xy1, world=True):
        """Return outer product of nodes with respect of  line from xy0 to xy1.

        To find which points that lie on one side of a line.

        Negative values lie to the left of the line and positive to the right.

        Example
        -------
        left  = gr.outer(xy0, xy1) < 0
        right = gr.outer(xy0, xy1) > 0

        Parameters
        ----------
        xy0: array_like 2 floats
            first point of line from xy0 to xy1.
        xy1: array_like of 2 floats
            second point of line from xy0 to xy1.
        world : boolean [True]
            use world or model coordinates.
        """
        x0, y0 = xy0
        x1, y1 = xy1
        if world:
            return (x1 - x0) * (self.YMw - y0) - (y1 - y0) * (self.XMw - x0)
        else:
            return (x1 - x0) * (self.YM  - y0) - (y1 - y0) * (self.XM  - x0)


    def ixyz2global_index(self, ix, iy, iz):
        """Return global cell index given ix, iy and iz.

        Rarameters
        ----------
        ix : numpy.ndarray of dtype=int
            column indices
        iy: numpy.ndarray of dtype=int
            row indices
        iz: numpy.ndarray of dtype=int
            layer indices

        Returns
        -------
        I : numpy.ndarray of int
            global cell indices

        Notice
        ------
        shape of ix, iy and iz must be the same.
        """
        ix = np.array(ix, dtype=int)
        iy = np.array(iy, dtype=int)
        iz = np.array(iz, dtype=int)

        if np.any(ix.shape != iy.shape) or np.any(ix.shape != iz.shape):
            print("The shapes of ix={}, iy={} and iz={} must be the same".\
                  format(ix.shape, iy.shape, iz.shape))
            raise ValueError("Shapes don't match.")

        Iglob = np.zeros(ix.shape, dtype=int) + intNaN

        L = AND(ix >= 0, ix <= self.nx,
                iy >= 0, iy <= self.ny,
                iz >= 0, iz <= self.nx)

        Iglob[L] = iz[L] * self.nx * self.ny + iy[L] * self.nx + ix[L]
        return   Iglob


    def xyz2global_index(self, xw, yw, zp=None, iLay=None):
        """Return global cell indices given points with world coords xw, yw zp.

        The shapes of xw, yw and zp must be the same.

        Rarameters
        ----------
        xw : arraylike
            world coordinates
        yw : arraylike
            world coordinates
        zp : arraylayk (grid or world coordinates (is the same))
             you can use ilay if the layer number is to be specified.
        iLay: arraylike
            layer number of points, default is 0

        Returns
        -------
        I : ndarray of dtype=int
            global index into grid for given points
        """
        if np.isscalar(xw):
            xw = [xw]
            yw = [yw]
            if zp is not None:
                zp = [zp]
            else:
                if iLay is not None:
                    iLay = [iLay]
                else:
                    iLay = np.zeros(len(xw), dtype=int)
        xw = np.array(xw, dtype=float).ravel()
        yw = np.array(yw, dtype=float).ravel()
        if zp is not None:
            zp = np.array(zp, dtype=float).ravel()
        elif iLay is not None:
            iLay = np.array(iLay, dtype=int).ravel()
        else:
            iLay = np.zeros_like(xw, dtype=int)

        assert np.all(xw.shape == yw.shape), "shape of xw must equal that of yw"
        if zp is not None:
            assert np.all(xw.shape == zp.shape), "shape of xw must equal that of zp"
        elif iLay is not None:
            assert np.all(xw.shape == iLay.shape), "shape of xw must equal that of iLay"
        else:
            pass


        xp, yp = self.world2model(xw, yw)
        ix = np.ones(xp.shape, dtype=int) * intNaN
        iy = np.ones(yp.shape, dtype=int) * intNaN
        if zp is None:
            zp = self.ZM[iLay, iy, ix] # mid of layers  defined by iLay
        I = self.inside(xp, yp, zp)
        iz = np.ones(zp.shape, dtype=int) * intNaN

        ixyz = np.vstack((ix, iy, iz)).T
        ixyz[I] = self.ixyz(xp[I], yp[I], zp[I], order='CRL')
        return self.ixyz2global_index(ixyz[:, 0], ixyz[:, 1], ixyz[:, 2])


    def U(self, xp):
        """Return normalized coordinate for xp.

        Normalized coordinate runs from 0 to nx

        Returns
        -------
        U  :  ndarray, dtype=float
            normalized coordinates within column iu
        iu :  ndarray, dtype = int
            column iu in which xp resides
        """
        xp = np.array(xp, ndmin=1, dtype=float).ravel()
        up = self.up(xp)
        iu = np.fmin( np.array(up, dtype=int), self.nx - 1)
        U = up - iu
        return U, iu


    def V(self, yp):
        """Return normalized coordinate for yp.

        Normalized coordinate runs from 0 to ny+1

        Returns
        -------
        V  :  ndarray, dtype=float
            normalized coordinates within row iv
        iv :  ndarray, dtype = int
            row iv in which yp resides
        """
        yp = np.array(yp, ndmin=1, dtype=float).ravel()
        vp = self.vp(yp)
        iv = np.fmin( np.array(vp, dtype=int), self.ny - 1)
        V = vp - iv
        return V, iv


    def W(self, xp, yp, zp):
        """Return normalized coordinate for zp.

        xp and yp are needed because z may vary for each ix,iy combination.
        """
        xp = np.array(xp, ndmin=1, dtype=float).ravel()
        yp = np.array(yp, ndmin=1, dtype=float).ravel()
        zp = np.array(zp, ndmin=1, dtype=float).ravel()

        if not ( np.all(xp.shape == yp.shape) and np.all(xp.shape == zp.shape) ):
            print("Shapes of xp={}, yp={} and zp{} must match".\
                  format(xp.shape, yp.shape, zp.shape))
            raise ValueError("Shapes don't match.")

        wp = self.wp(xp, yp, zp)
        iw = np.fmin( np.array(wp, dtype=int), self.nz - 1)
        W = wp - iw
        return W, iw


    def up(self, xp, left=None, right=None):
        """Return normallized coordinate along x axis (0 to nx+1).

        Any point outside the range will projected onto the left or right edge.
        So up<0 -> 0 and up>nx -> nx.
        """
        xp = np.array(xp, ndmin=1, dtype=float).ravel()
        up = np.interp( xp, self._x, np.arange(self.nx+1),
                       left=left, right=right)
        return up # up will always be between 0 and nx + 1


    def vp(self, yp, left=None, right=None):
        """Return normalized coordinate along y axis (0 to ny+1).

        Any point outside the range will projected onto the left or right edge.
        So up<0 -> 0 and up>ny -> ny
        """
        yp = np.array(yp, ndmin=1, dtype=float).ravel()
        vp = np.interp(-yp,-self._y, np.arange(self.ny+1),
                       left=left, right=right)
        return vp


    def wp(self, xp, yp, zp, left=None, right=None):
        """Return normalized coordinate along z axis (0 to nz+1)."""
        xp = np.array(xp, ndmin=1, dtype=float).ravel()
        yp = np.array(yp, ndmin=1, dtype=float).ravel()
        zp = np.array(zp, ndmin=1, dtype=float).ravel()

        if not (np.all(xp.shape == yp.shape) and np.all(xp.shape == zp.shape)):
            print("Shapes of xp={}, yp={} and zp{} must match".\
                  format(xp.shape, yp.shape, zp.shape))
            raise ValueError("Shapes don't match.")

        iu = self.ix(xp) # always between 0 and nx - 1
        iv = self.iy(yp) # always between 0 and nz - 1

        Lu = AND(xp >= self._x[ 0], xp <= self._x[-1])
        Lv = AND(yp >= self._y[-1], yp <= self._y[ 0])

        L = AND(Lu, Lv)
        wp = np.zeros(zp.shape, dtype=float) * np.NaN
        if self.full == False:
            wp = np.interp( -zp, -self.z, np.arange(self.nz+1),
                           left=left, right=right)
        else:
            for i in range(len(L)):
                if L[i] == True:
                    wp[i] = np.interp( -zp[i], -self._Z[:, iv[i], iu[i]],\
                        np.arange(self.nz+1) ,\
                            left=left, right=right)
        return wp # always between 0 and nz


    def xyz2uvw(self, xp, yp, zp, left=np.NaN, right=np.NaN):
        """Return normalized coordinate for xp, yp, zp.

        Normalized coordinate runs from 0 to nx+1,0 to ny+1 and 0 to nz+1

        Parameters
        ----------
        xp : ndarray, x-coordinates in grid
        yp : ndarray, y-coordinates in grid
        zp : ndarray, z-coordinates in grid

        Returns
        -------
        up  :  ndarray, dtype=float, 0 to nx+1
            normalized coordinate in column of xp
        vp  :  ndarray, dtype=float, 0 to ny+1
            normalized coordinates within row yp
        wp :  ndarray, dtype = float, 0 to nz+1
            normalized coordinates within col-row of zp

        @ TO161031
        """
        xp = np.array(xp, ndmin=1, dtype=float).ravel()
        yp = np.array(yp, ndmin=1, dtype=float).ravel()
        zp = np.array(zp, ndmin=1, dtype=float).ravel()

        if not (np.all(xp.shape == yp.shape) and np.all(xp.shape == zp.shape)):
            print("xp.shape={} and yp.shape={} and or zp.shape={} do not match.".\
                  format(xp.shape, yp.shape, zp.shape))
            raise ValueError("Shapes don't match.")

        #@@@
        Ic = points_in_cells(self, points)
        verts = self.vertices[self.cell2d[Ic][-1]]
        xq, yq = self.vertices[verts]

        up = self.up(xp) # always between 0 and nx
        vp = self.vp(yp) # always between 0 and ny
        wp = self.wp(xp, yp, zp, left, right) # always between 0 and ny
        return up, vp, wp


    def uvw2xyz(self, up,vp,wp):
        """Concert from nomalized coords (up, vp, wp) to grid coords (xp, yp, zp).

        Whatever the normalized coordinates are,
        the resulting grid coordinates will always be between
        and including the extreme grid coordinates. That is
        points outside the model will be project on its outer boundaries.

        This prevents NaNs. To eliminate such points filter the input.
        """
        def LOCAL(up, nx):
            """Return local coordinates u,iy given normalized coordinate up."""
            iu = np.fmin( np.array(up, dtype=int), nx - 1)
            u = up - iu
            return u, iu

        # Notice that underscores are used to prevent confusion with pdb cmds
        up = np.array(up, ndmin=1, dtype=float).ravel()
        vp = np.array(vp, ndmin=1, dtype=float).ravel()
        wp = np.array(wp, ndmin=1, dtype=float).ravel()

        if not (np.all(up.shape == vp.shape) and np.all(up.shape==wp.shape)):
            print("up.shape={} and vp.shape={} and or wp.shape={} do not match".\
                  format(up.shape, vp.shape, wp.shape))
            raise ValueError("Shapes don't match.")

        U, iu = LOCAL(up, self.nx)
        V, iv = LOCAL(vp, self.ny)
        W, iw = LOCAL(wp, self.nz)

        L = AND(up >= 0, up <= self.nx,\
                vp >= 0, vp <= self.ny,\
                wp >= 0, wp <= self.nz)

        xp = np.zeros(up.shape, dtype=float) * np.NaN
        yp = np.zeros(vp.shape, dtype=float) * np.NaN
        zp = np.zeros(wp.shape, dtype=float) * np.NaN

        xp[L] = self._x[iu[L]] +  U[L] * self.dx[iu[L]]
        yp[L] = self._y[iv[L]] -  V[L] * self.dy[iv[L]]
        if self.full == False:
            zp[L] = self.z[iw[L]] - W[L] * self.dz[iw[L]]
        else:
            for i in range(len(iw)):
                if L[i]==True:
                    zp[i] = self._Z[iw[i], iv[i], iu[i]] - W[i] *\
                             self.DZ[iw[i], iv[i], iu[i]]
        return xp, yp, zp


    def inside(self, xp, yp, zp):
        """Return logical array telling which of the points are inside the grid.

        xp, yp and zp must be ndarrays of equal shape of type
        """
        
        Ic = points_in_cells(self.Xv, self.Yv, np.vstack(((xp, yp))).T)
        Ir, Ic = self.Ic2rc(Ic)
        return [True if ic >=0 and self.Z[0, ir, ic] >= z_ and self.Z[-1, ir, ic] <= zp else False for ic, ir, ic, z_ in zip(Ic, Ir, Ic, zp)]

        
    def asdict(self):
        """Return a dict containing the grid lines that can be plotted or saved as a shapfile.

        Returns
        -------
        dict {'name': 'mfGrid', 'x': x, 'y':y, 'xw': xw, 'yw': yw}

        to save it as a shapefile use dict2shp in module shapefile.shapetools

        from shapefile import dict2shp

        dict2shp(gr.asdict, shapefilename, xy=('xw', 'yw'), shapetype=sf.LINE)
        dict2shp(gr.asdict, shapefilename, xy=('x', 'y'), shapetype=sf.LINE)

        TO 180607
        """
        x, y = [], []
        for i, x_ in enumerate(self.x):
            if i%2 == 0:
                y.append(self.y[ 0])
                y.append(self.y[-1])
            else:
                y.append(self.y[-1])
                y.append(self.y[ 0])
            x.append(x_)
            x.append(x_)

        #x.append(np.nan)
        #y.append(np.nan)

        for i, y_ in enumerate(self.y):
            if i%2 == 0:
                x.append(self.x[ 0])
                x.append(self.x[-1])
            else:
                x.append(self.x[-1])
                x.append(self.x[ 0])
            y.append(y_)
            y.append(y_)

        x = np.array(x)
        y = np.array(y)
        xw, yw = self.m2world(x, y)

        return {'name': 'mfGrid',
                'x' : x,  'y' : y,
                'xw': xw, 'yw': yw}


    def interp_map(self, surf):
        """Return array of (self.ny, self.nx) with interpolated surf values.

        Returns the map (called surf) cell values. `surf['z'] is an array (my, mx)
        in which grid world coordinates surf['x'], surf['y'] lie. So surf['x']
        and surf['y'] are aligned with world axes.
        This interpolater pickes the surf['z'] value in which the grid coordinate
        lies. No smoothing is done.

        This function is useful to for instance get elevations, when they
        are given as a high-resolution array aligned with world coordinates.

        Parameters
        ----------
        surf : dict {'x', 'y', 'z'}
           x : x_coordinates (world coordinates)
           y : y_coordinates (world coordinates)

           z : value array of shape (len(y), len(x))

            make sure x and y cover total model to prevent NaNs.

        TO 180609

        """
        x, y = surf['x'], surf['y']

        sign = 1 if y[-1] > y[0] else -1

        ix = np.interp(       self.Xmw.ravel(),        x, np.arange(len(x)))
        iy = np.interp(sign * self.Ymw.ravel(), sign * y, np.arange(len(y)))
        ix = np.asarray(ix, dtype=int)
        iy = np.asarray(iy, dtype=int)
        return surf['z'][[iy], [ix]].reshape((self.ny, self.nx))


    def plot_grid(self, ax=None, row=None, col=None, world=False, **kwargs):
        """Plot the grid in model or world coordinates.

        Parameters
        ----------
        ax : plt.Axes
            axes to plot on.
        row: int
            plot vertical (zx) along specified row
        col: int
            plot vertical (zy) grid along specified column
        other kwargs:
            are passed on as plt.plot(... **kwargs)
        """
        if ax is None:
            fig, ax = plt.subplots()
            ax. set_title('fdm grid')
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
        else:
            pass

        if not row is None:
            if world:
                X = self.XW[ :, row, :]
            else:
                X = self.XV[:, row, :]

            z = self.Zgr[:, row, :].copy()
            x = X.copy()
            x[1::2, :] = x[1::2, ::-1]
            z[1::2, :] = z[1::2, ::-1]
            ax.plot(x.ravel(), z.ravel(), **kwargs)

            z = self.Zgr[:, row, :].copy().T
            x = X.copy().T
            x[1::2, :] = x[1::2, ::-1]
            z[1::2, :] = z[1::2, ::-1]
            ax.plot(x.ravel(), z.ravel(), **kwargs)

        elif not col is None:
            if world:
                Y = self.YW[:, :, col]
            else:
                Y = self.YV[:, :, col]

            z = self.Zgr[:, : col].copy()
            y = Y.copy()
            y[1::2, :] = y[1::2, ::-1]
            z[1::2, :] = z[1::2, ::-1]
            ax.plot(y.ravel(), z.ravel(), **kwargs)

            z = self.Zgr[:, : col].copy().T
            y = Y.copy().T
            y[1::2, :] = y[1::2, ::-1]
            z[1::2, :] = z[1::2, ::-1]
            ax.plot(y.ravel(), z.ravel(), **kwargs)

        else:
            if world:
                X = self.Xw
                Y = self.Yw
            else:
                X = self.Xv
                Y = self.Yv

            xr = X.copy(); xr[1::2, :] = xr[1::2, ::-1]
            yr = Y.copy(); yr[1::2, :] = yr[1::2, ::-1]
            ax.plot(xr.ravel(), yr.ravel(), **kwargs)

            xc = X.copy().T; xc[1::2, :] = xc[1::2, ::-1]
            yc = Y.copy().T; yc[1::2, :] = yc[1::2, ::-1]
            ax.plot(xc.ravel(), yc.ravel(), **kwargs)

        return ax


    def plot_grid_world(self, **kwargs):
        """Plot the grid in world coordinates."""
        self.plot_grid(world=True, **kwargs)

    def plot_grid_model(self, **kwargs):
        """Plot the grid in model coordinates."""
        self.plot_grid(world=False, **kwargs)


    def plot_ugrid(self,color='grb', axes=None, **kwargs):
        """Plot the normlized grid."""
        color = color[0:3] if len(color)>3\
                     else color + (3 - len(color)) * color[-1]

        if axes is None:
            axes = plt.figure().add_subplot(211, projection='3d')

        if isinstance(axes, Axes3D):
            #plot 3D
            axes.set_xlim([0, self.nx])
            axes.set_ylim([0, self.ny])
            axes.set_zlim([0, self.nz])
            axes.yaxis_inverted()
            axes.zaxis_inverted()
            for iz in range(self.nz + 1):
                for ix in range(self.nx + 1):
                    plt.plot([ix, ix], [0, self.ny],
                             [iz, iz], color[0], **kwargs)
                for iy in range(self.ny + 1):
                    plt.plot([0, self.nx], [iy, iy],
                             [iz, iz], color[1], **kwargs)
            for ix in range(self.nx + 1):
                for iy in range(self.ny + 1):
                    plt.plot([ix, ix], [iy, iy],
                             [0, self.nz], color[2], **kwargs)
        elif isinstance(axes, Axes):
            # plot 2D
            axes.invert_yaxis()
            plt.setp(axes, xlabel='u [-]', ylabel='v [-]',
                     title='normalized fdm grid',
                     xlim=(0, self.nx), ylim=(self.ny, 0))
            for u in self.up(self._x):
                plt.plot([u, u], [0, self.ny], color[0], **kwargs)
            for v in self.vp(self._y):
                plt.plot([0,self.nx], [v, v], color[1], **kwargs)
        else:
            print("axes must be of type Axes or Axes3D not {}".format(type(axes)))
            raise ValueError("Incorrect axis type.")
        return axes


    def plot_us(self, color=None, axes=None):
        """Plot a vertical cross section along any row in normalized grid."""
        if axes is None:
            axes = plt.gca()
            plt.setp(axes, xlabel='u [-]', ylabel='v [-]',
                     title='XS normalized grid',
                     xlim=(0, self.nx+1), ylim=(0, self.ny+1),
                     ydir='reverse' )
        if color is None:
            color = 'c'
        for v_ in np.arange(self.nz + 1):
            plt.plot( [0, self.nx], [v_, v_], color)
        for u_ in np.arange(self.nx+1):
            plt.plot([u_, u_], [0, self.ny], color)
        return axes

    def plot_grid3d(self, color='ccc', world=False, **kwargs):
        """Plot 3D wire frame of grid.

        Parameters
        ----------
        color : str
            a trhee-character string telling the color of each axis, like
            'ccc', 'rgb' etc.

        Kwargs
        ------
        'world' : boolean
            use world coordinates world=True or model coordinags world=False
        'ax' : Axis
            axis to plot onto, if None or not present then a new axes is generated.
        any argements that can be passed on to plt.plot(...)
        color : str, color of the wire frame lines. One to three characters
            that idicate the wire colors in the, x, y and z directions.
            If less than 3 charaters are used, the last char is used for the
            missing colors.
            examples: 'c', 'cmc', 'cb'
            Default: 'ccc'
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title('3D plot of the model grid')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')

        def getxy(world, perturb):
            if world:
                return (self.Xw.transpose(perturb).copy(),
                        self.Yw.transpose(perturb).copy(),
                        self.Zgr.transpose(perturb).copy())
            else:
                return (self.Xv.transpose(perturb).copy(),
                        self.Yv.transpose(perturb).copy(),
                        self.Zgr.transpose(perturb).copy())

        if len(color)<3:
            color[-1:3] = color[-1]

        X, Y, Z = getxy(world, (0, 1, 2))

        X[:, 1::2, :] = X[:, 1::2, ::-1]
        Y[:, 1::2, :] = Y[:, 1::2, ::-1]
        Z[:, 1::2, :] = Z[:, 1::2, ::-1]
        for  iz in range(self.nz + 1):
            ax.plot(X[iz, :, :].ravel(),
                     Y[iz, :, :].ravel(),
                     Z[iz, :, :].ravel(), color=color[0], **kwargs)

        X, Y, Z = getxy(world, (0, 2, 1))

        X[:, 1::2, :] = X[:, 1::2, ::-1]
        Y[:, 1::2, :] = Y[:, 1::2, ::-1]
        Z[:, 1::2, :] = Z[:, 1::2, ::-1]
        for iz in range(self.nz + 1):
            ax.plot(X[iz, :, :].ravel(),
                     Y[iz, :, :].ravel(),
                     Z[iz, :, :].ravel(), color[1], **kwargs)

        X, Y, Z = getxy(world, (2, 1, 0))

        X[:, 1::2, :] = X[:, 1::2, ::-1]
        Y[:, 1::2, :] = Y[:, 1::2, ::-1]
        Z[:, 1::2, :] = Z[:, 1::2, ::-1]
        for  iz in range(X.shape[0]):
            ax.plot( X[iz, :, :].ravel(),
                     Y[iz, :, :].ravel(),
                     Z[iz, :, :].ravel(), color[2], **kwargs)
        return ax


    def show(self, A, filled=False, **kwargs):
        """Contour array A and show it.

        Same as method gr.contour().

        Parameters
        ----------
        A : np.ndarray or 3D np.array(nz, ny, nx) or list of 2D np.arrays(ny, nx)
            Array(s) to be plotted. Their shape is (self.ny, self.nx)
        labels: list of labels for the arrays (for the legend).
        filled : bool
            if True, ax.contourf is used if False, ax.contour is used

        Additional kwargs
        -----------------
        kwargs ['title', 'xlabel', 'ylabel', 'xlim', 'ylim',
        'xscale', 'yscale'] --> ax

        kwargs ['fmt', 'fz',, 'inline', 'inline_spacing'] --> clabel
        will be passed on to plt.contourf

        all other kwargs --> ax.contour, ax.contourf resp.

        See doc of ax.contour for details
        """
        return self.contour(A, filled=filled, **kwargs)


    def imshow(self, A, **kwargs):
        """Show array A using imshow.

        This implies that axes are cell numbers !!

        Parameters
        ----------
        A: ndarray of size self.ny, self.nx
            the array to be shown
        kwargs: additional kwargs
            xlabel, ylabel, title, xlim, ylim, set_inches
            addtional kwargs are passed on to ax.imshow
        """
        fig, ax = plt.subplots()

        ax.set_xlabel(kwargs.pop('xlabel', 'ix'))
        ax.set_ylabel(kwargs.pop('ylabel', 'iy'))
        ax.set_title(kwargs.pop('title', 'gr.imshow()'))
        if 'xlim' in kwargs: ax.set_xlim(kwargs.pop('xlim'))
        if 'ylim' in kwargs: ax.set_ylim(kwargs.pop('xlim'))
        if 'size_inches' in kwargs: fig.set_size_inches(kwargs.pop('size_inches'))
        ax.grid()
        ax.imshow(A)  #, extent=(*self.xm[[0, -1]], *self.ym[[-1, 0]]), **kwargs)


    def contour(self, A, filled=False,  **kwargs):
        """Contour array A and show it.

        Parameters
        ----------
        A : np.ndarray or 3D np.array(nz, ny, nx) or list of 2D np.arrays(ny, nx)
            Array(s) to be plotted. Their shape is (self.ny, self.nx)
        labels: list of labels for the arrays (for the legend).
        filled : bool
            if True, ax.contourf is used if False, ax.contour is used

        Additional kwargs
        -----------------
            kwargs ['title', 'xlabel', 'ylabel', 'xlim', 'ylim',
            'xscale', 'yscale'] --> ax

            kwargs ['fmt', 'fz',, 'inline', 'inline_spacing'] --> clabel
            will be passed on to plt.contourf

            all other kwargs --> ax.contour, ax.contourf resp.

            See doc of ax.contour for details
        """
        if A.ndim < 3:
            M = [A]
        else:
            M = A

        lblkw = {'fmt':'%.1f', 'fz': 8, 'inline': 1, 'inline_spacing': 1.0}
        for k in lblkw:
            lblkw[k] = kwargs.pop(k, lblkw[k])
        lblkw['fontsize'] = lblkw['fz']

        ax = kwargs.pop('ax', None)
        if ax is None:
            fig, ax = plt.subplots()

        ax.set_title(kwargs.pop('title', 'Showing contour of given 2D array'))
        ax.set_xlabel(kwargs.pop('xlabel', 'x [m]'))
        ax.set_ylabel(kwargs.pop('ylabel', 'y [m]'))
        xlim   = kwargs.pop('xlim',   None)
        ylim   = kwargs.pop('ylim',   None)
        xscale = kwargs.pop('xscale', None)
        yscale = kwargs.pop('yscale', None)

        labels   = kwargs.pop('labels', np.arange(len(M), dtype=int))
        fontsize = kwargs.pop('fontsize', None)
        if fontsize is None:
            fontsize = kwargs.pop('fz', 8)

        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)
        if xscale: ax.set_xscale(xscale)
        if yscale: ax.set_yscale(yscale)
        ax.grid()

        if filled:
            cfun = ax.contourf
        else:
            cfun = ax.contour

        for B, label in zip(M, labels):
            cs  = cfun(self.xm, self.ym, B, **kwargs)
            plt.clabel(cs, **lblkw)
            if filled:
                plt.colorbar(cs, ax=ax)

        plt.show()
        return



    def const(self, u, vdis=True, dtype=float):
        """Return an ndarray of gr.shape with value u.

        U can be a scaler or a vector in which case the values of u are used
        to set the layers of the resulting array with the corresponding value
        of u. The number of resulting layers is either equal to the length of u
        or to nlay if lay is True.
        Lay works when u is scalar.
        If u is any sequence, then u is first converted to an array and then
        the ravel() is used. Once u is always used as a vector that will
        be turned into the (downward) z-direction (the highest index is the
        lowest layer of the grid in space).

        Parameters
        ----------
        u: float or sequence
            value or layer values to be converted in 3D arrays (nlay, nrow, nocl) if vdis is False,
            or 2D arrays (nlay, nrow * ncol) if vdis is True
        vids: bool
            if True output as dimanded by vdis (ilay, icell)
            if False output as demanded by dis (ilay, irow, icol)

        Examples
        --------
        U = gr.const(3.), U = gr.const(np.random.randn((1, 1, gr.nz)))
        U = gr.const(np.random.randn(gr.nz))

        TO 161007
        """
        if isinstance(u, (np.ndarray, list, tuple)):
            u = np.asarray(u, dtype=dtype).ravel()[:, np.newaxis, np.newaxis]
            nz = len(u)
        else:
            nz = self.nz

        if isinstance(u, (float, int)):
            out = u * np.ones((nz, self.ny, self.nx), dtype=dtype)
        elif isinstance(u, np.ndarray):
            out = u * np.ones((1, self.ny, self.nx), dtype=dtype)

        if vdis:
            out = np.array([o.ravel() for o in out])

        return out


    def ckD2k(self, c, kD):
        """Return kh and kv given c and kD.

        Parameters
        ----------
        c : sequence or array of aquitard resistance values
        kD: sequence or array of aquifer  avlues
        """
        if len(c) + len(kD) != self.nz:
            assert len(c) + len(kD) == self.nz and \
                   (len(c) == len(kD) or len(kD) == len(c) + 1), \
                "len(c) + len(kD) must equal nz and " +\
                "len(c) must equal len(kD) or " +\
                "len(kD) must equal len len(c) + 1"

        kh = self.const(0)
        kv = self.const(0)
        D  = self.DZ

        if len(c) == len(kD): # then c on top
            for i in range(0, len(c)):
                j = 2 * i
                kv[j] = D[j] / c[i]
                kh[j] = D[j] / c[i]
            for i in range(0, len(kD)):
                j = 2 * i + 1
                kv[j] = kD[i] / D[j]
                kh[j] = kD[i] / D[j]
        else:  # kD on top
            for i in range(0, len(kD)):
                j = 2 * i
                kv[j] = kD[i] / D[j]
                kh[j] = kD[i] / D[j]
            for i in range(0, len(c)):
                j = 2 * i + 1
                kv[j] = D[j] / c[i]
                kh[j] = D[j] / c[i]

        return kh, kv

    def s2ss(self, S):
        """Return ss array given S.

        Parameters
        ----------
        S : np.ndarray of floats
            sequence of storage coefficients of all layers [-]
        """
        assert len(S) == self.nz, "len(S) must equal numbe of layers"

        return self.const(S) / self.DZ


    def well(self, x, y, z, Q, kh=None, order='LRC', world=True):
        """Return well in grid as lRCQ.

        Parameters
        ----------
        x, y: floats
            well x and y location
        z : tuple
            well screen top and bottom
        Q : float
            extraction of well
        kh: 3D array with horizontal conductivities
        order : str: LRC | RLC | CRL order of index columns
        world : [True] | False
            whether the coordinates are in world coordinate or model coordinates

        Returns
        -------
        tuple LRC, Q
            LRC : Iz, Iy, Ix [nx3] cell indeces that screen penetrates
            Q is the injection per cell propertional to its transmissivity
        """
        # make sure len x an len y are the same as len z
        assert np.isscalar(x), 'x must be a float'
        assert np.isscalar(y), 'y must be a float'
        assert len(z) == 2, 'len(z) must be2'

        x *= np.ones_like(z)
        y *= np.ones_like(z)

        LRC = self.ixyz(x, y, z, order, world)
        I = self.I(LRC)
        n = self.ny * self.nx
        I  = np.arange(I[0], I[-1] + 1, n, dtype=int)
        kD = kh.ravel()[I] * self.DZ.ravel()[I] # cell transmissivty
        Q  = kD / np.sum(kD) * Q # cell flows
        return I, Q


    def quivdata(self, Out, iz=0):
        """Return vector data for plotting velocity vectors.

        Takes Qx from fdm3 and returns the tuple X, Y, U, V containing
        the velocity vectors in the xy plane at the center of the cells
        of the chosen layer for plotting them with matplotlib.pyplot's quiver()

        Qx --- field in named tuple returned by fdm3.
        x  --- grid line coordinates.
        y  --- grid line coordinates.
        iz --- layer for which vectors are computed (default 0)

        """
        ny = self.ny
        nx = self.nx

        X, Y = np.meshgrid(self.xm, self.ym) # coordinates of cell centers

        # Flows at cell centers
        U = np.concatenate((Out.Qx[iz, :, 0].reshape((1, ny, 1)), \
                            0.5 * (Out.Qx[iz, :, :-1].reshape((1, ny, nx-2)) +\
                                   Out.Qx[iz, :, 1: ].reshape((1, ny, nx-2))), \
                            Out.Qx[iz, :, -1].reshape((1, ny, 1))), axis=2).reshape((ny,nx))
        V = np.concatenate((Out.Qy[iz, 0, : ].reshape((1, 1, nx)), \
                            0.5 * (Out.Qy[iz, :-1, :].reshape((1, ny-2, nx)) +\
                                   Out.Qy[iz, 1:,  :].reshape((1, ny-2, nx))), \
                            Out.Qy[iz, -1, :].reshape((1, 1, nx))), axis=1).reshape((ny,nx))
        return X, Y, U, V


    def inblock(self, xx=None, yy=None, zz=None):
        """Return logical array denoding cells in given block."""
        if xx == None:
            Lx = self.const(0) > -np.inf # alsways false
        else:
            Lx = np.logical_and(self.XM >= min(xx), self.XM <= max(xx))
        if yy== None:
            Ly = self.const(0) > -np.inf  # always false
        else:
            Ly = np.logical_and(self.YM >= min(yy), self.YM <= max(yy))
        if zz == None:
            Lz = self.const(0) > -np.inf  # always false
        else:
            Lz = np.logical_and(self.ZM >= min(zz), self.ZM <= max(zz))
            #L = np.logical_and( np.logical_and(Lx, Ly), Lz)
        return np.logical_and( np.logical_and(Lx, Ly), Lz)


    def from_shape(self, rdr, var, out=None, dtype=float, row=None):
        """Return array filled by value indicated by fieldnm var.

        Always world coordinates are used.

        Parameters
        ----------
        rdr : shapefile.Reader
        var: str
            name ov variable to pick form recrod to fill array
        out: 2D ndarray [ny, nx]
            array to fill inplace
            out is also returned
        dtype: numpy.dtype
            dtype to use if new array is generated. (not for Out)
        row: int
            row number if not None, the the zx cross section is assumed
            both for the grid as for the coordinate in the shape.
            The x is then local grid x-coordinate and the z the z as usual.
        """
        fldNames = [p[0] for p in rdr.fields][1:]  # first item of each field.

        idx = fldNames.index(var)

        if row is None:
            if out is None:
                out = np.zeros((self.ny, self.nx), dtype=dtype) * np.nan

            for sr in rdr.shapeRecords():
                val = sr.record[idx]
                pgon = np.array(sr.shape.points)
                I = inpoly(self.Xmw, self.Ymw, pgon) # world coordinates
                out[I] = val
            return out
        else:
            if out is None:
                out = np.zeros((self.nz, self.nx), dtype=dtype) * np.nan

            for sr in rdr.shapeRecords():
                val = sr.record[idx]
                pgon = np.array(sr.shape.points)
                I = inpoly(self.Xm[:, row, :], self.Zm[:, row, :])
                out[I] = val
            return out


    def extend (self, dx, dy, dz, nx=(10, 10), ny=(0, 10), nz=(0, 10)):
        """Return new grid extended according to parameters.

        Parameters
        ----------
        dx : tuple of floats
            dx[0] = extension to the left in m
            dx[1] = extension to the right in m
        dy : tuple of floats
            dy[0] = extensio to the front in m
            dy[1] = extension to the back in m
        dz : tuple of floats
            dz[0] = extension on top in m
            dz[-1] = extension to the bottom in m
        nx : tuple of ints
            nx[0] number of columns prepended
            nx[1] number of columns appended
        ny : tuple of ints
            ny[0] = number of rows prepended
            ny[1] = number of rows appended
        nz : tupel of ints
            nz[0] = number of layers added on top
            nz[1] = number of layers added to bottom
        """
        def co (L,n):
            """Return array [n-1] where coordinate distances doulbles.

            Coordinate distance double between successive points, but
            total length is L.

            Parameters
            ----------
            L : float
                final length returnd
            n : int
                number of points returned minus 1
            """
            w = np.array([2**n for n in range(n)])
            w = w*L/sum(w)
            w = np.cumsum(w)
            return w

        xL = self.x[ 0] + co(-dx[0], nx[0])[::-1]
        xR = self.x[-1] + co( dx[1], nx[1])
        yF = self.y[ 0] + co( dy[0], ny[0])[::-1]
        yB = self.y[-1] + co(-dy[1], ny[1])
        zT = self.z[ 0] + co( dz[0], nz[0])[::-1]
        zB = self.z[-1] + co(-dz[1], nz[1])

        return Grid(np.hstack((xL, self.x, xR)),
                    np.hstack((yF, self.y, yB)),
                    np.hstack((zT, self.z, zB)))

    def inpoly(self, pgcoords, row=None, world=False):
        """Return bool array [ny, nx] or [nz, nx].

        Depending on row with true if cell centers are inside horizontal
        or vertical polygon

        Parameters
        ----------
        pgcoords: like [(0, 1),(2, 3), ...] or an (n, 2) np.ndarray
            polygon coordinates
        row: if None, polygon is assumed xy coordinates and grid is in the xy plane
             if row is int, then grid is cross section at given row
             and pgcoords are interpreted as x,z
        """
        if world:
            if row is None:
                return inpoly(self.Xmw, self.Ymw, pgcoords)
            else:
                return inpoly(self.XMw[:, row, :], self.ZM[:, row, :], pgcoords)
        else:
            if row is None:
                return inpoly(self.Xm, self.Ym, pgcoords)
            else:
                return inpoly(self.XM[:, row, :], self.ZM[:, row, :], pgcoords)

def array2tuples(A):
    """Return array as a list of tuples."""
    try:
        return [tuple(a) for a in A]
    except:
        raise "Can't convert array to list of tuples"

def extend_array(A, nx, ny, nz):
    """Return extended array specified by  nx, ny nz.

    Parameters
    ----------
    nx : tuple of ints (nxL, nxR)
        nxL = exstension to the left
        nxR = extension to the right
    ny : tuple of ints (nyF, nyB)
        nyF = extension to the front
        nyB = extension to the back
    nz : tupel of ints (nzT, nzB)
        nzT = extension to the top
        nzB = extension to the bottom
    """
    # extend in x-direction
    nxL, nxR = nx # extend in x-direction
    nyF, nyB = ny # extend in y-direction
    nzT, nzB = nz # extend in z-direction

    # extend in x-direction (axis=2)
    Nz, Ny, Nx = A.shape
    if nxL > 0:
        AxL = A[:,:,0:1] * np.ones((Nz, Ny, nxL))
        A   = np.concatenate((AxL, A), axis=2)
    if nxR > 0:
        AxR = A[:,:,-1:] * np.ones((Nz, Ny, nxR))
        A   = np.concatenate((A, AxR), axis=2)

    # extend in y-direction (axis=1)
    Nz, Ny, Nx = A.shape
    if nyF > 0:
        AyF = A[:, 0:1, :] * np.ones((Nz, nyF, Nx))
        A   = np.concatenate((AyF, A), axis=1)
    if nyB > 0:
        AyB = A[:, -1:, :] * np.ones((Nz, nyB, Nx))
        A   = np.concatenate((A, AyB), axis=1)

    # extend in z-direction (axis=0)
    Nz, Ny, Nx = A.shape
    if nzT > 0:
        AzT = A[0:1,:,:] * np.ones((nzT, Ny, Nx))
        A = np.concatenate((AzT, A), axis=0)
    if nzB > 0:
        AzB = A[-1:,:,:] * np.ones((nzB, Ny, Nx))
        A = np.concatenate((A, AzB), axis=0)
    return A


def gridspace(X, W):
    """Return grid spacing defined by points X and cell width W.

    Useful for defining cell grid coordinates of a rectangular grids.

    Parameters
    ----------
        X : np.array(, dtype=float)
                points along axis where cell width is defined by W
        W : np.array_like(X)
                cell widths at points defined by X
    Returns
    -------
        xGr: np.array( , dtype=float)
            coordinates between X[0] and X[-1] np.diff(X) corresponds to
            the relation given by X and W
    Usage
    -----
        x= np.array([0, 200, 400, 600, 1000]) #[::-1]
        w =np.array([5, 20, 10, 20, 5])
        xgr = gridspace(x, w)
        plt.plot(xgr[:-1], np.abs(np.diff(xgr)), '.-')
        plt.grid()
        plt.show()
    """
    X = np.array(X)
    W = np.array(W)
    assert len(X)==len(W), "len(x)={} must be equal to len(w)={}".format(len(X), len(W))
    if np.any(np.diff(X) < 0):
        sign = -1
        X = X[::-1]
        W = W[::-1]
    else:
        sign = +1

    assert np.all(np.diff(X) > 0), "X must be monotonously increasing or decreasing"
    assert np.all(W > 0), "W must all by > 0 (cell widths)"

    dx = lambda xi: np.interp(xi, X, W)

    XGR = [X[0]]
    x   = X[0]
    while True:
        if x > X[-1]:
            break
        x += dx(x)
        XGR.append(x)


    if X[-1] - XGR[-1] < 0.2 * dx(XGR[-1]):
        XGR[-1] = X[-1]
    else:
        XGR.append(X[-1])

    if sign < 0:
        XGR = XGR[::-1]

    return XGR

def sinespace(x1, x2, N=25, a1=0, a2=np.pi/2):
    """Return coordinates with distances according to the sine function."""
    return sinspace(x1, x2, N=N, a1=a1, a2=a2)

def sinspace(x1, x2, N=25, a1=0, a2=np.pi/2):
    """Return N points between x1 and x2, spaced according to sin betwwn a1 and a2.

    alike np.linspace and np.logspace
    parameters:
    -----------
    x1, x2 : the boundaries for distance to divide
    N=25 : number of desired points including x1 and x2
    a1=0, a2=np.pi/2 the angle in radians defaults to 0 and np.pi/a
    """
    a = np.linspace(a1, a2, N)   # angle space
    am = 0.5 * (a[:-1] + a[1:])  # centered
    dx = np.abs(np.sin(am))      # relative dx
    dx = dx/np.sum(dx) * (x2 -x1) # correcrted to cover x1..x2
    x = x1 + np.zeros(a.shape)   # initialize x wih x1
    x[1:] = x1 + np.cumsum(dx)   # fill x[1:]
    return x


def inpoly(x, y, pgcoords):
    """Return bool array [ny, nx] telling which grid points are inside polygon."""
    if not isinstance(pgcoords, np.ndarray):
        pgcoords = np.array(pgcoords)

    assert pgcoords.shape[1]==2 and pgcoords.ndim==2,\
        "coordinates must be an array of [n, 2]"
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


import unittest

class TestGridMethods(unittest.TestCase):
    """Tes the methods of teh grid class."""

    def setUp(self):
        """Set up the self tests."""
        self._x = np.linspace(0, 100, 11)
        self._y = np.linspace(0, 100, 7)
        self.z = np.linspace(0, -100, 5)
        self.gr = GridVdis(self._x, self._y, self.z, axial=False)

        self.up = np.random.rand(10) * self.gr.nx
        self.vp = np.random.rand(10) * self.gr.ny
        self.wp = np.random.rand(10) * self.gr.nz

        self.Xp, self.Yp, self.Zp = self.gr.uvw2xyz(self.up, self.vp, self.wp)

    def test_2normlized(self):
        """Return True if test is successful."""
        ue, ve, we = self.gr.xyz2uvw(self.Xp, self.Yp, self.Zp)
        for i in range(len(self.Xp)):
            print(("{:10.4g} "*6)
                  .format(self.up[i] ,ue[i], self.vp[i], ve[i], self.wp[i], we[i]))
        self.assertTrue (np.all(
                    AND(self.up == ue,
                        AND(self.vp == ve, self.wp == we))))

    def test_2regular(self):
        """Return True if test is successful."""
        Xe, Ye, Ze = self.gr.uvw2xyz(self.up, self.vp, self.wp)
        self.assertTrue( np.all(
                    AND(self.Xp == Xe,
                        AND(self.Yp == Ye, self.Zp == Ze))))

    def test_const1(self):
        """Return True if test is successful."""
        U = self.gr.const(3.)
        self.assertAlmostEqual(np.mean(U), 3.)

    def test_const2(self):
        """Return True if test is successful."""
        values = np.random.rand(self.gr.nz)
        digits = 5
        U = self.gr.const(values)
        self.assertTrue( np.all(
            np.round( np.mean(np.mean(U, axis=0), axis=0), decimals=digits) ==\
            np.round( values, decimals=digits)))

    def test_const3(self):
        """Return True if test is successful."""
        testVal = 3
        U = self.gr.const(testVal)
        self.assertTrue(self.gr.shape == U.shape)

    def test_order_and_remove_small_gaps(self):
        """Return True if test is successful."""
        tol = 1e-4
        digits = int(-np.log10(tol))
        x = np.array([3, 0., 2, 1e-5, 2, 3.000001, 4, 3])
        b = np.array([0., 2, 3, 4])
        gr = Grid(x, self._y, self.z)
        print("\n\n")
        for i in range(gr.nx+1):
            print( ('{:10.3g}'*2).format(np.round(gr.x[i], digits), np.round(b[i], digits) ))
        self.assertTrue(np.all(np.round(gr.x,digits) == np.round(b,digits)))

    def test_inpoly(self):
        """Return True if test is successful."""
        x = np.linspace(0., 100, 101)
        y = np.linspace(100, 0., 101)
        pgcoords = [(30, 0), (80, 50), (10, 80)]

        # individual inpoly function
        plt.spy(self.inpoly(x, y, pgcoords))
        plt.show()

        z = np.array([1, 0])
        gr = self.Grid(x, y, z)

        # grid_method inpoly
        plt.spy(gr.inpoly(pgcoords))
        plt.show()
        return True


def plot_shapes(rdr, ax=None, title='Shapes from shapefile'):
    """Plot polygons in rdr as patches.

    TODO: add filling with patches

    Parameters
    ----------
    rdr: shapefile.Reader
        opened by shapefile
    ax: matplotlib.axis
    title: str
        title in case no axes are given
    """
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_xlabel("x [m]")
        ax.set_ylabel("m +NAP")
        ax.set_title(title)

    for sr in rdr.shapeRecords():
        id = sr.record[0]
        p = np.array(sr.shape.points)
        ax.plot(p[:, 0], p[:, 1], label="shape {}".format(id))
    ax.legend(loc='best')
    return ax

def sub_grid(rdr, dx=0.1, dz=0.1, dy=1.):
    """Return the coordinates of a subgrid."""
    xL, xR  = rdr.bbox[0], rdr.bbox[2]
    zT, zB  = rdr.bbox[3], rdr.bbox[1]
    yN, yZ = dy / 2, -dy /2

    nz = int(np.ceil((zT - zB) / dz))
    ny = int(np.ceil((yN - yZ) / dy))
    nx = int(np.ceil((xR - xL)/ dx))

    x = np.linspace(xL, xR, nx + 1)
    z = np.linspace(zT, zB, nz + 1)
    y = np.linspace(yN, yZ, ny + 1)

    return x, y, z

# %%

if __name__ == '__main__':
    #unittest.main()
    Xp = np.array([ [  0., 15., 30., 40., 65],
                    [ -5., 10., 25., 45., 70],
                    [ -3., 13., 27., 41., 65],
                    [-10., 11., 23., 37., 67]])
    Yp = np.array([ [50., 45., 55., 50., 52.],
                    [37., 35., 40., 36., 35],
                    [18., 22., 19., 22., 20],
                    [ 5., -3.,  1., 5.,  4]])
    ncellsx = np.array([5, 3, 12, 5])
    ncellsy = np.array([6, 4, 10])
    
    z = np.array([0., -10., -12., -20.])

    gr = GridVdis(Xp, Yp, z, ncellsx=ncellsx, ncellsy=ncellsy, spline=True)


    ax = newfig("GridVdis object", "x", "y")
    
    gr.plot_grid(ax)
    
    ax.plot(gr.Xc.ravel(), gr.Yc.ravel(), 'r.')
    
    plt.show()
    
    # %%
    
    
    # Pairs for the HDF package
    # x = np.linspace(-100., 100., 21)
    # y = np.linspace(-100., 100., 21)
    # z = [0, -10, -20]

    # gr = GridVdis(x, y, z)

    # polygon = np.array([(23, 15), (45, 50.), (10., 81.), (-5., 78), (-61., 51.), (-31., 11.),
    #            (-6., -4.), (-42., -20.), (-50., -63.), (-7., -95.),
    #            (31., -80.), (60., -71.), (81., -31.), (5., -63.), (25., -15.), (95., 40.),
    #            (23, 15)])

    # pairs = gr.cell_pairs(polygon, open=False) # a closed polygon
    # polygon = polygon[:10]
    # pairs = gr.cell_pairs(polygon, open=True) # an open line


    # fig, ax = plt.subplots()
    # ax.set_title('Node pairs for the hor. flow-barrier package of Modflow')
    # ax.set_xlabel('x [m]')
    # ax.set_ylabel('y [m]')
    # gr.plot_grid(world=False, ax=ax)
    # ax.plot(polygon[:,0], polygon[:,1])
    # ax.plot(gr.Xm.ravel()[pairs[:,0]], gr.Ym.ravel()[pairs[:,0]], '.r', label='column 1')
    # ax.plot(gr.Xm.ravel()[pairs[:,1]], gr.Ym.ravel()[pairs[:,1]], '.b', label='column 2')
    # for pair in pairs:
    #     ax.plot(gr.Xm.ravel()[pair], gr.Ym.ravel()[pair], 'k-')
