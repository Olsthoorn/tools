#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 14:23:15 2018

Interopolate the Be and Breda tops using Kriging

@author: Theo 20180615
"""



import os
import shelve
import numpy as np
import fdm
import matplotlib.pyplot as plt
import logging
import pykrige as kr
import shape
import pandas as pd
import coords

#%% logging setup

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)-15s -  %(levelname)s - %(message)s')
logger = logging.getLogger('__name__')

def varfun_LT(args, x):
    '''
    try varfun_LT(x, slope=0.0092, nugget=10, alpha=150)
    '''
    slope, new_nugget, old_nugget, alpha = args
    return new_nugget + old_nugget * (1 - np.exp(-x / alpha)) + slope * x

def varfun_MT(args, x):
    '''
    try this
    ax.plot(x, varfun_MT(x, sill=70, range=12000, nugget=10.5, alpha=150))
    '''
    sill, range, new_nugget, old_nugget, alpha = args
    psill = sill - (old_nugget - new_nugget)
    return new_nugget + (old_nugget - new_nugget) * (1 - np.exp(-x / alpha))+ psill * (1 - np.exp(-x / (3.5 * range)))

# Kriging parameters, a dict with key is tuple (terraceName, formation)
krigpars = {('Laagterras', 'ztopBreda'): {
                        'variogram_model': 'linear',
                        'drift_terms': 'regional_linear',
                        'variogram_function': None,
                        'variogram_parameters': None,
                        'anisotropy_scaling': 2.25,
                        'anisotropy_angle'  : 27.5},
            ('Middenterras', 'ztopBreda'): {
                        'variogram_model': 'exponential',
                        'drift_terms': 'regional_linear',
                        'variogram_function': None,
                        'variogram_parameters': None, # {'sill': 65, 'range': 12000, 'nugget': 1},
                        'anisotropy_scaling':  2.0,
                        'anisotropy_angle'  :  0.0}}

if False:
    # Kriging parameters, a dict with key is tuple (terraceName, formation)
    new_nugget = 10
    krigpars = {('Laagterras', 'ztopBreda'): {
                            'variogram_model': 'custom',
                            'drift_terms': 'regional_linear',
                            'variogram_function': varfun_LT,
                            'variogram_parameters': [0.0092, new_nugget, 10 - new_nugget, 150], #{'slope': 0.0092, 'nugget': 10, 'alpha': 150 },
                            'anisotropy_scaling': 2.25,
                            'anisotropy_angle'  : 27.5},
                ('Middenterras', 'ztopBreda'): {
                            'variogram_model': 'custom',
                            'drift_terms': 'regional_linear',
                            'variogram_function': varfun_MT,
                            'variogram_parameters': [70, 1200, new_nugget, 10.5 - new_nugget, 150],
                            #{'sill': 70, 'range': 12000,'nugget': 10.5, 'alpha':150},
                            'anisotropy_scaling':  2.0,
                            'anisotropy_angle'  :  0.0}}



#varpars = {('Laagterras', 'ztopBreda'): {'slope':, 'nugget': 0},
#            ('Middenterras', 'ztopBreda'): {'still':, 'range': 'nugget': n}}

#%%

def krige_it(gr, data=None, x='x', y='y', key=None, column=None, boundary=None,  krigpars=None,
                     verbose=True, enable_plotting=True, **kwargs):
    '''Return z and and variance s with parts universally kriged.

    parameters
    ----------
        gr : fdm.mfgrid.Grid object
            grid object of which the xm, ym are used as the coordinates to interpolate to.
        data : pd.DataFrame
            index is the point name, columns are the data with 'x', 'y' the coordinates
            Chose the column name to krige
        boundary: polygon as [x, y]. May be list of boundaries
        column: str
            the column name to be kriged
            data are produced only with the boundary or boundaries if list
        krigpars: dict
            kriging parameters like
            kriging variances like
                       {'variogram_model': 'gaussian',
                        'drift_terms': 'regional_linear',
                        'anisotropy_scaling': 4.0,
                        'anisotropy_angle'  :  25}
        verbose : bool
            whether or not print variogram parameters
        enable_plotting: bool
            whetther or not show variogram plot.
        **kwargs :
    returns
    -------
        z : 2D ndarray of kriged elevation
        s : 2D ndarray of kriged variances
    '''
    # Universal kriging.

    #short name for kriging parametersf for this terrace,
    kp = krigpars[key]

    # fix key order, where each key is a borehole name
    #names = list(data.index)

    x, y, zf = data[x].values, data[y].values, data[column].values

    try:
        # Universal kriging with formation and terrace-specific paameters
        U = kr.uk.UniversalKriging(x, y, zf, variogram_model=kp['variogram_model'],
                               variogram_parameters=kp['variogram_parameters'],
                               variogram_function=kp['variogram_function'],
                               nlags=20,
                               weight=True,
                               anisotropy_scaling=kp['anisotropy_scaling'],
                               anisotropy_angle=kp['anisotropy_angle'],
                               drift_terms=kp['drift_terms'],
                               point_drift=None,
                               external_drift=None,
                               external_drift_x=None,
                               external_drift_y=None,
                               specified_drift=None,
                               functional_drift=None,
                               verbose=verbose,
                               enable_plotting=False)

        if enable_plotting:
            vario_title = kwargs.pop('title', 'semi-variogram {1} in {0}'.format(*key))
            # Parameters for plotting the variogram
            vario_kwargs = {'title': ', '.join([vario_title,
                                               "gamma: '{}'".format(kp['variogram_model']),
                                               'aniso=({:.1f},{:.0f})'.\
                                               format(kp['anisotropy_scaling'], kp['anisotropy_angle']),
                                               'drift={}'.format(kp['drift_terms'])]),
                            'xlabel': 'lags',
                            'ylabel': 'gamma'}

            U.display_variogram_model(**vario_kwargs)

            plt.gcf().set_size_inches((8.5, 5))
            plt.gca().title.set_fontsize('x-small')
            fmt = 'model={} drift={} aniso=({:.1f}, {:.1f}), Q1={:.2f} Q2={:.2f}, rc={:.2f}'

        if verbose:
            print(fmt.format(kp['variogram_model'], kp['drift_terms'],
                             kp['anisotropy_scaling'], kp['anisotropy_angle'],
                             *U.get_statistics()))

        # generate masked_arrays for x and y for points to use in kriging

        In = np.ones(gr.shape[1:], dtype=bool)

        In = gr.inpoly(boundary['points'], world=True)
        xOut = gr.Xmw[In]
        yOut = gr.Ymw[In]

        # actual kriging to desired points
        zk, ss = U.execute('points', xOut, yOut)

    except: # as err:
        logger.debug('Error for {}, no results'.format(col))
        raise

    return zk, ss, In


def krige_layer(gr, data=None, col=None, terraces=None, krigpars=None):
    Z = gr.Z[0]
    S = np.zeros_like(Z)
    for terraceName in terraces:
        key = (terraceName, col)

        z, s, In = krige_it(gr, data=bores, column=col, key=key,
                                 krigpars=krigpars,
                                 boundary=terraces[terraceName],
                                 verbose=True,
                                 enable_plotting=True)
        Z[In] = z
        S[In] = s
    return Z, S




if __name__ == '__main__' :

    datadir = './data'

    x = np.arange(181185, 185000 + 1, 10)
    y = np.arange(333470, 339280 + 1, 10)
    gr = fdm.mfgrid.Grid(x, y, [0, -1])


    # Get contour for selecting the boreholes for kriging
    #selectionContour = 'bs_pgon' # borehole_selection_polygon
    #wrld['bs_pgon'] = shape.shapes2dict(
    #                    os.path.join(GIS, 'boringenSelectiePolygon'),
    #                    key='name')['Selectiepgon']['points']

    fname = os.path.join(datadir, 'bs_pgon')
    bspgon = shape.shapes2dict(fname)[1]

    bores    = pd.DataFrame(shape.shapes2dict(os.path.join(datadir, 'boresTopBeBreDino+DEME'))).T
    bores['x'] = np.array([a[0][0] for a in list(bores['points'].values)])
    bores['y'] = np.array([a[0][1] for a in list(bores['points'].values)])
    bores = bores.drop(columns='points').astype(float)

    # only the bores in bspgon
    In = coords.inpoly(bores['x'].values, bores['y'].values, bspgon['points'])
    bores = bores.iloc[In]

    terraces = shape.shapes2dict(os.path.join(datadir, 'modelterrassen'))  #.pop('Hoogterras')
    terraces.pop('Hoogterras')

    col = 'ztopBreda'

    Z, S = krige_layer(gr, bores, col=col, terraces=terraces, krigpars=krigpars)

    x_ = np.linspace(0, 12000, 100)
    #ax = plt.gca()
    #ax.plot(x_, varfun_MT(x_, sill=70, range=1200, nugget=10.5, alpha=150), 'b')
    #ax.plot(x_, varfun_LT(x_, slope=0.009, nugget=10, alpha=150), 'b', label='custom')


    fig, ax = plt.subplots()
    ax.set_title('Elevations {} in {} and {}.'.format(col, *[nm for nm in terraces]))
    ax.set_xlabel('xRD [m]')
    ax.set_ylabel('yDR [m]')
    ax.grid()
    ax.imshow(Z, vmin=29, vmax=35, extent=(x[0], x[-1], y[0], y[-1]))
    for nm in terraces:
        ax.plot(*terraces[nm]['points'].T, label=nm)


    zKriged = gr.interpxy(Z, np.vstack((bores['x'], bores['y'])).T)
    bores['zKriged'] = zKriged[0]
    bores['zDiff']   = bores['ztopBreda'] - bores['zKriged']

    bores1 = bores.dropna()


    OK = (bores1['zKriged'] >  10).values  # not ok are 5 bores in the Hoogterras (outside scope)
    bores1 = bores1.loc[OK]
    ax.plot(bores1['x'], bores1['y'], '.')

    fig, ax = plt.subplots()
    bores1[['ztopBreda', 'zKriged', 'zDiff']].plot()
    ax = plt.gca()
    ax.set_title('{} in borehole and kriged'.format(col))
    ax.set_xlabel('point number')
    ax.set_ylabel('elevatiion')
    ax.grid()


    #bores[['ztopBreda', 'zKriged']].plot()
    '''
    ax.plot(bores['x'].values, bores['y'].values, '.')
    for k in bores.index:
        b = bores.loc[k]
        s = k + ' {:.2f}'.format(b['ztopBreda'])
        if not np.isnan(b['zKriged']):
            s = s + ' {:.2f}'.format(b['zKriged'])
            ax.text(b['x'] + 25, b['y'], s)
    '''
