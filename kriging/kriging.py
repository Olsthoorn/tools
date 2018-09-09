#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 14:23:15 2018

Interopolate the Be and Breda tops using Kriging

@author: Theo 20180613 20180908
"""

import os
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
    '''Custom variogram function for the Laagterras near Obbicht Limburg NL
    The function is used by pyKrige.uk.UniversalKirging()
    usage varfun_LT([slope, new_nugget, nugget, alpha], x)
    Paramters for this case: [slope=0.0092, new_nugget=0.2, nugget=10, alpha=150)
    The nugget is the value in the optimized case and the new_nugget is the one
    in the custom variogram.
    new_nugget must be > 0, pyKrige can't handle a zero nugget.
    '''
    slope, new_nugget, old_nugget, alpha = args
    return new_nugget + old_nugget * (1 - np.exp(-x / alpha)) + slope * x

def varfun_MT(args, x):
    '''Custom variogram function for the Middenterras near Obbicht Limburg NL
    The function is used by pyKrige.uk.UniversalKirging()
    usage varfun_MT([sil, range, new_nugget, nugget, alpha])
    parameters for thi case [sill=70, range=12000, new_nugget=0.2, nugget=10.5, alpha=150]
    The nugget is the value in the optimized standard case (exponential variogram) and
    the new_nugget the one of the adapted semi-variogram.
    new_nugget must be > 0, pyKrige can't handle nugget=0'
    '''
    sill, range, new_nugget, old_nugget, alpha = args
    psill = sill - (old_nugget - new_nugget)
    return new_nugget + (old_nugget - new_nugget) * (1 - np.exp(-x / alpha))+ psill * (1 - np.exp(-x / (3.5 * range)))


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

    x, y, zf = data['x'].values, data['y'].values, data[column].values

    try:
        # Universal kriging with formation and terrace-specific parameters
        # In pyKrige, first an kriging object is generated which holds the
        # parameters, after which methods are applied, like generating the
        # semi-vartiograms and kriging the data onto a grid.
        # Se documentation for details
        # https://pykrige.readthedocs.io/en/latest/generated/pykrige
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

            # Use of vario_kwargs has been added by my to the U.display_variogram_model function
            U.display_variogram_model(**vario_kwargs)

            # set figure size
            plt.gcf().set_size_inches((8.5, 5))
            plt.gca().title.set_fontsize('x-small')

        if verbose:
            fmt = 'model={} drift={} aniso=({:.1f}, {:.1f}), Q1={:.2f} Q2={:.2f}, rc={:.2f}'
            print(fmt.format(kp['variogram_model'], kp['drift_terms'],
                             kp['anisotropy_scaling'], kp['anisotropy_angle'],
                             *U.get_statistics()))

        # generate masked_arrays for x and y for points to use in kriging

        # Mask to seelect the points that are inside the provided boundary
        In = gr.inpoly(boundary['points'], world=True)
        xOut = gr.Xmw[In]
        yOut = gr.Ymw[In]

        # Carry out the actual kriging to desired grid points xOut and yOut
        zk, ss = U.execute('points', xOut, yOut)

    except: # as err:
        logger.debug('Error for {}, no results'.format(col))
        raise

    return zk, ss, In # zk and sk are vectors mathing the mask In


def krige_layer(gr, data=None, col=None, terraces=None, krigpars=None):
    '''Returns the kriged grid and the variance grid.

    The function is a wrapper to hide some details and prevent clitter below

    parameters
    ----------
        data : pd.DaraFrame containing columns 'x', 'y' and col for 'z'
            The borehole data as a DataFrame
        col: str
            name of the column to be interpolated
        terraces: dict
            dict containing the names and the ourlines of the river terraces.
        krigpars: dict
            the kriging parameters in a dict
    '''

    Z = np.zeros((gr.ny, gr.nx))
    S = np.zeros_like(Z)
    for terraceName in terraces:
        key = (terraceName, col)

        z, s, In = krige_it(gr, data=data, column=col, key=key,
                                 krigpars=krigpars,
                                 boundary=terraces[terraceName],
                                 verbose=True,
                                 enable_plotting=True)
        Z[In] = z
        S[In] = s
    return Z, S



if __name__ == '__main__' :

    # Example of using pyKrige with and without custom semi-variograms
    # Here the elevation of the top of the Breda Formation is interpolated
    # using universal kriging. The situation is near Obbicht in the Province
    # of Limburg, NL. But this is irrelevant for the illustration.

    # The data directory contains data in the form of shapefiles. There is the
    # shapefile containing the borehole locations with the elevation of interest.
    # There is the shapefile containing the outlines of two river Meuse Terraces
    # There is a selection polygon shapefile, used to select the boreholes that
    # will be used in the analysis.

    # Univeral kriging is applied. The kriging parameters as conveniently specified
    # in a dict, one for each of the two terraces. The parameters have been
    # optimized by running this file many times and thereby adapting the
    # variogram_model, the anisotropy_scaling and the anisotropy_angle.
    #These valuesa are used and the variograms will be plotted.

    # The variograms thus obtained cannot deal with variabiliy for small lags due
    # to lack of flexibility of each of the variogram_models. Combining two
    # variogram models, one for the longer lags and one for the shorter lags would
    # be a solution, which can be done with pyKirge by specifying a custom
    # variogram. The second set of krigpars, that will be used when setting if True
    # instead of if False, defines such custom variograms. These custom variograms
    # are the same for the long lags but use an exponential variogram for the short
    # lags, such that the whole variogram is smooth and continuous. With the
    # custom variograms one can specify arbitrary small nuggets while the
    # the variogram for long lags is not changed. Note that pyKrige cannot
    # handle nugget=0. The variogram functions were formulated with an extra
    # parameters new_nugget that specifies the custom nugget while nugget is
    # the nugget of the old variogram.
    # The variogram functions for the two terraces are define above as
    # varfun_LT andvarfun_MT. A custom variogram function must take two
    # parameters. The first is the list of parameter values use in the function
    # in the right order. The second is the x or lag (will be used by pyKrige.)

    #%% Kriging parameters
    # Kriging parameters can be conveniently packed in a dict and be selected by their
    # which represents a case, in this case the combination tuple (terraceName, formation)
    # The first set of krigpars uses the standers mathematical variograms with anisotropy
    # parameters that have been optimized by trial and error.

    # Note that I adapted the variogram plotting function of pykrige by adding
    # **kwargs and some code that allows passing title, xlabel and ylabel

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

    # This is an alternative set of krigeng parameters, but now we use a custom
    # variogram model. This requires a function (defined above) and its parameters.
    # the function must take two arguments one list and one value. The list is
    # the list of parameters, the value is for the x. Is called f(list, x)
    # The list is given in the variogram parameters.
    # Set if True to use the custom variograms.
    if True:
        # Kriging parameters, a dict with key is tuple (terraceName, formation)
        new_nugget = 0.2
        krigpars = {('Laagterras', 'ztopBreda'): {
                                'variogram_model': 'custom',
                                'drift_terms': 'regional_linear',
                                'variogram_function': varfun_LT,
                                'variogram_parameters':
                                    [0.0092, new_nugget, 10 - new_nugget, 150],
                                    #{'slope': 0.0092, 'nugget': 10, 'alpha': 150 },
                                'anisotropy_scaling': 2.25,
                                'anisotropy_angle'  : 27.5},
                    ('Middenterras', 'ztopBreda'): {
                                'variogram_model': 'custom',
                                'drift_terms': 'regional_linear',
                                'variogram_function': varfun_MT,
                                'variogram_parameters':
                                    [70, 1200, new_nugget, 10.5 - new_nugget, 150],
                                    #{'sill': 70, 'range': 12000,'nugget': 10.5, 'alpha':150},
                                'anisotropy_scaling':  2.0,
                                'anisotropy_angle'  :  0.0}}


    # Get the borehole data from the data folder
    datadir = './data'

    # This is the grid to interpolate onto.
    x = np.arange(181185, 185000 + 1, 10)
    y = np.arange(333470, 339280 + 1, 10)
    gr = fdm.mfgrid.Grid(x, y, [0, -1])


    # Get contour for selecting the boreholes for kriging
    fname = os.path.join(datadir, 'bs_pgon')
    bspgon = shape.shapes2dict(fname)[1]

    # Get the borehole information (In this case from a shapefile)
    bores    = pd.DataFrame(shape.shapes2dict(os.path.join(datadir, 'boresTopBeBreDino+DEME'))).T
    bores['x'] = np.array([a[0][0] for a in list(bores['points'].values)])
    bores['y'] = np.array([a[0][1] for a in list(bores['points'].values)])
    bores = bores.drop(columns='points').astype(float)

    # Retain only the bores in bspgon
    In = coords.inpoly(bores['x'].values, bores['y'].values, bspgon['points'])
    bores = bores.iloc[In]

    # Get the terrace contours, we will krige within each terrace separately
    terraces = shape.shapes2dict(os.path.join(datadir, 'modelterrassen'))  #.pop('Hoogterras')
    terraces.pop('Hoogterras')

    # The data column to be interpolated.
    col = 'ztopBreda'

    # The details of kriging are hidden in krige_layern and in krige_it within krige_layer
    # We gt a resulting interpolated value and variance for the grid
    # Interpolated in each terrace and combined. Layer points outside the terraces
    # Have np.nan.
    Z, S = krige_layer(gr, bores, col=col, terraces=terraces, krigpars=krigpars)

    # Plot the elevation of the kriged values as a layer
    fig, ax = plt.subplots()
    ax.set_title('Elevations {} in {} and {}.'.format(col, *[nm for nm in terraces]))
    ax.set_xlabel('xRD [m]')
    ax.set_ylabel('yDR [m]')
    ax.grid()
    # Take 29 and 35 as limits to accentuate the range of interest
    ax.imshow(Z, vmin=29, vmax=35, extent=(x[0], x[-1], y[0], y[-1]))

    # Plot the outline of the terraces
    for nm in terraces:
        ax.plot(*terraces[nm]['points'].T, label=nm)

    # Get the kriged values as the borehole locations for verification
    zKriged = gr.interpxy(Z, np.vstack((bores['x'], bores['y'])).T)
    bores['zKriged'] = zKriged[0] # add results as a column to the dataframe
    bores['zDiff']   = bores['ztopBreda'] - bores['zKriged'] # and also the difference

    # Retain only the boreholes in the terraces that is, without np.nan in their values.
    bores1 = bores.dropna()

    # There happen to be 5 boreholes outside the terraces in the data, also remove them
    OK = (bores1['zKriged'] >  10).values  # not ok are 5 bores in the Hoogterras (outside scope)
    bores1 = bores1.loc[OK]

    # plot the retained borehole locatons.
    ax.plot(bores1['x'], bores1['y'], '.')

    # Compare the measured and interpolated elevation and their difference
    #fig, ax = plt.subplots()
    bores1[['ztopBreda', 'zKriged', 'zDiff']].plot() # plots on current axes
    ax = plt.gca()
    ax.set_title('{} in borehole and kriged for nugget = {} $m^2$'.format(col, new_nugget))
    ax.set_xlabel('{} boreholes in alphabetical order -->'.format(len(bores1)))
    ax.set_ylabel('elevatiion')
    ax.grid()

    # We may plot the measured and interpolatted elevations on the map but
    # this becomes a mess with so many boreholes.
    '''
    ax.plot(bores['x'].values, bores['y'].values, '.')
    for k in bores.index:
        b = bores.loc[k]
        s = k + ' {:.2f}'.format(b['ztopBreda'])
        if not np.isnan(b['zKriged']):
            s = s + ' {:.2f}'.format(b['zKriged'])
            ax.text(b['x'] + 25, b['y'], s)
    '''
