#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" This module contains functions suitable for analysis of the results of
    calibration

Created on Tue Mar  7 08:55:42 2017

@author: Theo
"""

import matplotlib.pylab as plt
import numpy as np
import mfetc

def par_contrib(cov, param_names=None, width=0.8,
                           title="Contribution of parameters to eigen vectors",
                           ylabel='Fraction of eigen vector [%]',
                           xlabel='Eigen vectors',
                           verbose=True):
    """Makes figure showing parameter contribution to eac eigen vector.
    parameters
    ----------
    cov : square ndarray
    param_names: list of strings of length cov.shape[0]
    width: float
        width of bars, 0.1<width<1.0
    title: str
        title of figure
    ylabel: str
        ylabel
    xlabel: str
        xlabel
    verbose: bool
        when True print eigen values and eigen vectors.

    returns
    -------
    vs2: ndarray of same shape as cov
        square of eigen vectors, sums vertically to 1
    """
    n = cov.shape[0]

    if not param_names is None:
        if not len(param_names) == n:
            raise ValueError("len(param_names) should be {}".format(n))
    else:
        param_names = ["p"+str(i) for i in range(n)]

    xTick = np.arange(n, dtype=float)
    width = max(0.1, min(1.0, width))


    w, vr = np.linalg.eig(cov)
    ipvt = w.argsort()[::-1]
    w = np.diag(w[ipvt])
    vr = np.take(vr, ipvt, axis=1)

    if verbose:
        mfetc.prar(w, "eig_vals")
        mfetc.prar(vr, "eig_vecs")

    vs2 = np.array(vr)**2 # contributions to each eigen vector

    bot = np.zeros_like(vs2)
    bot[:-1, :] = 1 - np.cumsum(vs2, axis=0)[:-1]
    colors =['r', 'b', 'g', 'm', 'c']
    colors *= (n//len(colors) + 1)

    fig, ax = plt.subplots()
    for i in range(n):
        ax.bar(xTick, vs2[i], width, color=colors[i], bottom=bot[i],
               label=param_names[i])

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.xticks(xTick + 0.4, ["v"+str(i) for i in range(n)])
    ax.set_ylim((0, 1.0))
    ax.set_yticks(np.linspace(0, 1.0, 11))
    ax.legend(loc='best', fontsize='small')
    # use plt.show() only at the very end of your program
    return vs2


def jac(x0, func, kwargs=dict(), d=0.01):
    """returns jacobian computed by forward differences.
    parameters
    ----------
    x0: array_like
        parameter vector
    func: function
        accepting x0 and possibly other parametres given in dict kwargs
        and yielding vector of function values
    d: float
        step size, default is 0.01
    returns
    -------
    jacobian
    """

    d = 0.01
    dx = np.zeros_like(x0)
    J0 = func(x0, **kwargs)
    J = np.zeros(len(J0), len(x0))
    for i in range(x0):
        dx[i] += d
        J[:,i] = (func(x0 + dx, **kwargs) - J0) / d
        dx[i] -= d
    return J