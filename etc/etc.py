#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 00:23:37 2018

@author: Theo
"""

import numpy as np
import pandas as pd

__all__ = ['ddir', 'linestyles_', 'colors_', 'linestyle_cycler', 'line_cycler']

import itertools

def ddir(obj):
    print([d for d in dir(obj) if not d.startswith('__')])
    
    
linestyles_ = ['-', '--', '-.', ':']
colors_ = ['b', 'r', 'g', 'k', 'm', 'c', 'brown', 'orange', 'gold', 'beige']

def linestyle_cycler():
    for l in itertools.cycle(linestyles_):
        for c in colors_:
            yield {'ls': l, 'color': c}

def line_cycler():
    for ls in itertools.cycle(linestyles_):
        yield {'ls': ls}



def get_outliers(ds, inner=1.5, outer=3.0):
    '''Report outliers in a pd.DataFrame column Col.
    
    parameters
    ----------
    ds : pd.Series or array like (np.ndarray, tuple, list)
    inner: float > 0
        inner outlier fence factor: potential outlier:
          OR ( < median - inner * (Q3 - Q1), > median + inner * (Q3 - Q1))
    outer: float > 0
        outer outlier fence factor: outlier : outer * Q75-Q25quarter range
          OR ( < median - outer * (Q3 - Q1), > median + outer * (Q3 - Q1))

    >>> df = pd.DataFrame(index=np.arange(10), data=np.random.randn(10, 3), columns=['a', 'b', 'c'])
    >>> df.loc[[2, 5], 'b'] = [2.4, -10]
    >>> col = 'b'
    >>> get_outliers(df[col])  # ds[col] is a pd.Series not a pd.DataFrame !

    
    @ TO 2018-06-22 12:30
    '''
    
    if isinstance(ds, pd.Series):
        pass
    elif isinstance(ds, (np.ndarray, tuple, list)):
        ds = pd.Series(ds)
    elif isinstance(ds, pd.DataFrame):
        raise KeyError('Specify a series by selecting a column of your DF like df[col].')
    else:
        raise KeyError('''Can only handle pd.Series, np.arrays, tuples and lists of floats.
                       Can't handle your type <{}>.'''.format(type(ds)))
        
    Q3 = ds.quantile(0.75)
    Q1 = ds.quantile(0.25)
    dQ = Q3 - Q1
    med = ds.median()

    innerFence = med - inner * dQ, med + inner * dQ
    outerFence = med - outer * dQ, med + outer * dQ
    
    potOutliers  = np.logical_or(ds < innerFence[0], ds > innerFence[1])
    truOutliers  = np.logical_or(ds < outerFence[0], ds > outerFence[1])

    if any(potOutliers):
        print('Prob. true  outliers ({} * inner_quartile_range):'.format(inner))
        print(ds[potOutliers])
    
    if any(truOutliers):
        print('Potentential outlier ({} * inner_quartile_range:'.format(outer))
        print(ds[truOutliers])
    else:
        print('No outliers in df.')
