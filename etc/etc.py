#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 00:23:37 2018

@author: Theo
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from pathlib import Path

#%%

import importlib.util

def where_is(module_name: str) -> str:
    """
    Return the file path of the given module if it can be found,
    or a message if it is a built-in or not found.

    Example:
    >>> where_is("numpy")
    '/Users/Theo/miniconda3/envs/flopy/lib/python3.10/site-packages/numpy/__init__.py'
    >>> where_is("sys")
    'sys is a built-in module'
    """
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return f"Module '{module_name}' not found"
    if spec.origin == "built-in":
        return f"{module_name} is a built-in module"
    return spec.origin

def check_project_root():
    """
    Ensure that the current project root (where src/, notebooks/, etc. live)
    is on sys.path so modules like `src.dutch_soils` can be imported safely.
    """
    project_root = os.getcwd()

    # Safety check: does this look like a project root?
    expected_dirs = {"src", "notebooks"}
    if not expected_dirs.issubset(set(os.listdir(project_root))):
        raise RuntimeError(
            f"{project_root} does not look like a valid project root. "
            f"Expected to see {expected_dirs} inside."
            "Maybe you didn't launch VScode from the project directory."
        )

    if project_root not in sys.path:
        sys.path.insert(0, project_root)  # prepend for priority
    return project_root

class Dirs:
    def __init__(self, home=None):
        """Return a standard project directory tree with home as root
        of home is given path, without overwriting).

        @ TO 2025-08-20 with ChatGPT

        Usage
        -----
        dirs = Dirs('~projects/my_project')
        dirs.create_standard_files()
        print(dirs.data)
        print(dirs.images)
        """
        if home is None:
            home = check_project_root()
        
        self.home = Path(home).expanduser().resolve()

        # Standard project layout
        self.src = self.home / 'src'
        self.data = self.home / 'data'
        self.images = self.home / 'images'
        self.docs = self.home / 'docs'
        self.LyX  = self.home / 'LyX'
        self.notebooks = self.home / 'notebooks'
        self.tests = self.home / 'tests'

        # Make directories (silently skip if they exist)
        for d in [self.src, self.data, self.images, self.docs, self.notebooks, self.tests]:
            d.mkdir(parents=True, exist_ok=True)
            
        self.create_standard_files()
            
    def create_standard_files(self):
        """Create common project files if they do not exist."""
        files = ['.gitignore', 'requirements.txt', 'pyproject.toml', 'README.md']
        for f in files:
            file_path = self.home / f  # self.home is a Path object, so '/' works
            if not file_path.exists():
                file_path.touch()  # creates an empty file

    def add_dir(self, name):
        """Create a new subdirectory under the project home and return its Path."""
        new_dir = self.home / name
        new_dir.mkdir(parents=True, exist_ok=True)
        return new_dir



def dayofweek(d):
    """Return dayofweek (0..7)."""
    d = np.datetime64(d) # d can be string or np.datetime64
    return  int((d - np.datetime64('1969-12-29', 'D')) / np.timedelta64(1, 'D') % 7)

def week(yr, wk):
    """return start and end date of week in year."""
    if wk > 53:
        yr, wk = wk, yr # in case you forgot order of yr and wk
    d = np.datetime64(str(yr), 'D') + (wk - 1) * np.timedelta64(7, 'D')
    week_start = d - dayofweek(d)
    week_end = week_start + np.timedelta64(6, 'D')
    return week_start, week_end


def mlDatenum2npDatetime(mlDatenum):
    """Return np.Datenum64(object) from matlab datenum.

    >>>739291.4753587963
    np.datetime64('2023-02-09 11:24:31')
    """
    unix_start_datetime = np.datetime64('1970-01-01')
    unix_start_datenum = 719529
    return (unix_start_datetime
            + (mlDatenum - unix_start_datenum) * 86400 * np.timedelta64(1, 's'))


def npDatetime2mlDatenum(datetime):
    """Return matlab datenum from np.Datenum64 object.

    >>>npDatetime642mlDatenum(np.datetime64('2023-02-09 11:24:31'))
    739291.4753587963
    """
    unix_start_datetime = np.datetime64('1970-01-01')
    unix_start_datenum = 719529
    return ((datetime - unix_start_datetime)
            / np.timedelta64(1, 's') / 86400 + unix_start_datenum)
    

#print(npDatetime2mlDatenum(np.datetime64('2023-02-09 11:24:31')))
#print(mlDatenum2npDatetime(738926.4753587963))

# Shorthand
dn2dt = mlDatenum2npDatetime
dt2dn = npDatetime2mlDatenum

def attr(obj):
    """Return the attributes of an object that do not start with '_'."""
    return [o for o in dir(obj) if not o.startswith('_')]
    

linestyles_ = ['-', '--', '-.', ':']
colors_ = ['b', 'r', 'g', 'k', 'm', 'c', 'brown', 'orange', 'gold', 'beige']

def color_cycler(colors=None):
    if not colors:
        colors = colors_
    for c in itertools.cycle(colors):
        yield c
    

def linestyle_cycler(linestyles=None, colors=None):
    if not linestyles:
        linestyles = linestyles_
    if not colors:
        colors = colors_
    for l in itertools.cycle(linestyles):
        for c in itertools.cycle(colors):
            yield {'ls': l, 'color': c}

def line_cycler():
    for ls in itertools.cycle(linestyles_):
        yield ls


def newfig(title='title?', xlabel='xlabel?', ylabel='ylabel?',
           xscale=None, yscale=None, xlim=None, ylim=None,
           figsize=(12, 8), fontsize=12, aspect=None, invert_yaxis=False):
    """Return a new figure.

    Parameters
    ----------
    title, xlabel, ylabel: str
        title of fig and of xaxis and yaxis
    xscale: str ('log', 'linear') or None
        type of axis
    yscale: str ('log', 'linear') or None
        type of axis
    figsize: 2-tuple
        fig_size in inches
    fontsize: int
        fontsize in points of titles
    """
    fig, ax = plt.subplots()
    fig.set_size_inches(figsize)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xscale: ax.set_xscale(xscale)
    if yscale: ax.set_yscale(yscale)
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    if aspect: ax.set_aspect(aspect)

    if fontsize:    
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                        ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fontsize)
    
    if invert_yaxis:
        ax.invert_yaxis()
            
    ax.grid()
    return ax


def newfigs(titles, xlabel, ylabels, xlim=None, ylims=None, xscale=None, yscales=None, invert_yaxes=None, sharex=False, sharey=False, figsize=(12, 12), fontsize=12):
    """Return array of xshared Axes on a single figure.

    @TO 2020-02-10
    
    Parameters
    ----------
    titles: seqence of str
        one title for each Axes.
    xlabel: str
        label of the xshared Axes.
    ylabels: sequence of str
        one label for each Axes.
    xlim: 2-sequence of left and right value of the shared x-axis.
        a single xlim used for all Axes.
    xscale: str (None, 'log' or 'linear').
        a single value for all Axes.
    yscales: a sequence of str, each of which is 'log' or 'linear'.
        yscales itself may be None inwich case all Axes have a linear y-scale.
    invert_yaxes: seq of booleans
        one value for each Axes. Tells to invert the vertical axis of the respective Axes.
        ivert_yaxes may be None, in which case no y-axis will be inverted.
    sharex: bool (False)
        if True, share x-axes.
    sharey: bool (False)
        if True , share y-axes.
    figsize: 2-tuple of floats
        figure size in inches (w, h)
    fontsize: int
        fontsize in points for the Axes and labels and ticklabeels
    """
    
    assert isinstance(titles, (tuple, list)), "titles must be a tuple or list."
    assert isinstance(xlabel, str), "xlabel must be string."
    assert isinstance(ylabels, (tuple, list)), "ylabels must be a tuple or list."
    if ylims:
        assert isinstance(ylims, (tuple, list, np.ndarray)), "ylims must be a sequence of ylim tuples"
    if yscales:
        assert isinstance(yscales, (tuple, list, np.ndarray)),\
                        "yscales must be a sequence of str ('log' or 'linear"
    if invert_yaxes:
            assert isinstance(yscales, (tuple, list, np.ndarray)),\
                        "invert_yaxes must be a sequence of booleans"
    N = len(titles)
    assert N == len(ylabels), "number titles {} must equal the number of ylabels {}.".format(N, len(ylabels))
    
    fig, axs = plt.subplots(N, sharex='all', squeeze=True)
    fig.set_size_inches(figsize)
    for i, (ax, title, ylabel) in enumerate(zip(axs, titles, ylabels)):
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        if xlim is not None: ax.set_xlim(xlim)
        if ylims is not None: ax.set_ylim(ylims[i])
        if xscale: ax.set_xscale(xscale)
        if yscales: ax.set_yscale(yscales[i])
        if invert_yaxes and invert_yaxes[i]: ax.invert_yaxis()
        if fontsize:   
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                        ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fontsize)
    axs[-1].set_xlabel(xlabel)
    return axs 


def newfigs2(titles, xlabels, ylabels, xlims=None, ylims=None, xscales=None, yscales=None, invert_yaxes=None, figsize=(12, 12), fontsize=12):
    """Return array of xshared Axes on a single figure.

    @TO 2020-02-10
    
    Parameters
    ----------
    titles: seqence of str
        one title for each Axes.
    xlabels: str
        labels of the xshared Axes
    ylabels: sequence of str
        one label for each Axes
    xlims: tuple of 2-sequence of left and right value of the shared x-axis
        a single xlim used for all Axes
    xscales: tuple of 2 str (None, 'log' or 'linear')
        a single value for all Axes
    yscales: a sequence of str, each of which is 'log' or 'linear'
        yscales itself may be None inwich case all Axes have a linear y-scale
    invert_yaxes: seq of booleans
        one value for each Axes. Tells to invert the vertical axis of the respective Axes.
        ivert_yaxes may be None, in which case no y-axis will be inverted.
    figsize: 2-tuple of floats
        figure size in inches (w, h)
    fontsize: int
        fontsize in points for the Axes and labels and ticklabeels
    """
    
    assert isinstance(titles, (tuple, list)), "titles must be a tuple or list."
    assert isinstance(xlabels, (tuple, list)), "xlabels must a tuple or list of strings."
    assert isinstance(ylabels, (tuple, list)), "ylabels must be a tuple or list of strings."
    if xlims:
        assert isinstance(xlims, (tuple, list, np.ndarray)), "xlims must be a sequence of xlim tuples"
    if xscales:
        assert isinstance(xscales, (tuple, list, np.ndarray)),\
                        "xscales must be a sequence of str ('log' or 'linear"
    if ylims:
        assert isinstance(ylims, (tuple, list, np.ndarray)), "ylims must be a sequence of ylim tuples"
    if xscales:
        assert isinstance(xscales, (tuple, list, np.ndarray)),\
                "xscales must be a sequence of str ('log' or 'linear"
    if yscales:
        assert isinstance(yscales, (tuple, list, np.ndarray)),\
                        "yscales must be a sequence of str ('log' or 'linear"
    if invert_yaxes:
            assert isinstance(yscales, (tuple, list, np.ndarray)),\
                        "invert_yaxes must be a sequence of booleans"
    N = len(titles)
    assert N == len(ylabels), "number titles {} must equal the number of ylabels {}.".format(N, len(ylabels))
    
    # Choose arrangement of axes
    if N <= 4:
        n, m = 1, N        
    else:
        n, m = int(N / 4), 4
        
    fig, axs = plt.subplots(n, m, squeeze=True)
    fig.set_size_inches(figsize)
    for i, (ax, title, xlabel, ylabel) in enumerate(zip(axs, titles, xlabels, ylabels)):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        if xlims: ax.set_xlim(xlims[i])
        if ylims: ax.set_ylim(ylims[i])
        if xscales: ax.set_xscale(xscales[i])
        if yscales: ax.set_yscale(yscales[i])
        if invert_yaxes and invert_yaxes[i]: ax.invert_yaxis()
        if fontsize:   
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                        ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fontsize)
    return axs 


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

__all__ =[name for name in locals() if callable(locals()[name])
           and not name.startswith('_')]

# %%