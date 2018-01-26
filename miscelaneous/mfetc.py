# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 04:26:57 2016

Various useful functions for working with models

@author: Theo
"""


import numpy as np
import matplotlib.pylab as plt
import mfexceptions as err

NOT = np.logical_not
AND = np.logical_and

def stat(A):
    """Returns statistics of ndarray as dict"""
    B = A.ravel()
    Out = {}
    Out['shape'] = A.shape
    Out['min'] = np.min(B)
    Out['mean'] = np.mean(B)
    Out['med'] = np.median(B)
    Out['max'] = np.max(B)
    Out['std'] = np.std(B)
    return Out


def plot_logo(ax, logo=None, pos=[1.00, 0.1, 0.3, 0.3]):
    '''plot the logo on current figure (axes)
    parameters
    ----------
    ax : matplotlib.Axes
    logo : str
        name of figure file to use as logo
    pos : sequence of 4 floats
        [llx,lly, wx, wy]
    TODO:
        test
    '''
    im = plt.imread(logo)
    LOLO = [1.00, 0.1, 0.3, 0.3] #logo position and size
    zorder = 1                     #z-order position
    logo_ax = ax.get_figure().add_axes(LOLO, anchor='SW', zorder = zorder)
    logo_ax.imshow(im)
    logo_ax.axis('off')


def display(*args, fmt='10.3g'):
    """Show a series if equal length vectors next to each other

    a bit like display() in Matlab, but without the headers:

    Example:
    show(z, y, z)
    """
    A = args if len(args)>1 else args[0]
    prarr(np.vstack(A).T, N=len(A), fmt=fmt)


def prar(A, name='var', tabs=False, ncol=10):
    """print ndarray matlab like.
    If A is string, prints name of array before the array.
    Otherwise, prints (?) as array name.
    """

    if isinstance(A, np.matrix):   A = A.copy()

    if len(A.shape)==1:  A = [A]

    A = np.array(A)

    m, n = A.shape

    K = int(np.ceil(A.shape[1]/ncol))

    fmt="\t{:4g}" if tabs else " {:9.4g}"

    for k in range(K):
        fr = ncol * k;
        to =  min(ncol * (k + 1), n)

        print("\n{}[:, {}:{}]=\n".format(name, fr, to))
        for r in A:
            print((fmt * len(r[fr:to])).format(*r[fr:to]))
        print()


def prarr(A, N=8, fmt='10.4g'):
    """matlab like 1D, 2D and 3D array Printer

    (Wraps around `pra, 2D print function`)

    examples:
        prarr(A)
        prarr(A, N=10)
        prarr(A, N=7, fmt='10.2f')

    Numeric format `fmt` must be fortran like format string:
        '8.4g', '10d', '12.4f', '15.3e'

    TO 160926
    """
    A = np.asarray(A)

    if len(A.shape)<2:  # make sure array is at least 2D
        A = A.reshape((1,len(A)))

    if len(A.shape)>2: # when 3D, print 2D layers repeatedly with 1 base layer prefix
        for iL in range(A.shape[2]):
            prefix = "Layer iz={0}, ".format(iL)
            pra(A[:,:,iL], N, fmt, prefix)
    else:
        pra(A, N, fmt, "") # regular 2D print


def pra(A, fmt='10.4g', ncol=10, prefix='', decimals=7):
    """Matlab like 2D array Printer.

    (Is called by wrapper `prarr`, see docstring of `prarr`)

    TO 160926
    """
    if A.dtype==complex:
        try:
            w,f = fmt.split('.')
            if int(w) < 15:
                fmt = '16.4f'
        except:
            pass
    ft = "{{0:{0}}}".format(fmt)     # construct format

    A = np.asarray(A)

    if len(A.shape) == 1:
        A.reshape((1, len(A))) # make sure A is at least 2D

    SHP =A.shape
    Nrows, Ncols = SHP[:2] # number of rows and columns to print
    Nblocks = int(np.ceil(Ncols / ncol))# number of column blocks to print

    cols=np.array([0, ncol]) # first block of columns
    for iBlock in range(Nblocks): # print each block of columns
        print("\n{} for columns {} to {}".\
              format(prefix, cols[0], min(cols[1], Ncols) - 1))
        for iRow in range(Nrows):  # print all rows
            a = np.round(A[iRow], decimals) # current data row
            for j in range(cols[0], min(cols[1], Ncols)):
                print(ft.format(a[j]), end="")  # print one row
            print()
        cols += ncol # next block of columns to be printed
        print # blank line


def interrogate(item):
    """Print useful information about item."""
    if hasattr(item, '__name__'):
        print("NAME:    ", item.__name__)
    if hasattr(item, '__class__'):
        print("CLASS:   ", item.__class__.__name__)
        print("ID:      ", id(item))
        print("TYPE:    ", type(item))
        print("VALUE:   ", repr(item))
        print("CALLABLE:  ",end="")
        if callable(item):
            print("Yes")
        else:
            print("No")
        if hasattr(item, '__doc__'):
            doc = getattr(item, '__doc__')
            doc = doc.strip()   # Remove leading/trailing whitespace.
            firstline = doc.split('\n')[0]
            print("DOC:     ", firstline)
        print()

