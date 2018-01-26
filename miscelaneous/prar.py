#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 22:56:54 2017

@author: Theo
"""
import numpy as np


def prar(A, name='var', tabs=False, ncol=10):
    """prints ndarray the Matlab way.

    parameters:
    -----------
    A : array to print
    name : name of array (used in print, otherwise a default is used)
    tabs : if True use tabse beween items, which is handy for copying to Excel
    ncol : The numbe of consecutive columns to print as a block.
    @TO 20170315
    """

    if isinstance(A, np.matrix):   A = A.copy()

    if len(A.shape)==1:  A = [A]

    A = np.array(A)

    m, n = A.shape

    K = int(np.ceil(A.shape[1]/ncol))

    fmt="\t{:4g}" if tabs else " {:9.4g}"

    for k in range(K):
        fr = ncol * k;
        to =  min(ncol*(k+1), n)

        print("\n{}[:, {}:{}]=\n".format(name, fr, to))
        for r in A:
            print((fmt * len(r[fr:to])).format(*r[fr:to]))
        print()
