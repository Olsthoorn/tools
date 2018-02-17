#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 23:13:56 2018

Checker class for checking arrays and lists associated with structured fdm grids

@author: Theo 20180216
"""

#%% imports

import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

#%% Showing NaNs in any of the data
'''
The only cells that can have NaN's in any of the arrays are those for which
IBOUND == 0. So the question to answer in order to verify the input of the
model is, are there any cells for any variable that are NaN within the area
determined by IBOUND==0.
'''

class Checker:
    def __init__(self, gr, IBOUND):
        self.gr = gr
        self.ibound = IBOUND

    def check(self, A, name=None):
        if isinstance(A, np.ndarray):
            if name is None:
                name = 'array'
            mask = np.logical_and( self.ibound!=0, np.isnan(A) )
            if np.any(mask):
                print('{} nans in {} where ibound!=0:'.format(np.sum(mask), name))
                self.gr.LRC(mask)
            else:
                print("There are no NaNs in {}.".format(name))
        elif isinstance(A, list):
            if name is None:
                name == 'list'
            B = [a for a in A if np.any(np.isnan(a))]
            if len(B) == 0:
                print("There are no NaNs in {}.".format(name))
                return
            I = self.gr.I([(b[0], b[1], b[2]) for b in B])
            B = [b for b, i in zip(B, I) if self.ibound.ravel()[i] != 0]
            if len(B) > 0:
                print('{} records contain at least one NaN where IBOUND !=0 in {}:'.format(len(B), name))
                pprint(B)
        elif isinstance(A, dict):
            for k in A.keys():
                self.check(A[k], name=name)
        else:
            print('Can only check arrays, lists and dicts of lists or arrays.')
        return

    def show(self, A, name=None, grid=True, outline=None):
        '''Plot non-nan and nan values in array A as dots'''
        if isinstance(A, np.ndarray):
            if name is None:
                name = 'array'
            if np.sum(np.isnan(A)) == 0:
                print("There are no NaNs in {}. Nothing to do. Quitting.".format(name))
                return
            if A.ndim == 2:
                A = A[np.newaxis, :, :]
            layers = A.shape[0]
            fig, ax = plt.subplots(layers, 1)
            for iL in range(layers):
                ax[iL].set_title('layer {}'.format(iL))
                ax[iL].set_xlabel('x [m]')
                ax[iL].set_ylabel('y [m]')
                if grid:
                    self.gr.plot_grid(ax=ax[iL], world=False)
                mask = np.logical_and(np.isnan(A[iL]), self.ibound[iL] !=0)
                ax[iL].plot(self.gr.Xm[mask], self.gr.Ym[mask], 'r.',
                          label='nans where ibound != 0')
                if outline is not None:
                    ax[iL].plot(outline[:,0], outline[:,1], label='model area')
                ax[iL].legend(loc='best')

        else: # asssume L is list of records
            if name is None:
                name = 'list'
            LRC    = np.array([rec[:3] for rec in A])
            L = LRC[:,0]
            R = LRC[:,1]
            C = LRC[:,2]
            nans  = np.isnan(np.sum(np.array(A), axis=1))
            if np.sum(nans) == 0:
                print('There are no NaNs in {} Nothing to do. Quitting.'.format(name))
                return

            layers = np.unique(L)
            fig, ax = plt.subplots(len(layers), 1)
            if not isinstance(ax, list):
                ax = [ax]
            for iL in layers:
                ax[iL].set_title('layer {}'.format(iL))
                ax[iL].set_xlabel('x [m]')
                ax[iL].set_ylabel('y [m]')
                ax[iL].grid()
                if grid:
                    self.gr.plot_grid(world=False)

                mask1 = np.logical_and(L == iL, np.logical_not(nans))
                mask2 = np.logical_and(L == iL, nans)
                ax[iL].plot(self.gr.xm[C[mask1]], self.gr.ym[R[mask1]], '.g', label='ok')
                ax[iL].plot(self.gr.xm[C[mask2]], self.gr.ym[R[mask2]], '.r', label='nans')
                if outline is not None:
                    ax[iL].plot(outline[:,0], outline[:,1], label='model area')
                ax[iL].legend(loc='best')

    def spy(self, A, name=None):
        '''spy nan values in array A'''
        if not isinstance(A, np.ndarray):
            if name is None:
                name = 'Input'
            raise Exception("{} must b an array".format(name))

        if not np.any(np.isnan(A)):
            if name is None:
                name = 'array'
            print('There are no NaNs in {}'.format(name))
        else:
            if name is None:
                name = 'list'

            if A.ndim == 2: # make it 3D to loop over layers
                A = A[np.newaxis, :, :]

            layers = A.shape[0]
            fig, ax = plt.subplots(layers, 1)
            ax.set_title('Nans in {}'.format(name))
            if not isinstance(ax, list):
                ax = [ax]
            for iL in layers:
                ax[iL].set_title('spy layer 1 for nans')
                ax[iL].set_xlabel('x [m]')
                ax[iL].set_ylabel('y [m]')
                plt.spy(A[iL], ax=ax[iL])

