#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 10:33:13 2018

@author: Theo 180618
"""

import colelctions
import numpy as np
import matplotlib.pyplot as plt

class CalibdPZ:
    '''Return a piezometer object suitable for calibration.
    
    This object has a pd.DataFrame with columns 'measured', 'computed', 'diff'.
    The index is the absolute time as timestamps ('np.datetime64[ns']) obtained
    from the fdm.mfgrid.stressPeriod object. It allows comparing computed with
    measured data at points in time of the computed values. The measured
    values are interpolated to te computed times. It also allows to see
    the drawdown (head change) beyond a given starting timestamp, specified
    by the the index into the DataFram.index. It can plot itself, on its own
    of as a contribution to a chart that already contains other data.
    
    CalibDict is a containter (collections.UserDict) that can hold CalibPZ
    objects and may plot all at once for visual comparison or for
    calibraton purposes.
    '''
    
    def __init__(self, gr, hds, peilb, name, stressPeriod):
        '''
        parameters
        ----------
        gr : fdm.mfgrid.Grid
            fdm mesh
        hds : nd.array
            modflow-simulated heads
        peilb : dict
            piezometer objects
        name : str
            piezometer name (must be key of peilb)
        stressPeriod : fdm.mfgrid.StressPeriod
            stress period object
        
        
        @TO 180618
        '''
        self.name = name
        
        pb = peilb[name]
        
        self.x, self.y, self.z = pb.meta['xRd'], pb.meta['yRd'], pb.meta['NAP']
        
        # Then get the times from the simulation stressPeriod object.
        self.timestamps = np.asarray(stressPeriod.get_datetimes(),
                                     dtype='datetime64[ns]')

        self.ddstart = self.timeStamps[0] # default
        
        # generate a pd.DataFrame with the meausured heads interpolated at the
        # timestamps        
        self.data = pd.DataFrame(pb.interpolate(self.timestamps), columns=['mNAP'])

        # replace the column name
        self.data.columns = ['measured']

        # add column computed, interpolating the heads at pb location.
        # We need to specify the model layer number in one way or another.
        self.data['computed'] =\
            pd.DataFrame(gr.interpxy(hds, (self.x, self.y), iz=iz),
                         index=self.timestamps, columns=['mNAP'])
            
        self.data['diff'] = self.data['computed'] - self.data['measured']
            
        return
    
    def tstart(self, fmt='%d-%m-%Y %H:%M'):
        '''Returns start time as string.'''
        return self.timestamps[0].strftime(fmt)

    def tend(self, fmt='%d-%m-%Y %H:%M'):
        '''Returns end time as string.'''
        return self.timestamps[-1].strftime(fmt)
    
    def set_ddstart(self, itime=0):
        '''Sets start time for drawdown calculations'''        
        self.ddstart = self.timestamps[itime]
    
    @property
    def err(self):
        '''Returns computed - measured heads as pd.Series.'''
        return self.data['diff']
    
    def map(self, fun, dd=False):
        '''Applies function fun on column 'diff' of data.
        
        parameters
        ----------
        fun : numpy function
            function like np.min, np.max, np.var, np.std
        dd : bool
            whether or not to compute drawdown.
            Use set_ddstart to set start time for drawdown.
        '''
        if dd==True:
            return fun(self.data['diff'] - self.data['diff'][itime])
        else:
            return fun(self.data['diff'])
    
    def var( self, dd=False): return self.map(np.var, dd=dd)
    def std( self, dd=False): return self.map(np.std, dd=dd)
    def min( self, dd=False): return self.map(np.min, dd=dd)
    def max( self, dd=False): return self.map(np.max, dd=dd)
    def mean(self, dd=False): return self.map(np.mean, dd=dd)
    
    
    def __getitem__(self, key):
        return self.data[key]
    
    def __len__(self):
        return len(self.data)
    
    def __str(self):
        return '{}[{}], x={:.2f} y={:.2f} z={:.3f}\n'.\
                format(self.__class__, self.name, self.x, self.y, self.iz) +\
                self.data.__str__

    def plot(self, what, dd=False, **kwargs):
        '''Plot self.data[what] or self.data[what] - self.data[what][ddstart]
        
        parameters
        ----------
            what : str
                header of column to be plotted
            dd : bool
                whether to plot heads or head-change (drawdown)
            kwargs: dict
                parameters to be passed to ax.set and plot.
                If kwargs['ax'] is plt.Axes then use it else create it        
        '''
        ax = kwargs.pop('ax', None)
        if ax is None:
            fig, ax = plt.subplots()
            if dd==True:    prep = 'Head '
            else:           prep ='Head change '
            
            title='{} {}[{}] at x={}, y={} layer={}'\
                .format(prep, self.__class__, self.name, self.x, self.y)
            ax.set(title=title,
                   xlabel=kwargs.pop('xlabel','time'),
                   ylabel=kwargs.pop('ylabel',prep + '[m]'),
                   xlim = kwargs.pop('xlim', None),
                   ylim = kwargs.pop('ylim', None),
                   xscale = kwargs.pop('xscale', 'linear'),
                   yscale = kwargs.pop('yscale', 'linear'))
            
        ax.grid(True)
        if dd==False:
            ax.plot(self.index(), self.data[what], label=self.name, **kwargs)
        else:
            ax.plot(self.data.index() - self.data.index[itime],
                    self.data[what] - self.data[what][itime], label=self.name, **kwargs) 


class Calibdict(collections.UserDict):
    def __init__(self, colllars):
        
        
        pass
    
    def plot(self, what, dd=False, kwargs):
        fig, ax = plt.subplots()
        ax.set(title='piezometers', xlabel='time', ylabel='head')
        for k in kwargs:
            try:
                fig.set(k, kwargs[k])
                kwargs.pop(k)
            except:
                try:
                    ax.set(k, kwargs[k])
                    kwargs.pop[k]
                except:
                    pass
        
        for pb in self.data:
            pb.plot(what, dd=dd, **kwargs)
            
    def __len__(self):
        return len(self.data)
        
    def __getitem(self, key):
        return self.data[key]
    
    def __setitem(self, key, pb):
        self.data[key] = pb

    def __append__(self, value):
        self.data.append(value)
        
    def __pop__(self, key):
        self.data.pop(key)
    
