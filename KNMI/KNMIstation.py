#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 23:33:55 2017

@author: Theo
"""
tools = '/Users/Theo/GRWMODELS/python/tools/'

import sys
import os

if not tools in sys.path:
    sys.path.insert(1, tools)

import numpy as np
import pandas as pd
import datetime as dt
from collections import UserDict
import matplotlib.pyplot as plt
from datetime import datetime
from coords import wgs2rd
from coords.kml import nederland


class KNMI_stations(UserDict):
    '''Class contains meta data of all KNMI weather stations

    The class is a subclass of collections.UserDict. The dict is
    held in the attribute 'data'.

    '''

    def __init__(self, stationsFileName):
        # scan the file

        self.data ={}
        
        if not os.path.isfile(stationsFileName):
            raise FileNotFoundError(stationsFileName)

        with open(stationsFileName) as f:
            s = f.readline()
            while not s.startswith('==='):
                s = f.readline()
            while True:
                try:
                    name = f.readline().split('  ')[0]
                    ln = f.readline().split('coordinates:')[-1]
                    ln =[s[:-1].replace(' ','') for s in ln.split(',')][:3]
                    N, E, elev = [float(s) for s in ln]

                    x, y = wgs2rd(E, N)

                    #print(N, E, elev)
                    ln = f.readline().split(' ')[2:4]
                    code = int(ln[0])
                    #print(code)
                    ln = f.readline().split(' ')
                    #print(ln)
                    nyears = int(ln[-6])
                    years = [int(y) for y in ln[-1].split('-')]
                    yr_start, yr_end = years
                    self.data[name] = {'nr': code,
                                 'N': N, 'E': E, 'elev' : elev,
                                 'x': np.round(x), 'y': np.round(y),
                                 'nyears': nyears,
                                 'start' : yr_start,
                                 'endyr' : yr_end}
                    next(f)
                except:
                    print('Number of stations: ', len(self.data))
                    break
                
    def keys(self):
        return self.data.keys()

    def __getitem__(self, key):
        return self.data[key]

    def __str__(self):
        for k in self:
            print('\nstation ', k, ':')
            for kk in self[k]:
                print('    ', kk,' : ', self[k][kk])
        return ''

    def look_up(self, nr):
        for k in self:
            if self[k]['nr'] == nr:
                return k

    def plot(self, **kwargs):
        '''plots the meteo stations as dots on map

        kwargs
         ======
         ax : plt.Axis
         va : top|center|[bottom]  # horizontal label alignment
         ha : [left]|center|right  # vertical label alignment
         rotation : float # label rotatio in degrees anti clockwise
         fs : fontsize for label text of stations
         xlim : (xmin, xmax)
         ylim : (xmin, xmax)
         title: title
         xlabel : str at x axis
         ylabel : str at y axis
         other kwargs are passed on to plt.plot(... **kwargs)

         returns
         =======
         ax : plt.Axis
        '''
        ax     = kwargs.pop('ax', None)
        ha     = kwargs.pop('ha', 'center')
        va     = kwargs.pop('va', 'bottom')
        rotation = kwargs.pop('rotation', 0)
        fs     = kwargs.pop('fs', 8)
        fontsize = kwargs.pop(fs, 8)
        xlim   = kwargs.pop('xlim', None)
        ylim   = kwargs.pop('ylim', None)
        title  = kwargs.pop('title', None)
        xlabel = kwargs.pop('xlabel', None)
        ylabel = kwargs.pop('ylabel', None)
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_xlabel('xrd [m]')
            ax.set_ylabel('yrd [m]')
            ax.set_title('KNMI stations')
            ax.grid()

        if xlim   is not None: ax.set_xlim(xlim)
        if ylim   is not None: ax.set_ylim(ylim)
        if title  is not None: ax.set_title(title)
        if xlabel is not None: ax.set_xlabel(xlabel)
        if ylabel is not None: ax.set_ylabel(ylabel)


        for k in self.keys():
            plt.plot(self[k]['x'], self[k]['y'], 'o', **kwargs)
            ax.text(self[k]['x'], self[k]['y'], '{}_{}'.format(k, self[k]['nr']),
                    ha=ha, va=va, rotation=rotation, fontsize=fontsize)

        nederland(ax=plt.gca(), color='brown')
        return ax


class KNMI_hourstation:
    '''Class returning a KNMI weather station with hourly data.
    The data are stored in a pandas DataFrame held in its `data` attribute.
    The `info` property gives information on these data.

    The instances have attributes `nr` (=station number), `info` and `data`.

    TO 810507
    '''

    def __init__(self, knmiDataFileName):
        
        if not os.path.isfile(knmiDataFileName):
            raise FileNotFoundError(knmiDataFileName)

        with open(knmiDataFileName, "r") as f:
            # explore the file
            blanks = 0
            for iHdr, s in enumerate(f):
                if s == '\n': blanks += 1
                if s.startswith('# STN'):
                    break
            f.seek(0) # rewind to start reading knowing iHdr
            

            # get the meta info aa list of strings
            self._info = [next(f) for i in range(iHdr-1)]

            # Then use pandsa to read the dta
            self.data = pd.read_csv(knmiDataFileName,
                                header=iHdr - blanks,
                                skipinitialspace=True,
                                parse_dates=True,
                                index_col='YYYYMMDD',
                                dtype={'HH':int})
            self.nr = self.data['# STN'][0]

            # convert read data to mbar and mm
            self.data['p_air'] = self.data['P'] / 10.
            self.data['prec']  = self.data['RH'] / 10.
            self.data.loc[self.data['RH'] < 0, 'prec'] = 0.05

            # convert index to datetimes by adding hour to date
            self.data.index = [d + dt.timedelta(hours=float(h))
                    for d, h in zip(self.data.index, self.data['HH'])]

            # don't need these columns anymore
            self.data.drop(['# STN', 'HH'], axis=1)

    @property
    def info(self):
        for s in self._info:
            print(s)

    def plot(self, what, **kwargs):
        plot_station(self.nr, self.data, what, **kwargs)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def keys(self):
        return self.data.columns

    def index(self):
        return self.data.index

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return self.data.__str__()

    def __repr__(self):
        return self.data.__repr__()


class KNMI_daystation:
    '''Class returning a KNMI weather station with daily data.
    The data are stored in a `pandas` `DataFrame` held in the `data` attribute.
    The `info` property gives background information on these data.

    The instances have attributes `nr` (=station number), `info` and `data`/

    TO 180507
    '''

    def __init__(self, knmiDataFileName):
        
        if not os.path.isfile(knmiDataFileName):
            raise FileNotFoundError(knmiDataFileName)

        with open(knmiDataFileName, "r") as f:
            # explore the file
            blanks = 0
            for iHdr, s in enumerate(f):
                if s == '\n': blanks += 1
                if (s.startswith('STN') or s.startswith('# STN')) and iHdr > 10:
                    break
            # rewind to start reading
            f.seek(0)

            # get the meta info aa list of strings
            self._info = [next(f).replace('# ','') for i in range(iHdr-1)]

            # Then use pandsa to read the dta
            self.data = pd.read_csv(knmiDataFileName,
                                header=iHdr - blanks,
                                skipinitialspace=True,
                                parse_dates=True,
                                index_col='YYYYMMDD'
                                )
            cols = self.data.columns
            if '# STN' in cols:
                cols[cols.index('# STN')] = 'STN'
                self.data.columns = cols
            self.nr = self.data['STN'][0]
            self.data.drop('STN', axis=1)

            self.data['p_air'] = self.data['PG'] / 10.
            self.data['p_max'] = self.data['PX'] / 10.
            self.data['prec'] = self.data['RH'] / 10.
            self.data.loc[self.data['RH'] < 0, 'prec'] = 0.05
            self.data['evap']  = self.data['EV24'] / 10.

    @property
    def info(self):
        for s in self._info:
            print(s)

    def plot(self, what, **kwargs):
            plot_station(self.nr, self.data, what, **kwargs)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def keys(self):
        return self.data.columns

    def index(self):
        return self.data.index

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return self.data.__str__()

    def __repr__(self):
        return self.data.__repr__()


def plot_station(nr, data, what,  **kwargs):

    if not isinstance(what, list):
        what = [what]

    ax     = kwargs.pop('ax', None)
    xlabel = kwargs.pop('xlabel', None)
    ylabel = kwargs.pop('ylabel', None)
    xlim   = kwargs.pop('xlim', None)
    ylim   = kwargs.pop('ylim', None)
    title  = kwargs.pop('title', 'KNMI station {}'.format(nr))
    grid   = kwargs.pop('grid', True)

    if ax is None:
        fig, ax = plt.subplots()
        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        if xlim:
            if (not isinstance(xlim[0], datetime) or
                not isinstance(xlim[1], datetime)):
                raise ValueError('xlim must be (datetime, datetime)')
            else:
                ax.set_xlim(xlim)

            if ylim: ax.set_ylim(ylim)
        if title: ax.set_title(title)

    ax.grid(grid)

    for w in what:
        ax.plot(data.index, data[w], label=w, **kwargs)

    ax.legend(loc='best')
    return ax


if __name__ == '__main__':

    stationsFile = './data/KNMI_stations.txt'
    stations = KNMI_stations(stationsFile)
    stations.plot(xlim=(-50000, 300000), ylim=(300000, 650000))

    fName_h = './data/uurgeg_240_2001-2010.txt'
    fName_d = './data/etmgeg_350.txt'

    stat_h = KNMI_hourstation(fName_h)
    stat_d = KNMI_daystation(fName_d)

    #the data are stored in a pd.DataFrame as self.data

    stat_d.plot(what='p_air', title='barometer pressure (daily)')
    stat_h.plot(what='p_air', title='barometer pressure (hourly)')
    t1 = datetime(1990, 1, 1)
    t2 = datetime(2001, 1, 1)
    stat_d.plot(what=['prec', 'evap'], xlim=(t1, t2), ylabel='mm/d')

    stat_d.data['rch'] = stat_d['prec'] - stat_d['evap']

    stat_d.plot(what='rch', xlim=(t1, t2), ylabel='mm/d')


