#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 23:33:55 2017

@author: Theo
"""

__all__ = ['KNMI_daystation',  'KNMI_hourstation',  'KNMI_stations', 'plot_station']
 
tools = '/Users/Theo/GRWMODELS/python/tools/'

import sys
import os

if not tools in sys.path:
    sys.path.insert(0, tools)

import numpy as np
import pandas as pd
import datetime as dt
from collections import UserDict
import matplotlib.pyplot as plt
from datetime import datetime
from coords.transformations import wgs2rd, rd2wgs
from etc import newfig


# %%
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

        #nederland(ax=plt.gca(), color='brown')
        return ax

# %%
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

# %%
class KNMI_daystation:
    '''Class returning a KNMI weather station with daily data.
    The data are stored in a `pandas` `DataFrame` held in the `data` attribute.
    The `info` property gives background information on these data.

    The instances have attributes `nr` (=station number), `info` and `data`/

    TO 180507
    '''

    def __init__(self, knmiDataFileName, start=None, end=None):
        """Return KNMI station DataFrame, with only columns 'p_air', 'p_max', 'prec', 'evap' in mm/d.
        
        Lines with no data are not dropped.
        
        Parameters 
        ----------
        knmiDataFileName: str (path)
            Path to the file of downloaded data text file for a particular KNMI "weerstation".
        start: np.datetime64 object 
            desired start of the data 
        end: np.datetime64 object 
            disired end of the data
        """
        
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
            data = pd.read_csv(knmiDataFileName,
                    header=iHdr - blanks,
                    skipinitialspace=True,
                    parse_dates=True,
                    index_col='YYYYMMDD'
                    )
            col_labels_to_drop = data.columns[1:]
            data['p_air'] = data['PG'] / 10.
            data['p_max'] = data['PX'] / 10.
            data['prec']  = data['RH'] / 10.
            data.loc[data['RH'] < 0, 'prec'] = 0.05
            data['evap']  = data['EV24'] / 10.
            
            self.data = data.drop(col_labels_to_drop, axis=1)
            if start:
                self.data = self.data.loc[self.data.index.values >= start]
            if end:
                self.data = self.data.loc[self.data.index.values <= end]

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

# %%
def plot_station(title, xlabel, ylabel, data=None, columns=None, xlim=None, ylim=None, xscale=None, yscale=None, figsize=(12, 8), **kwargs):
    """From the wheather station plot the desired columns. 
    
    Parameters
    ----------
    title, xlabel, ylabel: str, str, str
        obvious
    data: KNMI_station object of pd.DataFrame with np.datetime64 as index 
        The data
    columns: list
        the names of the columns that should be plotted.
    xlim: two np.datetime64 objects 
        start and end of plot 
    ylim: two float 
        ylim 
    xscale, yscale: str 
        either 'log' or 'linear', leave None for 'linear' 
    figsize: tuple of 2 floats 
        desired figsize in inches
    kwargs: dict 
        other parmeters to pass to pyplot.plot
    """

    if not isinstance(columns, list):
        columns = [columns]

    fig, ax     = plt.subplots()
    fig.set_size_inches(figsize)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xlim:
        ax.set_xlim(xlim)
        assert isinstance(xlim[0], np.datetime64) and isinstance(xlim[1], np.datetime64), "xlim must be two values of type np.datetime64"
    if ylim: ax.set_ylim(ylim)
    if xscale: ax.set_xscale(xscale)
    if yscale: ax.set_yscale(yscale)
    ax.grid(True)

    if xlim:
        data = data.loc[np.logical_and(data.index >= xlim[0], data.index <= xlim[1])]
        
    for column in columns:
        ax.plot(data.index, data[column], label=column, **kwargs)

    ax.legend(loc='best')
    return ax

def show_neerslag_stations_groningen():
    neerslagstations = pd.read_table('./Neerslagstations_mei2021_1d.csv', header=0, sep=',', decimal='.', index_col=0)
    ns, xcol, ycol = neerslagstations, 'STN_POS_X', 'STN_POS_Y'

    xlim = (225, 245)
    ylim = (570, 590)

    groningen = neerslagstations.loc[ np.logical_and(ns[xcol] > xlim[0], ns[xcol] < xlim[1], ns[ycol] > ylim[0], ns[ycol] < ylim[1])]

    ax = newfig("Neeslagstatons KNMI ), rond Groningen", "xRD km", "yRD km", xlim=xlim, ylim=ylim)

    for stn in groningen.index:
        station = groningen.loc[stn]
        x, y, name = station['STN_POS_X'], station['STN_POS_Y'], station['LOCATIE']
        ax.plot(x, y, 'r.')
        ax.text(x, y, "  {} {}".format(stn, name))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.show()

# %%
if __name__ == '__main__':

    os.chdir('/Users/Theo/GRWMODELS/python/tools/KNMI/')

    stationsFile = './data/KNMI_stations.txt'
    stations = KNMI_stations(stationsFile)
    stations.plot(xlim=(-50000, 300000), ylim=(300000, 650000))

    start = np.datetime64("2016-01-01")
    end   = np.datetime64(datetime.now())
    xlim = (start, end)

    stn = 350 # Gilze_rijen
    stn = 280 # Eelde

    #fName_h = './data/uurgeg_240_2001-2010.txt'
    fName_d = './data/etmgeg_350.txt' # Gilze-rijen
    fName_d = './data/etmgeg_{}.txt'.format(stn) # Eelde
    
    #stat_h = KNMI_hourstation(fName_h)
    station = KNMI_daystation(fName_d, start=start)
    data = station.data.loc[station.data.index >= start]

    #plot_station('barometer pressure (daily) ' + fName_d, "time", "mbar", data=data, columns='p_air', xlim=xlim)
    #stat_h.plot(what='p_air', title='barometer pressure (hourly)')
    
    plot_station(fName_d, "time", "mm/d", data=data, columns=['prec', 'evap'], xlim=xlim)

    data['rch'] = data['prec'] - data['evap']

    plot_station(fName_d, "time", "mm/d", data=data, columns='rch', xlim=xlim)
    
    data.to_pickle('eeld280.pkl')
    data_back = pd.read_pickle('eeld280.pkl')

# %%
