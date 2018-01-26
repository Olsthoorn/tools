#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 13:31:19 2017

Reads diver times series from a directory and plots them between
desired dates.
Diver series have been converted to .txt files. Each series has at least
two columns with names DateTime and NAP
The dates with times are assumed in the dayfirst format '03/11/2017 20:13'
This is the way the come from an Excel file for example.

@author: Theo
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

NOT = np.logical_not
AND = np.logical_and


def dateFromStr(s, dayfirst=True):
    '''returns datetime object from date given as string
    parameters
    ----------
    s : str
        string readable as a legal data and time like  like '01/03/2017 20:13'
    dayfirst : bool
        if true, day/month/year h:min is assumed else year/month/day h:min
    '''
    s = s.replace('/',' ').replace(':',' ').split(' ')
    if dayfirst:
        s = s[2::-1] + s[3:]
        f = [int(f) for f in s]
    return datetime(*f)


class Diver:
    '''Stores the data sefies and knows how to plot it.'''


    def __init__(self, fName=None):

        if fName == None:
            return

        # First geneate object using super
        # Afterwards add extra properties
        ext = os.path.splitext(fName)[-1]
        if ext == '.txt':
            self.dframe= pd.read_table(
                             fName,
                             sep='\t',
                             skiprows=[0],
                             index_col='DateTime',
                             parse_dates=True,
                             dayfirst=True,
                             dtype={'NAP': float},
                             usecols=['DateTime', 'NAP'])
        elif ext == '.csv':
            self.dframe= pd.read_csv(
                         fName,
                         index_col='Date',
                         skipinitialspace=True,
                         parse_dates=True,
                         usecols=['Date','Waterstand gemeten'])
            self.dframe.columns =  ['NAP']
           #              na_values = {'Waterstand gemeten': [
           #                  '   NaN', '    NaN', '     NaN', '      NaN',
           #                  '       NaN', '        NaN']},
        else:
            raise ValueError('Unknown extension {}, use .txt or .csv'.format(self.ext))

        # clean off NaNs
        self.dframe = self.dframe[np.isnan(self.dframe['NAP'])==False]

        # clean off outliers:
        med = self.dframe['NAP'].median()
        std = self.dframe['NAP'].std()
        nap = self.dframe['NAP']
        self.dframe = self.dframe[
                AND(nap < med + 3 * std,
                    nap > med - 3 * std)]

        self.name = os.path.splitext(os.path.basename(fName))[0]
        self.ext  = os.path.splitext(fName)[-1]
        self.shortName = self.name[0] + self.name[-2:]

    def merge(self, other):
        '''merges other with self, making sure data don't overlap
        parameters
        ----------
        other : Diver
            must be another Diver, same type as self.
        '''
        if not isinstance(other, type(self)):
            raise ValueError("Can't merge, other not same type as sel")
            other = other.index() > self.index[-1]
            other = other.index() < self.index[0]
            self.dframe = pd.merge(self.dframe,
                                   other[other.index  > self.dframe.index[-1]])
            self.dframe = pd.merge(self.dframe,
                                   other[other.index < self.dframe.index[0]])

    def __getitem__(self, key):
        return self.dframe[key]
    def __setitem__(self, key, value):
        return self.dframe.__setitem__(key, value)
    def __iter__(self):
        return self.dframe.__iter__()
    def index(self):
        return self.dframe.index
    def keys(self, *args):
        return self.dframe.keys(*args)
    def values(self):
        return self.dframe.values
    @property
    def iloc(self):
        return self.dframe.iloc


    def plot(self, start=None, end=None, dayfirst=True, **kwargs):
        '''plots a timeline
        parameters
        ----------
        start : datetime obj or a string convertable to one
            start moment for plot
        end   : datetime obj or a string convertable to one
            end of plot
        dayfirst: bool
            if start or stop are given as str, dayfirst interprets
            str as a dd/mm/yyyy hh:mm string instead of yyy/mm/dd hh:mm
        additional kwargs
            ax : plt.Axes object or None
                axes object if None, axes will be generated.
            all other kwargs will be passed on to plot
        '''
        ax = kwargs.pop('ax', None)
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_xlabel('time')
            ax.set_ylabel('NAP [m]')
            plt.grid()
            plt.xticks(rotation=45, horizontalalignment='right',
                      verticalalignment='top')
            legend = True
        else:
            legend = False

        if 'title' in kwargs:
            ax.set_title(kwargs.pop('title'))

        if start is None:
            start = self.dframe.index[0]
        else:
            start = dateFromStr(start, dayfirst)

        if end is None:
            end = self.dframe.index[-1]
        else:
            end   = dateFromStr(end, dayfirst)

        ind = AND(self.dframe.index >= start, self.dframe.index <= end)

        ax.plot(self.dframe.index[ind], np.asarray(self.dframe['NAP'])[ind],
                label=self.name, **kwargs)
        if legend:
            plt.legend(loc='best')

from collections import UserDict

class Divers(UserDict):
    'Stores a series of time series and knows howto plot them.'''

    def __init__(self, files=None, n=None):
        '''returns a UserDict of diverSeries
        parameters:
            files: list
                list of files with .txt or .csv extension
                each holing a single diver time series
            The txt files are assumed ot have columns 'DateTime' and 'NAP'
            The csv files are assumed to hae a column 'Date' and 'Waterstand gemeten'
        '''

        self.data = dict()

        if files is None:
            return

        if n is not None:
            files = files[:n]
        if len(files) == 0:
            raise ValueError('No diver files read')
        for f in files:
            name = os.path.splitext(os.path.basename(f))[0]
            print('reading ', f, )
            if name in self.keys():
                self[name].merge(Diver(os.path.join(diverdir, f)))
            else:
                self[name] = Diver(os.path.join(diverdir, f))
        print('... done')

    def keys(self):
        return self.data.keys()
    def values(self):
        return self.data.values()


    def plot(self, start=None, end=None, dayfirst=True, **kwargs):
        '''plots all diverseries on a single time-chart
        parameters
        ----------
        start : datetime obj or a str convertable to one
        end   : datetime obj or a str convertable to one
        dayfirst : bool
            if True interpret start end as 'dd/mm/yyy hh:mm'
        kwargs:
            all kwargs are passed on to the plotting routine
            for a single series
        '''
        ax = kwargs.pop('ax', None)
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_title(kwargs.pop('title','Diver series'))
            ax.set_xlabel('time')
            ax.set_ylabel('NAP m')
            plt.grid()
            plt.xticks(rotation=45, ha='right', va='top', fontsize=8)
            kwargs['ax'] = ax
        for key in self.data:
            self[key].plot(start, end, dayfirst, **kwargs)
        ax.legend(loc='best')


if __name__ == '__main__':
    'Example usecase'

    HOME = '/Users/Theo/GRWMODELS/python/DEME-juliana/analyse'
    os.chdir(HOME)
    diverdir = '../diverTxtData'
    diverFiles = [os.path.join(diverdir, f) for f in os.listdir(diverdir) if f.endswith('.txt')]
    diverdir = '../diverCsvData'
    diverFiles += [os.path.join(diverdir, f) for f in os.listdir(diverdir) if f.endswith('.csv')]

    dfile = 'mydivers.pckl'
    mustread = True

    if mustread:
        divers = Divers(diverFiles, n=None)
        with open(dfile, 'wb') as file:
            pickle.dump(divers, file)
    else:
        with open(dfile, 'rb') as file:
            divers = pickle.load(file)

    for k in divers.keys():
        divers[k].plot(start='01/01/2000', end='31/12/2020')

    divers.plot(start='31/03/2016', end='31/03/2017')

    d = divers['PBU060']
    d.plot()

    L = ['PBGRA2', 'PBGRA3', 'PBGRA4', 'PBU060', 'PBU031',  'PBU059']

    L = ['PBGRA2', 'PBGRA3', 'PBGRA4', 'PBU060',            'PBU059']
    d = Divers()
    for k in L:
        d[k] = divers[k]
    d.plot()
