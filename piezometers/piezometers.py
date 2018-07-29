#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 11:16:31 2018

Piezometer tools

Analysis of the piezometers.

@author: Theo Olsthoorn 20180601
"""
__all__ = ['Piezom', 'Piezoms', 'Calib', 'Calibs']

import os
import collections
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shape
import logging
import etc
import datetime


# timedelta to float
# example td2float(piezom['PBGRA3'].dd.index)
td2float = lambda td : np.asarray(td, dtype=float)/1e9/86400
t2str    = lambda t : t.__str__()

# back from timedeltas to seconds
td2s = lambda td: np.timedelta64(int(td * 86400), 's')
Td2s = lambda td: np.asarray(np.asarray(td * 86400, dtype=int), dtype='timedelta64[s]')

# Convert times to utc.
cest2utc  = lambda index: index - np.timedelta64(2, 'h') # not used
cet2utc   = lambda index: index - np.timedelta64(1, 'h') # not used
elli2CEST = lambda index: index - np.timedelta64(2, 'h') # used


AND = np.logical_and

pz_logger = logging.getLogger()

#%%

def outliers(df, cols=None, fence=3.0):
    '''Report outliers in a pd.DataFrame column Col.

    parameters
    ----------
    df : pd.DagaFrame
        pandas DataFrame to be inspected.
    name : str
        identifier used in reporting. (Default is 'name??')
    cols : str or series of str
        column names in df to check for outliers.
        names must be columns of df.
    fencee : a float
        fence limit such that
          < median - fence  * (Q3 - Q1), > median + fence * (Q3 - Q1)

    returns
    -------
        None

    >>> df = pd.DataFrame(index=np.arange(10), data=np.random.randn(10, 3), columns=['a', 'b', 'c'])
    >>> df.loc[[2, 5], 'b'] = [2.4, -10]
    >>> col = 'b'
    >>> get_outliers(df[col])  # ds[col] is a pd.Series not a pd.DataFrame !


    @ TO 2018-06-22 12:30
    '''

    D = df

    if isinstance(cols, str):
        cols = [cols]
    for col in cols:
        if col not in D.columns:
            raise KeyError('Column <{}> not in data frame with columns\n[{}]'
                           .format(col, ', '.join(D.columns)))


    D['med'] = D.rolling(9).median()
    D.plot()

    if False:
        Q3 = D.quantile(0.75)[cols]
        Q1 = D.quantile(0.25)[cols]
        dQ = Q3 - Q1
        med = D.median()[cols]

        fence = med - fence * dQ, med + fence * dQ

        outliers  = np.logical_or(D[cols] < fence[0], D[cols] > fence[1])

        if np.any(outliers):
            print('{} is outside ({} * inner_quartile_range):'.format(df.name, fence))
            I = np.where(outliers)[0]
            for i in I:
                i0, i1 = max(0,i-5), min(len(D),i+6)
                print('{} {}:{}'.format(df.name, i0,i1))
                K = np.arange(i0, i1, dtype=int)
                print(D.iloc[K, :])
                print()


#%%
def theis_analysis(obj, t0dd=None, well=None, col='measured'):
    '''Return steepest gradient per logcycle and t where s=0.

    This is based on the simplified Theis solution that yield
    straight drawdown lines of half-logarithmic time scale.

        $s = \frac {Q} {4 \pi kD} ln(\frac {2.25 kD t} {r^2 S})$

    uses self.dd, the drawdown DataFrame

    parameters
    ----------
    obj: piezoms.Piezom or piezoms.Calib object
        piezometer, having the drawdown data on board
        as the pd.DatFrame self.dds
    well:  dict {'x': xwell, 'y': ywell, 'Q': Qwell} None
        The location and extraction used in the analysis.
        If None, kD and S will not be computed.
    plot: bool
        If True then plot the log approximation and at the dd curves.

    returns
    -------
    Dict containing the result of the analys allowing to plot
    the straignt drawdown approximation tangent to the curved drawdown
    as obtained in simplified analysis of Theis drawdown using the
    log approximation.

    `dd_logcycle` : float
        Drawdown (or head change) per log10 cycle [m]. (Sim)
    `t_dd0` : float (t,  not logt)
        The time since start pumping at which the straight drawdown curve
        hits dd=0. (Simlified Theis analysis.)
    `t_at_maxGrad` : float (t, nog logt)
        Time where the straigt line apporximation intersects the curved line.
    `dd_at_maxGrad` : float
        The drawdown at this time.
    `maxGrad` : float (ds / dlogt)
        The maximum gradient is the gradient of the straight dd vs logt
        line. Where the gradiet of the measured curved line is maximum
        is taken as the tangent point of the straigh line.
    '''

    #import pdb; pdb.set_trace()

    if isinstance(col, (list, tuple)):
        raise ValueError('''Col must be a str corresponding to one of the
                         header in self.dhs, preferably 'measured''')

    if not col in obj.hds.columns:
        raise ValueError('''Col must be one of the column headers
                in obj.hds DataFrame: [{}]'''.format(', '.join(obj.hds.columns)))


    D = obj.dds.copy().dropna(axis=0, how='any')  # the drawdown pd.DataFrame

    logt = np.log10((D.index - t0dd) / pd.Timedelta('1 days')) # in days

    # step size at log time scale
    lg_step       = np.log10(1.25)

    # discrete time points at equal distance on log time scale
    logt_pnts  = np.arange(logt[0], logt[-1], lg_step)
    dd_pnts    = np.interp(logt_pnts, logt, D[col] - D[col][0])

    # gradients between the discrte points
    gradients  = np.diff(dd_pnts) / np.diff(logt_pnts)

    # index of highest gradient of dd on log time scale.
    imax = gradients.argmax()

    # max gradient iself
    maxGrad = gradients[imax]

    # dd at maximum gradient and log10(t) of maximum gradient
    dd_maxGrad = 0.5 * (  dd_pnts[imax]   + dd_pnts[imax + 1])
    lt_maxGrad = 0.5 * (logt_pnts[imax] + logt_pnts[imax + 1])

    # log time where straight line of maximum gradient passes zero drawdown
    lt_dd0    = lt_maxGrad - dd_maxGrad / maxGrad

    # drawdown per log cycle
    dd_logcycle = maxGrad

    # store this in dict.
    theis = {
        't0dd'       : t0dd, # absolute reference, pd.Timestamp
        'dd_logcycle': dd_logcycle,
        't_dd0'      : 10**(lt_dd0),  # in days (floats)
        't_maxGrad'  : 10**(lt_maxGrad), # in days (floats)
        'dd_maxGrad' : dd_maxGrad,
        'maxGrad'    : maxGrad,
        'dd_max'     : np.max(D[col])}

    if well is not None:
        try:
            xWell = well['x']
            yWell = well['y']
            Q     = well['Q']
        except:
            raise ValueError('You must provide well=(t, xwell, ywell, Qwell)')

        # kD and S
        R = obj.distance(xWell, yWell)

        kD = Q / (4 * np.pi * theis['dd_maxGrad'])

        S  = 2.25 * kD * theis['t_dd0'] / R ** 2

        theis.update({'r': R, 'kD': kD, 'S': S, 'well': well})

        pz_logger.info("Theis for {} {}: kD = {:10.4g} and S = {:8.4g}"
                       .format(obj.name, col, kD, S))
    else:
        pz_logger.info('Theis for {} {}, no well -> no kD, no S'
                       .format(obj.name, col))

    return theis # a dict


#%%
class Base_Piezom:
    '''Base class for piezometer. It shoulde not be instantiate directly

    __init__ should be defined in the sub_class

    @TO 20180620
    '''


    def msre(self, dd=False):
        '''Return rean root squared error.'''
        if dd==False:
            return (self.hds['diff'] ** 2).mean().sqrt()
        else:
            return (self.drwdn['diff'] ** 2).mean().sqrt()


    def mae(self, dd=False):
        '''Return mean absolute error.'''
        if dd==False:
            return self.hds['diff'].abs().mean()
        else:
            return self.drwdn['diff'].abs().mean()


    def merr(self, dd=False):
        'Return mean error.'''
        if dd==False:
            return self.hds['diff'].mean()
        else:
            return self.drwdn['diff'].mean()

    def distance(self, x0, y0):
        '''Return the distance form self to point x0, y0 (e.g. a well)

        parameters
        ----------
        x0 : float
            x coordinate of point
        y0 : float
            y coordinate of point
        '''
        return np.sqrt((self.meta['x'] - x0) ** 2 + (self.meta['y'] - y0) **2 )



    def drwdn(self, t0dd=None, well=None, theis=False, out=True, cols=None):
        '''Return drawdown with respect to timestamp t0dd.

        In all cases, the resutting drawdown DataFrame is stored as self.dds

        t0dd is stored in self.meta['t0dd']

        If theis == True, then the rsults of the theis analysis, i.e. dd per
        logcycle and the time at which the log dd is 0 are stored as
        self.meta['theis']

        If well is not None, then kD and S are computed and stored along with theis.

        The drawdown is computed for all columns in self.hds and stored in self.dds.

        The argument col is only used in thei_analysis.


        Parameters
        ----------
        t0dd : np.datetime64 or str representing a datatiem64 object
            start of drawdown, absolute time like '2018-06-20 09:05'
        well: dict {'x': xWell, 'y': yWell, 'Q': Qwell}
            well should contain (xwell, ywell, Q), Q starts at t0dd
            If well is None, kD and S will not be computed.
        theis : bool
            whether or not to carry out the analysis according to the
            simplief formulat of Theis
                s = Q/(4 pi kD) ln((2.25 kD t) / (r**2 S))
        out : bool
            if True return dd DataFrame and theis of theis==True
            if False return None
        cols : list of str or str
            names of columns in the created self.dds that will be used in
            theis_analysis when it is invoked. Default = 'measured'

        Returns
        -------
            None
        '''

        #import pdb; pdb.set_trace()

        if t0dd is None:
            raise ValueError(
                    't0dd must be specified when calling self.drwdn()')

        if not isinstance(t0dd, (str,
                            pd.Timestamp, np.datetime64, datetime.datetime)):
            raise ValueError(
                't0dd must be a pd.Timestamp or a str representing one, not {}'
                             .format(str(t0dd)))

        t0dd = pd.Timestamp(t0dd)

        self.meta['t0dd'] = t0dd

        # To genrerate drawdown strat by copying the heads
        # and truncate anything before t0dd
        self.dds = self.hds.loc[self.hds.index > t0dd].copy()

        # Get the values in self.hds for all columns at t0dd
        att = self.attime(t=t0dd, dd=True)

        # Subtract the values at t0dd, not that att is a dictionary
        for _col_ in self.dds.columns:
            self.dds[_col_] -= att[_col_]

        # store the t0dd for this drawdown object
        if theis: # in case a Theis approximation analysis is requested:
            # Do theis analysis and store retulting dict in self.meta
            # Theis uses the default col name "measured"

            if cols is None:
                cols = self.dds.columns
            elif isinstance(cols, str):
                cols = [cols]

            self.meta['theis'] = dict()
            for col in cols:
                if not col=='diff':
                    self.meta['theis'][col] = theis_analysis(
                                self, t0dd=t0dd, well=well, col=col)

            # In case user wants to catch it directly return this Theis object
            return self.dds, self.meta['theis']

        if out: # In case theis was not requested and user want output return the drawdown DataFrame
            return self.dds
        else:
            return # nothing, self.dds is generated inplace


    def attime(self, t=None, dd=False, label=None):
        '''Return adict with the values interpolated at time t.

        Atttime resturns the heads or drawdown of the object at the given
        time as geneated by linear interpolation of time.
        Therefore, the time must always be a pd.Timestamp. You can
        always compute a timedelta index by subtrating a Timestamp from the
        axis.

        the resulting data will be added to the self.att dictionary if it
        exists, else it's generated. This dictionary allows collecting
        multipel attime results, borth for hds as for dd.

        The key in the self.att dictionary is label. If not specified a
        int is used, 0 if the self.att dict is generated or len(self.att)
        otherwise. It is usefull to pass your own label to allow easy access
        to the stored data at a later time.

        attime is compute for all columns are used.

        if dd is True than the drawdowns are computed at the given time.
        Else the hds are computed. Drawdowns can only be computed of the
        objects already contains a dds DataFrame, which can be generated
        by calloing self.drwdn or along with the generation of the container.

        parameters
        ----------
            t  : pd.Timestamp object or str str representing one.
                time that head or drawdown is resired.
            dd: bool
                if True the drawdown is computed else the head at t.
        returns
        -------
        a dict: {'t' : t, 'dd': True|False, col1: value, col1: value, ... }
            where col1, col2 etc are the names of the columns used.
        at the same time the self.att will be update with this dict like so
            self.att[label] = {'t': t, 'dd': dd, col1: value, col2, value ...}

        TO 20180709 update for calibration
        '''
        t = pd.Timestamp(t)

        # make sure self.att exists as dict and label is not None
        if not 'att' in self.__dict__:
            self.att = dict()

        if label is None:
            label = len(self.att)

        self.att[label] = {'t': t, 'dd': dd}

        if dd == False:
            DF = self.hds
        else:
            DF = self.dds

        cols = DF.columns

        td_f     = (t        - DF.index[0]) / pd.Timedelta(1, 'D')
        td_ind_f = (DF.index - DF.index[0]) / pd.Timedelta(1, 'D')
        for col in cols:
            self.att[label][col] = np.interp(td_f,
                        td_ind_f[~np.isnan(DF[col].values)],
                        DF[col].values[~np.isnan(DF[col].values)])

        return self.att[label]


    def apply(self, fun):
        '''Return df with function fun applied on all its columns.

        works like pd.DataFrame.apply()

        Returns
        -------
            pd.Series with the function fun applie on self.hds
            TODO: check

        @ TO 2018-06-22 16:02
        '''
        funName = str(fun).split(' ')[1]

        df = self.hds.apply(fun).to_frame # TODO check, looks strange
        df.columns = [funName]

        return df.T



    def plot(self, cols=None, tstart=None, tend=None, dd=False,
                             t0dd=None, well=None, theis=None, **kwargs):
        '''Plot self.data[what].

        parameters
        ----------
            what : str
                header of column to be plotted ['computed', 'measured', 'diff']
            dd : bool
                whether to plot heads or head-change (drawdown)
            kwargs: dict
                parameters to be passed to ax.set and plot.
                If kwargs['ax'] is plt.Axes then use it else create it

        returns
        -------
            None
        '''
        ax = kwargs.pop('ax', None)

        # Make sure self.dds exists.
        if dd == True:
            if t0dd is None:
                raise ValueError('''Because drawdowns are computed on the fly,
                    you must specify t0dd.
                    Specify t0dd as a pd.Timestamp or a str representing one.''')

            try:
                t0dd = pd.Timestamp(t0dd)
            except:
                raise ValueError('''Can't handle t0dd given as {}.
                        t0dd must be convertable to pd.Timestamp.'''.format(t0dd))

            # Generate drawdowns on the fly to ensure t0dd is up to data
            self.dds = self.drwdn(t0dd=t0dd, well=well, theis=theis, out=True, cols=cols)[0]

            DF = self.dds
            plot_index = (DF.index - t0dd) / pd.Timedelta(1, 'D')
        else:
            DF = self.hds
            plot_index = DF.index

        if cols is None:
            cols = list(DF.columns)
        elif not isinstance(cols, (tuple, list)):
            cols=[cols]

        if ax is None:
            legend = True

            fig, ax = plt.subplots()
            fig.set_size_inches(kwargs.pop('size_inches', size_inches))

            prep = 'Heads' if dd == False else 'Drawdowns'

            title='{} piezom. {} [{}] at x={:.0f}, y={:.0f} layer={}'\
                        .format(prep, self.name, ', '.join(cols),
                                self.meta['x'], self.meta['y'], self.meta['iz'])

            ax.set_title(title)
            ax.set_xlabel(kwargs.pop('xlabel','time'))
            ax.set_ylabel(kwargs.pop('ylabel',prep + '[m]'))
            ax.set_xscale(kwargs.pop('xscale', 'linear'))
            ax.set_yscale(kwargs.pop('yscale', 'linear'))
            if 'xlim' in  kwargs: ax.set_xlim(kwargs.pop('xlim'))
            if 'ylim' in  kwargs: ax.set_ylim(kwargs.pop('ylim'))
            ax.grid(True)
        else:
            legend=False


        for col, ls in zip(cols, etc.line_cycler()):

            kwargs.update(ls)

            ax.plot(plot_index, DF[col], label='{} {}'.format(self.name, col), **kwargs)

        if theis:

            for col in cols:
                Th = self.meta['theis'][col]
                lgt = np.linspace(np.log10(Th['t_dd0']),
                                  np.log10(Th['t_dd0']) + 1., 21)
                dd = np.linspace(0, Th['dd_logcycle'], 21)

                if len(lgt) > 0:
                    kwargs.update({'ls': '-', 'lw': 0.5, 'marker': None})

                    ax.plot(10 ** lgt, dd, label=self.name + ' Th/Jac approx.', **kwargs)

                    kwargs.update({'ls': '', 'marker': 'o'})

                    ax.plot([Th[ 't_maxGrad'], Th['t_dd0' ]],
                            [Th['dd_maxGrad'], 0],
                            label=self.name + ' tan./t0dd', **kwargs)

        if legend:
            ax.legend(loc='best')
        return



#%%
class Base_Piezoms(collections.UserDict):
    '''Base class for piezometer collection. It shoulde not be instantiate directly

    __init__ should be defined in the sub_class

    @TO 20180620
    '''



    def drwdn(self, t0dd=None, well=None, theis=False, cols=None):
        '''Sets the drawdown for all piezometers as self[name].dds

        time of start drawdown is saved in self[name].mta['t0dd']

        if theis, then results are saved as dict in self[name].meta['theis']

        if well is specified, then theis will also compute kD and S

        the resulting drawdown dataframes will be stored as

        self[name],dds with index as timedeltas relative to t0dd.

        parameters
        ----------
            t0dd: np.datetime64 object or str representing it
                absolute time from which drawdown is computed.
            well: dict {'x' : xwell, 'y': ywell, 'Q': Qwell}
                if  None, then kD and S cannot be computed.
            theis: bool
                if True, compute the maximum gradient per log cycle of time
                and the time at which this tangent crosses zero drawdown.
                kD and S will be computed of well is specified.
                Computation is based on simplified logarithmic Theis drawdown:
                    s = Q/(4 pi kD) ln ((2.25 kD t) / (r**2 S))
            cols: list of str or str
                names of columns on which to apply theis analysis.
        '''
        missed=[]
        for name in self:
            try:
                self[name].dds = \
                    self[name].drwdn(t0dd=t0dd, well=well, theis=theis,
                        out=False, cols=cols)
            except:
                missed.append(name)

        pz_logger.info("Drawdowns are saved in self[name].dds for {} piezometers."
                    .format(len(self) - len(missed)))
        if missed:
            pz_logger.debug('Could not compute drawdown for [{}]'.format(', '.join(missed)))
        pz_logger.info("t0dd <{}> saved in self[name].meta['t0dd']"
                    .format(self[name].meta['t0dd']))
        if theis:
            pz_logger.info("Theis analysis results are saved in self[name].meta['theis']")
        return


    def attime(self, t, dd=False, shapefile=None, label=None):
        '''Return head (interpolated) at t.

        parameters
        ----------
            t  : pd.Timestamp object or a str representing one
                time that head or drawdown is resired.
                self.dds must exist to compute the drawdown at the
                spefied time. It can be generated using the
                drwdn method of self or along with the generation of its
                container by specifying t0dd.
            dd: bool
                if True compute drawdown else compute head
            cols: str or list of str
                name of column(s) in hds or dds DataFrame to use, e.g.:
                    cols = ['meausured', 'computed', 'diff'] or
                    cols = 'computed'
            label: str
                label allows retrieving the att from every piezom or calib
                using self[name].att[label]. Of not specified then increasing
                integers will be used automatically
            shapefile : str or None
                if not None, shapefile is the path to the shapefile to be generated.
                if None, then don't generate a shapefile.
        returns
        -------
        att : dict with attime values for all columns
            with key self[name].name and fiels
            ['x', 'y', 'z', 'iz', 't', 'dd']
        '''

        att = dict()
        missed=[]
        for name in self:
            try:
                att[name]= self[name].attime(t, dd=dd, label=label)
            except:
                missed.append(name)

        if missed:
            pz_logger.debug("Couln't do attime for [{}]".format(', '.join(missed)))

        if shapefile:
            if not os.path.isfile(shapefile):
                raise FileNotFoundError("Can't find file {}".format(shapefile))

            shape.dict2shp(att, shapefile, shapetype='POINT', xy=('x', 'y'))
            pz_logger.info('Shapefile <{}> generated, containing {} points'.format(shapefile, len(att)))

        return att


    def plot(self, cols=None,
             tstart=None, tend=None, t0dd=None, dd=False, well=None, theis=False, **kwargs):
        '''Plot the measurements, not the drawdown h(t).


        parameters
        ----------
        cols : list of str
            names of DataFrame columns to be plotted e.d:
                cols=['computed', 'measured', 'diff']
        tstart : np.datetime64 obj or a str that represents a datatime.
            starting time of plot or moment relative to which drawdown is computed
        tend: same as tstart
            ending time of plot. Ignored in case dd=True
        dd : bool
            plot drawdown if True, heads if False
        kwargs
        ------
        kwargs : plot options
            all kwargs are pased on to plt.plot
        returns
        -------
            None
        '''

        if not cols is None and not isinstance(cols, (tuple, list)):  cols = [cols]

        fig, ax = plt.subplots()
        if 'size_inches' in kwargs:
            fig.set_size_inches(kwargs.pop('size_inches'))

        typeStr = str(type(self)).replace("'>","").split('.')[-1]

        if cols is None:
            ax.set_title('{}, {}'.format(typeStr, 'column names unspecified'))
        else:
            ax.set_title('{}, {}'.format(typeStr, ', '.join(cols)))
        if dd == False:
            ax.set_xlabel('date and time')
            ax.set_ylabel('m NAP')
        else:
            ax.set_xlabel('time since start [d] @ {}'.format(t0dd))
            ax.set_ylabel('head change [m]')
        if 'xlim'     in kwargs: ax.set_xlim(    kwargs.pop('xlim'))
        if 'ylim'     in kwargs: ax.set_ylim(    kwargs.pop('ylim'))
        if 'xscale'   in kwargs: ax.set_xscale(  kwargs.pop('xscale'))
        if 'yscale'   in kwargs: ax.set_xscale(  kwargs.pop('yscale'))
        if 'fontsize' in kwargs: ax.set_fontsize(kwargs.pop('fontsize'))

        ax.grid()
        #import pdb; pdb.set_trace()
        missed = []
        for name, ls in zip(self, etc.linestyle_cycler()):
            try:
                self[name].plot(cols=cols, dd=dd,
                        t0dd=t0dd, tstart=tstart, tend=tend,
                        well=well, theis=theis, **ls, ax=ax, **kwargs)
            except Exception as err:
                pz_logger.debug("Couldn't plot []: {}".format(name, err))
                missed.append(name)

        ax.legend(loc='best')
        if missed:
            print('Not plotted: [{}]'.format(', '.join(missed)))


    def apply(self, funs, tstart='1900-01-01', tend='2250-01-01'):
        '''Return df with function fun applied on all its columns.

        works like pd.DataFrame.apply()

        parameters
        ----------
        funs :  list of ufunc to apply (single string ok)
            the function to apply in turn ['np.var', 'np.std', ...]
        tstart: pd.Timestamp or str representing one
            start time to limit application range
        tend: pd.Timestamp or str represending one
            end time to truncate application range

        Returns
        -------
            pd.DataFrame with index the name of the piezometer with
            with the functions applied to all columns

        @ TO 2018-06-22 16:02
        '''

        #import pdb; pdb.set_trace()

        # Verify that no ufuncs are used.
        for f in funs:
            if 'ufunc' in str(f):
                raise ValueError(
                '''Don't' use {}. You can only use functions that
                   reduce, like np.sum, np.var, np.mean,
                   not ufuncs like np.sin, np.log etc.'''.format(str(f)))


        funsplit = lambda fun : str(fun).replace("'"," ").replace('  ',' ').split(" ")[1]

        #import pdb; pdb.set_trace()

        if not isinstance(funs, (tuple, list)):
            funs     = [funs]

        names    = list(self.keys()) # list of piezometer names
        funNames = [funsplit(fun) for fun in funs]
        columns  = self[names[0]].hds.columns # column used in each piez[name].hds

        Dout = pd.DataFrame(columns=pd.MultiIndex.from_product([funNames, columns]))

        missed = []
        for fun, funName in zip(funs, funNames):
            for name in names:
                try:
                    # Get the hds of this piezom and truncate time
                    D      = self[name].hds
                    ts = D.index[ 0] if tstart is None else pd.Timestamp(tstart)
                    te = D.index[-1] if tend   is None else pd.Timestamp(tend)
                    values = D[AND(D.index >= ts, D.index <= te)].apply(fun).values

                    Dout.loc[name, funName] = values
                except Exception as err:
                    pz_logger.info("Can't handle {} because {}".format(name, str(err)))
                    missed.append(name)

        if missed:
            print('Piezometers for which not all functions could be computed (perhaps empty dataFrame)')
            print(missed)

        return Dout.swaplevel(0, 1, axis=1).sort_index(level=0, axis=1)



    def to_shape(self, t, shapefile=None, dd=False):
        '''Generate a point shapefile with the head or drawdown data.

        parameters
        ----------
            itime: int
                index in drawdown index to extract drawdown for given time.
            shapefileNm : str
                the name of ths shapefile to generate.
            fldnms : list of str
                fldnams form piez[names].dd to use
            dd : bool
                if True drawdown will be saved else heads will be saved
        returns
        -------
            None
        '''

        self.attime(t, dd=dd, shapefile=shapefile)


#%%
class Calib(Base_Piezom):
    '''Return a Calib_hd object

    This class inherits all methods from its base_class. It only provides
    its __init__ constructor.

    The Calib_hd object is to compare computed with measured heads. It does so
    at the times tha Modflow used during the simlation. The measured
    column in the dataframe is computed by interpolating in the measured
    heads dataframe. This way both are available at the same times.

    The columns used are 'measured', 'computed', 'diff' which means
    computed minus measured heads.

    The index is the absolute time as timestamps ('np.datetime64[ns']) obtained
    from the fdm.mfgrid.stressPeriod object of Modflow HDS objectdirectly.

    '''

    def __init__(self, piez=None, gr=None, HDS=None, t0sim=None,
                 t0dd=None, well=None, theis=None):
        '''
        parameters
        ----------
        gr : fdm.mfgrid.Grid
            fdm mesh
        HDS : flopy.binary head file object
            modflow-simulated heads
        piez : dict
            piezometer object, parent for Calib
        t0hds: np.datetime64 obj or str representing it
            starting time of simulation, to link MODFLOW times to abs. datetime.
        t0dd : np.datetime64 objec, or str representing it
            starting time of drawdown, must be >= t0hds


        @TO 180618
        '''

        #import pdb; pdb.set_trace()

        self.name = piez.name
        self.meta = piez.meta.copy()
        self.gr = gr

        if t0sim is None:
            raise ValueError(
                    '''You must specify t0sim, the start time of the
                    Modlfow simulation, like t0sim='2018-06-20 22:23''')
        else:
            t0sim = pd.Timestamp(t0sim)

        self.hds = self.interpolate(gr=gr, HDS=HDS, piez=piez, t0sim=t0sim)

        self.hds['diff'] = self.hds['computed'] - self.hds['measured']

        if t0dd is not None:
            # only if t0dd is explicityl specified wil the ddrawdown be computed here.
            self.drwdn(t0dd=t0dd, well=well, theis=theis, out=False)

        return


    def interpolate(self, gr=None, HDS=None, piez=None, t0sim=None):
        '''Interpolate series at t where t are pd.Timestamps.

        Parameters
        ----------
        gr: fdsm.mfgrid.Grid
            object, holding the modflow grid.
        HDS : flopy.utils.binaryfile object
            Modflow-computd heads
        piez: piezoms.Piezom object
            piezometer instance.
        t0sim : pd.Timestamp or str representing a pdTimestamp
            Absolute start time of Modflow simulation,
            this fixes HDS.times to absolute datetime.

        Returns
        -------
            pd.DataFrame with computed and measured heads at
            the simulation times times at each piezometer.
        '''

        aday = pd.Timedelta(1, 'D')

        piezom_times = (piez.hds.index - t0sim) / aday # in days
        mflow_times  = HDS.times

        h_mflow  = HDS.get_alldata();

        mask = h_mflow < -900.;

        h_mflow[mask] = np.nan

        M = self.meta

        LRC = gr.ixyz(M['x'], M['y'], 0.5 * (M['z1'] + M['z2']),
                                                  order='LRC', world=True)

        L, R, C = LRC[0]

        # verify and change exchange if necessary (i.e. for PBU094A)
        if M['z1'] < M['z2']:
            M['z1'], M['z2'] = M['z2'], M['z1']

        if M['z2']<M['topBreda']: L=1  # not dominant
        if M['z1']>M['topBreda']: L=0  # this is dominant, overwrites previous

        # also set whether this piezometer is dry or not
        M['dry'] = M['hdate'] < M['z2']

        self.meta['iz'] = L # store this, it's the model layer

        # Get interpolated heads for all times at piezometer location:
        ht_mflow_at_piezom = [ht_mflow[L, R, C] for ht_mflow in h_mflow]

        timestamps = [t0sim + t_mflow * aday for t_mflow in mflow_times]

        # Get measured and computed heads in a DataFrame
        ht_dframe =  pd.DataFrame({'computed' :ht_mflow_at_piezom}, index=timestamps)

        for col in piez.hds.columns:
            ht_dframe[col] = np.interp(mflow_times, piezom_times, piez.hds[col],
                     left=np.nan, right=np.nan)

        #return ht_dframe.dropna(axis=0, how='any')
        return ht_dframe



#%%
class Calibs(Base_Piezoms):
    '''
    Collection of Piezom-like objects. Instead of the measured heads, it
    will containt the computed heads at the measurement locations at the
    computed times and the meaured heads at the computed times.

    The Piezoms object is used to store the measured heads as such, without
    interference with any model and, therefore, without any interpolation.

    This class inherits all methods from its base_class. It only provides
    its __init__ constructor.

    @TO 2018-06-20
    '''

    def __init__(self, piezoms=None, gr=None, HDS=None, t0sim=None,
                 t0dd=None, well=None, theis=None):

        '''Return a Calibs collection.

        Calibs is like piezoms. For each piezom it holds a
        meta dictionary, a hds DataFrame and possibly a dds DataFrame both with
        columns ['computed', 'measured', 'diff']

        The times in hds are pd.Timestamps that correspond exactly
        with the times of the MODFLOW computation.

        The times in dds are are also pd.Timestamps corresponding
        exactly with the times of the modflow computation. Therefore
        t0dd is a requirement to comute the drawdown, as drawdonw always
        has to be relative to some point in time.

        Either the drawdown or the heads DataFrame can contain NaN, namely
        when the length of the hds time series differs from that of the dds
        time series.

        The drawdow, that is self[name].dds will only be computed
        next to self[name].hds when t0dd is given.
        Theis will only be computed if theis==True and well is specified.

        parameters
        ----------
            piezoms:
                the piezometers
            gr: fdm.mfgrid.Grid object
                gr object holding the Modflow mesh
            HDS: flopy.utils.binaryfile.Headfile
                computed heads
            t0sim: pd.Timestamp or a str representing one
                start time of simulation
            t0dd: pd.Timestamp or a str representing one
                time relative to which the drawdown is computed
            well: dict {x: x, y: y, Q: Q}
                well info required when computing Theis/Jacob analysis
            theis : bool
                true of Theis is analysis is to be done


        for name in piezomes:

            self[name] - Calib(piezoms[name], gr. HDS, t0sim, t0dd)
        '''

        self.data = dict()

        if piezoms is None:
            return # allows generating an empty Calibs

        #import pdb; pdb.set_trace()

        missed = []

        for name in piezoms:
            try:
                self.data[name] = Calib(piez=piezoms[name],
                         gr=gr, HDS=HDS, t0sim=t0sim,
                         t0dd=t0dd, well=well, theis=theis)
            except Exception as err:
                pz_logger.debug("piezom[{}]: {}".format(name, err.args[0]))
                missed.append(name)

        if missed:
            pz_logger.info('Missed piezometers: [{}]'.format(', '.join(missed)))


#%%

class Piezom(Base_Piezom):
    '''Defintion of the pieozmeter class

    A piezometer has meta containing meta data

    It further has two pd.DataFrames
        hds: the measured heads
        dds: the measured drawdown (relative to a t0dd time.
    It also carries a Calib object useful for calibration
    because it compares computed and measured heads and drawdowns.
    The Calib object has a piezometer as parent and shares its meta data.
    It then has two pd.DataFrames
        hds : The heads
        dds : The drawdowns
    The hds and dds DataFrames in Calib has columns ['computed', 'measured', 'diff']
    However, the times are fixed to those of the simulation. The measured heads
    are here time-interpolated to the simulaton times.
    On the other hand, the computed heads are space-iterpolated to the
    measurement locations.

    A Theis analysis is possible for the dds of the piezometer and also for
    those of the Calib.dds DataFrame. object. However it is only implemented
    for the drawdowns of the piezometer, because it mainly serves for
    a priori analysis of drawdown data.

    This Theis dd analysis find the maximum drawdown rate as dd increase per
    logcycle and finds the t0dd, the time at zero drawdown according to the
    simplified Theis formula: s = Q/(4 pi kD) ln(2.25 kD t/ (r^2 S)).

    From these the kD and S can be and are computed in this analysis. However
    this requires the coordinate of the extraction to compute the distance
    to each piezometer and it requires the extraction Q as well. Of course,
    this analysis is only valid of the drawdown becomes straight on the
    graph of drawdown versus time.

    @TO 2018-06-18
    '''

    def __init__(self, path2csv, filtNm=None, csvNm=None, meta=None,
                 tstart=None, tend=None, verbose=None, csvparams=None,
                 outlier_col=None, outlier_fence=3.0, outlier_window=11, **kwargs):
        '''Return Piezometers read from csv file.

        parameters
        ----------
        path2scsv : str
            name of piezometer
        filtNm: str
            name of piezometer (hole name + filter, e.g. PBU060 + A = PBU060A)
        csvNm : str
            name of csv file without path, e.g. PBU060.csv
        meta : dict holding meta data containing keys
                        'x', 'y', 'z1','z2'
        threshold: float
            used in self.cleanup() to remove outliers in the data
            that show up as spikes. The value 0.025 proofs practical.
        tstart : np.datetime64[ns] or str indicating datetime
            truncate time before
        tend : one of same type as tstart.
            truncate time after.
        verbose: bool
            if true show graph.
        outlier_cols : str or series of column names
            columns to check for ourliers
            Outlier reporting is only done of outlier_cols is not None
        outlier_fences : tuple of 2 floats default (1.5, 3.0)
            inner and outer fences to use for finding ourliers.

        kwargs: additional kwargs
        '''

        self.name  = filtNm
        self.meta = meta

        self.path = os.path.join(path2csv, csvNm)

        data = pd.read_csv(self.path, **csvparams)

        #self.data = self.cleanup(verbose=verbose, threshold=threshold)
        # this removes all lines
        data = data.dropna(how='all')


        tstart = data.index[ 0] if tstart is None else pd.Timestamp(tstart)
        tend   = data.index[-1] if tend   is None else pd.Timestamp(tend  )


        self.hds = data.loc[AND(data.index >= tstart, data.index <= tend), :].copy()

        if outlier_col is not None:
            self.remove_outliers(col=outlier_col,
                                 fence=outlier_fence,
                                 window=outlier_window)


    def remove_outliers(self, col=None, fence=3.0, window=11):
        '''Return self.hds with outliers removed.

        parameters
        ----------
        col: str
            column to check for outliers
        fence: postivie float
            threshold -->
            D < med - (Q3 - Q2) * fence OR D > med + (Q3 - Q2) * fence
        window: postive int
            width in centered rolling window to compute median

        returns
        -------
            The self.hds datafram is replaced.
        '''
        D  = self.hds[col]
        Q3 = D.quantile(0.75)
        Q1 = D.quantile(0.25)
        dQ = Q3 - Q1

        E = np.abs((D - D.rolling(window, center=True).median()) / dQ) < fence
        E = D[E]
        E.dropna()
        self.hds = self.hds.loc[E.index]



    def report_outliers(self, col='measured', fence=3.0, window=11):
        '''Report outliers in a pd.DataFrame column Col.

        parameters
        ----------
        df : pd.DagaFrame
            pandas DataFrame to be inspected.
        name : str
            identifier used in reporting. (Default is 'name??')
        col : str
            column name in df to check for outliers.
        fencee : a float
            fence limit such that
              < median - fence  * (Q3 - Q1), > median + fence * (Q3 - Q1))
                window: postive int
        window: positive int
            width in centered rolling window to compute median

        returns
        -------
            None

        >>> df = pd.DataFrame(index=np.arange(10), data=np.random.randn(10, 3), columns=['a', 'b', 'c'])
        >>> df.loc[[2, 5], 'b'] = [2.4, -10]
        >>> col = 'b'
        >>> get_outliers(df[col])  # ds[col] is a pd.Series not a pd.DataFrame !


        @ TO 2018-06-22 12:30
        '''

        if col is None:
            D = self.hds
        else:
            E = self.hds[col]

        Q3 = D.quantile(0.75)[col]
        Q1 = D.quantile(0.25)[col]
        dQ = Q3 - Q1

        E = np.abs((D - D.rolling(13, center=True).median()) / dQ) > fence
        E = D[E]
        E.dropna()

        if np.any(E):
            fig, ax = plt.subplots()
            ax.plot(D.index, D, label='measured', color='b', ls='-')
            ax.plot(E.index, E.loc[E.index], label='extremes', color='r', ls='', marker='.')


class Piezoms(Base_Piezoms):
    '''Piezoms is a collection of Piezom objects, representing piezometers with
    measured head and their meta data.

    This class inherits almost all its methods from its base class. Only its
    constructor __init__ is supplied here.

    @TO 2018-06-20
    '''

    def __init__(self, path2csv, csv2filt=None, ext='.csv', collars=None,
                 csvparams=None, tstart=None, tend=None, verbose=True,
                 outlier_col=None, outlier_fence=None, outlier_window=None, **kwargs):
        '''Return Piezometers collection (based on UserDict)

        parameters
        ----------
            path2csv : str
                path to folder with the piezometer data
            csv2filt: dict
                dictionary with csvBaseName, screenName
                e.g.: {'KBP21': 'KBP21A', 'PBU099': 'PBU099A', etc}
                see get_csv2filt() in jukaTools.py
            ext : str
                csv file extension, default = '.csv'
            collars : dict with fields see below.
                Collars cn be obtained by 'get_collars() in jukaTools.py.
            csvparams : dict
                the parameters required by pandas to read the csv file with the
                head time series. E.g.
                  csvparams = {'header': 0,
                 'index_col': 0,
                 'usecols': usecols,        # a list of ints
                 'names' : ['UTC', 'mNAP'], # names for used cols
                 'parse_dates':True,
                 'squeeze': False,
                 'skipinitialspace': True,
                 'dtype': {k : np.float64 for k in usecols}, # always floats
                 'na_values': {k: ['NaN'] for k in usecols}} # include ''NaN'
                Where usecols is a list of column numbers to use.

            tstart : np.datetime64[ns] obj or str indicating date time
                Truncate at front of time series. E.g. '2018-05-14 15:20'
            tend:    np.datetime64[ns] obj or str indicating date time
                Truncate at end of tiem series. E.g. '2018-06-18 20:00'
            outlier_col : str
                column to check for outliers (default 'measured')
            outlier_fence : float
                fence to use for eliminating outliers (default 3.0)
            outlier_window: int
                width of rolling window to check for outliers (default 11)

        '''
        self.data = dict()

        if not path2csv:
            return  # allows intializing an empty Piezoms

        missing = []

        #import pdb
        #pdb.set_trace()

        # Generate Piezom objects from reading CSV files
        for csvNm in csv2filt: #csv2Filt see get_csv2filt()

            filtNm = csv2filt[csvNm] # filter/screen name --> Hole + filterID

            try:
                if isinstance(collars, dict):
                    meta= collars[filtNm]
                else: # assuming collars is a pd.DataFrame
                    meta = dict(collars.loc[filtNm])
                meta['iz'] = np.nan # will be set in Calib not in Piezom becaus needs model grid

                self.data[filtNm] = Piezom(path2csv, csvNm=csvNm + ext,
                         filtNm=filtNm, meta=meta, threshold=0.025,
                         tstart=tstart, tend=tend, csvparams=csvparams,
                         verbose=verbose,
                         outlier_col='measured',
                         **kwargs)
            except:
                missing.append(filtNm)
        if missing:
            pz_logger.debug('''The following CSV files could not be read because
                         their corresponding collar is missing''')
            pz_logger.debug('[{}]'.format(', '.join(missing)))
        return


#%%

if __name__ == '__main__':

    import shelve
    import flopy.utils.binaryfile as bf

    size_inches=(11.5, 7.0)

    pz_logger = logging.getLogger(__name__)

    testpath = './testdata/'
    path2csvs = './testdata/csvfiles'
    shelvefile = os.path.join(testpath, 'testdata')

    collarfile = 'BoreholeData.xlsx'

    HDS = bf.HeadFile(os.path.join(testpath, 'testdata.hds'))

    t0sim  = '2018-05-13 00:00' # time of start of modflow simulation
    t0dd   = '2018-05-15 12:00' # time relative to which ddn is computed
    tstart = '2018-05-13 00:00' # start of graph
    tend   = '2018-06-01 00:00' # end of graph
    tstart = None # start of graph
    tend   = None # end of graph

    well = {'x': 187008, 'y': 346037, 'Q': 28600}
    theis = True


    with shelve.open(shelvefile) as s:
        gr = s['gr']


    excel_collars = {'sheet_name': 'Collars',
                   'header' :0,
                   'usecols': 'B:F',
                   'index_col': 'Hole'}

    excel_screens = {'sheet_name': 'filters',
                   'header':0,
                   'usecols': 'B:E',
                   'index_col': 'name'}

    collars = pd.read_excel(os.path.join(testpath, collarfile), **excel_collars)
    screens = pd.read_excel(os.path.join(testpath, collarfile), **excel_screens)

    collars.columns = ['x', 'y', 'elev', 'end_depth']

    collars = pd.merge(collars, screens, how='right', on='Hole')
    collars.index = screens.index

    collars['z1']   = collars['elev'] - screens['From']
    collars['z2']   = collars['elev'] - screens['To']
    collars['zEnd'] = collars['elev'] - collars['end_depth']

    collars.drop(columns=['From', 'To', 'end_depth'])


    usecols = [0, 1]

    csvparams = {'header': 0,
                 'index_col': 0,
                 'usecols': usecols,
                 'names' : ['time', 'measured'],
                 'parse_dates':True,
                 'squeeze': False,
                 'skipinitialspace': True,
                 'dtype': {k : np.float64 for k in usecols},
                 'na_values': {k: ['NaN'] for k in usecols}}

    csv2filt = {'PBGRA1': 'PBGRA1A',
                'PBGRA2': 'PBGRA2A',
                'PBGRA3': 'PBGRA3A',
                'PBGRA4': 'PBGRA4A'}

    piezoms = Piezoms(path2csvs, csv2filt=csv2filt,
                      collars=collars, csvparams=csvparams,
                      tstart=tstart, tend=tend, verbose=True,
                      outlier_col='measured')

    PBGRA3A = piezoms['PBGRA3A']

    PBGRA3A.plot(cols='measured')

    piezoms.plot(cols='measured')

    piezoms.apply(funs=[np.var, np.min, np.max],
                  tstart='2018-05-14 20:00', tend='2018-05-18 14:00')


    calibs = Calibs(piezoms, gr=gr, HDS=HDS, t0sim=t0sim)

    calibs.plot()

    calibs.plot(dd=True, t0dd=t0dd)

    #

    calibs.plot(dd=True, t0dd=t0dd, cols=['computed', 'measured'],
                theis=theis, well=well, xscale='log')


    # Check whether piezom A is the deepest.
    pzs= []
    names = [k for k in piezoms if k[-1] == 'A']
    for name in names:
        pz = [piezoms[name]]
        for L  in ['B', 'C']:
            try:
                pz.append(piezoms[name[:-1] + L])
            except:
                pass
        pzs.append(pz)
    for pz in pzs:
        if len(pz) > 1:
            for p in pz:
                print('{} {:.2f} {:.2f}'
                      .format(p.name, p.meta['z1'], p.meta['z2']), end=', ')
            print()


