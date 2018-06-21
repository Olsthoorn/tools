#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 11:16:31 2018

Piezometer tools

Analysis of the piezometers.

@author: Theo Olsthoorn 20180601
"""

import os
import glob
import collections
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shape
import logging
import etc

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)-15s -  %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
#logger.disable(logging.INFO)

# timedelta to float
# example td2float(piezom['PBGRA3'].dd.index)
td2float = lambda td : np.asarray(td, dtype=float)/1e9/86400
t2str = lambda t : t.__str__()

# back from timedeltas to seconds
td2s = lambda td: np.timedelta64(int(td * 86400), 's')
Td2s = lambda td: np.asarray(np.asarray(td * 86400, dtype=int), dtype='timedelta64[s]')

# Convert times to utc.
cest2utc  = lambda index: index - np.timedelta64(2, 'h') # not used
cet2utc   = lambda index: index - np.timedelta64(1, 'h') # not used
elli2CEST = lambda index: index - np.timedelta64(2, 'h') # used

AND = np.logical_end

class Calib:
    '''Return a Calib_hd object
    
    This object is internal to Piezom, so that it shares its meta object.
    
    The Calib_hd object is to compare computed with measured heads. It does so
    at the times tha Modflow used during the simlation. The measured
    column in the dataframe is computed by interpolating in the measured
    heads dataframe. This way both are available at the same times.
    
    The columns used are 'measured', 'computed', 'diff' which means
    computed minus measured heads.
    
    The index is the absolute time as timestamps ('np.datetime64[ns']) obtained
    from the fdm.mfgrid.stressPeriod object of Modflow HDS objectdirectly.

    '''    

    def __init__(self, gr=None, HDS=None, piez=None, t0hds=None, t0dd=None):
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
        
        self.parent = piez
        self.gr = gr
        self.meta = self.parent.meta # a view, so complete shared data.

        if t0dd is None: t0dd=tstart
        if isinstance(t0hds, str): self.t0hds = np.datetime64(t0hds)
        if isinstance(t0dd,  str): self.t0dd  = np.datetime64(t0dd)
            
        self.hds = piez.interpolate(self, gr, HDS, tstart=t0hds)

        self.hds['diff'] = self.data['computed'] - self.data['measured']
        
        self.dds = self.drawdowns(tstart=t0dd)
        
        return
    
    
    def interpolate(self, gr, HDS, tstart):
        '''Interpolate series at t where t are pd.Timestamps.
        
        Parameters
        ----------
        gr
        HDS : flopy HDS file object
            Modflow-computd heads
        tstart : np.datetime64 or str representing a datatiem64
            start of simulation, fix HDS.times to absolute time
        '''

        if isinstance(tstart, str):
            tstart = np.datetime64(tstart)
            
        if tstart < self.data.index[0]:
            raise ValueError('tstart = {} must be > start of data = {}'
                             .format(t2str(tstart), t2str(self.data.index[0])))

        if tstart > self.data.index[-1]:
            raise ValueError('tstart = {} must be < end of data = {}'
                             .format(t2str(tstart), t2str(self.data.index[-1])))
        
        hds = HDS.get_all_data
        D = self.meta
        ix, iy, iz = gr.ixyz(D.x, D.y, np.mean(D.z0, D.z1), order='LRC', world=True)

        # Get interpolated heads:
        h = [hd[iz, iy, ix] for hd in hds]
        
        times = tstart + Td2s(HDS.times)
        # Get measured and computed heads in a DataFrame
        hds =  pd.DataFrame(index=times, values=h, columns='computed')
        hds['measured'] = np.interp(times, self.peilb.index, self.peilb['mNAP'])
        return hds


    def drawdowns(self, t0):
        '''Interpolate series at t where t are pd.Timestamps.
        
        Parameters
        ----------

        t0 : np.datetime64 or str representing a datatiem64
            start of stradown, absolute time
        '''
        
        if isinstance(t0, str):
            return np.datetime64(str)
        
        self.t0 = t0
        
        # set drawdown
        before = self.hds.index < t0

        # Get measured and computed drawdowns in a DataFrame

        self.dd = self.hds.loc[~before, :].copy() # data frame copy

        D = self.dd        
        D.loc[:, 'measured'] = D.loc[:, 'measured'] - D.loc[0, 'measured']
        D.loc[:, 'computed'] = D.loc[:, 'computed'] - D.loc[0, 'computed']
        D['diff'] = D['computed'] - D['measured']
        D.index = D.index - D.index[0]

        return
        

    def msre(self, dd=False):
        if dd==False:
            return (self.hds['diff'] ** 2).mean().sqrt()
        else:
            return (self.dds['diff'] ** 2).mean().sqrt()

    def abs_err(self, dd=False):
        if dd==False:
            return self.data['diff'].abs().mean()
        else:
            return self.dds['diff'].abs().mean()
    
    def mean_err(self, dd=False):
        if dd==False:
            return self.hds['diff'].mean()
        else:
            return self.dds['diff'].mean()


    def plot(self, what=['computed','measured'],dd=False, **kwargs):
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
            legend = True
            fig, ax = plt.subplots()
           
            if dd==False:
                prep = 'Heads'
            else:
                prep = 'Drawdowns'
            title='{} piezom. {} at x={}, y={} layer={}'\
                        .format(prep, self.parent.name, self.x, self.y)
            ax.set(title=title,
                   xlabel=kwargs.pop('xlabel','time'),
                   ylabel=kwargs.pop('ylabel',prep + '[m]'),
                   xscale = kwargs.pop('xscale', 'linear'),
                   yscale = kwargs.pop('yscale', 'linear'))
            if 'xlim' in kwargs: ax.set_xlim = kwargs.pop('xlim')
            if 'ylim' in kwargs: ax.set_ylim = kwargs.pop('ylim')

        else:
            legend=False
            
        ax.grid(True)
        for w in what:
            if dd==False:
                ax.plot(self.hds.index(), self.hds[w], label='{} {}'.format(self.parent.name, w), **kwargs)
            else:
                ax.plot(self.dds.index(), self.dds[w], label='{} {}'.format(self.parent.name, w), **kwargs)

        if legend:
            ax.legend(loc='best')


class Piezom:
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
    '''
    
    def __init__(self, path2csv, key, meta=None, threshold=0.025,
                 tstart=None, tend=None, verbose=None, csvparams=None, **kwargs):
        '''Return Piezometers read from csv files..
        
        parameters
        ----------
        path2scsv : str
            name of piezometer
        key: str
            name of piezometer
        meta : dict holding meta data containing keys
                        'x', 'y', 'z0','z1'
        threshold: float
            used in self.cleanup() to remove outliers in the data
            that show up as spikes. The value 0.025 proofs practical.
        tstart : np.datetime64[ns] or str indicating datetime
            truncate time before
        tend : one of same type as tstart.
            truncate time after.
        verbose: bool
            if true show graph.
        kwargs: additional kwargs
        '''
        
        self.name  = key
        self.meta = meta
        
        self.path = os.path.join(path2csv, key + '.csv')
        
        data = pd.read_csv(self.path, **csvparams)

        #self.data = self.cleanup(verbose=verbose, threshold=threshold)
        data = data.dropna()
        
        tstart = data.index[ 0] if tstart is None else np.datetime64(tstart)
        tend   = data.index[-1] if tstart is None else np.datetime64(tend  )
        
        self.hds = data.loc[AND(data.index >= tstart, data.index <= tend), :].copy()
        

        
    def drwdn(self, t0dd=None):
        
        D = self.hds
        
        t0dd = D.index[0] if t0dd is None else np.max(D.index[0], np.datetime64(t0dd))

        dd = D.loc[D.index >= t0dd].copy()

        dd.index = dd.index - dd.index[0]
        dd['m']   = dd['mNAP'] - dd['mNAP'].iloc[0]

        dd['m'].drop(labels=['mNAP'], axis=1)

        return dd


 
    def drwdn_plot(self, plot_theis_tangent=False, **kwargs):            
        '''Plot the drawdown for this piezometer relative to tstart.
        
        parameters
        ----------
        tstart : np.datetime64        
            timestap of start of drawdown (assumed within data time span)
        plot_tangent : bool
            whether or not to plot the straight line approximation
            according to the simplified Theis analysis given by
            $$ s = \frac {Q} {4 \pi kD} ln \left( \frac {2.25 kD t} {r^2 S} \right)$$
        xscale : str
            'log' or 'linear'
        yscale : str
            'log' or 'linear'
            
        additional kwargs are passed on to plt.plot.
        '''


        if plot_theis_tangent:
            
            # Use same color and linestyle as the dd line
            color=ln[0].get_color()
            ls   =ln[0].get_ls()
    
            t = self.dd.index[self.dd.index > self.dd_theis['t_dd0']]
            y = self.dd_theis['maxGrad'] * (np.log10(t) - np.log10(self.dd_theis['t_dd0']))
    
            ax.plot(t, y, color=color, ls=ls)
            ax.plot(self.dd_theis['t_maxGrad'], self.dd_theis['dd_maxGrad'], 'o', color=color)
            ax.plot(self.dd_theis['t_dd0'], 0, 'o', color=color)

        
    def attime(self, t, col='mNAP'):
        '''Return head (interpolated) at t.
        
        parameters
        ----------
            t  : np.datetime64 object or a str representing one (a single datetime)
                time that head is resired.

                t will be set to self.hds.index[ 0] if < than that;
                t will be set to self.hds.index[-1] if > than that.
        '''
        D = self.hds
            
        t = D.index[ 0] if t is None else np.datetime64(t)
        t = D.index[-1] if t > D.index[-1] else t

        return t, np.interp(t, D.index, np.asarray(D[col]))

            
    def plot(self, tstart=None, tend=None, dd=False, **kwargs):
        '''Plot the measurements, not the drawdown h(t).
        
        
        parameters
        ----------
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
        '''
        AND = np.logical_end
        
        ax = kwargs.pop('ax', None)
        
        what = 'drawdown' if dd==True else 'head'

        D = self.Data

        tstart = D.index[ 0] if tstart is None else np.datetime64(tstart)
        tend   = D.index[-1] if tend   is None else np.datetime64(tend  )
        
        I = np.where(AND(D.index >= tstart, D.index <= tend))[0]
        
        if dd == False:
            what = 'head'
        else:
            D = self.drwdn(t0dd=tstart)
            what = 'drawdown'

        
        if ax is None:
            fig, ax = plt.subplots()
            fig.set_size_inches(kwargs.pop('size_inches', (11.5, 7.0)))
            ax.set_title('piezom {} {}'.format(self.name, what))
            ax.set_xlabel('utc')
            ax.set_ylabel('m NAP')
            if 'xlim' in kwargs: ax.set_xlim(kwargs.pop('xlim'))
            if 'ylim' in kwargs: ax.set_ylim(kwargs.pop('ylim'))
            ax.grid()
            
        label = self.name
        
        if not 'color' in kwargs:
            kwargs.update({'ls': '-', 'color': 'blue'})
        
        ax.plot(D.index[I], D.iloc[I], label=label, **kwargs)


    def distance(self, x, y):
        '''Return distance from well to point x, y
        parameters
        ----------
        x : float
            x coordinate
        y : float
            y coordinaten
        '''
        try:
            return np.sqrt(
                    (self.meta['x'] - x)**2 + (self.meta['y'] - y)**2)
        except:
            return np.nan

        
    def __len__(self, dd=False):
        return len(self.hds)
    

    def __str__(self, dd=False):
        try:
            return '''piez[{}]
        data = pd.DataFrame(len={})
        columns = {}
        meta = {}'''.format(
            self.name, len(self.hds),
            ', '.join(self.hds.columns),
            ', '.join([k + ':' + str(v) for k, v in 
             zip(self.hds.keys(), self.hds.values())]))
        except:
            return '''peib[{}]
        data = pd.DataFrame(len{})
        columns = {}
        meta = None'''.format(
            self.name, len(self.hds),
            ', '.join(self.hds.columns))


    def cleanup(self, verbose=False, threshold=0.025):
        '''Removes spikes (outliers) form self.data.
        
        parameters
        ----------
        verbose : bool
            plot graph of piezom of True
        threshold : postive float
            spike must be larger than threshold to be detected.
        '''
        
        # remove NaNs
        self.hds = self.hds.dropna()
        
        # remove outliers (peaks in the data)
        window = [3, 5, 7]
        self.hds['u'] = self.hds['mNAP']
        for wd in window:
            self.hds['u'] = self.hds['u'].rolling(
                            window=wd, center=True)\
                            .median()\
                            .fillna(method='bfill')\
                            .fillna(method='ffill')

        self.outlier_index = np.abs(self.hds['mNAP'] - self.hds['u']) > threshold

        self.hds['mNAP_orig'] = self.hds['mNAP']
        self.hds['mNAP']      = self.hds['u']

        return self.hds

        if verbose:
            fig, ax = plt.subplots()
            ax.set_title('piezuis {}'.format(self.name))
            ax.set_xlabel('utc')
            ax.set_ylabel('m NAP')
            ax.grid()
            #ax.plot(pb_data.index, pb_data['Waterstand gemeten'] - pb_data['u'], 'b', label=name)
            ax.plot(self.hds.index, self.hds['mNAP'], 'b', label=self.name)
            ax.plot(self.hds.index[self.outlier_index],
                    self.hds['mNAP'][self.outlier_index], '.r', label=self.name)
            ax.legend()


    def theis_analysis(self, x0=None, y0=None, Q=None, out=None, **kwargs):
        '''Return steepest gradient per logcycle and t where s=0.
        
        This is based on the simplified Theis solution that yield
        straight drawdown lines of half-logarithmic time scale.
        
            $s = \frac {Q} {4 \pi kD} ln(\frac {2.25 kD t} {r^2 S})$
        
        uses self.dd, the drawdown DataFrame
        
        parametrs
        ---------
        if self.dd has not yet been created using self.drwdn(tstart)
        then tstart must be in kwargs.
        
        returns
        -------
        dict containing the result of the analys allowing to plot
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
        kwargs may contain tstart to compute drawdown
        '''
        
        if not 'dd' in self.__dict__.keys():
            try:
                self.drwdn(tstart=kwargs.pop('tstart'))
            except:
                raise LookupError('{} not in kwargs'.format('tstart'))
        
        logt = np.log10(self.dd.index)
        
        # points to interpolate the drawdown at fix distance on log time scale
        step       = np.log10(2.)
        logt_pnts  = np.arange(logt[0], logt[-1], step)
        dd_pnts    = np.interp(logt_pnts, logt, self.dd['m'])
        gradient   = np.diff(dd_pnts) / np.diff(logt_pnts)
    
        # lcoation of maximum gradient
        i = np.argmax(gradient)
        
        # maximum gradient iself
        maxGrad = gradient[i]
        dd_maxGrad   = 0.5 * (  dd_pnts[i]   + dd_pnts[i + 1])
        logt_maxGrad = 0.5 * (logt_pnts[i] + logt_pnts[i + 1])
        
        logt_dd0    = logt_maxGrad - dd_maxGrad / maxGrad
        dd_logcycle = maxGrad # drawdown per log10 cycle

        self.dd_theis = {'dd_logcycle': dd_logcycle,
                't_dd0' : 10**(logt_dd0),
                't_maxGrad': 10**(logt_maxGrad),
                'dd_maxGrad': dd_maxGrad,
                'maxGrad': maxGrad,
                'dd_max' : np.max(self.dd['m'])}
        
        
        # kD and S
        R = self.distance(x0, y0)
        kD = Q / (4 * np.pi * self.dd_theis['dd_maxGrad'])
        S  = 2.25 * kD * self.dd_theis['t_dd0'] / R**2
        self.dd_theis.update({'r': R, 'kD': kD, 'S': S})
        
        logger.info('sef.dd._theis set for {}'.format(self.name))
        
        if out:
            return self.dd_theis
      

class Piezoms(collections.UserDict):
    
    def __init__(self, path2csv, pattern='*.csv', namefun=None, collars=None,
                 csvparams=None, tstart=None, tend=None, verbose=True, **kwargs):
        '''Return Piezometers collection (based on UserDict)
        
        parameters
        ----------
            path2csv : str
                path to folder with the piezometer data
            pattern : str
                file pattern used by glob.glob to select files in folder.
                defalt = '*.csv'
            namefun: fun
                function to convetbasename of file to name piezometer key.
            collars : pd.DataFrame with columns see below.
                Collars may be obtained in advance by reading the data fiom
                an Excel workbook, such that the resulting 
                pandas.DataFrame has at least columns ['x' 'y', 'z0', 'z1']
                with z0 and z1 the top and bottom elevation of the piezomeer screen,
                and the index of the DataFrame is the name of the piezometer.

                To read a proper sheet from an Excel workbook, you may use
                parameters like below (see documentation).
                
                pd.read_excel(path, **excelparams)
                where (see pandad) excel_params may look like:
                    excelparams =  {'sheet_name': 'collars',
                               'header': 0,
                               'usecols': ['A:C, K:M'],
                               'index' : 0}
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
        '''
        
        self.data = dict()
        
        # Generate Piezom objects from reading CSV files
        for csvName in glob.glob(os.path.join(path2csv, pattern)):
            
            name = os.path.splitext(os.path.basename(csvName))[0]
            
            try:
                meta = dict(collars.T[name])
            except:
                raise LookupError(
                    'name {} not in collars with our without namefun applied.'
                            .format(name))
            
            self[name] = Piezom(path2csv, name, meta=meta, threshold=0.025,
                 tstart=tstart, tend=tend, csvparams=csvparams, verbose=verbose, **kwargs)
            
    
    def plot(self, tstart=None, tend=None, **kwargs):
        '''Plot the measurements, not the drawdown h(t).
        
        
        parameters
        ----------
        tstart : np.datetime64 obj or a str that represents a datatime.
            starting time of plot
        tend: same as tstart
            ending time of plot
        kwargs
        ------
        kwargs : plot options
            all kwargs are pased on to plt.plot
        '''
        
        fig, ax = plt.subplots()
        fig.set_size_inches((8, 7))
        fig.set_size_inches(kwargs.pop('size_inches', (11.5, 7.0)))
        ax.set_title('Measured heads')
        ax.set_xlabel('utc')
        ax.set_ylabel('m NAP')
        if 'xlim' in kwargs: ax.set_xlim(kwargs.pop('xlim'))
        if 'ylim' in kwargs: ax.set_ylim(kwargs.pop('ylim'))
        ax.grid()
            
        # time span to plot
        for name, ls in zip(self, etc.linestyle_cycler()):
            self[name].plot(ax=ax, tstart=tstart, tend=tend, **ls)
        ax.legend(loc='best')

            
    def set_calib(self, gr, HDS, t0hds, t0dd):
        
        for piezom in self:
            piezom.set_calib(gr, HDS, t0hds, t0dd)


    def theis_analysis_table(self, fname=None, **kwargs):
        '''Show a table of the results of drawdown analysis and inject results into piez[name].dd.
        
        parameters
        ----------
        names : sequence of str
            the keys/names of the piez to be used.
        fname : str
            name of csv file to save results table.
        kwargs: dict
            additional arguwments (not used)        
        '''
        
        for name in self:
            self[name].theis_analysis(self, x0=None, y0=None, Q=None, out=None, **kwargs)
            
        
        columns = ['name', 'x', 'y', 'r',
                'h0drwdn', 'd_water0', 'dd_logcycle',
                't_dd0', 't_maxGrad', 'dd_maxGrad',
                'dd_max', 'kD', 'S', 't0drwdn',
                ]

        table = pd.DataFrame(columns=columns)            
                
        # print one line for each piezometer given its name
        for name in self:

            ta = self[name].dd_theis_analysis

            table.loc[self.name] = pd.Series(ta).T

        return table

    def dd2shape(self, itime=int, path2shape=None, fldNms=None):
        '''Generate shapefile with the dawdown data.
        
        The shapefile is a point shapefile it uses dd['xRd'] and dd['yRd'] and
        dd['topBreda'] for the coordinates of the piezoemter.
    
        parameters
        ----------
            itime: index in drawdown index to extract drawdown for given time.
            shapefileNm : str
                the name of ths shapefile to generate.
            fldnms : list of str
                fldnams form piez[names].dd to use
        '''
        
        
        
        fldNms = ['x', 'y', 'r', 'kD', 'S']
            
        self.dd_dict = {'name': 'dummy'}
        self.dd_dict.update({k: np.nan for k in fldNms})
        
        # Geneate a dict
        for name in self:
            D = self[name].dd_theis
            self.dd_dict[name] = {k: D[k] for k in D}
            self.dd_dict[name].update({'t':D.index[itime], 'dd': D['m'].iloc[itime] })
            
        shape.dict2shape(path2shape, self.dd_dict)
            
        logger.debug('Drawdiwb written to shapefile {} in\n{}'
                                 .format(*os.path.split(path2shape)))

   
#%%

if __name__ == '__main__':
    
    path = './testdata/BoreholeData.xlsx'
    path2csvs = './testdata/csvfiles'
    
    excel_collars = {'sheet_name': 'Collars',
                   'header' :0,
                   'usecols': 'A:E',
                   'index_col': 0}
    
    excel_screens = {'sheet_name': 'screen',
                   'header':0,
                   'usecols': 'A:C',
                   'index_col': 'Hole ID'}
    
    collars = pd.read_excel(path, **excel_collars)
    screens = pd.read_excel(path, **excel_screens)
    
    collars.columns = ['x', 'y', 'elev', 'end_depth']
    collars['z0']   = collars['elev'] - screens['From']
    collars['z1']   = collars['elev'] - screens['To']
    collars['zEnd'] = collars['elev'] - collars['end_depth']
    collars.drop(columns=['end_depth'])

    # convert the colloar names to those of the csv files, so that both match
    collars.index = [k.replace('-','') for k in collars.index]

    usecols = [0, 1]
    tstart = '2018-05-14 14:00'
    tend   = '2018-06-01 00:00'
    
    csvparams = {'header': 0,
                 'index_col': 0,
                 'usecols': usecols,
                 'names' : ['UTC', 'mNAP'],
                 'parse_dates':True,
                 'squeeze': False,
                 'skipinitialspace': True,
                 'dtype': {k : np.float64 for k in usecols},
                 'na_values': {k: ['NaN'] for k in usecols}}
        
    
    piezoms = Piezoms(path2csvs,
                      collars=collars, csvparams=csvparams,
                      tstart=tstart, tend=tend, verbose=True)
    
    PBGRA3 = piezoms['PBGRA3']
        
    PBGRA3.plot()
    
    piezoms.plot()
    
    piezoms.plot()
    #piezoms.set_calib(gr, HDS, t0dd='2018-05-17 14:00')

