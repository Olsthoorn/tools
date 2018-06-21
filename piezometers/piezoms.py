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
import glob
import collections
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shape
import logging
import etc

size_inches=(11.5, 7.0)

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


AND = np.logical_and


#%%
def theis_analysis(obj, well=None, col='measured'):
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
    
    D = obj.dds  # the drawdown pd.DataFrame
    
    logt = np.log10(D.index / np.timedelta64(1, 'D')) # in days
    
    # step size at log time scale
    step       = np.log10(2.)

    # discrete time points at equal distance on log time scale    
    logt_pnts  = np.arange(logt[0], logt[-1], step)
    dd_pnts    = np.interp(logt_pnts, logt, D[col])

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
        'dd_logcycle': dd_logcycle,
        't_dd0' : 10**(lt_dd0),
        't_maxGrad': 10**(lt_maxGrad),
        'dd_maxGrad': dd_maxGrad,
        'maxGrad': maxGrad,
        'dd_max' : np.max(D[col])}
    
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
        
        logger.info('kD and S added to self.dd_theis set for piezometer {}'.format(obj.name))
    
    return theis


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
    


    def drwdn(self, t0dd=None, well=None, theis=False, out=True):
        '''Return drawdown with respect to timestamp t0dd.
    
        In all cases, the resutting drawdown DataFrame is stored as self.dds
        
        t0dd is stored in self.meta['t0dd']
        
        If theis == True, then the rsults of the theis analysis, i.e. dd per
        logcycle and the time at which the log dd is 0 are stored as
        self.meta['theis']
        
        If well is not None, then kD and S are computed and stored along with theis.
    
        
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
        '''
        
        if t0dd is None:
            raise ValueError('t0dd must be specified when calling self.drwdn()')
        else:
            t0dd = pd.Timestamp(t0dd)
            
        att = self.attime(t=t0dd, dd=False, cols=None)
        
        # Also truncates the ddn
        self.dds = self.hds.loc[self.hds.index > t0dd].copy()
        
        for col in self.dds.columns:
            self.dds[col] = self.dds[col] - att[col]
            
        self.dds.index = self.dds.index - t0dd
    
        self.meta['t0dd'] = t0dd
            
        if theis:
            self.meta['theis'] = theis_analysis(self, well=well, col=col)
            
            logger.info('self.dd_theis set for piezometer {}'.format(self.name))

            if out:
                return self.dds, self.meta['theis']
        elif out:
            return self.dds,
    

    def attime(self, t=None, dd=False, cols=None):
        '''Return head (interpolated) at t.
        
        parameters
        ----------        
            t  : np.datetime64 object or a str representing one (a single datetime)
                time that head is resired.
            dd: bool
                if True compute drawdown else compute head
            cols: str or list of str
                name of column(s) in hds or dds DataFrame to use, e.g.:
                    cols = ['meausured', 'computed', 'diff'] or
                    cols = 'computed'                
        '''
        if isinstance(cols, str): cols = [cols]
        
        att = {'name': self.name, 't': t, 'dd': dd}
        att.update({k: self.meta[k] for k in self.meta})
        
        if dd == False:
            t        = pd.Timestamp(t)
            td_f     = (t - self.hds.index[0]) / np.timedelta64(1, 's')
            td_ind_f = (self.hds.index - t)    / np.timedelta64(1, 's')
            for col in self.hds.columns:
                att[col] = np.interp(td_f, td_ind_f, self.hds[col])
        else:
            '''
            The drawdown index is always in timedelta relative to t0dd
            so the t must be given as float or timedelta in case
            dd==True.
            '''            
            if isinstance(t, float):
                td = t * np.timedelta64( np.timedelta64( 1, 'D'), 's') # seconds for accuracy
            elif isinstance(t, np.timedelta64): # ok
                td = np.timedelta64(t, 's')
                pass
            else:
                raise ValueError(
                        '''
                        t must be instance of (float, or np.timedelta64) or
                            to denote relative time after t0dd
                        t must be instance of (str, np.timdelta64 or pd.Timestamp)
                            to denote absolute time of drawdown.
                        ''')

            td_f     = td / np.timedelta64(1, 's')  # float because t = timedela64

            try:            
                td_idx_f = np.asarray(self.dds.index, dtype='timedelta[s]') / np.timedelta64(1, 's')

                for col in cols:
                        att[col] = np.interp(td_f, td_idx_f, self.dds[col])
            except:
                raise ValueError('self.dds not found. Remedy: Initialize spycifying t0dd explictly')

        return att


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
        '''
        ax = kwargs.pop('ax', None)
        
        # Make sure self.dds exists.
        if dd == True:
            self.drwdn(t0dd=t0dd, well=well, theis=theis, out=False)
        
        
        if cols is None:
            cols = list(self.hds.columns if dd==False else self.dds.columns)
        
        if not isinstance(cols, (tuple, list)): cols=[cols]
    
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
            
            
        for i, col in enumerate(cols):
            
            # overwrite ls if more than one column
            if i > 0: kwargs['ls'] = etc.linestyles_[i % len(etc.linestyles_)]
            
            if dd==False:
                ax.plot(self.hds.index, self.hds[col], label='{} {}'.format(self.name, col), **kwargs)
            else:
                if 'dds' not in self.__dict__:
                    raise ValueError(
                            ''''self.dds not found. Remedy:
                                Initialize specifying t0dd and possible well and theis explicitly''')
                ax.plot(self.dds.index / np.timedelta64(1, 'D'), self.dds[col], label='{} {}'.format(self.name, col), **kwargs)
    
            if theis and col=='measured':
                Th = self.meta['theis']
                lgt = np.linspace(np.log10(Th['t_dd0']),
                                  np.log10(Th['t_dd0']) + 1., 21)            
                dd = np.linspace(0, Th['dd_logcycle'], 21)
                
                kwargs.update(
                    {'ls': etc.linestyles_[(i + 1) % len(etc.linestyles_)], 'lw': 2})
                         
                
                if len(lgt) > 0:
                    ax.plot(10 ** lgt, dd, label=self.name + ' Th/Jac approx.', **kwargs)
            
        if legend:
            ax.legend(loc='best')
        return



#%%
class Base_Piezoms(collections.UserDict):
    '''Base class for piezometer collection. It shoulde not be instantiate directly
    
    __init__ should be defined in the sub_class

    @TO 20180620    
    '''



    def drwdn(self, t0dd=None, well=None, theis=False):
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
        '''
        for name in self:
            self[name].dds, theis_ = \
                        self[name].drwdn(t0dd=t0dd, well=well, theis=theis, out=False)
            
        logger.info("Drawdown computed are saved in self[name].dds for {} piezometers."
                    .format(len(self)))
        logger.info("t0dd <{}> saved in self[name].meta['t0dd']"
                    .format(self[name].meta['t0dd']))
        if theis:
            logger.info("Theis analysis resutls are saved in self[name].meta['theis']")
        return


    def attime(self, t, dd=False, cols=None, shapefile=None):
        '''Return head (interpolated) at t.
        
        parameters
        ----------        
            t  : np.datetime64 object or a str representing one (a single datetime)
                time that head is resired.
            dd: bool
                if True compute drawdown else compute head
            cols: str or list of str
                name of column(s) in hds or dds DataFrame to use, e.g.:
                    cols = ['meausured', 'computed', 'diff'] or
                    cols = 'computed'
            shapefile : str or None
                if not None, shapefile is the path to the shapefile to be generated.
                if None, then don't generate a shapefile.
        returns
        -------
        att : dict
            with key self[name].name and fiels
            ['x', 'y', 'z', 'iz', 't', usedcols, 'dd']
        '''
        
        att = dict()
        for name in self:
            att[name] = self[name].attime(self[name], t, dd=dd, cols=cols)
            
        if shapefile:
            if not os.path.isfile(shapefile):
                raise FileNotFoundError("Can't find file {}".format(shapefile))
                
            shape.dict2shp(att, shapefile, shapetype='POINT', xy=('x', 'y'), usecols=cols)
            logger.info('Shapefile <{}> generated, containing {} points'.format(shapefile, len(att)))
    
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
        '''
           
        if not cols is None and not isinstance(cols, (tuple, list)):  cols = [cols]
        
        fig, ax = plt.subplots()
        fig.set_size_inches(kwargs.pop('size_inches', size_inches))
        
        if cols is None:
            ax.set_title('Calibs, {}'.format('column names unspecified'))
        else:
            ax.set_title('Calibs, {}'.format(', '.join(cols)))
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
    
        for name, ls in zip(self, etc.linestyle_cycler()):
            self[name].plot(cols=cols, dd=dd,
                    t0dd=t0dd, tstart=tstart, tend=tend,
                    well=well, theis=theis, **ls, ax=ax, **kwargs)
            
        ax.legend(loc='best')


        
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
        '''

        self.attime(t, dd=dd, cols=None, shapefile=shapefile)       


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
            self.dds = self.drwdn(t0dd=t0dd, well=well, theis=theis, out=False)

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
        '''

        
        hmflow  = HDS.get_alldata()
        mask = hmflow < -900.
        hmflow[mask] = np.nan

        M    = self.meta
        
        LRC = gr.ixyz(M['x'], M['y'], 0.5 * (M['z0'] + M['z1']), order='LRC', world=True)

        L, R, C = LRC[0]
        
        self.meta['iz'] = L # store this, it's the model layer

        # Get interpolated heads for all times at piezometer location:
        htmf_at_piezom = [htmf[L, R, C] for htmf in hmflow]
        
        aday = np.timedelta64(np.timedelta64(1, 'D'), 's')

        timedeltas = np.array([t * aday for t in HDS.times])
        timestamps = np.array([t0sim + td for td in timedeltas])
        
        # Get measured and computed heads in a DataFrame
        htdframe =  pd.DataFrame({'computed' :htmf_at_piezom}, index=timestamps)
        
        mflo_idx_as_floats = np.array(timedeltas, dtype=float)
        piez_idx_as_floats = np.array(np.asarray(piez.hds.index -t0sim,
                                        dtype=timedeltas.dtype), dtype=float)

        for col in piez.hds.columns:
            htdframe[col] = np.interp(mflo_idx_as_floats, piez_idx_as_floats, piez.hds[col])
        return htdframe


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

    def __init__(self, piezoms=None, gr=None, HDS=None, t0sim=None):

        '''Return a Calibs collection.
        
        Calibs is a collocation like piezoms. For each piezom it holds a
        meta dictionary, a hds DataFrame and a dds DataFrame both with
        columns ['computed', 'measured', 'diff']
        
        The times in hds are np.datetime64 timestamps that correspond exactly
        with the times of the MODFLOW computation.
        
        The times in dds are np.timedelta[D] objects that also correspond
        exactly with the times of the modflow computation, however, they
        are with respect of t0dd, the starting time of the drawdown as
        given by the user. This is 
        for hds, the index is the same as for the piezoms
        
        for name in piezomes:
            
            self[name] - Calib(piezoms[name], gr. HDS, t0sim, t0dd)
        '''
        self.data = dict()
        
        for name in piezoms:
            self.data[name] = Calib(piez=piezoms[name], gr=gr, HDS=HDS, t0sim=t0sim)
    
  
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
    
    def __init__(self, path2csv, key, meta=None, tstart=None, tend=None,
                 verbose=None, csvparams=None, **kwargs):
        '''Return Piezometers read from csv file.
        
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
        tend   = data.index[-1] if tend   is None else np.datetime64(tend  )
        
        self.hds = data.loc[AND(data.index >= tstart, data.index <= tend), :].copy()
    
    
class Piezoms(Base_Piezoms):
    '''Piezoms is a collection of Piezom objects, representing piezometers with
    measured head and their meta data.
    
    This class inherits almost all its methods from its base class. Only its
    constructor __init__ is supplied here.
    
    @TO 2018-06-20
    '''
    
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
                meta['iz'] = np.nan # will be set in Calib not in Piezom becaus needs model grid
            except:
                raise LookupError(
                    'name {} not in collars with our without namefun applied.'
                            .format(name))
            
            self.data[name] = Piezom(path2csv, name, meta=meta, threshold=0.025,
                 tstart=tstart, tend=tend, csvparams=csvparams, verbose=verbose, **kwargs)
        return


   
#%%

if __name__ == '__main__':
    
    import shelve
    import flopy.utils.binaryfile as bf

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
                   'usecols': 'A:E',
                   'index_col': 0}
    
    excel_screens = {'sheet_name': 'screen',
                   'header':0,
                   'usecols': 'A:C',
                   'index_col': 'Hole ID'}
    
    collars = pd.read_excel(os.path.join(testpath, collarfile), **excel_collars)
    screens = pd.read_excel(os.path.join(testpath, collarfile), **excel_screens)
    
    collars.columns = ['x', 'y', 'elev', 'end_depth']
    collars['z0']   = collars['elev'] - screens['From']
    collars['z1']   = collars['elev'] - screens['To']
    collars['zEnd'] = collars['elev'] - collars['end_depth']
    collars.drop(columns=['end_depth'])

    # convert the colloar names to those of the csv files, so that both match
    collars.index = [k.replace('-','') for k in collars.index]

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
        
    
    piezoms = Piezoms(path2csvs,
                      collars=collars, csvparams=csvparams,
                      tstart=tstart, tend=tend, verbose=True)
    
    PBGRA3 = piezoms['PBGRA3']
        
    PBGRA3.plot(cols='measured')
    
    piezoms.plot(cols='measured')
    
    calibs = Calibs(piezoms, gr=gr, HDS=HDS, t0sim=t0sim)
    
    calibs.plot()
    
    calibs.plot(dd=True, t0dd=t0dd)
    
    calibs.plot(dd=True, t0dd=t0dd, cols=['computed', 'measured'], theis=theis, well=well, xscale='log')


