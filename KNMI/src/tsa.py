'''
Created on 29 Mar 2017

@author: Theo
'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
'''
Module ts_analysis.py, holds functionality for Groundwater time-series analysis

Created on Sun Feb  5 23:27:10 2017

@author: Theo
"""
import sys
import os

myModules = os.path.join(os.path.expanduser('~'), 'GRWMODELS', 'Python_projects', 'modules')

if not myModules in sys.path:
    sys.path.insert(1, myModules)

from mfetc import prar

import numpy as np
import scipy.stats as stats
import scipy.signal as signal
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.optimize import leastsq
import pickle
import pdb

intervals = lambda index: np.diff(index)/np.timedelta64(1, 'D')
unpack    =  lambda parDict, isLog:\
     np.array([np.log(parDict[k]) if k in isLog else parDict[k] for k in parDict])
repack    = lambda x, par, isLog: OrderedDict(tuple(
        [(k, np.exp(v)) if k in isLog else (k, v) for k, v in zip(par, x)]))

def ppars(parD):
    """prints parameters in OrderedDict parD"""
    for k in parD:
        print(" {} = {:8.3g}, ".format(k, parD[k]), end="")
    print()



def generate(NE, d=0., c=1.0, S=0.2, mode=0., a=30., sigma=None):
    """Returns a newly generated time series with noise
    parameters:
        NE    : pandas dataframe with headers 'RH', 'EV24' and timestamp index
                it is generated from a KNMI meteo file.
        d     : float, drainge base [elevation m]
        c      : [d] drainge resistance
        k      : [-] exponent - 1
        Sc     : [T] characteristic drainage decay time
        sigma : float, innovation std reached after 1 day
    returns:
        pandas dataframe, containing NE plus columns h, y, n and v
        where:
            h : generated heads (including noise)
            y : model, (no noise)
            n : residual (h - y)
            v : innovations
    """

    # simulate without noise
    key = 'y'
    simulate(NE, d=d, c=c, S=S, mode=mode, key=key, verbose=False) # genearates column NE['y']

    dt     = intervals(NE.index)
    v      = sigma * np.random.randn(len(NE)) # innovations
    v[1:] *= np.sqrt( (a/2) * (1 - np.exp(-2 * dt / a)))

    n      = v.copy()
    for i,_dt_ in enumerate(dt):
        n[i+1] += n[i] * np.exp(-_dt_ / a)

    # generated head series with measurements
    NE['n'] = n
    NE['v'] = v
    NE['h'] = NE[key] + NE['n']
    return

def gamma_block_response(mode=0, loc=0, scale=1.0, dtau=1.0, verbose=False):
    """Resturns the numerical block response for timestep dtau
       computed from gamma distribution as gamma.cdf(tau+dtay) - gamma.cdf(tau)
    parameters:
    -----------
    mode   : float [T], mode of gamma distribution in user dimension [T]
    scale  : float [T], theta of gamma distribution in user dimension [T}
    dtau   : float [T], constant time step size in simulation [T]
    loc    : float [T], shift to the right of gamma ditribution [T]
    verbose: boolean, if verbose, plot the block response
    returns:
    --------
    Block response BR [-] for time step dtau [T]
    """

    theta = scale # for clarity translate to theory which uses theta instead of scale
    n = 1 + mode/theta # convert mode to exponent n in gamma ditribution (=shape factor)
    b = loc  # default is zero. Not used, but could be. Changes gamma to pearsonIII

    # Geneerate a frozen gamma distribution, to derive all other info from:
    gamma = stats.gamma(n, scale=theta, loc=b) # frozen distribution

    # tau long enough to capture distribution
    tau0    = 0.
    tau_end = gamma.ppf(0.999)  # dimension [T] already throug theta in gamma
    tau     = np.linspace(tau0, tau_end, int(np.ceil((tau_end - tau0)/dtau)))

    # Compute a 1 dtau block response to filter input series
    filt  = gamma.cdf(tau + dtau) - gamma.cdf(tau)

    if verbose:
        fig, ax = plt.subplots()
        ax.set_title("Block response, n={}, theta={}".format(n, theta))
        ax.set_xlabel('tau [d]')
        ax.set_ylabel('Normalized BR [-] for dtau={:.2g} day'.format(dtau))
        ax.plot(tau, filt)

    return filt

def simulate(NE, d=0., c=1., S=0.2, mode=0., key='y', verbose=False):
    """Returns the simulated time series of heads
    parameters:
    -----------
    NE     : pandas dataFrame with keys ['rch'] and ['dt']
    d      : [m] drainage elevation
    c      : [d] drainge resistance
    S      : [-] specific yield (storage coefficient)
    mode   : [d] mode of impulse response [d]
    key    : key to store simulated series in dataFrame as NE[key]
    returns:
    --------
    NE[key] has simulated values (Returned by reference)
    None
    """

    dtau = intervals(NE.index[:2])[0]

    theta = c * S

    filt  = gamma_block_response(mode=mode, scale=theta, dtau=dtau, verbose=False)

    #y = scipy.signal.lfilter(filt, 1., RCH['rch'].values) + d
    NE[key] = c * signal.lfilter(filt, 1., NE['rch']) + d

    if verbose:
        fig, ax = plt.subplots()
        ax.set_title("Simulated time series without noise")
        ax.set_xlabel("Time [d]")
        ax.set_ylabel("head [m]")
        NE[['y']].plot(ax=ax)
        plt.show()

    return


def sim_with_noise(NE, ts, d=0., c=1., S=0.2, mode=0., a=30., key='y'):
    """Returns tuple (pd.df, sigma) where ts augmented with simulation results

    OU stands for Ornstein-Uhlenbeck noise model. This is the noise-model used
    for instance in the time-series analysis package Menyanthes
    by Von Asmuth (2012)

    ts is a pd.timeseries of head that will be augemented with simulation
    results.
    sigma is the std dev of the noise process after delta_t= 1d

    parameters:
    -----------
        NE   : pandas dataframe with columns ['rch'] and ['dt']

               it is generated from a KNMI meteo file.
               This is the explaining time series containing two columns, 'RH'
               and 'EV24', the precipitation and the Makking evaportranspiration
               both in m/d.
               The NE time series contains daily values from a KNMI weather station.
        ts   : a pandas time series, containing the head measurements for a given
               piezometer. It has a timestep index.
        d    : float [m], drainage base [elevation m]
        c    : float [d] drainage resistance
        mode : float [d], mode of impulse response
        theta: float [d], characteristic drainage decay time = cS
        a    : float [T], charateristic noise decay time
        yroff: runin time in years (before first measurement)
    returns:
    --------
        The results appear as an extra column in the dataFrame NE[key] where
        key is a string passed into this function.
        Notice that the column with key will be added or updated in both
        the NE and the ts dataframes!!!
        upon output, ts will be a pandas DataFrame with columns h, key, n and v
        where:
            h   : measured heads
            key : model, (no noise)
            n   : residual (h - y)
            v   : innovations (possibly weighted innovations)
    """

    # simulate, results in NE being updated with column key
    simulate(NE, d=d, c=c, S=S, mode=mode, key=key)  # adds field 'y'

    # sample simulated y at ts index of meausured ts
    ts[key]  = NE[key].loc[ts.index].copy()

    # residuals are measurd h minus simulated y
    n  = np.array(ts['h'] - ts[key])
    v  = n.copy()            # Initialize innovations v
    dt = intervals(ts.index)

    # Extract noise from measurements and innovations accordingto Ornstein Uhlenbeck
    for i, _dt_ in enumerate(dt):
        v[i+1] = n[i+1] - n[i] * np.exp(-_dt_ / a)

    # Ornstein Uhlenbeck wieights (reciprocals !)
    w      = np.ones_like(v)
    w[1:]  = np.sqrt(1 - np.exp(-2 * dt / a))

    # update ts dataframe
    ts['n']  = n   # residuals
    ts['v']  = v   # innovations
    ts['w']  = w   # Ornstein Uhlenbeck weights
    ts['vw'] = v/w # weighted innovations

    return # return entire simulated data frame by reference

def jac(x0, NE, ts, simulator=None, parD=None, isLog=None, key='y', verbose=False):
    """Returns Jacobian

    parameters
    ----------
    x0   : ndarray of floats. The parameter vector.
           x0 should not be touched inside this function as it's used by leassq
           via reference. Touching interferes unpredictably with the optimation.
    NE   : pandas dataFrame containing columns 'RH' and 'EV24' with respectively
           the precipitaiton and the Makkink evapotranspiration for[24h] intervals
    ts   : pandas dataFrame with the head time series in column 'h' or
           pandas time series with name 'h. It must have a timestamp index.
    simulator: function that simulates the time series.
    parD : dict of {parName: parValue, ...}
    isLog: list of parD keys that are log-transformed (ln-transformed)
    returns
    -------
    array of weighted innovations
    TO 20170319
    """

    # generate a parameter dictionary from x0 for use by the simulator
    # converting log-stransformed parameters to normal ones if necessary
    simulator(NE, ts, **parD, key=key)
    J0 = np.array(ts[key])

    Jac = np.zeros((len(ts), len(parD)))
    d = 0.01
    for i, k in enumerate(parD.keys()):
        x0[i] +=  d
        parD = repack(x0, parD.keys(), isLog)
        simulator(NE, ts, **parD, key=key)
        Jac[:,i] = (np.array(ts[key]) - J0) / d
        x0[i] -= d

    return Jac


def func(x0, NE, ts, simulator=None, parD=None, isLog=None, key='y', verbose=False):
    """Returns weighted innovations given the parameter vector x0
    and the simulator function

    This function is passed to optimize.leastsq(...) to optimize parmameters.
    It returns the weighted innovations for the time series.
    ts will be updated with columns of simulated values (key), residuals (n)
    innovations (v) and weighted innovations(vw).
    The dataframe NE should be complete, i.e. it must contain the input data
    for every time step. On the contrary, the measurement series ts may
    have been sampled at irregular intervals.

    parameters
    ----------
    x0   : ndarray of floats. The parameter vector.
           x0 should not be touched inside this function as it's used by leassq
           via reference. Touching interferes unpredictably with the optimation.
    NE   : pandas dataFrame containing columns 'RH' and 'EV24' with respectively
           the precipitaiton and the Makkink evapotranspiration for[24h] intervals
    ts   : pandas dataFrame with the head time series in column 'h' or
           pandas time series with name 'h. It must have a timestamp index.
    simulator: function that simulates the time series.
    parD : dict of {parName: parValue, ...}
    isLog: list of parD keys that are log-transformed (ln-transformed)
    returns
    -------
    array of weighted innovations
    TO 20170319
    """

    # generate a parameter dictionary from x0 for use by the simulator
    # converting log-stransformed parameters to normal ones if necessary
    parD = repack(x0, parD.keys(), isLog)

    simulator(NE, ts, **parD, key=key)

    if verbose:
        ppars(parD)
        #print('std of ts :', ts['vw'].std())
        #print(ts['v'])

    return np.array(ts['vw']) #.values


def plot(ts,key='y'):    # Plot the reslting series
    """
    Show the input and measurement series with residuals and innovations
    """
    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].set_title('measured and modelled head')
    ax[1].set_title('residuals')
    ax[2].set_title('innovations')

    ax[0].set_ylabel('elevation [m]')
    ax[1].set_ylabel('[m]')
    ax[2].set_ylabel('[m]')
    ax[2].set_xlabel('time [d]')

    ts[key].plot(style='b-',  ax=ax[0])
    ts['h'].plot(style='r.',  ax=ax[0])
    ts['n'].plot(ax=ax[1])
    ts['v'].plot(ax=ax[2])


def __main__():
    """test the time series analysis functions defined in this module

    # generator generates a time series based on a recharge dataframe and noise
    # simulator simulates a time series with noise
    # simulate simulates a time serie without noise
    # gamma_block_response : generates a block response based on gamma function
    """
    #pdb.set_trace()

    # Choose model for simulations
    simulator = sim_with_noise

    # refer to data directory on this computer
    datapath = os.path.join(os.path.expanduser("~"), 'GRWMODELS', 'Python_projects', 'ts_analysis')
    datafile = os.path.join(datapath, 'NE.pkl')
    if not os.path.isfile(datafile):
        raise FileExistsError("Can't find file <{}>".format(datafile))

    # Get recharge data:
    with open(datafile, 'rb') as fd:
        NE = pickle.load(fd)
        try:
            NE['rch'] = (NE['RH'] - NE['EV24']) / 1000. # To m/d
        except:
            raise KeyError("dataFrame in <{}> has not column 'RH' or 'EV24'".format(datafile))

    # Set true parameters for test
    d_True      = 1.25 # [m], eelvation, draiange basis
    c_True      = 300. # [d] drainage resistance
    S_True      = 0.2 # [-] specific yield
    mode_True   = 20. # [T] characteristic head decay time
    a_True      = 30. # deday of noise

    isLog = ('c', 'S', 'mode', 'a')

    p_True = OrderedDict((('d', d_True),
                          ('c', c_True),
                          ('S', S_True),
                          ('mode', mode_True),
                          ('a', a_True)))

    p_init = OrderedDict((('d', d_True),
                          ('c', c_True),
                          ('S', S_True),
                          ('mode', mode_True),
                          ('a', a_True  * 0.003)))

    sigma = 0.025  # innovation noise stddev

    # Generate and show a new time series
    generate(NE, **p_True, sigma=sigma)


    # Sample to get a simulted measurement head series at 14d measurement interval
    ts = NE['h'].loc['1/14/2010':'12/13/2016':14].to_frame()


    # Initial parameters to optimize by leastsq
    x0 = unpack(p_init, isLog)

    # Optimize parameters using leastsq
    xOpt, cov_x, infoDict, msg, ier = leastsq(func, x0,
                   args=(NE, ts, simulator, p_init, isLog, 'opt', True),
                   full_output=True, epsfcn=1.0e-6)

    # Convert final parametres result[0] back to parameter dictionary
    p_opt = repack(xOpt, p_True.keys(), isLog)

    # Compare true, initial and final parameters
    print("                {:8s} {:8s} {:8s}".format("True", "start", "final" ))
    for k in p_True:
        print(" {:<10s}".format(k), end="")
        print(" {:8.2f}".format(p_True[k]), end="")
        print(" {:8.2f}".format(p_init[k]), end="")
        print(" {:8.2f}".format(p_opt[k]))

    #Jac0 =jac(unpack(p_True, isLog), NE, ts, simulator=simulator,
    #         parD=p_opt, isLog=isLog, key='y', verbose=False)
    #prar(Jac0, "Jac")
    #Jac1 =jac(unpack(p_init, isLog), NE, ts, simulator=simulator,
    #         parD=p_opt, isLog=isLog, key='y', verbose=False)
    #prar(Jac1, "Jac")
    #Jac2 =jac(unpack(p_opt, isLog), NE, ts, simulator=simulator,
    #         parD=p_opt, isLog=isLog, key='y', verbose=False)
    #prar(Jac2, "Jac")

    if not cov_x is None:
        cov = np.sum(infoDict['fvec'] ** 2) / (infoDict['nfev'] - len(p_opt)) * cov_x
        prar(cov, 'cov')

        psigma= np.sqrt(np.diag(cov))
        print("Parameter uncertainty :")
        ppars(repack(psigma, p_opt, isLog))


        print(infoDict['fvec'])
        cor = cov / (psigma[:, np.newaxis] * psigma[np.newaxis, :])
        prar(cor, 'cor')

    # Simulate the model with the final parameters
    key='opt'
    simulator(NE, ts, **p_opt, key=key)
    tse = NE[['h', key]].loc[ts.index]
    n = np.array(tse['h'] - tse[key])    # residuals
    v = n.copy()
    a = p_opt['a']
    dt = intervals(ts.index)
    for i, _dt_ in enumerate(dt):
        v[i+1] = n[i+1] - n[i] * np.exp(-_dt_ / a)
    tse['n'] = n
    tse['v'] = v

    plot(tse, key=key)
    plt.show()

    print("Finished")
    return


if __name__=='__main__':
    __main__()
