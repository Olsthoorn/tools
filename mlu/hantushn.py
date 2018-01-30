
import sys

tools = 'Users/Theo/GRWMODELS/python/tools/'

if not tools in sys.path:
    sys.path.insert(1, tools)

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import k0 as K0
from math import factorial

from fdm.fdm3t import fdm3t
from fdm.mfgrid import Grid
from scipy.interpolate import interp1d
from colors import colors

def bcoth(z):
    z[z>20.  ] = 20.
    z[z<1.e-6] = 1e-6
    return z * (np.exp(z) + np.exp(-z)) / (np.exp(z) - np.exp(-z))


def hantushn(Q=None, r=None, t=None, Sat=None, Saq=None, c=None, T=None, N=10):
    '''
    Return analytical drawdown of n-layer transient solution multiple aquifer system.


    Laplace transform is applied with back-transformation from Laplace space
    according to Stehfest.

    Implies the solution given by Hemker and Maas (1987)

    Example:
        hantushn();  % selfTest
        s = hantushn(Q,r,t,Sat,S,c,T[,N]);

    The solution has been used for the interpretation of the pumping test
    De Zilk, Pastoorslaan by Jochem Fritz and Rob Rijsen. Sept 3-14 2007.

    parameters
    ----------
        Q : ndarray of floats
            extraction vector (positive extractions will create positive dd')
        r : nadarray of floats
            distance vector
        t : ndarray of floats


        Sat: ndarray of floats
            storage coefficients of aquitards
        S : ndarray of floats
            storage coefficient of aquifers
        c : ndarray of floats
            hydraulic resistance vector
            to model absence of a top or bottom aquifer, just make the
            resistance of the top and or bottom aquifer infinite. You can
            make any or all aquitard resistances infinite.
        T : ndarray of floats
            transmissivity vector
        N : int
            Stehfest's parameter, default 10


    The first resistance layer is on top of the first aquifer by definition.
    apply Inf to close off the given zero drawdown on top of this layer.
    if length(c)and Sat must be the same
    if length(c)=length(T)+1, this means that there is a resistance
    layer also at the bottom of the system and a fixed zero drawdown
    below it.
    If, however, length(T)==length(c), then the lowest layer is an
    aquifer closed at its bottom.

    returns
    -------
        drawdown (Nt, Naq, Nr) ndarray

        If instead r= list of tubples with (r, iLyer, t), the drawdown will
        be in list format
        providing the drawdown for every coordinate pair
        in that case t is not used as a separate input is dummy !

    See also
    --------
        stehfest (in this modulle)
    %
    % TO 090329 090330 180122 converstion to Python
    '''

    print('Running Hantushn')

    Q   = np.array(Q)
    Sat = np.array(Sat)
    Saq = np.array(Saq)
    T   = np.array(T)
    c   = np.array(c)
    #if len(c) < len(T):
    #   c = np.hstack((np.inf, c, np.inf)) # add top and bot aquifer
    #   Sat = np.hstack((1e-5, Sat, 1e-5)) # must not be zero
    #if len(c)==len(T):
    #   c = np.hstack((c, np.inf)) # add bot aquifer
    #   Sat = np.hstack((Sat, 1e-5))  # must not be zero

    assert len(Sat) == len(c), 'len(Sat) !=len(c)'
    assert len(Saq) == len(T), 'len(S) != len(T)'
    assert len(Q   )== len(T), 'len(Q) != len(T)'


    # compute Stehfest's coeffient
    v = stehfest(N);

    # Compute drawdown
    # T, r and t together form an array of values (nL,nR,nt)

    drawdown = np.zeros((len(t), len(T), len(r)))
    for ir, rr in enumerate(r):
        for it, tt in enumerate(t):
            drawdown[it, :, ir] = ddOnePoint(\
                    Q, rr, tt, Sat=Sat, Saq=Saq, c=c, T=T, v=v)

    return drawdown

eig = np.linalg.eig

def ddOnePoint(Q=None, r=None, t=None,
               Sat=None, Saq=None, c=None, T=None, top_aquitard=True, v=None):
    '''Return drawdown for all layers for this r and t.
    '''

    s = np.zeros(T.shape)

    if len(c) < len(T):
        top_aquitard = False

    bot_aquitard = ( len(c)  > len(T)) or \
                   ((len(c) == len(T)) and (top_aquitard == False))

    if not top_aquitard:  # prepend dummy to c and Sat
        c   = np.hstack((1.e6, c))
        Sat = np.hstack((0. , Sat))
    if not bot_aquitard:  # postpand dummy to c and Sat
        c = np.hstack((c, 1.e6))
        Sat = np.hstack((Sat, 0.))

    for iStehfest, vv in enumerate(v):
        p = (iStehfest + 1) * np.log(2) / t
        d = p * Saq / T  # number of aquifers

        b = np.sqrt(p * Sat * c)  # number of aquitards

        if len(T) == 1:
            # This should yield Hantush directly
            eii = bcoth(b[0]) / (c[0] * T)
            eij = bcoth(b[1]) / (c[1] * T)
            A = eii + eij + d
        else:
            bcothb = bcoth(b)
            bsinhb = b / np.sinh(b)

            eii=  bcothb[:-1] / (c[:-1] * T)
            eij=  bcothb[ 1:] / (c[ 1:] * T)
            fii=  bsinhb[ 1:-1] / (c[ 1:-1] * T[ 1:])
            fij=  bsinhb[ 1:-1] / (c[ 1:-1] * T[:-1])

            if not top_aquitard: eii[0]  = 0.
            if not bot_aquitard: eij[-1] = 0.

            A = np.diag(eii + eij + d) - np.diag(fii, -1) - np.diag(fij, +1)

        # This is suggested by Hemker in his papers, but seems to give
        # not perfectly the same results
        #         if False:
        #            D = np.diag(T) ** (1/2) * A * np.diag(T) ** (-1/2)
        #            R, W = np.eig(A)  # D
        #            V = np.diag(T) ** (-1/2) * R
        #         else:
        W, V = eig(A)  # D

        s += v[iStehfest] / (2 * np.pi * p) * \
            np.dot(np.dot(np.dot(V, np.diag(K0(r * np.sqrt(W)))), np.linalg.inv(V)),(Q / T))

    dd = s * np.log(2) / t

    return dd


def selfTestHM87Fig2(N=8):
    ''' Solves problem stated in figure 2 of Hemker and Maas (1987)

    N = 8 seems optimal.

    TO 180125
    '''

    c   = np.array([[1e12, 100, 1e12],
                    [100, 100, 100],
                    [100, 100, 100],
                    [100, 100, 100]])
    Sat = np.array([[1.6, 1.6, 1.6],
                    [0, 1.6,    0],
                    [1.6, 1.6, 1.6]])



    T   = np.array([100, 100])   # T of aquifers
    Saq = np.array([0.1, 0.1]) * 1e-3   # Stor coef of aquifers
    Q   = np.array([   0,4 * np.pi * 100])          #Extracion from aquifers

    r = np.array([0.1 * np.sqrt(T[0] * c[1])])
    t = np.logspace(-1, 5, 61)

    dd = hantushn(Q=Q, r=r, t=t, Sat=Sat, Saq=Saq, c=c, T=T, N=N)


    # =============fig 3 Hemker and Maas (1987) ===========
    #defaults = {'nextplot','add','ydir','reverse','xGrid','on','yGrid','on'};

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(12,6)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('t [d]')
    ax.set_ylabel('sd [m]')
    ax.set_title('H+M87 fig 2,  dd vs t')
    ax.set_xlim((1e-1, 1e5))
    ax.set_ylim((1e-2, 10))
    ax.grid(True)

    for rr in r:
        for iL in [0, 1]: # size(dd,1)
            lw = 2   if iL == 0 else 1
            ls = '-' if iL == 0 else '--'
            ax.plot(t, dd[:, iL, 0], label='L{}, r={:.1g}'.format(iL+1, rr), ls=ls, lw=lw)
    ax.legend(loc='best')

    return dd

def selfTestHM87Fig3(N=8):
    ''' Solves problem stated in figure 2 of Hemker and Maas (1987)

    N = 8 seems optimal.

    TO 180125
    '''

    c   = np.array([1000., 1500., 1000., 4000., 20000.])   # resistance of aquitards
    # also possible: c   = np.array([np.inf, np.inf, np.inf, np.inf, np.inf])   # resistance of aquitards
    Sat = np.array([   3,  0.5,  0.3,  0.2,     1]) * 1e-3 # Stor coef of aquitards
    T   = np.array([2000, 1500,  500, 2000])   # T of aquifers
    Saq = np.array([   1,  0.4,  0.1,  0.3]) * 1e-3   # Stor coef of aquifers
    Q   = np.array([   0., 10000., 0., 0.])          #Extracion from aquifers

    r = np.logspace(1, np.log10(1e4), 41)
    t = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10])

    dd = hantushn(Q=Q, r=r, t=t, Sat=Sat, Saq=Saq, c=c, T=T, N=N)


    # =============fig 3 Hemker and Maas (1987) ===========
    #defaults = {'nextplot','add','ydir','reverse','xGrid','on','yGrid','on'};

    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(12,6)

    ax[0].set(xscale='linear', yscale='linear', xlim=(0,  6000), ylim=(0, 1))
    ax[1].set(xscale='log',    yscale='log',    xlim=(10,10000), ylim=(0.001, 10))

    ax[0].set_xlabel('r [m]')
    ax[0].set_ylabel('dd [m]')
    ax[1].set_xlabel('r [m]')
    ax[1].set_ylabel('dd [m]')
    ax[0].set_title('H+M87 fig 3a, dd vs r')
    ax[1].set_title('H+M87 fig 3b, dd vs r')
    ax[0].set_ylim((1, 0))
    ax[1].set_ylim(10, 1e-3)
    ax[0].grid(True)
    ax[1].grid(True)

    for it, t in enumerate(t):
        for iL in [0, 1]: # size(dd,1)
            lw = 2   if iL == 0 else 1
            ls = '-' if iL == 0 else '--'
            ax[0].plot(r, dd[it, iL, :], label='L{}, t={:.1g}'.format(iL+1, t), ls=ls, lw=lw)
            ax[1].plot(r, dd[it, iL, :], label='L{}, t={:.1g}'.format(iL+1, t), ls=ls, lw=lw)
    ax[0].legend(loc='best')
    ax[1].legend(loc='best')

    return dd


def selfTestHM87Fig4(N=8):
    ''' Solves problem stated in figure 2 of Hemker and Maas (1987)

    N = 8 seems optimal.

    TO 180125
    '''

    c   = np.array([[1e12, 100, 1e12],
                    [100, 100, 100],
                    [100, 100, 100],
                    [100, 100, 100]])
    Sat = np.array([1.6, 1.6, 1.6],
                    [0, 1.6,    0],
                    [1.6, 1.6, 1.6],
                    []



                   ) * 1e-3 # Stor coef of aquitards
    T   = np.array([100, 100])   # T of aquifers
    Saq = np.array([0.1, 0.1]) * 1e-3   # Stor coef of aquifers
    Q   = np.array([   0,4 * np.pi * 100])          #Extracion from aquifers

    r = np.array([0.1 * np.sqrt(T[0] * c[1])])
    t = np.logspace(-1, 5, 61)

    dd = hantushn(Q=Q, r=r, t=t, Sat=Sat, Saq=Saq, c=c, T=T, N=N)


    # =============fig 3 Hemker and Maas (1987) ===========
    #defaults = {'nextplot','add','ydir','reverse','xGrid','on','yGrid','on'};

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(12,6)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('t [d]')
    ax.set_ylabel('sd [m]')
    ax.set_title('H+M87 fig 2,  dd vs t')
    ax.set_xlim((1e-1, 1e5))
    ax.set_ylim((1e-2, 10))
    ax.grid(True)

    for rr in r:
        for iL in [0, 1]: # size(dd,1)
            lw = 2   if iL == 0 else 1
            ls = '-' if iL == 0 else '--'
            ax.plot(t, dd[:, iL, 0], label='L{}, r={:.1g}'.format(iL+1, rr), ls=ls, lw=lw)
    ax.legend(loc='best')

    return dd

def stehfest(N=10):
    '''Return Stehfest's v-coefficients.

    parameters
    ----------
        N : int
            Numberof Stefest coefficients

    This needs to be calculated only once
    '''

    v = np.zeros((N, 1))

    for i in range(1, N+1):
        S = np.zeros(N)
        for j, k in enumerate(range(int((i+1)/2), int(min(i,N/2)) + 1)):
            S[j] = k ** int(N/2) * factorial(2 * k) / \
                (factorial(int(N / 2) - k) * factorial(k) * \
                 factorial(k - 1) * factorial(i - k) *\
                 factorial( 2 * k - i))
        S = np.sum(sorted(S))
        v[i - 1] = (-1) ** (i + int(N/2)) * S
    return v


def compare_hant_fdm(aqSys, obsWells, t=None, Q=None, epsilon=1.0, **kwargs):
    '''Compare hantushn vor r's and all t with fdm3t
    obsWells : dictionarary with keyys ['name', 'r', 'layer')

    Input prerequisites
    aqSys : Aquifer_system
    epsilon : float
        implicitness, use value between 0.5 to 1 (Modflow uses 1.0)
    '''

    N = kwargs.pop('N', 10)

    # simulate analytically using Hantushn
    dd = hantushn(Q=Q, r=obsWells.r, t=t,
                  Sat=aqSys.Sat,
                  Saq=aqSys.Saq, c=aqSys.c, T=aqSys.kD, N=N)

    dd = dd[:, obsWells.aquifer, np.arange(dd.shape[-1], dtype=int)]


    # simulate numerically using fdm3t
    FQ = aqSys.gr.const(0.);     FQ[1::2, 0, 0] = Q

    HI = aqSys.gr.const(0.)

    IBOUND = aqSys.gr.const(1, dtype=int)
    IBOUND[[0, -1], :, :] = -1

    Kh = aqSys.gr.const(aqSys.kh)
    Kv = aqSys.gr.const(aqSys.kv)
    Ss = aqSys.gr.const(aqSys.Ss)

    out = fdm3t(gr=aqSys.gr, t=t, kxyz=(Kh, Kh, Kv),
                 Ss=Ss, FQ=FQ, HI=HI, IBOUND=IBOUND,
                 epsilon=epsilon)

    phi = out['Phi'][:, :, 0, :] # squeeze y (axis 2)

    # interpolator
    interpolator = interp1d(aqSys.gr.xm, phi, axis=2)

    phi = interpolator(obsWells.r) # shape [Nt, Nz, len(obsWells)]

    # select using fancy indexing for last two axes
    phi = phi[:, obsWells.layer, np.arange(phi.shape[-1])]


    # plot the results of both Hantush and fdm3t
    ax = kwargs.pop('ax', None)
    if ax is None:
        fig, ax = plt.subplots()
        ax.set_title (kwargs.pop(
            'title', 'Hantushn versus fdm3 for a set of observatio points'))
        ax.set_xlabel(kwargs.pop('xlabel', 't [d]'))
        ax.set_ylabel(kwargs.pop('ylabel', 's [m]'))
        ax.set_xscale(kwargs.pop('xscale', 'log'))
        ax.set_yscale(kwargs.pop('yscale', 'linear'))
        ax.grid(True)

    xlim = kwargs.pop('xlim', None)
    ylim = kwargs.pop('ylim', None)

    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)

    for i, name in enumerate(obsWells.names):
        ax.plot(t, phi[:, i], label='fdm ' + name, color=colors[i], ls='-', lw=2)
        ax.plot(t, dd[ :, i], label='han ' + name, color=colors[i], ls='--')
    ax.legend(loc='best')

    return ax


class Aquifer_system:
    def __init__(self, z=None, c=None, kD=None, Sat=None, Saq=None,
                 rw=0.1, R=1e4,
                 top_aquitard=True):

        r = np.logspace(np.log10(rw), np.log10(R),
                int(10 * np.ceil(np.log10(R/rw))  + 1) )

        self.gr  = Grid(r, [-0.5, 0.5], np.array(z), axial=True)


        self.top_aquitard = top_aquitard
        self.bot_aquitard =\
            (     self.top_aquitard  and len(c) > len(kD)  ) or\
            ((not self.top_aquitard) and len(c) == len(kD) )

        assert self.gr.nz == len(c) + len(kD), 'nz != len(c) + len(kD)'
        assert self.gr.nz == len(Sat) + len(Saq), 'nz = len(Sat)  + len(Saq)'

        self.c   = np.array(c)
        self.kD  = np.array(kD)
        self.Sat = np.array(Sat)
        self.Saq = np.array(Saq)

        self.kh = np.zeros(self.gr.nz)
        self.kv = np.zeros(self.gr.nz)
        if self.top_aquitard:
            self.kv[::2]  = self.gr.dz[::2] / c
            self.kh[::2]  = 0.
            self.kh[1::2] = self.kD  / self.gr.dz[1::2]
            self.kv[1::2] = np.inf
        else:
            self.kh[::2]  = self.kD / self.gr.dz[::2]
            self.kv[::2]  = np.inf
            self.kv[1::2] = self.gr.dz[1::2] / c
            self.kh[1::2] = 0.
        if self.top_aquitard: self.kv[ 0] *= 0.5
        if self.bot_aquitard: self.kv[-1] *= 0.5

        self.iaquif = np.ones(self.gr.nz, dtype=bool)
        if self.top_aquitard:
            self.iaquif[::2] = False
        else:
            self.iaquif[1::2] = False
        self.iatard = np.logical_not(self.iaquif)

        self.Ss = np.zeros(self.gr.nz)
        self.Ss[self.iaquif] = self.Saq / self.gr.dz[self.iaquif]
        self.Ss[self.iatard] = self.Sat / self.gr.dz[self.iatard]


class ObsWells:
    def __init__(self, points, aqSys):
        '''generate an instance of observation points.

        parameters
        ----------
            points: (name, r, z)
                observation point locations
        '''

        if isinstance(points, dict):
            self.names = [p['name'] for p in points]
            self.r     = [p['r']    for p in points]
            self.z     = [p['z']    for p in points]
        else:
            self.names = [p[0] for p in points]
            self.r     = [p[1] for p in points]
            self.z     = [p[2] for p in points]
        self.layer   = aqSys.gr.lrc(self.r, np.zeros(len(self.r)), self.z)[:, 0]

        aqNr = -np.ones(aqSys.gr.nz, dtype=int)
        j = 0
        for i, ia in enumerate(aqSys.iaquif):
            if ia:
                aqNr[i] = j
                j += 1

        self.aquifer = aqNr[self.layer]
        for i, ia in enumerate(self.aquifer):
            assert ia >= 0,\
            'obsPoint {} not in aquifer'.format(self.names[i])



if __name__=="__main__":

    #v = stehfest()

    #selfTestHM87Fig2(N=8)

    selfTestHM87Fig3(N=8)

    z   = [0, -10, -20, -30, -40, -50]
    c   = [1000., 600, 900]
    kD  = [450., 1250]
    Sat = [1.e-5, 1.e-5, 1.e-5]
    Saq = [1.e-3, 1.e-3]
    top_aquitard = True

    aqSys = Aquifer_system(z=z, c=c, kD=kD,
                           Sat=Sat, Saq=Saq, top_aquitard=True)

    points = [('wp1', 10, -15),
              ('wp2', 20, -35),
              ('wp3', 25, -15),
              ('wp4', 50, -35)]

    Q = [1000, 600.]

    t = np.logspace(-3, 2, 51)

    obsWells = ObsWells(points, aqSys)

    compare_hant_fdm(aqSys, obsWells, xscale='log', t=t, Q=Q, N=8)




