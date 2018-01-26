
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import k0 as K0
from math import factorial

def coth(z):
    z[z>20] = 20.
    return (np.exp(z) + np.exp(-z)) / (np.exp(z) - np.exp(-z))


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
               Sat=None, Saq=None, c=None, T=None, v=None):
    '''Return drawdown for all layers for this r and t.
    '''

    s = np.zeros(T.shape)

    for iStehfest, vv in enumerate(v):
        p = (iStehfest + 1) * np.log(2) / t
        d = p * Saq / T  # number of aquifers
        b = np.sqrt(p * Sat * c)  # number of aquitards
        if len(T) == 1:
            eii = (b[0] * coth(b[0])) / (c[0] * T)
            eij = (b[1] * coth(b[1])) / (c[1] * T)
            A = eii + eij + d
        else:
            bcothb = b * coth(b)
            bsinhb = b / np.sinh(b)

            if len(c) > len(T): # aquitard with zero drawdown at bottom
                eii=  bcothb[ :-1] / (c[ :-1] * T)   #links to overlying  aquitard
                eij=  bcothb[  1:] / (c[  1:] * T)   # links to underlying aquitard
                fii=  bsinhb[1:-1] / (c[1:-1] * T[1:])
                fij=  bsinhb[1:-1] / (c[1:-1] * T[:-1])
            else: # no bottom aquitard
                eii=  bcothb       / (c * T)
                eij=  bcothb[1:-1] / (c[1:] * T[:-1])
                fii=  bsinhb[  1:] / (c[1:] * T[1:])
                fij=  bsinhb[  1:] / (c[1:] * T[:-1])

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

def selfTestHM87Fig3(N=8):
    ''' Solves problem stated in figure 2 of Hemker and Maas (1987)

    N = 8 seems optimal.

    TO 180125
    '''

    c   = np.array([1000, 1500, 1000, 4000, 20000])   # resistance of aquitards
    Sat = np.array([   3,  0.5,  0.3,  0.2,     1]) * 1e-3 # Stor coef of aquitards
    T   = np.array([2000, 1500,  500, 2000])   # T of aquifers
    Saq = np.array([   1,  0.4,  0.1,  0.3]) * 1e-3   # Stor coef of aquifers
    Q   = np.array([   0,10000,    0,    0])          #Extracion from aquifers

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
        S = 0
        for k in range(int((i+1)/2), int(min(i,N/2)) + 1):
            S += k ** int(N/2) * factorial(2 * k) / \
                (factorial(int(N / 2) - k) * factorial(k) * \
                 factorial(k - 1) * factorial(i - k) *\
                 factorial( 2 * k - i))
        v[i - 1] = (-1) ** (i + int(N/2)) * S
    return v

if __name__=="__main__":

    #v = stehfest()

    selfTestHM87Fig2(N=8)

    #selfTestHM87Fig3(N=8)

