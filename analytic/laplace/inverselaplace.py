# This code's origin is as follows:
# It was first copied from mpmath source code.
#
# contributed to mpmath by Kristopher L. Kuhlman, February 2017
# contributed to mpmath by Guillermo Navas-Palencia, February 2022
#
# Then the mpmath code was altered to work without mpmath's arbitrary precision. for speed.
# The code was then altered to work with time vectors at once, also for speed.
# For for method of deHoog e.a., the code from Kuhlman was copied in as it is present in ttim,
# This was necessary to get this method to properly work.
# Finally, the class structure was replace by simple functions, as the class structure provided no benefit.

# Copyright 2019 Kristopher L. Kuhlman <klkuhlm _at_ sandia _dot_ gov>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from itertools import cycle
from scipy.special import factorial as fac, kv as Kv

def talbot(flpl, times, **kwargs):
    r"""Compute time-domain inversion of the laplace transform using the fixed Talbot algorithm .
    
    Parameters
    ----------
    flpl: fucnction
        the pointer to the laplace tranform function to be inverted.
        Must only depend on the laplace parameter. Use functools.partial to
        fix all other parameters before calling this function.
    times: sequence
        times for hich the inverse it be retured.
    kwargs: optional dict of extra prameters which include
            degree: <20> Number of terms            
            tmax: maximum time associated with vector of times
                    (typically just the time requested)
            r: abscissa for `p_0` (otherwise computed using rule
                of thumb `2M/5`)

    Returns
    -------
    Inversion of the laplace function for the given times.

    Theory
    ------    
    
    The "fixed" Talbot method deforms the Bromwich contour towards
    `-\infty` in the shape of a parabola. Traditionally the Talbot
    algorithm has adjustable parameters, but the "fixed" version
    does not. The `r` parameter could be passed in as a parameter,
    if you want to override the default given by (Abate & Valko,
    2004).

    The Laplace parameter is sampled along a parabola opening
    along the negative imaginary axis, with the base of the
    parabola along the real axis at
    `p=\frac{r}{t_\mathrm{max}}`. As the number of terms used in
    the approximation (degree) grows, the abscissa required for
    function evaluation tend towards `-\infty`, requiring high
    precision to prevent overflow.  If any poles, branch cuts or
    other singularities exist such that the deformed Bromwich
    contour lies to the left of the singularity, the method will
    fail.

    The working precision will be increased according to a rule of
    thumb. If 'degree' is not specified, the working precision and
    degree are chosen to hopefully achieve the dps of the calling
    context. If 'degree' is specified, the working precision is
    chosen to achieve maximum resulting precision for the
    specified degree.

    .. math ::

        p_0=\frac{r}{t}

    .. math ::

        p_i=\frac{i r \pi}{Mt_\mathrm{max}}\left[\cot\left(
        \frac{i\pi}{M}\right) + j \right] \qquad 1\le i <M

    where `j=\sqrt{-1}`, `r=2M/5`, and `t_\mathrm{max}` is the
    maximum specified time.

    The fixed Talbot time-domain solution is computed from the
    Laplace-space function evaluations using

    .. math ::

        f(t,M)=\frac{2}{5t}\sum_{k=0}^{M-1}\Re \left[
        \gamma_k \bar{f}(p_k)\right]

    where

    .. math ::

        \gamma_0 = \frac{1}{2}e^{r}\bar{f}(p_0)

    .. math ::

        \gamma_k = e^{tp_k}\left\lbrace 1 + \frac{jk\pi}{M}\left[1 +
        \cot \left( \frac{k \pi}{M} \right)^2 \right] - j\cot\left(
        \frac{k \pi}{M}\right)\right \rbrace \qquad 1\le k<M.

    Again, `j=\sqrt{-1}`.

    **References**

    1. Abate, J., P. Valko (2004). Multi-precision Laplace
        transform inversion. *International Journal for Numerical
        Methods in Engineering* 60:979-993,
        http://dx.doi.org/10.1002/nme.995
    2. Talbot, A. (1979). The accurate numerical inversion of
        Laplace transforms. *IMA Journal of Applied Mathematics*
        23(1):97, http://dx.doi.org/10.1093/imamat/23.1.97
    """
    if np.isscalar(times):
        times = np.array([times])
    # tmax = times[-1]
    
    M = kwargs.get('degree', 34)

    # Abate & Valko rule of thumb for r parameter
    r = kwargs.get('r', 2 / 5 * M)

    theta = (np.arange(M + 1, dtype=np.complex128) * np.pi / M)[:-1]
    cot = np.zeros_like(theta)
    cot[1:] = 1 / np.tan(theta[1:])
    
    # All but time-dependent part of p
    # This is p_i in the formula
    gamma  = np.zeros_like(theta)
    p      = np.zeros_like(theta)
    FP     = np.zeros_like(theta)
    results = np.zeros(len(times), dtype=np.complex128)
                    
    #p[1:] = self.r / tmax * theta[1:] * (cot[1:] + 1j)        
    #FP[1:]= np.array([fp(p_) for p_ in p[1:]], dtype=np.complex128)
    
    for it, t in enumerate(times):
        p = r / t * theta * (cot + 1j)
        p[0] = r / t
        FP= np.array([flpl(p_) for p_ in p], dtype=np.complex128)
        #FP[0] = fp(p[0])
        
        gamma[0] = np.exp(t * p[0])  / 2
        gamma[1:] =np.exp(t * p[1:]) * (
            1 + 1j * theta[1:] * (1 + cot[1:] ** 2) - 1j * cot[1:]) 

        results[it] = 2 / (5 * t) * np.dot(gamma, FP)

    return results.real

# ****************************************

def stehfest(flpl, times, **kwargs):
    r"""Compute time-domain inversion of the laplace transform using the Graver-Stehfest's algorithm .
    
    Parameters
    ----------
    flpl: fucnction
        the pointer to the laplace tranform function to be inverted.
        Must only depend on the laplace parameter. Use functools.partial to
        fix all other parameters before calling this function.
    times: sequence
        times for hich the inverse it be retured.
    kwargs: optional dict of extra prameters which include
            alpha <0.>,
            tol (1e-9>),
            degree <20>
            
    Returns
    -------
    Inversion of the laplace function for the given times.

    Theory
    ------    
    The Gaver-Stehfest method is a discrete approximation of the
    Widder-Post inversion algorithm, rather than a direct
    approximation of the Bromwich contour integral.
    The method abscissa along the real axis, and therefore has
    issues inverting oscillatory functions (which have poles in
    pairs away from the real axis).
    The working precision will be increased according to a rule of
    thumb. If 'degree' is not specified, the working precision and
    degree are chosen to hopefully achieve the dps of the calling
    context. If 'degree' is specified, the working precision is
    chosen to achieve maximum resulting precision for the
    specified degree.
    .. math ::
        p_k = \frac{k \log 2}{t} \qquad 1 \le k \le M
        
    
    Compute time-domain Stehfest algorithm solution.

    .. math ::

        f(t,M) = \frac{\log 2}{t} \sum_{k=1}^{M} V_k \bar{f}\left(
        p_k \right)

    where

    .. math ::

        V_k = (-1)^{k + N/2} \sum^{\min(k,N/2)}_{i=\lfloor(k+1)/2 \rfloor}
        \frac{i^{\frac{N}{2}}(2i)!}{\left(\frac{N}{2}-i \right)! \, i! \,
        \left(i-1 \right)! \, \left(k-i\right)! \, \left(2i-k \right)!}

    As the degree increases, the abscissa (`p_k`) only increase
    linearly towards `\infty`, but the Stehfest coefficients
    (`V_k`) alternate in sign and increase rapidly in sign,
    requiring high precision to prevent overflow or loss of
    significance when evaluating the sum.

    **References**

    1. Widder, D. (1941). *The Laplace Transform*. Princeton.
    2. Stehfest, H. (1970). Algorithm 368: numerical inversion of
        Laplace transforms. *Communications of the ACM* 13(1):47-49,
        http://dx.doi.org/10.1145/361953.361969           
    """
    
    def _coeff(M):
        r"""Salzer summation weights (aka, "Stehfest coefficients")
        only depend on the approximation order (M) and the precision.
        
        Salzer summation weights
        get very large in magnitude and oscillate in sign,
        if the precision is not high enough, there will be
        catastrophic cancellation.
        """
        if M % 2 != 0:
            M = int(M) + 1
        M2 = M // 2  # checked earlier that M is even
        
        V = np.zeros(M)

        for k in range(1, M + 1):
            z = np.zeros(min(k, M2) + 1)
            for j in range((k + 1) // 2, min(k, M2) + 1):
                z[j] = (j ** M2 * fac(2 * j) / (
                    fac(M2 - j) * fac(j) * fac(j - 1) * fac(k - j) * fac(2*j - k)))
            V[k-1] = (-1) ** (k + M2) * np.sum(z)

        return V

    if np.isscalar(times):
        times = [times]
        
    M = 2 * (kwargs.get('degree', 10) // 2)

    V = _coeff(M)
    
    result = np.zeros(len(times), dtype=float)

    p_no_t = np.arange(1, M+1) * np.log(2.)
    for it, t in enumerate(times):
        p = p_no_t / t            
        result[it] = np.sum(V * np.array([flpl(p_) for p_ in p])) * np.log(2.) / t

    return result.real

# ****************************************

def dehoog(flpl, times, **kwargs):
    r"""Compute time-domain inversion of the laplace transform using theDeHoog-Knight & Stokes algorithm .
    
    Parameters
    ----------
    flpl: fucnction
        the pointer to the laplace tranform function to be inverted.
        Must only depend on the laplace parameter. Use functools.partial to
        fix all other parameters before calling this function.
    times: sequence
        times for hich the inverse it be retured.
    kwargs: optional dict of extra prameters which include
            alpha <0.>,
            tol (1e-9>),
            degree <20>
            
    Returns
    -------
    Inversion of the laplace function for the given times.

    Theory
    ------    
    The de Hoog, Knight & Stokes algorithm is an
    accelerated form of the Fourier series numerical
    inverse Laplace transform algorithms.
    .. math ::
        p_k = \gamma + \frac{jk}{T} \qquad 0 \le k < 2M+1
    where
    .. math ::
        \gamma = \alpha - \frac{\log \mathrm{tol}}{2T},
    `j=\sqrt{-1}`, `T = 2t_\mathrm{max}` is a scaled time,
    `\alpha=10^{-\mathrm{dps\_goal}}` is the real part of the
    rightmost pole or singularity, which is chosen based on the
    desired accuracy (assuming the rightmost singularity is 0),
    and `\mathrm{tol}=10\alpha` is the desired tolerance, which is
    chosen in relation to `\alpha`.`
    When increasing the degree, the abscissa increase towards
    `j\infty`, but more slowly than the fixed Talbot
    algorithm. The de Hoog et al. algorithm typically does better
    with oscillatory functions of time, and less well-behaved
    functions. The method tends to be slower than the Talbot and
    Stehfest algorithsm, especially so at very high precision
    (e.g., `>500` digits precision).
    print("This method does nothing)
    """
    
    alpha = kwargs.get('alpha', 0.)
    tol = kwargs.get('tol', 1e-9)
    M = kwargs.get('degree', 20)

    # split up t vector in pieces of same order of magnitude, invert one piece
    #   at a time. simultaneous inversion for times covering several orders of 
    #   magnitudes gives inaccurate results for the small times.

    allt = np.asarray(times)			# save full times vector
    logallt = np.log10(allt)
    iminlogallt = int(np.floor(min(logallt)))
    imaxlogallt = int(np.ceil(max(logallt)))
    
    F = []
    
    # loop through all pieces, decimal cycles
    for ilogt in range(iminlogallt, imaxlogallt + 1):
    
        ts = allt[np.logical_and(logallt >= ilogt, logallt < ilogt + 1)]        
                
        if len(ts) > 0:			# maybe no elements in that magnitude

            T = np.max(ts) * 2
            gamma = alpha - np.log(tol) / (2 * T)
            
            # NOTE: The correction alpha -> alpha-log(tol)/(2*T) is not in de Hoog's
            #   paper, but in Mathematica's Mathsource (NLapInv.m) implementation of 
            #   inverse transforms
                        
            NP = 2 * M + 1

            # find F argument, call F with it, get 'a' coefficients in power series
            p = gamma + 1j * np.pi * np.arange(0, 2 * M + 1) / T # 2 M + 1 terms
            fp = np.array([flpl(p_) for p_ in p], dtype=np.complex128)    

            # build up e and q tables. superscript is now row index, subscript column
            #   CAREFUL: paper uses null index, so all indeces are shifted by 1 here
            
            # would it be useful to try re-using
            # space between e&q and A&B?
            e = np.zeros((NP, M + 1), dtype=np.complex128)
            q = np.zeros((NP, M),     dtype=np.complex128)
            d = np.zeros((NP,),       dtype=np.complex128)
            A = np.zeros((NP + 2,),   dtype=np.complex128)
            B = np.zeros((NP + 2,),   dtype=np.complex128)

            # initialize Q-D table
            e[0 : 2 * M, 0] = 0.0
            q[0, 0] = fp[1] / (fp[0] / 2.0)
            for i in range(1, 2 * M):
                q[i, 0] = fp[i + 1] / fp[i]

            # rhombus rule for filling triangular Q-D table (e & q)
            for r in range(1, M + 1):
                # start with e, column 1, 0:2*M-2
                mr = 2 * (M - r)
                e[0:mr, r] = q[1 : mr + 1, r - 1] - q[0:mr, r - 1] + e[1 : mr + 1, r - 1]
                if not r == M:
                    rq = r + 1
                    mr = 2 * (M - rq) + 1
                    for i in range(mr):
                        q[i, rq - 1] = q[i + 1, rq - 2] * e[i + 1, rq - 1] / e[i, rq - 1]

            # build up continued fraction coefficients (d)
            d[0] = fp[0] / 2.0
            for r in range(1, M + 1):
                d[2 * r - 1] = -q[0, r - 1]  # even terms
                d[2 * r] = -e[0, r]  # odd terms

            # seed A and B for recurrence
            A[0] = 0.0
            A[1] = d[0]
            B[0:2] = 1.0
            
            A_ = A.copy()
            B_ = B.copy()
            
            for t in ts:
                
                A[:] = A_[:]
                B[:] = B_[:]

                # base of the power series
                z = np.exp(1j * np.pi * t / T)

                # coefficients of Pade approximation (A & B)
                # using recurrence for all but last term
                for i in range(1, 2 * M):
                    A[i + 1] = A[i] + d[i] * A[i - 1] * z
                    B[i + 1] = B[i] + d[i] * B[i - 1] * z

                # "improved remainder" to continued fraction
                brem = (1.0 + (d[2 * M - 1] - d[2 * M]) * z) / 2.0
                rem = -brem * (1.0 - np.sqrt(1.0 + d[2 * M] * z / brem**2))
                if np.isnan(rem):
                    print(f"rem is {rem}")

                # last term of recurrence using new remainder
                A[NP] = A[2 * M] + rem * A[2 * M - 1]
                B[NP] = B[2 * M] + rem * B[2 * M - 1]

                # diagonal Pade approximation
                # F=A/B represents accelerated trapezoid rule
                F.append(np.exp(gamma * t) / T * (A[NP] / B[NP]).real)

    return np.array(F) # loop through time vector pieces

# ****************************************

def cohen(flpl, times, **kwargs):
    r"""Compute time-domain inversion of the laplace transform using theDeHoog-Knight & Stokes algorithm .
    
    Parameters
    ----------
    flpl: fucnction
        the pointer to the laplace tranform function to be inverted.
        Must only depend on the laplace parameter. Use functools.partial to
        fix all other parameters before calling this function.
    times: sequence
        times for hich the inverse it be retured.
    kwargs: optional dict of extra prameters which include
            alpha <0.>,
            tol (1e-9>),
            degree <20>
            
    Returns
    -------
    Inversion of the laplace function for the given times.

    Theory
    ------    
    The Cohen algorithm accelerates the convergence of the nearly
    alternating series resulting from the application of the trapezoidal
    rule to the Bromwich contour inversion integral.
    
    .. math ::
        p_k = \frac{\gamma}{2 t} + \frac{\pi i k}{t} \qquad 0 \le k < M
    
    where
    
    .. math ::
        \gamma = \frac{2}{3} (d + \log(10) + \log(2 t)),
    
    `d = \mathrm{dps\_goal}`, which is chosen based on the desired
    accuracy using the method developed in [1] to improve numerical
    stability. The Cohen algorithm shows robustness similar to the de Hoog
    et al. algorithm, but it is faster than the fixed Talbot algorithm.
    
    **Optional arguments**
    
    *degree*
        integer order of the approximation (M = number of terms)
    
    *alpha*
        abscissa for `p_0` (controls the discretization error)
    The working precision will be increased according to a rule of
    thumb. If 'degree' is not specified, the working precision and
    degree are chosen to hopefully achieve the dps of the calling
    context. If 'degree' is specified, the working precision is
    chosen to achieve maximum resulting precision for the
    specified degree.
    
    **References**
    1. P. Glasserman, J. Ruiz-Mata (2006). Computing the credit loss
    distribution in the Gaussian copula model: a comparison of methods.
    *Journal of Credit Risk* 2(4):33-66, 10.21314/JCR.2006.057
    
    The accelerated nearly alternating series is:

    .. math ::

        f(t, M) = \frac{e^{\gamma / 2}}{t} \left[\frac{1}{2}
        \Re\left(\bar{f}\left(\frac{\gamma}{2t}\right) \right) -
        \sum_{k=0}^{M-1}\frac{c_{M,k}}{d_M}\Re\left(\bar{f}
        \left(\frac{\gamma + 2(k+1) \pi i}{2t}\right)\right)\right],

    where coefficients `\frac{c_{M, k}}{d_M}` are described in [1].

    1. H. Cohen, F. Rodriguez Villegas, D. Zagier (2000). Convergence
    acceleration of alternating series. *Experiment. Math* 9(1):3-12

    """
    if np.isscalar(times):
        times = [times]

    N = kwargs.get('degree', 34)
    M = N + 1

    def gamma(t):
        dps = 26
        return 2 / 3 * (dps * np.log(10) + np.log(2 * t))        
    
    A = np.zeros(M)
    results = np.zeros(len(times))

    for it, t in enumerate(times):
        
        A[:] = 0.    
        d = (3 + np.sqrt(8)) ** N
        d = (d + 1 / d) / 2
        b = -1
        c = -d
        s = 0
                    
        p = gamma(t) / (2 * t) + 1j * np.pi * np.arange(
            M, dtype=np.complex128) / t    
        
        for m in range(M):
            A[m] = flpl(p[m]).real

        for k in range(N):
            c = b - c
            s = s + c * A[k + 1]
            b = 2 * (k + N) * (k - N) * b / ((2 * k + 1) * (k + 1))

        results[it] = np.exp(gamma(t) / 2) / t * (A[0] / 2 - s / d)

    return results

# ****************************************


# Laplace transforms of Bruggeman's solution 223_02
def fhat64(p, hb=None, r=None, R=None, S=None, kD=None):
    """Laplace transform of Phi Burgeman 223_02.
    
    using 64 bit float accuracy.
    """
    beta = np.sqrt(S / kD)
    return (hb / p * Kv(0, beta * r * np.sqrt(p)) /
                     Kv(0, beta * R * np.sqrt(p)))

def qhat64(p, hb=None, r=None, R=None, S=None, kD=None):
    """Lapace transform of Q of Bruggeman 223_02.
    
    using normal 64 bit float accuracy.
    """
    beta = np.sqrt(S / kD)
    return (2 * np.pi * r * np.sqrt(S * kD) * hb  / np.sqrt(p) *
            Kv(1, beta * r * np.sqrt(p)) /
            Kv(0, beta * R * np.sqrt(p)))


if __name__ == '__main__':

    invertors = {'talbot':   talbot,
                 'stehfest': stehfest,
                 'dehoog':   dehoog,
                 'cohen':    cohen,
                 }

    methods = ['talbot', 'dehoog', 'stehfest', 'cohen']
    
 
    print(f"Laplace inversion: {methods}")
    
    ts = np.logspace(-4, 6, 36)
    
    hb, r, R, S, kD = 1., 30., 30., 0.1, 1000.
    rs = 30., 60., 90, 120
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
    fig.suptitle(f"Bruggeman (1999) Solution 223_02\nNumerieke inversie volgens {methods}\nhb={hb}, S={S}, kD={kD}\nfloat64")
    ax1.set_title("Head, different numerical backtransformations")
    ax2.set_title("Q, different numerical backtransformations")
    ax1.set_ylabel("Phi")
    ax1.set_ylim(-0.1, 1.1)
    ax2.set_ylabel("Q")
    ax2.set_xlabel("t [d]")
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_ylim(1e-3, 1e4)
    ax1.grid(True)
    ax2.grid(True)

    use_mpm = False
    # ts = [10.]
    
    kwargs = {'tol': 1e-9, 'degree': 10}
    
    #for r in rs[1:2]:
    for r in rs:        
        fpF = partial(fhat64, hb=hb, r=r, R=rs[0], S=S, kD=kD)
        fpQ = partial(qhat64, hb=hb, r=r, R=rs[0], S=S, kD=kD)

        clrs = cycle("rbgmck")
        mks  = cycle("osx+s")
        fcs  = cycle(1.1 ** np.array([0, 1, 2, 3, 4, 4]))
        for method in methods:
            clr = next(clrs)
            mk  = next(mks)
            fc  = next(fcs)

            F_ = invertors[method](fpF, ts * fc)
            Q_ = invertors[method](fpQ, ts * fc)

            ax1.plot(ts * fc, F_, '-', marker=mk, color=clr, mfc='none', label=f'{method}, r={r}')
            ax2.plot(ts * fc, Q_, '-', marker=mk, color=clr, mfc='none', label=f'{method}, r={r}')
            
            # Finv = invlap(fpF, ts, **kwargs)
            # Qinv = invlap(fpQ, ts, **kwargs)
            # 
            # ax1.plot(ts, Finv.T, 'x-', color=clr, mfc='none', label='invlap, r={r}')
            # ax2.plot(ts, Qinv.T, 'x-', color=clr, mfc='none', label='invlap, r={r}')

            print(f"method={method:>20s}, r={r:6.0f}, F[::10]=", F_[::10])
            print(f"method={method:>20s}, r={r:6.0f}, Q[::10]=", Q_[::10])
        print()
        
    print("Done")

    ax1.legend(loc='lower right')
    ax2.legend(loc='lower right')

# fig.savefig(f"{method}.png")

plt.show()
# %%    
    
    

def invertlaplace(f, t, **kwargs):
    r"""Computes the numerical inverse Laplace transform for a
    Laplace-space function at a given time.  The function being
    evaluated is assumed to be a real-valued function of time.

    The user must supply a Laplace-space function `\bar{f}(p)`,
    and a desired time at which to estimate the time-domain
    solution `f(t)`.

    A few basic examples of Laplace-space functions with known
    inverses (see references [1,2]) :

    .. math ::

        \mathcal{L}\left\lbrace f(t) \right\rbrace=\bar{f}(p)

    .. math ::

        \mathcal{L}^{-1}\left\lbrace \bar{f}(p) \right\rbrace = f(t)

    .. math ::

        \bar{f}(p) = \frac{1}{(p+1)^2}

    .. math ::

        f(t) = t e^{-t}

    >>> from mpmath import *
    >>> tt = [0.001, 0.01, 0.1, 1, 10]
    >>> fp = lambda p: 1/(p+1)**2
    >>> ft = lambda t: t*exp(-t)
    >>> ft(tt),ft(tt)-dehoog(fp, tt[0])
    (0.000999000499833375, 8.57923043561212e-20)
    >>> ft(tt),ft(tt)-stehfest(fp,tt)
    (0.00990049833749168, 3.27007646698047e-19)
    >>> ft(tt),ft(tt)-cohen(fp,tt)
    (0.090483741803596, -1.75215800052168e-18)
    >>> ft(tt),ft(tt)-talbot(fp,tt)
    (0.367879441171442, 1.2428864009344e-17)
    """