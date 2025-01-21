# INVLAP  numerical inverse Laplace transform
#
# f = invlap(F, t, alpha, tol, P1,P2,P3,P4,P5,P6,P7,P8,P9);
#         
# F       laplace-space function (string refering to an m-file), 
#           must have form F(s, P1,..,P9), where s is the Laplace parameter,
#           and return column vector as result
# t       column vector of times for which real-space function values are
#           sought
# alpha   largest pole of F (default zero)
# tol     numerical tolerance of approaching pole (default 1e-9)
# P1-P9   optional parameters to be passed on to F
# f       vector of real-space values f(t)
#
# example: identity function in Laplace space:
#   function F = identity(s);                    # save these two lines
#            F = 1./(s.^2);                      # ...  as "identity.m"
#   invlap('identity', [1;2;3])                  # gives [1;2;3]
#
# algorithm: de Hoog et al's quotient difference method with accelerated 
#   convergence for the continued fraction expansion
#   [de Hoog, F. R., Knight, J. H., and Stokes, A. N. (1982). An improved 
#    method for numerical inversion of Laplace transforms. S.I.A.M. J. Sci. 
#    and Stat. Comput., 3, 357-366.]
# Modification: The time vector is split in segments of equal magnitude
#   which are inverted individually. This gives a better overall accuracy.   
##  details: de Hoog et al's algorithm f4 with modifications (T->2*T and 
#    introduction of tol). Corrected error in formulation of z.
#
#  Copyright: Karl Hollenbeck
#             Department of Hydrodynamics and Water Resources
#             Technical University of Denmark, DK-2800 Lyngby
#             email: karl@isv16.isva.dtu.dk
#  22 Nov 1996, MATLAB 5 version 27 Jun 1997 updated 1 Oct 1998
#  IF YOU PUBLISH WORK BENEFITING FROM THIS M-FILE, PLEASE CITE IT AS:
#    Hollenbeck, K. J. (1998) INVLAP.M: A matlab function for numerical 
#    inversion of Laplace transforms by the de Hoog algorithm, 
#    http://www.isva.dtu.dk/staff/karl/invlap.htm 


import numpy as np

def invlap(flpl, t, **kwargs):

    alpha = kwargs.get('alpha', 0.)
    tol = kwargs.get('tol', 1e-9)
    
    # split up t vector in pieces of same order of magnitude, invert one piece
    #   at a time. simultaneous inversion for times covering several orders of 
    #   magnitudes gives inaccurate results for the small times.

    allt = np.asarray(t)			# save full times vector
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
                        
            M = kwargs.get('degree', 20)
            NP = 2 * M + 1

            # find F argument, call F with it, get 'a' coefficients in power series
            p = gamma + 1j * np.pi * np.arange(0, 2 * M + 1) / T # 2 M + 1 terms
            fp = np.array([flpl(p_) for p_ in p], dtype=np.complex128)    

            # build up e and q tables. superscript is now row index, subscript column
            #   CAREFUL: paper uses null index, so all indeces are shifted by 1 here
            
            # would it be useful to try re-using
            # space between e&q and A&B?
            e = np.empty((NP, M + 1), dtype=np.complex64)
            q = np.empty((NP, M), dtype=np.complex64)
            d = np.empty((NP,), dtype=np.complex64)
            A = np.empty((NP + 2,), dtype=np.complex64)
            B = np.empty((NP + 2,), dtype=np.complex64)

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

                # last term of recurrence using new remainder
                A[NP] = A[2 * M] + rem * A[2 * M - 1]
                B[NP] = B[2 * M] + rem * B[2 * M - 1]

                # diagonal Pade approximation
                # F=A/B represents accelerated trapezoid rule
                F.append(np.exp(gamma * t) / T * (A[NP] / B[NP]).real)

        
    return np.array(F) # loop through time vector pieces