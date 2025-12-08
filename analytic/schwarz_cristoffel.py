# %% [markdown]
# 
# Schwarz Cristoffel for ditch cross section


# %%
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from scipy.special import erf


# %%

# --- Given angles / k_i ---
# Example values for the corners B, C, D, E
k = [0.5, -0.5, 0.5, 0.5]
b, c, e = 2, 0.5, 1.5
A, B, C, D, E, F = 5 + 0j, b + 0j, b - c * 1j, 0 - c * 1j, 0 - e * 1j, 5 - e * 1j

BE = np.array([B, C, D, E])
# %% Intgration points between a and b density of points increases towards a and b

def unpack_u(u):
    """Return new parameters (robust)
    
    Make sure that during optiomisations subsequent points are always increasing
    
    Paramters
    ---------
    u: floats (u0, u1)
    """

    a = np.exp(u[0])     # BC
    b = np.exp(u[1])     # CD
    # c = np.exp(u[2])     # DE
        
    # cumulative χ-coordinates
    xB = 1.0        # xB is kept fixed
    xC = 2.0
    xD = xC + a
    xE = xD + b

    return np.array([xB, xC, xD, xE])


def get_pq(Afrom, Bfrom, Ato, Bto):
    """Return p and q to transform A, B to An Bn.
    
    Parameters
    ----------
    An, Bn: complex numbers (coordinates)
        Coordinates of the new points
    A, B: complex numbers (coordinates)
        Coordinates of the old points
    
    Usage:
    ------
    Pnew = p * Pold + q
    """
    p = (Ato - Bto) / (Afrom - Bfrom)
    q = (Ato * Bfrom - Afrom * Bto) / (Bfrom - Afrom)
    return p, q

def erf_map(a, b, N=101, W=2.5):
    """Return points between a and b concentrated near a and b.

    Parameters
    ----------
    a, b: floats
        boundary of segment along the real axis
    N: int
        Number of points
    W: float, default 2..5
        max min value defined by (b - a) * erf(W)
    """
    s = np.linspace(-W, W, N)
    M = 0.5 * (a + b)
    L = 0.5 * (a - b)
    x = M - L * erf(s)
    return x[1:-1]

def integrate_trapz_complex(x, y):
    Ireal = np.trapz(np.real(y), x)
    Iimag = np.trapz(np.imag(y), x)
    return Ireal + 1j * Iimag

def sc_arg(Z, xP=None, k=None):
    """Return the argument of the Schwarz-Cristoffel expression.
    
    
    Parameters
    ----------
    Z: np.ndarray of complex | complex 
        Coordinates in the z-plane
    xP: scalars
        Points along real axis in z plane causing aburpt bending of polygon in w-plane
    k: scalars
        internal angles between successive polygon edges / pi
    """
    fvals = 1
    for xi, ki in zip(xP, k):
        fvals *= (s - xi + 0j) ** (-ki)
    return fvals

def sc_integral(Z, xP, k):
    """Return integration of Scharz-Cristoffel argument at x given points (sigularities at xP and k).
    
    The integration path is first vertical (along the imaginary axis) and then
    hoirzontal along the real axis.

    For each point we integrate from 0 vertically along the imaginary axis and then horizontally
    along the real axis.
    
    Parameters
    ----------
    Z np.array of complex | complex
        Points in the z-plane
    xP: floats (real)
        points along real axis causing boundary in z-plane to bend.
    k: float
        exponents arg will be (x - xi) ** (-ki), where ki = alpha / pi (alpha inner angle between sections)
    """    
    # along vertical
    ds = 1e-3
    zv = np.arange(0, Z.imag + eps, ds)
    zh = np.arange(0, Z.real + eps, ds)

    argv = sc_arg(1j * zv, xP, k)
    argh = sc_arg(     zh, xP, k)
     
    Iv = integrate_trapz_complex(1j * zv, 1j * argv)
    Ih = integrate_trapz_complex(     zh,      argh)       
    return Iv + Ih # w = p z + q


def sc_along_real_ax(x, xP, k):
    """Return integration of Scharz-Cristoffel argument at x given points (sigularities at xP).
    
    Parameters
    ----------
    x: float (real)
        Point on the real axis
    xP: floats (real)
        points along real axis causing boundary in z-plane to bend.
    k: float
        exponents arg will be (x - xi) ** (-ki), where ki = alpha / pi (alpha inner angle between sections)
    """
    xP = np.asarray(xP)
    sc_integral = 0 + 0j # Force complex
    
    # --- Always integrate from zero and pass points between 0 and x
    # --- It is assumed that all xP > 0
    assert np.all(xP > 0) and np.all(np.diff(xP) > 0), "All points must be > 0 and series must increase."
    
    as_ = np.hstack((0, xP))
    bs_ = np.hstack((xP, np.inf))
    
    for a, b in zip(as_, bs_):            
        if x <= a:
            break
        
        b = min(x, b) # x may be < than b
        s = erf_map(a, b, N=100, W=2.5)
        
        # --- compute argument of SC     
        y = 1.0
        for xi, ki in zip(xP, k):
            y *= (s - xi + 0j) ** (-ki)  # force complex
        
        sc_integral += integrate_trapz_complex(s, y)
    return sc_integral


def sc_segment(i, xP, k):
    """"Return integrate between xP[i] and xP[i+1], gets segment in z plane before rotation stretching and translation.
   
    Parameters
    ----------
    i: int 
        segment number
    xP: floats (real)
        points along x-axis, singularities
    k: floats (real)
        neg exponents SC argument ...(x - xi) ** (-ki) ...
    """
    a, b= xP[i], xP[i+1]
    s = erf_map(a, b, N=100, W=2.5)
    
    # --- compute argument of SC     
    y = 1.0
    for xi, ki in zip(xP, k):
        y *= (s - xi + 0j) ** (-ki)  # force complex
    
    return integrate_trapz_complex(s, y)


# --- Objective function for least-squares ---
def objective(u, lengths):
    """
    u = np.log(scales), used to enforce positivity / better scaling
    Returns vector of differences between computed and desired lengths
    """
    # unpack u into χ-points: here we fix xB = 1.0, solve for xC, xD, xE
    xP = unpack_u(u)
    
    # compute segment lengths
    computed = []
    for i in range(len(lengths)):
        L = sc_segment(i, xP, k)
        computed.append(np.abs(L)) # The length
    print("computed:" , computed)
    
    computed_ratios = np.array([computed[1] / computed[0], computed[2] / computed[0]])
    desired_ratios  = np.array([lengths[1] / lengths[0], lengths[2] / lengths[0]])
    
    print("computed_ratios ", computed_ratios)
    print("desired_ratio s  ", desired_ratios)
    return computed_ratios - desired_ratios

# ---- Leg B en D op -1 en 1 voor de arcsin
def chi_to_chi1(chi, idxB=0, idxD=2):
    """
    Mapping from points on x-axis to -1 and +1 for arcsin. The indices of the points are given.
    
    Linear mapping: chi[idxB] -> -1, chi[idxD] -> +1
    chi : array of chi-points (optimized pooints along the axis)
    idxB, idxD : indices of B and D in chi
    """
    chi = np.asarray(chi)
    B = chi[idxB]
    D = chi[idxD]
    scale = 2.0 / (D - B)
    chi1 = scale * (chi - B) - 1.0
    return chi1

def w_from_omega(Omega, Q=None, k=None, xP=None, BE=None):
    """Return w from omega using Scharz-Cristoffel for simple ditch profile.""" 
    zeta = np.sin(1j * np.pi/Q * Omega)
    p, q = get_pq(-1 + 0j, 1 + 0j, xP[0], xP[1])
    Z = p * zeta + q
    
    w0 = sc_along_real_ax(xP[0], xP, k)
    w2 = sc_along_real_ax(xP[2], xP, k)
    B, D = BE[0], BE[2]
    p1, q1 = get_pq(w0, w2, B, D)
        
    W = p1 * sc_integral(Z, xP, k) + q
    return W
    

# %%
if __name__ == '__main__':
    # --- Given angles / k_i ---
    k = [0.5, -0.5, 0.5, 0.5]
    
    # --- The cross section is defined by these lenghts
    b, c, e = 2, 0.5, 1.5

    # --- Corner points B, C, D, E and extra points A and F
    A, B, C, D, E, F = 5 + 0j, b + 0j, b - c * 1j, 0 - c * 1j, 0 - e * 1j, 5 - e * 1j

    # --- Segment lengths used by the objective function which mathes segment-length ratios
    segment_lengths = np.abs(C - B), np.abs(D - C), abs(E -D)

    # --- Initial guess ---
    u0 = np.log([1.0, 1.0])  # rough starting guesses for xC-xB, xD-xC, xE-xD
    
    # --- Solve with least squares ---
    res = least_squares(objective, u0,
                xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=200, method='lm', args=[segment_lengths])
    print("Optimization success:", res.success, res.message)
    print(res)

    # --- the optimized points
    xP = unpack_u(res.x)
    print("xp:", xP)

    # --- put points B and D in the z plane to -1 and 1 to use the arcsin
    chi = chi_to_chi1(xP, idxB=0, idxD=2)
    print("chi", chi)

    # --- Compute Omega in the omega plane
    Q = 1
    Omega =Q * (0.5j -1j / np.pi * np.arcsin(chi + 0j)) # Between 0 at bottom to Q at top
    print(Omega.imag)

    # --- Verify be computing the points in the w plane, the segment ratios are now
    # correct, which is enough for us to transfrom to the omega plane. But if we
    # wish to transfer the omega values back to the z-plane we need to transform
    # the points in the we plane to match the actual coordinates of the cross section.
    w = []
    for x in xP:
        # --- Compute w for xP points
        w.append(sc_along_real_ax(x, xP, k))
    w = np.array(w)

    p, q = get_pq(w[0], w[2], B, D)
    
    def z_to_w(x, xP, k):
        x = np.asaray(x, dtype=complex)
        w = []
        for xi in x.ravel():
            w.append(sc_along_real_ax(xi, xP, k))
        w = np.array(w).reshape(x.shape)
        W = p * w + q
        return W
        
    ditch = [A]
    for wi in w:
        ditch.append(p * wi + q)
    ditch.append(F)
    ditch = np.array(ditch)
        
    
    fig, ax = plt.subplots()
    ax.plot(w.real, w.imag, 'bo--', label='w-plane')
    ax.plot(ditch.real, ditch.imag, 'go-', label='ditch in w plane' )
    ax.legend()
    
    # %%
    # Back from omega:
    eps = 1e-6
    psi = np.linspace(-1, 1, 21).clip(-1 + eps, 1 - eps) * Q / 2
    phi = np.linspace(0, 2, 41).clip(eps, None)
    Phi, Psi = np.meshgrid(phi, psi)
    Omega = Phi + 1j * Psi
    
    Z = w_from_omega(Omega, Q=1, k=k, xP=xP, BE=BE)
    
    fig, ax = plt.subplots()

    ax.plot(Z.real, Z.imag, 'b-', lw=0.5, label='Phi')
    ax.plot(Z.real.T, Z.imag.T, 'g-', lw=0.5, label='Psi')
    ax.set_aspect(1)
    ax.legend()

    plt.show()

    # move to xB, xD
