# %% [markdown]
# 
# Schwarz Cristoffel for ditch cross section

# %%
import os
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from scipy.special import erf


def unpack_u(u):
    """Return new parameters (robust)
    
    Make sure that during optiomisations subsequent points are always increasing
    
    Paramters
    ---------
    u: floats (u0, u1)
    """

    delta1 = np.exp(u[0])     # BC
    delta2 = np.exp(u[1])     # CD
    
        
    # cumulative χ-coordinates, xB and xC are kept fixed
    xB = 1.0
    xC = 2.0
    xD = xC + delta1
    xE = xD + delta2

    return np.array([xB, xC, xD, xE])


def get_pq(Afr, Bfr, Ato, Bto, test = False):
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
    p = (Ato - Bto) / (Afr - Bfr)
    q = (Ato * Bfr - Afr * Bto) / (Bfr - Afr)
    
    if test:
        print(f"Ato={Ato:.4g} Afr={Afr:.4g}, Bto={Bto}, Bfr={Bfr:.4g}")
        print(f"p={p:.4g}, q={q:.4g}")
        print(f"p Afr + q ={p * Afr + q:.4g}, p Bfr + q ={p * Bfr + q:.4g}")
    
    return p, q

def erf_grid(a, b, N=100, W=2.5, dense_side=None):
    """Return points between a and b concentrated near a and b.

    Parameters
    ----------
    a, b: floats
        boundary of segment along the real axis
    N: int
        Number of points
    W: float, default 2..5
        max min value defined by (b - a) * erf(W)
    dense_side: str | None default = 'both'
        which side has dense coordinates ('left', 'right' or 'both')?
    """
    valid = {'l': 'L', 'r': 'R', 'b': 'B'}

    try:
        dense_side = valid[dense_side.strip().lower()[0]]
    except Exception:
        raise ValueError("dense_side must be 'left', 'right' or 'both'")
    
    if dense_side in ['L', 'R']:
        N = N + 1
    else:
        N = N + 2
    
    s = np.linspace(-W, W, N)
    M = 0.5 * (a + b)
    L = 0.5 * (a - b)
    x = M - L * erf(s)

    if dense_side == 'L':
        return x[1:]

    if dense_side == 'R':
        return x[:-1]

    return x[1:-1]

def integrate_trapz_complex(z, fz):
    """Return complex line integral ∫ f(z) dz along a polyline.
    
    Parameters
    ----------
    z : complex array
        Complex points along the path, in order.
    fz : complex array
        Values f(z) at those points.
    """
    assert len(z) == len(fz), "z and fz must have the same length"

    fm = 0.5 * (fz[:-1] + fz[1:])
    dz = np.diff(z)
    return np.sum(fm * dz)


# def sc_arg(Z, xP=None, k=None):
#     """Return the argument of the Schwarz-Cristoffel expression.
#     
#     
#     Parameters
#     ----------
#     Z: np.ndarray of complex | complex 
#         Coordinates in the z-plane
#     xP: scalars
#         Points along real axis in z plane causing aburpt bending of polygon in w-plane
#     k: scalars
#         internal angles between successive polygon edges / pi
#     """
#     fvals = 1
#     for xi, ki in zip(xP, k):
#         fvals *= (Z - xi + 0j) ** (-ki)
#     return fvals


def sc_arg(Z, xP, k):
    f_total = np.ones_like(Z, dtype=complex)

    for xi, ki in zip(xP, k):
        dz = Z - xi

        r = np.abs(dz)
        ang = np.angle(dz)
        ang_corr = np.unwrap(ang)

        log_dz = np.log(r) + 1j*ang_corr
        term = np.exp(-ki * log_dz)

        f_total *= term

    return f_total


def zeta0_fr_omega(omega, Q):
    """Transform omega = Phi + i Psi to the zeta_plane (0 < Psi / Q < 0.5).
    
    Procedure, rotate (* i) multiply by pi/Q and subtract 1.2 to centralize    
    so that -pi/2 x < pi/2. Centralization implies that 0 < Psi< Q
    Finally use sin to flatten the lines +/- pi/2    
    """
    return np.sin(1j * np.pi / Q * (omega + 0j) + np.pi / 2)

def zeta_fr_zeta0(zeta0, xP0, xP1):
    """Linearly map zeta_0 to zeta xP0 -> zeta0=-1, xP1-> zeta0=+1 for arcsin.
    """
    p, q = get_pq(-1, 1, xP0, xP1)
    return zeta0 * p + q
    
def w_fr_x(X, xP=None, k=None):
    X = np.asarray(X)
    w = []
    for x in X:
        w.append(w_fr_one_x(x, xP, k))        
    w = np.array(w).reshape(X.shape)
    return w if w.size > 1 else w.item()
        
def w_fr_one_x(x, xP=None, k=None):
    """Return Scharz-Cristoffel w-points from xP on real axis.
    
    This integrates along the singularities xP if passed.
    w_fr_zeta, uses w_fr_x and add path along imaginary axis to reach genearl point zeta
    
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
    w = 0 + 0j # Force complex
    
    # --- It is assumed that all xP > 0
    assert np.all(xP > 0) and np.all(np.diff(xP) > 0), "All points must be > 0 and series must increase."
    
    # --- Integrate from 0 along segments between singular points on real zeta axis
    xP_ext = np.hstack((0, xP, np.inf))
    
    for xa, xb in zip(xP_ext[:-1], xP_ext[1:]):
        if x < 0:
            a, b = 0, x
            s = erf_grid(a, b, dense_side='left')
            argc = sc_arg(s, xP=xP, k=k)
            return integrate_trapz_complex(s, argc)
        if x <= xa:
            break
            
        a, b = xa, min(xb, x)
        if np.isclose(a, b):
            return w
        if a == xP[-1]:
            s = erf_grid(a, b, dense_side='left')
        else:
            s = erf_grid(a, b, dense_side='both')
        argc = sc_arg(s, xP=xP, k=k)
        w += integrate_trapz_complex(s, argc)
    return w


def w_fr_zeta(Zeta, xP, k):
    """Return integration of Scharz-Cristoffel argument at x given points (sigularities at xP and k).
    
    The integration path is a detaul in 3 branches that avoids passing closely by the
    prevertices. This way, far less discrtization points are necessary.
    Could be further improved.
    First branch from 0 to 2j to avoid horizontal branch close to real axis with vertices.
    Second branch horizontal to real part of zete (z1 + zeta.real).
    Third path vertically from z1 + zeta.real to zeta.
    
    Parameters
    ----------
    Z np.array of complex | complex
        Points in the z-plane
    xP: floats (real)
        points along real axis causing boundary in z-plane to bend.
    k: float
        exponents arg will be (x - xi) ** (-ki), where ki = alpha / pi (alpha inner angle between sections)
    """    
    w  = []
    for zeta in Zeta.ravel():
        z1 = 2j       
        # --- Vertical to get away from the pervertices take 2
        zf = 0 + 1j * erf_grid(0, z1.imag, dense_side='both')
        zf_arg = sc_arg(zf, xP, k)
        If = integrate_trapz_complex(zf, zf_arg)

        # --- Horizontal path from 1j * zeta.imag
        zh = zf[-1] + erf_grid(0, zeta.real, dense_side='both')
        zh_arg = sc_arg(zh, xP, k)
        Ih = integrate_trapz_complex(zh, zh_arg)

        # --- Vertical path from zero
        zv = zh[-1] + 1j * erf_grid(0, zeta.imag - z1.imag, dense_side='both')
        zv_arg = sc_arg(zv, xP, k)
        Iv = integrate_trapz_complex(zv, zv_arg)
        
        # --- Total path
        w.append(If + Ih + Iv)
    return np.array(w).reshape(Zeta.shape)


def z_fr_w(w, wA, wB, zA, zB):
    """Map points in w-plain to final real-world z-plane.
    
    Parameters
    ----------
    w: complex number(s)
        points in the w-plane
    wA, wB, zA, zB: complex numbers
        Two points in the w-plane that map on two points in the z_plane.
    """
    w = np.asarray(w, dtype=complex)
    p, q = get_pq(wA, wB, zA, zB)
    return  p * w + q


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
    s = erf_grid(a, b, dense_side='both')
    arg = sc_arg(s, xP=xP, k=k)
    return integrate_trapz_complex(s, arg)


# --- Objective function for least-squares ---
def objective(u, k, lengths):
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
    # print("computed:" , computed)
    
    computed_ratios = np.array([computed[1] / computed[0], computed[2] / computed[0]])
    desired_ratios  = np.array([lengths[1] / lengths[0], lengths[2] / lengths[0]])
    
    print("computed_ratios ", computed_ratios)
    print("desired_ratio s  ", desired_ratios)
    return computed_ratios - desired_ratios
    
    
def phase(z):
    return np.angle(z)   # in (-pi, pi]

def detect_phase_jumps(Z, F, threshold=np.pi/2):
    """
    Detect large phase jumps (branch cuts) in a 2D complex field F(Z).
    Returns mask of same shape as Z/F.
    """
    ph = np.angle(F)

    # robust wrapped phase difference (mod 2π)
    dpx = np.abs(np.angle(np.exp(1j * (ph[:, 1:] - ph[:, :-1]))))   # (M, N-1)
    dpy = np.abs(np.angle(np.exp(1j * (ph[1:, :] - ph[:-1, :]))))   # (M-1, N)

    # initialize mask
    mask = np.zeros_like(ph, dtype=bool)

    # place the differences back into full grid shape
    mask[:, 1:] |= dpx > threshold
    mask[1:, :] |= dpy > threshold

    return mask


# %%

def test_sc_mapping(case=1):
    # --- Sets of combinations of points along x and corners (k = alpha/pi values)

    if case == 0:
        title = "triangle"
        xP = np.array([1, 2, 3, 4])
        k  = np.ones_like(xP) * 2 / 3
    elif case == 1:
        title = r"Four left corners of $\pi/2$"
        xP = np.array([1, 2, 3, 4, 5])
        k  = np.ones_like(xP) / 2
    elif case == 2:
        title="Pentagon"
        xP = np.array([1, 2, 3, 4, 5, 6])
        k  = np.ones_like(xP) * 2 / 5
    elif case == 3:
        title="There is not way to tell where the points will land"
        xP = np.array([1, 2, 3.5, 4.0, 4.75, 4.9, 5])
        k  = np.ones_like(xP) / 3
    elif case == 4:
        title="Ditch half cross section"
        xP = np.array([1, 2, 3, 4])
        k  = np.array([1, -1, 1, 1]) / 2
 
    w = w_fr_x(xP, xP, k)

    fig, ax = plt.subplots()
    ax.plot(w.real, w.imag, '.-')

    ax.set_title(title)
    for ip, wi in enumerate(w):
        ax.text(wi.real, wi.imag, f"{ip}", ha='left', va='bottom')
    ax.set_aspect(1)
    
    plt.show()

    
def main():
    
    # --- Given angles / k_i ---
    # Example values for the corners B, C, D, E
    k = [0.5, -0.5, 0.5, 0.5]
 
    # --- The cross section is defined by these lenghts
    b, c, d = 0.5, 1, 2.5
    A, B, C, D, E, F = 5 + 0j, c + 0j, c - b * 1j, 0 - b * 1j, 0 - (b + d) * 1j, 5 - (b + d) * 1j

    AF = np.array([A, B, C, D, E, F])
    BE = AF[1:-1]

    # --- Segment lengths used by the objective function which mathes segment-length ratios
    segment_lengths = np.abs(C - B), np.abs(D - C), abs(E -D)

    # --- Initial guess ---
    u0 = np.log([1.0, 1.0])  # rough starting guesses for xC-xB, xD-xC, xE-xD
    
    # --- Solve with least squares ---
    res = least_squares(objective, u0,
                xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=200, method='lm', args=[k, segment_lengths])
    print("Optimization success:", res.success, res.message)
    print(res)

    # --- the optimized points
    xP = unpack_u(res.x)
    print("xp:", xP)
    
    # --- Plot zeta, w and Z
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Schwarz-Cristoffel half-ditch X-section")
    
    ax1.set(title="zeta", xlabel='xi', ylabel='ypsilon')
    ax2.set(title="w-plane", xlabel='u', ylabel='v')
    ax3.set(title='z-plane', xlabel='x', ylabel='y')
    
    ax1.grid()
    ax2.grid()
    ax3.grid()
    
    # --- Plot the xP points in the zeta plane
    ax1.plot(xP.real, xP.imag, 'bo-', label='points xP')

    # --- Compute the images w(xP) and show them in the w-plane
    wP = w_fr_x(xP, xP, k)
    ax2.plot(wP.real, wP.imag, 'bo-', label='xP --> w')
    
    # --- Plot these points together with the original ditch corner points
    zP = z_fr_w(wP, wP[0], wP[2], BE[0], BE[2])
    ax3.plot(zP.real, zP.imag, 'bo-', label='zP (back transformed)')
    ax3.plot(AF.real, AF.imag, 'go-', ms=12, mfc='none', label='original points')
    
    # --- Try an arbitrary polygon in the zeta plane
    # zetaPgon = np.array([0 + 1j, 2 + 2j, 2 + 3j, 0 + 4j, -2 + 3j, -2 + 2j, 0 + 1j])
    # wPgon = w_fr_zeta(zetaPgon, xP, k)
    # zPgon = z_fr_w(wPgon, wP[0], wP[2], BE[0], BE[2])
    # 
    # ax1.plot(zetaPgon.real, zetaPgon.imag, 'r.--', label='polygon in zeta')
    # ax2.plot(wPgon.real, wPgon.imag, 'r.--', label='polygon in w')
    # ax3.plot(zPgon.real, zPgon.imag, 'r.--', label='polygon in Z')
            
    # --- From Omega to Z
    eps = 1e-3
    Q = 1.0
    psi = np.linspace(1, 0, 21).clip(eps, 1 - eps) * Q # highest psi on top for convenience
    # psi = np.linspace(1, 0, 21)[1:20] * Q # highest psi on top for convenience
    
    phi = np.linspace(0, 2, 41).clip(eps, None) * Q
    phi = np.linspace(0, 2, 41).clip(eps, None) * Q
    Phi, Psi = np.meshgrid(phi, psi)
    Omega = (Phi + 1j * Psi) 
    
    zeta0 = zeta0_fr_omega(Omega, Q)
    zeta  = zeta_fr_zeta0(zeta0, xP[0], xP[2])
    w     = w_fr_zeta(zeta, xP=xP, k=k)
    Z     = z_fr_w(w, wP[0], wP[2], BE[0], BE[2])
    
    ax1.plot(zeta.real,   zeta.imag,   'b', lw=0.35)
    ax1.plot(zeta.real.T, zeta.imag.T, 'g', lw=0.35)
    ax2.plot(w.real,   w.imag,   'b', lw=0.35)
    ax2.plot(w.real.T, w.imag.T, 'g', lw=0.35)
    ax3.plot(Z.real,   Z.imag,   'b', lw=0.35)
    ax3.plot(Z.real.T, Z.imag.T, 'g', lw=0.35 )
    
    for ax in [ax1, ax2, ax3]:
        ax.grid(True)
        ax.set_aspect(1)
        #ax.legend(loc='best')
        
    images = '/Users/Theo/GRWMODELS/python/tools/analytic/images'
    fig.savefig(os.path.join(images, 'SC_ditch.png'))
        
    return
    
if __name__ == '__main__':
    #for case in [0, 1, 2, 3, 4]:
    #    test_sc_mapping(case=case)
    
    main()
    
    plt.show()


