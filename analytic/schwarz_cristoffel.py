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

# %% Intgration points between a and b density of points increases towards a and b

def unpack_u(u):
    """Return new parameters (robust)
    
    Make sure that during optiomisations subsequent points are always increasing
    
    Paramters
    ---------
    u: floats (u0, u1, u2, u3)
    """

    a = np.exp(u[0])      # BC
    b = np.exp(u[1])     # DE
    c = np.exp(u[2])  # extra marge
    
    # cumulative χ-coordinates
    xB = 1.0        # xB is kept fixed
    xC = xB + a
    xD = xB + a + b
    xE = xB + a + b + c

    return np.array([xB, xC, xD, xE])


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


def sc_along_real_ufunc(x, xP, k):
    """Return integration of Scharz-Cristoffel argument at x given points (sigularities at xP).
    
    Parameters
    ----------
    x: np.array
        points on the real axis
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
    
    for ix, xi in enumerate(xP):
        assert not np.isclose(np.abs(xi - xP).min()), f"x must not be close to xP[{ix}]={xP}"
    
    y = np.ones_like(x)
    for xi, ki in zip(xP, k):
        y *= (x - xi + 0j) ** (-ki)
        
    ym = 0.5 * (y[:-1] + y[1:])
    dx = np.diff(x)
    int_arg = ym * dx
    sc_integral = np.cumsum(int_arg.real) + 1j * np.cumsum(int_arg.imag)
    xm = 0.5 * (x[:-1] + x[1:])
    return sc_integral, xm


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
        ds = np.diff(s)        
        sm = 0.5 * (s[:-1] + s[1:]) # excludes a and b
        
        # --- compute argument of SC     
        prod = 1.0
        for xi, ki in zip(xP, k):
            prod *= (sm - xi + 0j) ** (-ki)  # force complex
        
        # --- integrate using trapezium rule
        sc_integral += np.sum(prod.real * ds + 1j * prod.imag * ds)
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
    ds = np.diff(s)        
    sm = 0.5 * (s[:-1] + s[1:]) # excludes a and b
    
    # --- compute argument of SC     
    prod = 1.0
    for xi, ki in zip(xP, k):
        prod *= (sm - xi + 0j) ** (-ki)  # force complex
    
    return  np.sum(prod.real * ds + 1j * prod.imag * ds)


# --- Objective function for least-squares ---
def objective(u, lengths):
    """
    u = np.log(scales), used to enforce positivity / better scaling
    Returns vector of differences between computed and desired lengths
    """
    # unpack u into χ-points: here we fix xB = 1.0, solve for xC, xD, xE
    xB = 1.0
    tC, tD, tE = np.exp(u)   # scale factors
    xC = xB + tC
    xD = xC + tD
    xE = xD + tE
    xP = [xB, xC, xD, xE]

    # desired lengths in z-plane
    desired_ratios = np.array([lengths[1] / lengths[0], lengths[2] / lengths[0]]) # [c/b e/b]
    
    # compute segment lengths
    computed = []
    for i in range(len(lengths)):
        L = sc_segment(i, xP, k)
        computed.append(np.abs(L)) # The length
    
    computed_ratios = np.array([computed[1] / computed[0], computed[2] / computed[0]])

    return np.sum((computed_ratios - desired_ratios) ** 2)

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

# %%
xP = 1, 2, 3, 4
k  = [1/2, 1/2, 1/2, 1/2]

X = np.linspace(5, 100)

w = []
for x in X[-2:]:
    w.append(sc_along_real_ax(x, xP, k))
w = np.array(w)

fig, ax = plt.subplots()
ax.plot(w.real, w.imag, '.')
ax.set_aspect(1)

# %%
if __name__ == '__main__':
    
    
    # %%
    # --- Initial guess ---
    u0 = np.log([0.5, 0.5, 1.0])  # rough starting guesses for xC-xB, xD-xC, xE-xD
    xP = unpack_u(u0)


    # --- Solve with least squares ---
    res = least_squares(objective, u0, xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=200, method='trf', args=[[b, c, e]])
    print("Optimization success:", res.success, res.message)

    # --- Extract χ-points ---
    xB = 1.0
    tC, tD, tE = np.exp(res.x)
    xC = xB + tC
    xD = xC + tD
    xE = xD + tE
    xP_new = [xB, xC, xD, xE]
    print("xp_new:", xP_new)

    chi = chi_to_chi1(xP_new, idxB=0, idxD=2)
    print("chi", chi)

    Q = 1
    Omega =Q * (0.5j -1j / np.pi * np.arcsin(chi + 0j)) # Between 0 at bottom to Q at top
    print(Omega.imag)

    # Let's compute a series of points and show them in the z-plain

    xp = np.hstack((np.linspace(        0 , xP_new[0], 20)[1:-1],
                    np.linspace(xP_new[1], xP_new[2], 20)[1:-1],
                    np.linspace(xP_new[2], xP_new[3], 20)[1:-1],
                    np.linspace(xP_new[3], xP_new[3] + 1., 20)[1:-1]
    ))

    Z = []
    for x in xp:
        Z.append(sc_along_real_ax(x, xP_new, k))
    Z = np.array(Z, dtype=complex)

    ZP = []
    for x in xP:
        ZP.append(sc_along_real_ax(x, xP_new, k))
    ZP = np.array(ZP, dtype=complex)


    fig, ax = plt.subplots()
    ax.set(title="Schwarz-Cristoffel plot")
    ax.plot(Z.real, Z.imag, 'b.', ms=1)
    ax.plot(ZP.real, ZP.imag, 'ro')

    ax.set_aspect(1.0)


    # Nu nog de ZP punten mappen naar de z-plane
    def get_pq(An, Bn, A, B):
        p = (An - Bn) / (A - B)
        q = (An * B - A * Bn) / (B - A)
        return p, q


    Pnts = np.array([A, B, C, D, E, F])

    p, q = get_pq(xP[0], xP[2], B, D)
    PntsA = p * np.array(xP_new) + q


    fig, ax = plt.subplots()
    ax.plot(Pnts.real, Pnts.imag, 'ro-')
    ax.plot(PntsA.real, PntsA.imag, 'b.-')
    ax.set_aspect(1)
    plt.show()
    print("Done")




# %%
