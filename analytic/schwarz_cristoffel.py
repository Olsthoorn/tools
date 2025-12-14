# %% [markdown]
# 
# Schwarz Cristoffel for ditch cross section

# %%
import os
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from scipy.special import erf

# %% Functions

def unpack_u(u):
    """Return the new parameters after during and after optimation in a robust way.
    
    Make sure that during optimisations subsequent points are always increasing
    
    Paramters
    ---------
    u: floats (u0, u1)
        The parameters passed to the objective function.
    """

    delta = np.exp(u)     # BC
        
    xP = np.hstack((1, 2, 2 * np.ones(len(u))))
    xP[2:] += np.cumsum(delta)
    
    return xP


def get_pq(Afr, Bfr, Ato, Bto, test=False):
    """
    Return p and q for the linear complex mapping

        z ↦ p*z + q

    such that Afr ↦ Ato and Bfr ↦ Bto.

    Parameters
    ----------
    Afr, Bfr : complex
        Points in the original domain.
    Ato, Bto : complex
        Target points after transformation.

    Returns
    -------
    p, q : complex
        Scale+rotation (p) and translation (q).

    Examples
    --------
    >>> p, q = get_pq(1+2j, 3+0.5j, 4+2j, 5+2j)
    >>> abs(p - (0.32+0.24j)) < 1e-12
    True
    >>> abs(q - (4.16+1.12j)) < 1e-12
    True
    """
    p = (Ato - Bto) / (Afr - Bfr)
    q = (Ato * Bfr - Afr * Bto) / (Bfr - Afr)
    
    if test:
        print(f"Ato={Ato:.4g} Afr={Afr:.4g}, Bto={Bto}, Bfr={Bfr:.4g}")
        print(f"p={p:.4g}, q={q:.4g}")
        print(f"p Afr + q ={p * Afr + q:.4g}, p Bfr + q ={p * Bfr + q:.4g}")
    
    return p, q

class AffineMap:
    """
    The unique affine map z = p * w + q that sends wA → zA and wB → zB.
    
    After construction, the object is callable:
        f = AffineMap(wA, wB, zA, zB)
        z = f(w)        # maps scalar or array-like
    """
    def __init__(self, wA, wB, zA, zB):
        if np.isclose(wA, wB):
            raise ValueError(f"wA and wB must be distinct. Got wA={wA}, wB={wB}.")

        self.p = (zA - zB) / (wA - wB)
        self.q = zA - self.p * wA

    def __call__(self, w):
        """Apply the affine map to w (scalar or array-like)."""
        return self.p * np.asarray(w) + self.q

    def inverse(self, z):
        """Evaluate the inverse affine map at z."""
        return (z - self.q) / self.p
    
    def inv_map(self):
        """If you *do* want the inverse *as a function object*, still possible."""
        p_inv = 1 / self.p
        q_inv = -self.q / self.p
        return lambda z: p_inv * z + q_inv

    def __repr__(self):
        return f"AffineMap(p={self.p}, q={self.q})"


def erf_grid(a, b, N=25, W=2.5, dense_side=None):
    """
    Return points between a and b concentrated near a and b.
    
    Parameters
    ----------
    a, b: floats
        boundary of segment along the real axis
    N: int
        Number of points
    W: float, default 2.5
        controls density
    dense_side: str | None
        which side has dense coordinates ('left', 'right', 'both')
    
    Returns
    -------
    np.ndarray
    
    Examples
    --------
    >>> x = erf_grid(0, 1, N=5, dense_side='both')
    >>> len(x)
    5
    >>> 0 <= x[0] < x[-1] <= 1
    True

    >>> x_left = erf_grid(0, 1, N=5, dense_side='left')
    >>> x_left[0] > 0  # first point slightly away from a
    True

    >>> x_right = erf_grid(0, 1, N=5, dense_side='right')
    >>> x_right[-1] < 1  # last point slightly before b
    True

    >>> np.all(np.diff(x) > 0)
    True
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


def subdiv(A, B, xP, N=25, eps=1e-6, endclip=1e-4, show=False):
    """Return subdivided line AB.
    
    Subdivides line AB such that de density of the points is iversily
    proportional to the distance towards the prevertices xP. This
    prevents taking too large steps near these singularities when
    carrying a complex integration along path AB.
    
    Parameters
    ----------
    A, B: complex numbers
        The ends of a straight line in complex space.
    xP: iterable of prevertices
        Points to take into considation.
    N: int
        Number of subdivisions
    eps: small int
        Prevents distance to become zero.
    endclip: small int
        Prevents the line AB to land on a prevertex.
    show: bool
        allows showing the subdivisions.
        
    Examples
    --------
    A simple subdivision with one prevertex in the middle::

    >>> Z = subdiv(0+0j, 1+0j, [0.5], N=4)
    >>> len(Z)
    5
    >>> Z[0]
    0j
    >>> Z[-1]
    (1+0j)

    The spacing is finer near the prevertex 0.5::

    >>> ds = np.abs(np.diff(Z))
    >>> ds[1] < ds[0]      # just left of 0.5 is tighter than near 0.0
    True
    >>> ds[2] < ds[0]      # just right of 0.5 also tighter
    True
    
    """
    # parameter x ∈ [0,1]
    alpha = np.linspace(0, 1, 100).clip(0, 1 - endclip)  # dense pre-grid
    
    # fysieke punten
    Z = A + alpha * (B - A)
    
    # distance-based weight
    w = np.min(np.abs(Z[:, None] - np.array(xP)[None, :]), axis=1) + eps
    
    # integrate 1/w
    s = np.cumsum(1/w)
    s /= s[-1]                       # normalize to [0,1]

    # invert s(α): given s_k → α_k
    
    alpha_new = np.interp(np.linspace(0, 1, N+1), s, alpha)

    # final points
    Z_new = A + alpha_new * (B - A)
    # ds = np.abs(np.diff(Z_new))

    if show:
        fig, ax = plt.subplots()
        ax.plot(xP, np.zeros_like(xP), 'o')
        ax.plot(Z_new.real, Z_new.imag, '.')
        plt.show()

    return Z_new # , ds



def integrate_trapz_complex(z, fz):
    """Return complex line integral ∫ f(z) dz along a polyline.
    
    The path does not have to be straight, but the integration
    wil require some density to be accurate enough.
    
    Parameters
    ----------
    z : complex array
        Complex points along the path.
    fz : complex array
        Values f(z) at those points.
    """
    assert len(z) == len(fz), "z and fz must have the same length"

    fm = 0.5 * (fz[:-1] + fz[1:])
    dz = np.diff(z)
    return np.sum(fm * dz)


def sc_arg(Z, xP, k):
    r"""Return the argument of the Scwarz-Christoffel integral.
    
    The argument is $\Pi_i=0^{len(x_P)} (z - x_i)^{-k_i}$
    
    The computation os robust using unwrap and the log.
    
    Parameters
    ----------
    Z: np.ndarray of complex numbers (coordinates)
        The points for which the argument is to be computed
    xP: iterable of prevertices
        The prevertices (points along the real axis)
    k: ierable of floats. Some length as xP
        the inner angles between successive edges of the domain divided by pi

    Examples
    --------
    With all exponents k = 0 the result is identically 1::

    >>> Z = np.array([0+1j, 1+1j])
    >>> sc_arg(Z, [0, 2], [0, 0])
    array([1.+0.j, 1.+0.j])

    With a single prevertex the function reduces to (z - x)^(-k)::

    >>> Z = np.array([1+1j, 2+1j])
    >>> out = sc_arg(Z, [0], [1])
    >>> expected = 1 / (Z - 0)
    >>> np.allclose(out, expected)
    True

    The function is multiplicative over prevertices::

    >>> Z = np.array([1+1j])
    >>> s1 = sc_arg(Z, [0], [1])
    >>> s2 = sc_arg(Z, [2], [2])
    >>> combined = sc_arg(Z, [0, 2], [1, 2])
    >>> np.allclose(combined, s1 * s2)
    True

    Output shape matches input shape::

    >>> Z = np.array([1+1j, 2+2j])
    >>> out = sc_arg(Z, [0], [1])
    >>> out.shape == Z.shape
    True

    """
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
    
    The zeta-plane is the start of the Schwarz-Cristoffel transformation, the
    plane in which the pervertices are on the real axis, that will produce
    the right shape in the w-plane by the SC-tranformation.

    Omega is the plane Phi + i Psi with Phi the heads straight vertical lines
    and Psi the striaght horizontal stream function lines.
    
    Procedure, rotate (* i) multiply by pi/Q and subtract 1.2 to centralize    
    so that -pi/2 x < pi/2. Centralization implies that 0 < Psi< Q
    Finally use sin to flatten the lines +/- pi/2    
    """
    return np.sin(1j * np.pi / Q * (omega + 0j) + np.pi / 2)


def zeta_fr_zeta0(zeta0, xP0, xP1):
    """Linearly map zeta_0 to zeta xP0 -> zeta0=-1, xP1-> zeta0=+1 for arcsin.
    
    In zeta0-plane the points -1 and 1 form two prevertices (of choice) this transoformation
    moves -1 and 1 in the zeta0 plane to xp0 and xP1 in the zeta_plane.    
    """
    f = AffineMap(-1, 1, xP0, xP1)
    return f(zeta0)
    
def w_fr_x(X, xP=None, k=None):
    r"""Return the w-plane coordinates of the points X along the real zeta-axis.
    
    Parameters
    ----------
    X point or points (real values)
        points along the real zeta-axis.
    xP: iterable of real numbers
        prevertices on the real zeta-axis.
    k: iterable of length (xP) of real numbers.
        Schwarz-Cristoffel angles/pi (negative powers) in
        $\Pi_i=0^N (x - xP_i)^{(-k_i)}$
    """
    X = np.asarray(X)
    w = []
    for x in X:
        w.append(w_fr_one_x(x, xP, k))        
    w = np.array(w).reshape(X.shape)
    return w if w.size > 1 else w.item()
        
def w_fr_one_x(x, xP=None, k=None):
    """Return Scharz-Cristoffel w-points from a single point on real axis.
    
    This integrates along the x-axis to x, segment-wise, to avoid
    hitting a singularity (from just after to just before the next singularity)
        
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


def w_fr_zeta(Zeta, xP, k, N=100, endclip=1e-4):
    """
    Return Schwarz-Cristoffel integral at arbitrary Zeta in the upper half plane.

    Parameters
    ----------
    Zeta : array_like of complex
        Points in the z-plane
    xP : array_like of floats
        Prevertices along real axis
    k : array_like of floats
        SC exponents at prevertices
    N : int
        Number of subdivisions for each segment
    endclip : float
        Small offset to avoid landing exactly on a prevertex

    Returns
    -------
    np.ndarray of complex
        SC integral values at Zeta

    Examples
    --------
    >>> import numpy as np
    >>> Z = np.array([0+0j, 1+1j])
    >>> xP = np.array([0.5])
    >>> k = np.array([0.5])
    >>> w = w_fr_zeta(Z, xP, k, N=10)
    >>> isinstance(w, np.ndarray)
    True
    >>> w.shape == Z.shape
    True
    >>> np.all(np.iscomplex(w))
    True
    >>> w[0]  # integration starts at 0
    0j
    """
    w  = []
    for zeta in Zeta.ravel():
        z1 = 2j       
        # --- Vertical path from 0 to z1
        zf = subdiv(0, z1, xP, N=N)        
        zf_arg = sc_arg(zf, xP, k)
        If = integrate_trapz_complex(zf, zf_arg)

        # --- Horizontal path from z1 to z1 + zeta.real
        zh = subdiv(zf[-1], zf[-1] + zeta.real, xP, N=N)
        zh_arg = sc_arg(zh, xP, k)
        Ih = integrate_trapz_complex(zh, zh_arg)

        # --- Vertical path from zh[-1]
        zv = zh[-1].real + 1j * erf_grid(zh[-1].imag, zeta.imag, N=N, dense_side='both')
        zv_arg = sc_arg(zv, xP, k)
        Iv = integrate_trapz_complex(zv, zv_arg)
        
        w.append(If + Ih + Iv)
    return np.array(w).reshape(Zeta.shape)


def z_fr_w(w, wA, wB, zA, zB):
    """Return the affine map sending wA→zA and wB→zB, applied to w.

    This computes the unique linear map z = p*w + q such that:
    - z(wA) = zA
    - z(wB) = zB

    Parameters
    ----------
    w : scalar or array-like
        Points in the w-plane to map.
    wA, wB : complex or float
        Two distinct points in the w-plane.
    zA, zB : complex
        Corresponding points in the final z-plane.

    Returns
    -------
    complex or ndarray of complex
        The mapped points.
    """
    f = AffineMap(wA, wB ,zA, zB)
    return f(w)
    

def sc_segment(i, xP, k):
    """"Return integrate between xP[i] and xP[i+1], gets segment in z plane before rotation stretching and translation.
   
    Parameters
    ----------
    i: int 
        segment number
    xP: floats (real)
        real prevertices
    k: floats (real)
        neg exponents SC argument ...(x - xi) ** (-ki) ...
    """
    a, b= xP[i], xP[i+1]
    s = erf_grid(a, b, dense_side='both')
    arg = sc_arg(s, xP=xP, k=k)
    return integrate_trapz_complex(s, arg)


# --- Objective function for least-squares ---
def objective(u, k, lengths):
    """Return u values that make the reatio of edge-lengths in the ransform match the real-world ratios.

    Parameters
    ----------
    u: iterable of floats same size as k
        log of the postive addition to parameters (prevertices) to make sure
        the successive scale factors keep their order. (xP[i] = xP[i - 1] + np.exp(ui))
        See unpack(u)
    k: floats (real)
        neg exponents SC argument ...(x - xi) ** (-ki) ...
    length iterable of floats
        length of the edges of the shape to be tranformed so that
        the ration between those lengths can be maintained by
        optimization.
    
    Returns vector of computed length rations and desired rations
    """
    lengths = np.asarray(lengths)
    
    # unpack u into χ-points: here we fix xB = 1.0, solve for xC, xD, xE
    xP = unpack_u(u)
    
    # compute segment lengths
    computed = []
    for i in range(len(lengths)):
        L = sc_segment(i, xP, k)
        computed.append(np.abs(L)) # The length
    
    computed = np.array(computed)
    
    computed_ratios = computed[1:] / computed[0]
    desired_ratios  = lengths[1:]  / lengths[0]
    
    print("computed_ratios ", computed_ratios)
    print("desired_ratio s  ", desired_ratios)
    return computed_ratios - desired_ratios

def test_sc_mapping(case=1):
    """Show the Scharz-Christoffel transformation for a set of shapes.
    
    Parameter
    ---------
    case: int
        One of the cases
    """
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
    elif case == 5:
        title = "Arbitrary shape"
        xP = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        k = 1 / np.array([np.inf, -4, 2, -4, 4, 2, 2, -4, 4, -2, 4, np.inf])
 
    # Do the transformation
    w = w_fr_zeta(xP + 0j, xP, k)
    #w = w_fr_x(xP, xP, k)

    fig, ax = plt.subplots()
    ax.plot(w.real, w.imag, '.-')

    ax.set_title(title)
    for ip, wi in enumerate(w):
        ax.text(wi.real, wi.imag, f"{ip}", ha='left', va='bottom')
    ax.set_aspect(1)
    
    
def rectangular_ditch():
    """Compute and show the stream and potential lines for a ditch cross section.
    
    The symmetrial half ditch cross-section has zero-head a vertical side
    and a zero-head horizontal bottom through which water enters the ditch.
    
    This problem is sovled using the sine and Schwarz-Cristoffel conformal
    transformation. But first the transformation is done from the Omega plane
    to the zeta0-upper half plane to the zeta-upper half plane in which two
    ditch-edges landed on two pervertices. From here the Schwarz-Cristoffel 
    integral takes us to the w-plane in which the shape of the ditch is
    corect but not its location and size and perhaps rotation. The last step, 
    finalle, takes us from the w-plane to the real-world z-plane using a
    linear transfromation that matches the corners of the ditch in the zeta-plane
    to those in the z-plane.
    """
    
    # --- The cross section is defined by these lenghts
    lengths = np.array([2, 1,  1, 1, 2]) # b, c, d, e
    angs    = np.array([180, 90, -90, 90, 90]) * np.pi / 180
    
    zd = 0 + 0j
    zDitch = np.array([zd])
    theta = 0.
    for le, ang in zip(lengths, angs):
        theta += ang        
        zd1 = zd + le * np.exp(1j * theta)        
        zDitch = np.hstack((zDitch, zd1))
        zd = zd1
    zDitch = np.array(zDitch)
    
    zDitch = zDitch - zDitch[-2].real - 1j * zDitch[1].imag # F.real=0, B.imag=0
        
    fig, ax = plt.subplots()
    ax.plot(zDitch.real, zDitch.imag)
    ax.set_aspect(1)
    
    # --- Given angles / k_i ---
    vecs = np.diff(zDitch)
    k = np.angle(vecs[1:] / vecs[:-1]) / np.pi
    
    # --- initial prevertices   
    xP = np.arange(len(k)) + 1.
    
    
    # --- Segment lengths used by the objective function which mathes segment-length ratios
    segment_lengths = np.abs(vecs[1:-1])

    # --- Initial guess ---
    u0 = np.log(np.ones(len(xP) - 2))  # rough starting guesses for xC-xB, xD-xC, xE-xD
    
    # --- Solve with least squares ---
    res = least_squares(objective, u0,
                xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=200, method='lm', args=[k, segment_lengths])
    print("Optimization success:", res.success, res.message)
    print(res)

    # --- the optimized points
    xP = unpack_u(res.x)
    print("xp:", xP)
    
    # --- Plot zeta, w and Z
    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    ax0, ax1, ax2, ax3 = axs.flatten()
    
    fig.suptitle("Schwarz-Cristoffel half-ditch X-section")

    ax0.set(title="Omega-plane", xlabel='Phi', ylabel='Psi')    
    ax1.set(title="zeta", xlabel='xi', ylabel='ypsilon')
    ax2.set(title="w-plane", xlabel='u', ylabel='v')
    ax3.set(title='z-plane', xlabel='x', ylabel='y')
    
    # --- Plot the xP points in the zeta plane
    ax1.plot(xP.real, xP.imag, 'bo-', label='points xP')

    # --- Compute the images w(xP) and show them in the w-plane
    wP = w_fr_x(xP, xP, k)
    ax2.plot(wP.real, wP.imag, 'bo-', label='xP --> w')
    
    # --- Plot these points together with the original ditch corner points
    # --- Tell which corners to match (wp and zditch)
    zP = z_fr_w(wP, wP[0], wP[2], zDitch[1], zDitch[3])
    ax3.plot(zP.real, zP.imag, '-', label='zP (back transformed)')
    ax3.plot(zDitch.real, zDitch.imag, '-', ms=12, mfc='none', label='original points')
    
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
    phi = np.linspace(0, 2, 41).clip(eps, None) * Q
    Phi, Psi = np.meshgrid(phi, psi)
    Omega = (Phi + 1j * Psi) 
    
    zeta0 = zeta0_fr_omega(Omega, Q)
    zeta  = zeta_fr_zeta0(zeta0, xP[0], xP[2])
    w     = w_fr_zeta(zeta, xP=xP, k=k, N=50, endclip=1e-3)
    Z     = z_fr_w(w, wP[0], wP[2], zDitch[1], zDitch[3])
    
    ax0.plot(Omega.real,   Omega.imag,   'b', lw=0.35)
    ax0.plot(Omega.real.T, Omega.imag.T, 'g', lw=0.35)    
    ax1.plot(zeta.real,   zeta.imag,   'b', lw=0.35)
    ax1.plot(zeta.real.T, zeta.imag.T, 'g', lw=0.35)
    ax2.plot(w.real,   w.imag,   'b', lw=0.35)
    ax2.plot(w.real.T, w.imag.T, 'g', lw=0.35)
    ax3.plot(Z.real,   Z.imag,   'b', lw=0.35)
    ax3.plot(Z.real.T, Z.imag.T, 'g', lw=0.35 )
    
    for ax in axs.ravel():
        ax.grid(True)
        ax.set_aspect(1)
        #ax.legend(loc='best') # Pictures too small for legend.
        
    # --- Save picture
    images = '/Users/Theo/GRWMODELS/python/tools/analytic/images'
    fig.savefig(os.path.join(images, 'SC_ditch_1.png'))
        
    return
    
def some_shape(case_title='Ditch',
               fig_name='SC_ditch_Xsec',
               lengths=None,
               angles=None,
               ip1=0,
               ip2=2):
    """Compute and show the stream and potential lines for a ditch cross section.
    
    The symmetrial half ditch cross-section has zero-head a vertical side
    and a zero-head horizontal bottom through which water enters the ditch.
    
    This problem is sovled using the sine and Schwarz-Cristoffel conformal
    transformation. But first the transformation is done from the Omega plane
    to the zeta0-upper half plane to the zeta-upper half plane in which two
    ditch-edges landed on two pervertices. From here the Schwarz-Cristoffel 
    integral takes us to the w-plane in which the shape of the ditch is
    corect but not its location and size and perhaps rotation. The last step, 
    finalle, takes us from the w-plane to the real-world z-plane using a
    linear transfromation that matches the corners of the ditch in the zeta-plane
    to those in the z-plane.
    
    Paramters
    ---------
    Lengths: ndarray of floats
        lengths of the segments (edges) defining the cross section.
        First and last is part of the line from and to infinity (needed for direction)
    angs: ndarray of floats (same size as lengths)
        Angles in degrees between current and next segment (edge). First should be 180.
        Follow the cross section in anti-clockwise direction (inside cross section is to the left).
        Anti-clockwise angles are postiive.
    i1, i2: ints
        indices of the points that correspond to the start and end of the fixed head (ditch bottom).
        Start counting with 0 at the second point (because the first point will be dropped).
    """

    # --- The cross section is defined by these lenghts and angles

    if (lengths is None) or (angles is None):
        # --- lengths of  segments starting on the line from infinity before the firs prevertex
        #     continuing with the line towards infinity (firs and last needed for the direction)
        lengths = np.array([2, 1,  1, 1, 2, 5, 5]) # a, b, c, d, e
    
        # --- Angles between successive edges. The first is always 180 (horizontal)
        #     Try out different angles for the first and last to see that happens great to see!
        angles    = np.array([180, 45, -45, -45, 45, 90, 90]) * np.pi / 180
    else:
        assert len(lengths) == len(angles), "lengths and angles must be of same size"
        lengths = np.asarray(lengths)
        angles  = np.asarray(angles)
        
    
    # --- Compute the ditch points (abrupt bending points + first and last)
    zd = 0 + 0j
    zDitch = np.array([zd])
    theta = 0.
    for le, ang in zip(lengths, angles):
        theta += ang        
        zd1 = zd + le * np.exp(1j * theta)        
        zDitch = np.hstack((zDitch, zd1))
        zd = zd1
    zDitch = np.array(zDitch)
    
    # --- Tell which points to translate the image to desired coordinates
    zDitch = zDitch - zDitch[-2].real - 1j * zDitch[1].imag # F.real=0, B.imag=0
        
    # --- Show the ditch and its points in the w plane for verification
    fig, ax = plt.subplots()
    ax.set_title("Show figure in original and w-plane")
    ax.plot(zDitch.real, zDitch.imag, 'b.-', label='ditch in z-plane')
    ax.set_aspect(1)
    
    # --- Get the neg. exponents (angles between edges / pi)
    vecs = np.diff(zDitch)
    k = np.angle(vecs[1:] / vecs[:-1]) / np.pi
    
    # --- initial prevertices   
    xP = np.arange(len(k)) + 1.
    
    # --- transform to w (just to check)
    wP = w_fr_zeta(xP, xP, k)
    
    ax.plot(wP.real, wP.imag, 'ro-', label='ditch in w-plane')
    ax.set_aspect(1)
    ax.legend()

    # --- Here we can throw away the first and the last point of zDitch
    # --- because they were only help point which should actually lie at infinity
    zDitch = zDitch[1:-1]

    # --- Segment lengths used by the objective function which mathes segment-length ratios
    segment_lengths = np.abs(zDitch[1:] - zDitch[:-1])    
 
    # --- Initial guess ---
    u0 = np.log(np.ones(len(xP) - 2))  # rough starting guesses for xC-xB, xD-xC, xE-xD
    
    # --- Solve with least squares ---
    res = least_squares(objective, u0,
                xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=200, method='trf', args=[k, segment_lengths])
    print("Optimization success:", res.success, res.message)
    print(res)

    # --- the optimized points
    xP = unpack_u(res.x)
    print("xp:", xP)
    
    # --- Plot zeta, w and Z
    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    ax0, ax1, ax2, ax3 = axs.flatten()
    
    fig.suptitle(f"Schwarz-Cristoffel, X-section: {case_title}")

    ax0.set(title="Omega-plane", xlabel='Phi', ylabel='Psi')    
    ax1.set(title="zeta", xlabel='xi', ylabel='ypsilon')
    ax2.set(title="w-plane", xlabel='u', ylabel='v')
    ax3.set(title='z-plane', xlabel='x', ylabel='y')
    
    # --- Plot the xP points in the zeta plane
    ax1.plot(xP.real, xP.imag, 'b.-', label='points xP')

    # --- Compute the images w(xP) and show them in the w-plane
    wP = w_fr_x(xP, xP, k)
    ax2.plot(wP.real, wP.imag, 'b.-', label='xP --> w')
    
    # --- Plot these points together with the original ditch corner points
    zP = z_fr_w(wP, wP[0], wP[1], zDitch[0], zDitch[1])
    ax3.plot(zP.real, zP.imag, 'b-', lw=0.35, label='zP (back transformed)')
    # ax3.plot(zDitch.real, zDitch.imag, 'go', ms=12, mfc='none', label='original points')
                
    # --- From Omega to Z
    eps = 1e-3
    Q = 1.0
    psi = np.linspace(1, 0, 21).clip(eps, 1 - eps) * Q # highest psi on top for convenience
    phi = np.linspace(0, 2, 41).clip(eps, None) * Q
    
    Phi, Psi = np.meshgrid(phi, psi)
    Omega = (Phi + 1j * Psi)
    zeta0 = zeta0_fr_omega(Omega, Q)
    
    # tell func to which points -1 and 1 will map
    zeta  = zeta_fr_zeta0(zeta0, xP[ip1], xP[ip2])
    
    w     = w_fr_zeta(zeta, xP=xP, k=k, N=100, endclip=1e-3)
    Z     = z_fr_w(w, wP[0], wP[1], zDitch[0], zDitch[1])

    # Plot Phi and Psi in the different planes
    ax0.plot(Omega.real,   Omega.imag,   'b', lw=0.35)
    ax0.plot(Omega.real.T, Omega.imag.T, 'g', lw=0.35)
    
    ax1.plot(zeta0.real,   zeta0.imag,   'b', lw=0.35)
    ax1.plot(zeta0.real.T, zeta0.imag.T, 'g', lw=0.35)
    
    ax2.plot(w.real,   w.imag,   'b', lw=0.35)
    ax2.plot(w.real.T, w.imag.T, 'g', lw=0.35)
    
    ax3.plot(Z.real,   Z.imag,   'b', lw=0.35)
    ax3.plot(Z.real.T, Z.imag.T, 'g', lw=0.35 )
    
    # --- Finish figures
    for ax in axs.ravel():
        ax.grid(True)
        ax.set_aspect(1)
        #ax.legend(loc='best') # Pictures too small for legend.
        
    # --- Save picture
    # --- Replace this if you are in another directory of on another computer
    images = '/Users/Theo/GRWMODELS/python/tools/analytic/images'
    fig.savefig(os.path.join(images, fig_name))
        
    return
    
def get_line(P, vecs):
    """Return a line as an array of complex numbers.
    
    Parameters
    ----------
    P: complex number
        Start of line
    vecs: iteratble of 2-tuples (len, angle)
        Segments of the  line    
    """
    theta = 0.
    line = [P]    
    for l, ang in vecs:
        theta += ang * np.pi / 180
        P1 = P + l * np.exp(1j * theta)
        line.append(P1)
        P = P1
    return np.array(line)

def mirror_line(line):
    """Return line horizontalle swapped around x=0.
    
    Parameters
    ----------
    line: np.array of complex numbers
        The line to be mirrored
    """
    return -line.real + 1j * line.imag
        
def get_puppet():
    """Return a dict with coordinates of a puppet used for demo"""
    cheek, temple, hat = 1/ 6 * np.sqrt(3), 1/6, 1/4
    up_arm, low_arm, up_leg, low_leg, foot = .8, .8, 1, 1, 1/3
    body, neck = 0.9, 1/8

    throat, forehead = 0 + 0j, 0 + 1j * (neck + cheek / 2 + temple)
    
    puppet = {
        'head': {
            'left':
                get_line(throat,
                    ((neck, 90), (cheek, -60), (temple, 60), (2 * cheek / np.sqrt(3), 90))),
        },
        'hat': {
            'left':
                get_line(forehead,
                    ((hat, 0), (hat, 0), (hat, 180),
                     (hat, -90), (hat, 90))),                     
        },
        'arm': { 
            'left':
                get_line(throat,
                     ((up_arm, -45), (low_arm, +90))),
        },
        'leg': { 
            'left':
                get_line(throat,
                    ((body, -90), (up_leg, 50), (low_leg, -90), (foot, 90))),
        },
    }
    
    for part in puppet.keys():
        puppet[part]['right'] = mirror_line(puppet[part]['left'])
    return puppet
    
    
case = {0: {'case_title': "Rectangular ditch",
            'fig_name': "SC_rect_ditch.png",
            'lengths': np.array([2, 1,  1, 1, 2]),
            'angles' : np.array([180, 90, -90, 90, 90]) * np.pi / 180,
            'ip1': 0,
            'ip2': 2
            },
        1: {'case_title': "Ditch with polygon profile",
            'fig_name': "SC_pgon_ditch.png",
            'lengths': np.array([2, 1,  1, 1, 1, 5]),
            'angles': np.array([180, 70, -25, -25, 70, 90]) * np.pi / 180,
            'ip1': 0,
            'ip2': 3,
            },
        2: {'case_title': "Ditch cuts in top of aquifer",
            'fig_name': "SC_cutin_ditch.png",
            'lengths': np.array([2, 1,  1, 1, 2, 5, 5]),
            'angles': np.array([180, 45, -45, -45, 45, 90, 90]) * np.pi / 180,
            'ip1': 0,
            'ip2': 3,
            }
        }

def show_puppet(throat=None, navel=None, ax=None):
    """Show the puppet matching the new points"""
    puppet = get_puppet()
    
    throat_puppet = puppet['head']['left'][0]
    navel_puppet  = puppet['leg']['left'][1]
    
    if throat is not None and navel is not None:
        f = AffineMap(throat_puppet, navel_puppet, throat, navel)

    if ax is None:
        fig, ax = plt.subplots()
    
    for part in puppet:
        left  = puppet[part]['left']
        right = puppet[part]['right']
        
        if throat is not None and navel is not None:
            left  = f(left)
            right = f(right)
        
        ax.plot(left.real, left.imag)
        ax.plot(right.real, right.imag)
        
    ax.set_aspect(1)
    return ax


if __name__ == '__main__':
    
    if False:
        for case in [0, 1, 2, 3, 4, 5]:
            test_sc_mapping(case=case)
    if False:
        rectangular_ditch()
    if False:        
        some_shape(**case[0])
        some_shape(**case[1])
        some_shape(**case[2])

    if True:
        # --- show puppet and its tranform using get_pq(..)
        puppet = get_puppet()
        ax = show_puppet()
        show_puppet(3 + 2j, 4 + 0j, ax=ax)
        ax.set_title('Puppet and transform using get_pq()')

    plt.show()


