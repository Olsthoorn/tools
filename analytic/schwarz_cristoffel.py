import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import least_squares

# --------------------------------------------------------
# Helper functions (blijven netjes buiten de class)
# --------------------------------------------------------
def unpack_u(u, x0=1.0, x1=2.0):
    """Return the new parameters after during and after optimation in a robust way.
    
    Make sure that during optimisations subsequent points are always increasing
    
    Paramters
    ---------
    u: floats (u0, u1)
        The parameters passed to the objective function.
    """

    d = np.log1p(np.exp(u))
    xP = np.empty(len(u) + 2)
    xP[0]  = x0
    xP[1]  = x1
    xP[2:] = x1 + np.cumsum(d)
    return xP


def erf_grid(a, b, N=25, W=2.5, dense_side='both'):
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


def subdiv(z1, z2, xP, N=50, endclip=1e-4, show=False):
    """Return subdivided line AB.
    
    Subdivides line AB such that de density of the points is iversily
    proportional to the distance towards the prevertices xP. This
    prevents taking too large steps near these singularities when
    carrying a complex integration along path AB.
    
    Parameters
    ----------
    zi, z2: complex numbers
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
    alpha = np.linspace(0, 1, N).clip(0, 1 - endclip)  # dense pre-grid
    
    # fysieke punten
    Z = z1 + alpha * (z2 - z1)
    
    # distance-based weight
    eps = 0
    p = 0.5
    w = np.min(np.abs(Z[:, None] - np.array(xP)[None, :])**p, axis=1) + eps
    
    # integrate 1/w
    s = np.cumsum(1/w)
    s /= s[-1]                       # normalize to [0,1]

    # invert s(α): given s_k → α_k
    
    alpha_new = np.interp(np.linspace(0, 1, N+1), s, alpha)

    # final points
    Z_new = z1 + alpha_new * (z2 - z1)
    # ds = np.abs(np.diff(Z_new))

    if show:
        fig, ax = plt.subplots()
        ax.plot(xP, np.zeros_like(xP), 'o')
        ax.plot(Z_new.real, Z_new.imag, '.')
        plt.show()

    return Z_new # , ds


def sc_arg(z, xP, k):
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
    f_total = np.ones_like(z, dtype=complex)

    for xi, ki in zip(xP, k):
        dz = z - xi

        r = np.abs(dz)
        ang = np.angle(dz)
        ang_corr = np.unwrap(ang)

        log_dz = np.log(r) + 1j*ang_corr
        term = np.exp(-ki * log_dz)

        f_total *= term

    return f_total


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

def w_fr_x(X, xP=None, k=None, N=100, W=2.5):
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
    N, W:
        passed on to w_fr_one_x(... N=N, W=W)
        number of subpoints
    W: float

    """
    X = np.asarray(X)
    w = []
    for x in X:
        w.append(w_fr_one_x(x, xP=xP, k=k, N=N, W=W))        
    w = np.array(w).reshape(X.shape)
    return w if w.size > 1 else w.item()
        
def w_fr_one_x(x, xP=None, k=None, N=25, W=2.5):
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
        
    for xa, xb, ka, kb in zip(xP_ext[:-1], xP_ext[1:], k[:-1], k[1:] ):
        if x < 0:
            a, b = 0, x       
            s = erf_grid(a, b, N=N, W=W,dense_side='left')
            argc = sc_arg(s, xP=xP, k=k)
            return integrate_trapz_complex(s, argc)
        if x <= xa:
            break
            
        a, b = xa, min(xb, x)
        if np.isclose(a, b):
            return w
        if a == xP[-1]:
            s = erf_grid(a, b, N=N, W=W, dense_side='left')
        else:
            s = erf_grid(a, b, N=N, W=W, dense_side='both')
        argc = sc_arg(s, xP=xP, k=k)
        w += integrate_trapz_complex(s, argc)
    return w


def w_fr_zeta(Zeta, xP, k, N=100, W=2.5, endclip=1e-4):
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
    >>> w = w_fr_zeta(Z, xP, k, N=10, W=2.5, endlcip=1e-4)
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
        # --- Vertical path from 0 to z1
        zf = subdiv(0, 2j, xP, N=N,  endclip=0)  
        # zf = 1j * erf_grid(0, 2, N=N, W=W, dense_side='left')
      
        zf_arg = sc_arg(zf, xP, k)
        If = integrate_trapz_complex(zf, zf_arg)

        # --- Horizontal path from z1 to z1 + zeta.real
        zh = subdiv(zf[-1], zf[-1] + zeta.real, xP, N=N, endclip=0)
        zh_arg = sc_arg(zh, xP, k)
        Ih = integrate_trapz_complex(zh, zh_arg)

        # --- Vertical path from zh[-1]
        zv = subdiv(zh[-1], zeta, xP, N=N, endclip=endclip)
        # zv = zh[-1].real + 1j * erf_grid(zh[-1].imag, zeta.imag, N=N,
        #                                  W=W, dense_side='both')
        zv_arg = sc_arg(zv, xP, k)
        Iv = integrate_trapz_complex(zv, zv_arg)
        
        w.append(If + Ih + Iv)
    return np.array(w).reshape(Zeta.shape)


def sc_segment(i, xP=None, k=None, N=25, W=2.5):
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
    s = erf_grid(a, b, N=N, W=W, dense_side='both')
    arg = sc_arg(s, xP=xP, k=k)
    return integrate_trapz_complex(s, arg)


# --- Objective function for least-squares ---
def objective(u, k, lengths, N, W):
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
        L = sc_segment(i, xP=xP, k=k, N=N, W=W)
        computed.append(np.abs(L)) # The length
    
    computed = np.array(computed)
    
    # Solve for best-fit positive scale s* (one scalar!)
    #s = np.dot(computed, lengths) / np.dot(lengths, lengths)  # LS-schaal

    # Return residuals
    
    print("computed / desired: ", np.round(computed / lengths, 4))
    print("xP :", xP)
    
    #return computed - s * lengths
    return (computed[1:] / computed[0]) / (lengths[1:] / lengths[0]) - 1

# --------------------------------------------------------
# AffineMap class (zoals jij al had)
# --------------------------------------------------------
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
    
    
class SC_Section:
    """
    Complete Schwarz–Christoffel pipeline for a ditch-like cross section:
    Ω-plane (phi+i psi)  →  ζ-plane (via complex sine)
                         →  w-plane (SC integral)
                         →  z-plane (affine map)
    """

    # ----------------------------------------------------------
    # 1. Orde & configuratie
    # ----------------------------------------------------------
    def __init__(self, lengths, angles, ip1=0, ip2=1, N=100, W=2.5, endclip=1e-4):
        """
        Store geometric parameters of the ditch, and settings
        for the SC mapping.
        
        Picture of cross section with 5 xP and 4 edges
        
        ~~~~~ 0 <---- from infinity (implied)
              |          Cross section with xP indices (0-4)
              1          Only the lengths of the edges between
             /           the xP points are specified (4).
        3---2            Angles at xP here are [90, -45, -45, 90, 90]
        |                First and last angle assume horizontal edge to infinity
        4 ---------> to infinity (implied)
        
        Parameters
        ----------
        lengths: np.iterable of floats
            The successive lengths of the edges between the vertices defining
            the X-section. leaving out the first and last from and to infinity.
        angles: np.iterable of floats
            Angles in degrees between successive edges at every prevertex.
            For the first angle it is assumed that the previous not specified
            (implied) edge is horizontal coming from infinity.
            The last angle should bend the last specified edge towards 
            infinity, which normally is a horizontal line to the right.
        ip1: int
            The ditch as the set of consequtive edges where the head is
            prescribed and zero.
            ip1 is the index of the prevertex of the left side of the ditch.
        ip2: int
            Prevertex index of the left sizde of the ditch.
        W: float
            erf(W) determines how dense points are near ends of segment
        endclip: float
            Small value to prevent that subdiv lands on singularity

        Returns
        -------
        Instantiated X-section object

        Examples:
        >>> xsec = SC_section(lengths, edges)
        >>> xsec.plot()
        """
        self.W = W
        self.ip1 = ip1
        self.ip2 = ip2
        
        self.zDitch = self.xsec_geometry(lengths, angles)
            
        # --- Compute prevertices self.xP and points self.wP       
        self.solve_prevertices(N=N, W=W, endclip=endclip)

        # ---Affine mapping is later computed
        self.affine = None

    def xsec_geometry(self, lengths, angles, show=False):
        """Compute and store xsec geometry.
        
        Picture of cross section with 5 xP and 4 edges
            
        ~~~~~ 0 <---- from infinity (implied)
                |          Cross section with xP indices (0-4)
                1          Only the lengths of the edges between
                /           the xP points are specified (4).
        3---2            Angles at xP here are [90, -45, -45, 90, 90]
        |                First and last angle assume horizontal edge to infinity
        4 ---------> to infinity (implied)
        
        Parameters
        ----------
        lengths: np.iterable of floats
            The successive lengths of the edges between the vertices defining
            the X-section. leaving out the first and last from and to infinity.
        angles: np.iterable of floats
            Angles in degrees between successive edges at every prevertex.
            For the first angle it is assumed that the previous not specified
            (implied) edge is horizontal coming from infinity.
            The last angle should bend the last specified edge towards 
            infinity, which normally is a horizontal line to the right.
                
        Returns
        -------
        Instantiated X-section object

        Examples:
        >>> xsec = SC_section(lengths, edges)
        >>> xsec.plot()
        """
        if lengths is None or angles is None:
            # --- Rectangular ditch
            lengths = np.array([1,  1, 1]) # b, c, d, e
            angles  = np.array([90, -90, 90, 90])
        
        lengths = np.asarray(lengths)
        angles  = np.asarray(angles) * np.pi / 180.
            
        assert len(lengths) == len(angles) - 1, (
            "Number of segment lengths must be one less than number of angles."
        )
        
        # --- Store them
        self.lengths = lengths
        self.angles = angles
        
        # --- Get Schwarz-Cristoffel exponents 
        self.k = self.angles / np.pi

        # --- Add the implied edges
        all_edges = np.hstack((lengths, 1))
        
        # --- Compute the the cross section vertices
        zp0 = 0 + 0j
        zP = np.array([zp0])
        theta = np.pi        
        for le, ang in zip(all_edges, angles):
            theta += ang        
            zp1 = zp0 + le * np.exp(1j * theta)        
            zP = np.hstack((zP, zp1))
            zp0 = zp1
            
        # --- Drop the implied vertices
        zP = np.array(zP)[:-1]
        
        # --- Align cross section with left-most and top-most zP
        zP = zP - zP.real.min() - 1j * zP.imag.max()
            
        if show:
            _, ax = plt.subplots()
            ax.plot(zP.real, zP.imag)
            ax.set_aspect(1)
        
        self.zP = zP
        self.zDitch = zP # For convenience


    # ----------------------------------------------------------
    # 2. Ω-plane: phi + i psi → rooster
    # ----------------------------------------------------------
    def get_omega_grid(self, phi_range, psi_range, Nphi, Npsi):
        """Return Omega_grid.
        
        Parameters
        ----------
        phi_range: two floats
            The min and max value of Phi
        psi_range: two floats
            The min and max valueo of Psi
        
        """
        phi = np.linspace(*phi_range, Nphi)
        psi = np.linspace(*psi_range, Npsi)
        Phi, Psi = np.meshgrid(phi, psi)
        Omega = Phi + 1j*Psi
        return Omega

    # ----------------------------------------------------------
    # 3. ζ-plane via complex sine
    # ----------------------------------------------------------
    def zeta0_fr_omega(self, Omega):
        """Return zeta0"""
        
        psi_max, psi_min = Omega.imag.max(), Omega.imag.min()
        psi_diff = psi_max - psi_min
        psi_avg = 0.5 * (psi_max + psi_min)
                
        return np.sin(1j * np.pi * (Omega - 1j * psi_avg) / psi_diff)
    
    
    def zeta_fr_zeta0(self, zeta0):
        """Linearly map zeta_0 to zeta xP0 -> zeta0=-1, xP1-> zeta0=+1 for arcsin.
        
        In zeta0-plane the points -1 and 1 form two prevertices (of choice) this transoformation
        moves -1 and 1 in the zeta0 plane to xp0 and xP1 in the zeta_plane.    
        """
        f = AffineMap(-1, 1, self.xP[self.ip1], self.xP[self.ip2])
        return f(zeta0)

    
    # ----------------------------------------------------------
    # 4. Bepalen prevertices xP via least squares
    # ----------------------------------------------------------
    def solve_prevertices(self, N=100, W=2.5, endclip=1e-4):
        """Solve prevertices and store in self.xP.
                
        """
        # --- Initial guess ---
        u0 = np.log(np.ones(len(self.angles) - 2))  # rough starting guesses
    
        # --- Solve with least squares ---
        res = least_squares(objective, u0,
                    xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=200, method='dogbox', args=[self.k, self.lengths, N, W])
        print("Optimization success:", res.success, res.message)
        print(res)

        # --- the optimized points
        xP = unpack_u(res.x)
        self.xP = xP
        
        # --- Comopute image points in the w-plane accurately
        self.wP = w_fr_x(self.xP, xP=self.xP, k=self.k, N=N, W=W)

    # ----------------------------------------------------------
    # 5. Kern: SC-integral w(ζ)
    # (jouw 3-segment detour integrator)
    # ----------------------------------------------------------
    def w_fr_x(self, x, N=None, W=2.5):
        return w_fr_x(x, xP=self.xP, k=self.k, N=N, W=W)
    
    def w_fr_zeta(self, Zeta, N=25, W=2.5, endclip=1e-4):
        return w_fr_zeta(Zeta, self.xP, self.k, N=N, W=W, endclip=endclip)

    # ----------------------------------------------------------
    # 6. Affine mapping w → z
    # ----------------------------------------------------------
    def z_fr_w(self, w):
        """Transform w-plane to z-plane (real world).
        
        Parameters
        ----------
        w: ndarray of complex
            the w-plane values
        """
        self.affine = AffineMap(*self.wP[:2], *self.zP[:2])
        return self.affine(w)

    # ----------------------------------------------------------
    # 7. Volledige pipeline Ω → ζ → w → z
    # ----------------------------------------------------------
    def compute_full_map(self, Omega, N=100, W=2.5, endclip=1e-4):
        zeta0 = self.zeta0_fr_omega(Omega)
        zeta  = self.zeta_fr_zeta0(zeta0)
        w     = self.w_fr_zeta(zeta, N=100, W=W, endclip=endclip)
        z     = self.z_fr_w(w)
        return z


def test_sc_mapping(case_nr=1):
    """Show the Scharz-Christoffel transformation for a set of shapes.
    
    Parameter
    ---------
    case: int
        One of the cases
    """
    # --- Sets of combinations of points along x and corners (k = alpha/pi values)

    endclip = 1e-5

    cases = {
        0: {
            'title': "triangle",
            'lengths': [1, 1, 1] ,
            'angles': [-60, 120, 120, 120],
            'N': 200,
            'W': 4.5,
            'endclip': endclip,
        },
        1: {
            'title': r"Four left corners of $\pi/2$",
            'lengths': [1, 1, 1, 1],
            'angles': [-90, 90, 90, 90, 0],
            'N': 200,
            'W': 4.5,
            'endclip': endclip,
        }  ,      
        2: {
            'title': "Pentagon",
            'lengths': [1, 2, 3, 4, 5],
            'angles' : [-72, 72, 72, 72, 72, 72],
            'N': 200,
            'W': 4.5,
            'endclip': endclip,
        },     
        3: {
            'title': "No way to tell where the points will land",
            'lengths': [1, 1, 1, 1, 1, 1],
            'angles': [-60, 60, 60, 60, 60, 60, 60],
            'N': 200,
            'W': 4.5,
            'endclip': endclip,
        },
        4: {
            'title': "Arbitrary shape",
            'lengths': [1, 1, 1, 1, 1, 1, 1, 1],
            'angles': [180, -45, 90, -45, 120, 90, -60, 30, 0],
            'N': 200,
            'W': 4.5,
            'endclip': endclip,
        },
    }
 
    # Do the transformation
    case = cases[case_nr]
    
    xsec = SC_Section(case['lengths'], case['angles'], N=case['N'], W=case['W'], endclip=case['endclip'])
    
    wP2 = xsec.w_fr_zeta(xsec.xP, N=case['N'], endclip=case['endclip'])
    wP3 = xsec.w_fr_x(   xsec.xP, N=case['N'], W=case['W'])
    
    fig, ax = plt.subplots()
    wP1 = xsec.wP
    ax.plot(wP1.real, wP1.imag, 'b.-', label='wP1')
    ax.plot(wP2.real, wP2.imag, 'r.-', label='wP2')
    ax.plot(wP3.real, wP3.imag, 'go-', mfc='none', label='wP3')

    ax.set_title(case['title'])
    for ip, wi in enumerate(xsec.wP):
        ax.text(wi.real, wi.imag, f"{ip}", ha='left', va='bottom')
    ax.set_aspect(1)
    ax.legend()
    
    # Compare final lengths
    lengths_Z = np.abs(np.diff(xsec.zP))
    lengths_W = np.abs(np.diff(xsec.wP))
    print(f"rel_len_zP = {lengths_Z/lengths_Z[0]}")
    print(f"rel_len_wP = {lengths_W/lengths_W[0]}")
          

def some_shape(case_title='Ditch',
               fig_name='SC_ditch_Xsec',
               lengths=None,
               angles=None,
               ip1=0,
               ip2=2,
               W=4.5,
               N=200,
               endclip=1e-5):
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
    xsec = SC_Section(lengths, angles, ip1=ip1, ip2=ip2, N=N, W=W, endclip=endclip)
        
    # --- Show the ditch and its points in the w plane for verification
    if False:
        fig, ax = plt.subplots()
        ax.set_title("Show figure in original and w-plane")
        ax.plot(xsec.zP.real, xsec.zP.imag, 'b.-', label='ditch in z-plane')
        ax.set_aspect(1)
    
        ax.plot(xsec.wP.real, xsec.wP.imag, 'ro-', label='ditch in w-plane')
        ax.set_aspect(1)
        ax.legend()
 
    # --- the optimized points
    print("xp:", xsec.xP)
    
    # --- Plot zeta, w and Z
    fig, axs = plt.subplots(2, 2, figsize=(12, 7))
    ax0, ax1, ax2, ax3 = axs.flatten()
    
    fig.suptitle(f"Schwarz-Cristoffel, X-section: {case_title}")

    ax0.set(title="Omega-plane", xlabel='Phi', ylabel='Psi')    
    ax1.set(title="zeta", xlabel='xi', ylabel='ypsilon')
    ax2.set(title="w-plane", xlabel='u', ylabel='v')
    ax3.set(title='z-plane', xlabel='x', ylabel='y')
    
    # --- Plot the xP points in the zeta plane
    ax1.plot(xsec.xP.real, xsec.xP.imag, 'b.-', label='points xP')

    # --- Compute the images w(xP) and show them in the w-plane
    ax2.plot(xsec.wP.real, xsec.wP.imag, 'b.-', label='xP --> w')
    
    # --- Plot these points together with the original ditch corner points
    zP = xsec.z_fr_w(xsec.wP)
    ax3.plot(zP.real, zP.imag, 'b-', lw=0.35, label='zP (back transformed)')
    # ax3.plot(xsec.zP.real, xsec.zP.imag, 'go', ms=12, mfc='none', label='original points')
                
    # --- From Omega to Z
    Q = 1.
    Omega = xsec.get_omega_grid((0, 2 * Q), (0, Q), Nphi=41, Npsi=21)
    zeta0 = xsec.zeta0_fr_omega(Omega)
    
    # tell func to which points -1 and 1 will map
    zeta  = xsec.zeta_fr_zeta0(zeta0)
    
    w     = xsec.w_fr_zeta(zeta, N=N, W=W, endclip=endclip)
    Z     = xsec.z_fr_w(w)

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

case = {0: {'case_title': "Rectangular ditch",
            'fig_name': "SC_rect_ditch.png",
            'lengths': [1,  1, 1],
            'angles' : [90, -90, 90, 90],
            'ip1': 0,
            'ip2': 2,
            'N': 200,
            'W': 4.5,
            'endclip': 1e-5,
            },
        1: {'case_title': "Ditch with polygon profile",
            'fig_name': "SC_pgon_ditch.png",
            'lengths': [1,  1, 1, 1],
            'angles': [70, -25, -25, 70, 90],
            'ip1': 0,
            'ip2': 3,
            'N': 200,
            'W': 4.5,
            'endclip': 1e-5,
            },
        2: {'case_title': "Ditch cuts in top of aquifer",
            'fig_name': "SC_cutin_ditch.png",
            'lengths': [1,  1, 1, 2, 5],
            'angles': [45, -45, -45, 45, 90, 90],
            'ip1': 0,
            'ip2': 3,
            'N':200,
            'W': 4.5,
            'endclip': 1e-5,
            },
        3: {'case_title': "Flow through a throat",
            'fig_name': "SC_throat1.png",
            'lengths': [1,  1, 1, 1, 1],
            'angles': [45, -45, 90, 90, -45, 45],
            'ip1': 2,
            'ip2': 3,            
            'N': 200,
            'W': 4.5,
            'endclip': 1e-5,
            },
        4: {'case_title': "Flow through another throat",
            'fig_name': "SC_throat2.png",
            'lengths': [1,  1, 1, 1, 3, 1, 1, 1, 1],
            'angles': [45, -45, -45, 45, 90, 90, 45, -45, -45, 45],
            'ip1': 4,
            'ip2': 5,            
            'N': 200,
            'W': 4.5,
            'endclip': 1e-5,
            }

        }

if __name__ == '__main__':
    
    if False:
        test_sc_mapping(case_nr=0)
        test_sc_mapping(case_nr=1)
        test_sc_mapping(case_nr=2)
        test_sc_mapping(case_nr=3)
        test_sc_mapping(case_nr=4)
    
    if True:        
        some_shape(**case[0])
        some_shape(**case[1])
        some_shape(**case[2])
        some_shape(**case[3])
        some_shape(**case[4])

    plt.show()

