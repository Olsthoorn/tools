# Phreatic 3D transient groundwater flow
# TO 2023-03-20

# TODO Still need to deal with vertical conductance under water table conditions 2023-03-24
import os
import sys

if not (tools_dir:='/Users/Theo/GRWMODELS/python/tools/') in sys.path:
    sys.path.insert(0, tools_dir)

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve # to use its short name
from scipy.special import erf
from shapely.geometry import Point
import warnings
from fdm import Grid

import geopandas as gpd

inside = lambda xlim, zlim, X, Z: \
    np.logical_and(X > xlim.min(),
        np.logical_and(X < xlim.max(),
            np.logical_and(Z < zlim.max(),Z > zlim.min())
        )
    )
    
def fillLayer(gr, shapes=None, field=None, dtype=float, fillValue=0):
    """Fill a vertical cross section with the parameter in field and the geometry shapes.
    
    it uses the filed "id" as the index in that order if id exists,
    otherwise it uses the ragular index
    
    Parameters
    ----------
    gr: fdm.Grid object
        Holds the grid for the model.
    shapes: Geopandas object with polygon geometry.
        The zones with their field/parameter values.
    field: str
        name of the field to use.
    dtype: int or float
        dtype of the output array.
    startvalue
    """
    if not field in shapes.columns:
        raise ValueError("Field {} not in shape.columns: {}".format(field, repr(shapes.columns)))
    
    if "id" in shapes.columns:
        shapes.index = sorted(shapes["id"])
    
    A = gr.const(fillValue, dtype=dtype)
    
    points = [Point(x, y) for x, y in zip(gr.XM.ravel(), gr.ZM.ravel())]
    
    for i in shapes.index:
        shp = shapes.loc[i]
        A[np.array([shp.geometry.contains(p) for p in points]).reshape(gr.shape)] = shp[field]
    return A  

    
def fillIndex(gr, shapes=None, field="id", dtype=int, fillValue=-99):
    """Fill a vertical cross section with the parameter in field and the geometry shapes.
    
    Parameters
    ----------
    gr: fdm.Grid object
        Holds the grid for the model.
    shapes: Geopandas object with polygon geometry.
        The zones with their field/parameter values.
    field: str must be the index starting at 0 and must be continous
        name of the field to use.
    dtype: int must  be int
        dtype of the output array.
    """
    
    if field != "id":
        raise ValueError("field must be 'id'.")
    print("dtype is {} and so the test dtype is int) yields {}.".
                    format(repr(int), dtype is int))
    if dtype is not int:
        raise ValueError("dtype must be int")
    
    return fillLayer(gr, shapes=shapes, field=field, dtype=dtype, fillValue=fillValue)


def setBound(gr=None, A=None, recarr=None):
    """Fill rectangles in given array and pass adapted array on.
    
    Parameters
    ----------
    gr: fdm.Grid object
        The modflow grid.
    IBOUND: npdaary of gr.shape
        Modflow 5 boundary array.
    A: ndarray of gr.shape
        The FQ or FH array to fill the data into
    reacarr: recarray with dtype
        [('x1': float), ('x2', float), ('z1', float), ('z2', float), ('v', float or int)]
        specifies rectangle for whicht to fill in FQ or Head.
        
    Usage:
    ------
    >>> dtype = np.dtype([('x1', float), ('x2', float), ('z1', float), ('z2', float), ('v', int)])
    >>> x = np.linspace(0, 20, 21)
    >>> z = [0., -5., -10., -20]
    >>> gr = fdm.Grid(x, [-0.5, 0.5], z, axial=False)
    >>> recarr = np.array([( 4.3,  7.2, -2., -15., 99),
                           (13.1, 17.0, -9., -18., 88)],
                            dtype=dtype)
    >>> A = gr.const(0., dtype=int)
    >>> A = setBound(gr=gr, A=A, recarr=recarr)
    >>> print(A)
    """
    for rec in recarr:
        x1, x2, z1, z2, v = rec
        if x2 < x1:
            x1, x2 = x2, x1
        if z1 > z2:
            z1, z2 = z2, z1        
        I = inside(np.array([x1, x2]), np.array([z1, z2]), gr.XM, gr.ZM)
        if np.sum(I) == 0:
            warnings.warn('No cells found for boundary condition {}.'.format(repr(rec)))
        else:
            A[I] = v
    return A

def boundPrepare(gr=None, A=None, bdictrec=None):
    """Return a dict with key as times and items as array.

    This dict is used to geneate inputObjects for the model
    
    Parameters
    ----------
    gr: fdm.Grid object
        the model mesh
    A: ndarray of gr.shape
        Array to fill in the boundary condition values
    bdictrec: dict with key is time and item is a recarray
        with dtype [('x1': float), ('x2': float), ('z1': float), ('z2': float), ('v': int or float)]
        where each line defines a rectangle in which to fill in the value v.
        There is no limit ot the number of rectangles to fill for each array (each time)
    """
    out = dict()
    print(bdictrec)
    for time, recarr in bdictrec.items():
        A1 = A.copy()
        out[time] = setBound(gr=gr, A=A1, recarr=recarr)
    return out


def DZtol(Z=None, Phi=None, eps=0.01):
    """Return active layer thickness dependent on actual phi.

    Returns the effective cell thickness which will be close to zero for head near or below
    cel bottom and close to cell top for heads near or above cell top and the actual wet
    thickness for heads between top - tol® and bot + eps. This make sure that cells
    will never run completely dry.

    Parameters
    ----------
    Z: np.ndarray (3D array)
        top and bottom of the cells Z[::2] top Z[1::2] bottom
    phi: np.ndarray
        head as a 3D array of cells.
    eps: float
        thicknes of layer close to bottom and top of cell where smoothing of cell thickness starts.
    """
    delta = lambda x: eps * np.exp(x / eps - 1.)
    
    d_top = Z[:-1]- Phi # head below top
    d_bot = Phi - Z[1:] # head above bottom
    
    DZ = np.abs(np.diff(Z, axis=0))
    
    D = Phi - Z[1:] # D is effective thickness of layer
    
    D[d_top < eps] = DZ[d_top < eps] - delta((DZ - D)[d_top < eps])
    D[d_bot < eps] = delta(D[d_bot < eps])
    return D

def Sy_tol(Sy=None, Z=None, Phi=None, sigma=0.01):
    """Return specific yield taking layer wetness into account.

    Function Sy / 2 (1 + erf(x / simga)) is used

    Parameters
    ----------
    Sy: np.ndarray
        specific yield of cells
    ZM: np.ndarray (3D)
        elevation of cell centers
    phi: np.ndarray
            head as a 3D array of cells.
    sigma: float
        standard dev for erfc function determining the steepness around top and bottom of the cell.
    """
    R = lambda x : x.ravel()
    
    S  =Sy.copy()
    S, Phi, ZT, ZM, ZB = R(S), R(Phi), R(Z[:-1]), R(0.5 * (Z[:-1] + Z[1:])), R(Z[1:])
    
    S[Phi > ZM] *= 0.5 * (1 + erf((ZT - Phi)[Phi > ZM] / sigma))
    S[Phi < ZM] *= 0.5 * (1 + erf((Phi - ZB)[Phi < ZM] / sigma))
    return S.reshape(Sy.shape)

class InputObj:
    """Returns an input object to select from during run.
    
    Input objects are essentially like dicts where the key is the time after which the array
    is to be used during the simulation.
    
    Each item is a key and an ndarray of gr.shape and key is the time after which the array is valid.
    
    The first item is used until t reaches its time.
    The last item is used after t passes its time
    The others are interpolated between the time of the previous and the next.
    
    This object is effective in specifying time-variying head and flow input.
    
    If used with IBOUND interpolate must be False, because IBOUND cannot be interpolated.
    But a very similar one can be used of perhaps with an option.
    
    
    @TO 2023-03-28
    """
    
    def __init__(self, o, interpolate=True):
        """Return input object.
        
        Parameters
        ----------
        o: np.ndarray of gr.shape or a dict with such arrays
            input with keys that are times (sim times or real times).
            For instance the array may contains HI or FQ or IBOUND values. The keys are the times after which the array is valid.
        interpolate: bool
            if True interpolate arrays between the times
            Interpolate must be False if used for time-varying IBOUND as IBOUND can change but cannot not be interpolated.
        """
        self.interpolate = interpolate
        
        if isinstance(o, np.ndarray):
            self.o = {0: o}  
        elif isinstance(o, dict):
            self.o = o
        else:
            raise ValueError("Object must be ndarray of gr.shape or dict of them.")
           
        self.times = np.array([t for t in o.keys()], dtype=float)
        self.times.sort()

        
    def __getitem__(self, t):
        """Return the interpolated array for time t (not index it).
        
        Parameters
        ----------
        t: float
            current simulation time.
        """
        if t <= self.times[0]:
            return self.o[self.times[0]]
        elif t >= self.times[-1]:
            return self.o[self.times[-1]]
        else:
            t0 = self.times[np.where(t >= self.times)[0][-1]]
            if not self.interpolate:                            
                return self.o[t0]
            else:
                t1 = self.times[np.where(t < self.times)[0][0]]
                return self.o[t0] + (self.o[t1] - self.o[t0]) * (t - t0) / (t1 - t0)
            
    def __setitem__(self, t, A):
        self.o[t] = A
        self.times = np.array([t for t in self.o.keys()], dtype=float)
        self.times.sort()


def fdm3wtt(gr=None, t=None, kxyz=None, Ss=None, Sy=None,
          FQ=None, HI=None, IBOUND=None, Nouter=100,
          eps=0.01, sigma=0.01, tol=1e-5, epsilon=0.67, verbose=True):
    """Transient 3D Finite Difference Water Table Model returning computed heads and flows.

    Heads and flows are returned as 3D arrays as specified under output parmeters.

    Parameters
    ----------
    'gr' : `grid_object`, generated by gr = Grid(x, y, z, ..)
        if `gr.axial`==True, then the model is run in axially symmetric model
    t : ndarray, shape: [Nt+1]
        times at which the heads and flows are desired including the start time,
        which is usually zero, but can have any value.
    `kx`, `ky`, `kz` : ndarray, shape: (Ny, Nx, Nz), [L/T]
        hydraulic conductivities along the three axes, 3D arrays.
    `Ss` : ndarray, shape: (Ny, Nx, Nz), [L-1]
        specific elastic storage
    `Sy` : ndarray, shape: (Ny, Nx, Nz), [L-1]
        specific yield for cells that become partially dry
    `FQ` : InputObj [L3/d], see class
        prescrived cell flows (injection positive, zero of no inflow/outflow)
    `IH` : InputObj [L], see class
        initial heads. `IH` has the prescribed heads for the cells with prescribed head.
    `IBOUND` : ndarray, shape: (Ny, Nx, Nz) of int
        boundary array like in MODFLOW with values denoting
        * IBOUND>0  the head in the corresponding cells will be computed
        * IBOUND=0  cells are inactive, will be given value NaN
        * IBOUND<0  coresponding cells have prescribed head
    eps: float
        distance from bottom and top of cell from where where D is smoothed
    sigma: float
        distance determining specific yield smoothing when head goes above top or below bottom of cell
    `epsilon` : float, dimension [-]
        degree of implicitness, choose value between 0.5 and 1.0
    'verbose': True or int >=0
        basic printing if verbose True or 1,
        more printing of verbose > 1
        no printing if verbose < 1 or zero.

    outputs
    -------
    `out` : namedtuple containing heads and flows:
        `out.Phi` : ndarray, shape: (Nt+1, Ny, Nx, Nz), [L3/T]
            computed heads. Inactive cells will have NaNs
            To get heads at time t[i], use Out.Phi[i]
            Out.Phi[0] = initial heads
        `out.Q`   : ndarray, shape: (Nt, Ny, Nx, Nz), [L3/T]
            net inflow in all cells during time step, inactive cells have 0
            Q during time step i, use Out.Q[i]
        `out.Qs`  : ndarray, shape: (Nt, Ny, Nx, Nz), [L3/T]
            release from storage during time step.
        `out.Qx   : ndarray, shape: (Nt, Ny, Nx-1, Nz), [L3/T]
            intercell flows in x-direction (parallel to the rows)
        `out.Qy`  : ndarray, shape: (Nt, Ny-1, Nx, Nz), [L3/T]
            intercell flows in y-direction (parallel to the columns)
        `out.Qz`  : ndarray, shape: (Nt, Ny, Nx, Nz-1), [L3/T]
            intercell flows in z-direction (vertially upward postitive)

    TO 161024
    """
    if gr.axial:
        print('Running in axial mode, y-values are ignored.')

    if isinstance(kxyz, (tuple, list)):
        kx, ky, kz = kxyz
    else:
        kx = ky = kz = kxyz

    if kx.shape != gr.shape:
        raise AssertionError("shape of kx {0} differs from that of model {1}".format(kx.shape,gr.shape))
    if ky.shape != gr.shape:
        raise AssertionError("shape of ky {0} differs from that of model {1}".format(ky.shape,gr.shape))
    if kz.shape != gr.shape:
        raise AssertionError("shape of kz {0} differs from that of model {1}".format(kz.shape,gr.shape))
    if Ss.shape != gr.shape:
        raise AssertionError("shape of Ss {0} differs from that of model {1}".format(Ss.shape,gr.shape))

    kx[kx<1e-20] = 1e-50
    ky[ky<1e-20] = 1e-50
    kz[kz<1e-20] = 1e-50

    Nt=len(t)  # for heads, at all times Phi at t[0] = initial head
    Ndt=len(np.diff(t)) # for flows, average within time step

    # reshaping shorthands
    dx = np.reshape(gr.dx, (1, 1, gr.nx))
    dy = np.reshape(gr.dy, (1, gr.ny, 1))

    #Initialize output arrays (= memory allocation)
    Phi = np.zeros((Nt, gr.nod)) # Nt+1 times
    Q   = np.zeros((Ndt  , gr.nod)) # Nt time steps
    Qs  = np.zeros((Ndt  , gr.nod))
    Qx  = np.zeros((Ndt, gr.nz, gr.ny, gr.nx-1))
    Qy  = np.zeros((Ndt, gr.nz, gr.ny-1, gr.nx))
    Qz  = np.zeros((Ndt, gr.nz-1, gr.ny, gr.nx))

    # cell number of neighboring cells
    IW = gr.NOD[:,:,:-1]  # east neighbor cell numbers
    IE = gr.NOD[:,:, 1:] # west neighbor cell numbers
    IN = gr.NOD[:,:-1,:] # north neighbor cell numbers
    IS = gr.NOD[:, 1:,:]  # south neighbor cell numbers
    IT = gr.NOD[:-1,:,:] # top neighbor cell numbers
    IB = gr.NOD[ 1:,:,:]  # bottom neighbor cell numbers
    R = lambda x : x.ravel()  # generate anonymous function R(x) as shorthand for x.ravel()

    steady = np.all(Ss == 0.) and np.all(Sy ==0)

    # reshape input arrays to vectors for use in system equation
    Phi[0, :]= HI[t[0]].flatten()
    for it, dt in enumerate(np.diff(t)):
        ibound = IBOUND[t[it]]
        active = (ibound >0).reshape(gr.nod,)  # boolean vector denoting the active cells
        inact  = (ibound==0).reshape(gr.nod,)  # boolean vector denoting inacive cells
        fxhd   = (ibound <0).reshape(gr.nod,)  # boolean vector denoting fixed-head cells

        if steady and np.all(fxhd == False):
            raise ValueError(
                "Any Ss must be > 0 for transient); some IBOUND < 0 for steady flow")

        Phi[it + 1, :] = HI[t[it]].flatten() if it == 0 else Phi[it, :]
        for iouter in range(Nouter):
            D = DZtol(Z=gr.Z, Phi=Phi[it + 1].reshape(gr.shape), eps=eps)
            # half cell flow resistances
            if not gr.axial:
                RxE = 0.5 *    dx / (dy * D ) / kx
                RxW = RxE
                Ry1 = 0.5 *    dy / (D  * dx) / ky
                Rz1 = 0.5 * gr.DZ / (dx * dy) / kz
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning) # Division  by zero when x==0, is ok --> np.inf as resistance.
                    RxW = 1 / (2 * np.pi * kx * D) * np.log(gr.xm / gr.x[:-1]).reshape((1, 1, gr.nx))
                    RxE = 1 / (2 * np.pi * kx * D) * np.log(gr.x[1:] /  gr.xm).reshape((1, 1, gr.nx))
                Ry1 = np.inf * np.ones(gr.shape)
                Rz1 = 0.5 * gr.DZ / (np.pi * (gr.x[1:]**2 - gr.x[:-1]**2).reshape((1, 1, gr.nx)) * kz)

            # set flow resistance in inactive cells to infinite
            RxE[inact.reshape(gr.shape)] = np.inf
            RxW[inact.reshape(gr.shape)] = np.inf
            Ry1[inact.reshape(gr.shape)] = np.inf
            Ry2 = Ry1
            Rz1[inact.reshape(gr.shape)] = np.inf
            Rz2 = Rz1

            # conductances between adjacent cells
            Cx = 1 / (RxW[: , :,1:] + RxE[:  ,:  ,:-1])
            Cy = 1 / (Ry1[: ,1:, :] + Ry2[:  ,:-1,:  ])
            Cz = 1 / (Rz1[1:, :, :] + Rz2[:-1,:  ,:  ])

            # storage term, variable dt not included
            S = Ss * D + Sy_tol(Sy=Sy, Z=gr.Z, Phi=Phi[it + 1].reshape(gr.shape), sigma=sigma)
            Cs = R(S * gr.Area[np.newaxis, :, :, ] / epsilon)

            # notice the call  csc_matrix( (data, (rowind, coind) ), (M,N))  tuple within tupple
            # also notice that Cij = negative but that Cii will be postive, namely -sum(Cij)
            A = sp.csc_matrix(( np.concatenate(( R(Cx), R(Cx), R(Cy), R(Cy), R(Cz), R(Cz)) ),\
                                (np.concatenate(( R(IE), R(IW), R(IN), R(IS), R(IT), R(IB)) ),\
                                np.concatenate(( R(IW), R(IE), R(IS), R(IN), R(IB), R(IT)) ),\
                            )),(gr.nod,gr.nod))

            A = -A + sp.diags( np.array(A.sum(axis=1)).ravel() ) # Change sign and add diagonal

            # solve heads at active locations at t_i+eps*dt_i

            # this A is not complete !!
            #RHS = FQ[t[it]].ravel() - (A + sp.diags(Cs / dt))[:,fxhd].dot(Phi[it][fxhd]) # Right-hand side vector
            RHS = FQ[t[it]].ravel() - (A + sp.diags(Cs / dt))[:,fxhd].dot(HI[t[it]].flatten()[fxhd]) # Right-hand side vector
            phi = HI[t[it]].flatten()
            phi[active] = spsolve( (A + sp.diags(Cs / dt))[active][:,active],
                                    RHS[active] + Cs[active] / dt * Phi[it][active])
            
            err = np.max(np.abs(phi[active] - Phi[it + 1][active]))
            if verbose > 1:
                print("iout = {:3d}, max_err = {:.6g}".format(iouter, err))
            
            Phi[it + 1, :] = phi
            
            if err  <= tol: 
                if verbose:   
                    print("Done it = {}".format(it))
                break
            if it == Nouter:
                if verbose:
                    print("Outer convergence failed at t[{it}] = {:.2f} d, err={}".format(it, t, err))

            # net cell inflow
            
        Q[it]  = A.dot(Phi[it + 1])
        Qs[it] = -Cs/dt * (Phi[it + 1]-Phi[it])
                           
        #Flows across cell faces
        Qx[it] =  -np.diff( Phi[it + 1].reshape(gr.shape), axis=2) * Cx
        Qy[it] =  +np.diff( Phi[it + 1].reshape(gr.shape), axis=1) * Cy
        Qz[it] =  +np.diff( Phi[it + 1].reshape(gr.shape), axis=0) * Cz
        
        # update head to end of time step
        Phi[it + 1][active] = Phi[it][active] + (Phi[it + 1]-Phi[it])[active]/epsilon
        Phi[it + 1][fxhd]   = Phi[it][fxhd]
        Phi[it + 1][inact]  = np.nan
    if verbose:
        print("Ready all time steps done!")

    # reshape Phi to shape of grid
    Phi = Phi.reshape((Nt,) + gr.shape)
    Q   = Q.reshape( (Ndt,) + gr.shape)
    Qs  = Qs.reshape((Ndt,) + gr.shape)

    return {'t': t, 'Phi': Phi, 'Q': Q, 'Qs': Qs, 'Qx': Qx, 'Qy': Qy, 'Qz': Qz}

if __name__ == '__main__':
    
    x = np.hstack((0, np.logspace(0, 4., 101)))
    y = [-0.5, 0.5]
    z = [5., 0., -20.]
    
    gr = Grid(x, y, z, axial=True)
    
    k1, k2 = 500., 1.
    ss1, ss2 = 1e-5, 1e-5
    sy1, sy2 = 2.5e-1, 1e-1
    
    k  = gr.const([k1, k2])
    Ss = gr.const([ss1, ss2])
    Sy = gr.const([sy1, sy2])
    IBOUND = gr.const(1, dtype=int); IBOUND[:, :, [-1]] = -1
    IBOUND = InputObj({0.: IBOUND})
    
    kxyz = (k, k, k)
    Q = 60 * 24
    
    # Use InputObj to specify input changes at any time
    HI = {0: gr.const(0.0), 1: gr.const(0.)}
    HI = InputObj(HI)
    
    # Use InputObj to specify input changes at any time
    FQ = {9.: gr.const(0.), 10.: gr.const(0.)} # First until t=9.0, second after t=10. interpolate int the middle
    FQ[10.][0, :, 0] = Q
    FQ = InputObj(FQ)
    
    t = np.linspace(0, 75, 51)
    
    out = fdm3wtt(gr=gr, t=t, kxyz=kxyz, Ss=Ss, Sy=Sy,
          FQ=FQ, HI=HI, IBOUND=IBOUND, tol=0.0001, eps=0.01, sigma=0.01, epsilon=1.0)

    fig_size = (10, 6)
    
    fig, ax = plt.subplots()
    fig.set_size_inches(fig_size)
    ax.set_title("h at diff.  t, top blue, bottom green. Q={:.0f} m3/d, k={:.0f} m/d, Sy={:.2f}".format(Q, k1, sy1))
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_xlim(0, 400)
    ax.set_ylim(-0.5, 2)
    #gr.plot_grid(ax=ax, row=0, color='k', lw=0.5)
    ax.grid()
    for it, phi in enumerate(out['Phi']):
        if it % 5:
            continue  
        for iL in range(gr.nz):
            clr = 'b' if iL % 2 == 0 else 'g'
            ax.plot(gr.xm, phi[iL, 0, :], clr, label='layer {}, t={:.2f} d'.format(iL, t[it]))
    # ax.legend()
    
    # Check the storage
    Stored = np.zeros((2, len(t)))
    for it, phi in enumerate(out['Phi']):
        Stored[0][it] = ((out['Phi'][it][0] - out['Phi'][0][0]) * Sy[0] * gr.Area).sum()
        Stored[1][it] = t[it] * Q
    
    fig, ax = plt.subplots()   
    fig.set_size_inches(fig_size) 
    ax.set_title("Stored Volume, Q={:.0f} m3/d, k={:.0f} m/d, Sy={:.2f}".format(Q, k1, sy1))
    ax.set_ylabel(' V [m3]')
    ax.set_xlabel('t [d]')
    
    ax.plot(t, Stored[1], label='from Q')
    ax.plot(t, Stored[0], '.', label='Model')
    ax.grid()
    ax.legend()
    
    xdesired1 =   5  # 280 # m
    xdesired2 =  80 # m
    ix1 = np.sum(gr.xm <= xdesired1)
    ix2 = np.sum(gr.xm <= xdesired2)
    fig, ax = plt.subplots()   
    fig.set_size_inches(fig_size) 
    ax.set_title("Head for Q={:.0f} m3/d, k={:.0f} m/d, Sy={:.2f}".format(Q, k1, sy1))
    ax.set_ylabel(' head')
    ax.set_xlabel('t [d]')
    ax.plot(t, out['Phi'][:, 0, 0, ix1], label="h at ix={} -> r={:.0f}".format(ix1, gr.xm[ix1]))
    ax.plot(t, out['Phi'][:, 0, 0, ix2], label="h at ix={} -> r={:.0f}".format(ix2, gr.xm[ix2]))
    ax.grid()
    ax.legend()

    plt.show()
    
    print('Ready')
