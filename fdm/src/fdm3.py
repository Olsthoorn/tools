# -*- coding: utf-8 -*-
# [markdown]
# # Steady state FD simulation
#
"""
Steady-state 3D Finite Difference Models as a function.
Stream line computation (Psi) and a function
Computaion of veclocity vector for plotting with quiver

Created on Fri Sep 30 04:26:57 2016

@author: Theo

"""
import warnings
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as la
import scipy.special as spec
import matplotlib.pylab as plt
import etc
from fdm.src import mfgrid

def quivdata(Out, x, y, iz=0):
    """Returns vector data for plotting velocity vectors.

    Takes Qx from fdm3 and returns the tuple X, Y, U, V containing
    the velocity vectors in the xy plane at the center of the cells
    of the chosen layer for plotting them with matplotlib.pyplot's quiver()

    Parameters
    ----------
    `Qx` : ndarray
        field in named tuple returned by fdm3.
    `x` : ndarray
        grid line coordinates.
    `y` : ndarray
        grid line coordinates.
    `iz` : int
        layer number for which vectors are computed (default 0)

    Returns:
        tuple: X, Y, U,V

        X : ndaarray
            2D ndArray of x-coordinates cell centers
        Y : ndarray
            2D ndarray of y-coordinate of cell centers
        U : ndarray
            2D ndarray of x component of cell flow [L3/T]
        V : ndarray
            2D ndarray of y component of cell flow [L3/T]

    """
    Ny = len(y)-1
    Nx = len(x)-1
    xm = 0.5 * (x[:-1] + x[1:])
    ym = 0.5 * (y[:-1] + y[1:])

    X, Y = np.meshgrid(xm, ym) # coordinates of cell centers

    # Flows at cell centers
    U = np.concatenate((Out.Qx[:,0,iz].reshape((Ny,1,1)), \
                        0.5 * (Out.Qx[:,:-1,iz].reshape((Ny,Nx-2,1)) +\
                               Out.Qx[:,1:,iz].reshape((Ny,Nx-2,1))), \
                        Out.Qx[:,-1,iz].reshape((Ny,1,1))), axis=1).reshape((Ny,Nx))
    V = np.concatenate((Out.Qy[0,:,iz].reshape((1,Nx,1)), \
                        0.5 * (Out.Qy[:-1,:,iz].reshape((Ny-2,Nx,1)) +\
                               Out.Qy[1:,:,iz].reshape((Ny-2,Nx,1))), \
                        Out.Qy[-1,:,iz].reshape((1,Nx,1))), axis=0).reshape((Ny,Nx))
    return X, Y, U, V


def psi(Qx, row=0):
    """Returns stream function values in z-x plane for a given grid row.

    The values are at the cell corners in an array of shape [Nz+1, Nx-1].
    The stream function can be vertically contoured using gr.Zp and gr.Xp as
    coordinates, where gr is an instance of the Grid class.

    Arguments:
    Qx --- the flow along the x-axis at the cell faces, excluding the outer
           two plains. Qx is one of the fields in the named tuple returned
           by fdm3.
    row --- The row of the cross section (default 0).

    It is assumed:
       1) that there is no flow perpendicular to that row
       2) that there is no storage within the cross section
       3) and no flow enters the model from below.
    The stream function is computed by integrating the facial flows
    from bottom to the top of the model.
    The outer grid lines, i.e. x[0] and x[-1] are excluded, as they are not in Qx
    The stream function will be zero along the bottom of the cross section.

    """
    Psi = Qx[:, row, :] # Copy the section for which the stream line is to be computed.
                        # and transpose to get the [z,x] orientation in 2D
    Psi = Psi[::-1].cumsum(axis=0)[::-1]         # cumsum from the bottom
    Psi = np.vstack((Psi, np.zeros(Psi[0,:].shape))) # add a row of zeros at the bottom
    return Psi


def fdm3(gr=None, K=None, c=None, FQ=None, HI=None, IBOUND=None, GHB=None, axial=False):
    '''Compute a 3D steady state finite diff. model

    Parameters
    ----------
    gr: mfgrid.Grid object instance
        object holding the grid/mesh (see mfgrid.Grid)
    K: np.ndarray of floats (nlay, nrow, nocl) or a 3-tuple of such array
        if 3-tuple then the 2 np.ndarrays are kx, ky and kz
            kx  --- array of cell conductivities along x-axis (Ny, Nx, Nz)
            ky  --- same for y direction (if None, then ky=kx )
            kz  --- same for z direction
        else: kx = ky = kz = K
    c: np.ndarray (nlay - 1, nrow, ncol) or None of not used
        Resistance agains vertical flow between the layers [d]
    GHB: array (ncell, 3)
        dtype = dtype([('cellid', 'O'), ('h', float), ('cond', float)])
        list of sequence, cellid may also be the global cell index
        [[(k,j ,i), h, cond],
         [...]]
    FQ: np.ndarray of floats (nlay, nrow, ncol)
        Prescrived cell flows (injection positive)
    IH: np.ndarray (nlay, nrow, ncol)
        Initial heads
    IBOUND: np.ndarray of ints (nlay, nrow, ncol)
        the boundary array like in MODFLOW
        with values denoting:
        * IBOUND>0  the head in the corresponding cells will be computed
        * IBOUND=0  cells are inactive, will be given value NaN
        * IBOUND<0  coresponding cells have prescribed head
        
    Returns
    -------
    out: dict
        a dict with fields Phi, Qx, Qy, Qz and cell flow Q
        Output shapes are
        (Ny,Nx,Nz) (Ny,Nx-1,Nz), (Ny-1,Nx,Nz), (Ny,Nx,Nz-1), (Ny, Nx, Nz)

    TO 160905
    '''
    # notice that Our is a class. It is instantiated in the return below
    Out = dict()

    Nz, Ny, Nx = SHP = gr.shape
    Nod = Ny * Nx * Nz
    NOD = np.arange(Nod).reshape(SHP) # generate cell numbers

    if gr.axial==True:
        print("axial==True so that y coordinates and ky are ignored")
        print("            and x stands for r, so that all x coordinates must be >= 0.")
    if isinstance(K, np.ndarray): # only one ndaray was given
        kx, ky, kz = K.copy(), K.copy(), K.copy()
    elif isinstance(K, tuple): # 3-tuple of ndarrays was given
        kx, ky, kz = K[0].copy(), K[1].copy(), K[2].copy()
    else:
        raise ValueError("", "K must be an narray of shape (Ny,Nx,Nz) or a 3tuple of ndarrays")

    if kx.shape != SHP:
        raise AssertionError("shape of kx {0} differs from that of model {1}".format(kx.shape, SHP))
    if ky.shape != SHP:
        raise AssertionError("shape of ky {0} differs from that of model {1}".format(ky.shape, SHP))
    if kz.shape != SHP:
        raise AssertionError("shape of kz {0} differs from that of model {1}".format(kz.shape, SHP))

    # from this we have the width of columns, rows and layers
    dx = gr.dx.reshape(1, 1, Nx)
    dy = gr.dy.reshape(1, Ny, 1)
    dz = gr.dz.reshape(Nz, 1, 1)

    active = (IBOUND>0 ).reshape(Nod,)  # boolean vector denoting the active cells
    inact  = (IBOUND==0).reshape(Nod,) # boolean vector denoting inacive cells
    fxhd   = (IBOUND<0 ).reshape(Nod,)  # boolean vector denoting fixed-head cells

    if gr.axial==False:
        Rx2 = 0.5 * dx / (dy * dz) / kx
        Rx1 = 0.5 * dx / (dy * dz) / kx
        Ry  = 0.5 * dy / (dz * dx) / ky
        Rz  = 0.5 * dz / (dx * dy) / kz
        if c is not None:
            Rc  =        c / (dx * dy)
        #half cell resistances regular grid
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning) # Division by zero for x=0
            Rx2 = 1 / (2 * np.pi * kx[:,:, 1: ] * dz) * np.log(gr.xm[ 1:]/gr.x[1:-1]).reshape((1, 1, Nx-1))
            Rx1 = 1 / (2 * np.pi * kx[:,:, :-1] * dz) * np.log(gr.x[1:-1]/gr.xm[:-1]).reshape((1, 1, Nx-1))
        Rx2 = np.concatenate((np.inf * np.ones((Nz, Ny, 1)), Rx2), axis=2)
        Rx1 = np.concatenate((Rx1, np.inf * np.ones((Nz, Ny, 1))), axis=2)
        Ry = np.inf * np.ones(SHP)
        Rz = 0.5 * dz.reshape((Nz, 1, 1))  / (np.pi * (gr.x[1:]**2 - gr.x[:-1]**2).reshape((1, 1, Nx)) * kz)
        if c is not None:
            Rc = c  / (np.pi * (gr.x[1:]**2 - gr.x[:-1]**2).reshape((1, 1, Nx)))
        #half cell resistances with grid interpreted as axially symmetric

    # set flow resistance in inactive cells to infinite
    Rx2 = Rx2.reshape(Nod,); Rx2[inact] = np.inf; Rx2=Rx2.reshape(SHP)
    Rx1 = Rx1.reshape(Nod,); Rx1[inact] = np.inf; Rx1=Rx1.reshape(SHP)
    Ry  = Ry.reshape( Nod,); Ry[ inact] = np.inf; Ry=Ry.reshape(SHP)
    Rz  = Rz.reshape( Nod,); Rz[ inact] = np.inf; Rz=Rz.reshape(SHP)
    #Grid resistances between nodes

    Cx = 1 / (Rx1[:, :,:-1] + Rx2[:, :,1:])
    Cy = 1 / (Ry[:, :-1, :] + Ry[:, 1:, :])
    if c is None:
        Cz = 1 / (Rz[:-1, :, :] + Rz[:-1, :,:])
    else:
        Cz = 1 / (Rz[:-1, :, :] + Rc + Rz[:-1, :,:])
        
    # General head bounaries
    if GHB is not None:
        Cghb = gr.const(0.)
        Hghb = gr.const(0.)
        I = GHB['I']
        Cghb.ravel()[I] = GHB['C'] 
        Hghb.ravel()[I] = GHB['h']
        
    #conductances between adjacent cells

    IE = NOD[:, :, 1: ]  # east neighbor cell numbers
    IW = NOD[:, :, :-1] # west neighbor cell numbers
    IN = NOD[:, :-1, :] # north neighbor cell numbers
    IS = NOD[:,  1:, :]  # south neighbor cell numbers
    IT = NOD[:-1, :, :] # top neighbor cell numbers
    IB = NOD[ 1:, :, :]  # bottom neighbor cell numbers
    #cell numbers for neighboors

    def R(x): # shorthand for x.ravel()
        return x.ravel()
    
    # notice the call  csc_matrix( (data, (rowind, coind) ), (M,N))  tuple within tupple
    # also notice that Cij = negative but that Cii will be postive, namely -sum(Cij)
    A = sp.csc_matrix((
            -np.concatenate(( R(Cx), R(Cx), R(Cy), R(Cy), R(Cz), R(Cz)) ),\
            (np.concatenate(( R(IE), R(IW), R(IN), R(IS), R(IB), R(IT)) ),\
             np.concatenate(( R(IW), R(IE), R(IS), R(IN), R(IT), R(IB)) ),\
                      )),(Nod,Nod))

    # to use the vector of diagonal values in a call of sp.diags() we need to have it aa a
    # standard nondimensional numpy vector.
    # To get this:
    # - first turn the matrix obtained by A.sum(axis=1) into a np.array by np.array( .. )
    # - then take the whole column to loose the array orientation (to get a dimensionless numpy vector)
    adiag = np.array(-A.sum(axis=1))[:,0]

    A += sp.diags(adiag)
    # diagonal matrix, a[i,i]

    RHS = FQ.reshape(Nod,1) - A[:,fxhd].dot(HI.reshape(Nod,1)[fxhd])
    
    # Right-hand side vector.

    Phi = HI.flatten()
    # Allocate space to store heads.

    if GHB is not None:
        Cghb_diag = sp.diags(R(Cghb), offsets=0, shape=A.shape)
        Phi[active] = la.spsolve( (A + Cghb_diag)[active][:,active],
                                 (RHS + R(Cghb * Hghb)[:, np.newaxis])[active] )
    else:
        Phi[active] = la.spsolve( A[active][:,active] ,RHS[active] )
        
    # Solve heads at active locations.

    # net cell inflow
    if GHB is not None:
        Q  = (A + Cghb_diag).dot(Phi).reshape(gr.shape)
    else:
        Q  = A.dot(Phi).reshape(gr.shape)

    # reshape Phi to shape of grid
    Phi = Phi.reshape(gr.shape)

    #Flows across cell faces
    Qx =  -np.diff(Phi, axis=2) * Cx
    Qy =  +np.diff(Phi, axis=1) * Cy
    Qz =  +np.diff(Phi, axis=0) * Cz
    
    out=dict()
    out.update(Phi=Phi, Q=Q, Qx=Qx, Qy=Qy, Qz=Qz)
    
    if GHB is not None:
        Qghb = (Hghb - Phi) * Cghb
        out.update(Qghb=Qghb)

        # set inactive cells to NaN
    out['Phi'][inact.reshape(gr.shape)] = np.nan # put NaN at inactive locations

    return out

# Examples that take the function of tests

def mazure(kw):
    """1D flow in semi-confined aquifer example
    Mazure was Dutch professor in the 1930s, concerned with leakage from
    polders that were pumped dry. His situation is a cross section perpendicular
    to the dike of a regional aquifer covered by a semi-confining layer with
    a maintained head in it. The head in the regional aquifer at the dike was
    given as well. The head obeys the following analytical expression
    phi(x) - hp = (phi(0)-hp) * exp(-x/B), B = sqrt(kDc)
    To compute we use 2 model layers and define the values such that we obtain
    the Mazure result.
    """
    z = kw['z0'] - np.cumsum(np.hstack((0., kw['D'])))
    gr = mfgrid.Grid(kw['x'], kw['y'], z, axial=False)
    c = gr.const(kw['c'])
    k = gr.const(kw['k'])
    kD = (kw['k'] * kw['D'])[-1] # m2/d, transmissivity of regional aquifer
    B = np.sqrt(kD * float(kw['c'])) # spreading length of semi-confined aquifer
    
    FQ = gr.const(0.) # prescribed flows
    
    s0 = 2.0 # head in aquifer at x=0
    HI = gr.const(0); HI[-1, :, 0] = s0 # prescribed heads
    
    IBOUND = gr.const(1); IBOUND[0, :, :] = -1; IBOUND[-1, :, 0]=-1
    
    out = fdm3(gr=gr, K=k, FQ=FQ, HI=HI, c=c, IBOUND=IBOUND)
    
    ax = etc.newfig(kw['title'], 'x[m]', 's [m]')
    
    ax.plot(gr.xm, out['Phi'][-1, 0 ,:], 'r.', label='fdm3')
    ax.plot(gr.xm, s0 * np.exp(-gr.xm / B),'b-', label='analytic')
    ax.legend()
    return out


def deGlee(kw):
    """Simulate steady axial flow to fully penetrating well in semi-confined aquifer"""
    z = kw['z0'] - np.cumsum(np.hstack((0., kw['D'])))
    r = np.hstack((0., kw['rw'], kw['r'][kw['r'] > kw['rw']]))
    gr = mfgrid.Grid(r, None, z, axial=True)
    k = gr.const(kw['k'])
    c = gr.const(kw['c'])
    kD = (kw['k'] * kw['D'])[-1]      # m2/d, transmissivity of regional aquifer
    B  = np.sqrt(kD * float(kw['c'])) # spreading length of regional aquifer
    
    FQ = gr.const(0.); FQ[-1, 0, 0] = kw['Q']   # m3/d fixed flows
    HI = gr.const(0.)                           # m, initial heads
    
    IBOUND = gr.const(1); IBOUND[0, :, :] = -1  # modflow like boundary array
    
    out = fdm3(gr=gr, K=k, FQ=FQ, HI=HI, c=c, IBOUND=IBOUND) # run model
    
    ax = etc.newfig(kw['title'], 'r [m]', 's [m]', xscale='log', xlim=[1e-3, r[-1]])
    
    ax.plot(gr.xm, out['Phi'][-1, 0, :], 'ro', label='fdm3')
    ax.plot(gr.xm, kw['Q']/(2 * np.pi * kD) * spec.k0(gr.xm / B) / (kw['rw']/ B * spec.k1(kw['rw']/ B)), 'b-',label='analytic')
    ax.legend()
    return out
    
    
def deGlee_GHB(kw):
    """Run Axial symmetric example, as before, but now using GHB instead of an extra layer on top.
    """
    kD = (kw['k'] * kw['D'])[-1]      # m2/d, transmissivity of regional aquifer
    B  = np.sqrt(kD * float(kw['c'])) # spreading length of regional aquifer
    
    r = np.hstack((0., kw['rw'], kw['r'][kw['r'] > kw['rw']]))

    z = (kw['z0'] - np.cumsum(np.hstack((0., kw['D']))))[1:] # no top layer
    
    gr = mfgrid.Grid(r, None, z, axial=True)   # generate grid
    
    FQ = gr.const(0.); FQ[-1, 0, 0] = kw['Q']     # m3/d fixed flows
    HI = gr.const(0.)                       # m, initial heads
    
    IBOUND = gr.const(1, dtype=int)         # modflow like boundary array
    
    k = gr.const(kw['k'][-1])                         # full 3D array of conductivities
    
    cells = np.zeros(gr.shape, dtype=bool); cells.ravel()[gr.NOD[0]] = True
    hds = HI[cells]
    C   = gr.Area.ravel() / float(kw['c'])
    GHB = gr.GHB(cells, hds, C)
    
    out = fdm3(gr=gr, K=k, FQ=FQ, HI=HI, IBOUND=IBOUND, GHB=GHB) # run model
    
    ax = etc.newfig(kw['title'] + ' (Using GHB)', 'r [m]', 's [m]', xscale='log', xlim=[1e-3, r[-1]])
    
    ax.plot(gr.xm, out['Phi'][-1, 0, :], 'ro', label='fdm3')
    ax.plot(gr.xm, kw['Q']/(2 * np.pi * kD) * spec.k0(gr.xm / B) / (kw['rw']/ B * spec.k1(kw['rw']/ B)), 'b-',label='analytic')
    ax.legend()
    return out

cases = {
    'Mazure': {
        'title': 'Mazure 1D flow',
        'comment': """1D flow in semi-confined aquifer example
        Mazure was Dutch professor in the 1930s, concerned with leakage from
        polders that were pumped dry. His situation is a cross section perpendicular
        to the dike of a regional aquifer covered by a semi-confining layer with
        a maintained head in it. The head in the regional aquifer at the dike was
        given as well. The head obeys the following analytical expression
        phi(x) - hp = (phi(0)-hp) * exp(-x/B), B = sqrt(kDc)
        To compute we use 2 model layers and define the values such that we obtain
        the Mazure result.
        """,
        'z0': 0.,
        'x': np.hstack((0.001, np.linspace(0., 2000., 101))), # column coordinates
        'y': np.array([-0.5, 0.5]), # m, model is 1 m thick
        'D': np.array([10., 50.]), # m, thickness of confining top layer
        'c': np.array([[250.]]), # d, vertical resistance of semi-confining layer
        'k': np.array([np.inf, 10.]),
        },
    'DeGlee': {
        'title': 'DeGlee axially symmetric flow',
        'comment': """Axially symmetric example, well in semi-confined aquifer (De Glee case)
            De Glee was a Dutch engineer/groundwater hydrologist and later the
            first director of the water company of the province of Groningen.
            His PhD (1930) solved the axial symmetric steady state flow to a well
            in a semi confined aquifer using the Besselfunctions of the second kind,
            known as K0 and K1.
            The example computes the heads in the regional aquifer below a semi confining
            layer with a fixed head above. It uses two model layers a confining one in
            which the heads are fixed and a semi-confined aquifer with a prescribed
            extraction at r=rw. If rw>>0, both K0 and K1 Bessel functions are needed.
            The grid is signaled to use inteprete the grid as axially symmetric.
            """,
        'z0': 0.,
        'Q': -1200.,
        'rw':   .25,
        'D' : np.array([10., 50.]),
        'c' :  np.array([[250.]]),
        'k' :  np.array([np.inf, 10]),  # m/d conductivity of regional aquifer
        'r' : np.logspace(-2, 4, 61),  # distance to well center
    }
}  

if __name__ == "__main__":
    out1 = mazure(cases['Mazure'])
    out2 = deGlee(cases['DeGlee'])
    out2 = deGlee_GHB(cases['DeGlee'])
    print('Done')
    
    plt.show()
