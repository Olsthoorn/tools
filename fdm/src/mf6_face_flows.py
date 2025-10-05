# Derived from inspect.getsource(flopy.mf6.utils.postprocessing.get_structured_faceflows)

import numpy as np
import flopy.mf6.utils.binarygrid_util as fpb

# Original code from flopy 3.3.4 ===================================
# from .binarygrid_util import MfGrdFile
# This is very slow when multiple sets of fxf have to be generated, i.e.
# For every stress period. The slowness if due to recomputing the
# pointers every time.
# Below this original is my fast version. It computes the pointers
# only once and then uses them to pick the right flows (frf, fff, flf)
# directly from the flowja
# TO 231115

def get_structured_faceflows(
    flowja,
    grb_file=None,
    ia=None,
    ja=None,
    nlay=None,
    nrow=None,
    ncol=None,
    verbose=False,
    asdict=False,
    ):
    """
    Get the face flows for the flow right face, flow front face, and
    flow lower face from the MODFLOW 6 flowja flows. This method can
    be useful for building face flow arrays for MT3DMS, MT3D-USGS, and
    RT3D. This method only works for a structured MODFLOW 6 model.

    Parameters
    ----------
    flowja : ndarray
        flowja array for a structured MODFLOW 6 model
    grbfile : str
        MODFLOW 6 binary grid file path
    ia : list or ndarray
        CRS row pointers. Only required if grb_file is not provided.
    ja : list or ndarray
        CRS column pointers. Only required if grb_file is not provided.
    nlay : int
        number of layers in the grid. Only required if grb_file is not provided.
    nrow : int
        number of rows in the grid. Only required if grb_file is not provided.
    ncol : int
        number of columns in the grid. Only required if grb_file is not provided.
    verbose: bool
        Write information to standard output
    asdict: bool
        return faceflows as a dict.

    Returns
    -------
    frf : ndarray
        right face flows
    fff : ndarray
        front face flows
    flf : ndarray
        lower face flows

    """
    if grb_file is not None:
        grb = fpb.MfGrdFile(grb_file, verbose=verbose)
        if grb.grid_type != "DIS":
            raise ValueError(
                "get_structured_faceflows method "
                "is only for structured DIS grids"
            )
        ia, ja = grb.ia, grb.ja
        nlay, nrow, ncol = grb.nlay, grb.nrow, grb.ncol
    else:
        if (
            ia is None
            or ja is None
            or nlay is None
            or nrow is None
            or ncol is None
        ):
            raise ValueError(
                "ia, ja, nlay, nrow, and ncol must be"
                "specified if a MODFLOW 6 binary grid"
                "file name is not specified."
            )

    # flatten flowja, if necessary
    if len(flowja.shape) > 0:
        flowja = flowja.flatten()

    # evaluate size of flowja relative to ja
    __check_flowja_size(flowja, ja)

    # create empty flat face flow arrays
    shape = (nlay, nrow, ncol)
    frf = np.zeros(shape, dtype=float).flatten()  # right
    fff = np.zeros(shape, dtype=float).flatten()  # front
    flf = np.zeros(shape, dtype=float).flatten()  # lower

    def get_face(m, n, nlay, nrow, ncol):
        """
        Determine connection direction at (m, n)
        in a connection or intercell flow matrix.

        Notes
        -----
        For visual intuition in 2 dimensions
        https://stackoverflow.com/a/16330162/6514033
        helps. MODFLOW uses the left-side scheme in 3D.

        Parameters
        ----------
        m : int
            row index
        n : int
            column index
        nlay : int
            number of layers in the grid
        nrow : int
            number of rows in the grid
        ncol : int
            number of columns in the grid

        Returns
        -------
        face : int
            0: right, 1: front, 2: lower
        """

        d = m - n
        if d == 1:
            # handle 1D cases
            if nrow == 1 and ncol == 1:
                return 2
            elif nlay == 1 and ncol == 1:
                return 1
            elif nlay == 1 and nrow == 1:
                return 0
            else:
                # handle 2D layers/rows case
                return 1 if ncol == 1 else 0
        elif d == nrow * ncol:
            return 2
        else:
            return 1

    # fill right, front and lower face flows
    # (below main diagonal)
    flows = [frf, fff, flf]
    for n in range(grb.nodes):
        for i in range(ia[n] + 1, ia[n + 1]):
            m = ja[i]
            if m <= n:
                continue
            face = get_face(m, n, nlay, nrow, ncol)
            flows[face][n] = -1 * flowja[i]

    # reshape and return
    if asdict:
        return {'frf': frf.reshape(shape),
                'fff': fff.reshape(shape), 
                'flf': flf.reshape(shape),
        }
    else:
        return frf.reshape(shape), fff.reshape(shape), flf.reshape(shape)



def get_residuals(
    flowja, grb_file=None, ia=None, ja=None, shape=None, verbose=False):
    """
    Get the residual from the MODFLOW 6 flowja flows. The residual is stored
    in the diagonal position of the flowja vector.

    Parameters
    ----------
    flowja : ndarray
        flowja array for a structured MODFLOW 6 model
    grbfile : str
        MODFLOW 6 binary grid file path
    ia : list or ndarray
        CRS row pointers. Only required if grb_file is not provided.
    ja : list or ndarray
        CRS column pointers. Only required if grb_file is not provided.
    shape : tuple
        shape of returned residual. A flat array is returned if shape is None
        and grbfile is None.
    verbose: bool
        Write information to standard output

    Returns
    -------
    residual : ndarray
        Residual for each cell

    """
    if grb_file is not None:
        grb = fpb.MfGrdFile(grb_file, verbose=verbose)
        shape = grb.shape
        ia, ja = grb.ia, grb.ja
    else:
        if ia is None or ja is None:
            raise ValueError(
                "ia and ja arrays must be specified if the MODFLOW 6 "
                "binary grid file name is not specified."
            )

    # flatten flowja, if necessary
    if len(flowja.shape) > 0:
        flowja = flowja.flatten()

    # evaluate size of flowja relative to ja
    __check_flowja_size(flowja, ja)

    # create residual
    nodes = grb.nodes
    residual = np.zeros(nodes, dtype=float)

    # fill flow terms
    for n in range(nodes):
        i0, i1 = ia[n], ia[n + 1]
        if i0 < i1:
            residual[n] = flowja[i0]
        else:
            residual[n] = np.nan

    # reshape residual terms
    if shape is not None:
        residual = residual.reshape(shape)
    return residual

# internal
def __check_flowja_size(flowja, ja):
    """
    Check the shape of flowja relative to ja.
    """
    if flowja.shape != ja.shape:
        raise ValueError(
            f"size of flowja ({flowja.shape}) not equal to {ja.shape}"
        )


# Fast version. First compute the pointers, then use them
# one or more times to pick the flows directly.
# TO 231115

def get_indices_to_pick_fxf(grb_file=None):
    """Return the indices to pick the flow lower fact from the flowja array.
    
    This method only works for a structured MODFLOW 6 model.

    Parameters
    ----------
    flowja : ndarray
        flowja array for a structured MODFLOW 6 model
    grbfile : str
        MODFLOW 6 binary grid file path
    verbose: bool
        Write information to standard output

    Returns
    -------
    i_flf : ndarray (grb.nodes)
        node index
    j_flf : ndarray (number of conncections)
        index of corresponding flf in the flowja array
    """
    grb = fpb.MfGrdFile(grb_file)
    if grb.grid_type != "DIS":
        raise ValueError(
                "get_structured_faceflows method "
                "is only for structured DIS grids"
        )
    ia = grb.ia # CRS row pointers.
    ja = grb.ja # CRS column pointers.

    nlay, nrow, ncol = grb.shape

    # get indices in flowja for flf
    j_frf = -np.ones(grb.nodes, dtype=int)
    j_fff = -np.ones(grb.nodes, dtype=int)
    j_flf = -np.ones(grb.nodes, dtype=int)
    for n in range(grb.nodes):
        j0, j1 = ia[n] + 1, ia[n + 1]        
        for j in range(j0, j1): # skips when i0 == i1 (no connected flows, inactive cell)
            k = ja[j]
            if ncol > 1:
                if k == n + 1:
                    j_frf[n] = j
                    continue
            if nrow > 1:
                if k == n + ncol:
                    j_fff[n] = j
                    continue
            if nlay > 1:
                if k == n + ncol * nrow:
                    j_flf[n] = j
                    continue
    
    NOD   = np.arange(grb.ncells)     
    dtype = np.dtype([('node', int), ('ja', int)])
    
    ja_ptr = dict()
    for face, j_fxf in zip(['frf', 'fff', 'flf'], [j_frf, j_fff, j_flf]):
        node, ja = NOD[j_fxf >= 0], j_fxf[j_fxf >= 0]
        ptr = np.zeros(len(node), dtype=dtype)
        ptr['node'], ptr['ja'] = node, ja        
        ja_ptr[face] = ptr
    return ja_ptr, grb


def get_struct_flows(flowjas, grb_file=None, verbose=False):
    """Return flow_lower_face for all flowja (len nper).
    
    From the MODFLOW 6 flowja flows.
    This method only works for a structured MODFLOW 6 model.
    
    Parameters
    ----------
    flowjas: list of nstp_nper np.arrays of connected cell flows
        from CBC.get_data(text='JA-FLOW-JA')
    grb: filename
        name of the binary flowpy grid file
        
    Returns
    -------
    rec array of length flowjas, with keys frf, fff, flf, the three
    strutured arrays for flowjas[i]
    
    @TO 20231201
    """
    ja_ptr, grb = get_indices_to_pick_fxf(grb_file=grb_file)

    flowjas = [flowja.flatten() for flowja in flowjas[0]]
    
    dtype = np.dtype([('frf', (float, grb.shape)),
                      ('fff', (float, grb.shape)),
                      ('flf', (float, grb.shape))])
    
    if verbose:
        print(f"Generating a recarray len {len(flowjas)} of structed flows of dtype\n{repr(dtype)}")

    
    fxfs = np.zeros(len(flowjas), dtype=dtype)
    
    for i, flowja in enumerate(flowjas):
        fxfs[i]['frf'].ravel()[ja_ptr['frf']['node']] = -flowja[ja_ptr['frf']['ja']]
        fxfs[i]['fff'].ravel()[ja_ptr['fff']['node']] = -flowja[ja_ptr['fff']['ja']]
        fxfs[i]['flf'].ravel()[ja_ptr['flf']['node']] = -flowja[ja_ptr['flf']['ja']] # + or -? @TO 20240110
    return fxfs

def get_structured_flows_as_dict(budgetObj, grb_file=None, verbose=False):
    """Return a dict with the structured flows flf, fff, flrf.
    
    Parameters
    ----------
    budgetObj: flopy.budgetObj
        budgetObj with the flows in the cell-by-cell flows of the simulated GWF model.
    grb_file: string
        name of the flopy-generated grid file with .grb extension.
    
    Returns
    -------
    Dictionary with the flf, fff and frf. The keys are kstpkper in de budgetObj.
    The items for each kstpkper are rec_arrays with keys 'flf', 'fff', 'frf'
    and the data are float arrays with the shape of the model (nz, ny, nx)
    
    Usage
    -----
    fflows = structured_flows_dict(budgetObj, grb_file=grb_file_name)
    flf[kstpkper] = fflows[kstpkper]['flf']
    fff[kstpkper] = fflows[kstpkper]['fff']
    frf[kstpkper] = fflows[kstpkper]['frf']
    
    @TO 20240125
    """
    if not grb_file.endswith('.grb'):
        grb_file = grb_file + '.grb'

    ja_ptr, grb = get_indices_to_pick_fxf(grb_file=grb_file)
    flowjas = [flowja.flatten() for flowja in budgetObj.get_data(text='FLOW-JA')]
    dtype = np.dtype([('frf', (float, grb.shape)),
                      ('fff', (float, grb.shape)),
                      ('flf', (float, grb.shape))])
    
    kstpkper = budgetObj.get_kstpkper()
    
    if verbose:
        print("Generating a dictionary of {} structed flows with dtype\n{}".format(len(kstpkper), repr(dtype)))

    fxfs = dict()
    for ksp, flowja in zip(kstpkper, flowjas):
        fxfs[ksp] = np.zeros(1, dtype=dtype)
        for f in ['flf', 'fff', 'frf']:
            fxfs[ksp][f].ravel()[ja_ptr[f]['node']] = -flowja[ja_ptr[f]['ja']]
    return fxfs

if __name__ == '__main__':
    print('Hello')