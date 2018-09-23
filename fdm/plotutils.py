"""
Module containing helper functions for plotting model data
using ModelMap and ModelCrossSection. Functions for plotting
shapefiles are also included.

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import juka_tools.pestTools as pt


from matplotlib.colors import LinearSegmentedColormap
'''Some vectorixed grid related functions from flopy plotutil.py

@ TO 170729
'''


def get_contour_levels(A, dh=0.5):
    '''Return array of suitable levels for contouring array A.

    parameters
    ----------
        A: an ndarray
            array to be contoured
        dh: postive float
            distance between succesive levels.
    '''
    m = np.floor(np.nanmin(A))
    M = np.ceil( np.nanmax(A))
    n = int((M - m) / dh) + 1
    return np.linspace(m, M, n)


def show_layers(gr, coords=None, titles=None, Z=None, world=False, **kwargs):
    '''Plot layer levations in full color with colorbars usng contourf.

    This could be a method of mfgrid

    parameters
    ----------
    gr: fdm.mfgrid.Grid object
        model mesh object.
    coords = dict
        dict of point arrays to show on each map in model coordinates
        coords[k] must be in the form of [[x1, y1], [x2, y2], ...]
    world == bool, Default = False
        if True, coords must be in world coordinates and plot is made
        in world coordinates
        if False, coords must be in model coordinates and plot will be
        in model coordinates
    titles: list
        list if layer titles to use in header of subplots
    Z : 3D array LRC
        Array of layers to plot, if None, gr.Z is used
    '''

    if Z is None:
        Z = gr.Z

    n = len(Z) // 2 if len(Z) % 2 == 0 else len(Z) // 2 + 1

    fig, ax = plt.subplots(n, 2, sharex=True, sharey=True)

    if ax.ndim == 1: ax = ax[np.newaxis, :]

    if 'size_inches' in kwargs:
        fig.set_size_inches(kwargs.pop('size_inches'))

    cs = np.zeros_like(ax, dtype=object)

    if titles is None:
        titles = ['layer +{:02d}'.format(i) for i in range(len(Z))]

    for i in range(2):
        for j in range(2):
            try: # incase j fig [n, 2] does not exist
                ax[i,j].set_title(titles[2 * i + j])
                ax[i,j].set_xlabel('x [m]')
                ax[i,j].set_ylabel('y [m]')
                if 'xlim' in kwargs: ax[i,j].set_xlim(kwargs.pop('xlim'))
                if 'ylim' in kwargs: ax[i,j].set_ylim(kwargs.pop('ylim'))
                ax[i,j].grid()
                if world==False:
                    cs[i,j] = ax[i,j].contourf(gr.xm, gr.ym, Z[2 * i + j],\
                          levels=get_contour_levels(Z[2 * i + j]))
                else:
                    cs[i,j] = ax[i,j].contourf(gr.Xmw, gr.Ymw, Z[2 * i + j],\
                          levels=get_contour_levels(Z[2 * i + j]))
                plt.colorbar(cs[i, j], ax=ax[i, j])
                for k in coords:
                    ax[i, j].plot(*coords[k].T)
            except:
                pass
    return


def _show_layers(gr, wrld):
    '''Plot the layers as full contours in world coordinates.

    parameters
    ---------
    gr: fdm.mfgrid.Grid object
        model grid objects
    wrld: dict
        dictionary with world coordinates of spatial features.
    '''

    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches((13.5, 4.5))
    ax[0].set_title('Layer elevation for top of Beegden formation')
    ax[1].set_title('Layer elevation of top of Breda formation')

    m, M, n, levels, cs = dict(), dict(), dict(), dict(), dict()

    for layer in [0, 1]:
        ax[layer].set(xlabel='x [m]', ylabel='y [m]')
        ax[layer].grid()
        m[layer] = np.floor(np.min(gr.Z[layer][~np.isnan(gr.Z[layer])]))
        M[layer] = np.ceil( np.max(gr.Z[layer][~np.isnan(gr.Z[layer])]))
        n[layer] = int(4 * ( M[layer] - m[layer]) + 1)
        levels[layer]= np.linspace(m[layer], M[layer], n[layer])
        cs[layer] = ax[layer].contourf(gr.XMw[0], gr.YMw[0], gr.Z[layer], levels[layer])
        ax[layer].plot(wrld['modelgebied'].T[0], wrld['modelgebied'].T[1],
            color='brown', label='model area')
        plt.colorbar(cs[layer], ax=ax[layer])

        for name in com.Obb['show_items']:
            try:
                ax[layer].plot(wrld[name].T[0], wrld[name].T[1], label=name)
            except:
                ax[layer].plot(wrld[name][0][0], wrld[name][0][1], label=name)
    return


def plot_x_sections(gr, locations=None, coords=None, Z=None,
                    labels=None, size_inches=(10.0, 7.5), world=True, mask=None,
                    **kwargs):
    '''Plot a umber of WE cross sections to show layer elevations through locations

    parameters
    ----------
    gr: fdm.mfgrid.Grid object
        grid object (needed for its coordinates)
    locations: list of str
        location names, these names must be keys in the coords dict
    coords: dict
        coordinates of locations, which must be of shape [[x, y]]
    Z : ndarray, 3D to use shape must be [n, gr.ny, gr.nx]
        default is gr.Z
    labels: list of str to name the layers in the legend
        One label for every layer in Z
    size_inches: tuple of 2 floats
        Size of figure in inches
    world: bool
        Whether coords are in world or model coordinates
    kwargs: additional kwargs
        will be passed on to plt.plot
    '''

    fig, ax = plt.subplots(len(locations), 1)
    fig.set_size_inches(size_inches)

    if 'fontsize' in kwargs : fz = kwargs.pop('fontsize', 'medium')
    if 'fz'       in kwargs:  fz = kwargs.pop('fz','medium')
    plt.rcParams.update({'axes.titlesize': fz})

    # make sure coords are in both wrld and mdl coordinates
    if world==True:
        wrld = coords
        mdl = dict()
        for location in locations:
            mdl[location] = gr.world2model(*wrld[location].T)
    else:
        mdl = coords
        wrld = dict()
        for location in locations:
            wrld[location] = gr.m2world(*mdl[location].T)


    for i, location in enumerate(locations):
        iy = gr.iy(mdl[location][0][1])

        ax[i].set_title('WE section, at {}, yM={:.0f} m, row={}'
              .format(location, wrld[location][1][0], iy))

        ax[i].title.set_fontsize(fz)


        ax[i].set_ylabel(kwargs.pop('ylabel', 'elevation [m]'))
        if location == locations[-1]: # only the last gets an xlabel
            ax[i].set_xlabel(kwargs.pop('xlabel', 'xMdl [m]'))

        if 'xlim' in kwargs:
            ax[i].set_xlin(kwargs.pop('xlim'))
        if 'ylim' in kwargs:
            ax[i].set_ylim(kwargs.pop('ylim'))
        ax[i].grid()

        if Z is None: Z = gr.Z # use default

        if mask is not None:
            Z = np.ma.masked_array(Z, mask=mask)

        for iz, label in enumerate(labels):
            ax[i].plot(gr.xm, Z[iz, iy, :],label=label)

        ax[i].legend(loc='best')
    return

def plot_section(gr, piezoms=None, t0sim=None, time=None, hds=None, Z=None,
                 znames=None, hnames=None, colors=None, gmap=None, wrld=None, **kwargs):
    '''Plot a section along the piezometers (or calibs)

    parameters
    ----------
        gr: tools.fdm.mfgrid.Grid object
            the modflow mesh
        piezoms: tools.piezoms.piezoms or calibs object
            piezometers along which the section is to be plotted
        t0sim: pd.Timestamp object
            time corresponding to the start of the simulation
        time: np.array of times as floats
            modflow simulation times [d]
        hds: 4d array [nt, nz, ny, nx]
            simulated heads
        Z: 3D array of [n, ny, nx]
            elevations to plot
        znames: list of str
            labels for the layers in the Z array for legend
        hnames: list of str
            labels for the head layers for legend
        colors: list of str
            layer colors for the heads
        gmap: googlemaps.getmap.Gmap
            map to plot the piezometer locations on
        wrld: dict
            wrld[key] had coordinates of item to be plotted
        kwargs: additional kwargs to specify ax and grid:
            title, xlabel ylabel xlim ylim size_inches
            other left-over kwargs are passed on to ax.plot(..., **kwargs)

    '''

    names = [n for n in piezoms] # names

    # get the points for the piezoms
    xy = np.array([(piezoms[nm].meta['x'], piezoms[nm].meta['y']) for nm in piezoms])

    # interpolated the Z
    s, z, SP = gr.interp(xy, Z=Z)

    # interpolate the hds
    _, h, _ = gr.interp(xy, hds)

    fig, ax = plt.subplots(2, 1)

    # absolute time of heads in plotted section
    t = pd.Timestamp(t0sim)

    ax[0].set_title(kwargs.pop('title', 'Section along piezometers with head at time {}'
                .format(t.strftime('%Y-%m-%d %H:%M'))))
    ax[0].set_xlabel(kwargs.pop('xlabel', 'Distance along section [m]'))
    ax[0].set_ylabel(kwargs.pop('ylabel', 'm NAP'))
    ax[0].grid()

    if 'xlim' in kwargs:
        ax[0].set_xlim(kwargs.pop('xlim'))
    if 'ylim' in kwargs:
        ax[0].set_ylim(kwargs.pop('ylim'))
    if 'size_inches' in kwargs:
        fig.set_size_inches(kwargs.pop('size_inches'))

    for z_, zname in zip(z.T, znames):
        ax[0].plot(s.T[0], z_, lw=3, label='layer {}'.format(zname))
    for color, h_, hname in zip(colors, h.T, hnames):
        ax[0].plot(s.T[0], h_, color=color, label='hds {}'.format(hname))

    xlim = ax[0].get_xlim()
    ylim = ax[0].get_ylim()
    xoff = (xlim[1] - xlim[0]) / 80.
    ytxt = ylim[0] + 0.95 * (ylim[1] - ylim[0])
    colors_used = np.ones_like(colors, dtype=bool) * False
    for nm, sp in zip(names, SP):
        iz = piezoms[nm].meta['iz']
        ax[0].axvline(sp[0], linewidth=0.5, color='k')
        ylim = ax[0].get_ylim()
        att = piezoms[nm].attime(t)
        marker = dict( mec=colors[iz], mfc=None, fillstyle='none',
                  markerfacecoloralt='black')
        if not colors_used[iz]:
            ax[0].plot(sp[0], att['computed'], 'o', markersize=3, label='computed', **marker)
            ax[0].plot(sp[0], att['measured'], 'o', markersize=6, label='measured', **marker)
        else:
            ax[0].plot(sp[0], att['computed'], 'o', markersize=3, **marker)
            ax[0].plot(sp[0], att['measured'], 'o', markersize=6, **marker)
        if iz == 0:   # same as nm.endswith('A'):
            ax[0].text(sp[0] -xoff, ytxt, nm, rotation=90)
        else:
            ax[0].text(sp[0], ytxt, nm, rotation = 90)
        colors_used[iz] = True

    ax[0].legend(loc='best')

    pt.plot_piezoms(piezoms, gmap, wrld=wrld, xoff=10, yoff=25, ax=ax[1], **kwargs)


def add_piezoms_map(piez, gmap, fs=10, pos1=[0.125, 0.55, 0.75, 0.35],
                    pos2 = [0.32, 0.12, 0.4, 0.35], size_inches=(12, 10),
                    wrld=None,
                    xoff=10, yoff=25):
    '''Adds map below the axis on the current figure.

    parameters
    ----------
     piez: piezometers.piezometers.Calibs or piezometers.piezometers.Piezoms collection object
         the piezometers to be plotted on the map
     gmap: a googlemaps.getmap.Gmap object
         the backgroound map
    fs : int
        fontsize to be used for all labels in the axes
     pos1: [l,b, w,h] default = [0.125, 0.55, 0.88, 0.35]
         resets the original axis default
     pos2: [l, b, w, h] default = [0.32, 0.1, 0.4, 0.35]
         sets the new axis position for the map
     size_inches: [w, h] default [16, 9]
         resets the size of the figure
    wrld: dict
        dict with coordinates of features wrld[nm]=[[x1, y1], [x2, y2], ...]
    xoff: float default 10
        horizontal label offset in map coorcinates
    yoff: float default 25
        vertical label offset in map coordinates

    @TO 20180923
    '''

    # get currenf figure and reset its size
    fig = plt.gcf()
    fig.set_size_inches(size_inches)

    # get current axis and reset its position
    ax0 = plt.gca()
    ax0.set_position(pos1)

    # add new axis of map
    ax1 = fig.add_axes(pos2)


    pt.plot_piezoms(piez, gmap, wrld=wrld, xoff=10, yoff=25, ax=ax1)

    ax1.set_xlabel('xRd[m]', size=fs)
    ax1.set_ylabel('yRd[m]', size=fs)

    for a in [ax0, ax1]:
        for label in (a.get_xticklabels() + a.get_yticklabels()):
            label.set_fontsize(fs)

    plt.show()


def findrowcolumn(pt, edges, fraction=False):
    """
    returns MODFLOW cells indices of points pt in grid defined by edges

    Parameters
    ----------
    pt : np.ndarray of dimension [n,1], [n,2] or [n,3]
    edges : tuple containing 1, 2 or 3 numpy.ndarrays of grid-face coordinates
        that correspond to the first, second or third column of pt

    Returns
    -------
    np.array(np, 2, dtype=int) of cell index pairs
        Row and column location containing the points passed to function.

    Examples
    --------
    >>> import flopy
    >>> irow, jcol = flopy.plotutil.findrowcolumn(pt, xedge, yedge)

    """

    # make sure xedge and yedge are numpy arrays
    pt = np.array(pt)
    if len(pt.shape) == 1:
        pt = pt[:, np.newaxis]
    assert pt.ndim == 2, "points must be an ndarray of shape (npoints, 2)"
    assert isinstance(edges, tuple) and isinstance(edges[0], np.ndarray)

    id_int  = []
    id_frac = []
    for u, uGr in zip(pt, edges):
        sgn = np.sign(u[-1] - u[0]) # ensure uGr increases
        Nu = len(uGr)
        i_rel = np.interp1(sgn * u, sgn * uGr, np.arange(Nu), left=np.nan, right=np.nan)

        i_frac = i_rel - np.floor(i_rel)
        i_int  = np.array(i_rel, dtype=int)

        i_frac[i_int == Nu] = 1.0
        i_frac[i_int == Nu] = Nu - 1.0

        id_int.append(i_int)
        id_frac.append(i_frac)
    return (id, id_frac)


def line_intersect_grid(pt, xGr, yGr, zGr, returnvertices=False):
    """
    Intersect a list of polyline vertices with a rectilinear MODFLOW
    grid. Vertices at the intersection of the polyline with the grid
    cell edges are returned. Optionally the original polyline vertices
    are returned.
    Points outside the grid are not returned unless returnvertices==True.

    Parameters
    ----------
    ptsin : list
        A list of x, y points defining the vertices of a polyline that will be
        intersected with the rectilinear MODFLOW grid
    xedge : numpy.ndarray
        x-coordinate of the edge of each MODFLOW column. xedge is dimensioned
        to NCOL + 1. If xedge is not a numpy.ndarray it is converted to a
        numpy.ndarray.
    yedge : numpy.ndarray
        y-coordinate of the edge of each MODFLOW row. yedge is dimensioned
        to NROW + 1. If yedge is not a numpy.ndarray it is converted to a
        numpy.ndarray.
    returnvertices: bool
        Return the original polyline vertices in the list of numpy.ndarray
        containing vertices resulting from intersection of the provided
        polygon and the MODFLOW model grid if returnvertices=True.
        (default is False).

    Returns
    -------
    (x, y, dlen) : numpy.ndarray of tuples
        numpy.ndarray of tuples containing the x, y, and segment length of the
        intersection of the provided polyline with the rectilinear MODFLOW
        grid.

    Examples
    --------
    >>> import flopy
    >>> ptsout = flopy.plotutil.line_intersect_grid(ptsin, xedge, yedge)

    """

    xGr = np.array(xGr)
    yGr = np.array(yGr)
    zGr = np.array(zGr)

    pt = np.matrix(pt)
    dp = np.matrix(np.diff(axi=0))

    lam = np.hstack(((xGr[np.newaxis, :] - pt[:-1, 0]) /  dp[:, 0],
                     (yGr[np.newaxis, :] - pt[:-1, 1]) /  dp[:, 1],
                     (zGr[np.newaxis, :] - pt[:-1, 2]) /  dp[:, 2]))
    p_out = []
    for i, la in enumerate(lam):
        la = np.sorted(la[np.logical_and(la>0.0, la<1.0)])
        if returnvertices:
            la = np.hstack((0, la))
        p_out += list(pt[i] + la * dp[i])
    if returnvertices:
        p_out.append(pt[-1])
    return np.array(p_out)


def get_contour_levels(A, dh=0.5):
    '''Return array of suitable levels for contouring array A.

    parameters
    ----------
        A: an ndarray
            array to be contoured
        dh: postive float
            distance between succesive levels.
    '''
    m = np.floor(np.nanmin(A))
    M = np.ceil( np.nanmax(A))
    n = int((M - m) / dh) + 1
    return np.linspace(m, M, n)



def cell_value_points(pts, xGr, yGr, zGr, vdata, mid=False):
    """
    Return vertices of points obtained by intersecting ponts pts with a grid
    defined by xGr, yGt, zGr and deliver thel values at these points interpolated
    within vdata.

    If mid==False, the cell values are computed at exactly these points.
    If mid==True,  the cell values at successive mid cell-points

    Parameters
    ----------
    pts : ndarray of shape [npnt, 3]
    xGr : numpy.ndarray, length == Nx + 1
    yGr : numpy.ndarray, length == Ny + 1
    zGr : numpy.ndarray, length == Nz + 1
    vdata : numpy.ndarray
        Data (i.e., head, hk, etc.) for a rectilinear MODFLOW model grid. The
        shape of vdata is (NROW, NCOL). If vdata is not a numpy.ndarray it is
        converted to a numpy.ndarray.

    Returns
    -------
    pnts, LRC, vCell:
        pnts : points, see pnts above
        LRC: numpy int array of coordinats of cell in which pnts fall
        vCell : numpy.ndarray, values

    Examples
    --------
    >>> import flopy
    >>> pnts, LRC, vcell = flopy.plotutil.cell_value_points(xpts, xedge, yedge, head[0, :, :])
    """

    # make sure xedge and yedge are numpy arrays
    Nx, Ny = len(xGr)-1, len(yGr)-1, len(zGr)-1
    pts = line_intersect_grid(pts, xGr, yGr, zGr, returnvertices=False)
    if mid:
        pts = 0.5 * (pts[:-1] + pts[1:])
    LRC = findrowcolumn(pts)
    I = LRC[0] * Ny * Nx + LRC[1] * Nx + LRC[2]
    return pts, LRC, vdata.ravel()[I]



cm_data = [[ 0.26700401,  0.00487433,  0.32941519],
       [ 0.26851048,  0.00960483,  0.33542652],
       [ 0.26994384,  0.01462494,  0.34137895],
       [ 0.27130489,  0.01994186,  0.34726862],
       [ 0.27259384,  0.02556309,  0.35309303],
       [ 0.27380934,  0.03149748,  0.35885256],
       [ 0.27495242,  0.03775181,  0.36454323],
       [ 0.27602238,  0.04416723,  0.37016418],
       [ 0.2770184 ,  0.05034437,  0.37571452],
       [ 0.27794143,  0.05632444,  0.38119074],
       [ 0.27879067,  0.06214536,  0.38659204],
       [ 0.2795655 ,  0.06783587,  0.39191723],
       [ 0.28026658,  0.07341724,  0.39716349],
       [ 0.28089358,  0.07890703,  0.40232944],
       [ 0.28144581,  0.0843197 ,  0.40741404],
       [ 0.28192358,  0.08966622,  0.41241521],
       [ 0.28232739,  0.09495545,  0.41733086],
       [ 0.28265633,  0.10019576,  0.42216032],
       [ 0.28291049,  0.10539345,  0.42690202],
       [ 0.28309095,  0.11055307,  0.43155375],
       [ 0.28319704,  0.11567966,  0.43611482],
       [ 0.28322882,  0.12077701,  0.44058404],
       [ 0.28318684,  0.12584799,  0.44496   ],
       [ 0.283072  ,  0.13089477,  0.44924127],
       [ 0.28288389,  0.13592005,  0.45342734],
       [ 0.28262297,  0.14092556,  0.45751726],
       [ 0.28229037,  0.14591233,  0.46150995],
       [ 0.28188676,  0.15088147,  0.46540474],
       [ 0.28141228,  0.15583425,  0.46920128],
       [ 0.28086773,  0.16077132,  0.47289909],
       [ 0.28025468,  0.16569272,  0.47649762],
       [ 0.27957399,  0.17059884,  0.47999675],
       [ 0.27882618,  0.1754902 ,  0.48339654],
       [ 0.27801236,  0.18036684,  0.48669702],
       [ 0.27713437,  0.18522836,  0.48989831],
       [ 0.27619376,  0.19007447,  0.49300074],
       [ 0.27519116,  0.1949054 ,  0.49600488],
       [ 0.27412802,  0.19972086,  0.49891131],
       [ 0.27300596,  0.20452049,  0.50172076],
       [ 0.27182812,  0.20930306,  0.50443413],
       [ 0.27059473,  0.21406899,  0.50705243],
       [ 0.26930756,  0.21881782,  0.50957678],
       [ 0.26796846,  0.22354911,  0.5120084 ],
       [ 0.26657984,  0.2282621 ,  0.5143487 ],
       [ 0.2651445 ,  0.23295593,  0.5165993 ],
       [ 0.2636632 ,  0.23763078,  0.51876163],
       [ 0.26213801,  0.24228619,  0.52083736],
       [ 0.26057103,  0.2469217 ,  0.52282822],
       [ 0.25896451,  0.25153685,  0.52473609],
       [ 0.25732244,  0.2561304 ,  0.52656332],
       [ 0.25564519,  0.26070284,  0.52831152],
       [ 0.25393498,  0.26525384,  0.52998273],
       [ 0.25219404,  0.26978306,  0.53157905],
       [ 0.25042462,  0.27429024,  0.53310261],
       [ 0.24862899,  0.27877509,  0.53455561],
       [ 0.2468114 ,  0.28323662,  0.53594093],
       [ 0.24497208,  0.28767547,  0.53726018],
       [ 0.24311324,  0.29209154,  0.53851561],
       [ 0.24123708,  0.29648471,  0.53970946],
       [ 0.23934575,  0.30085494,  0.54084398],
       [ 0.23744138,  0.30520222,  0.5419214 ],
       [ 0.23552606,  0.30952657,  0.54294396],
       [ 0.23360277,  0.31382773,  0.54391424],
       [ 0.2316735 ,  0.3181058 ,  0.54483444],
       [ 0.22973926,  0.32236127,  0.54570633],
       [ 0.22780192,  0.32659432,  0.546532  ],
       [ 0.2258633 ,  0.33080515,  0.54731353],
       [ 0.22392515,  0.334994  ,  0.54805291],
       [ 0.22198915,  0.33916114,  0.54875211],
       [ 0.22005691,  0.34330688,  0.54941304],
       [ 0.21812995,  0.34743154,  0.55003755],
       [ 0.21620971,  0.35153548,  0.55062743],
       [ 0.21429757,  0.35561907,  0.5511844 ],
       [ 0.21239477,  0.35968273,  0.55171011],
       [ 0.2105031 ,  0.36372671,  0.55220646],
       [ 0.20862342,  0.36775151,  0.55267486],
       [ 0.20675628,  0.37175775,  0.55311653],
       [ 0.20490257,  0.37574589,  0.55353282],
       [ 0.20306309,  0.37971644,  0.55392505],
       [ 0.20123854,  0.38366989,  0.55429441],
       [ 0.1994295 ,  0.38760678,  0.55464205],
       [ 0.1976365 ,  0.39152762,  0.55496905],
       [ 0.19585993,  0.39543297,  0.55527637],
       [ 0.19410009,  0.39932336,  0.55556494],
       [ 0.19235719,  0.40319934,  0.55583559],
       [ 0.19063135,  0.40706148,  0.55608907],
       [ 0.18892259,  0.41091033,  0.55632606],
       [ 0.18723083,  0.41474645,  0.55654717],
       [ 0.18555593,  0.4185704 ,  0.55675292],
       [ 0.18389763,  0.42238275,  0.55694377],
       [ 0.18225561,  0.42618405,  0.5571201 ],
       [ 0.18062949,  0.42997486,  0.55728221],
       [ 0.17901879,  0.43375572,  0.55743035],
       [ 0.17742298,  0.4375272 ,  0.55756466],
       [ 0.17584148,  0.44128981,  0.55768526],
       [ 0.17427363,  0.4450441 ,  0.55779216],
       [ 0.17271876,  0.4487906 ,  0.55788532],
       [ 0.17117615,  0.4525298 ,  0.55796464],
       [ 0.16964573,  0.45626209,  0.55803034],
       [ 0.16812641,  0.45998802,  0.55808199],
       [ 0.1666171 ,  0.46370813,  0.55811913],
       [ 0.16511703,  0.4674229 ,  0.55814141],
       [ 0.16362543,  0.47113278,  0.55814842],
       [ 0.16214155,  0.47483821,  0.55813967],
       [ 0.16066467,  0.47853961,  0.55811466],
       [ 0.15919413,  0.4822374 ,  0.5580728 ],
       [ 0.15772933,  0.48593197,  0.55801347],
       [ 0.15626973,  0.4896237 ,  0.557936  ],
       [ 0.15481488,  0.49331293,  0.55783967],
       [ 0.15336445,  0.49700003,  0.55772371],
       [ 0.1519182 ,  0.50068529,  0.55758733],
       [ 0.15047605,  0.50436904,  0.55742968],
       [ 0.14903918,  0.50805136,  0.5572505 ],
       [ 0.14760731,  0.51173263,  0.55704861],
       [ 0.14618026,  0.51541316,  0.55682271],
       [ 0.14475863,  0.51909319,  0.55657181],
       [ 0.14334327,  0.52277292,  0.55629491],
       [ 0.14193527,  0.52645254,  0.55599097],
       [ 0.14053599,  0.53013219,  0.55565893],
       [ 0.13914708,  0.53381201,  0.55529773],
       [ 0.13777048,  0.53749213,  0.55490625],
       [ 0.1364085 ,  0.54117264,  0.55448339],
       [ 0.13506561,  0.54485335,  0.55402906],
       [ 0.13374299,  0.54853458,  0.55354108],
       [ 0.13244401,  0.55221637,  0.55301828],
       [ 0.13117249,  0.55589872,  0.55245948],
       [ 0.1299327 ,  0.55958162,  0.55186354],
       [ 0.12872938,  0.56326503,  0.55122927],
       [ 0.12756771,  0.56694891,  0.55055551],
       [ 0.12645338,  0.57063316,  0.5498411 ],
       [ 0.12539383,  0.57431754,  0.54908564],
       [ 0.12439474,  0.57800205,  0.5482874 ],
       [ 0.12346281,  0.58168661,  0.54744498],
       [ 0.12260562,  0.58537105,  0.54655722],
       [ 0.12183122,  0.58905521,  0.54562298],
       [ 0.12114807,  0.59273889,  0.54464114],
       [ 0.12056501,  0.59642187,  0.54361058],
       [ 0.12009154,  0.60010387,  0.54253043],
       [ 0.11973756,  0.60378459,  0.54139999],
       [ 0.11951163,  0.60746388,  0.54021751],
       [ 0.11942341,  0.61114146,  0.53898192],
       [ 0.11948255,  0.61481702,  0.53769219],
       [ 0.11969858,  0.61849025,  0.53634733],
       [ 0.12008079,  0.62216081,  0.53494633],
       [ 0.12063824,  0.62582833,  0.53348834],
       [ 0.12137972,  0.62949242,  0.53197275],
       [ 0.12231244,  0.63315277,  0.53039808],
       [ 0.12344358,  0.63680899,  0.52876343],
       [ 0.12477953,  0.64046069,  0.52706792],
       [ 0.12632581,  0.64410744,  0.52531069],
       [ 0.12808703,  0.64774881,  0.52349092],
       [ 0.13006688,  0.65138436,  0.52160791],
       [ 0.13226797,  0.65501363,  0.51966086],
       [ 0.13469183,  0.65863619,  0.5176488 ],
       [ 0.13733921,  0.66225157,  0.51557101],
       [ 0.14020991,  0.66585927,  0.5134268 ],
       [ 0.14330291,  0.66945881,  0.51121549],
       [ 0.1466164 ,  0.67304968,  0.50893644],
       [ 0.15014782,  0.67663139,  0.5065889 ],
       [ 0.15389405,  0.68020343,  0.50417217],
       [ 0.15785146,  0.68376525,  0.50168574],
       [ 0.16201598,  0.68731632,  0.49912906],
       [ 0.1663832 ,  0.69085611,  0.49650163],
       [ 0.1709484 ,  0.69438405,  0.49380294],
       [ 0.17570671,  0.6978996 ,  0.49103252],
       [ 0.18065314,  0.70140222,  0.48818938],
       [ 0.18578266,  0.70489133,  0.48527326],
       [ 0.19109018,  0.70836635,  0.48228395],
       [ 0.19657063,  0.71182668,  0.47922108],
       [ 0.20221902,  0.71527175,  0.47608431],
       [ 0.20803045,  0.71870095,  0.4728733 ],
       [ 0.21400015,  0.72211371,  0.46958774],
       [ 0.22012381,  0.72550945,  0.46622638],
       [ 0.2263969 ,  0.72888753,  0.46278934],
       [ 0.23281498,  0.73224735,  0.45927675],
       [ 0.2393739 ,  0.73558828,  0.45568838],
       [ 0.24606968,  0.73890972,  0.45202405],
       [ 0.25289851,  0.74221104,  0.44828355],
       [ 0.25985676,  0.74549162,  0.44446673],
       [ 0.26694127,  0.74875084,  0.44057284],
       [ 0.27414922,  0.75198807,  0.4366009 ],
       [ 0.28147681,  0.75520266,  0.43255207],
       [ 0.28892102,  0.75839399,  0.42842626],
       [ 0.29647899,  0.76156142,  0.42422341],
       [ 0.30414796,  0.76470433,  0.41994346],
       [ 0.31192534,  0.76782207,  0.41558638],
       [ 0.3198086 ,  0.77091403,  0.41115215],
       [ 0.3277958 ,  0.77397953,  0.40664011],
       [ 0.33588539,  0.7770179 ,  0.40204917],
       [ 0.34407411,  0.78002855,  0.39738103],
       [ 0.35235985,  0.78301086,  0.39263579],
       [ 0.36074053,  0.78596419,  0.38781353],
       [ 0.3692142 ,  0.78888793,  0.38291438],
       [ 0.37777892,  0.79178146,  0.3779385 ],
       [ 0.38643282,  0.79464415,  0.37288606],
       [ 0.39517408,  0.79747541,  0.36775726],
       [ 0.40400101,  0.80027461,  0.36255223],
       [ 0.4129135 ,  0.80304099,  0.35726893],
       [ 0.42190813,  0.80577412,  0.35191009],
       [ 0.43098317,  0.80847343,  0.34647607],
       [ 0.44013691,  0.81113836,  0.3409673 ],
       [ 0.44936763,  0.81376835,  0.33538426],
       [ 0.45867362,  0.81636288,  0.32972749],
       [ 0.46805314,  0.81892143,  0.32399761],
       [ 0.47750446,  0.82144351,  0.31819529],
       [ 0.4870258 ,  0.82392862,  0.31232133],
       [ 0.49661536,  0.82637633,  0.30637661],
       [ 0.5062713 ,  0.82878621,  0.30036211],
       [ 0.51599182,  0.83115784,  0.29427888],
       [ 0.52577622,  0.83349064,  0.2881265 ],
       [ 0.5356211 ,  0.83578452,  0.28190832],
       [ 0.5455244 ,  0.83803918,  0.27562602],
       [ 0.55548397,  0.84025437,  0.26928147],
       [ 0.5654976 ,  0.8424299 ,  0.26287683],
       [ 0.57556297,  0.84456561,  0.25641457],
       [ 0.58567772,  0.84666139,  0.24989748],
       [ 0.59583934,  0.84871722,  0.24332878],
       [ 0.60604528,  0.8507331 ,  0.23671214],
       [ 0.61629283,  0.85270912,  0.23005179],
       [ 0.62657923,  0.85464543,  0.22335258],
       [ 0.63690157,  0.85654226,  0.21662012],
       [ 0.64725685,  0.85839991,  0.20986086],
       [ 0.65764197,  0.86021878,  0.20308229],
       [ 0.66805369,  0.86199932,  0.19629307],
       [ 0.67848868,  0.86374211,  0.18950326],
       [ 0.68894351,  0.86544779,  0.18272455],
       [ 0.69941463,  0.86711711,  0.17597055],
       [ 0.70989842,  0.86875092,  0.16925712],
       [ 0.72039115,  0.87035015,  0.16260273],
       [ 0.73088902,  0.87191584,  0.15602894],
       [ 0.74138803,  0.87344918,  0.14956101],
       [ 0.75188414,  0.87495143,  0.14322828],
       [ 0.76237342,  0.87642392,  0.13706449],
       [ 0.77285183,  0.87786808,  0.13110864],
       [ 0.78331535,  0.87928545,  0.12540538],
       [ 0.79375994,  0.88067763,  0.12000532],
       [ 0.80418159,  0.88204632,  0.11496505],
       [ 0.81457634,  0.88339329,  0.11034678],
       [ 0.82494028,  0.88472036,  0.10621724],
       [ 0.83526959,  0.88602943,  0.1026459 ],
       [ 0.84556056,  0.88732243,  0.09970219],
       [ 0.8558096 ,  0.88860134,  0.09745186],
       [ 0.86601325,  0.88986815,  0.09595277],
       [ 0.87616824,  0.89112487,  0.09525046],
       [ 0.88627146,  0.89237353,  0.09537439],
       [ 0.89632002,  0.89361614,  0.09633538],
       [ 0.90631121,  0.89485467,  0.09812496],
       [ 0.91624212,  0.89609127,  0.1007168 ],
       [ 0.92610579,  0.89732977,  0.10407067],
       [ 0.93590444,  0.8985704 ,  0.10813094],
       [ 0.94563626,  0.899815  ,  0.11283773],
       [ 0.95529972,  0.90106534,  0.11812832],
       [ 0.96489353,  0.90232311,  0.12394051],
       [ 0.97441665,  0.90358991,  0.13021494],
       [ 0.98386829,  0.90486726,  0.13689671],
       [ 0.99324789,  0.90615657,  0.1439362 ]]

viridis = LinearSegmentedColormap.from_list(__file__, cm_data)

if __name__ == '__main__':
    pass
    #use this to show how functions are used

