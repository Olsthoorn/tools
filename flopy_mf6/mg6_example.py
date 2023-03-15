#!/usr/bin/env python3

# %%

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import flopy
from importlib import reload
from types import SimpleNamespace
from inspect import signature as sig
from scipy.interpolate import splprep, splev

tools = '/Users/Theo/GRWMODELS/python/tools/'
sys.path.append(tools)

from etc import newfig, newfigs, attr

reload(grv)

# %% Using supports to generate the grid 
rv = lambda Z: Z.ravel()

def get_params_from_excel(wbk):
    """Read mf6 parameters from workbook (sheetname='mf6').
    
    Parameters
    ----------
    wbk: str (path)
        Excel workbook name that holds the mf6 parameters in sheet 'MF6'
        
    Returns
    -------
    params: dictionary
        a dict with package name as key, in which each item is a dictionary
        specifying the values for this key.
        
    @TO 220413
    """
    paramsDf = pd.read_excel(wbk, 'MF6', header=0, usecols="A:D").dropna(axis=0)
    params = dict()
    for pkg in np.unique(paramsDf['package'].values):
        params[pkg]=dict()
        
    for i in paramsDf.index:
        pkg, parameter, value, type = paramsDf.loc[i]
        if value == 'None':
            continue
        if   type == 'None': value = None
        elif type == 'str': pass
        elif type == 'float': value = float(value)
        elif type == 'int': value = int(value)
        elif type == 'bool': value = True if value == 'True' else False
        elif type == 'list': value = exec(value)
        else: raise ValueError("Unknown parameter type pkg={}, {}, {}, {}".format(
                            pkg, parameter, value, type))
        params[pkg][parameter]=value
    return params

def get_ff(sim=None, ws=None, gr=None, cbc=None, kstpkper=None):
    """Return old fashoned frf, fff, flr for this structured dsv grid.
    
    Modfow 6 cbc files holds the cell-connection flow data in compressed sparse row (CSR) format.
    To interpret these data we need the IA and JA pointers that are in the <sim_name>.disv.grd file.
    With the IA and Ja pointers the data can be linked to cells.
    For this routine to work and for frf, fff and flf to make sense, the grid must consist of
    regular old fashioned layers, rows and columns. Only the vertice need not form rectangles;
    they may form quadrilaterals instead.
    
    The IA and JA pointers use global cell indices, so we reshape afterwards using gr.shape.
    
    Parameters
    ----------
    sim: sim object
        simulation object
    ws: str path to workspace
        path to the modflow input and output files
    gr: tools.fdm.GridDisv object
        The disv grid object.
    cbc: the cbc object
        as obtained by gwf.output.budget()
        
    Returns
    -------
    frf (flow right face) as a (nlay, nrow, ncol) array for each kstpkper in cbc object
        flow right face
    fff (flow front face) as a (nlay, nrow, ncol) array for each kstpkper in cbc object
        flow front face
    flf (flow lower face) as a (nlay, nrow, ncol) array for each kstpkper in cbc object
        flow lower face
        
    @ TO 220412
    """
    gwf_name = sim.get_model().name
    grdfile = os.path.join(ws, gwf_name + '.disv.grb')
    try:
        gobj = flopy.mf6.utils.MfGrdFile(grdfile)    
    except FileNotFoundError:
        raise FileNotFoundError("Can't find file {}".format(grdfile))
        
    IA, JA = gobj.ia, gobj.ja

    fff, frf, flf = gr.const(0.).ravel(), gr.const(0.).ravel(), gr.const(0.).ravel()
    Qcell = gr.const(0.).ravel()
    
    # TODO: still have to find out how Q is stored in case of model with multiple layers and simulation with multiple transient steps
    Q = cbc.get_data(text='FLOW-JA-FACE', kstpkper=kstpkper)[0].squeeze() # This may have to be adapted
    #print(Q)

    # Retrieve fff, frf, flf from grb file and budget
    for i1, i2 in zip(IA[:-1], IA[1:]):
        if i2 <= i1:
            continue # inactive cell    
        J = JA[i1:i2]
        ic = J[0]
        Qcell[ic] = Q[ic]
        for j in J[1:]:
            if j - ic == 1:
                frf[ic] = Q[j]
            elif j - ic == gr.ncol:
                fff[ic] = Q[j]
            elif j - ic >= gr.ncpl:
                flf[ic] = Q[j]
            else:
                pass
            
    return frf.reshape(gr.shape), fff.reshape(gr.shape), flf.reshape(gr.shape), Qcell.reshape(gr.shape)

# %%
def build_mf6model(config=None, params=None):
    if config.build_model:
        packages = {}
        packages['sim']  = flopy.mf6.MFSimulation(**params['sim'])
        packages['tdis'] = flopy.mf6.ModflowTdis(    packages['sim'], **params['tdis'])
        packages['ims']  = flopy.mf6.ModflowIms(     packages['sim'], **params['ims'])
        packages['gwf']  = flopy.mf6.ModflowGwf(     packages['sim'], **params['gwf'])
        packages['disv'] = flopy.mf6.ModflowGwfdisv( packages['gwf'], **params['disv'])
        packages['npf']  = flopy.mf6.ModflowGwfnpf(  packages['gwf'], **params['npf'])
        packages['sto'] = flopy.mf6.ModflowGwfsto(   packages['gwf'], **params['sto'])
        packages['ic']   = flopy.mf6.ModflowGwfic(   packages['gwf'], **params['ic'])
        packages['chd']  = flopy.mf6.ModflowGwfchd(  packages['gwf'], **params['chd'])
        packages['drn']  = flopy.mf6.ModflowGwfdrn(  packages['gwf'], **params['drn'])
        packages['wel']  = flopy.mf6.ModflowGwfwel(  packages['gwf'], **params['wel'])
        packages['rcha'] = flopy.mf6.ModflowGwfrcha( packages['gwf'], **params['rcha'])
        packages['oc']   = flopy.mf6.ModflowGwfoc(   packages['gwf'], **params['oc'])
        return packages
    return None
    
# %%
def write_model(config=None, packages=None, silent=True):
    if config.write_model:
        packages['sim'].write_simulation(silent=silent)
    
# %%
def run_model(config=None, packages=None, silent=True):
    success = True
    if config.run_model:
        success, buff = packages['sim'].run_simulation(silent=silent)
        if not success:
            print(buff)
    return success

# %%
def plot_results(config=None, packages=None, kstpkper=None, gr=None, nlevels=25):
    if config.plot_model:
        
        # Plot the head for the aqufers (layers [0, 2, 4])
        gwf = packages['gwf']
        hobj = gwf.output.head()
        if not kstpkper:
            kstpkper = (np.array(hobj.kstpkper) - 1)[-1]  # the last one coverted to base zero
        heads = hobj.get_data(kstpkper=kstpkper).squeeze()
        
        #ylim = heads.min(), heads.max()
        
        xlim, ylim = gr.extent[:2], gr.extent[2:]
        
        cobj = gwf.output.budget()

        # extract specific discharge
        qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(
            cobj.get_data(text="DATA-SPDIS", kstpkper=kstpkper)[0],
            packages['gwf'],
        )
        
        # modflow 6 layers to extract
        titles = ["Unconfined aquifer", "Middle aquifer", "Lower aquifer"]
        layers_mf6 = [0, 2, 4]
        ylims = (ylim, ylim, ylim)

        axs = newfigs(titles, "x", ["hd", "hd", "hd"] , figsize=(8, 12),
                      xlim=xlim, ylims=ylims)

        for ax, layer, head in zip(axs, layers_mf6, heads[::2]):
            # Plot a map, first instantiate a PlotMapView
            fmp = flopy.plot.PlotMapView(model=packages['gwf'], ax=ax, layer=layer)                
            plot_obj = fmp.plot_array(head, vmin=head.min(), vmax=head.max())

            fmp.plot_grid(lw=0.5)
            fmp.plot_bc("DRN", color="green")
            fmp.plot_bc("WEL", color="0.5")
            cv = ax.contour(gr.Xc.reshape(gr.nrow, gr.ncol), gr.Yc.reshape(gr.nrow, gr.ncol),
                            head.reshape(gr.shape[1:]), linewidth=0.25,
                            colors='black',
                            levels=np.linspace(head.min(), head.max(), nlevels)
                            )
            plt.clabel(cv, fmt="%.2f")
       
            fmp.plot_vector(qx[layer], qy[layer], normalize=True, color="0.75", ax=ax)

        # # plot colorbar
        cax = plt.axes([0.325, 0.125, 0.35, 0.025])
        cbar = plt.colorbar(
            plot_obj, shrink=0.8, orientation="horizontal", cax=cax
        )
        cbar.ax.tick_params(size=0)
        cbar.ax.set_xlabel(r"Head, $ft$", fontsize=9)
            
# %%
def simulation(config=None, params=None, kstpkper=None, gr=None, nlevels=25, silent=True):
    """Build, write, run model and plot results.
    
    Parameters
    ----------
    config: SimpleNamespace
        Class/name space to hold config options as booleans
    pparams: Dictionary
        dict of packages each with dict of package parameters.
    silent: bool
        whether or not plot information bfore end of this run.
    """
    packages = build_mf6model(config=config, params=params)
    write_model(config=config, packages=packages, silent=silent)
    success = run_model(config=config, packages=packages, silent=silent)

    if success:
        plot_results(config=config, packages=packages, kstpkper=kstpkper, gr=gr, nlevels=nlevels)
    return packages


# === RUNNING the file ===
if __name__ == '__main__':
    
    last_kstpkper = lambda cbc: (np.array(cbc.kstpkper) - 1)[-1] # also works for head
    
    # === FOLDERS and NAMES ===
    ws = '/Users/Theo/GRWMODELS/python/tools/flopy_mf6/examples/test/'
    executables = '/Users/Theo/GRWMODELS/mflab/trunk/bin/'
    exe_name = os.path.join(executables, 'mf6.mac')
    sim_name = 'test'
    model_name = sim_name
    
    # === parameter workbook, sheet_name = 'MF6' ===
    params_wbk = os.path.join(tools, 'flopy_mf6/mf_parameters.xlsx')

    # === confi for what to do ===
    config = SimpleNamespace(build_model=True, write_model=True, run_model=True, plot_model=True, plot_grid=False)

    # === MODFLOW GRID ===
    Xp = np.array([ [  0., 15., 30., 40., 65],
                    [ -5., 10., 25., 45., 70],
                    [ -3., 13., 27., 41., 65],
                    [-10., 11., 23., 37., 67]]) * 10.
    Yp = np.array([ [50., 45., 55., 50., 52.],
                    [37., 35., 40., 36., 35],
                    [18., 22., 19., 22., 20],
                    [ 5., -3.,  1., 5.,  4]]) * 10.

    z = [0, -25., -35., -50., -60., -100.]
    
    gr = grv.GridVdis(Xp, Yp, z, ncellsx=50, ncellsy=50, min_dz=0.001, spline=True)

    if config.plot_grid:
        ax = plt.gca()
        ax = newfig("Spline grid", "x", "y")
        gr.plot_grid()
        ax.plot(gr.Xc, gr.Yc, '.')
    
    # === DEFAULT parameters followed by parameter updates for current simulation ===
    params = get_params_from_excel(params_wbk)

    # === SIM ===
    params['sim'].update(sim_ws=ws, sim_name=sim_name, exe_name=exe_name)

    # === IC ===
    params['ic'].update(strt=0.0)

    # === TDIS ===
    perlen, nstp, tsmult = 100, 5, 1.25
    tdis_ds = ((perlen, nstp, tsmult), (perlen, nstp, tsmult), (perlen, nstp, tsmult))
    params['tdis'].update(nper=len(tdis_ds), perioddata=tdis_ds)

    # === GWF ===
    # If model name is not specified 'model' is used
    # model_rel_path can also be '.' then relatieve to sim_ws.
    params['gwf'].update(modelname=model_name, model_rel_path=ws, exe_name=exe_name) #Use same name as sim and same exe_name

    # === DISV ===
    top = gr.top
    botm= gr.botm    
    params['disv'].update(top=top, botm = botm,
                        nlay=gr.nlay, ncpl=gr.ncpl, nvert=gr.nvert,
                        vertices=gr.vertices, cell2d=gr.cell2d)
    
    # === STO ===
    params['sto'].update(steady_state=True, transient=[False, False, True])

    # === NPF ===
    k11 = gr.const([10., 0.1, 10., 0.1, 10.])
    k33 = k11
    params['npf'].update(k=k11, k33=k33)

    # === OC ===
    params['oc'].update(head_filerecord = "{}.hds".format(sim_name),
                        budget_filerecord = "{}.cbc".format(sim_name),
                        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")])

    # === CHD ===
    LIc_CHD = gr.LIcell(gr.NOD[0, :, [0, -1]].ravel(), astuple=True)
    chd_spd = [[cellid, 0.0] for cellid in LIc_CHD]
    params['chd'].update(stress_period_data={0: chd_spd}, maxbound=len(chd_spd))

    # === WEL ===
    LIc_WEL = gr.LIcell(gr.NOD[2][[15, 30]][:, [15, 30]].ravel(), astuple=True) \
            + gr.LIcell(gr.NOD[4][[10, 35]][:, [10, 35]].ravel(), astuple=True)
    wel_spd = [[cellid, -25.0] for cellid in LIc_WEL]
    params['wel'].update(stress_period_data={0: wel_spd}, maxbound=len(wel_spd))

    # === DRN ===
    LIc_DRN = gr.LIcell(gr.NOD[0, [0, -1], 1:-1].ravel(), astuple=True)
    drn_spd = [[cellid, 100, -1.0] for cellid in LIc_DRN]    
    params['drn'].update(stress_period_data={0: drn_spd}, maxbound=len(drn_spd))
    
    # === GHB ===
    # === RIV ===

    # === RCHA ===    
    params['rcha'].update(recharge=0.001)

    # === EVTA ===
    
    # === BUILD, WRITE, SIMULATE and PLOT and finally output the packages for inspection ===
    packages = simulation(config, params=params, kstpkper=None, gr=gr, nlevels=25, silent=False)
    
    # === Show how to construct old-fashioned frf, fff, flf from FLOW-JA-FACE in cbc and grid file
    cbc  = packages['gwf'].output.budget()
    
    grdfile = os.path.join(ws, sim_name + '.disv.grb')
    gobj = flopy.mf6.utils.MfGrdFile(grdfile)
    frf, fff, flf, Qcell = get_ff(packages['sim'], ws=ws, gr=gr, cbc=cbc, kstpkper=last_kstpkper(cbc))
    
    # === show all plots and wait for user to close them ===
    plt.show()
    # %%