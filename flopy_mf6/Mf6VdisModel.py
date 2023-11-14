#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 15 16:51:50 2021

First Flopy mdf6 tutorial from the flopy documentation

See for the examples on how to use the ts (time series) and tas (time array series)

https://github.com/modflowpy/flopy/blob/develop/examples/Notebooks/flopy3_mf6_obs_ts_tas.ipynb

To see how stress period data can be specified see
https://flopy.readthedocs.io/en/3.3.4/_notebooks/tutorial06_mf6_data.html
Note that the cellid information (layer, row, column) is encapsulated in a tuple.

This model was set up for the Eastern Aquifer model. It is here converted
to a general method to set up a MF6 model with flopy using its full
set of parameters, which are read from an Excel workbook.

@author: Theo 2021-22-16, 2023-08-03
"""
# %%
#import inspect
import os
import sys
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd
import pickle
import warnings
import flopy
import re
from fdm.mfgridVDIS import GridVdis
import logging
from types import SimpleNamespace
logging.basicConfig(level=logging.WARNING, format=' %(asctime)s - %(levelname)s - %(message)s')

# Folders is a dict in the module setPaths.
from setPaths import folders

os.chdir(folders['src'])

logging.warning("cwd = {}".format(os.getcwd()))

from setPaths import folders
import mountainAquiferShapes as mashps
import etc

import fdm.mfgrid # tools.fdm.mfgrid
import modelTiffsLib

logging.warning(sys.version)
logging.warning(sys.executable)

def plot_sections(grv=None, rows=None, zstp=None, inactive=None,formations=None, figsize=None):
    """Plot cross sections showing the formations along rows.
    
    Parameters
    ----------
    grv: grid object for vdis grid
        see mfgridVDIS
    rows: list of ints
        The rows of the grid (E-W) for which a cross section is desired.
    zstp: float
        Used for rounding vmax and vmin
    inactive 
    """
    Ztop = np.ma.masked_array(grv.ZT, mask=inactive)
    Zbot = np.ma.masked_array(grv.ZB, mask=inactive)
    zstp = 100. # For rounding of vmax and vmin
    vmax = np.ceil( np.max(Ztop) / zstp) * zstp
    vmin = np.floor(np.min(Zbot) / zstp) * zstp
    # plot layer cross sections using matpotlib patches
    axs = []
    for row in rows:
        ax = plot_one_section(grv=grv, formations=formations, row=row, Ztop=Ztop,
                              Zbot=Zbot, vmin=vmin, vmax=vmax, inactive=inactive, figsize=figsize)
        axs.append(ax)
    return axs

def ground_surface(grv, row, inactive):
    """Return ground surface taking inactive cells into account."""
    active = np.logical_not(inactive)
 
    z_gs = grv.ZT[-1, row]
    a_gs = active[-1, row]
    
    for zt, at in zip(grv.ZT[:-1, row][::-1], active[:-1, row][::-1]):
        z_gs[at] = zt[at]
        a_gs[at] = at[at]
        
    x = np.vstack((grv.Xv[row][:-1], grv.Xv[row][1:])).T.flatten()
    z = np.vstack((z_gs, z_gs)).T.flatten()
    m = np.logical_not(np.vstack((a_gs, a_gs)).T.flatten())
    return np.ma.MaskedArray(x, mask=m), np.ma.MaskedArray(z, mask=m) 
        

def plot_one_section(grv, formations=None, row=None, Ztop=None, Zbot=None, vmin=None, vmax=None,
                     inactive=None, figsize=None):
        "Plot a single cross section"
        ax = etc.newfig("Cross section Mountain Aquifer, row {}, y_EPGS6984 ={:.0f} m"
                        .format(row, np.mean(grv.Ym[row])),
                            "x[m] (EPGS6984)", "elev [m]", 
                            xlim = (grv.Xv[row, 0], grv.Xv[row, -1]),
                            ylim=(vmin, vmax),
                            figsize=figsize)
        
        # Plot the top (ground surface)
        x_gs, z_gs = ground_surface(grv, row, inactive)
        ax.plot(x_gs, z_gs, '-g', lw=2, label='Ground surface')            
        
        for ilay in range(grv.nlay):
            I = np.arange(grv.ncol, dtype=int)[inactive[ilay, row] == False]
            if len(I) == 0:
                continue
            
            # Turn this in a step curve that is also closed
            zb = np.vstack((grv.ZB[ilay, row, I],
                            grv.ZB[ilay, row, I])).T.flatten()
            zt = np.vstack((grv.ZT[ilay, row, I],
                            grv.ZT[ilay, row, I])).T.flatten()[::-1]
            xb = np.vstack((grv.Xv[row, I], grv.Xv[row, I + 1])).T.flatten()
            xt = xb[::-1]
            z = np.hstack((zb, zt, zb[0]))
            x = np.hstack((xb, xt, xb[0]))
            vertices = np.vstack((x, z)).T
            
            # Turn coordinates into a Path
            codes = np.ones(len(vertices), dtype=int)
            codes[0]  = patches.Path.MOVETO
            codes[1:] = patches.Path.LINETO
            codes[-1] = patches.Path.CLOSEPOLY
            
            # Generate patch and add to ax
            p = patches.PathPatch(patches.Path(vertices=vertices,
                            codes=codes, closed=True),
                            fc=formations[ilay]['color'], ec='gray', alpha=1)
            ax.add_patch(p)
            p.set_label(formations[ilay]['name'])
            
            # handles.append(patches.Patch(color=formations[ilay]['color'], label=formations[ilay]['name']))
            # labels.append(formations[ilay]['name'])
                           
        # ax.legend(handles, labels)
        ax.legend()
        return ax

def plot_heads(grv=None, HDS=None, rows=None, axs=None, iper=None, inactive=None):
    """Plot the heads for all layers in all cross sections"""
    
    markers = ['.', 'x', '+', 's']

    if HDS is None:
        return
    
    if isinstance(HDS, np.ndarray):
        hds = HDS
        color = 'blue'
        mdl = 'yasin (1999)'
    else:
        kskp = HDS.get_kstpkper()[iper]
        hds = HDS.get_data(kstpkper=kskp).reshape(grv.shape)
        color = 'black'
        mdl = 'mf6 (2022)'
        
    hds = np.ma.masked_array(hds, mask=inactive).reshape(grv.shape)
    
    for ax, row in zip(axs, rows):
        lss  = etc.line_cycler()
        mrk = etc.marker_cycler(markers=markers)
        for ilay in range(grv.nlay):
            ls = next(lss)
            mk = next(mrk)
            ax.plot(grv.Xm[row], hds[ilay, row, :], color=color, ls=ls, marker=mk,
                    mfc='none', label='head layer {}, {}'.format(ilay, mdl))
        ax.legend(loc='best')
    return None



def plot_cross_sections(mf6_dict, rows=None, HDS=None, iper=None,  figsize=(12, 6)):
    """Plot model cross sections along given rows.
    
    Parameters
    ----------
    mf6_dict: dict
        data for the mf6 model as a dictionary
    rows: sequence of ints
        numbers of the rows for which a cross section is desired.
    HDS: HDS file object (flopy)
        The head file object holding the heads and related info.
    iper: int
        Stess period number of which the heads of the last
        time step will be plotted.
    """
    
    zstp = 100.
    
    grv = mf6_dict['disv']['grv']
    formations = mf6_dict['formations']
    inactive = mf6_dict['disv']['idomain'] == 0
    
    axs = plot_sections(grv=grv, rows=rows, zstp=zstp, inactive=inactive,
                        formations=formations, figsize=figsize)
    
    plot_heads(grv=grv, HDS=HDS, rows=rows, axs=axs, inactive=inactive, iper=iper)
    
    return axs
    

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
    paramsDf = pd.read_excel(wbk, 'MF6', header=0, usecols="A:D", engine="openpyxl").dropna(axis=0)
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


def build_mf6model(used_packages=None, params=None):
    """Set the parameters for all simulation packages.
    
    The parameters have been read in from an excel workbook.
    Here they are added to the flopy packages.
    
    Extra paramters, like arrays still have to be added.
    
    Original:
        # packages['tdis'] = flopy.mf6.ModflowTdis(    packages['sim'], **params['tdis'])
        # packages['ims']  = flopy.mf6.ModflowIms(     packages['sim'], **params['ims'])
        # packages['gwf']  = flopy.mf6.ModflowGwf(     packages['sim'], **params['gwf'])
        
        # packages['disv'] = flopy.mf6.ModflowGwfdisv( packages['gwf'], **params['disv'])
        # packages['npf']  = flopy.mf6.ModflowGwfnpf(  packages['gwf'], **params['npf'])
        # packages['sto']  = flopy.mf6.ModflowGwfsto(   packages['gwf'], **params['sto'])
        # packages['ic']   = flopy.mf6.ModflowGwfic(   packages['gwf'], **params['ic'])
        # packages['chd']  = flopy.mf6.ModflowGwfchd(  packages['gwf'], **params['chd'])
        # packages['ghb']  = flopy.mf6.ModflowGwfghb(  packages['gwf'], **params['ghb'])
        # packages['drn']  = flopy.mf6.ModflowGwfdrn(  packages['gwf'], **params['drn'])
        # packages['wel']  = flopy.mf6.ModflowGwfwel(  packages['gwf'], **params['wel'])
        # packages['rcha'] = flopy.mf6.ModflowGwfrcha( packages['gwf'], **params['rcha'])
        # packages['evta'] = flopy.mf6.ModflowGwfevta( packages['gwf'], **params['evta'])
        # packages['oc']   = flopy.mf6.ModflowGwfoc(   packages['gwf'], **params['oc'])
    """ 
    # TODO: add missing sim modules
    flopy_sim_modules = {'tdis': flopy.mf6.ModflowTdis,
                    'ims': flopy.mf6.ModflowIms,
                    'gwf': flopy.mf6.ModflowGwf}

    # TODO: add missing gwf modules
    flopy_gwf_modules = {'disv': flopy.mf6.ModflowGwfdisv,
                    'npf':  flopy.mf6.ModflowGwfnpf,
                    'sto':  flopy.mf6.ModflowGwfsto,
                    'ic':   flopy.mf6.ModflowGwfic,
                    'chd':  flopy.mf6.ModflowGwfchd,
                    'ghb':  flopy.mf6.ModflowGwfghb,
                    'riv':  flopy.mf6.ModflowGwfriv,
                    'drn':  flopy.mf6.ModflowGwfdrn,
                    'wel':  flopy.mf6.ModflowGwfwel,
                    'rcha': flopy.mf6.ModflowGwfrcha,
                    'evta': flopy.mf6.ModflowGwfevta,
                    'oc':  flopy.mf6.ModflowGwfoc}

    packages = {}
    packages['sim']  = flopy.mf6.MFSimulation(**params['sim']) # always needed
    
    for pkg in flopy_sim_modules.keys():
        if pkg in used_packages:
            packages[pkg] = flopy_sim_modules[pkg](packages['sim'], **params[pkg])
    for pkg in flopy_gwf_modules.keys():
        if pkg in used_packages:  
            packages[pkg] = flopy_gwf_modules[pkg](packages['gwf'], **params[pkg])
    
    return packages


def write_model(packages=None, silent=True):
    """Write modflow/simulation input files
    
    Parameters
    ----------
    config: SimpleNamepace
        Configuration and simulation settings
    packages: ??
        ??
    silent: bool
        whether or not to write to screen during execution.
    """
    packages['sim'].write_simulation(silent=silent)


def run_model(packages=None, silent=True):
    """Run  modflow/simulation using the writeen input files.
    
    Parameters
    ----------
    config: SimpleNamepace
        Configuration and simulation settings
    packages: ??
        ??
    silent: bool
        whether or not to write to screen during execution.
    """
    success, buff = packages['sim'].run_simulation(silent=silent)
    print("========================================")
    print(f"# MODFLOW 6 did{' ' if success else ' not '}terminate normally. #")
    print("========================================")
    
    logging.critical(f"MODFLOW 6 did{' ' if success else ' not '}terminate normally")
        
    if not success:
        print(buff)
        logging.critical("Buffer printed because MODFLOW die not terminate normally.")
    return success


if __name__ == '__main__':

#%% Modflow 6 model as a Class
# Set up model_dict
# Store the case-specific model parameters in model_dict dictionary

# Generate model_dict
def generate_model_dict():
model_dict = {}

# sim
sim_name = 'my_simulation'
model_dict['sim'] = {'name': sim_name, 'sim_ws': '.'}

# tdis (perioddata)
nper = 10
model_dict['tdis'] = {'nper': nper, 'perioddata': [[365., 1, 1] for _ in range(nper)]}

# ims (solver,geneally use defaults from workbook)
model_dict['ims'] = {}

# gwf (groundwater flow model, use name from 'sim')
model_dict['gwf'] = {'model_name': model_dict['sim'][sim_name],
                     'exe_name': model_dict['sim']['exe_name']}

# disv (grid), use disv, unstructured layered grid
x = np.arange(-5, 5.1, 1.0)
y = np.arange(-6, 3.1, 1.0)
z = np.arange(-30, 01, 10.)
grv = GridVdis(x=x, y=y, z=z, axial=False, min_dz=0.1)
idomain = grv.const(0)

model_dict['disv'] = {'grv': grv}
model_dict['disv'].update(top=grv.top, botm = grv.botm,
                          idomain=model_dict['disv']['idomain'],
                          nlay=grv.nlay, ncpl=grv.ncpl, nvert=grv.nvert,
                          vertices=grv.vertices, cell2d=grv.cell2d)

# sto (Storage coefficients)
Ss = grv.const(0.0001)
Sy = grv.const(0.1)
model_dict['sto'] = {'steady_state': True, 'transient': None,
                     'Ss': Ss, 'Sy': Sy}

# npf (Node properties)
K1 = grv.const(10.)
K3 = grv.const(30.)
model_dict['npf'] = {'k1': K1, 'k3': K3} # Conductivities

# ic (Initial conditions)
strt = grv.const(0.3)
model_dict['ic'] = {'srt': strt}

# chd (time varying fixed heads)
LIc_CHD = grv.LRC2LIc(model_dict['chd'][0], bc=True)
#inactCHD = grv.LIc2LRC(grv.inactive(LIc_CHD, idomain=idomain)[0])
model_dict['chd'] = {'stress_period_data' :{0: LIc_CHD}, 'maxbound' :len(LIc_CHD)}

# drn (drains)
# drains are placed on top of each layer where domain > 0 and layer is exposed (is ground surface)
isdrn = np.ones_like(grv.NOD)
for iL, (isdr, idom) in enumerate(zip(isdrn, model_dict['disv']['idomain'])):
    isdr[idom == 0] = 0 # inactive cells don't get drains
    if iL > 0: # sub layers
        isdr[idomain[iL - 1] != 0] = 0 # No drain if there is already a drain in a layer above
    # Convert this idrn boolean array to a list of cells with a drain
    LIc_DRN = grv.I2LIc(grv.NOD[isdrn > 0], astuples=True) # The Layer-Icell cell index pair
    #inactDRN = grv.inactive(LIc_DRN, idomain=idomain)[0]
    drn_elev = grv.Z[:-1][isdrn > 0]  # No drain if already a drain above this cell
    c_dr = 0.4 # d, # drain resistance (~ arbitrary, think of it as Q = (h - elev) * Conductance)
    Conductance = (grv.Area[np.newaxis, :, :] * isdrn / c_dr)[isdrn > 0]
    drn_spd =[((lic), c, elev) for lic, c, elev in zip(LIc_DRN, drn_elev, Conductance)]
    model_dict['drn'] = {'stress_period_data' :{0: drn_spd}, 'maxbound' : len(drn_spd)}

# ghb (general head boundaries)
LIc_GHB = grv.LRC2LIc(model_dict['ghb'][0], bc=True)
#inactGHB = grv.LIc2LRC(grv.inactive(LIc_GHB, idomain=idomain)[0])
model_dict['ghb'] = {'stress_period_data' : {0: LIc_GHB}, 'maxbound': len(LIc_GHB)}

# riv
LIc_RIV = grv.LRC2LIc(model_dict['riv'][0], bc=True)
model_dict['ghb'] = {'stress_period_data' : {0: LIc_GHB}, 'maxbound': len(LIc_GHB)}

riv = [[(0, 2, 3), 10., 1.2, 0.2], [[0, 3, 4], 10., 1.2, 0.2]]
model_dict['riv'] = {'stress_period_data': riv, 'maxbound' : len(LIc_RIV)}

# wel
LIc_WEL = grv.LRC2LIc(model_dict['wel'][0], bc=True)
model_dict['wel'] = {'stress_period_data': wel, 'maxbound': len(LIc_WEL)}

# rcha (recharge time series)
rcha = {}
model_dict['rcha'] = rcha

# evta (evaporation time series)
evta = {}
model_dict['evta'] = {}

model_dict['oc']  = {'head_filerecord' :   "{}.hds".format(model_dict['sim']['sim_name']),
                     'budget_filerecord' : "{}.cbc".format(model_dict['sim']['sim_name']),
                     'saverecord': [("HEAD", "ALL"), ("BUDGET", "ALL")]}


class Mf6model():
    """Modflow6 model class, using VDIS (layer icell) grid but keeping the familiar Structure
    that is the redular rectangular shape in the horizonal plane for ease of handling data and
    grid locations as well. However, the nodes do not have to lie on straight lines.
    """
    
    def __init__(self, model_dict=None, params_wbk=None, use_packages=None):
        """Return an MF6model object.
        
        Parameters
        ----------
        model_dict: dict
            the non-detault, case specific, data that make up the mf6 model
            keys are the package and the values are a dict of which the items are
            parameter name, parameter values pairs
            
            The model_dict is used to update the default parameters dict,
            which is read from an excel workbook.
            
        params_wbk: path
            Excel workbook file name with the mf6 default parameters in sheet `mf6`
        use_packages = set or list of the package names to be used in the simulation, e.g.:
            ['sim', 'ic', 'tdis', 'gwf', 'disv', 'sto', 'npf', 'oc', 'chd', 'wel', 'drn', 'ghb',
                    'rcha', 'ims']
        """
        
        # === read the default params from the parameter workbook, sheet_name = 'MF6' ===
        params = get_params_from_excel(params_wbk)

        # === Keep and update sets of used and unused package names
        self.use_packages  = set(use_packages) # the package names to be used as a set
        self.used_packages = set() # The package names actually used as a set
    
        # === Filter out the params for the packages to be used
        params = {key: params[key] for key in params if key in use_packages}
        
        # === The new disv mf6 model grid
        # (however the old 3d structure :nlay, nrow, ncol, is always used)
        grv = model_dict['disv']['grv']
        
        # === Update the default parameters with the case specifi ones for each of the packages
        for pkg in model_dict:
            self.used_packages.add(pkg)
            self.use_packages.remove(pkg)
            params[pkg].update(**model_dict['pkg'])
            
        # === Store grid info  and the parameters in this model object
        self._grv = grv
        self.params = params
        
        # === Inform user which packages are not used or not implemented
        self.unused          = use_packages.difference(self.used_packages)
        self.not_implemented = set(params.keys()).difference(self.used_packages)
        
        logging.info(f"Unused packages: {self.unused}")
        logging.info(f"not implemented but requested packages {self.not_implemented}")
        
        if self.unused:
            print(f"Available packages but not used at this time: {self.unused}")
        if self.not_implemented:
            print(f"Requested but not implemented packages: {self.not_implemented}")
        
        return
    
    @property
    def grv(self):
        """Return vdis grid object."""
        return self._grv
    
    def build(self):
        self.packages = build_mf6model(used_packages=self.used_packages, params=self.params)
        return
    
    def write(self, silent=True):
        write_model(packages=self.packages, silent=silent)
        return

    def run(self, silent=True):
        success = run_model(packages=self.packages, silent=silent)
        return success
    
    def simulate(self, silent=True):
        """Build, write, run model and plot results all at once.

        Parameters
        ----------
        silent: bool
            whether or not plot information bfore end of this run.
        """
        packages = self.build_mf6model()
        self.write(silent=silent)
        success = self.runl(silent=silent)
        return success
        
    def plot_cross_sections(self, mf6_dict, rows=None,
                    HDS=None, iper=None, figsize=(12,6)):
        axs = plot_cross_sections(mf6_dict, rows=rows,
                    HDS=HDS, iper=iper, figsize=figsize)
        return axs
        
        
class MF5_model:
    """Class for a Modflow 5 type of modelrun by mf6 (structured grid).
    
    Model object type Modflow 5, constracted from the reactored data
    from the original Yasin (1999) model, which was run in Modflow version 1998.
    
    The refactoring consitst of constructing a fullly 3D model from the original quasi-3D model.

    The number of layers will be still be 4. Most of the inputs are actually the same
    for the new and the old model. The only exception are the vertical conductivities.
    The original model did not have vertical conductivities, because it used a quasi-3D approach
    with VCONT between the aquifers. However, the diretory with the original files also has
    a file 'trial2.lea' which, according to Chiang & Kinzelbach (2003) is a file with vertical
    conductivities. I checked that these vertical conductivities are fully compatible
    with the VCONT of the quasi-3D model. Therefre, these vertical conductivities
    can be used instead of the VCONT for the new fully 3D model. This makes sure the
    vertical conductivities of the new fully 3D model's are uniquely defined, which
    is impossible if only the VCONT were given as VCONT combines both the vertical
    conductivties and the thicknesses of the two overlying and underlyaing aquifers.
    """
    
    def __init__(yasin_old, steady=False, tif_path=None, chd_dead_sea_coast=None, drains_present=False, strategy=3, nstp=3, tsmult=1.5):
        """Return an MF5_model object constucted from the old Yasin (1999) mf 98 model.
    
        Parameters
        ----------
        yasin_old: dict
            the data extraced from the original Yasin 1999 data files.
        steady: boolean 
            weather the model is steady or transient (using the long-term monthly data from satellites)
        tif_path: str or None
            Full path to the tif hold in the ground surface elevation. The tif must contain the area
            area of interest given by gr.x and gr.y and be in new Israel-Palestine coordinates.
            If None, the top laye of the grid is not replaced.
        chd_dead_sea_coast: float or None
            The level of the Dead Sea to be used. If None, no direct connection to the
            Dead Sea is used, exactly as it was in the original 1999 Yasin model.
            Dead Sea Levels:
                None   : don't use Dead Sea
                -380   : (1950)
                -411   : (1999)
                -430.5 : (Recent, Wikipedia)
            Set hDeadSea to one of the concrete values to use the Dead Sea as boundary.
        drains_present: bool
            Whether drains on ground surface are required or not.
        strategy: int, oneof 1, 2, 3
            Boundary-head correction strategy to correct for head elevation not within specified layer in old model
            1): raise h to above the bottom of the cell of the current, specified layer.
                This assumes cell was right, but specified h was wrong.
            2): remove the record altogether (assuming situation is impossible)
            3): transfer the boundary condition to highst cell with bottom below h.
                This assumes that the h was right, but cell layer was wrong.
        nstp: int 
            number of time steps in each stress period (used only when simulation is transient)
        tsmult: float
            multiplier for length of successive time steps  (used only when simlation is transient)
        """
        
        self.model = yasin_old
        
        mf6_dict = 'mf6_dict'
        logging.warning("Constructing the new model using function `{}`".format(etc.whoIsCalling()))
        # The original model data
        gr_old = yasin_old['bcf']['gr']

        ibound = yasin_old['bas']['ibound']

        logging.warning("Using dem as new roof on `{}'.".format(mf6_dict))
        gr = get_new_roof_of_model(gr_old, tif_path=tif_path, plot=False, ibound=ibound)    
    
        # Initialize the new model arrays
        K1     = yasin_old['lpf']['HK']
        K3     = yasin_old['lpf']['VK']
        strt   = yasin_old['bas']['strt']
        
        # This still has the old elevation for the top of the model (data=Z)

        # %% Using mf6 idomain
        idomain = np.ones_like(ibound, dtype=int) # start with all cells active
        idomain[ibound == 0] = 0 # only ibound == 0 are inactive cells in mf6
        idomain[gr.DZ <= gr.min_dz] = -1  # Inactive, vertically flow-through cell
        #idomain[np.logical_and(idomain != 0, gr.DZ <= gr.min_dz)] = -1  # Inactive, vertically flow-through cell
        # Special: Don't allow inactive cells in layer 3 while layer 2 is active
        idomain[-1][np.logical_and(idomain[-2] > 0, idomain[-1] <= 0)] = 1


        # The dict that will hold the data for the new modflow 6 model (by recactoring the yasin_old model)
        # In fact, anything of our liking can be put into this dict as well, for instance the "formations" 
        # that will be use in drawing cross sections later on.
        mf6_dict = dict()
        bas = yasin_old['bas']

        # Formations defined here to allow drawing them in the cross sections.
        mf6_dict['formations'] = {
            0: {'name': 'Sononian', 'color': 'magenta'},
            1: {'name': 'Upper Cenomanien', 'color': 'yellow'},
            2: {'name': 'Yatta', 'color': 'darkblue'},
            3: {'name': 'Lower Cenomanien', 'color': 'green'}
            }


        # input for mfsim.nam
        # Here we need to set the transient
        mf6_dict['tdis'] = {'time_units': 'days',
                            'nper': len(yasin_old['tdis']['perioddata']),
                            'perioddata': yasin_old['tdis']['perioddata'],
                            'start_datetime': '2021-05-22 18:00',
                            }

        if steady:        
            mf6_dict['sto']  = {'storagecoefficient': False,
                                
                            'iconvert':  yasin_old['sto']['iconvert'],
                            'ss':        yasin_old['sto']['ss'],
                            'sy':        yasin_old['sto']['sy'],
                            'steady_state': np.ones (mf6_dict['tdis']['nper'], dtype=bool),
                            'transient':    np.zeros(mf6_dict['tdis']['nper'], dtype=bool),
                            }
        else:
            lta = {} # Long term average (for first steady state stress period.)
            stress_period_data = {}
            
            # See rch further down
            lta['rch'] = modelTiffsLib.get_lta_array(folders=folders, key='rch')        
            stress_period_data['rch'] = modelTiffsLib.getBoundArrayFromTifs(folders=folders, key='rch')
            
            # See wel further down
            lta['sup'] = modelTiffsLib.get_lta_array(folders=folders, key='sup')
            stress_period_data['sup'] = modelTiffsLib.getBoundArrayFromTifs(folders=folders, key='sup')
            
            # Generate a new key equal to a year before the first month.
            key1 = sorted(stress_period_data['rch'].keys())[0]
            y, m, d = re.compile(r"\d+").findall(key1)
            key0 = "{:4d}-{}-{}".format(int(y)-1, m, d) # One year before first month
            perlen = (np.datetime64(key1) - np.datetime64(key0)) / np.timedelta64(1, 'D')
            lta['rch']['perlen'] = perlen
            lta['sup']['perlen'] = perlen
            stress_period_data['rch'][key0] = lta['rch'] # Place lta here
            stress_period_data['sup'][key0] = lta['rch'] # Place lta here

            # Notice that the first SP will be steady to initialyze
            
            # Overwrite steady-state values from old model:
            mf6_dict['tdis']['PERLEN'] = [stress_period_data['rch'][k]['perlen'] for k in stress_period_data['rch']]
            mf6_dict['tdis']['nper']   = len(stress_period_data['sup'])
            mf6_dict['tdis']['NSTP']   = np.ones(mf6_dict['tdis']['nper'], dtype=int) * nstp
            mf6_dict['tdis']['TSMULT'] = np.ones(mf6_dict['tdis']['nper']) * tsmult
            
            mf6_dict['tdis']['start_datetime'] = sorted(stress_period_data['sup'].keys())[0]

            steady_state = np.zeros(mf6_dict['tdis']['nper'], dtype=bool)
            transient    = np.ones( mf6_dict['tdis']['nper'], dtype=bool)
            
            steady_state[0] = True  # First SP steady state
            transient[0]    = False # First SP not transient
            
            mf6_dict['sto']  = {'storagecoefficient': False,
                        'iconvert':  idomain >0, # exclude -1 (flow-through, inactive) and 0  (inactive)
                        'ss': 1e-5,  # should be a 3D array
                        'sy': 0.15,  # should be a 3D array
                        'steady_state': steady_state,
                        'transient':    transient,
                    }

        mf6_dict['dis'] = {'length_units': 'meters',
                            'gr': gr,
                            'proj': yasin_old['bcf']['proj'],
                            'idomain': idomain,
                            'ibound': ibound, # comes in handy, don't stricktly need it.
                            }
        mf6_dict['npf'] = {'k1': K1,
                            'k3': K3,
                            'perched': True,
                            'icelltype': 0, # 'icelltype': 1
                            }
        mf6_dict['ic'] = {'strt': strt}
        
        mf6_dict['boundHeadCorrStrategy'] = strategy # 1, 2, 3 

        ghb_spd = raise_hd_above_bottom(yasin_old['ghb'], gr=gr, idomain=idomain, strategy=strategy)

    
        # CHD
        chd_spd = yasin_old['chd']
        dead_sea_present =  chd_dead_sea_coast is not None   
        if  dead_sea_present:
            chd_spd = combine_chd_spd_with_dead_sea(chd_spd, chd_dead_sea_coast, idomain=idomain)
        chd_spd = raise_hd_above_bottom(chd_spd, gr=gr, idomain=idomain, strategy=strategy)

        mf6_dict['ghb'] = {'stress_period_data': ghb_spd}
        mf6_dict['chd'] = {'stress_period_data': chd_spd,
                            'chd_dead_sea_coast': chd_dead_sea_coast,
                            'dead_sea_present' : dead_sea_present,
                            }

        # ////// TRANSIENT /// --> only for rch and wel:
        # //// RCH_AND_WEL package ///
        if steady: # if steady-state:
            mf6_dict['rch'] = {'stress_period_data': yasin_old['rch']['RECH'],
                                'irch':get_irch(idomain, zero_based=True)}
            well_spd = put_well_in_proper_layer(yasin_old['wel'], gr=gr)
        else:
            rch_spd = modelTiffsLib.get_rch_spd(stress_period_data['rch'])
            mf6_dict['rch'] = {'stress_period_data': rch_spd,
                                'irch':get_irch(idomain, zero_based=True)}
            well_spd = modelTiffsLib.get_spd_as_recarray_for_active_bottom_cells(stress_period_data['sup'],
                                                        gr=gr, ibound=ibound)
            mf6_dict['wel'] = {'stress_period_data': well_spd}

        # Surface runoff conductance per m, hard_wired
        mf6_dict['drn'] = {'present': drains_present, 'conductance' : 10.}

        if drains_present:
            drn_spd = get_drains(gr=gr, idomain=idomain)
            mf6_dict['drn'] = {'stress_period_data': drn_spd,
                                'conductance': 10,
                            }

        # side-effect: pickle the new model
        with open(os.path.join(folders['test'], 'mf6_dict.pickle'), 'wb') as fp:
            logging.warning("Dumping (pickle) the new model `{}` in folders[{}]".format(mf6_dict, 'test'))
            pickle.dump(mf6_dict, fp)
        logging.warning("returning `{}`".format(mf6_dict))
        return mf6_dict

# %% Define how to plot hydraulic boundary cells

def plot_bounds(mf6_dict, iper=-1, layer=0, type='wel', ax=None, **kwargs):
    """Plot the boundaries of given type on map with ax.
    
    Parameters
    ----------
    mf6_dict: dict with the data
        The data that construct the mf6 model.
    blist: list of boundary cells [(j, j, i), h, ...), (...), ...]
        the cells to be plotted
    layer: int
        zero-base model layer number
    type: str
        boundary package name
    ax: plt.axes
        axis on which to plot
    kwargs dictionary
        passed on to ax.plot(..)
    """
    from etc import plot_kw, text_kw

    plt_kw = set(kwargs.keys()).intersection(plot_kw.keys())
    txt_kw = set(kwargs.keys()).intersection(text_kw.keys())
    pkwargs = {k:kwargs[k] for k in plt_kw}
    tkwargs = {k:kwargs[k] for k in txt_kw}

    gr = mf6_dict['dis']['gr']

    dtypes = {
        'chd': np.dtype([('cellid', 'O'), ('head', '<f8')]),
        'wel': np.dtype([('cellid', 'O'), ('q', '<f8')]),
        'drn': np.dtype([('cellid', 'O'), ('elev', '<f8'),
                                ('cond', '<f8')]),
        'ghb': np.dtype([('cellid', 'O'), ('bhead', '<f8'),
                                ('cond', '<f8')]),
        'riv': np.dtype([('cellid', 'O'), ('stage', '<f8'),
                        ('cond', '<f8'), ('zbot', '<f8')])}
        
    assert type in dtypes, "type {} not in [{}]".format(dtypes.keys())

    try:
        # bounds is converted to a recarray of given dtype
        blist = mf6_dict[type]['stress_period_data'][iper]
    except:
        warnings.warn("Model may not have boundaries of type {} "
                .format(type))
        return
    try:
        bounds = np.array(blist, dtype=dtypes[type])
    except:
        # TODO --> not compatible with transient data
        raise ValueError('Boundary cell list may not be ' +
                'compatible with the dtype for the given package.')

        btype = bounds.dtype

    # Always print the head, flow or elevation, the name of which is
    pname = bounds.dtype.names[1] # parameter name from recarray fields

    for cellid, pvalue in zip(bounds['cellid'], bounds[pname]):
        k, j, i = cellid        
        if k  == layer:
            ax.plot(gr.xm[i], gr.ym[j], **pkwargs)
            ax.text(gr.xm[i], gr.ym[j],
                '_{}={:.0f}'.format(type, pvalue), **tkwargs)
    return bounds


def get_irch(idomain, zero_based=True):
    """Return an array (ny, nx), i.e. irch, with the layer number of the top active cells.
    
    Also useful to place drains on ground surface (in first active vertical cell).

    Irch using in RCH package is the array telling in which layer the recharge will work. This is the top most layer in a column that is active.

    According to flopy.mfrch source code docstring, layer must be zero-based that
    implies top model layer has zero as layer number when the cell in the top layer is
    inactive or when it is the first active cell. This, however is tricky to implement.
    The easiest way is to assume irch is one-based, then the top layer having zero
    means inactive. Afterwards, we can subtract 1 from all cells with value > 0.

    If you want the top layer which is active just use zero_based = False.
    The the array will have 0 for inactive, 1 for cells that are active in the
    top layer, 2 for cells that are active in the second layer but not in the first,
    3 for cells that are active in the 3rd layer but not in the first and second, etc.
    This can be used to determine the cells that are actually at ground surface, which
    is what irch does express.

    Parameters
    ----------
    idomain : ndarray (nlay, nrow, ncol)
        mf6 array telling wheather a cell is active [>0] or inactive [0] or
        a vertical flow-through cell [-1].
    zero_based : bool
        if True first layer has index 0 (flopy), else 1 (modflow)

    @TO 20210630
    """
    irch = np.zeros_like(idomain[0])
    for iz in range(len(idomain)):     
        laynr = iz + 1
        irch[np.logical_and(irch == 0, idomain[iz] > 0)] = laynr
    if zero_based:
        irch[irch > 0] -= 1
    return irch


def put_well_in_proper_layer(well_data, gr=None):
    """Make sure the well is in a sufficient thick active layer.s

    Parameters
    ----------
    well_data: dict
        stress-period indexed stress period data for well extraction
    gr: tools.fdm.Grid object
        Structured Modflow Grid
    """
    for iper in well_data.keys():
        new_list = [] # rectified boundary condion list
        for rec in well_data[iper]: # rec must be (k, j, i, h, ...)
            # Put the well in a thicker lower layerse if it exists
            (k, j, i), q = rec # note k, j, i are zero based
            while k < gr.nz and gr.DZ[k, j, i] <= gr.min_dz:
                k += 1
                rec = (k, j, i), q
            new_list.append(rec) # Store this rec (always succeeds)
        well_data[iper] = new_list

    return well_data


def raise_hd_above_bottom(sp_data, gr=None, dz=0.1,
                                idomain=None, strategy=3):
    """Raise head/elev in stress period data to at least dz above layer bottom, in place.
    
    Essential to ensure boundary pakage heads are valid to convertible modflow6 cells.

    The idea is that the j, j, i are correct, but then we cannot have a head below
    the bottom of the layer for which the condition is defined.
    It is not clear which of these strategies is correct.

    Parameters
    ----------
    sp_data: dict with keys sp number and list of cell records [(k, j, i, h, ...), ...]
        Stress period data.
    stratagy: oneof 1, 2, 3
        1): raise h to above the bottom of the cell of the current, specified layer.
            This assumes cell was right, but specified h was wrong.
        2): remove the record altogether (assuming situation is impossible)
        3): transfer the boundary condition to highst cell with bottom below h.
            This assumes that the h was right, but cell layer was wrong.
    gr: fdm.Grid object
        The modflow Grid (block shape, regular)
    idomain: int array of size gr.shape
        tells wether a cell is active (>0) inactive (<0) and or
        flow=through (-1) which is also inactive
    dz: float > 0
        The minimum elevation of head above bottom of current layer.
    """
    active = idomain > 0
    for iper in sp_data.keys():
        new_list = [] # rectified boundary condion list
        for rec in sp_data[iper]: # rec must be (k, j, i, h, ...)
            (k, j, i), h, *rest = rec # note k, j, i are zero based
            if strategy == 1: # Raise head or elev to dz above bottom of layer
                # rest[0] is the first value of rest, i.e. the head (or flow if wel)
                # Adapt the head, it must be above the bottom of the given cell.
                h = max(h, gr.Zbot[k, j, i] + dz)
                # Generate a new record with adapted head
                rec = (k, j, i), h, *rest
                new_list.append(rec) # Store this rec (always succeeds)
            elif strategy == 2: # Remove record
                if h >= gr.Zbot[k, j, i]:
                    new_list.append(rec) # Success, store this rec
                else:
                    pass # No success, ignore this rec
            elif strategy == 3:
                # Change k to that of the cell with bottom below h,
                # which must also be active
                # This assumes that h < Zbot(nlay, j, i) !!
                for kk in range(k, len(gr.Zbot)):
                    if np.logical_and(
                        gr.Zbot[kk, j, i] < h, active[kk, j, i]):
                        rec = (kk, j, i), h, *rest
                        new_list.append(rec) # Store this rec
                        break # Found, ready
                    else:
                        pass # try next deeper cell,
                             # if deepest, ignore rec
            else:
                raise ValueError("strategy must be oneof [1, 2, 3].")
        sp_data[iper] = new_list

    return sp_data

# %% Simulation

def get_modflow_grid(gr_old, gr_surface_tif=None): # Obsolete
    """Get the modflow grid for the mf6_dict model

    The old grid see yasin_old['bcf']['gr'] already has the x and y coordinates in EPGS6984 projection. However, its top layer elevation is still arbitrary;
    it was set aribtrrily to 100 m above the base of the  first layer.
    Therefore, the true ground elevation has to be used as the roof of the
    modflow grid, i.e. as top of the first layer.

    The elevation (dem) data are in ../mdl_elev.tif

    Parameters
    ----------
    gr_old: tools.fdm.Grid object
        Blockshaped modflow grid
    """
    # Check layer thickness
    def spy_layer_thickness(gr_old, gr_new):
        """Shee where ground surface is lower than the top of the second layer (bottom of layer 0)"""
        DZold = gr_old.DZ
        DZnew = gr_new.DZ

        inactive = yasin_old['bas']['ibound'] == 0

        dz = (gr_new.Z[0][np.newaxis, :, :] - gr_old.Z[1:]).reshape(gr_new.nlay * gr_new.nrow, gr_new.ncol)
        dv = 10
        vmin, vmax = np.floor(dz.min()  / dv) * dv, np.ceil(dz.max() / dv) * dv

        f0 = DZold.reshape(gr_old.nlay * gr_old.nrow, gr_old.ncol) >= gr_old.min_dz
        f1 = DZnew.reshape(gr_new.nlay * gr_new.nrow, gr_new.ncol) >= gr_new.min_dz
        f0 = dz > 0
        f1 = dz < 0

        fig, axs = plt.subplots(1, 3)
        fig.set_size_inches(12, 10)

        for ax,title, xlabel, ylabel, what in zip(axs, ['dz > 0', 'dz < 0', 'dz'], ['ixm', 'ixm', 'x[m]'], ['iym', 'iym', 'z'], [f0, f1, None]):
            ax.grid()
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_aspect(1.0)
            if what is not None:
                ax.spy(what)
            else:
                #c = ax.imshow(dz, extent=gr_new.extent, vmin=vmin, vmax=vmax, cmap='viridis')
                #c = ax.imshow(dz, extent=gr_new.extent, vmin=-500, vmax=500, cmap='viridis')
                c = ax.imshow(dz, extent=gr_new.extent, vmin=-500, vmax=500, cmap='Accent')
                #c = ax.contourf(dz, levels=np.linspace(-500, 500, 21), cmap='viridis')
                #ax.contour(dz, levels=np.linspace(-500, 500, 21), colors='k', linewidths=0.25)
                ax.set_aspect(4.)
                plt.colorbar(c)

        fig, axs = plt.subplots(1, 7, sharex=True, sharey=True)
        fig.set_size_inches(12, 4)
        rows = [20, 30 , 40, 50, 60, 60, 80]

        Ztop = gr_new.Ztop
        Zbot = np.ma.masked_array(gr_old.Zbot, mask=inactive)

        for ax,row in zip(axs, rows): 
            ax.grid()
            ax.set_title('row {}'.format(row))
            ax.set_xlabel('x[m]')
            ax.set_ylabel('y[m')
            ax.plot(gr_new.xm, Ztop[0, row], 'b-', label='top0')
            ax.plot(gr_new.xm, Zbot[1, row], 'r-', label='bot1')
            ax.plot(gr_new.xm, Zbot[2, row], 'g-', label='bot2')
            ax.plot(gr_new.xm, Zbot[3, row], 'k-', label='bot3')
            ax.legend()

        fig, axs = plt.subplots(1, 4, sharex=True, sharey=True)
        fig.set_size_inches(12, 4)
        cols = [10, 20, 30 , 40]

        for ax,col in zip(axs, cols): 
            ax.grid()
            ax.set_title('col {}'.format(col))
            ax.set_xlabel('x[m]')
            ax.set_ylabel('y[m')
            ax.plot(gr_new.ym, Ztop[0, :, col], 'b-', label='top0')
            ax.plot(gr_new.ym, Zbot[1, :, col], 'r-', label='bot1')
            ax.plot(gr_new.ym, Zbot[2, :, col], 'g-', label='bot2')
            ax.plot(gr_new.ym, Zbot[3, :, col], 'k-', label='bot3')
            ax.legend()

        plt.show(block=True)

        return None

    # New: Get the elevations from the tif file mdl_elec.tif
    with rasterio.open(gr_surface_tif) as elev_data:
        ztop = elev_data.read(1)
        xm = [(elev_data.transform * (i, 0))[0] for i in range(elev_data.width)]
        ym = [(elev_data.transform * (0, i))[1] for i in range(elev_data.height)]
        assert np.all(gr_old.shape[1:] == ztop.shape), "gr_old.shape = ({},{}) !=\
            ztop. shape ({},{})".format(*gr_old.shape, *ztop.shape
        )
        assert np.all(np.isclose(gr_old.xm, xm)), "gr_old.x not close to x"
        assert np.all(np.isclose(gr_old.ym, ym)), "gr_old.y not close to y"
        Z = gr_old.Z.copy()
        Z[0, :, :] = ztop[:, :]
        gr_new = mfgrid.Grid(gr_old.x, gr_old.y, Z, min_dz = 1.0)
        gr_new.crs = elev_data.crs

    spy_layer_thickness(gr_old, gr_new)

    return gr_new

def set_chd_dead_sea(mf6_dict, gr=None, idomain=None):
    """Return chd cells where model touches the Dead Sea in deepest model layer.

    Also the Dead Sea cells will be set to active in idomain deepest layer.
    
    Dedicated function for this model and model grid.

    Parameters
    ----------
    mf6_dict: dict
        The data for the the model as a dict.
    hDeadSea: float
        level of Dead Sea (with respect to datum or MSL)
            None (for no Dead Sea cells)
            -380 # (1950)
            -411 # (1999)
            -430.5 # (Recent, Wikipedia)
    gr: fdm.Grid object
        the model grid
    idomain: ndarray (nz, ny, nx) of int
        tells which cells are active or inactive
        will be updated
    """
    if mf6_dict['chd']['dead_sea_present'] == False:
        return None

    xDS_west = 220000
    yDS_south, yDS_north = 603000, 630000
    # The range along which the Dead Sea touches the model (y coords)
    dead_sea_touches_model = np.logical_and(gr.ym > yDS_south, gr.ym < yDS_north)

    # To find the right most active cells in the lowest layer, which actually
    # touches the Dea Sea between y = 603000 and 630000, use a copy of domain[-1]
    # and make all its cells west of Xm_west equal to 1 (the right cells are
    # zero becaus inactive. Then sum the rows to get first inactive column number
    # at the y-range where the model touches the Dead Sea.
    ido = idomain[-1].copy() # Only the lowest aquifer touches the Dead Sea
    ido[gr.Xm < xDS_west] = 1
    icol = (np.sum(ido, axis=1) - 1)[dead_sea_touches_model]

    # for rows simply take all rows and select those where dead_sea_touches_model
    irow = np.arange(gr.ny)[dead_sea_touches_model]

    chd_dead_sea_coast = [((gr.nz - 1, irow[i], icol[i]), hDeadSea) for i in range(len(icol))]

    return chd_dead_sea_coast

def get_drains(gr=None, idomain=None):
    """Return drn_spd for drains on ground surface.
    
    Surface runoff is computed using the DRN package, it is the amount
    of water that can't infiltrate and will thus be drained from ground
    surface. We just put drains on top of the model in the top most
    active cells.
    
    In reality, the head stays way below ground surface,
    so DRN is not needed. But, in case the vertical resistance is
    too large, the head will rise far above ground surface.
    """

    # Top active cells
    irch1 = get_irch(idomain, zero_based=False)
    
    # Get the indices of the highest active cells, i.e. of ground surface
    Idr = gr.NOD[0][irch1 > 0] # top layer cells, omitting inactive columns
    lrc = gr.LRC(Idr) 

    # Replace layer number of that of first active cell in vertical column,
    # but now zero based (subtract 1)
    lrc.T[0] = irch1[irch1 > 0] - 1 # return to zero-based

    # Use ground surface as drain elevation (Z[0] covers entire area).
    hdr = gr.Z[0].ravel()[Idr]

    # Use some easily draining conductance for all drains, to capture
    # water that can't infiltrate or seeps out.
    # Notice:  # 1000 m3/d (1 mm/d) for 1 km2 discharges as 1 m head difference.
    Cdr = np.ones(len(Idr)) * mf6_dict['drn']['conductance']

    # Set up the drains input array for mf6, we use a record array 
    dtype = np.dtype([('cellid', 'O'), ('elev', '<f8'), ('cond', '<f8')])
    drn_spd = np.zeros(len(Idr), dtype=dtype)
    drn_spd['cellid'] = np.asarray(lrc, dtype=[('cellid', 'O')])
    drn_spd['elev'] = hdr
    drn_spd['cond'] = Cdr

    # Turn into list, because flopy does not accept this perfectly legal record array.
    drn_spd = {0: [(tuple(p[0]), p[1], p[2]) for p in drn_spd]}
    return drn_spd

def combine_chd_spd_with_dead_sea(chd_spd, chd_list, idomain=None):
    """Combine stress period data dict with a list of sp records.
    
    Parameters
    ----------
    sp_data: dict keyed with zero-base sp numbers.
        stess period data
    reclist: list with with each item a cell record (k, j, i, ...)
        the permanent records to be added to each stress period
    idomain: ndarray of ints
        integer denoting cell type (active, inactive or flow-through)
        idomain is changed in place.
    """
    for iper in chd_spd:
        rec_list = chd_spd[iper]
        cells = {rec[0] for rec in rec_list}  # set
        for rec in chd_list:
            k, j, i = rec[0]
            if not (k, j, i) in cells:
                rec_list.append(rec)
            else: # replace head with that of rec, i.e. rec[1]
                idx = [ii for ii, x in enumerate(rec_list) if x[0]==(k, j, i)]
                for ii in idx: # There should be only 1 idx, but nevermind
                    rec_list[ii] = rec
    return chd_spd


def set_up_modflow_sim(mf6_dict, simname=None, folders=None, USEWELLS=False, well_reduction_factor = 1.0):
    """Set_up the data for the modflow mf6 simulation.
    
    Parameters
    ----------
    mf6_dict: dict of
        model data
    simname: str
        basename of simulation
    src_dir: str
        path to directory to put the simulation files. 
    USEWELLS: boolean 
        wheather or not use any wells in the model 
    well_reduction_factor: float 
        factor to multiply well flow with as a debug feature, to make sure wells do not completely dryout the aquifer. 
        TODO Make sure that well flows are automatically reduces of the aquifer may run dry.
    """
    sim_src_dir  = os.path.join(folders['src_python'], simname) # working directory simulation
    exe_path = os.path.join(folders['exe'], 'mf6.mac')
    version = 'mf6'
    name = 'mf6_dict'

    # if the sim directory does not yet exist, make it.
    if not os.path.isdir(sim_src_dir):
        os.mkdir(sim_src_dir)

    os.chdir(sim_src_dir)

    gr = mf6_dict['dis']['gr']

    idomain = mf6_dict['dis']['idomain']

    # Layer properties:
    # There is too much vertical resistance in the aquifers if k33 == k;
    # all vertical resistance in the original Yasin model was in VCONT,
    # which are separate layers in the new model with their own vertical
    # conductivity has to match the original VCONT. So we have to
    # nullify (almost) the vertical resistance within the aquifers to
    # match with the original 1999 model.
 
    # Oc output control
    headfile    = "{}.hds".format(name)
    budgetfile  = "{}.cbb".format(name)
    saverecord  = [("HEAD", "LAST"), ("BUDGET", "LAST")]
    printrecord = ("HEAD", "LAST"),

    # IMS (interative ... solver)
    ims_kwargs = dict(
        print_option='ALL', # 'SUMMARY',
        complexity='COMPLEX',
        csv_outer_output_filerecord='ims_outer.csv',
        csv_inner_output_filerecord='ims_inner.csv',
        no_ptcrecord='FIRST', # 'ALL',  # or FIRST
        outer_maximum=200,
        outer_dvclose=0.1,
        under_relaxation='DBD',
        under_relaxation_theta=0.5,
        under_relaxation_kappa=0.3,
        under_relaxation_gamma=0.3,
        under_relaxation_momentum=0.001,
        backtracking_number=20,
        backtracking_tolerance=1.05,
        backtracking_reduction_factor=0.2,
        backtracking_residual_limit=10,

        inner_maximum=100,
        inner_dvclose=0.0001,
        rcloserecord=[10., 'STRICT'],
        linear_acceleration='BICGSTAB',
        relaxation_factor=0.97,  # may be zero (default)
        preconditioner_levels=5,
        preconditioner_drop_tolerance=1e-4,
        number_orthogonalizations=0,  # default = 0
        scaling_method='NONE', # 'L2NORM', # 'POLCG', # 'DIAGONAL', # 'NONE',
        reordering_method='NONE', # 'MD', # 'RCM', # 'NONE',
    )

    # Adding the packages this should normally stay untouched for any model
    # Except for which packagres to add. All the data are specified above.
    sim = flopy.mf6.MFSimulation(sim_name=name, exe_name=exe_path, version=version)

    td = mf6_dict['tdis']
    tdis = flopy.mf6.ModflowTdis(sim, time_units=td['time_units'],
                                start_date_time=td['start_datetime'],
                                nper=td['nper'],
                                perioddata=[
            [perlen, nstep, tsmult] for perlen, nstep, tsmult in zip(td['PERLEN'], td['NSTP'], td['TSMULT'])
            ])

    # Solver
    ims = flopy.mf6.ModflowIms(sim, pname='ims', **ims_kwargs)

    # Add the groundwater flow model
    gwf = flopy.mf6.ModflowGwf(sim, modelname=name, model_nam_file="{}.nam".format(name),
                            version=version, exe_name=exe_path, newtonoptions='under_relaxation',
                            )

    # Dis package
    dis = flopy.mf6.ModflowGwfdis(gwf,
                                nlay=gr.nlay, nrow=gr.nrow, ncol=gr.ncol, delr=gr.dx, delc=gr.dy, idomain=idomain,
                                top=gr.Ztop[0], botm=gr.Zbot)

    # Storage
    # sto = flopy.mf6.ModflowGwfsto(gwf, storagecoefficient=True, iconvert=1,
    #                              ss=ss, sy=sy, transient=transient,
    #                              save_flows=True
    #                              )
    # Initial conditions
    # Initial conditions
    strt = mf6_dict['ic']['strt']
    ic = flopy.mf6.ModflowGwfic(gwf, strt=strt)

    # Layer rproperties
    npf = flopy.mf6.ModflowGwfnpf(gwf,
                            k   = mf6_dict['npf']['k1'],
                            k33 = mf6_dict['npf']['k3'],
                            icelltype = mf6_dict['npf']['icelltype'],
                            save_flows=True,
                            )

    sto = flopy.mf6.ModflowGwfsto(gwf, save_flows=True,
                        storagecoefficient=False,
                        iconvert=mf6_dict['sto']['iconvert'],  # per cell
                        ss=mf6_dict['sto']['ss'],              # per cell
                        sy=mf6_dict['sto']['sy'],              # per cell
                        steady_state=mf6_dict['sto']['steady_state'], # per stress period
                        transient=mf6_dict['sto']['transient'], # per stress period
                        )

    
    maxbound = lambda data: np.max(
                np.array([len(data[k]) for k in data], dtype=int))
    
    chd_spd = mf6_dict['chd']['stress_period_data']
    chd = flopy.mf6.ModflowGwfchd(gwf,
            maxbound=maxbound(chd_spd),
            stress_period_data=chd_spd,
            save_flows=True
            )

    ghb_spd = mf6_dict['ghb']['stress_period_data']
    ghb = flopy.mf6.ModflowGwfghb(gwf,
            maxbound=maxbound(ghb_spd), stress_period_data=ghb_spd, save_flows=True)

    if mf6_dict['drn']['present']:
        drn_spd = mf6_dict['drn']['stress_period_data']
        drn = flopy.mf6.ModflowGwfdrn(gwf,
            maxbound=maxbound(drn_spd), stress_period_data=drn_spd, save_flows=True)

    if mf6_dict['drn']['present']:
        drn_spd = mf6_dict['drn']['stress_period_data']

    wel_spd = mf6_dict['wel']['stress_period_data']
    wspd = {}
    for key in wel_spd: # eliminate very small values to reduce the number of wells somewhat
        wspd[key] = [(cellid, -q * well_reduction_factor) for cellid, q in wel_spd[key][np.abs(wel_spd[key]['flux'])> 0.1]]
        
        
    if USEWELLS:
        wel = flopy.mf6.ModflowGwfwel(gwf,
                maxbound=maxbound(wel_spd),
                auto_flow_reduce=0.1, timeseries=None,
                stress_period_data=wspd,
                save_flows=True)

    # For transient flow
    # rch = flopy.mf6.ModflowGwfrcha(gwf, save_flows=True, readasarrays=None,
    #                                 timearrayseries=None,
    #                                 recharge=rch_rec)

    # recharge
    # tas_name = 'ema_rech'
    # tas_fname = tas_name + '.tas'
    # interpolation_method = 'stepwise'
    # #tas = {0.0: 0.002, 1000.: 0.}
    # tas = {   0.0: mf6_dict['rch']['stress_period_data'][0],
    #        1000.0: 0.}
    # tas.update(filename=tas_fname,
    #            time_series_namerecord=tas_name,
    #            interpolation_methodrecord=interpolation_method)

    # rcha = flopy.mf6.ModflowGwfrcha(gwf, save_flows=True, readasarrays=True,
    #                                 timearrayseries=tas, irch=get_irch(idomain, zero_based=True),
    #                                 recharge='timearrayseries ' + tas_name)
    rcha = flopy.mf6.ModflowGwfrcha(gwf, save_flows=True, readasarrays=True,
                                    timearrayseries=None,
                                    irch=mf6_dict['rch']['irch'],
                                    recharge=mf6_dict['rch']['stress_period_data'],
                                    )

    # Output control ('OC') package
    oc = flopy.mf6.ModflowGwfoc(gwf,
                                saverecord=saverecord,
                                head_filerecord=[headfile],
                                budget_filerecord=[budgetfile],
                                printrecord=printrecord,
                                )
    return sim


def run_simulation(sim):
    """Write simulation files and run the model.

    This should not have to be touched.
    """
    logging.warning("Runing simulation.")
    sim.write_simulation()

    # Run the model
    success, buff = sim.run_simulation()
    try:
        if not success:
            print("========================================")
            print("# MODFLOW 6 did not terminate normally. #")
            print("========================================")
            raise Exception("MODFLOW 6 did not terminate normally.")
    except:
        #mfsim_lst_fname = os.path.join(src_dir, 'mf6_dict', 'mfsim.lst')
        #conv_test.iter_progress()
        pass


def post_process(mf6_dict, HDS=None, iper=0, cmap='viridis', shps_dict=None, gis_dir=None):
    """Read and show the head results of the simulation in plane projection.

    Parameters
    ----------
    mf6_dict: dict
        the model data
    HDS: flopy headsfile oject
        the heads from modflow read in by flopy
    iper: int
        stress period index (zero-based)
    cmap: str
        name of colormap to be using in colorbar
    shps_dict: dictionary
        names of the shape files to include in the maps
    """ 
    logging.warning("Postprocessing.")
    gr = mf6_dict['dis']['gr']
    idomain = mf6_dict['dis']['idomain']
    inactive = idomain <= 0
    
    # Last step of given iper
    kstpkper = [(ks, ip) for (ks, ip) in HDS.get_kstpkper() if ip==iper][-1]

    Zbot = np.ma.MaskedArray(data=gr.Zbot, mask=inactive)

    elev     = Zbot
    hstep = 100.
    elev_max = np.ceil( np.max(elev) / hstep) * hstep
    elev_min = np.floor(np.min(elev) / hstep) * hstep

    hraw = HDS.get_data(kstpkper=kstpkper)
    h = np.ma.MaskedArray(data=hraw,
            mask=np.logical_or(inactive, hraw <= elev))

    # Neat top and bottom for the ylim of the graphs
    hmax = np.ceil( np.max(h) / hstep) * hstep
    hmin = np.floor(np.min(h) / hstep) * hstep
    
    hd_levels = np.arange(hmin, hmax + hstep, hstep) #includes both hmin and hmax

    btypes    = ['chd', 'wel', 'ghb', 'drn', 'riv']
    clrs      = ['r', 'b', 'm', 'c', 'y']
    markers   = ['s', 'o', 'p', '^', 'v']
    rot = 30
    fsz = 5 # fontsize

    formation_name = [mf6_dict['formations'][i]['name'] for i in range(gr.nz)]

    for layer in range(gr.nlay):
        ax = etc.newfig('Heads for layer {}, [{}]'.format(
            layer, formation_name[layer]),
            'x_EPGS6984 [m]', 'y_EPGS6984 [m]', figsize=(6, 10), aspect=1)

        gr.plot_grid(ax=ax, lw=0.1, color='gray')

        c1 = ax.imshow(elev[layer], cmap=cmap, vmin=elev_min, vmax=elev_max,
                    extent=gr.extent, alpha=0.5)

        # Overlay with contour lines
        c2 = ax.contour(
            gr.xm, gr.ym, h[layer], levels=hd_levels, linestyles='-', colors="black", linewidths=0.5)
        ax.clabel(c2, fmt="%.0f m", fontsize=6)

        cb = plt.colorbar(c1, ax=ax)
        cb.ax.set_title('elevation [m]')

        for btype, marker, clr in zip(btypes, markers, clrs):
            plot_bounds(mf6_dict, iper=iper, layer=layer, type=btype, ax=ax,
                            color=clr, mec=clr, mfc='none', fontsize=fsz,
                            marker=marker, rotation=rot)

        mashps.plot_objects(gis_dir=gis_dir, shps_dict=mashps.shps_dict, ax=ax)
    return None



def get_new_roof_of_model(gr_old, tif_path=None, ibound=None, plot=False):
    """Replace top of model with dem.

    Parameters
    ----------
    gr_old: fdm.Grid object
        the model grid
    tif_path: str
        Full path of the actual tif file with accurate grid.
        Notice that the file 'wbank_6984_km.tif has been prepared to
        cover the potential model area and has pixel size (1000, -1000)
        while the elevations are exactly at half-km interval in the
        new Israel-Palestine grid with EPGS6984.
        Therefore, a shift of the model should work seamlessly along as its
        cell size remains 1000 x 1000 m. Otherwize update the tif file
        using `read_tiffs_final.py`.
    plot: boolean
        produces plots if True
    out: Boolean
        if True return grid else return None
    """
    if tif_path is None:
        return gr_old
    
    from rasterio.plot import show 

    with rasterio.Env():    
        with rasterio.open(tif_path, 'r') as src:
            # The source tif file has been prepared such that it has the proper coordinates
            # compatible with the of the model with ground surface as data.
            
            # Get the dem from fp_tif
            Ztop0 = src.read(1)
            if plot:
                ax = etc.newfig("Dem over the entire potential model area",
                            "x [m] EPGS6984", "y [m] EPGS6984", aspect=1.0)
                show(src, ax=ax)

            # Compute the dem elevation coordinates
            xm = np.array([(src.transform * (i, 0))[0] for i in range(src.width)])
            ym = np.array([(src.transform * (0, i))[1] for i in range(src.height)])

            # Match with those of the current grid, and hence, the current model
            tol = 0.01
            Lx = np.logical_and(xm >= gr_old.xm[0] -tol, xm <= gr_old.xm[-1] + tol)
            Ly = np.logical_and(ym <= gr_old.ym[0] +tol, ym >= gr_old.ym[-1] - tol)

            # Select the part of the dem that matches the model grid exactly
            ztop = Ztop0[np.ix_(Ly, Lx)][np.newaxis, :, :]

            # ground surface correction, make layers at least 25 m thick
            # Set top to at least 25 m above each lower layer
            Zbot = gr_old.Zbot.copy()
            Zbot[ibound == 0] = -3000          # inactive must not have any impact on the outcome
            Zbot = np.max(Zbot, axis=0)        # maximum (=top) of bottoms
            ztop = np.fmax(ztop, Zbot + 25)    # makes layers at least 25 m thick unless inactive

            # Put this on top of the existing elevations
            Z = np.concatenate((ztop, gr_old.Zbot), axis=0)

            # Reconstruct the grid
            gr_new = mfgrid.Grid(gr_old.x, gr_old.y, Z, LAYCBD=gr_old.LAYCBD, min_dz=gr_old.min_dz)
    
            if plot:
                iz = 0
                ax = etc.newfig("Ground-surface of the model",
                        'x [m] EPGS6984', 'y [m] EPGS6984', aspect=1.0)
                ax.imshow(gr_new.Ztop[iz], extent = gr_new.extent)

    return gr_new
     
# %% Cross sections

# %% Show the model
def show_model(mf6_dict, gis_dir, iper=0, useimshow=True, showDZ=True):
    """Show the model elevation or layer thickness with ghb locations and heads.
    
    Parameters
    ----------
    mf6_dict: Dict
        Model Data
    gis_dir: str (path)
        Directory with the shape files.
    iper: Int
        Stress period number (zero-based)
    useimshow: Boolean
        Use imshow if true, else use contourf.
    showDZ: Boolean
        Show DZ else show Zbot
    """
    gr = mf6_dict['dis']['gr']
    inactive = mf6_dict['dis']['idomain'] <= 0

    formation_name = [mf6_dict['formations'][i]['name'] for i in range(gr.nz)]

    if showDZ:
        A = np.ma.masked_array(gr.DZ, mask=inactive, fill_value=-9999.99)
        title = "Thickness of model layer {}, [{}]"
    else:
        A = np.ma.masked_array(gr.Zbot, mask=inactive, fill_value=-9999.999)
        title = "Elevation of bottom of model layer {} [{}]"
    for iz in range(gr.nz): # for all aquifers, note: aquifers start at 1.
        chd_spd = mf6_dict['chd']['stress_period_data']
        ax = etc.newfig(title.format(iz, formation_name[iz]),
                    "x_EPGS6984 [m]", "y_EPGS6984 [m]", figsize=(6, 10))

        if useimshow:
            c = ax.imshow(A[iz], extent=gr.extent, alpha=1)
        else:
            c = ax.contourf(gr.xm, gr.ym, A[iz])

        # Plot the ghb points for period iper.
        for rec in chd_spd[iper]:
            (iz_mf, iy, ix), h_ = rec
            # But only if point is in current aquifer.
            if iz_mf == iz + 1:
                ax.plot(gr.xm[ix], gr.ym[iy], 'ro')
                ax.text(gr.xm[ix], gr.ym[iy], ' {} m'.format(str(h_)))

        mashps.plot_objects(gis_dir=gis_dir, shps_dict=mashps.shps_dict, ax=ax)

        # Contour elev of aquifer
        plt.colorbar(c, ax=ax) # For elevation.


#%% Verifications

def verify(mf6_dict, HDS, CBC):
    """Verify aspects of the model hydrologically.
    """
    NOT = np.logical_not
    print("Verify the recharge:")

    gr = mf6_dict['dis']['gr']

    # Surface area occipied by active cells is
    idomain = mf6_dict['dis']['idomain']

    # Total reacharge specified
    irch1 = get_irch(idomain, zero_based=False)
    active = irch1 !=0

    # Total recharge on active area
    RECH = mf6_dict['rch']['stress_period_data'][0]
    rch_tot       = np.sum(RECH * gr.Area)
    rch_tot_act   = np.sum(RECH * gr.Area * active)
    rch_tot_inact = np.sum(RECH * gr.Area * NOT(active))
    act_frac = np.sum(active * gr.Area) / np.sum(gr.Area)

    print("Active frac of model area     = {:8.4f} [ - ]".format(act_frac))
    print("Total recharge model area     = {:8.4f} m3/d".format(rch_tot))
    print("Total recharge active area    = {:8.4f} m3/d".format(rch_tot_act))
    print("Total recharge inactive area  = {:8.4f} m3/d".format(rch_tot_inact))

    # Wells:
    WEL = mf6_dict['wel']['stress_period_data'][0]
    Q1 = np.array([w[1] for w in WEL if w[0][0] == 1]).sum()
    Q3 = np.array([w[1] for w in WEL if w[0][0] == 3]).sum()
    Qt = np.array([w[1] for w in WEL]).sum()
    print("Qwells layer 1 = {:8.4f}".format(Q1))
    print("Qwells layer 3 = {:8.4f}".format(Q3))
    print("Qwells total   = {:8.4f}".format(Qt))
    
    # General head boundary
    GHB = mf6_dict['ghb']['stress_period_data'][0]
    hds = HDS.get_data(kstpkper=HDS.get_kstpkper()[-1])
    ghb = np.array([(hds[w[0]], w[1], w[2]) for w in GHB])
    Qghb = np.sum((ghb[:, 1] - ghb[:, 0]) * ghb[:, 2])
    print("Qghb total = {:8.4f}".format(Qghb))

    # Using the CellBudgetFile
    record_names = {'WEL', 'GHB', 'CHD', 'RCH'}.intersection([t.strip().decode('utf-8') for t in CBC.textlist])
    kstpkper = CBC.get_kstpkper()[-1]
    for recname in record_names:
        recs = CBC.get_data(kstpkper=kstpkper, text=recname)[0]
        qin = recs['q'][recs['q'] > 0]
        qout= recs['q'][recs['q'] < 0]
        print("Q {} = in: {:8.4f} out: {:8.4f}".format(
            recname, np.sum(qin), np.sum(qout)))
        
        
def verify2(mf6_dict, sim):
    """Verify aspects of the model hydrologically fram the aspecit of the sim object.
    
    Parameters
    ----------
    mf6_dict: dictionary 
        input and sturcture data for the model
    sim: mf6.sim object 
        simulation object that can be requested for simulation results.
    """
    NOT = np.logical_not
    print("Verify the recharge:")

    # Getting HDS and CBC object from the sim object
    mdl = sim.get_model()
    HDS = mdl.output.head()
    CBC = mdl.output.budget()

    gr = mf6_dict['dis']['gr']

    # Surface area occipied by active cells is
    idomain = mdl.dis.idomain.array

    # Total reacharge specified
    irch1 = get_irch(idomain, zero_based=False)
    active = irch1 !=0

    # Total recharge on active area
    # RECH = mf6_dict['rch']['stress_period_data'][0]
    
    # Getting recharge from the simulation object.
    RECH = mdl.rch.recharge.get_data()[0]
    rch_tot       = np.sum(RECH * gr.Area)
    rch_tot_act   = np.sum(RECH * gr.Area * active)
    rch_tot_inact = np.sum(RECH * gr.Area * NOT(active))
    act_frac = np.sum(active * gr.Area) / np.sum(gr.Area)

    print("Active frac of model area     = {:8.4f} [ - ]".format(act_frac))
    print("Total recharge model area     = {:8.4f} m3/d".format(rch_tot))
    print("Total recharge active area    = {:8.4f} m3/d".format(rch_tot_act))
    print("Total recharge inactive area  = {:8.4f} m3/d".format(rch_tot_inact))

    # Wells:
    # TODO: getting wells from the sim object. Probably not so. It is in the 
    # CBC object.
    WEL = mf6_dict['wel']['stress_period_data'][0]
    Q1 = np.array([w[1] for w in WEL if w[0][0] == 1]).sum()
    Q3 = np.array([w[1] for w in WEL if w[0][0] == 3]).sum()
    Qt = np.array([w[1] for w in WEL]).sum()
    print("Qwells layer 1 = {:8.4f}".format(Q1))
    print("Qwells layer 3 = {:8.4f}".format(Q3))
    print("Qwells total   = {:8.4f}".format(Qt))

    kstpkper = HDS.get_kstpkper()[-1]
    
    # General head boundary
    #GHB = mf6_dict['ghb']['stress_period_data'][0]
    GHB = CBC.get_data(kstpkper=kstpkper, text='GHB')
    Qghb = GHB[0].q.sum()
    print("Qghb total = {:8.4f}".format(Qghb))
    
    hds = HDS.get_data(kstpkper=kstpkper)
    #ghb = np.array([(hds[w[0]], w[1], w[2]) for w in GHB])
    #ghb = hds.ravel()[GHB.node - 1]

    # Using the CellBudgetFile
    record_names = {'WEL', 'GHB', 'CHD', 'RCH'}.intersection([t.strip().decode('utf-8') for t in CBC.textlist])
    for recname in record_names:
        recs = CBC.get_data(kstpkper=kstpkper, text=recname)[0] # is a list, unapack it
        try: # ghb, riv, drn
            qin = recs['q'][recs['q'] > 0]
            qout= recs['q'][recs['q'] < 0]
        except: # wells
            qin = recs['flux'][recs['flux'] > 0]
            qout= recs['flux'][recs['flux'] < 0]
            
        print("Q {} = in: {:8.4f} out: {:8.4f}".format(
            recname, np.sum(qin), np.sum(qout)))
        


#%%
def analyze_convergence(mf6_dict, HDS, iper=0, showDZ=False, folders=None):
    """Anlayze the convergence by location of max_dv an dmax_dr."""

    # The location of the output files of modflow:
    data = os.path.join(folders['src_python'], 'mf6_dict')

    gr = mf6_dict['dis']['gr']

    inactive = mf6_dict['dis']['idomain'] <= 0
    

    Zbot = np.ma.masked_array(data=gr.Zbot, mask=inactive)

    kstpkper = [kskp for kskp in HDS.get_kstpkper() if kskp[-1]==iper][-1]
    hds = np.ma.masked_array(HDS.get_data(kstpkper=kstpkper), mask=inactive)
    vstp = 100.
    vmin = np.floor(Zbot.max() / vstp) * vstp
    vmax = np.ceil( Zbot.min() / vstp) * vstp

    dstp = 10.
    DZ   = np.ma.masked_array(data=gr.DZ,   mask=inactive)
    dmin, dmax = 0, np.ceil(DZ.max() / dstp) * dstp

    # The output of the IMS, showing progress of convergence
    ims_inner = os.path.join(data, 'ims_inner.csv')
    ims_outer = os.path.join(data, 'ims_outer.csv')

    ii = pd.read_csv(ims_inner) # Follow inner iterations
    io = pd.read_csv(ims_outer) # TODO Follow the outer iterations

    # Show progress of convergence
    ax = etc.newfig("Solution_inner_dvmax", "total_inner_iterations", "solution_inner_dvmax", yscale='log')
    ax.plot(ii['total_inner_iterations'], ii['solution_inner_dvmax'],
     label='dvmax', lw=0.25)
    ax.plot(ii['total_inner_iterations'], ii['solution_inner_drmax'],
     label='drmax', lw=0.25)

    ax = etc.newfig("Solution_inner_dvmax_node", "total_inner_iterations", "solution_inner_dvmax", yscale='log')
    ax.plot(io['total_inner_iterations'], io['solution_outer_dvmax'],
     label='dvmax', lw=0.25)

    # Show where the maximum error occurs
    lrc_idv = np.asarray(gr.LRClrc(ii['solution_inner_dvmax_node'].values,
     shape=gr.shape, zero_based=True), dtype=int).T
    lrc_idr = np.asarray(gr.LRC(ii['solution_inner_drmax_node'].values,
     shape=gr.shape, zero_based=True), dtype=int).T
    lrc_odv = np.asarray(gr.LRC(io['solution_outer_dvmax_node'].values,
     shape=gr.shape, zero_based=True), dtype=int).T

    btypes = ['chd', 'wel', 'ghb', 'drn', 'riv']
    bclrs  = ['r', 'b', 'm', 'c', 'y']
    bmarkers   = ['s', 'o', 'p', '^', 'v']
    rot = 30
    fsz = 7 # fontsize

    # Find the locations where the layer thickness is gr.min_dz, while
    # idomain >1. These are suspected causing trouble with convergence.
    # TODO: find the origin of this nuisanse.
    active = mf6_dict['dis']['idomain'] > 0
    cells_too_thin = np.isclose(gr.DZ, gr.min_dz)
    bad = np.logical_and(active, cells_too_thin)
    assert not gr.LRC(gr.NOD[bad]),\
        "There must be no active cells with thickness <= gr.min_dz"

    # Plot the boundary points, so where they match the largest errors
    # Especially wells may be pumped more than the aquifer can deliver.
    # We can check this automatically or let the well be reduced until
    # its water level remains above the bottom of the aquifer.
    for ilay in range(gr.shape[0]):
        clrs = ['w', 'y','m']
    
        ax = etc.newfig("location of max error, ilay={})".format(ilay),
                            "x [m]", "y [m]", figsize=(6, 10))

        if not showDZ:     
            c = ax.imshow(Zbot[ilay], extent=gr.extent, aspect='equal',
                            vmin=vmin, vmax=vmax, cmap='viridis')
        else:
            c = ax.imshow(DZ[ilay], extent=gr.extent, aspect='equal',
                            vmin=dmin, vmax=dmax, cmap='viridis')

        ax.plot(gr.xm[lrc_idv[2][lrc_idv[0]==ilay]],
                  gr.ym[lrc_idv[1][lrc_idv[0]==ilay]],
            color=clrs[0],  ls='none', marker='.', label='inner_dv')

        ax.plot(gr.xm[lrc_idr[2][lrc_idr[0]==ilay]],
                  gr.ym[lrc_idr[1][lrc_idr[0]==ilay]],
            color=clrs[1], ls='none', marker='.', label='inner_dr')

        ax.plot(gr.xm[lrc_odv[2][lrc_odv[0]==ilay]],
                  gr.ym[lrc_odv[1][lrc_odv[0]==ilay]],
            color=clrs[2], ls='none', marker='.', label='outer_dv')
        
        for btype, clr, marker in zip(btypes, bclrs, bmarkers):
            plot_bounds(mf6_dict, iper=iper, layer=ilay, type=btype, ax=ax,
                            color=clr, mec=clr, mfc='none', fontsize=fsz,
                            marker=marker, rotation=rot)

        plt.colorbar(c, cax=None, ax=ax)

        ax.legend()

        mashps.plot_objects(gis_dir=folders['gis'], shps_dict=mashps.shps_dict, ax=ax)

    return

# %%
if __name__ == '__main__':
    
    # %%
    # src_dir is the directory with the python source files defined in setpaths
    # It is the startup diretory, otherwise it can't work.
    logging.warning("Running main in file {}".format(__file__))

    os.chdir(folders['src_python'])
    logging.warning("cwd = {}".format(os.getcwd()))
    
    config = SimpleNamespace(ws=folders['src_python'],
                             sim_name='yasin_old3d',
                             model_name=None,
                             exe_name=os.path.join(folders['exe'], 'mf6.mac'),
                             build_model=True,
                             write_model=True,
                             run_model=True,
                             plot_model=True,
                             plot_grid=False,
                             gwf_pkgs = ['chd', 'wel', 'drn', 'ghb', 'rcha'],
                             well_reduction_factor = 1.0,
                             strategy = 3,
                             showDZ = False,
                             gr_surface_tif = None,
    )

    # Scenario parameter settings:
    newmodel = config.sim_name
    
    USEWELLS = 'wel' in config.gwf_pkgs
    
    # For debugging multiply all well flows by this factor to prevent drying out of aquifers and allow convergence.
    well_reduction_factor = config.well_reduction_factor
    
     # Boundary-head correction stratagy to ensure head in within correct layer
    strategy = config.strategy
    
     # IF True DZ is shown in cross sections, if Ffalse Zbot is shown
    showDZ = config.showDZ

    # Get Yasin_old model dictionary, from pickle file or generated it anew
    with open(os.path.join(folders['test'], 'yasin_old.pickle'), 'rb') as fp:
        logging.warning("Retrieving `yasin_old.pickle` from folders[{}]".format('test'))
        yasin_old = pickle.load(fp)

    # Replace the top of the top layer (ground surface) with a USGS dem derived layer
    if config.gr_surface_tif:
        gr_surface_tif = os.path.join(folders['dems'], 'mdl_elev.tif')
        gr_surface_tif = os.path.join(folders['src_python'], 'wbank_6984_km.tif' )
        yasin_old['bcf']['.TOP'][0] = gr_surface_tif
    else:
        gr_surface_tif = None

    #model_dict = to_model_dict(yasin_old) # to be defined

    # Rows for which to show cross sections
    cross_section_rows = [2, 5, 10, 30, 50, 63, 70]
    
    if 'Mountain peaks.shp' in mashps.shps_dict.keys():
        mashps.shps_dict.pop('Mountain peaks.shp')



    # Generate the model in mf6 (its grid and data)
    # Note that this is the steady=state version
    mf6_dict = get_new_mf6_model(yasin_old, steady=False, tif_path=gr_surface_tif, chd_dead_sea_coast=None, drains_present=False, strategy=strategy)

    # Show what the new model looks like (maps, cross sections)
    show_model(mf6_dict, folders['gis'], useimshow=True, showDZ=showDZ)

    #plot_cross_sections(mf6_dict, rows=cross_section_rows, HDS=None, iper=iper, step=True)

    # Instantiate a modflow 6 simulation object
    sim = set_up_modflow_sim(mf6_dict, simname=newmodel, folders=folders, USEWELLS=USEWELLS, well_reduction_factor=well_reduction_factor)
    run_simulation(sim)

    mdl = sim.get_model()
    print("Available methods in mdl.output object: ", mdl.output.methods())
    # Read the binary head and cell-by-cell budget files
    
    HDS = mdl.output.head()
    CBC = mdl.output.budget()
    
    #headfile    = "{}.hds".format(newmodel)  # Name of the head file produced by modflow 6
    #budgetfile  = "{}.cbb".format(newmodel)  # Name of the budget file produced by modflow 6

    #HDS = flopy.utils.binaryfile.HeadFile(headfile)  # Read the binary head file
    #CBC = flopy.utils.binaryfile.CellBudgetFile(budgetfile) # Read the binary cell-by-cell budget file

    #mdl = sim.get_model("mf6_dict")
    #bud = mdl.ooutput.budget()
    #bud.get_data(idx=-1, full3D=True)

    # Choose a stress period to show the simulation results
    iper = 0

    plot_cross_sections(mf6_dict, rows=cross_section_rows, HDS=HDS, iper=iper, figsize=(12, 6))  # Step is True --> cross sections shown with stepped elevations (better)
    #plot_cross_sections(mf6_dict, rows=cross_section_rows, HDS=HDS, iper=iper,) # Step is Ffalse --> cross section shown with continuous elevations (not so good)
    

    # Post-processing, i.e. contouring the heads in the different layers
    post_process(mf6_dict, HDS=HDS, iper=iper, cmap='viridis', shps_dict=mashps.shps_dict, gis_dir=folders['gis'])

    # %% Carry out some verifications (see function)
    verify(mf6_dict, HDS, CBC)
    
    verify2(mf6_dict, sim)

    # %% Show some figures to follow the progress of the convergence 
    #analyze_convergence(mf6_dict, HDS, iper=0, showDZ=showDZ, folders=folders)

    #print("Done1")
    plt.show(block=True)

# %%
