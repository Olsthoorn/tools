# %% [markdown]
# # Check the implementation of the Dutch soils

# The Van Genughten parameters describing the standard Dutch Soil series, called "De Staringreeks"
# have been copied from the report by Heinen, Bakker and Wösten (2020), update 2018.
# The implementation NL_VG_soils allows computing values of the properties of these soils
# in an application. To verify the implementation table 4 was also copied from Heinen et al. (2020),
# which enumerates the values of the properties computed from the given parameter values.
# The current module loads Dutch soils and table 4, computes the values listed in table 4
# and compares them with the implementation.

# The comparison shows that the implementation perfectly matches the data in table4.

# TO 13-09-2025

# %% --- imports

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import cycle

import etc

dirs = etc.Dirs(os.getcwd())
# dirs.src = str(dirs.src)

#if dirs.src not in sys.path:
#    sys.path.insert(0, str(dirs.src))

from src.NL_soils import Soil # noqa

# %%  --- load the Dutch soils and import table 4.

wbook = os.path.join(dirs.data, 'NL_VG_soilprops.xlsx')
if not os.path.isfile(wbook):
    raise FileNotFoundError(f"Can't find file `{wbook}`")

Soil.load_soils(wbook=wbook)

# %% --- load table 4 and put it in a dictionary

sheet_name = 'tabel4'

tab4 = pd.read_excel(wbook, sheet_name=sheet_name, index_col=(0,1), header=0, usecols='A:Q')
tab4 = tab4.dropna().sort_index()

# --- the suction values pertaining to the values
psi = np.array(tab4.columns[:-2], dtype=float)
psi[0] = 1. # we'll use pF

# --- turn the complex table in to an accessible dict
table4 = dict()
table4['psi'] = psi  

# --- the unique soil codes and parameters
kind = np.unique([idx[0] for idx in tab4.index[1:]])
pars = np.unique([idx[1] for idx in tab4.index[1:]])
 
for k in kind:
    # --- for each soil code (kind of soil)
    table4[k] = dict()  # subdict to fill next
    
    # --- get the main type and the description of the soil
    k_row = tab4.loc[(k, 'K')]
    cols  = list(k_row.columns[-2:])
    table4[k]['type']  = k_row[cols[0]].values[0]
    table4[k]['descr'] = k_row[cols[1]].values[0]

    for p in pars:
        # --- for each parameter with the same soil code
        row = np.asarray(tab4.loc[(k, p)].values[0][:-2], dtype=float)
        table4[k][p] = row
        
        # --- verify what's going on
        print(k, p, row[:5], ' ... ', row[-5:])

# %% --- Compute the values in the table using the parameters

# --- choose bovengronden or ondergronden
codes = [k for k in table4.keys() if k[0] == 'B']
codes = [k for k in table4.keys() if k[0] == 'O']

# --- set up fig with two axes
titles = ("psi(theta)", "K(theta)")
ax1, ax2 = etc.newfigs(titles, "theta", ("pF", "K [cm/d]"), sharex=True)
ax2.set_yscale('log')
pF = np.log10(table4['psi'])

# --- run for all soil codes
clrs = cycle('rbgkmcy')
for code in codes:
    clr = next(clrs)
    soil = Soil(code)
    theta = soil.theta_fr_psi(table4["psi"])
    K = soil.K_fr_theta(theta)
    
    ax1.plot(table4[code]['theta'], pF,                'o', color=clr, label=f"{code} table4")
    ax2.plot(table4[code]['theta'], table4[code]['K'], 'o', color=clr, label=f"{code} table4")
    
    ax1.plot(theta, pF, '-', color=clr, label=f"{code} Theta")
    ax2.plot(theta, K,  '-', color=clr, label=f"{code} K")
ax1.legend(fontsize=6)
ax2.legend(fontsize=6)


# %% Verify dK/dtheta

# --- set up fig with two axes
title = ("V = dK/dtheta")
ax = etc.newfig(title, "theta", "K [cm/d]")
ax.set_yscale('log')

dtheta = 1e-4

# --- run for all soil codes
clrs = cycle('rbgkmcy')
for code in codes:
    clr = next(clrs)
    soil = Soil(code)
    theta = soil.theta_fr_psi(table4["psi"])
    V_analytic = soil.dK_dtheta(theta)
    
    V_numeric  = (soil.K_fr_theta(theta + 0.5 * dtheta) -
                  soil.K_fr_theta(theta - 0.5 * dtheta)) / dtheta
    
    ax.plot(theta, V_analytic, color=clr, label=f"{code} V analytic")
    ax.plot(theta, V_numeric,  'o', color=clr, label=f"{code} V numeric")
ax.legend(fontsize=8)





# %%
