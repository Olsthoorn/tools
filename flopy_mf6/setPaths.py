#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" set_paths.py

Module to add directories to sys.path specific to this workspace
See: https://stackoverflow.com/questions/15208615/using-pth-files

>>> setPaths
All folders in dict 'folders' exist.
"""

# %%
import sys
import os
import logging
logging.basicConfig(level=logging.WARNING,
        format=' %(asctime)s - %(levelname)s - %(message)s')

#logging.warning("Importing setPaths --> see project folders, see `folders`")

def get_folders():
    """Return dictionary with all folders used in this project (mountain aquifer)."""
    #sys.path.append('/whatever/dir/you/want')
    
    fjoin = lambda folder, subfolder: os.path.join(folders[folder], subfolder)
    
    folders ={'hygea': '/Users/Theo/Entiteiten/Hygea/'}
    folders['exe'            ] =  '/Users/Theo/GRWMODELS/mfLab/trunk/bin/'
    folders['tools'          ] =  '/Users/Theo/GRWMODELS/python/tools/'

    folders['proj'           ] = fjoin('hygea'       , '2020_BUZA_ISR_PAL_T/')
    folders['models'         ] = fjoin('proj'        , 'mountain_aquifers/')
    folders['data'           ] = fjoin('models'      , 'data/')
    folders['model'          ] = fjoin('models'      , 'eastern_aquifer/')
    folders['gis'            ] = fjoin('models'      , 'QGIS/')
    folders['src'            ] = fjoin('model'       , 'python/')
    folders['mf6'            ] = fjoin('model'       , 'yasin_mf6/')

    folders['test'           ] = fjoin('src'         , 'tests/')

    # Adding certain path to sys.path for importing modules
    for key in ['src_python', 'dems_python', 'bast_python', 'tools']:
        if not folders[key] in sys.path:
            sys.path.append(folders[key])
            
    for key in folders:
        assert os.path.isdir(folders[key]), f'Folder {folders[key]:} does not exist.'
    return folders
            
folders = get_folders()

if __name__ == '__main__':
    
    if not folders: folders = get_folders()
    
    os.chdir(folders['python'])
    #logging.warning("cwd = {}".format(os.getcwd()))

    # Show the folders (They have already been asserted).
    mxlen = 80
    #print("\nFolders used, truncated if path len is > {} :".format(mxlen))
    for f in folders:
        folder_name = folders[f] if len(folders[f]) < mxlen + 3 else '...' + folders[f][-mxlen:]
        if not os.path.isdir(folders[f]):
            raise FileNotFoundError(folder_name)
        else:
            # print(f"{f:12s} :, {folder_name}")
            pass        
    print("\nAll folders in dict 'folders' exist.\n")