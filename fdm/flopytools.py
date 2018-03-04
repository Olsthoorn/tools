#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 10:39:53 2018

@author: Theo
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import flopy.utils.binaryfile as bf

#%% Water balance

class modflow_budget:
        
    # labels for water balance
    cbc_label ={
                'STO': {'text': b'         STORAGE', 'color':'c', 'name':'Storage'},
                'RIV': {'text': b'   RIVER LEAKAGE', 'color':'r', 'name':'Jukanaal'},
                'RCH': {'text': b'        RECHARGE', 'color':'g', 'name':'Recharge'},
                'WEL': {'text': b'           WELLS', 'color':'y', 'name':'EastInfl'},
                'DRN': {'text': b'          DRAINS', 'color':'m', 'name':'Drainage'},
                'CHD': {'text': b'   CONSTANT HEAD', 'color':'b', 'name':'Maas'}
                }

    txt2lbl = dict([(v['text'], k)
                  for k, v in zip(cbc_label.keys(), cbc_label.values())])

    def __init__(self, workspace, modelname, ext='.cbc'):
        
        self.CBC = bf.CellBudgetFile(os.path.join(workspace, modelname + ext))

        # get active labels
        self.labels=[self.__class__.txt2lbl[txt] for txt in self.CBC.textlist
                if txt in self.__class__.txt2lbl.keys()]

        kstpkper = self.CBC.get_kstpkper()
        nstp = len(kstpkper)
        
        # initialize budget for all labels, per label np.array(n, 2)
        # where np.array[:,0] = inflows and np.array[:,1] = outflows
        self.budget = {lbl : np.zeros((nstp, 2)) for lbl in self.labels}
        
        cbclbl = self.__class__.cbc_label # shorthand
        
        for i, sp in enumerate(kstpkper):
            for label in self.labels:
                try:
                    rec = self.CBC.get_data( kstpkper=sp,
                                             text=cbclbl[label]['text'])[0]
                    if isinstance(rec, np.recarray):
                        self.budget[label][i, 0] = np.sum(rec['q'][rec['q']>0])
                        self.budget[label][i, 1] = np.sum(rec['q'][rec['q']<0])
                    elif label in ['RCH', 'EVT']:
                        self.budget[label][i, 0] = np.sum(rec[1][rec[1]>0])
                        self.budget[label][i, 1] = np.sum(rec[1][rec[1]<0])
                    else: # 3D array
                        self.budget[label][i, 0] = np.sum(rec[rec>0])
                        self.budget[label][i, 1] = np.sum(rec[rec<0])
                except:
                    pass # skips when label not avaiable for stress period
        
    def show(self, order=['STO', 'RIV', 'RCH', 'WEL', 'DRN', 'EVT', 'CHD']):
        # Labels to use in for water budget time graph of toplayer
        fig, ax = plt.subplots()
        ax.set_title('Budget over time')
        ax.set_xlabel('time -->')
        ax.set_ylabel('m3/d')
        ax.grid()
        
        self.labels = [lbl for lbl in order if lbl in self.labels][::-1] #.reverse()
        

        inflows = [ self.budget[label][:,0] for label in self.labels]
        outflows= [ self.budget[label][:,1] for label in self.labels]
        colors = [self.__class__.cbc_label[label]['color'] for label in self.labels]
        names  = [self.__class__.cbc_label[label]['name' ] for label in self.labels]

        t = np.array(self.CBC.times)
        ax.stackplot(t,  *inflows, labels=names, colors=colors)
        ax.stackplot(t,  *outflows, colors=colors)
        ax.legend(loc='best')
