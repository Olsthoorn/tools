#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 22:54:30 2018

@author: Theo
"""

import os
from mlu2xml import mlu2xml, Mluobj


if __name__=='__main__':


    fnames = ['gat_boomse_klei.mlu']
    #,
    #        'zuidelijke_landtong.mlu']
    folder = './testdata'

    timeshifts = [0.60e-3, 0] #, 3e-3, 3e-2]

    for fname, tshift in zip(fnames,timeshifts):
        fname = os.path.join(folder, fname)
        fnxml = os.path.splitext(fname)[0] + 'xml'

        #tree = mlu2xml(fname)

        #print('mlu file {} parsed'.format(fname))

        #tree.write(fnxml, xml_declaration=True)

        #print('mlu file {} written'.format(fnxml))


        mu = Mluobj(fnxml)
        mu.plotDrawdown(mu.obsNames, yscale='log', xlim=(1e-3, 5.), xscale='log', marker='.', tshift=tshift)
        mu.section(xscale='linear')
        #mu.planview()