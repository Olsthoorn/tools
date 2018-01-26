#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 17:24:18 2017

Copy files of drillings from dinoloket, but only the xml files version 3.
This reduces the burden on the computer.

This is to be used only once

@author: Theo
"""

import os
import shutil

boredir = '/Users/Theo/Instituten-Groepen-Overleggen/HYGEA/Consult/2017/' +\
        'DEME-julianakanaal/DINO_boringen'

destlocaldir = 'dump/'

destdir = os.path.join(boredir, destlocaldir)

for d in os.listdir(boredir):
    p = os.path.join(boredir, d, 'Boormonsterprofiel_Geologisch booronderzoek')
    if os.path.isdir(p):
        print('\ndirectory found: ', p)
        xml14files = [f for  f in os.listdir(p) if f.endswith('1.3.xml')]
        i = 0
        for f in xml14files:
            #shutil.move(os.path.join(p, f), os.path.join(destdir, f))
            #shutil.copy(os.path.join(p, f), os.path.join(destdir, f))
            print(f)
            i += 1
            if i==10:
                print(i)



