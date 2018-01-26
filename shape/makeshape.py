#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 12:33:24 2017

@author: Theo
"""
import os
import sys
import pandas as pd
import shapefile as sf
from pprint import pprint

tools = '/Users/Theo/GRWMODELS/python/tools'
proj = '/Users/Theo/Instituten-Groepen-Overleggen/HYGEA/Consult/2017/' +\
    'DEME-julianakanaal/'
dataDir = 'Data_DEME/Boringen_Juka'
dataPath = os.path.join(proj, dataDir)

def excel2shape(excelFile):

fields = [('Hole ID', 'C', 13, 0),
            ('Elevation', 'F', 15, 0),
            ('Starting Depth', 'F', 15, 3),
            ('Ending Depth', 'F', 15, 3),
            ('Boormeester', 'C', 21, 0),
            ('Boorbedrijf', 'C', 30, 0),
            ('Projectleider', 'C', 26, 0),
            ('Klant', 'C', 33, 0),
            ('Plaatsnaam', 'C', 26, 0),
            ('Boordatum', 'D', 15, 0),
            ('Boormethode', 'C', 29, 0),
            ('Boordiameter', 'F', 15, 0),
            ('Afwerking', 'C', 20, 0),
            ('Projectreferentie', 'C', 34, 0),
            ('Opdrachtgever', 'C', 28, 0)
            ]


if __name__ == '__main__:


    excelFile ='Boorgatgegevens.xlsx'
    os.path.isfile(os.path.join(dataPath, excelFile))

    boreholes = pd.read_excel(os.path.join(dataPath, excelFile),
                  sheetname='Collars')

    for c in boreholes.columns:
        print("('{}', 'C', {}, 0),".format(c,
                np.max(
                    [len(f) if isinstance(f, str) else 10 for f in boreholes[c]
                        ]) + 5))

    wr = sf.Writer(shapeType=sf.POINTZ)

    flds = [f[0] for f in fields]
    for f in fields:
        wr.field(*f)
    for i in boreholes.index:
        wr.point(boreholes['Easting'][i],
                 boreholes['Northing'][i],
                 boreholes['Elevation'][i])
        wr.record(**dict(boreholes.iloc[i, :]))
    wr.save('Boorgaten')
        #wr.record(**{fld:db[fld] for fld in flds})



