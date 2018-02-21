#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 30 10:03:21 2017

Generates xml files from the bores in the file
Boorgatgegevens.xlsx in which the DEME borings have
been placed by AGT as an extract or dump from the program
stater.

The excelfile Boorgatgegevens.xlsx has been hard-wired in the
function agtborings2xml.

One the xml files have been generated they can be plotted using
dinobores.py. Also see the file parsebores in directory

DEME-juliana/analyse

It requires the py file agtborecodes to interprete the
phrase used in the excel file

@author: Theo
"""

import os
import sys
import numpy as np

analyse = '/Users/Theo/GRWMODELS/python/2017_DEME-juliana/analyse'

if not analyse in sys.path:
    sys.path.insert(1, analyse)

import pandas as pd
import xml.etree.ElementTree as ET
import dino.bores.dinobores as db
import dino.bores.agtborecodes as bc
import matplotlib.pyplot as plt

def full(): plt.get_current_fig_manager().full_screen_toggle()


#%% functions

def extract(parent, what):

    newNode(parent, 'description', text=what)

    slist = what.lower().split(',')

    # lithology is always the first part of the description
    lith = slist[0].strip()
    newNode(parent, 'lithology', {'code': bc.lithology[lith]}, text=lith)


    for s in [s.strip() for s in slist[1:]]:
        if lith == 'zand' and s in bc.sandMedianClass:
            newNode(parent, 'sandMedianClass',
                                {'code' : bc.sandMedianClass[s]},   text=s)

        elif lith == 'grind' and s in bc.gravelMedianClass:
            newNode(parent, 'gravelMedianClass',
                                {'code' : bc.gravelMedianClass[s]}, text=s)

        if s in bc.siltAdmix:
            newNode(parent, 'siltAdmix', {'code': bc.siltAdmix[s]}, text=s)
        elif s in bc.clayAdmix:
            newNode(parent, 'clayAdmix', {'code': bc.clayAdmix[s]}, text=s)
        elif s in bc.sandAdmix:
            newNode(parent, 'sandAdmix', {'code': bc.sandAdmix[s]}, text=s)
        elif s in bc.gravelAdmix:
            newNode(parent, 'gravelAdmix',
                                {'code' : bc.gravelAdmix[s]},       text=s)
        elif s in bc.organicAdmix:
            newNode(parent, 'organicAdmix',
                                {'code' : bc.organicAdmix[s]},      text=s)
        elif s in bc.humusAdmix:
            newNode(parent, 'humusAdmix',
                                {'code' : bc.humusAdmix[s]},        text=s)

        elif s in bc.primSec:
            pc = bc.primSec[s]
            if pc[0] is not None:
                newNode(parent, 'colorIntensity', {'code': pc[0]}, text=pc[0])
            if pc[1] is not None:
                newNode(parent, 'primaryColor',   {'code': pc[1]}, text=pc[1])
            if pc[2] is not None:
                newNode(parent, 'secondaryColor', {'code': pc[2]}, text=pc[2])
            if pc[3] is not None:
                newNode(parent, 'matplotlibColor',{'code': pc[3]}, text=pc[3])

        elif s in bc.shellFrac:
            newNode(parent, 'shellFrac', {'code': bc.shellFrac[s]}, text=s)
        elif s in bc.plantType:
            newNode(parent, 'plantType', {'code': bc.plantType[s]}, text=s)
        elif s in bc.glaucFrac:
            newNode(parent, 'glaucFrac', {'code': bc.glaucFrac[s]}, text=s)
        elif s in bc.lithology:
            newNode(parent, 'lithology', {'code': bc.lithology[s]}, text=s)


def newNode(parent, tag, attrib=None, text=None):
    el = ET.Element(tag)
    if attrib is not None: el.attrib = attrib
    if text   is not None: el.text = text
    if parent is not None: parent.append(el)
    return el

def tofile(root, path=None):
    fName = root[0][0].text

    print(fName)

    et = ET.ElementTree(root)

    if not path is None:
        fName = os.path.join(path, fName)

    et.write(fName + '.xml',
              encoding='utf-8',
              xml_declaration=True,
              default_namespace=None,
              method='xml',
              short_empty_elements=True)


def agtbores2xml(path=None):
    '''returns borehole info and generates XML files for the boreholes
    from the excel file Boorgatgegevens.xlsx that contains the information
    of the boreholes commissioned by De Vries en van de Wiel for het
    project Verruiming en verdieping Julianakanaal.
    De file is een extract van Strater, verkregen van consultancy AGT
    december 2017.
    De filename is hard wired hier.
    De boringen kunnen worden getekend met dinobores precies
    zoals dat voor de boringen van dinoloket het geval is.
    De XML files die hier worden gegenereerd hebben naas de boring
    informatie ook de gegevens van de peilfilters.
    parameters
    ----------
    path : str
        pad waar de files moeten worden geplaatst

    opmerking: In de sheets van de excel file zijn enkele taalfoutjes verbeterd
    en zijn lege rijen van gebruikte kolommen aangevuld met "onbekend" of "bkp"
    voor de filters. Dit was nodig om alle gegevens te kunnen gebruiken en
    ervoor te zorgen dan xml.elementtree.ElementTree niet struikelt over nan's


    @theo 20180102

    '''
    #hardwiring the date file
    excel = '/Users/Theo/GRWMODELS/python/2017_DEME-juliana/bores/Boorgatgegevens.xlsx'
    assert os.path.isfile(excel), FileNotFoundError(excel)

    print('Generating files from workbook: ', os.path.basename(excel))
    print('reading sheet "Collars"')
    collars = pd.read_excel(excel, sheetname='Collars',
                            header=0, index_col='Hole ID', parse_dates=True)
    print('reading sheet "Lithology"')
    lith  = pd.read_excel(excel, sheetname='Lithology2',
                            header=0, index_col='Hole ID', parse_dates=True)
    print('reading sheet "Filter1"')
    filt1 = pd.read_excel(excel, sheetname='Filter1',
                                          header=0, index_col='Hole ID')
    print('reading sheet "Filter2"')
    filt2 = pd.read_excel(excel, sheetname='Filter2',
                                          header=0, index_col='Hole ID')
    print('reading sheet "Filter3"')
    filt3 = pd.read_excel(excel, sheetname='Filter3',
                                          header=0, index_col='Hole ID')

    print('reading sheet "Waterpeil"')
    waterlevel = pd.read_excel(excel, sheetname='Waterpeil',
                                          header=0, index_col='Hole ID')

    print('total number borings: ', len(collars))


    bore = dict()

    for k in collars.index:
        bore[k]=dict()
        bore[k]['meta'] = collars.loc[k]
        bore[k]['lith'] = lith.loc[k]

        meta = collars.loc[k]

        root = newNode(None, 'set')
        psv =  newNode(root, 'pointSurvey',{'version': "1.0",
                                    'embargo': 'OPENBAAR',
                                    'drillingCompany': meta['Boorbedrijf'],
                                    'commisionedBy' : meta['Opdrachtgever'],
                                    'projectRef': meta['Projectreferentie']})
        _    = newNode(psv, 'identification', {'id' : k}, k)

        svl = newNode(psv, 'surveyLocation')
        coo = newNode(svl, 'coordinates',
                      {'originalMeasurement':"JA",
                          'UoM':"METER",
                          'coordSystem':"RD"})
        _   = newNode(coo, 'coordinateX', text=str(meta['Easting' ]))
        _   = newNode(coo, 'coordinateY', text=str(meta['Northing']))

        sve = newNode(psv, 'surfaceElevation')

        _   = newNode(sve, 'elevation', {"originalMeasurement":"JA",
                                         "levelReference":"NAP",
                                         "levelValue": str(meta['Elevation']),
                                         "UoM":"METER"})

        gpe = newNode(sve, 'geoPoliticalLocation')
        _   = newNode(gpe, 'coountryName', text='Nederland')
        _   = newNode(gpe, 'locationName', text=meta['Plaatsnaam'])

        bho = newNode(psv, 'borehole', {"version":"1.0",
                                        "baseDepthUoM":"METER",
                                        "baseDepth": str(meta['Ending Depth'])})
        _ = newNode(bho, 'operatorOrg', text= meta['Klant'])

        date = meta['Boordatum']
        _  = newNode(bho, 'date', {"startYear" : str(date.year),
                                   "startMonth": str(date.month),
                                   "startDay"  : str(date.day) } )

        ldescr = newNode(bho, 'lithoDescr', {"version":"1.0",
                                             "layerDepthUoM":"METER",
                                             "layerDepthReference":"MV",
                                             "lithoDescrStandard":"AGT",
                                             "lithoDescrVersion":"1",
                                             "lithoDescrQuality":"C"})

        # lithology
        for i in range(len(bore[k]['lith'])):

            linterv = bore[k]['lith'].iloc[i]

            lithinterval = newNode(ldescr, 'lithoInterval',
                                   {'topDepth'  : str(linterv['From']),
                                    'baseDepth' : str(linterv['To']) } )

            extract(lithinterval, linterv['Lithology Description'])

        # Lithostrat description (if available)
        if any([f!='onbekend' for f in bore[k]['lith']['Lithostratigrafie']]):
            lsd = newNode(bho, 'lithostratDescr',
                                  {"embargo":"OPENBAAR",
                                   "layerDepthReference":"MV",
                                   "layerDepthUoM":"METER",
                                   "lithostratDescrStandard":"SGT"})

            _   = newNode(lsd, 'describerName', text = meta['Projectleider'])

            for i in range(len(bore[k]['lith'])):

                linterv = bore[k]['lith'].iloc[i]

                liv = newNode(lsd, 'lithostratInterval', {
                                'topDepth' : str(linterv['From']),
                                'baseDepth': str(linterv['To'])})

                F = linterv['Lithostratigrafie']
                _ = newNode(liv, 'lithostrat', {'code': bc.lithocode[F]}, F)

        # drilling methods
        drm = newNode(bho, 'drillMethod', {"depthReference": "MV",
                                           "depthUoM": "METER",
                                           "diameterUoM": "MILLIMETER",
                                           "version": "AGT",
                                         "diameter": str(meta['Boordiameter']),
                                           "afwerking": meta['Afwerking']})

        #boormeth = meta['Boormethode']
        boormeth = meta['Boormethode']
        _   = newNode(drm, 'drillMethodInterval',
                      {'code': boormeth.upper()[:3]},
                      text=boormeth)


        # filter screen positions:
        flt = newNode(bho, 'filters',{'filterDepthUoM': 'METER'})
        for filt, filtNm in zip([filt1, filt2, filt3],
                            ['filter1', 'filter2', 'filter3']):
            if k in filt.index:
                for i in range(len(filt.loc[k])):
                    F = filt.loc[k].iloc[i]
                    _ = newNode(flt, filtNm, {'topDepth'  : str(F['From']),
                                              'baseDepth' : str(F['To']),
                                    'outerDiam': str(F['Outer Diameter'])},
                                               text=F['Item'])

        # water level postion
        try:
            levels = waterlevel.loc[k:k, :]
            if len(levels) > 0:
                wlevel = newNode(bho, 'waterlevels', {'UoM': 'METER'},
                                 text='initial waterdepths')
                for i in range(len(levels)):
                    level = levels.iloc[i]
                    if not np.isnan(level['From']) and not np.isnan(level['To']):
                        _ = newNode(wlevel, 'initialWaterlevel',
                                    {  'filter'    : str(level['Filter']),
                                       'item'      : level['Item'],
                                       'topDepth'  : str(level['From']),
                                       'baseDepth' : str(level['To'])},
                                       text='initial waterlevel')
        except:
            pass

        if False: print(_) # dummy to do something with unused variable _

        tofile(root, path)

    print('...done, {} files!'.format(len(bore)))

    print('\nsee folder :\n',
      path if path is not None else os.path.abspath('.'), '\n')


    return bore

#%% main

if __name__ == '__main__':


    borepath = '/Users/Theo/GRWMODELS/python/2017_DEME-juliana/bores/agtdeme'

    # to generate xml files
    bores = agtbores2xml(borepath)


    agtbores = db.Bores(borepath)

    Y = sorted([(k, agtbores[k].y) for k in agtbores], key=lambda x: x[1], reverse=True)
    north2south = [y[0] for y in Y ]

    line = [(180000, 338360), (185000, 332640)]
    if True:
        agtbores.plot(fw=80, line=line, admix=False,
                      verbose=True, title='Afstand tot Heerlerheidebreuk'
                      )
    else:
        agtbores.plot(fw=80, order=north2south, admix=False, lith=True,
                      verbose=True, title='North to south', xlabel='yRD [m]',
                      name=True, fs=6, ha='center', va='top')
        full()
