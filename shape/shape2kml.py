#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 21:55:06 2017

@author: Theo
"""

tools   = '/Users/Theo/GRWMODELS/python/tools/'

import os
import sys

if not tools in sys.path:
    sys.path.insert(1, tools)

import shapefile
import xml.etree.ElementTree as ET
import copy
import coords
import numpy as np

template = '/Users/Theo/GRWMODELS/python/tools/kml/data/shape2kmlTemplate.kml'

def xy2en(co):
    '''return [(e,n),...] form [(x,y),...]
    parameters
    ----------
    co :
        list of coordinates tuples from shape
    returns
    -------
    string of coordinates in e, n, 0
    '''
    co = np.array(co)
    E, N = coords.rd2wgs(co[:, 0], co[:, 1])
    s = '\n\t\t\t\t\t\t\t'
    for e, n in zip(E, N):
        s = s + '{},{},0 '.format(e, n)
    s = s[:-1] + '\n\t\t\t\t\t\t'
    return s


def shape2kml(shapefilename, kmlfilename=None, verbose=False):
    '''convert shapefile (rd coordinates) to kmlfile (EN) coordinates

    The shapefile is assumed to consist of  polygones in rd coordinates.
    paramters
    ---------
    shapefilename : str
        full path to shapefilename
    kmlfilename : str
        basname of kmlfilename to be written
    '''

    if verbose:
        print('Converting {} to {}'.format(shapefilename, kmlfilename))

    if kmlfilename is None:
        kmlfilename = os.path.basename(shapefilename)

    # First open the shapefile to be converted to kml
    rdr = shapefile.Reader(shapefilename)
    records = rdr.records()
    shapes  = rdr.shapes()
    if verbose:
        print('{} read'.format(shapefilename))

    # Open the kml folder template
    tree   = ET.parse(template)

    # find the string that is prepended to tags
    root   = tree.getroot()
    pre    = root.tag.split('}')[0] + '}'

    # get the Foler annd Placemak nodes
    folder = tree.find('.//' + pre + 'Folder')
    foldNm = folder.find('.//' + pre + 'name')

    # replace folder name in that of kmlfile
    foldNm.text = os.path.basename(kmlfilename)

    plmark = tree.find('.//' + pre + 'Placemark')
    if verbose:
        print('template analyzed: {}'.format(template))

    # deepcopy the placemark node
    p      = copy.deepcopy(plmark)

    # and tet name an coordinate nodes in this copy
    # which are to be replaced by values from shapefiles
    nm     = p.find('.//' + pre + 'name')
    co     = p.find('.//' + pre + 'coordinates')

    # remove placemark node from template
    folder.remove(plmark)

    # iter over all shapes in the shapefile
    for rec, shp in zip(records, shapes):

        # replace the text of the name node
        nm.text = rec[0]
        if verbose:
            print('record name =:', nm.text)

        # replace coordinate string by new one
        co.text = xy2en(shp.points)
        if verbose:
            print('{} coordinates added to placemark.'.format(len(shp.points)))

        # append a deep copy of this placemark to folder node
        folder.append(copy.deepcopy(p))
        if verbose:
            print('record appended')

    kmlfilename = os.path.splitext(kmlfilename)[0] + '.kml'

    tree.write(kmlfilename,
               encoding='UTF-8',
               xml_declaration=True,
               short_empty_elements=True)

    if verbose:
        print('{} written in folder\n{}\n'.
              format(kmlfilename, os.path.abspath('.')))


if __name__ == '__main__':

    shpdir = '/Users/Theo/GRWMODELS/python/DEME-juliana/shapes/'

    maasterrassen  = 'Maasterras'

    shapefilename = os.path.join(shpdir, maasterrassen)
    kmlfilename = 'Maasterrassen3'

    shape2kml(shapefilename, kmlfilename, verbose=True)

