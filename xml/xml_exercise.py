#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 14:36:12 2017

Exercises to parse XML files (as obtained from dinoloket for instance)

@author: Theo
"""
import os
import sys
from collections import UserDict

tools = '/Users/Theo/GRWMODELS/python/tools/'

if not tools in sys.path:
    sys.path.insert(1, tools)

import xml.etree.ElementTree as ET

def printXML(xmlfile, outline=False):
    '''prints entire XML file with indentation
    parameters
    ----------
    xmlFile : str
        name of xmlfile
    outline : bool
        whether or not to print the attributes of the nodes
    '''

    tree = ET.parse(xmlfile)
    root = tree.getroot()
    print('\n\n\n')
    print_tree(root, spaces='  ', outline=outline)


def print_tree(root, spaces='', outline=False):
    '''traverses the xml and prints tag and possibly the attrib and text
    parameters
    ----------
    root : ET.element
    space : str
        indentation spaces for next level
    outline : bool
        whether to print the attrib or not (=outline)
    '''
    print(spaces, root.tag, end='')
    if outline:
        print()
    else:
        print(root.attrib)
        spaces += '  '
    for child in root:
        print_tree(child, spaces, outline)


#%% Getting the names to parse

class Bore:
    '''borehole definition parsed form xml files from www.dinoloket.nl

    Thisis a simple test version, see the files under module tools.dino.bores
    for extended ones used in actual large sets of drillings from dinoloket.

    '''

    def __init__(self, xmlfile):
        '''returns bore read and parsed from dinoloket xml file
        parameters
        ----------
        xmlfile : str
            coplete path to xmlfile
        '''

        tree = ET.parse(xmlfile)

        for co in tree.findall('.//coordinates'):
            if co.attrib['coordSystem'] == 'RD':
                self.x = float(co.find('.//coordinateX').text)
                self.y = float(co.find('.//coordinateY').text)

        sfe = tree.find('.//surfaceElevation')
        self.se_date = (sfe.attrib['levelYear'], \
                        sfe.attrib['levelMonth'],
                        sfe.attrib['levelDay'])

        ld = root.find('.//lithoDescr')
        LI = ld.findall('.//lithoInterval')
        self.lg = []
        for i, li in enumerate(LI):
            lg = {'zt' : float(li.attrib['topDepth']) / 100., \
                  'zb' : float(li.attrib['baseDepth']) / 100.}
            for ch in li.getchildren():
                if ch.attrib:
                    lg[ch.tag] = ch.attrib[list(ch.keys())[-1]]
                elif ch.text:
                    lg[ch.tag] = ch.text
            self.lg.append(lg)

    def plot(self):
        pass

    def __repr__(self):
        '''Readable representation of bores'''
        s = 'borehole {:10}:  '.format(self.id)
        s += 'x = {:8.0f}, y = {:8.0f}, z = {:8.2f}, startDate = {}\n'.\
            format(self.x, self.y, self.z, self.startDate)
        for i, lg in enumerate(self.lg):
            s += 'Laag {:2d}:  '.format(i)
            s += '  ' + str(lg) + '\n'
        s += '\n'
        return s

class Bores(UserDict):
    def __init__(self, boredir, version=1.4, max=None):
        '''return descriptions of boreholes from dinoloket given in xml files
        parameters
        ----------
        boredir : str
            directory where the xml files are
        version : str
            version id used in name of file ('1.3 or 1.4')
        max: int
            maximum number of files to read
        '''

        self.data={}

        print(boredir, ' ', version)
        os.listdir(boredir)
        LD =[f for f in os.listdir(boredir) if f[-7:-4] == version]
        print(LD)
        print('LD=', len(LD))
        if max is None:
            max = len(LD)
        for file in LD[:max]:
            name = os.path.extsep(os.path.basename(file))[0][:-4]
            print(file)
            self.data[name] = Bore(os.path.join(boredir, file))

        def keys(self):
            return self.data.keys()
        def __getitem__(key):
            return self.data[key]

    def plot(self):
        for b in self:
            b.plot()



if __name__ == '__main__':

    '''
    xml files or drillholes as supplied by dinoloket are read
    in module tools/dino/bores. Thee files are converted to drawings
    of the borehole profiles and plotted on screen.
    Therefore, refer to dino/bores as this file is only to illustrate
    the use of Element_tree as a general way to deal with xml files.

    The module kml deals with Google Earth's kml files, which works exactly
    the same.

    TO 20171227

    '''

    boredir = os.path.join(tools, 'dino/bores/testbores')

    if not os.path.isdir(boredir):
        raise FileNotFoundError("Can't find dirctory {}".format(boredir))

    # get full path of first xml file in directory boredir:
    xmlfile = os.path.join(boredir, os.listdir(boredir)[0])

    # to plot the xml files as an indented ouline of its tags:
    printXML(xmlfile, outline=True)

    # to print the xml file in full with tags, attributes and text
    printXML(xmlfile, outline=False)

    # As an alternative drop the xml file in the browser to view it
    # in a human friendly way. Note that kml files may trigger Google Earth.

    # Parse an xmlfile
    tree = ET.parse(xmlfile)

    # It's root gives acces to anything below it.
    root = tree.getroot()

    # You may iterate through the tree to find a certain node
    #Look up some tag in the tree:
    for node in tree.iter('lithoDescr'):
        print(node.tag, node.attrib)
        nd = node

    # Here's how to grab it's attribute (if it has one)
    # You can also get its tag (nd.tag) or its text (nd.text)
    # nd.attrib.get(tag) --> give attribure of specified node in element tree
    nd.attrib.get('layerDepthUoM')

    for node in tree.iter():
        if nd.tag == 'borehole':
            print(node.tag, node.attrib)
            nd = node

    # To find an element using tree.find() or tree.findall()
    # You must precede the element string with './/'
    # Otherwise, None will be retured unless the name corresponds
    # to one of the children of the node.
    node = tree.findall('.//lithology')

    node = root.findall('.//lithology') # is also fine. Any node is legal

    bores = Bores(boredir, max=None)


