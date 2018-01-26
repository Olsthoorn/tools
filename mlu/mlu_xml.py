#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 21:55:52 2018

Module to convert a mlu data file to an xml file format.
mlu is pumping test software www.microfem.com
It stores its data and results in the so-called mlu file which is defined
in the manual also downloadable from the mentioned site.
This module converts a mlu file to an xml file.
It further defined the Mluobj class that reads an mlu.xml file and
can show its contents and plot the time drawdown curvecs for all the
observation wells that it contains.

@author: Theo 180120
"""

import os
import sys

tools ='/Users/Theo/GRWMODELS/python/tools/'

if not tools in sys.path:
    sys.path.insert(1, tools)

import numpy as np
import matplotlib.pyplot as plt
from   matplotlib.patches import Rectangle
import xml.etree.ElementTree as ET
import googlemaps.get_google_map_image as gm
from coords import rd2wgs

from colors import colors


# full screen toggle
def full(): plt.get_current_fig_manager().full_screen_toggle()

convert = {
     0: '=== MLU data file (c) 2011 Kick Hemker & Vincent Post',
     1: '=== Multilayer aquifer system - Pumping test analysis',
     2: '=== MLU Version',
     3: '=== General Information =============================',
     4: '=== Parameter Optimization ==========================',
     5: '=== Aquifer System ==================================',
     6: "=== Thickness ===  T|c = x === S|S' = x === Name ====",
     7: '=== Pumping Wells and Discharges ====================',
     8: '=== X     Y     Casing x Screen x Skin x Periods Name',
     9: '=== Well screens per layer for all pumping wells ====',
    10: '=== Observation wells and Drawdowns =================',
    11: '=== X Y Layer Casing x Screen x Skin x drawdowns Name',
    12: '=== Time graph settings ============================='}


# helper functions
def line2keys(s):
    s = s.strip().replace('=',' ')\
                 .replace("S|S'",'S')\
                 .replace('|','_or_')
    return [k.strip() for k in s.split()]

def line2toks(s):
    s = s.replace('(','').replace(')','')
    return [t.strip().replace('.','').replace(' ','_') for t in s.lower().split(':')]


# mlu file to xml file converter function
def mlu2xml(fname, fout=None):
    '''Converts an mlu file to an xml file and writes it to disk.

    parameters
    ----------
        fname: str
            full path name of mlu file
        fout : str
            full path name of xlm output file
    '''

    with open(fname, 'r') as file:
        lines = file.readlines()

    # lineNrs of key lines lines inf mlu files for navigation
    loc = dict()
    for i, line in enumerate(lines):
        for k in convert.keys():
            if convert[k] in line:
                loc[k] = i

    # generate the root element for the elementtree
    root = ET.Element('mlu_data_file')
    root.text = lines[loc[0]] + lines[loc[1]]

    toks = lines[loc[2]].split(' ') # split line in tokens
    root.attrib = {'version': toks[3][:-1], 'url': toks[-1]}

    # generate the elementtree with root as its root
    tree = ET.ElementTree(root)

    # general info node
    info = ET.Element('generalInfo')
    for line in lines[loc[3] + 1 : loc[4]]:
        toks = line2toks(line)
        if len(toks) > 1:
            info.attrib[toks[0]] = toks[-1]
    root.append(info)

    # parameter Optimization node
    parOpt = ET.Element('parOpt')
    iline = loc[4]
    toks = line2toks(lines[iline + 1])
    parOpt.attrib[toks[0]] = toks[-1]

    toks = line2toks(lines[iline + 2])
    parOpt.attrib[toks[0]] = toks[-1]

    parOpt.attrib['method'] = lines[loc[4] + 3]
    root.append(parOpt)

    # Aquifer system node
    aqSys = ET.Element('aquiferSystem')
    for line in lines[loc[5] + 1 : loc[6]]:
        toks = line2toks(line)
        aqSys.attrib[toks[0]] = toks[-1]
    root.append(aqSys)

    # aquifer and aquitard layers nodes
    nr_of_layers = loc[7] - loc[6] - 1

    if aqSys.attrib['top_aquitard'] == 'absent':
        isaquif = np.mod(np.arange(nr_of_layers, dtype=int), 2) == 0
    else:
        isaquif = np.mod(np.arange(nr_of_layers, dtype=int), 2) == 1

    for aq, iline in zip(isaquif, range(loc[6] + 1, loc[7])):
        keys = line2keys(lines[loc[6]])
        if aq:
            layer = ET.Element('aquifer')
            keys[keys.index('T_or_c')] = 'T'
        else:
            layer = ET.Element('aquitard')
            keys[keys.index('T_or_c')] = 'c'
        values = lines[iline].split()
        for v in values[6:len(values)]:
            values[5] = values[5] + '_' + v
        for k, value in zip(keys, values):
            layer.attrib[k] = value
        layer.attrib.pop('x')
        aqSys.append(layer)


    # wells and discharges (wells node)
    wells = ET.Element('wells')
    toks = line2toks(lines[loc[7] + 1])
    wells.attrib[toks[0]] = toks[-1]
    root.append(wells)

    nr_of_wells = int(toks[-1])

    keys = line2keys(lines[loc[8]])
    iline = loc[8]
    # wells (well nodes)
    for iw in range(nr_of_wells):
        well = ET.Element('well')
        values = lines[iline + 1].split()
        for k, value in zip(keys, values):
            well.attrib[k] = value
        values = lines[iline + 2].split()
        well.attrib['tstart'] = values[0]
        well.attrib['Q'] = values[-1]
        well.attrib.pop('x')
        wells.append(well)
        iline += 2

    iline = loc[9] + 1
    naquif = len(aqSys.findall('aquifer'))
    screens = ''
    for i in range(naquif):
        n = lines[iline + i].split()[0]
        if n == '.':
            screens = screens + '0'
        else:
            screens = screens + n
    well.attrib['screens'] = screens

    # observation well nodes
    obsWells = ET.Element('obsWells')
    nr_of_obsWells = lines[loc[10] + 1].split(':')[-1].strip()
    obsWells.attrib['nr_of_obsWells'] = nr_of_obsWells
    root.append(obsWells)

    keys = line2keys(lines[loc[11]])
    iline = loc[11] + 1
    for iobs in range(int(nr_of_obsWells)):
        values = lines[iline].split()
        obsWell = ET.Element('obsWell')
        for key, value in zip(keys, values):
            obsWell.attrib[key] = value
        obsWell.attrib.pop('x')
        n = abs(int(obsWell.attrib['drawdowns']))
        obsWell.text ='\n'
        for iL in range(iline + 1, iline + n + 1):
            a, b = lines[iL].split()
            obsWell.text = obsWell.text + a + ',' + b + '\n'
        obsWells.append(obsWell)
        iline += n + 1

    if fout is not None:
        tree.write(fout)

    return tree

# ############# Mlu xml file as a python object ###############################

#====== helper classes used in Mluobj class====================================

class Layer:
    '''Layer object, will be inherited from'''
    def __init__(self, layer):
        '''Create and return Layer object
        parameters
        ----------
            layer : ET.Element
        '''
        self.name =       layer.attrib['Name']
        self.D    = float(layer.attrib['Thickness'])


class Aquitard(Layer):
    '''Aquitard object'''
    def __init__(self, layer):
        '''Create and return an Aquitard instance.
        parameters
        ----------
            layer: ET.Element with tag 'aquitard'
        '''
        super().__init__(layer)

        self.type =       layer.tag
        self.c    = float(layer.attrib['c'])
        self.S    = float(layer.attrib['S'])

    def plot(self, xlim=(0, 500), **kwargs):
        ax = kwargs.pop('ax')
        p = Rectangle( (xlim[0], self.zbot ),
                      np.diff(xlim), self.ztop - self.zbot, **kwargs)
        ax.add_patch(p)


class Aquifer(Layer):
    '''Aquifer object'''
    def __init__(self, layer):
        '''Create and return an Aquifer instance.
        parameters
        ----------
            layer: ET.Element with tag 'aquifer'
        '''
        super().__init__(layer)

        self.type =       layer.tag
        self.T    = float(layer.attrib['T'])
        self.S    = float(layer.attrib['S'])

    def plot(self, xlim=(0, 500), **kwargs):
        ax = kwargs.pop('ax')
        p = Rectangle( (xlim[0], self.zbot),
                      np.diff(xlim), self.ztop - self.zbot, **kwargs)
        ax.add_patch(p)


class AqSys:
    '''Aquifer system class.'''
    def __init__(self, aqSys):
        '''Create an aquifer system instance.
        parameters
        ----------
            aqSys: ET.Element with tag 'aquiferSystem'
        '''
        self.base_aquitard =       aqSys.attrib['base_aquitard']
        self.top_aquitard  =       aqSys.attrib['top_aquitard']
        self.naquif        = int(  aqSys.attrib['no_of_subaquifers'])
        self.UoL           =       aqSys.attrib['length']
        self.UoT           =       aqSys.attrib['time']
        self.UoQ           = '{}3/{}'.format(self.UoL, self.UoT)
        self.top_level     = float(aqSys.attrib['top_level'])

        if self.top_level > 1000: self.top_level = 0.0

        self.aquifs = list()
        self.atards = list()

        ztop = self.top_level
        for i, layer in enumerate(aqSys):
            if layer.tag == 'aquifer':
                aquif = Aquifer(layer)
                aquif.ztop = ztop
                aquif.zbot = ztop - aquif.D
                aquif.zmid = ztop - aquif.D / 2
                ztop -= aquif.D
                self.aquifs.append(aquif)
            else:
                atard = Aquitard(layer)
                atard.ztop = ztop
                atard.zbot = ztop - atard.D
                ztop -= atard.D
                self.atards.append(atard)


    def plot(self, **kwargs):
        '''Plot aquifer system.'''
        atColor = kwargs.pop('atColor', 'g')
        aqColor = kwargs.pop('aqColor', 'y')
        for atrd in self.atards:
            atrd.plot(facecolor=atColor, edgecolor='k', **kwargs)
        for aqui in self.aquifs:
            aqui.plot(facecolor=aqColor, edgecolor='k', **kwargs)


class GeneralWell:
    '''General well class.'''
    def __init__(self, well):
        '''Create a generalWell instance.
        parameters
        ----------
            well: ET.Element object with tag 'well'
        '''
        self.name   = well.attrib['Name']
        self.casing = float(well.attrib['Casing'])
        self.screen = float(well.attrib['Screen'])
        self.skin   = float(well.attrib['Skin'])
        self.x      = float(well.attrib['X'])
        self.y      = float(well.attrib['Y'])


class Well(GeneralWell):
    '''Well class.'''
    def __init__(self, well):
        '''Create a Well instance.
        parameters
        ----------
            well: ET.Element object with tag 'well'
        '''
        super().__init__(well)
        self.tstart = float(well.attrib['tstart'])
        self.periods= int(  well.attrib['Periods'])
        self.Q      = float(well.attrib['Q'])
        self.screens=       well.attrib['screens']

    def plot(self, aqSys, **kwargs):
        '''Plot a Well.
        parameters
        ----------
            aqSys: aqSys instance
        '''
        if not 'ax' in kwargs:
            raise ValueError('ax not in kwargs')
        ax = kwargs.pop('ax')
        for i, aqf in enumerate(aqSys.aquifs):
            if int(self.screens[i]) > 0:
                #p = Rectangle((self.screen, aqf.zbot),
                #             self.screen,
                #             aqf.ztop - aqf.zbot, **kwargs)
                #ax.add_patch(p)
                ax.plot(self.screen, aqf.zmid, 'ro')
                ax.text(self.screen, aqf.zmid, self.name,
                        rotation=45, ha='left', va='bottom')


class ObsWell(GeneralWell):
    '''Observation well class.'''
    def __init__(self, ow):
        '''Create insance of an observation well.
        parameters
        ----------
            ow : ET.Element with tag obsWell
        '''
        super().__init__(ow)
        self.layer  = int(ow.attrib['Layer'])
        self.data = [d.split(',') for d in ow.text.split()]
        self.data = np.array([[float(a), float(b)] for a, b in self.data])


    def dist(self, well):
        '''Return distance between this obsWell and well.
        parameters
        ----------
            well: Well instance
        '''
        return np.sqrt((self.x - well.x)**2 + (self.y - well.y)**2)

    def plot(self, aqSys, r, lscreen=1.0, **kwargs):
        '''Plot observation well.
        parameters
        ----------
        aqSys : aquifer system instance.
        r : float
            distance between this obsWell to well
        '''
        if not 'ax' in kwargs:
            raise ValueError('ax not in kwargs')
        ax = kwargs.pop('ax')
        aqf = aqSys.aquifs[self.layer - 1]
        z = aqf.zmid -0.5 * lscreen
        #h = self.screen
        #p = Rectangle( (r, z), h, lscreen, **kwargs)
        #ax.add_patch(p)
        ax.plot(r, z, 'ro')
        ax.text(r, z, self.name, rotation=45, ha='left', va='bottom')


# =================== Class Mluobj ============================================
class Mluobj:
    "Class that contains an mlu data file"

    def __init__(self, xmlfileName):
        '''
        parameters
        ----------
            xmlfileName : str
                name of the xml file that constains the mlu data
        '''
        self.tree = ET.parse(xmlfileName)

        self.meta = self.tree.find('.//generalInfo').attrib

        self.aqSys     = AqSys(self.tree.find('.//aquiferSystem'))

        self.wells = list()
        for w in self.tree.findall('.//well'):
            self.wells.append(Well(w))

        self.obsWells = list()
        for ow in self.tree.findall('.//obsWell'):
            self.obsWells.append(ObsWell(ow))

    def r(self, ow, wellNr=0):
        r = np.sqrt((ow.x - self.wells[wellNr].x)**2 +
                    (ow.y - self.wells[wellNr].y)**2)
        return r

    @property
    def obsNames(self):
        '''Return the list of the names of the observation wells.'''

        obsWells = self.tree.find('.//obsWells')
        return [ow.attrib['Name'] for ow in obsWells]


    @property
    def rmax(self):
        '''Return distance to observatin well farthest from wells[0].'''
        r = 0.0
        for ow in self.obsWells:
            r = max(r,
             np.sqrt((self.wells[0].x - ow.x)**2 + (self.wells[0].y - ow.y)**2))
        return r

    @property
    def rmin(self):
        '''REturn screen radius of  wells[0]'''
        return self.wells[0].screen / 2.0


    # ~~~~~~~~~~~~~ plotting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def plot_aquif(self, **kwargs):
        self.aqSys.plot(**kwargs)


    def plot_wells(self, **kwargs):
        for w in self.wells:
            w.plot(**kwargs)


    def plot_obswells(self, **kwargs):
        for ow in self.obswells:
            ow.plot(**kwargs)


    def plot_drawdown(self, piezomNames, tshift=0., **kwargs):

        if isinstance(piezomNames, str):
            piezomNames = [piezomNames]

        obsNames = [ow.name for ow in self.obsWells]

        notFound = []
        for piezNm in piezomNames:
            if piezNm not in obsNames:
                notFound.append(piezNm)
        if len(notFound) > 0:
            print('The following piezom names were not in obsNames:')
            print(notFound)
            print('Known obsNames are:')
            print(piezomNames)
            raise FileNotFoundError('piezom not found')

        kw = {'title': '{}, {}, {}, Q={:.0f} {}'.format(
                  self.meta['name'],
                  self.meta['test'],
                  self.meta['date'],
                  self.wells[0].Q, self.aqSys.UoQ),
              'xlabel': 't ' + self.aqSys.UoT,
              'ylabel': '<-- drawdown [' + self.aqSys.UoL + ']',
              'xscale': 'log',
              'yscale': 'linear',
              'grid' : True}

        fig, ax, kwargs = getFigAx(kw, kwargs)

        if 'xlim' in kwargs:
           ax.set_xlim(kwargs.pop('xlim'))
        if 'ylim' in kwargs:
            ax.set_ylim(kwargs.pop('ylim'))

        ax.invert_yaxis()

        for piezom, color in zip(piezomNames, colors):
            ow = self.obsWells[obsNames.index(piezom)]
            plt.plot(ow.data[:, 0] - tshift, ow.data[:, 1],
                     label=piezom + ' [r = {:.3g} {}]'\
                     .format(self.r(ow), self.aqSys.UoL),
                     color=color, **kwargs)

        plt.legend(loc='best')

        return ax


    def plot_plan(self, **kwargs):
        '''Plot a plan view'''

        marker = kwargs.pop('marker','o')
        ha    = kwargs.pop('ha', 'right')
        va    = kwargs.pop('va', 'bottom')
        color = kwargs.pop('color', 'red')

        kw = {'title': '{}, {}, {}, Q={:.0f} {}'.format(
                  self.meta['name'],
                  self.meta['test'],
                  self.meta['date'],
                  self.wells[0].Q, self.aqSys.UoQ),
          'xlabel': 'x ' + self.aqSys.UoL,
          'ylabel': 'y ' + self.aqSys.UoL,
          'xscale': 'linear',
          'yscale': 'linear',
          'grid' : True}

        fig, ax, kwargs = getFigAx(kw, kwargs)

        colors  =[(f, f, 0) for f in np.arange(0.1,1.1,0.1)]
        msize = np.arange(5, 25, 3)
        dy = 10.
        for ow in self.obsWells:
            ax.plot(ow.x, ow.y ,
                        marker,
                        mfc='none',
                        mec=colors[ow.layer],
                        ms=msize[ow.layer])
        dy = np.diff(ax.get_ylim()) / 60
        for ow in self.obsWells:
            ax.text(ow.x, ow.y + dy * ow.layer, '{}, aq {}'.format(ow.name, ow.layer), ha=ha, va=va, **kwargs)

        for w in self.wells:
            ax.plot(w.x, w.y, marker, mfc='none', mec='red', ms=msize[-1])
            ax.text(w.x, w.y, w.name, ha=ha, va=va, color=color)


    def plot_map(self, **kwargs):
        '''Plots a map of pumping test situation, using Google Maps.'''

        x = np.array([o.x for o in self.obsWells])
        y = np.array([o.y for o in self.obsWells])

        Lon, Lat = rd2wgs(x, y)

        Loc = ['{:6f},{:6f}'.format(lon,lat) for lon, lat in zip(Lon, Lat)]

        center = '{:.6f},{:.6f}'.format(np.mean(Lon), np.mean(Lat))

        markers=list()
        for loc, lbl in zip(Loc, 'AQuickBrownFoxJumpsOverTheLaZyDog'.upper()):
            markers.append('size:mid|color:red|label:{}|{}'.format(lbl, loc))

            #print("{} {} {}".format(o.name, o.x, o.y))

        img = gm.get_image(center, crs='RD', markers=markers, **kwargs)

        return img



    def plot_section(self, aqColor='y', atColor='g', **kwargs):
        '''Plot a cross sectional view in r,z coordinates.'''

        kw = {'title': '{}, {}, {}, Q={:.0f} {}'.format(
                        self.meta['name'],
                        self.meta['test'],
                        self.meta['date'],
                        self.wells[0].Q, self.aqSys.UoQ),
          'xlabel': 'r [{}]'.format(self.aqSys.UoL),
          'ylabel': 'z [{}]'.format(self.aqSys.UoL),
          'xscale': 'log',
          'yscale': 'linear',
          'grid' : True}

        fig, ax, kwargs = getFigAx(kw, kwargs)

        xlim = (self.rmin, self.rmax * 1.2)
        ylim = (min(self.aqSys.aquifs[-1].zbot, self.aqSys.atards[-1].zbot),
                max(self.aqSys.aquifs[ 0].ztop, self.aqSys.atards[ 0].ztop))

        self.aqSys.plot(ax=ax, xlim=xlim, aqColor=aqColor, atColor=atColor, **kwargs)

        for w in self.wells:
            w.plot(self.aqSys, ax=ax, **kwargs)
            ax.text(w.x, w.y, w.name, rotation=45)
        for o in self.obsWells:
            o.plot(self.aqSys, self.r(o), ax=ax, **kwargs)
            ax.text(o.x, o.y, o.name, rotation=45)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)


# ============ helper function to set up a plot axis ==========================
def getFigAx(kw, kwargs):
    '''Return fig, ax an remaining kwargs after setting up a figure axis.

    Convenience function to set up a figure with axis.
    '''
    ax     = kwargs.pop('ax',     None)
    title  = kwargs.pop('title',  kw['title'])
    xlabel = kwargs.pop('xlabel', kw['xlabel'])
    ylabel = kwargs.pop('ylabel', kw['ylabel'])
    xscale = kwargs.pop('xscale', kw['xscale'])
    yscale = kwargs.pop('yscale', kw['yscale'])
    grid   = kwargs.pop('grid',   kw['grid'])

    if ax     is None:
        fig, ax = plt.subplots()

    ax.set(title=title, xlabel=xlabel, ylabel=ylabel,
           xscale=xscale, yscale=yscale)
    ax.grid(grid)

    return fig, ax, kwargs


# =====================  MAIN =================================================
if __name__ == '__main__':

    import warnings
    warnings.simplefilter("error")

    fname= os.path.join(tools, 'mlu/testdata/gat_boomse_klei.mlu')
    fname= os.path.join(tools, 'mlu/testdata/zuidelijke_landtong.mlu')
    tree = mlu2xml(fname)
    print('Done')

    fn_out = os.path.splitext(os.path.basename(fname))[0] + '.xml'
    tree.write(fn_out, xml_declaration=True)


    mu = Mluobj(fn_out)

    #mu.plot_drawdown('PB1-1')
    mu.plot_drawdown(mu.obsNames, yscale='linear', xscale='log', marker='.')

    #mu.plot_drawdown(['PB2-1', 'PB2-2'])


    mu.plot_section(xscale='log')

    #mu.plot_map(maptype='hybrid')

    #img = mu.plot_plan()

    #img