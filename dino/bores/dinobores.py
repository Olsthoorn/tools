#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 14:36:12 2017

Exercises to parse XML files (as obtained from dinoloket for instance)

@author: Theo
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import matplotlib.patches as patches
import warnings

# todo: change for the actual user!
tools = '/Users/Theo/GRWMODELS/python/tools'
if not tools in sys.path:
    sys.path.insert(1, tools)

import dino.bores.dinoborecodes as dcodes
from coords import inpoly

# import shapefile  replace import to specific module

import logging

logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
logging.debug('Start of '.format(__name__))

AND = np.logical_and
NOT = np.logical_not
OR  = np.logical_or

def full(): plt.get_current_fig_manager().full_screen_toggle()

#%% Getting the names to parse

def xmlprint(xmlfile):
    '''Pretty print xmlfile.'''

    def treeprint(sp, root):
        sp = sp + '  '
        print('{}<{}'.format(sp, root.tag), end='')
        if root.attrib:
            for k in root.attrib:
                print(' ""{}": "{}"'.format(k, root.attrib[k]), end='')
        print(">")
        if root.text:
            print('{}  {}'.format(sp, root.text))
        print('{}</{}>'.format(sp, root.tag))
        for child in root.getchildren():
            treeprint(sp, child)
    tree = ET.parse(xmlfile)
    treeprint('', tree.getroot())


class Bore:
    '''Borehole definition parsed form xml files from www.dinoloket.nl
    '''

    def __init__(self, xmlfile=None, verbose=True):
        '''Returns a bore that was read and parsed from dinoloket xml file

        Args:
            xmlfile (str):
                coplete path to xmlfile
        '''

        if verbose:
            print(xmlfile)

        self.name = os.path.basename(xmlfile).split('_')[0]

        tree = ET.parse(xmlfile)

        pointSurvey    = tree.find('.//pointSurvey')
        surveyLocation = pointSurvey.find('.//surveyLocation')
        borehole       = pointSurvey.find('.//borehole')

        self.baseDepth = float(borehole.attrib['baseDepth'])
        if borehole.attrib['baseDepthUoM']=='CENTIMETER':
                                self.baseDepth /= 100.

        for id in pointSurvey.findall('.//identification'):
            self.id = id.attrib['id']

        for co in surveyLocation.findall('.//coordinates'):
            if co.attrib['coordSystem'] == 'RD':
                self.x = float(co.find('.//coordinateX').text)
                self.y = float(co.find('.//coordinateY').text)

        surfaceElevation = pointSurvey.find('.//surfaceElevation')
        try:
            try:
                self.levelDate = \
                           (surfaceElevation.attrib['levelYear'],
                            surfaceElevation.attrib['levelMonth'],
                            surfaceElevation.attrib['levelDay'])
            except:
                pass

            elevation = surfaceElevation.find('.//elevation')

            self.ref  = elevation.attrib['levelReference']
            self.ztop = float(elevation.attrib['levelValue'])
            if elevation.attrib['UoM'] == 'CENTIMETER':
                                    self.ztop /= 100.
            self.zbase = self.ztop - self.baseDepth
        except:
            print('Except: bore {} has no elevation.'.format(xmlfile))
            self.ztop = np.nan
            return


        date = borehole.find('.//date')
        if date:
            if date.attrib:
                self.startDate = (date.attrib['startYear'],
                                  date.attrib['startMonth'],
                                  date.attrib['startDay'])
            else:
                self.startDate=''

        # Lithodescription
        lithoDescr = borehole.find('.//lithoDescr')
        UoM = lithoDescr.attrib['layerDepthUoM']
        lithoInterval = lithoDescr.findall('.//lithoInterval')
        self.lith = {}
        for i, lith in enumerate(lithoInterval):
            self.lith[i] = {'topDepth' : float(lith.attrib['topDepth']), \
                            'baseDepth' : float(lith.attrib['baseDepth'])}
            if UoM == 'CENTIMETER':
                self.lith[i]['topDepth']  /= 100.
                self.lith[i]['baseDepth'] /= 100.
            for child in lith.getchildren():
                try:
                    self.lith[i][child.tag] = child.attrib['code']
                except:
                    pass

        try:
            lithostratDescr = borehole.find('.//lithostratDescr')
            UoM = lithostratDescr.attrib['layerDepthUoM']
            lithostratInterval = lithostratDescr.findall('.//lithostratInterval')
            self.strat = dict()
            for i, lisi in enumerate(lithostratInterval):
                self.strat[i] = {'baseDepth': float(lisi.attrib['baseDepth'])}
                if UoM == 'CENTIMETER' : self.strat[i]['baseDepth'] /= 100.
                for child in lisi.getchildren():
                    try:
                        self.strat[i][child.tag] = child.attrib['code']
                        self.strat[i]['text'] = child.text
                    except:
                        pass
        except:
            pass

        self.filters=[]
        try:
            filtersNd = borehole.find('.//filters')
            UoM = filtersNd.attrib['filterDepthUoM']
            for F in ['filter1', 'filter2', 'filter3']:
                try:
                    filts = filtersNd.findall('.//' + F)
                    if len(filts) > 0:
                        for flt in filts:
                            if flt.text =='bkb':
                                bkb = float(flt.attrib['topDepth'])
                                diameter = float(flt.attrib['outerDiam'])
                            if flt.text.lower().startswith('filter'):
                                topDepth  = float(flt.attrib['topDepth'])
                                baseDepth = float(flt.attrib['baseDepth'])
                                if UoM == 'CENTIMETER':
                                    bkb /= 100.
                                    topDepth  /= 100.
                                    baseDepth /= 100.
                        self.filters.append(
                                (self.ztop - bkb, topDepth, baseDepth, diameter))
                except:
                    pass
        except:
            pass


        try:
            waterlevels = borehole.find('.//waterlevels')
            initialWaterlevel = waterlevels.findall('.//initialWaterlevel')
            if len(initialWaterlevel) > 0:
                self.initialWaterlevel = dict()
                for level in initialWaterlevel:
                    f = 'filter' + level.attrib['filter']
                    self.initialWaterlevel[f]=\
                     {'UoM': waterlevels.attrib['UoM'], **level.attrib,
                        'elev' : self.ztop - 0.5 *
                        (float(level.attrib['topDepth']) +\
                         float(level.attrib['baseDepth']))}
                    self.initialWaterlevel[f].pop('filter')
        except:
            pass

        try:
            drillMethod = borehole.find('.//drillMethod')
            drillMethodInterval = drillMethod.find('.//drillMethodInterval')
            self.drillMethod = dict()
            self.drillMethod['code'] = drillMethodInterval.attrib['code']
            self.drillMethod['text'] = drillMethodInterval.text
        except:
            pass


    def __repr__(self):
        '''Return readable representation of bores'''

        s = 'borehole {:10}:  '.format(self.id)
        s += 'x = {:8.0f}, y = {:8.0f}, z = {:8.2f}\n'.\
            format(self.x, self.y, self.ztop)
        for i in self.lith.keys():
            s += 'Lith {:2d}:  '.format(i)
            s += '  ' + str(self.lith[i]) + '\n'
        try:
            for i in self.strat.keys():
                s += 'Strat {:2d}: '.format(i)
                s += '  ' + str(self.strat[i]) + '\n'
                s += 'drillMethod = ' + str(self.drillMethod) + '\n'
        except:
            pass
        s += '\n'
        return s


    def plot(self, fs=6, fw=80, lith=True, admix=True, strat=True,
                                                 filters=True, waterlevel=True, **kwargs):
        '''Plot single borehole.

        Args:
            fs (int):
                fontsize for text at drillings
            fw (float):
                Faction of xlim to used as width of drawn borehole.
            lith ([True] | False):
                Plot lithology.
            admix ([True] | False):
                Plat admixtures.
            strat ([True] | False):
                Plot stratigraphy is available
            filters ([True] | False):
                Plot filter screens if available.
        '''
        name = kwargs.pop('name', False)
        rotation = kwargs.pop('rotation', 90)
        ha = kwargs.pop('ha', 'left')
        va = kwargs.pop('va', 'bottom')

        toff = 0.1

        x     = kwargs.pop('x', self.x)

        w    = {'V'  : 0.4,
                'K'  : 0.4,
                'BRK': 0.4,
                'L'  : 0.6,
                'Z'  : 0.8,
                'GKZ': 0.8,
                'G'  : 1.0}

        clr  = {'V'  : 'black',
                'K'  : 'darkolivegreen',
                'BRK': 'sienna',
                'L'  : 'green',
                'Z'  : 'yellow',
                'GKZ': 'green',
                'G'  : 'darkorange'}

        dcol = 0.1

        dxAdmixDict= {'humusAdmix': -dcol * 7,
                  'clayAdmix':      -dcol * 6,
                  'siltAdmix':      -dcol * 5,
                  'sandAdmix':      -dcol * 4,
                  'gravelAdmix':    -dcol * 3}

        admixCode= {'humusAdmix': 'H',
                  'clayAdmix':    'K',
                  'siltAdmix':    'S',
                  'sandAdmix':    'Z',
                  'gravelAdmix':  'G'}

        clrAdmixDict= {'humusAdmix':     dcodes.humusAdmixColor,
                  'clayAdmix':      dcodes.clayAdmixColor,
                  'siltAdmix':      dcodes.siltAdmixColor,
                  'sandAdmix':      dcodes.sandAdmixColor,
                  'gravelAdmix':    dcodes.gravelAdmixColor}


        ax = kwargs.pop('ax', None)
        if ax is None:
            fw = 1.0
            fig, ax = plt.subplots()
            ax.set_title(self.id)
            ax.set_xlabel("distance from first well [m]")
            ax.set_ylabel("elevation relative to NAP [m]")
            ax.grid(axis='y')
            ax.set_xlim(x + np.array([-1.25, 1.25]))
            ax.set_ylim(np.array([self.zbase, self.ztop]) + np.array([-2., 2.]))
            ax.text(self.x, self.ztop, "{}, ({:.0f}, {:.0f})".
                    format(self.id, self.x, self.y),
                    va='bottom', ha='left')
        else:
            fw = float(np.diff(ax.get_xlim()) / fw)

        if lith == True or admix==True:
            for i in self.lith.keys():
                zt = self.ztop - self.lith[i]['topDepth']
                zb = self.ztop - self.lith[i]['baseDepth']
                zm = 0.5 * (zt + zb)
                L  = self.lith[i]['lithology']

                try:
                    C = clr[L]
                    W = w[L]
                    if L == 'Z':
                        try:
                            code = self.lith[i]['sandMedianClass']
                            dw = dcol * (1 - 6 / (dcodes.sandMedianClassWidth[code] + 1))
                            W += dw
                            C = dcodes.sandMedianClassColor[code]
                        except:
                            pass
                    if L == 'G':
                        try:
                            code = self.lith[i]['gravelMedianClass']
                            dw = dcol * (1 - 4 / (dcodes.gravelMedianClassWidth[code] + 1))
                            W += dw
                            C = dcodes.gravelMedianClassColor[code]
                        except:
                            pass
                except:
                    C = 'lightgray'
                    W = 0.75

                if lith==True:
                    p = patches.Rectangle((x, zb), fw * W, zt - zb,
                                          faceColor=C, edgeColor='k', lineWidth=0.5)
                    ax.add_patch(p)
                    ax.text(x + toff, zm, L, va='center', ha='left', fontsize=fs)


                if admix==True:
                    for adm in dxAdmixDict:
                        dx = dxAdmixDict[adm]
                        p = patches.Rectangle((x + fw * dx, zb), fw * dcol, zt - zb,
                              faceColor='w', edgeColor='k', lineWidth=0.1)
                        ax.add_patch(p)
                        if i==0:
                            ax.text(x + (dx + 0.5 * dcol) * fw, zt, admixCode[adm],
                                    ha='center', va='bottom', fontsize=fs)

                    for adm in set(dxAdmixDict.keys()).intersection(set(self.lith[i].keys())):
                        try:
                            code = self.lith[i][adm]
                            self.plotAdmix(ax, adm, code, x, zt, zb, dxAdmixDict[adm],
                                           dcol, fw, clrAdmixDict[adm][code])
                        except:
                            pass

        if strat: #plot stratygraphy
            try:
                zt = self.ztop
                if self.id == 'PB-T-010':
                    print(self.id)
                for i in range(len(self.strat)):
                    zb = self.ztop - self.strat[i]['baseDepth']
                    zm = 0.5 * (zt + zb)
                    lithoCode = self.strat[i]['lithostrat']
                    fc = dcodes.lithostratColor[lithoCode]
                    dcol = 0.5
                    p = patches.Rectangle((x - 2 * dcol * fw, zb), fw * dcol, zt - zb,
                                          faceColor=fc, edgeColor='k', lineWidth=0.5)
                    ax.add_patch(p)
                    ax.text(x - 1.5 * dcol * fw, zm, lithoCode,
                            va='center', ha='center', fontsize=fs)
                    zt = zb
            except:
                pass

        if filters:
            if len(self.filters) > 0:
                self.plotFilters(ax, x, dcol, fw, fc='blue')

        if waterlevel:
            if 'initialWaterlevel' in dir(self):
                self.plotWaterlevel(ax, x, fw, **kwargs)

        if name == True: # Plot the name above the boring
            #ax.plot(x, self.ztop + 0.25, 'r.')
            ax.text(x, self.ztop + 0.25, '  ' + self.name,
                    ha=ha, va=va, rotation=rotation, fontsize=fs)


    def plotAdmix(self, ax, adm, code, x, zt, zb, dx, dcol, fw, fc):
        '''plots admixture.

        Args:
            ax (Axes):
                axes to plot on
            adm (str):
                admixture name
            code (str):
                admixture code (short name acc. to TNO)
            x, zt, zb (floats):
                location and elevations of well
            dx (float):
                column offset relative coordinates
            dcol (float):
                column width relative coordinate
            fw (float):
                factor to multiply widths to world coordinates
            fc (str) :
                facecolor used for patch indicating admixture type
        '''
        try:
            p = patches.Rectangle((x + dx *fw, zb), fw * dcol, zt - zb,
                    faceColor=fc, edgeColor='k', lineWidth=0.5)
            ax.add_patch(p)
            #ax.plot(x, zb, 'o', color=fc)

        except:
            raise Warning('check color {}  for admix = {})'.format(fc, adm))


    def plotFilters(self, ax, x, dcol, fw, fc):
        '''plots the screens in the borehole.

        Args:
            ax (Axes):
                axes o plot on
            x (float):
                location of well on plot
            fw (float):
                scale factor x-axis for well drawing
            fc (str):
                facecolor used for filling the filter rectangle.
        '''

        for f in self.filters:
            zbkb, d1, d2, diam = f
            diam /= 1000.
            if True:
                p = patches.Rectangle((x - fw * diam, self.ztop - d2), 2 * fw * diam, d2 - d1,
                                  faceColor=fc, edgeColor='k', lineWidth=0.5)
            else:
                p = patches.Rectangle((x - fw * dcol, self.ztop - d2), fw * dcol, d2 - d1,
                                  faceColor=fc, edgeColor='k', lineWidth=0.5)
            ax.add_patch(p)


    def plotWaterlevel(self, ax, x, fw, **kwargs):
        '''plots the initialWaterlevel in the well.

        Args:
            ax (Axes):
                axes o plot on
            x (float):
                location of well on plot
            fw (float):
                scale factor x-axis for well drawing
        kwargs:
            w (float):
                with to plot water level (w=0.25 is default.
            lw (float):
                linewidth (default=3.)
            color (str):
                color used to plot water level, color='blue' is default.
        '''
        color = kwargs.pop('color', 'blue')
        w     = kwargs.pop('w', 0.25)

        for key in self.initialWaterlevel.keys():
            if self.initialWaterlevel[key]['item'] != 'Droog':
                p = self.initialWaterlevel[key]['elev']
                ax.plot([x - fw * w, x + fw * w], [p, p], color=color, lw = 2)


from collections import UserDict

class Bores(UserDict):
    '''Container for bores (boreholes) read from xml files

    plot :
        plots the borehole stratigraphy on a section
    select :
        returns a subset of bores

    '''
    def __init__(self, boredir=None, **kwargs):
        '''Return descriptions of boreholes from dinoloket given in xml files.

        Args:
            boredir (str):
                path to folder where the xml files are
        Kwargs:
            version (str | [None]):
                version string used in xml file name ('1.3 or 1.4')
            n (int | [None]):
                maximum number of files to read
            verbose ([False] | True):
                yields more info during processing
        '''

        self.data = {}

        if boredir is None:
            return

        version = kwargs.pop('version', None)
        verbose = kwargs.pop('verbose', False)
        n       = kwargs.pop('n'      , None)

        assert os.path.isdir(boredir), "Not a directory:\n{}".format(boredir)

        if version is None:
            LD =[f for f in os.listdir(boredir) if f[-7:-4] == '1.4']
            # Default xml version of TNO, use of available
            if len(LD) == 0:
                LD =[f for f in os.listdir(boredir) if f[-4:] == '.xml']
        else:
            ext = version
            LD =[f for f in os.listdir(boredir) if f[-7:-4] == ext]

        if len(LD) == 0:
            raise FileNotFoundError(
                    "No files found with version = {} in directory\n{}"
                    .format(version, boredir))

        if n is None:
            n = len(LD)
        else:
            n = min(abs(int(n)), len(LD))

        for i, file in enumerate(LD[:n]):
            if verbose:
                print(file)
            else:
                if np.remainder(i + 1, 50) == 0:
                    print('.', end='')
                if np.remainder(i + 1, 500) == 0:
                    print(i)
            bore = Bore(os.path.join(boredir, file))
            if not np.isnan(bore.ztop):
                self.data[bore.name]= bore
            else:
                print('\n', file, " has no elevation.")


        def keys(self):
            return self.data.keys()

        def __getitem__(self, key):
            return self.data[key]

        def __setitem__(self, key, value):
            self.data[key] = value

        def __iter__(self):
            return self.data.__iter()


    def select(self, polygon):
        '''Return a subset of Bores lying inside the polygon.

        Args:
            polygon ([[x, y],...] coordinates or a n,2 ndarray):'
                contour to check whether bores are inside it.
        Returns:
            bores inside polygons
        '''
        Xb = [self[b].x for b in self]
        Yb = [self[b].y for b in self]
        I = list(inpoly(Xb, Yb, polygon))

        B = Bores()
        for tru, k in zip(I, self.keys()):
            if tru:
                B[k] = self[k]
        return B


    def plot(self, **kwargs):
        '''Plot the boreholes according to kwargs.

        When neither a line or a polyline is specified, then plot all boreholes
        as if they were in a profile in the order as given and accumulating x
        using the distance between succussive holes.

        Using line:
            When line is speciied, then plot all boreholes with x the distance of
            each borehole to the line.

        Using polylines:
            When polyline is specified, then plot the boreholes with x the distance
            to the polyline measured along the direction given by angle alpha. If
            alpha is not given, the averge direction computed form the edges of the
            polyline is used instead.

            This distance is computed from each borehole along the direction given
            by angle alpha. This is a simplistic, yet flexible way to compute the
            distance from a bunch of points to a cruved line.

            When using polyline, only the bores are plotted whos line under
            the direction alpha interseces the polyline..


        Either a 2-point line or the combination of a polyline and an angle
        must be specified.

        Kwargs:
            order (list of bore id_s):
                plot profile in given order.
            figsize (tuple):
            figsize  (float, float)
                figsize (W, H) in inches.
            fw (float > 1):
               Ffraction of figure width used for each borehole.
            line (floats as (x0, y0, x1, y1) or (x0, y0, alpha) | [None]):
                Line to which distance must be computed perpendicularly.
            polyline (arraylike X, Y) | [None]:
                Coordinates of polyline.
            alpha (float | [None]):
                Angle in degrees under which the lines from points intersect
                the polyline. The angle is with respect to east.
            lith ([True] | False):
                Plot lithology.
            admix ([True] | False):
                Plot admixtures of lithology.
            strat ([True] | False):
                Plot geologic interpretation.
            maxdist (float | (float, float) | [None] ):
                Set maximum distance to line or polyline. If two valules are
                given, the first is to the left of the line and the second
                to the right, with repect to the line direction. (Compare
                left and right hand shore of a river.)
            fs (int > 0 | [None])
                Set fontsize.

            futher kwargs are passed on to Bore.plot, see its docstring.


        '''



        order    = kwargs.pop('order', None)
        verbose  = kwargs.pop('verbose', False)
        figsize  = kwargs.pop('figsize', (16.,7.))
        fw       = kwargs.pop('fw', 40)
        line     = kwargs.pop('line', None)
        polyline = kwargs.pop('polyline', None)
        alpha    = kwargs.pop('alpha', None)
        lith     = kwargs.pop('lith', True)
        admix    = kwargs.pop('admix', True)
        strat    = kwargs.pop('strat', True)
        maxdist  = kwargs.pop('maxdist', None)
        fs       = kwargs.pop('fs', 5)

        dz   = 2.0 # space above highest and lowest drilling elevation
        fext   = 0.1 # extension factor to convert s to xlim
        rect = 0.05, 0.05, 0.9, 0.9  # axis on figure
        fdz  = 0.1 # fraction of dz to plot the line with distances

        keys = order if order is not None else self.keys()


        Xp    = np.array([self[k].x for k in keys])
        Yp    = np.array([self[k].y for k in keys])
        top  = np.max(np.array([self[k].ztop for k in keys]))
        base = np.min(np.array([self[k].zbase for k in keys]))
        s    = np.hstack((0., np.cumsum(np.sqrt(np.diff(Xp)**2 + np.diff(Yp)**2))))
        sm   = 0.5 * (s[:-1] + s[1:])
        ds   = np.diff(s)

        if np.all(np.isnan(s)):
            raise Warning("all distances are nan, can't plot the boreholes.")
            return s

        if line is not None:
            s = dist2line(Xp, Yp, line, verbose=verbose)
        elif polyline is not None:
            if not isinstance(alpha,(float, int)):
                raise ValueError("direction alpha must be float in degrees anticlockwise")
            s = dist2polyline((Xp, Yp), polyline, alpha, verbose=verbose,
                              maxdist=maxdist)
            if np.all(np.isnan(s)):
                warnings.warn(
                    'No bores to be plotted. All s == NaN\n' + \
                    'Check that lines from any bores intersect\n' + \
                    'the given line')
                return
        else:
            pass # this will plot all borehole in line with x their
                 # mutual cumulative distance

        mns = min((np.nanmin(s), 0))
        mxs = max((np.nanmax(s), 0))

        assert not np.isnan(mns), 'how come mns is nan?'
        assert not np.isnan(mxs), 'how come mxs is nan?'

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(rect)
        ax.set_title(kwargs.pop('title', 'Test drawing a set of boreholes'))
        ax.set_xlabel(kwargs.pop('xlabel', 'x [m]'))
        ax.set_ylabel(kwargs.pop('ylabel', 'm +NAP'))
        ax.set_xlim((mns - fext * (mxs - mns), mxs + fext * (mxs-mns)))
        ax.set_ylim(kwargs.pop('ylim', (base - dz, top + dz)))
        ax.grid(axis='y', linewidth=0.5)

        for i, k in enumerate(keys):
            if not np.isnan(s[i]):
                print('bore {}, name={}, s={:.0} m'.format(i,self[k].name, s[i]))
                self[k].plot(x=s[i], ax=ax, fw=fw, fs=fs,
                          lith=lith, admix=admix, strat=strat,
                          verbose=verbose, **kwargs)

        if line is None and polyline is None:
            for i in range(len(ds)):
                #print('top=', top, 'bot=', base)
                ax.plot(s[[i, i+1]], [top + fdz * dz, top + fdz * dz], 'k', linewidth=0.5)
                ax.plot(s, np.zeros_like(s) + top + fdz * dz, 'b.', markersize=2)
                ax.text(sm[i], top + fdz * dz, '{:.0f}'.format(ds[i]),
                        ha='center', va='bottom',
                        fontdict={'fontsize':6, 'rotation': 45})
        return s

    def toline(self, line, verbose=False):
        '''Returns distances of all bores to line

        Args:
            line:
                line wo which the istances are to be computed
            verbose:
                print/plot info when True
        '''
        Xp = [self[b].x for b in self]
        Yp = [self[b].y for b in self]
        return dist2line(Xp, Yp, line, verbose)


    def toshape(self, shapename, verbose=False):
        '''Generates shapefile of bore locations with record [name, ztop, zbot].
        '''

        import shapefile
        wr = shapefile.Writer(shapeType=shapefile.POINT)
        wr.field('name', fieldType='C', size='20')
        wr.field('ztop', fieldType='N', size='20', decimal=3)
        wr.field('zbase', fieldType='N', size='20', decimal=3)
        wr.field('depth', fieldType='N', size='20', decimal=3)

        if verbose:
            print(wr.fields)
            print(len(self.data.keys()))

        for k in self.data.keys():
           d = {'name': k,
                'ztop': self.data[k].ztop,
                'zbase': self.data[k].zbase,
                'depth': self.data[k].baseDepth}
           print(d)
           wr.record(**d)
           wr.point(self.data[k].x, self.data[k].y)

        wr.save(shapename)

        if verbose:
            print('{} records and point-shapes geneated'.format(len(self.data)))
            print('shapefile {} generated.'.format(shapename))


def ln2alpha(line):
    '''Return lines as (x0, y0, alpha) when given (x0, y0, x1, y1).

    Args:
        line (tuple):
            tuple of 3 or 4 floats

            line = (x0, y0, x1, y1) or line= (x0, y0, alpha)
            with alpha in degrees
    Returns:
        line (tuple) :
            line = (x0, y0, alpha)
            (alpha in degrees)
    '''
    if len(line) == 3:
        return line
    else:
        if len(line) != 4:
            raise ValueError("len(line) must be 3 or 4")
        x0, y0, x1, y1 = line
        if x0 > y0: # orient line northward
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        alpha = np.arctan2(y1 - y0, x1 - x0) * 180. / np.pi
        return x0, y0, alpha


def dist2line(Xp, Yp, line=None, verbose=False):
    '''Return and possibly plot distance from Xp, Yp to line.

    Args:
        Xp, Yp (np.ndarrays):
            x and y coordinates of points
        line (sequence of 3 or 4 floats or 2 tuples):
            (x0, y0, angle) | (x0, y0, x1, y1) | [(x0, y0), (x1, y1)]
            Ppoint on line + angle or two points on line.
        verbose (True | [False]):
            Plot points Xp, Yp, line and lines through points perpendicular to line.
    Returns:
        mu : np.ndarray
            distance of bores to line

    TO 20171127
    '''
    assert isinstance(line, (tuple, list)) and len(line) in (2, 3, 4),\
        ValueError('line must be tuple (x, y, alpha) or (x0, y0, z1, y1)')

    toAngle = lambda x0, y0, x1, y1 : np.arctan2(y1 - y0, x1 - x0) * 180 / np.pi

    if len(line) == 2:
        x0, y0 = line[0]
        x1, y1 = line[1]
    if len(line) == 4:
        x0, y0, x1, y1 = line
    if len(line) == 3:
        x0, y0, alpha = line
    else:
        alpha = toAngle(x0, y0, x1, y1)

    Xp = np.array(Xp)
    Yp = np.array(Yp)

    ex = np.cos(np.pi / 180. * alpha)
    ey = np.sin(np.pi / 180. * alpha)

    Mi = np.linalg.inv(np.array([[ex, ey], [ey, -ex]]))

    lammu = np.dot(Mi, np.array([Xp - x0, Yp - y0]))

    lam = lammu[0]
    mu  = lammu[1]

    if verbose==False: return mu

    Xa = x0 + lam * ex
    Ya = y0 + lam * ey

    Xb = Xp + mu * -ey
    Yb = Yp + mu *  ex

    fig, ax = plt.subplots()
    ax.set_title('Testing perpendicular lines to n points')
    ax.set_xlabel('x')
    ax.set_ylabel('x')

    X0 = np.zeros_like(Xa) + x0
    Y0 = np.zeros_like(Ya) + y0

    xmin = min((np.nanmin(Xa), np.nanmin(Xp), np.nanmin(X0)))
    xmax = max((np.nanmax(Xa), np.nanmax(Xp), np.nanmax(X0)))
    ymin = min((np.nanmin(Ya), np.nanmin(Yp), np.nanmin(Y0)))
    ymax = max((np.nanmax(Ya), np.nanmax(Yp), np.nanmax(Y0)))

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    ax.text(x0, y0, '(x0, y0)', ha='left')

    ax.plot(x0, y0, 'r.')

    ax.plot([Xp, Xb], [Yp, Yb], 'b-')
    ax.plot( Xp,       Yp,      'bo')
    ax.plot( Xb,       Yb,      'bx')
    ax.plot( Xa,       Ya,      'r+')


    ax.plot([X0, Xa], [Y0, Ya], 'r-')

    for xpp, ypp, x, y, m in zip(Xp, Yp, Xb, Yb, mu):
        xm = 0.5 * (xpp + x)
        ym = 0.5 * (ypp + y)
        ax.text(xm, ym, 'mu={:.3g}'.format(m), ha='left')

    return mu


def dist2polyline(points, polyline, alpha, verbose=False, maxdist=None):
    '''Return distances from points to polyline given angle alpha.

    Args:
        points tuple (Xp, Yp) or sequence of tuples (Xpi, Ypi) or np.ndarray
            the coordinates from which the distance to line is tob computed.
        polyle : like points, but interpreted as a polyline
            line to which the distance is to be computed.
        alpha : float
            angle in degrees with respect to east under whicht the lines
            through the points will intersect the polyline
        verbose: bool
            if true then plot points, line and perpendicular lines
    Returns:
        mu (np.ndarray):
            distance of bores to line

    TO 20171127
    '''

    if isinstance(maxdist, (int, float)):
        maxdist = (-maxdist, maxdist)

    points = np.array(points)
    Xp = points[0]
    Yp = points[1]
    line = np.array(polyline)
    X0 = line[:, 0]
    Y0 = line[:, 1]
    Dx = np.diff(X0)
    Dy = np.diff(Y0)

    ex = np.cos(np.pi / 180. * alpha)
    ey = np.sin(np.pi / 180. * alpha)

    Mu = np.zeros_like(Xp) * np.nan

    for i, (dx, dy, x0, y0) in enumerate(zip(Dx, Dy, X0[:-1], Y0[:-1])):

        Mi = np.linalg.inv(np.array([[dx, -ex], [dy, -ey]]))

        lammu = np.dot(Mi, np.array([Xp - x0, Yp - y0]))

        lam = lammu[0]
        mu  = lammu[1]

        Mu[AND(lam>=0, lam<=1)] = mu[AND(lam>=0, lam<=1)]

    if maxdist is not None:
        Mu[OR(Mu<maxdist[0], Mu>maxdist[1])] = np.nan


    if not verbose == False:  return Mu

    if np.all(np.isnan(Mu)):
        raise Warning("All distances are Nan, Can't plot the points")
        return Mu

    Xa = Xp + Mu * ex
    Ya = y0 + Mu * ey

    fig, ax = plt.subplots()
    ax.set_title('Testing distance to polyline')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    xmin = min((np.nanmin(Xa), np.nanmin(Xp), np.nanmin(X0)))
    xmax = max((np.nanmax(Xa), np.nanmax(Xp), np.nanmax(X0)))
    ymin = min((np.nanmin(Ya), np.nanmin(Yp), np.nanmin(Y0)))
    ymax = max((np.nanmax(Ya), np.nanmax(Yp), np.nanmax(Y0)))

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    ax.plot(X0, Y0, 'b', linewidth=3, label='polyline')
    ax.plot(Xp, Yp, 'r.')

    for xp, yp, mu in zip(Xp, Yp, Mu):
        if not np.isnan(mu):
            ax.plot([xp, xp + ex * mu], [yp, yp + ey * mu], 'r')

    ax.legend(loc='best')

    return Mu

#%% getting union of tags, text and attrib keys from all bores

def get_tags_and_fields(boredir, xmlfiles, tags, flds, verbose):

    def getfields(root, tags,flds, spaces, verbose):
        tag = root.tag
        if verbose:
            print(spaces, tag)
        if not tag in tags:
            tags.add(tag)
            flds[tag] = dict()
        if root.attrib is not None:
            if 'code' in root.attrib:
                flds[tag][root.attrib['code']] = root.text
            else:
                if root.attrib is not None:
                    if 'attrib' not in flds[tag]:
                        flds[tag]['attrib'] = set()
                    flds[tag]['attrib'] = flds[tag]['attrib'].union(root.attrib.keys())
                if root.text is not None and root.text != '\n':
                    if 'text' not in flds[tag]:
                        flds[tag]['text'] = set()
                    flds[tag]['text'].add(root.text)
        for ch in root.getchildren():
            getfields(ch,tags, flds, spaces + '  ', verbose)

    for xml in xmlfiles:
        print(xml)
        tree = ET.parse(os.path.join(boredir, xml))
        root = tree.getroot()
        getfields(root, tags, flds,'', verbose)

def gettags(boredir, verbose=True):
    tags=set()
    flds = dict()
    xmlfiles= [xml for xml in os.listdir(boredir) if xml[-7:-4]=='1.4']
    get_tags_and_fields(boredir, xmlfiles, tags, flds, verbose)
    return tags, flds


def get_bores_with_strat(boredir):
    '''Return bores that have stratigraphy'''

    fileNamesOut = []
    xmlfiles = [f for f in os.listdir(boredir) if f.endswith('1.4.xml')]
    print('processing ', len(xmlfiles), ' files')
    for i, xml in enumerate(xmlfiles):
        tree = ET.parse(os.path.join(boredir,xml))
        root = tree.getroot()
        if np.mod(i, 50) == 0:
            print('.', end='')
        if root.find('.//lithostratDescr') is not None:
            fileNamesOut.append(xml)
            print(xml, end='')
    print()
    return fileNamesOut

#%% examples of results

mytags = {
 'biogenAdmix',
 'carbonateFracNEN5104',
 'clasticAdmix',
 'clayAdmix',
 'coarseMatAdmix',
 'colorIntensity',
 'colorMain',
 'colorSecondary',
 'geolInterpret',
 'glaucFrac',
 'gravelAdmix',
 'gravelFrac',
 'gravelMedianClass',
 'gravelRound',
 'humusAdmix',
 'lithoInterval',
 'lithoLayerBoundary',
 'lithoLayerTrend',
 'lithology',
 'lithostrat',
 'lithostratInterval',
 'plantFrac',
 'plantType',
 'sandAdmix',
 'sandCompact',
 'sandMedCoarse',
 'sandMedFine',
 'sandMedianClass',
 'sandPerc',
 'sandRound',
 'sandSorting',
 'sandVarieg',
 'sedimentStructure',
 'shellFrac',
 'siltAdmix',
 'siltPerc',
 'subLayerLithology',
 'subLayerThickness',
 'subLithoLayer',
}


# Borings with strat
xmls = ['B60C0816_1.4.xml',
 'B60C0763_1.4.xml',
 'B60A1703_1.4.xml',
 'B60C1077_1.4.xml',
 'B60A1609_1.4.xml',
 'B60A1619_1.4.xml',
 'B60A0150_1.4.xml',
 'B60C0061_1.4.xml',
 'B60A1654_1.4.xml',
 'B60A1644_1.4.xml',
 'B60C0941_1.4.xml',
 'B60A1785_1.4.xml',
 'B60A1636_1.4.xml']


#%%

if __name__ == '__main__':

    boredir = '/Users/Theo/GRWMODELS/python/tools/dino/bores/testbores'
    '''
    #% read an xml file

    xmlfile = "B60A0143_1.4.xml"
    xmlprint(os.path.join(boredir, xmlfile))

    bores = Bores(boredir=boredir, n=5)
    #bores[-1].plot()
    bores.plot(fw=80, strat=False, lith=True, admix=False, fs=5)

    xmls = get_bores_with_strat(boredir)
    '''
    boredir = '/Users/Theo/GRWMODELS/python/tools/dino/bores/testbores'
    bore = Bore(os.path.join(boredir, 'B60A0143_1.4.xml'))
    bore.plot()
    '''
    tags, flds = gettags(boredir, verbose=True)
    showtags(tags)


    line = (182625, 337210, 183050, 337580) # obbicht
    line = (182612, 335900, 182850, 336345 ) # kingbeek

    mu = bores.toline(line, verbose=True)

    bores.plot(fw=40, line=line)

    shapename = 'bores'
    bores.toshape(shapename, verbose=True)
    '''

    borepath = '/Users/Theo/GRWMODELS/python/DEME-juliana/bores/agtdeme'

    agtbores = Bores(borepath)


    def northsouth(yN, yS):
        Y = sorted([(k, agtbores[k].y) for k in agtbores], key=lambda x: x[1], reverse=True)
        Y = [y[0] for y in Y if y[1] >= yS and y[1] <= yN]
        return Y

    north2south = [y[0] for y in Y ]

    testset = ['PB-GRA-5.xml', 'PB-GRA-1.xml', 'PB-GRA-4.xml', 'PB-GRA-2.xml',
         'PB-U-099.xml'] #, 'PB-GRA-3.xml', 'PB-U-060.xml', 'PB-U-031.xml',
         # 'PP-U-003.xml', 'PB-U-094.xml']
    #north2south =[y for y in north2south if y in testset]


    line = [(180000, 338360), (185000, 332640)] # Heerlerheide
    line = [(182832, 336309), (183090, 336204)] # through PB-U-203 loorecht Juka
    if True:
        yN = 336700
        yS = 335700
        order = northsouth(yN, yS)
        xlabel = 'Afstand [m] langs Julianakanaal vanaf wnp PB-U-203'
        title  = 'Waarnemingsputten en putten van N naar Z uit "Boorgatgegevens.xlsx ter hoogte van de Kingbeekbronnen"'
        ylabel = 'NAP [m]'
        agtbores.plot(fw=100, line=line, order=order,
                      admix=False, lith=True, strat=True, filters=True,
                      verbose=True,
                      title=title,
                      xlabel=xlabel,
                      ylabel=ylabel,
                      name=True, fs=5, maxdist=(-500, 500))
    else:
        yN = 336700
        yS = 335700
        order = northsouth(yN, yS)
        yWNP = agtbores[order[0]].y
        xlabel = '(yRD={:.0f}) - yRD wnp [m]'.format(yWNP)
        title  = '{} wnp uit "Boorgatgegevens.xlsx" tussen y={:.0f} en {:.0f} m'.\
                format(len(order), yN, yS)
        ylabel = 'NAP [m]'
        agtbores.plot(fw=100, order=order, line=None,
                      admix=False, lith=True, strat=True, filters=True,
                      verbose=True,
                      title=title,
                      xlabel=xlabel,
                      ylabel=ylabel,
                      name=True, fs=5)
        plt.text(0, plt.ylim()[0], '  yRD={:.0f}m'.format(yWNP),
                 va='bottom', ha='center', rotation=90)
    full()
