#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 21:45:08 2017

This module is OBSOLETE (use kml)

plots of contents of a kml folder with polygons or polylines

@author: Theo 171226
"""
tools = '/Users/Theo/GRWMODELS/python/tools'
import sys

if not tools in sys.path:
    sys.path.insert(1, tools)

import os
import xml.etree.ElementTree as ET
import numpy as np
from collections import UserDict
import matplotlib.pyplot as plt
from coords import wgs2rd, rd2wgs
import matplotlib.patches as patches
import shapefile

class Patch:
    '''returns Patch object
    parameters
    ==========
    name : str
        name of the object
    EN : ndarray [n, 2] coordinates in E, N (wgs84, Google Earth)
        if EN is given, RD is computed automatically
    RD : ndarray [n, 2] coordinates in x, y of Dutch rd systems
        if RD is given, EN is computed automatically
    '''
    def __init__(self, name, EN=None, RD=None):
        self.name = name
        if EN is None and RD is None:
            raise ValueError('EN and RD cannot both be None')
        if EN is None:
            self.rd = RD
            x, y = RD[:, 0], RD[:, 1]
            E, N = rd2wgs(x, y)
            self.EN = np.vstack((E, N)).T
        if RD is None:
            self.EN = EN
            e, n = EN[:, 0], EN[:, 1]
            x, y = wgs2rd(e, n)
            self.rd = np.vstack((x, y)).T

        self.Em = np.mean(self.EN[:, 0])
        self.Nm = np.mean(self.EN[:, 1])
        self.xm = np.mean(self.rd[:, 0])
        self.ym = np.mean(self.rd[:, 1])
        self.xlim = (np.min(self.rd[:, 0]), np.max(self.rd[:, 0]))
        self.ylim = (np.min(self.rd[:, 1]), np.max(self.rd[:, 1]))

    @property
    def bbox(self):
        return self.xlim[0], self.ylim[0], self.xlim[1], self.ylim[1]

    def plot_bbox(self, **kwargs):
        '''plots bounding box around patch'''
        ax = kwargs.plot('ax', plt.gca())
        ax.plot(self.xlim[[0, 1, 1, 0, 0]],
                self.ylim[[0, 0, 1, 1, 0]],
                **kwargs)

    def plot(self, *args, co='rd', labels=False, **kwargs):
        '''plots the patch'''
        if not co == 'rd': co = 'EN'
        ax = kwargs.pop('ax', None)
        xlim = kwargs.pop('xlim', None)
        ylim = kwargs.pop('ylim', None)
        title = kwargs.pop('title', None)
        fs = kwargs.pop('fs', kwargs.pop('fontsize', 8))
        ha = kwargs.pop('ha', 'center')
        va = kwargs.pop('va', 'bottom')
        rotation = kwargs.pop('rotation', 0)

        color = kwargs.pop('color', None)
        ec = kwargs.pop('edgecolor', kwargs.pop('ec', 'black'))
        fc = kwargs.pop('facecolor', kwargs.pop('fc', 'orange'))
        alpha = kwargs.pop('alpha', 0.5)

        if ax is None:
            fig, ax = plt.subplots()
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_title(self.name)
            ax.grid(True)
        if xlim is not None: ax.set_xlim(xlim)
        if ylim is not None: ax.set_ylim(ylim)
        if title is not None: ax.set_title(title)

        if co == 'rd':
            ax.plot(self.rd[:,0], self.rd[:,1], *args, color=color, **kwargs)
            p = patches.Polygon(self.rd, ec=ec, fc=fc, alpha=alpha,  **kwargs)
            ax.add_patch(p)
            if labels == True:
                ax.text(self.xm, self.ym, self.name,
                        ha=ha, va=va, rotation=rotation,
                        fontsize=fs)
        else:
            ax.plot(self.EN[:,0], self.EN[:,1], *args, **kwargs)
            if labels == True:
                ax.text(self.Em, self.Nm, self.name,
                        ha=ha, va=va, rotation=rotation,
                        fontsize=fs)
        return ax



class Patches(UserDict):
    '''A dict of patches that can plot itself'''

    def __init__(self, kmlfile=None, verbose=True):
        '''returns Patches object with Patches read from a KML folder file
        parameters
        ----------
        kmflfile : str
            coplete path to kmflfile that contains multiple objeces
            i.e. multiple polylines of polygons. It is a saved folder
            in Google Eartch
        '''
        if not os.path.isfile(kmlfile):
            raise FileNotFoundError(kmlfile)

        tree = ET.parse(kmlfile)
        root = tree.getroot()

        self.data = {}

        # text automatially preprended to all tags
        pre = './/' + root.tag.split('}')[0] + '}'

        if verbose:
            print(kmlfile)

        placemarks = root.findall(pre + 'Placemark')
        for pm in placemarks:
            name = pm.find(pre + 'name').text
            coords = pm.find(pre + 'coordinates').text
            coords = coords.replace('\t','').replace('\n','').split(' ')
            if len(coords[-1]) == 0:
                coords.pop(-1)
            coords = [f.split(',') for f in coords]
            EN     = np.array([(float(f[0]), float(f[1])) for f in coords])
            self.data[name] = Patch(name, EN=EN)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, key):
        return self.data[key]

    def plot(self, *args, **kwargs):
        '''plots the Patches using rd coordinates as default'''
        ax =kwargs.pop('ax', None)
        title = kwargs.pop('title', None)
        xlabel = kwargs.pop('xlabel', None)
        ylabel = kwargs.pop('ylabel', None)
        grid  = kwargs.pop('grid', True)
        xlim = kwargs.pop('xlim', None)
        ylim = kwargs.pop('ylim', None)
        fs = kwargs.pop('fs', kwargs.pop('fontsize', 8))

        if ax is None:
            fig, ax = plt.subplots()
            ax.set_title('Patches')
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
        if title is not None: ax.set_title(title)
        if xlabel is not None: ax.set_xlabel(xlabel)
        if ylabel is not None: ax.set_ylabel(ylabel)
        if xlim   is not None: ax.set_xlim(xlim)
        if ylim   is not None: ax.set_ylim(ylim)
        ax.grid(grid)

        kwargs['ax'] = ax
        kwargs['fontsize'] = fs

        for k in self:
            self[k].plot(*args, **kwargs)


    def makeShape(self, shapeFilenName):

        wr = shapefile.Writer(shapeType=shapefile.POLYGON)

        # two text fields only
        wr.field(name='name',      fieldType='C', size='30')
        wr.field(name='shortName', fieldType='C',  size='10')

        for name in self.keys():
            nm = name.split()[1]
            sn = nm[:2] + nm[-1]  # Generate short names
            wr.record(nm, sn)      # add a record for the dbf file

            # To add a shape we need the coordinates as a list of coordinate pairs
            points = [[p, q] for (p, q) in zip(self[name].rd[:,0], self[name].rd[:,1])]
            wr.poly(parts=[points]) # add the polygon shape

        wr.save(shapeFilenName)


def nederland(**kwargs):
    kmlfile =os.path.join( tools, 'coords/data/Nederland.kml')
    nl = Patches(kmlfile)
    nl.plot(**kwargs)


if __name__ == '__main__':

    kmlfile = os.path.join(tools, 'coords/data/Nederland.kml')

    map = Patches(kmlfile)
    map.plot(color='brown')

    nederland(color='orange')

    kmlfile = os.path.join(tools, 'coords/data/Maasterrassen.kml')
    maasterrassen = Patches(kmlfile)
    maasterrassen.plot('--', color='brown', labels=True,
                       alpha=0.1,
                       ec=None,
                       fc='green',
                       rotation=40)


    maasterrassen.makeShape('Maasterrassen')


