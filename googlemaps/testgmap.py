#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 03:05:54 2018

@author: Theo
"""

import sys

tools ='/Users/Theo/GRWMODELS/python/tools/'

if tools not in sys.path:
    sys.path.insert(1, tools)

import numpy as np
from coords import rd2wgs
import matplotlib.pyplot as plt
from googlemaps import get_google_map_image as gm

Omega = 40000000 # cicumference of the world


x = lambda lon : Omega * (lon / 360 + 0.5)
y = lambda lat : Omega / (2 * np.pi) * np.log(np.tan(np.pi/360  * (90 + lat)))

dxdlon = lambda lon : Omega / 360
dydlat = lambda lat : Omega / 360 / np.sin(np.pi / 180 * (90 + lat))

z = 12


if __name__ == '__main__':

    xlim= (182300, 183700) # 1400 m
    ylim= (336400, 337800) # 1400 m

    xc = np.mean(xlim)
    yc = np.mean(ylim)

    lon1, latc = rd2wgs(xlim[0], yc)
    lon2, latc = rd2wgs(xlim[1], yc)
    lonc, lat1 = rd2wgs(xc, ylim[0])
    lonc, lat2 = rd2wgs(xc, ylim[1])

    # check, should be equal to xlim and ylim [km]
    dxGlobe = (lon2 - lon1) / 360 * Omega  * np.cos(np.pi * latc / 180) # 1400
    dyGlobe = (lat2 - lat1) / 360 * Omega # 1400

    print("Compare globe distances:")
    print('dxGlobe = {:.1f}, xlim[1] - xlim[0] = {:.1f}'.format(dxGlobe, xlim[1] - xlim[0]))
    print('dyGlobe = {:.1f}, ylim[1] - ylim[0] = {:.1f}'.format(dyGlobe, ylim[1] - ylim[0]))

    #Because of equal area and equal distance rations dxMap and dyMap should
    #be almost the same
    dxMap = x(lon2) - x(lon1)
    dyMap = y(lat2) - y(lat1)

    z = 15 # zoom level

    npx = dxMap / Omega * 2 ** z * 256
    npy = dyMap / Omega * 2 ** z * 256
    print("See that npx and npy are almost equal")
    print('npx = {:.1f}, npy = {:.1f}'.format(npx, npy))

    dxdlon(lonc) * (lon2 - lon1)
    dydlat(latc) * (lat2 - lat1)

    print("Compare dxMap, dyMap with the results from the derivatives:")
    print('dxdlon deltaLon = {:.1f}, dydlat deltalat  = {:.1f}'
          .format(dxdlon(lonc) * (lon2 - lon1), dydlat(latc) * (lat2 - lat1)))
    print('dxMap = {:.1f}, dyMap  = {:.1f}'
          .format(dxMap, dyMap))

    #Generate a map
    gmap = gm.get_gmap_from_RDwindow(xlim, ylim)

    gmap.imshow(extent=(*xlim, *ylim))
    ax = plt.gca()
    #ax.set_aspect('equal')
    plt.show()

    Npx, Npy= gmap.size

    gmap.xlim # compre with xlim above
    gmap.ylim # compare with ylim above
