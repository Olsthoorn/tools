#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 14:15:16 2018

Get a google map image with labels

see
    https://maps.googleapis.com/maps/api/staticmap?parameters


@author: Theo 180120

"""

#%% Import modules

import os
import sys

tools ='/Users/Theo/GRWMODELS/python/tools/'

if tools not in sys.path:
    sys.path.insert(1, tools)

import numpy as np
import requests
from PIL import Image
from io  import BytesIO
from coords import rd2wgs
import matplotlib.pyplot as plt
from coords import wgs2rd

mykey = 'AIzaSyDRETF3BLxtT-W3c7dXlk-7t3j3Z7wLTN8'

API   = 'https://maps.googleapis.com/maps/api/staticmap?'

r_earth = 6378137 # m


class Gmap:

    offset = 2 ** 21 * 128 # half of world circumf in pixels at zoom level 21


    def __init__(self, lon, lat, zoomlevel=14, size=(640, 640), maptype='roadmap'):

        self.center = (lon, lat)
        self.zoomlev= zoomlevel
        self.size = size
        self.maptype=maptype
        self.language = 'en'
        self.img = get_image(self.center,
                    crs='RD', zoom=self.zoomlevel, maptype=self.maptype,
                        size=self.size, language=self.language)

    def px(self, lam):
        '''return pixel coordinate relative to  mapcenter.
        '''
        return px(lam, lon, self.center[0], self.zoomlevel)
    def py(self, phi):
        '''return pixel coordinate relatie to map center.
        '''
        return py(phi, self.center[0], self.zoomlevel)
    def pixy(self, lam, phi):
        '''return pixel coordinates relative to map center.
        '''
        return self.px(lam), self.py(phi)
    def lon(self, px):
        '''return longitude (degrees) from px relative to map center.
        '''
        return lon(px, 0, self.center[0], self.zoomlevel)

    def lat(self, py):
        '''return latitude (degrees) from py relative to max center.
        '''
        return lat(py, 0, self.center[1], self.zoomlevel)

    def lonlat(self, px, py):
        '''return lat, lon from px, py relative to map center.

        parameters
        --------
            px, py: float or its
                pixel coordinates of point relative to self.center having 0, 0
                as pixel coordinates
        '''
        return lonlat(px, py, 0, 0, *self.center, self.zoomlevel)

    def xy(px, py):
        lat, lon = lonlat(px, py)
        return wgs2rd(lat, lon, lat)

    def xlim(self):
        lam1 = self.lon(-0.5 * self.size[0]) + self.center[0]
        lam2 = self.lon(+0.5 * self.size[0]) + self.center[0]
        return lam1, lam2

    def ylim(self):
        phi1 = self.lat(-0.5 * self.size[1]) + self.center[1]
        phi2 = self.lat(+0.5 * self.size[1]) + self.center[1]
        return phi1, phi2

    def xlim_rd(self):
        lam1, lam2 = self.xlim()
        phi0 = self.lat(0)
        xL, y0 = wgs2rd(lam1, phi0)
        xR, y0 = wgs2rd(lam2, phi0)
        return xL, xR

    def ylim_rd(self):
        phi1, phi2 = self.ylim()
        lam0 = self.lon(0)
        x0, y1 = wgs2rd(lam0, phi1)
        x0, y2 = wgs2rd(lam0, phi2)
        return y1, y2

    def ULrd(self):
        lam, phi = self.lon(-self.size[0]), self.lat(+self.size[1])
        return wgs2rd(lam, phi)
    def LLrd(self):
        lam, phi = self.lon(-self.size[0]), self.lat(-self.size[1])
        return wgs2rd(lam, phi)
    def LRrd(self):
        lam, phi = self.lon(+self.size[0]), self.lat(-self.size[1])
        return wgs2rd(lam, phi)
    def URrd(self):
        lam, phi = self.lon(+self.size[0]), self.lat(+self.size[1])
        return wgs2rd(lam, phi)

    def bb_rd(self):
        lam1, lam2 = self.xlim_rd()
        phi1, phi2 = self.ylim_rd()
        x1, _ = wgs2rd(lam1, self.center[1])
        x2, _ = wgs2rd(lam2, self.center[1])
        _, y1 = wgs2rd(self.center[0], phi1)
        _, y2 = wgs2rd(self.center[0], phi2)
        return x1, y1, x2, y2

    def pgbb(self):
        lam1, lam2 = self.xlim()
        phi1, phi2 = self.ylim()
        lam = [lam1, lam2, lam2, lam1, lam1]
        phi = [phi1, phi1, phi1, phi1, phi1]
        return np.array([lam, phi]).T

    def pgbb_rd(self):
        x1, x2 = self.xlim_rd()
        y1, y2 = self.ylim_rd()
        x = [x1, x2, x2, x1, x1]
        y = [y1, y1, y1, y1, y1]
        return np.array([x, y]).T

    def xy2pxy(self, x, y):
        lon, lat = rd2wgs(xy)
        px = self.px(lon)
        py = self.py(lat)
        return px, py

    def pxy2xy(self, px, py):
        lon = self.lon(px)
        lat = self.lat(py)
        x, y = wgs2rd(lon, lat)
        return x, y

    def xcyc(self):
        return wgs2rd(self.center[0], self.center[1])

# this eas
def px(lam, lam0, zoomlevel):
    '''Return number of pixels from lam0 to lam.

    parameters
    ---------
        lam : float
            Easting (X) in degrees
        lam0 : float
            Easting from which the distance in pixels is computed
        zoomlevel : int
            Same value as in the google static maps URL


    @TO 20180819
    '''

    px = (128 / np.pi) * 2 ** zoomlevel * (lam - lam0)

    return px

def py(phi, phi0, zoomlevel):
    '''Return position number pixels going from phi0 to phi.

    parameters
    ---------
        phi: float
            Northing (Y) in degrees
        phi0: float
            latitude from which the distance in pixels is computed
        zoomlevel : int
            Same value as in the google static maps URL

    @TO 20180819
    '''
    py = (128 / np.pi) * 2 ** zoomlevel * (
          np.log(np.tan(np.pi / 4 + phi0 / 2)) /
          np.log(np.tan(np.pi / 4 + phi  / 2)))

    return py

def pxpy(lam, phi, lam0, phi0, zoomlevel):
    '''Return position of point (lon, lat) in pixels relative to center of bitmap.

    parameters
    ---------
        lam : float
            Easting (X) in degrees
        phi: float
            Northing (Y) in degrees
        lam0 : float
            longitude from which distance is computed
        phi0: float
            latitude from which distance is computed
        zoomlevel : int
            Same value as in the google static maps URL
            '''
    return px(lam, lam0, zoomlevel), py(phi, phi0, zoomlevel)


def lon(px, px0, lam0, zoomlevel):
    '''return longitude (degrees) from px0 to px, lam0 and zoomlevel

    paamters
    --------
        px, px0: float or its
            desired and known pixel east-west (x)
        lam0: float
            loingitude at point px0
        zoomlevel: int
            Same value as in the google static maps URL
    '''
    return lam0 + np.pi / 128 * (px - px0) * 2 ** (-zoomlevel)

def lat(py, py0, phi0, zoomlevel):
        '''return latitude (degrees) from py0, py, lam0 and zoomlevel

    paamters
    --------
        py, py0: float or its
            desired and known pixel north-south (y)
        phi0: float
            latitute at point py0
        zoomlevel: int
            Same value as in the google static maps URL
    '''

        phi = np.arctan(np.tan(np.pi / 4 + phi0 / 2) *\
            np.exp(np.pi / 128 * (py - py0)) * (2 ** (-zoomlevel))) - np.pi / 2
        return phi

def lonlat(px, py, px0, py0, lam0, phi0, zoomlevel):
    '''return logitude (degrees) from pixel0 to pixel1, lam0 and zoomlevel

    paamters
    --------
        px, px0: float or its
            desired and known pixel east-west (x)
        py, py0: float or its
            desired and known pixel north-south (y)
        lam0: float
            longitue at point px0
        phi0: float
            latitude at point py0
        zoomlevel: int
            Same value as in the google static maps URL
    '''

    return lon(px, px0, lam0, zoomlevel), lat(py, py0, phi0, zoomlevel)



def zoomlev(lat, w):
    '''Return zoomlevel.

    paramters
    ---------
        lat : float
            latitude (northing) in degrees
        w  : float
            desired width in m
    '''
    zlev = -np.log(w / (2 * np.pi * r_earth * np.cos(lat * np.pi / 180.))
                        ) / np.log(2)

    return int(min(21, np.floor(zlev)))


def get_image(center, crs='RD', width=None, zoom=14, maptype='roadmap',
              path=None, polygon=None, markers=None, size=(640, 640),
              scale=None, language='en', outfile=None):
    '''Return a google maps image

    parameters
    ----------
        center: sequence of tuples of floats or str
            if center is a string, then it must be a legal geolocation (address)
            accoring to google maps
            else:
                xy are x, y or E, N (lon, lat coordinates interpreted
                according to crs)
        crs : str
            coordinate system, either 'RD' or 'WGS' or 'GE'
        width: float
            desired width of image in m (it overwrites zoomlevel)
        zooml : int
            zoomlevel 0-21 google maps (overwritten by width if width not None)
        maptype : str
            roadmap|satellite|hybrid|terrain
        path : sequence of x, y tuples (default = None)
            line to be drawn
        polygon : sequence of x, y tuples, (default = None)
            polygon to be drawn
        size : tuple of two float
            hor and vert number of pixels, max (640, 640)
        markers : markers
            markers as understood by google maps API
        outfile: str
            name of image file to be written (if not None)
            use correct extension (.png or .jpg, or .gif)

        markers=markerStyles|markerLocation1|markerlocaion2| ... etc
        marker=[{size: .., color: .., label: .., locations ...}] where
            size  = (optional) {tiny, mid, small}
            color = (optional) {black, brown, green, purple, yellow, blue, gay, red, white}
            label = (optional) single afpha numerid code [A..Z, 0..9]

        center='Heemstede,Netherlands'
        markers:size:mid|color:blue|label:A|2102CR54|2102CR76!2102CR80
    '''

    if isinstance(size, (list, tuple)):
        size = '{:.0f}x{:.0f}'.format(min(640, size[0]), min(640, size[0]))

    if not isinstance(center, str):
        xy = np.array(center)
        if xy.ndim == 1:
            xy = xy[np.newaxis, :]

        if crs=='RD':
            x, y = xy[:,0], xy[:,1]
            lon, lat = rd2wgs(x, y)
        else:
            lon, lat = xy[:, 0], xy[:, 1]

        if width == None:
            if zoom == None:
                if len(lon) > 1:
                    w = 4 * max(np.std(lat), np.std(lon)) *\
                    (2 * np.pi * r_earth) * np.cos(np.pi * np.mean(lat)/180)
                    zoom = zoomlev(np.mean(lat), w)
                else:
                    zoom = 14
            else:
                pass
        else:
            lat = 52.
            zoom =zoomlev(lat, width)
            width = None
        center = '{:.6f},{:.6f}'.format(np.mean(lat), np.mean(lon))
    else:
        if width is None:
            if zoom == None:
                zoom = 14
            else:
                pass
        else:
            lat   = 52.
            zoom  = zoomlev(lat, width)
            width = None

    fmt = 'png'
    if outfile is not None:
        outfile, ext = os.path.splitext(outfile)
        if not ext: ext='.png'
        exts = ['png', 'jpg', 'gif', 'png8', 'png32', 'jpg_baseline']
        if ext[1:] not in exts:
            ValueError('Outfile extension <> not in <>'.format(ext[1:], exts))
        else:
            fmt = ext[1:]


    params = {'center' : center,  # home can be any address
              "zoom"   : zoom,    # between 0 and 21
              "size"   : size,    # max 640x640
              "maptype": maptype, # satellite, hybrid, ...  4 types
              "timeout": 0.01,    # timeout should always be used
              "format" : fmt,     # image format
              }

    if scale==2:
        params['scale'] = '2'
    if path is not None:
        params['path'] = path
    if polygon is not None:
        params['polygon'] = polygon
    if markers is not None:
        params['markers'] = markers

    params["key"] = mykey

    response = requests.get(API, params=params)

    img = Image.open(BytesIO(response.content)) # this reads and image object

    # write image to a file, chunk length can be chosen freely
    if outfile is not None:
        with open(outfile, 'wb') as fd:
            for chunk in response.iter_content(chunk_size=128):
                fd.write(chunk)
        print("image written to file <>".format(outfile + ext))

    return img


# ===============  Example  ===================================================

if __name__=="__main__":

    API = 'https://maps.googleapis.com/maps/api/staticmap?'

    xy = 'Heemstede,Netherlands'

    center = (np.round(52 + (21 + (24.25)/60)/60, decimals=6),
              np.round( 4 + (37 + (47.63)/60)/60, decimals=6))

    home = '{:.6f},{:6f}'.format(*center)

    markers='size:mid|color:yellow|label:M|' + home

    markers='size:mid|color:yellow|label:M|Heemstede,Netherlands,Strawinskylaan 54'

    markers=['size:mid|color:yellow|label:T|Strawinskylaan 54',
             'size:mid|color:red|label:Z|Strawinskylaan 74']


    img = get_image( xy, crs='RD', width=None, zoom=14, maptype='roadmap',
                  path=None, polygon=None, markers=markers, size=(640, 640),
                  scale=None, language='en', outfile=None)

    fig, ax = plt.subplots()
    h=plt.imshow(img) # show the image
    ax.set_title('xy')

    fig, ax = plt.subplots()
    img = get_image(center, crs='RD', width=None, zoom=14, maptype='roadmap',
              path=None, polygon=None, markers=markers, size=(640, 640),
              scale=None, language='en', outfile=None)
    h=plt.imshow(img) # show the image
    ax.set_title(center)

    fig, ax = plt.subplots()
    ax.set_title('Obbicht')
    ax.set_xlim(-130, 250)
    ax.set_ylim(-450, 193)
    img = get_image('Obbicht, Netherlands', crs='RD', width=None, zoom=14, maptype='roadmap',
              path=None, polygon=None, markers=markers, size=(640, 640),
              scale=None, language='en', outfile=None)
    h = plt.imshow(img) # show the image
    plt.show()
