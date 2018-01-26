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
from coords.rd2vswgs84 import rd2wgs

mykey = 'AIzaSyDRETF3BLxtT-W3c7dXlk-7t3j3Z7wLTN8'

API   = 'https://maps.googleapis.com/maps/api/staticmap?'

r_earth = 6378137 # m


def zoomw(lat, zoomlevel):
    '''Return zoomwidth.

    parameters
    ----------
        lat : float
            latitude (northing) in degrees
        zoomlevel : int
            google map's zoom level 0-21
    '''
    return  2 * np.pi * r_earth * np.cos(lat * np.pi / 180.) * 2**(-zoomlevel)


def zoomlev(lat, w):
    '''Return zoomlevel.

    paramters
    ---------
        lat : float
            latitude (northing) in degrees
        w  : float
            desired width in m
    '''
    zlev = -np.log(w/( 2 * np.pi * r_earth * np.cos(lat * np.pi / 180.)))\
            / np.log(2)
    return int(min(21, np.floor(zlev)))


def get_image(center, crs='RD', width=None, zoom=14, maptype='roadmap',
              path=None, polygon=None, markers=None, size=(640, 640),
              scale=None, language='en', outfile=None):
    '''Return a google maps image

    parameters
    ----------
        xy: sequence of tuples of floats or str
            if xy is a string, then it must be a legal geolocation (address)
            accoring to google maps
            else:
                xy are x, y or E, N (lon, lat coordinates interpreted
                according to crs

    additional kwargs
    -----------------
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
            makers as understoog by google maps API
        outfile: str
            name of image file to be written (if not None)
            use correct extension (.png or .jpg, or .gif)

        makers=markerStyles|markerLocation1|markerlocaion2| ... etc
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

    lat= np.round(52 + (21 + (24.25)/60)/60, decimals=6)
    lon= np.round( 4 + (37 + (47.63)/60)/60, decimals=6)
    home = '{:.6f},{:6f}'.format(lat, lon)

    markers='size:mid|color:yellow|label:M|' + home

    markers='size:mid|color:yellow|label:M|Heemstede,Netherlands,Strawinskylaan 54'

    markers=['size:mid|color:yellow|label:T|Strawinskylaan 54',
             'size:mid|color:red|label:Z|Strawinskylaan 74']


    img = get_image( xy, crs='RD', width=None, zoom=14, maptype='roadmap',
                  path=None, polygon=None, markers=markers, size=(640, 640),
                  scale=None, language='en', outfile=None)

    img # show the image
