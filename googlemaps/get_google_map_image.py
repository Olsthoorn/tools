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

def gmkey():
    #subprocess.run(['env', '| grep -e GMAPI /Users/Theo/.bash_profile'], capture_output=True)
    #return gmapi[0].split('=')[1].replace('"','')
    return 'AIzaSyBRcG5lwN1Mlfu-RDkunJBNL0fDHZMR_sI'

r_earth = 6378137 # m


def get_gmap_from_RDwindow(xlim, ylim, maptype='roadmap', npx_max=640, npy_max=640, **kwargs):
    '''Return a Gmap (google map) from xlim ylim in RD coordinates.

    parameters
    ----------
        xlim : (x1, x2)
            desired map lim in RD coordinates
        ylim : (y1, y2)
            desired map lim in RD coordinates
        maptype: str
            roadmap|satellite|hybrid
        npx_max : int < = 640
            desired number of pixels of image in x-direction
        npy_max : int < = 640
            desired numver of pixels of image in y-direction
        kwargs:
            additional options are passed on to Gmap
    '''

    lonlim, latlim = xylim2LL(xlim, ylim)

    zoomx = zoom_fr_lonlim(lonlim, npx=npx_max)
    zoomy = zoom_fr_latlim(latlim, npy=npy_max)

    zoom = int(min(zoomx, zoomy))

    npx = 256 * 2 ** zoom * (lonlim[1] - lonlim[0]) / 360

    npy = (256 * 2 ** zoom) * (
            np.log(np.tan(np.pi/4 + latlim[1] * np.pi / 360) /
                   np.tan(np.pi/4 + latlim[0] * np.pi / 360))
            / (2 * np.pi))

    npx, npy = int(np.round(npx)), int(np.round(npy))

    center = np.mean(latlim), np.mean(lonlim)

    gmap = Gmap(center=center,
                zoom=zoom,
                size=(npx, npy),
                maptype=maptype)

    return gmap


def xylim2LL(xlim, ylim):
    '''Return RD limits as (lon1, lon2), (lat1, lat2) limits

    parameters
    ----------
        xlim : (x1, x2)
            xlim in RD coordinates
        ylim: (y1, y2)
            ylim in RD coordinates
    '''
    if xlim[1] < xlim[0]:
        xlim[0], xlim[1] = xlim[1], xlim[0]
    if ylim[1] < ylim[0]:
        ylim[0], ylim[1] = ylim[1], ylim[0]

    xc, yc = np.mean(xlim), np.mean(ylim)

    XY = np.array([[xlim[0], yc], [xlim[1], yc], [xc, ylim[0]], [xc, ylim[1]]])

    lon, lat = rd2wgs(*XY.T)

    lonlim = lon[0], lon[1]
    latlim = lat[2], lat[3]

    return lonlim, latlim


def zoom_fr_lonlim(lonlim, npx=640):
    '''Return zoom level that matches lonlim with npx pixels.

    parameters
    ----------
        lonlim : (lon1, lon2)
            longitude extends in degrees
        npx: max number of pixesl within these limits
    '''

    if lonlim[1] < lonlim[0]:
        lonlim[0], lonlim[1] = lonlim[1], lonlim[0]

    zoom  = np.log(npx / 256 * 360 / (lonlim[1] - lonlim[0])) / np.log(2)

    return zoom


def zoom_fr_latlim(latlim, npy=640):

    if latlim[1] < latlim[0]:
        latlim[0], latlim[1] = latlim[1], latlim[0]

    npy = 256 * 2 ** 15 * np.log(np.tan(np.pi / 4 + np.pi * latlim[1] / 360)  /
           np.tan(np.pi / 4 + np.pi * latlim[0] / 360)) / (2 * np.pi)

    zoom = (np.log((2 * np.pi) *(npy / 256)) -
            np.log( np.log(np.tan(np.pi / 4 + np.pi * latlim[1] / 360)  /
                           np.tan(np.pi / 4 + np.pi * latlim[0] / 360)))
                ) / np.log(2)
    return zoom


class Gmap:

    def __init__(self, center, zoom=14, maptype='roadmap',
                  size=(640, 640),
                  scale=1, format='png', language='en', region=None,
                  markers=None, path=None, visible=None,
                  style=None, timeout=0.01, signature=None):
        '''Return a google maps image

        for API see
        (https://developers.google.com/maps/documentation/' +
             'maps-static/dev-guide?hl=pt-BR)

        parameters
        ----------
            center: (required of markers are not present)
                defines the center of the map, equidistant from all edges of the
                map. This paramter takes a location s either a comma-separated
                `latitute`, `longitude` pair or a string address ('e.g. "city hall,
                new york, ny") identifying a unique location on the face of the
                earth. For more information see Locations.
                sequence of 2 floats or str
                center='Heemstede,Netherlands'
            zooml : int (required of makers not present)
                defines the zoom level of the map, which determines the
                magnification leel of the map. This parmaeter takes a numerical
                value corresponding to the zoom level of the region desired. E.g.
                an integer value between 0 and 21.
            maptype : str
                roadmap|satellite|hybrid|terrain
            size : (npx, npy) tuple of two floats
                hor and vert number of pixels, max (640, 640)
            scale: (1 default) or 2
                higher number of pixels for the same map coverage and contens.
            format: png (default), jpg, gif etc.
                a few of the formats for saving the image
            language: en default
                language in which to print labels
            timeout: float (seconds)
                timeout in case response is slow.
            region:
                defines appropriate borders to display, based on geo-political
                sensitivities. Accepts a region code specified as a two-character
                TLD('top level domain') value.
        Feature parameters
        ------------------
            markers : markers
                defin one or more markers to attach to the image at specifie
                locations. This parameter takes a single marker definition with
                parameters separated by teh pip character (|). Multiple markers
                may be place within the same markers parameter as l ong as they
                exhibit the same style; you may add additional markers of differenig
                styles by adding addtional markers parametes. Not that if you supply
                markers for a map, you do nit need to specify the (normally required)
                center and zoom parameters. For more informaiton, see Maps Static API
                Markers.
                markers=markerStyles|markerLocation1|markerlocaion2| ... etc
                marker=[{size: .., color: .., label: .., locations ...}] where
                size  = (optional) {tiny, mid, small}
                color = (optional) {black, brown, green, purple, yellow, blue, gay, red, white}
                label = (optional) single afpha numerid code [A..Z, 0..9]
                markers:size:mid|color:blue|label:A|2102CR54|2102CR76!2102CR80

            path : sequence of x, y tuples (default = None)
                Defines a single path of tow ormore connected points to overlay
                on the image at speciied locations. This parameter takes a string
                of point defnitions separated by the pipe (|) character, or an encoded
                polyline using th eenc: prefix within teh location declaration of the
                path. You may supply addtional paths by adding addtional path
                paramters.Note that if yo supply a path, you do not need to specify
                the (normally required) center and zoom paramters. For more
                information, see Maps Static API Paths.
            Visible: speifies one or more locations that should remain visible on
                the map, though no markers or other indicators will be displayed.
                Use this paramters to ensure that certain features or map locations
                are shown on the Maps Static API
            Style: defines a custom style to alter the presentation of a specific
                feature (roads, parks and other features) of the map. This parameter
                takes feature and element arguments identifying the features to
                style, and a set of style operations to apply t othe selected features.
                You can supplmultiple styles by adding additional styple parameters.
                For more information, see the guide to styled maps.
        Key and signature parameters
        ----------------------------
            key: (requied) allows you to monitor your application's API usage in the
                'Google Cloud Platform Consol', enables access to generous
                (free daily quota) and ensure that Google can contact you bout your
                application if necessary. For more information see 'Get a Key and
                Signature'
            Signature: (recommended) is a digital signature use to verify that any
                site generating requests using your API is authorized to do so.
                Note: if you enable billing, the digital signature is required.
                If you exceed the free daily limit of map loads, additional map
                loads are billable for the remainder of that day. Billable map
                loads that do not include a digital signature will fail. For more
                information see 'Get a Key and Signature'

        '''

        # Google statib map KEY was put in in os.environ['mapapi']

        center_ = center if isinstance(center, str) else '{},{}'.format(*center)
        size_  = '{}x{}'.format(*size)

        # params for requests
        params = {'center' : center_,  # home can be any address
                  "zoom"   : zoom,    # between 0 and 21
                  "size"   : size_,    # max 640x640
                  "maptype": maptype, # satellite, hybrid, ...  4 types
                  "timeout": timeout,    # timeout should always be used
                  "format" : format,     # image format
                  }

        if scale==2:  params['scale'] = '2'
        if path:      params['path'] = path
        if markers:   params['markers'] = markers
        if region:    params['region'] = region
        if visible:   params['visible'] = visible
        if style:     params['style'] = style
        if signature: params['signature'] = signature


        params["key"] = gmkey()

        self.__dict__.update(**params)

        self.center = center # keep original
        self.size = size     # keep original

        # Do the request
        API = 'https://maps.googleapis.com/maps/api/staticmap?'
        response = requests.get(API, params=params)

        # Get the image from the response
        self.img = Image.open(BytesIO(response.content)) # this reads and image object

        return

    def imshow(self, **kwargs):
        '''Show the image and return its axis.

        kwargs
        ------
            extent: (xlim1, xlim2, ylim1, ylim2)
                extent of the image in user coordinates
                can be specified conveniently as
                extend=(*xlim, *ylim)
            aspect: aspect ration
                set to 'equal' by default

            all kwargs are passed to plt.imshow

        returns
        -------
        ax: plt.Axis
        '''
        if not 'aspect' in kwargs:
            kwargs['aspect'] = 'equal'

        plt.imshow(self.img, **kwargs)
        return plt.gca()


    def px(self, lon):
        '''Return number of pixels from  self.center to lam.

        parameters
        ---------
            lin : float
                Easting (X) in degrees


        @TO 20180819
        '''

        px = 256 * 2 ** self.zoom * (lon - self.center[1]) / 360.

        return px

    def py(self, lat):
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
        py = 256 * 2 ** self.zoom * np.log(
              np.tan(np.pi / 4 + np.pi * (lat - self.center[0]) / 360))

        return py

    def pxpy(self, lon, lat):
        '''Return position of point (lon, lat) in pixels relative to center of bitmap.

        parameters
        ---------
            lon : float
                Easting (X) in degrees
            lat: float
                Northing (Y) in degrees
                '''
        return self.px(lon), self.py(lat)


    def lon(self, px):
        '''return longitude (degrees) given pixel coordinate px.

        parameters
        --------
            px : float/int
                pixel coordinate on map
        '''
        lon = self.center[1] + (px / 256) * 360 * 2 ** (-self.zoom)
        return lon

    def lat(self, py):
            '''return latitude (degrees) from py0, py, lam0 and zoomlevel

        parameters
        --------
            py, py0: float or its
                desired and known pixel north-south (y)
        '''

            lat = (360/np.pi) * np.arctan(
                    np.tan(np.pi / 4 + self.center[0]* np.pi / 360) *
                    np.exp( 2 * np.pi * py / 256 * 2 ** (-self.zoom))) - 90
            return lat

    def lonlat(self, px, py):
        '''return lonlat (degrees) from px and py of map.

        paamters
        --------
            px, px0: float or its
                desired and known pixel east-west (x)
            py, py0: float or its
                desired and known pixel north-south (y)
        '''

        return self.lon(px), self.lat(py)

    def xy(self, px, py):
        lat, lon = self.lonlat(px, py)
        return wgs2rd(lon, lat)

    @property
    def xlim(self):
        '''Return the xlim of the map.'''
        lon1 = self.lon(-0.5 * self.size[0])
        lon2 = self.lon(+0.5 * self.size[0])
        x1, y = wgs2rd(lon1, self.center[0])
        x2, y = wgs2rd(lon2, self.center[0])
        return x1, x2

    @property
    def ylim(self):
        '''Return the ylim of the map'''
        lat1 = self.lat(-0.5 * self.size[1])
        lat2 = self.lat(+0.5 * self.size[1])
        x, y1 = wgs2rd(self.center[1], lat1)
        x, y2 = wgs2rd(self.center[1], lat2)
        return y1, y2

    @property
    def UL(self):
        return self.lon(-self.size[0]), self.lat(+self.size[1])

    @property
    def LL(self):
        return self.lon(-self.size[0]), self.lat(-self.size[1])

    @property
    def LR(self):
        return self.lon(+self.size[0]), self.lat(-self.size[1])

    @property
    def UR(self):
        return self.lon(+self.size[0]), self.lat(+self.size[1])

    @property
    def ULrd(self):
        return wgs2rd(*self.UL)

    @property
    def LLrd(self):
        return wgs2rd(*self.LL)

    @property
    def LRrd(self):
        return wgs2rd(*self.LR)

    @property
    def URrd(self):
        return wgs2rd(*self.UR)

    @property
    def bb(self):
        lon1 = self.lon(-0.5 * self.size[0])
        lon2 = self.lon(+0.5 * self.size[0])
        lat1 = self.lat(-0.5 * self.size[1])
        lat2 = self.lat(+0.5 * self.size[1])
        return lon1, lat1, lon2, lat2

    @property
    def bb_rd(self):
        x1, x2 = self.xlim
        y1, y2 = self.ylim
        return x1, y1, x2, y2

    @property
    def pgon_bb(self):
        lon1, lat1, lon2, lat2 = self.bb

        lon = [lon1, lon2, lon2, lon1, lon1]
        lat = [lat1, lat1, lat2, lat2, lat1]
        return np.array([lon, lat]).T

    @property
    def pgbb_rd(self):
        xy = self.pgon_bb
        return wgs2rd(*xy.T)

    def xcyc(self):
        return wgs2rd(self.center[1], self.center[0])

# ===============  Example  ===================================================

if __name__=="__main__":

    #center = 'Heemstede, Netherlands'
    #gmap1 = Gmap(center, zoom=14, maptype='roadmap',
    #              size=(640, 640),
    #              scale=1, format='png', language='en', region=None,
    #              markers=None, path=None, visible=None,
    #              style=None, signature=None)
    #gmap1.imshow()
    #plt.show()


    '''
    home = '{:.6f},{:6f}'.format(*center)

    markers='size:mid|color:yellow|label:M|' + home

    markers='size:mid|color:yellow|label:M|Heemstede,Netherlands,Strawinskylaan 54'

    markers=['size:mid|color:yellow|label:T|Strawinskylaan 54',
             'size:mid|color:red|label:Z|Strawinskylaan 74']
    '''

    xlim = (182300, 183700) # 1400
    ylim = (336400, 337600) # 1200


    gmap = get_gmap_from_RDwindow(xlim, ylim, maptype='hybrid',  timout=1.0, scale=2)

    ax = gmap.imshow(extent=(*xlim, *ylim), aspect='equal')
    ax = plt.gca()

