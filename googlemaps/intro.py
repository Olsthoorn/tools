# -*- coding: utf-8 -*-
"""
Spyder Editor

Google static maps API, see

https://developers.google.com/maps/documentation/static-maps/intro


Shows how to import a map from Google Maps using requests module

@ Theo 180118
"""



#%% Import modules

'''
# first part of Google Static Maps URL
https://maps.googleapis.com/maps/api/staticmap?parameters

# parameteres
    all separated using &
    they should be properly URL-encoded
    "|" = %7C
    see


    Location parmaeters
        center
            comma separted lat lon pair like (e.g. "40.71428,-73.998672")
            or a string address like "city hal, new york, ny"
        zoom
            int 0, .. 21
        scale
            used together wih size determines the actual ouput size in pixels
            size 1 or 2 (4 is for Google premium plan)
    Map parameters
        size
            {xpix}x{ypix} eg 500x400
            maps smaller than 180 pix display with reduced Google logo.
        format (optional)
            default png
            other: gif, jpeg
        maptype (optional)
            roadmap|satellite|hybrid|terrain|
        language (optional)
            used for labels
        region (optional)
            defines appropriate borders based on geo-political sensitivities
            is a two-character code value
    Feature parameters
        markers
            markers=markerStyles|markerLocatio1|markerLocaion2|...
            markerStyles
                size
                color  24bit color e.g. color=0xFFFFCC or one of
                    black, brown, green, purple, yellow, blue, orange, red, white
                label
                    a single uppercase character from {A-Z, 0-9}
                custom icon
                markers=icon:URLofIcon|markerLocation
                may use URL shorting service
                    https://googl.gl
                    URLs are automatically encoded.
        path
            a parh of two or more connected points to overlay the image.
            requires a sing of points sebarated by a pip character (|)
        polygon this is an encoded oplyline like
            &path=fillcolor:0xAA00033%7color:0xFFFFFF00%7Cenc:....(comlicated code)...
            see Google Static Maps API Paths
        visible
            specifies one or more locations that should remain visible on the map.
        style
            seems complicated. See later.
    key and signature parameters
        key (required)
            get a key signature
        signature (recommende)
            is a digital signature used to verify that any site generating
            requests using your API is authorized to do so.
            Free daily limit of map loads
    URL size restriction: 8192 characters
        latitudes and longitudes
            numerals within a comma-separated text string having a precision of
            up to 6 decimals.
        adresses must be URL escapde
            ' ' -> '+'
'''
import numpy as np
import requests
from PIL import Image
from io  import BytesIO


# Strawinskylaan 54, Heemstede, my home
lat= np.round(52 + (21 + (24.25)/60)/60, decimals=6)
lon= np.round( 4 + (37 + (47.63)/60)/60, decimals=6)
home = '{:.6f},{:6f}'.format(lat, lon)

# my personal key for downloading maps from Google maps
mykey = 'AIzaSyDRETF3BLxtT-W3c7dXlk-7t3j3Z7wLTN8'
API = 'https://maps.googleapis.com/maps/api/staticmap?'

# Example
params = {'center': home, # home can be any address
          "zoom"  : 14,   # between 0 and 21
          "size"  : "640x640", # max 640x640
          "maptype": "satillite", # satellite, hybrid, ...  4 types
          "markers": "2|yellow|A|{}".format(home), # a pin
          "timeout": 0.01, # timeout should always be used
          "key"   : mykey}

r = requests.get(API, params=params)

img = Image.open(BytesIO(r.content)) # this reads and image object

img # this shows the image object inline, the how must be in its __repr__

# write image to a file, chunk length can be chosen freely
with open('strawinsky_home.png', 'wb') as fd:
    for chunk in r.iter_content(chunk_size=128):
        fd.write(chunk)


# example from a Hitchhiker's guid to python
response = requests.get('http://pypi.python.org/pypi/requests/json')
type(response)
response.ok


