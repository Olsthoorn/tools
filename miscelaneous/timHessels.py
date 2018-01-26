#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 19:35:42 2017

@author: Theo
"""
# Chips has daily rainfall data of the world of very goog quality

import pandas as pd

startdate = '2016-12-01'
enddate   = '2016-12-31'

dates = pd.date_range(startdate, enddate, freq='D')

for date in dates:
    year = date.year
    month = date.month
    day = date.month
    # match the file name on the ftb servuer.
    name_CHIRPS_tif = "chirpsv2.0.%d.%02d/%02d'.tif % (year, month day)
    print(name_CHIRPS_tif) # to download


# glob is more efficient than with pandas
imiport os
import glob
os.chdir(input_folder)

CHrIPSALLFIles = glob.flog('*.tif')  # more than one wildcard permitted


import gdal,glob os
input_folder where al chirps data are
chanbe working directory
search with all the files that start with ...
File = Files[0]

Filname_name = os.path.join(input_folder, file)

os.sep

g =gdal.Open(File_name)
# now you can read all kind of dat out of g
g.GetRasterBand(1)  #normally gtiff data only has one band
Band_data = g.GetRasterBand(1)
Array = Band_data.ReadAsArray()
print("Maximum Vlaue", Array.std())
Array[Array > -20] = 20
print("max value", Array.max())
Array[Array == 0] = -9999.
# Geolocation data
# Six numbers: = Geotransform parameters
"""
   1 Origin X  # upper left corner
   2 Origin Y
   3 Pixel Width
   4 Pixel Height
   5 Rotation in Y diretion (is 0 in Case 1) # mostly 0
   6 Rotation in X direction(is 0 in Case 1) # mostly 0
    """

GeoTransform = g.GetGeoTransform()  # weird order of the 6 parameters
Proj = g.GetProjection()  # necessary to recreate the goetiff again

'''
Open in QGIS

mostly red due to NoData values. Set -9999. s nodata values.

Array = g.GetRasterBand(1).ReadAsArray()
Array[Array == 0.0] = -9999.
file_out = 'CHIRPS_NVisZero.tif'
file_out_path = os.path.join(Input_folder, file_out)
driver = gdal.GetDriverByName(''GTIFF')
dst_ds =driver.Create(file_out_path, int(Array.shape[0]), 1, gdal.GDT_Float32)
dst_ds.SetProjection(Proj)
dst_ds.SetGeoTransform(GeoTransform)
dst_ds.GEtRasterBand(1).SetNoDataValue(-9999)
dst_ds.GetrasterBand(1).WriteArray(Array)
dst_ds = None