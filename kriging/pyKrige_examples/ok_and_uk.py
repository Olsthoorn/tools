#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 10:56:41 2018

@author: Theo
"""

import pykrige
#import pykrige.kriging_tools as kt
import numpy as np

data = np.array([[0.3, 1.2, 0.47],
                 [1.9, 0.6, 0.56],
                 [1.1, 3.2, 0.74],
                 [3.3, 4.4, 1.47],
                 [4.7, 3.8, 1.74]])

gridx = np.arange(0.0, 5.5, 0.5)
gridy = np.arange(0.0, 5.5, 0.5)

OK = pykrige.OrdinaryKriging(data[:, 0], data[:, 1], data[:, 2], variogram_model='linear',
                     verbose=False, enable_plotting=False)

zOK, ssOK = OK.execute('grid', gridx, gridy)

#kt.write_asc_grid(gridx, gridy, z, filename="output.asc")


UK = pykrige.UniversalKriging(data[:, 0], data[:, 1], data[:, 2], variogram_model='linear',
                      drift_terms=['regional_linear'])

zUK, ssUK = UK.execute('grid', gridx, gridy)

print(zOK)

print(zUK)


