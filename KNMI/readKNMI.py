#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 23:33:55 2017

@author: Theo
"""
import numpy as np
import pandas as pd
import pickle

fname = "KNMI_20170130.txt"

with open(fname, "rb") as fd:
    meteo = pd.read_csv(fname, index_col=1,
                        parse_dates=True,
                        header=47,
                        skipinitialspace=True)

# RH       = Etmaalsom van de neerslag (in 0.1 mm) (-1 voor <0.05 mm);
# EV24     = Referentiegewasverdamping (Makkink) (in 0.1 mm);

#NE will be in mm/d
NE = meteo[['RH', 'EV24']]/10.
NE['RH'][NE['RH']<0] = 0.05
NE['EV24'][NE['EV24']<0] = 0.05

# keep only series where makkink is available
NE = NE[pd.notnull(NE['EV24'])]  # now starts at 3 april 1987

times = np.array(NE.index)

print("{} lines of data between {} and {}".format(len(NE), NE.index[0], NE.index[-1]))

NE.plot()

with open("NE.pkl", 'wb') as fd:
    pickle.dump(NE,fd)