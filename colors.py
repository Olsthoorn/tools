#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 23:43:06 2018

# generate sufficient useful colors

@author: Theo
"""

import matplotlib

col = [c for c in matplotlib.colors.cnames.keys()\
              if not c.startswith('white') or c.endswith('white')]
colors = []
n = 20
for i in range(n):
    colors =colors + col[i::n]
