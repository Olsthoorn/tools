#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 00:23:37 2018

@author: Theo
"""

__all__ = ['ddir', 'linestyles_', 'colors_', 'linestyle_cycler']

import itertools

def ddir(obj):
    print([d for d in dir(obj) if not d.startswith('__')])
    
    
linestyles_ = ['-', '--', '-.', ':']
colors_ = ['b', 'r', 'g', 'k', 'm', 'c', 'brown', 'orange', 'gold', 'beige']

def linestyle_cycler():
    for l in itertools.cycle(linestyles_):
        for c in colors_:
            yield {'ls': l, 'color': c}



