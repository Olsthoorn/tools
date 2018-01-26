#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 13:03:27 2017

Demo of packages in python see pages 234 ff in the book:
    The Quick Python Book by Naomi R. Ceder, (Manning 2013)

This example is placed in the mathproj directory to prevent clutter.

@author: Theo
"""

import sys

if not '..' in sys.path:
    sys.path.insert(1, '..')

import mathproj

mathproj.version

mathproj.comp  # generate tracback because not (yet) loaded

mathproj.comp.numeric.n1

import mathproj.comp.numeric.n1

mathproj.comp.numeric.n1.g()

mathproj.comp

mathproj.comp.numeric