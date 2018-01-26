#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 17:45:17 2018

@author: Theo
"""

def pra(X, n=6):
    'Print an 2D array the matlab way.'
    [print(''.join(['{:10.4f}'.format(a) for a in x[:n]])) for x in X]