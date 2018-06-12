#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 13:32:41 2018

@author: Theo 180609
"""

import os

def test4illegalChars(path, ext, chars=['&']):
    '''Return filenames with illegal characters.
    
    parameters
    ----------
    path : str
        path to file (such that os.path.isfile() is True)
    ext : str
        file extension to inspect
    chars: list or str
        list of illegal charactors or str with illegal characters to inspect.    
    '''
        
    
    count = 1
    fnames = []
    for fname in os.listdir(path):
        if fname.endswith('1.4.xml'):
            with open(os.path.join(p, fname)) as f:
                s = f.read()
            for c in chars:
                if c in s:
                    fnames.append(f)
                    print('{} contains illegal char <{}>'.format(f, c))
                    count +=1
    print(count, 'files contained illegal char &.')
    return fnames
    

def seeChars(path, fname, r=None):
    '''Return a few characters of text file for inspection.

    parameters
    ----------
    path : str
        path
    fname  : str
        filename
    r : tuple (int, int)
        start and end index of chars to be inspected.
    '''
    if r is None: r = (0, 10)
    
    with open(os.path.join(path, fname), 'r') as f:
        s = f.read()
        print(s[slice(r[0],r[1])])
        return s


if __name__ == '__main__':

    p = '/Users/Theo/Instituten-Groepen-Overleggen/HYGEA/Consult/2017_DEME-julianakanaal/DINO_boringen/cfb9fea6-38cb-48b2-b44f-0d1fb8be67c7/Boormonsterprofiel_Geologisch booronderzoek/'
    
    fnames = test4illegalChars(p, '1.4.xml', '&')

