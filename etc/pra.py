#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 11:08:55 2018

@author: Theo
"""
__all__ = ['pra']

import numpy as  np
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)-15s -  %(levelname)s - %(message)s')
logger = logging.getLogger('__name__')


def pra(A, cols=10, fmt='10.2f', name='', **kwargs):
    '''pprint a 2D array.
    
    parameters
    ----------
    A: np.ndarray
        a 2D array
    cols : int or tuple
        if int:
            cols is number of columns to print
        if tuple:
            cols indicates slice of indices of columns to print.
            Use cols=(i,i+1) to print only column i
    fmt : str
        format code useful in '{:fmt}.format() like '10.2f' 
    '''
    Fmt = '{:' + fmt + '}'
    
    shape = A.shape
    if len(shape) == 1:
        pra(A[np.newaxis, :], cols=cols, fmt=fmt, name=name, **kwargs)
    if len(shape) == 2:
            if isinstance(cols, int):
                cols = (0, min(cols, A.shape[-1])) # makes cols a tuple.
            if name:
                print("Array '{}', ".format(name), end='')
            if 'block' in kwargs:
                print('Block={}, '.format(kwargs.pop('block')), end='')
            if 'layer' in kwargs:
                print('Layer={}, '.format(kwargs.pop('layer')), end='')
            print('Columns=[{}:{}]'.format(*cols))
            incr = 1 if cols[1] > cols[0] else -1
            for aRow in A:
                print(' '.join([Fmt.format(v) for v in aRow[slice(*cols, incr)]]))
            print()

    elif len(shape) == 3:
        for i, Aplane in enumerate(A):
            kwargs['layer'] = i
            pra(Aplane, cols=cols, fmt=fmt, name=name, **kwargs)
    elif len(shape) == 4:
        for i, Ablock in enumerate(A):
            kwargs['block'] = i
            pra(Ablock, cols=cols, fmt=fmt, name=name, **kwargs)            
    elif len(shape) > 4:
        raise ValueError('A.ndim must be <5')

    return
        
        
if __name__ == '__main__':
    
    print('Vector:')
    pra(np.random.rand(6), name='A vector')

    print('2D array:')    
    pra(np.random.rand(5, 6), name='A 2D array', fmt='.5g')

    print('3D array:')
    pra(np.random.rand(5, 6, 7), name='Heads')
    
    print('4D array:')
    pra(np.random.rand(5, 6, 3, 4), name='My_4D_array')    
