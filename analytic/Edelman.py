# Edelman.py
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import pandas as pd


def ierfc(n:int, u:float | np.ndarray)->float|np.ndarray:
    """Return repeated intefral of the complementary error function."""
    assert n >= -1, f'n must be >= -1, not {n}'
    if n == -1:
        return 2 / np.sqrt(np.pi) * np.exp(-u ** 2)
    if n == 0:
        return erfc(u)
    
    return -u/n * ierfc(n-1, u) + 1 / (2*n) * ierfc(n-2, u)

class Eleman:
    def __init__(self, kD, S):
        self.kD = kD
        self.S = S
        
    def s(self, A, n, x, t):
        """Return s and Q given that s0 = A * t ** (n/2)"""
        ie0 = ierfc(n, 0)
        u = x * np.sqrt(self.S / (4 * self.kD * t))
        s = A * t ** (n/2) * ierfc(n, u) / ie0
        Q = A / 2 * np.sqrt(self.kD * self.S) * t ** ((n-1)/2) * ierfc(n-1, u) / ie0
        return (s, Q)
    
    def Q(self, B, n, x, t):
        """Return Q and s given Q(0) = B t **(n/2)"""
        ie0 = ierfc(n+1, 0)
        u = x * np.sqrt(self.S / (4 * self.kD * t))
        Q = B * t ** (n/2) * ierfc(n, u) / ie0
        s = 2 * B / np.sqrt(self.kD * self.S) * t**((n+1)/2) * ierfc(n+1, u) / ie0
        return (Q, s)

def get_etable()->pd.DataFrame:    
    idx = [0, 0.1, 0.3, 0.65, 1.0, 1.6]
    data = [(1.0000, 1.0000, 1.0000, 1.0000, 1.0000),
        (0.8875, 0.9900, 0.8327, 0.7935, 0.7624),
        (0.6714, 0.9139, 0.5569, 0.4829, 0.4286),
        (0.3580, 0.6554, 0.2430, 0.1798, 0.1394),
        (0.1573, 0.3679, 0.0891, 0.0568, 0.0388),
        (0.0237, 0.0773, 0.0102, 0.0052, 0.0029), 
    ]
    columns = ['E1', 'E2', 'E3', 'E4', 'E5']
    return pd.DataFrame(index=idx, data=data, columns=columns)

def compare(etable, cols=[1, 2, 3, 30, 4, 40, 5]):
    """
    E1 = ierfc(0, u) / i_efc(0, 0)
    E2 = np.sqrt(np.pi) * ierfc(0, u) / ierfc(1, 0)
    E3 = np.sqrt(np.pi) * ierfc(1, u) / ierfc(1, 0)        
    E3 = ierfc(2, u) / ierfc(2, 0)
    E4 = ierfc(2, u) / i_efrc(2, 0)
    E4 = ierfc(2, u) / i_efrc(3, 0)        
    E5 = np.sqrt(np.pi / (4 / 9)) * ierfc(3, u) / ierfc(3, 0)                
    """
    u = np.asaray(etable.index)
    btable = etable.copy()
    
    for col in cols:
        btable[f'E{col}'] = edelm(col, u)
        
    return btable
    
    
def edelm(n, u):
    """Return Edelman table values for case n."""
    if n==1:
        return ierfc(0, u) / ierfc(0, 0)
    elif n==2:
        return ierfc(0, u) / ierfc(1, 0) * np.sqrt(np.pi)
    elif n==3:
        return ierfc(1, u)  / ierfc(1, 0) * np.sqrt(np.pi)
    elif n==30:
        return ierfc(2, 0) / ierfc(2, 0)
    elif n==4:
        return ierfc(2, u) / ierfc(2, 0)
    elif n==40:
        return ierfc(2, u) / ierfc(3, 0)
    elif n==5:
        return ierfc(3, u) / ierfc(3, 0) * np.sqrt(np.pi)
    raise ValueError(f"n={n} must be one of [1, 2, 3, 30, 4, 40, 5]")

    
    

    
    
    