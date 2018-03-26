# -*- coding: utf-8 -*-
"""
Spyder Editor

Try to define  grammar to analyze the lithology in the file "Boorgatgegevens.xlsx"
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import dinoborecodes as bc
import Boorgatgegevens_textreplacements as txrep
from importlib import reload

#%% Lets first read the sheet with the boreholes lithology

colors=set('blauw bruin geel groen grijs oranje rood wit zwart beige'.split())
admix=set('lemig kleig siltig zandig grindig'.split())


lith = pd.read_excel("20180316_Boorgatgegevens.xlsx", sheet_name='Lithology')
lith.columns
lithkey = set(lith['Lithology Keyword'])


#%%
    
class LithConv:

    def __init__(self, lithDescr, replacements=None):
    
        self.lithDescr = lithDescr
        self.replacements = replacements
    
    def phrases(self, i):
        phr = []
        for line in self.lithDescr:
            try:
                phr.append(line.split(',')[i].lower().strip())
            except:
                phr.append('')
        return phr
    
    def unique_phrases(self, i):
        return set(self.phrases(i))

    def replaced(self, i):
        rep = []
        for s in  self.phrases(i):
            for r in self.replacements[i]:
                if s==r[0]:
                    s=r[1]
            rep.append(s)
        return rep
    
    def replaced_set(self, i):
        return set(self.replaced(i))
    
  
lc = LithConv(lith['Lithology Description'], txrep.replacements)
    
i=7
lc.unique_phrases(i)
print('\n\n')
lc.replaced_set(i)


lc.phrases(0)

# mer ge Lithology Description

# replace the 'Lithology Keywords' where necessary
lc = LithConv(lith['Lithology Keyword'], txrep.keyw_replacements)
lc.unique_phrases(0)
lc.replaced_set(0)




#%%
gravel = re.compile('(matig |uiterst |zeer )*(fijn |grof |medium )*(grind)', re.IGNORECASE)
sand = re.compile('(matig |uiterst |zeer )*(fijn |grof |medium )*(zand)', re.IGNORECASE)
aarde = re.compile('(teelaarde|gras|humus)')
antro = re.compile('(antropogeen|beton|geen monster|puin|stenen|tegel|mijnsteen)')
klei = re.compile('(klei)')
leem = re.compile('(leem)')
for k in (k.lower() for k in lithkey):
    rg = gravel.search(k)
    rs = sand.search(k)
    aa = aarde.search(k)
    an = antro.search(k)
    kl = klei.search(k)
    lm = leem.search(k)
    print('{:>20}'.format(k), end='    -->  ')
    if rg: print(rg.groups())
    if rs: print(rs.groups())
    if an: print(an.groups())
    if aa: print(aa.groups())
    if kl: print(kl.groups())
    if lm: print(lm.groups())

 #%%       
    
tokens = (
        MAT,  # zand, klei, grind
        DEGR,  # zwak, matig, sterk
        GRAIN,  #
        COLOR, # bruin, rood, beige, bruinrood, geelgrijs
        MEDC,  # fijn, grof
        MCDEG, # zeer, matig, uiterst
        ADMIX, # zandig, grindig, kleiig, met grind, met zand
        
        )



lith_inv = invert_dict(bc.lithology)

def missing(combs, expected=set(lith_inv.keys())):
    
    


[DEG* + COARSENESS* + MAT]
