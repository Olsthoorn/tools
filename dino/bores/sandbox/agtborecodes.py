#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 18:06:01 2018

@author: Theo
"""

# Codes to generate xml file borehole specification for DEME boreholes
# As given in spreadsheet "Boorgatgegevens"

lithocode = {'onbekend':      'ONB',
         'Dijklichaam':       'AAOP',
         'Breda/Rupel':       'BR',
         'Breda':             'BR',
         'Maarland':          'BEOM',
         'Schimmert':         'BXSC',
         'Beegden/Breda':     'BE',
         'Beegden':           'BE',
         'Rupel':             'RU',
         'Schimmert/Beegden': 'BXSC'}


lithology= {
    'geen monster':  'GM',
    'puin'   :  'PU',
    'opvulling' : 'PU',
    'leem'   :  'L',
    'zand'   :  'Z',
    'veen'   :  'V',
    'klei'   :  'K',
    'zeer harde/vaste klei':  'K',
    'humus'  :  'DET',
    'teelaarde': 'TLA',
    'gras'   :  'GRA',
    'grind'  :  'G',
    'beton'  : 'STN',
    'steen'  : 'STN',
    'stenen' : 'STN',
    'tegel'  : 'STN',
    'slakken': 'MI'
}

siltAdmix = {
    'sterk siltig':  'S3',
    'siltig':  'SX',
    'siltfractie aanwezig': 'S2',
    'siltige bijmenging':  'S2',
    'zwak siltig':  'S1',
    'matig siltig':  'S2',
    'zwak siltig bruin':  'S1',
    'matig leemhoudend':  'S2',
    'brokken leem':  'S2',
    'laagjes leem':  'S2',
    'sterk leemhoudend':  'S3',
    'zwak leemhoudend':  'S1',
    'lenzen leem':  'S2'
}

organicAdmix = {
    'sterk veenhoudend':  'VEB3',
    'sporen veen':  'VEB1'
}

humusAdmix = {
     'zwak humeus': 'H1',
     'met humeus materiaal': 'H1',
     'matig humeus': 'H2',
     'sterk humeus': 'H3',
     'humeus': 'HX'
 }

coarseMatAdmix = \
{'sporadisch stenen': 'ST1',
 'met stenen' : 'STX',
 'met stenen en weinig zand': 'STX',
 'stenen (grind meestal 3 cm  en max 10 cm = steen)': 'STX'
}

clasticAdmix = \
{'leembrokken tussen grind': 'LEBx',
 'leembrokken' : 'LEBx'
}

gravelAdmix = {
    'enkel grindje':  'G1',
    'sporadisch grind': 'G1',
    'sporatdisch zeer grof grind':'G1',
    'zwak grindig':  'G1',
    'zwak grindige bijmenging':  'G1',
    'met weinig grind' : 'G1',
    'laagjes grind':  'G2',
    'matig grindig':  'G2',
    'met matig grof grind': 'G2',
    'met fijn tot zeer grof grind (max 2cm)':'G2',
    'met zeer grof grind (max 2cm)':'G2',
    'platte korrels van grind van max 3 cm':'G2',
    'sterk grindig':  'G3',
    'grindige bijmenging':  'GX',
    'met grind' : 'GX',
    'bijmenging zeer grof grind':  'GX',
    'grof grindig':  'G3'
}



gravelMedianClass = {
    'zeer fijn': 'GZF',
    'fijn': 'GFN',
    'fijn tot medium' : 'GMF',
    'fijn tot dominant medium' : 'GMF',
    'fijn tot dominant matig grof': 'GMG',
    'matig fijn': 'GMF',
    'matig grof': 'GMG',
    'grof': 'GGR',
    'zeer grof' : 'GZG',
    'uiterst grof' : 'GUG',
    'grof tot matig grof grind':  'GMG',
    'zeer grof grind':  'GZG',
    'fijn tot grof grind':  'GMG',
    'fijn tot dominant grof': 'GMG',
    'matig grof tot grof grind':  'GMG',
    'matig grof grind':  'GMG',
    'matig grof grind met een enkele steen':  'GMG',
    'grof grind':  'GGR',
    'fijn grind':  'GFN',
    'grind tot 15cm':  'GZG',
    'fijn tot grof': 'GMG',
    'fijn tot zeer grof': 'GGR',
    'matig tot zeer grof': 'GGR',
    'dominant matig grof to grof': 'GGR',
    'zeer grof grind (10-15cm)':  'GZG',
    'matig grof grind met grote keien (10-15cm)':  'GMG'
}

sandMedianClass = {
    '<125µm': 'ZZF',
    '125µm -180µm': 'ZFN',
    '<180µm': 'ZFN',
    '180µm': 'ZFN',
    '180µm - 250µm' : 'ZMF',
    '>250µm': 'ZGX',
    '250µm - 355µm': 'ZGX',
    '<355µm' : 'ZGX',
    '>355µm' : 'ZZG',
    '355µm - 500µm' : 'ZZG',
    '>710 µm': 'ZUG',
    'uiterst fijn': 'ZUF',
    'zeer fijn': 'ZZF',
    'zeer fijn tot matig fijn': 'ZFN',
    'zeer fijn tot matig grof': 'ZFN',
    'fijn': 'ZFN',
    'fijn tot zeer grof': 'ZMG',
    'matig to zeer fijn' : 'ZFN',
    'matig fijn': 'ZMF',
    'matig grof': 'ZMG',
    'grof': 'ZGX',
    'zeer grof': 'ZZG',
    'zeer grof tot uiterst grof': 'ZUG',
    'uiterst grof': 'ZUG',
}


sandAdmix = {
    'sporadisch zand': 'Z1',
    'zwak zandig' : 'Z1',
    'weinig zandig': 'Z1',
    'matig zandig':  'Z2',
    'zandige bijmenging':  'Z2',
    'grof zandig':  'Z3',
    'verkitte bruine zandbrokjes':  'Z1',
    'zwak zandig':  'Z1',
    'verkitte zandbrokjes':  'Z1',
    'laagjes zand':  'Z2',
    'matig tot sterk zandig': 'Z2',
    'matig zandhoudend':  'Z2',
    'zandig':  'ZX',
    'matig to sterk zandig':'ZX',
    'matig zandig is uiterst grof': 'ZX',
    'matig zandig is zeer to uiterst grof':'ZX',
    'sterk siltige zandbrokken':  'Z2',
    'brokken zand':  'Z2',
    'sterk zandig': 'z3',
    'uiterst zandig':  'Z4'
}

clayAdmix = {
    'matig kleiig':  'K2',
    'laagjes klei':  'K2',
    'sterk kleiig':  'K3',
    'weinig kleiig':  'K1',
    'zwak kleiig':  'K1',
    'zandige klei' : 'K2',
    'kleiig' : 'K2',
    'kleiig (kan je worstjes mee draaien)': 'K2',
    'kleilaagjes zijn zwartbruin':  'K2',
    'kleilaagjes <5cm':  'K2',
    'brokken klei':  'K2',
    'lenzen klei':  'K2'
}

subLayerLithology = \
{'kleilenzen aanwezig in grind als brokken': 'SKL2',
 'laagje in breda' : 'SKL1',
 'met kleilensjes' : 'SKL2',
 'met leemlagen' :  'SLLX',
 'zwak lemig' : 'SLL1',
 'matig tot sterk lemig': 'SLLX',
 'lemig' : 'SLLX',
 'zwak lemig': 'SLL1',
 'leem is zandig' : 'SLLX'
}


shellFrac = {'resten schelpen':  'SCHX'}


plantType = {
    'zwak wortelhoudend': 'WOR1',
    'matig wortelhoudend': 'WOR2',
    'wortels': 'WOR2',
    'resten wortels': 'WOR2',
    'sterk wortelhoudend':  'WOR2',
    'zwak plantenhoudend':  'PLA1',
    'matig plantenhoudend':  'PLA2',
    'zwak houthoudend':  'HOR2',
    'resten hout':  'HOR3'
}

glaucFrac = {
    'spoor glauconiet' : 'GC1',
    'weinig glauconiet': 'GC2',
    'veel glauconiet'  : 'GC3',
    'zeer veel glauconiet' : 'GC4',
    'glauconiet'       : 'GCX',
    'zwarte spikkels in zand':  'GCX',
    'zwarte korrels'   : 'GCX',
    'met groene schijn': 'GCX'
 }


      # in descr        li/dark    primary   second   matplotlibcolor
primSec = {'beige'      : (None,   'beige',  None,   'beige'),
     'beigebruin'       : (None,   'beige', 'brown', 'khaky'),
     'beigegrijs'       : (None,   'beige', 'gray',  'blanchdalmond'),
     'bruin'            : (None,   'brown',  None,   'chocolate'),
     'bruin groen geel' : (None,   'green', 'yellow', 'y'),
     'bruinbeige'       : (None,   'brown', 'beige', 'khaky'),
     'bruingeel'        : (None,   'brown', 'yellow','gold'),
     'bruingrijs'       : (None,   'brown', 'gray',  'tan'),
     'bruingroen'       : (None,   'brown', 'green', 'darkkhaky'),
     'bruinoranje'      : (None,   'brown', 'orange','darkorange'),
     'bruinrood'        : (None,   'brown', 'red',   'brown'),
     'bruinzwart'       : (None,   'brown', 'black', '....'),
     'donker bruin'     : ('dark', 'brown',  None,   'saddlebrown'),
     'donker bruingrijs': ('dark', 'brown', 'gray',  'sienna'),
     'donker grijsbruin': ('dark', 'gray',  'brown', 'sienna'),
     'donker grijsgroen': ('dark', 'gray',  'green', '.....'),
     'donker groengeel' : ('dark', 'green', 'yellow', '....'),
     'donker groengrijs': ('dark', 'green', 'gray',  'cadetblue'),
     'donker roodbruin' : ('dark', 'red',   'brown', 'darkred'),
     'donker zwartbruin': ('dark', 'black', 'brown', 'maroon'),
     'donker zwartgrijs': ('dark', 'black', 'gray',  'gray'),
     'donkerbruin'      : ('dark', 'brown',  None,   'sienna'),
     'donkergrijs'      : ('dark', 'gray',   None,   'darkgray'),
     'geel'             : (None,   'yellow', None,   'yellow'),
     'geelbruin'        : (None,   'yellow','brown', 'khaky'),
     'geelgrijs'        : (None,   'yellow','gray',  'lightyellow'),
     'grijs'            : (None,   'gray',   None,   'grey'),
     'grijs-lichtbruin' : ('light','gray',  'brown', 'moccassin'),
     'grijsbeige'       : (None,   'gray',  'beige', 'floralwhite'),
     'grijsbruin'       : (None,   'gray',  'brown', 'burlywood'),
     'grijsbruinrood'   : (None,   'gray',  'brown', 'rosybrown'),
     'grijsgeel'        : (None,   'gray',  'yellow','palegoldenrod'),
     'grijsgroen'       : (None,   'gray',  'green', 'khaky'),
     'grijsrood'        : (None,   'gray',  'red',   'rosybrown'),
     'groen'            : (None,   'green',  None,   'green'),
     'groenbruin'       : (None,   'green', 'brown', 'y'),
     'groengeel'        : (None,   'green', 'yellow','greenyellow'),
     'groengrijs'       : (None,   'green', 'gray',  'mediumaquamarine'),
     'licht beigebruin' : ('light','beige', 'brown', 'peachpuff'),
     'licht bruingeel'  : ('light','brown', 'yellow', '....'),
     'licht beigegrijs' : ('light','beige', 'gray',  'seashell'),
     'licht bruinbruin' : ('light','brown', 'brown', 'sandybrown'),
     'licht bruingrijs' : ('light','brown', 'gray',  'peachpuff'),
     'licht bruinrood'  : ('light','brown', 'red',   'lightsalmon'),
     'licht grijsbruin' : ('light','gray',  'brown', 'peachpuff'),
     'licht grijsgroen' : ('light','gray',  'green', 'darkseagreen'),
     'licht groenbruin' : ('light','green', 'brown', 'khaky'),
     'licht groengrijs' : ('light','green', 'gray',  'aquamarine'),
     'licht oranjebruin': ('light','orange','brown', 'darkorange'),
     'lichtbeige'       : ('light', 'beige', None,   'beige'),
     'lichtbruin'       : ('light', 'brown', None,   'burlywood'),
     'lichtgrijs'       : ('light', 'gray',  None,   'lightgrey'),
     'okerbruin'        : (None,    'beige','Brown', 'goldenrod'),
     'oranje'           : (None,    'orange',None,   'orange'),
     'oranjebruin'      : (None,    'orange','brown','darkoranje'),
     'oranjegrijs'      : (None,    'orange','gray', 'peachpuff'),
     'roestbruin'       : (None,    'brown', None,   'darkorange'),
     'rood'             : (None,    'red',   None,   'red'),
     'roodbruin'        : (None,    'red',  'brown', 'firebrick'),
     'witbeige'         : (None,    'white','beige', 'beige'),
     'witgrijs'         : (None,    'white','gray',  'gainsboro'),
     'zwart'            : (None,    'black', None,   'black'),
     'zwartbruin'       : (None,    'black','brown', 'maroon'),
     'zwartgrijs'       : (None,    'black','gray',  'gray'),
     'zwartrood'        : (None,    'black','red',   'darkred')
     }

# example lithography

import pandas as pd

def split_lithography(workbook, sheetname=None):
    '''Return catagorized lithography

    Each line  in the lithololgy column is split on the comma's
    and catagorzed based on the material.
    '''

    try:
        # read the litholoties into a list
        lth = list(pd.read_excel(workbook, sheetname=sheetname, header=None)[0])

    except:
        Exception("For this the sheet must exist in the workbook !")

    # turn each line in the list into a list split on commas
    lth1 = [b.lower().replace('.','').split(',') for b in lth]
    lth2 = [[a.strip() for a in b] for b in lth1]
    lth3 = {k[0]: set(k[1:]) for k in lth2}

    for k in lth3:
        for L in lth2:
            if k == L[0]:
                lth3[k] = lth3[k].union(L[1:])

    for k in lth3:
        print()
        print(k)
        print()
        for a in sorted(lth3[k]):
            print(a)
        print()

    # total set of subphrases
    subphrases = set()
    for k in lth3:
        subphrases = subphrases.union(lth3[k])

    print("\nSubphrases for admix and medium classes:\n")
    for sphr in subphrases:
        print(sphr)

import re
def get_colors(str):

    colorRE = re.compile(r'(licht|donker)(grijs|bruin)+')
    mo = colorRE.findall('donkergrijs met bruine met spikkels')
    #colorRE.findall('licht bruingrijs met bruine met spikkels')
    mo



    if __name__ == '__main__':

        workbook = 'Boorgatgegevens.xlsx'
        sheetname= 'Sheet1'

        split_lithography(workbook, sheetname)



