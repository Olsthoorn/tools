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

#%% Lets first read the sheet with the boreholes lithology

colors=set('blauw bruin geel groen grijs oranje rood wit zwart beige'.split())
admix=set('lemig kleig siltig zandig grindig'.split())


lith = pd.read_excel("20180316_Boorgatgegevens.xlsx", sheet_name='Lithology')
lith.columns

#%% First Idea

    # split the columns 'Lithology Description' in subphrases
    # split off the first subphrase that contains the material
    # and compare it with the column 'Lithology Keyword'


#%% # interpret Lithology keyword
lithkey = set(lith['Lithology Keyword'])


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
    if aa: print(aa.groups())
    if an: print(an.groups())
    if kl: print(kl.groups())
    if lm: print(lm.groups())



#%%
lithkey1 = set([k.split(',')[0].lower() for k in lith['Lithology Description']])

replacements = [
 ('beton','antropogeen'),
 ('bruin tot zwart fijn zand met groene schijn (glauconiet)',
      'fijn zand, bruinzwart, glauconiethoudend'),
 ('bruin zand','zand, bruin'),
 ('dominant fijn tot grof grind met stenen ','grind, ongesorteerd, met stenen'),
 ('fijn tot grof grind laagje in breda','grind, laagje in Breda'),
 ('fijn tot grof grind met stenen', 'grind, met stenen'),
 ('fijn tot medium grind', 'matig fijn grind'),
 ('geen monster','antropogeen'),
 ('gras','teelaarde'),
 ('groenstrook','teelaarde'),
 ('klei met fijn tot zeer grof grind (max 2 cm)','klei, met ongesorteerd grind'),
 ('klei met zeer grof grind (max 2 cm)','klei, met zeer grof grind'),
 ('klei. klei is vettig en grijs en bevat oranje oxidatie plekken (limoniet/pyriet?). ',
      'klei, grijs, met roestplekken'),
 ('leem (vanaf 1.3 grind in leem)','leem met grind'),
 ('leem - teeltaarde','teelaarde'),
 ('leem met weinig grind','leem, zwak grindhoudend'),
 ('lemig grind','grind, met leem'),
 ('lemige grond','teelaarde, met leem'),
 ('mengeling van grind','grind, ongesorteerd'),
 ('opvulling (wegenwerken)','antropogeen'),
 ('steen','stenen'),
 ('teel aarde','teelaarde'),
 ('tegel','antropogeen'),
 ('volledig slakken','mijnsteen'),
 ('zand + grind','zan, grindhoudend'),
 ('zandig fijn tot dominant grof grind met stenen','grind, ongesorteerd, met zand, met stenen'),
 ('zandig fijn tot dominant medium grind','grind, ongesorteerd, met zand'),
 ('zandig fijn tot grof grind','grind, ongesorteerd, met zand'),
 ]

lithkey1 = [k.split(',')[0].lower() for k in lith['Lithology Description']]

res = []
for s in  lithkey1:
    for r in replacements:
        if s==r[0]:
            s=r[1]
    res.append(s)
res = set(res)

#%%
#%%
lithkey2=[]
for line in lith['Lithology Description']:
    try:
        s = line.split(',')[1].lower().strip()
    except:
        s = ''
    lithkey2.append(s)
#lithkey2 = set(lithkey2)      

replace2 = [
 ("afgeleid van foto's",''),
 ('dominant fijn','fijn'),
 ('dominant fijn tot grof','matig fijn'),
 ('dominant fijn tot grof. grijs tot witte kleur. met organisch materiaal (zwart spoelwater) zoals stukjes hout',
  'ongesorteerd, grijswit, met humus, met hout'),
 ('dominant matig grof tot grof','ongesorteerd, matig grof'),
 ('fijn tot dominant grof','ongesorteerd, grof'),
 ('fijn tot dominant matig grof','ongesorteerd, matig grof'),
 ('fijn tot grof','matig grof'),
 ('fijn tot matig grof','matig fijn'),
 ('fijn tot matig grof. zand is grijs tot wit.','matig fijn, grijswit'),
 ('fijn tot medium','matig fijn'),
 ('fijn tot weinig grof. rijk aan glauconiet (opmerkelijk sterkere groene kleur t.o.v. eerdere boringen).',
      'medium, glauconiethoudend'),
 ('fijn tot zeer grof','ongesorteerd'),
 ('fijn. rijk aan glauconiet (opmerkelijk sterkere groene kleur t.o.v. eerdere boringen).',
      'fijn, glauconiethoudend'),
 ('fjin tot dominant grof','ongesorteerd'),
 ('grijze stijve klei met matig grof grind','met klei, met grind'),
 ('grindige bijmenging','met grind'),
 ('grof grindig','grof, met grind'),
 ('grof zandig','grof', 'met zand'),
 ('grof. gesorteerd.','grof, gesorteerd'),
 ('grote stenen','met stenen'),
 ('kleiig (kan je worstjes mee draaien)','met klei'),
 ('lemig','met leem'),
 ('matig fijn tot grof','ongesorteerd'),
 ('matig fijn tot matig grof','ongesorteerd'),
 ('matig grindig','matig grindhoudend'),
 ('matig grof tot grof','grof'),
 ('matig humeus','matig humushoudend'),
 ('matig kleiig','matig kleihoudend'),
 ('matig siltig','matig silthoudend'),
 ('matig tot zeer fijn','fijn'),
 ('matig tot zeer grof','grof'),
 ('matig tot zeer grof. kleilenzen aanwezig in grind als brokken','grof, met kleilenzen'),
 ('matig tot zeer grof. kleilenzen aanwezig in grind als brokken (grind meestal 3 cm','grof, met kleilenzen',
        'grof, met keilenzen'),
 ('matig vast',''),
 ('matig zandig','matig zandhoudend'),
 ('met fijn tot grof grind','met grind'),
 ('met fijn tot grof grind en met zand. ongesorteerd.','met grind'),
 ('met fijn tot grof grind en weinig tot sporadisch zand.','met grind, zwak zandhoudend'),
 ('met fijn tot grof grind en weinig zand. klei is vettig en zwaar en grijs-bruine kleur.',
      'grijsbruin, met grind, zwak zandhoudend'),
 ('met fijn tot grof grind. klei is vettig en lichtbruin.','lichtbruin, met grind'),
 ('met fijn tot grof grind. klei is vettig en zwaar','met grind'),
 ('met fijn tot grof grind. sporadisch stenen. leem is lichtbruin.','lichtbruin, met grind'),
 ('met fijn tot matig grof grind. leem is beige-bruin.','beigebruin, met grind'),
 ('met fijn tot matig grof grind. leem komt voor als brokken','met grind'),
 ('met humeus materiaal','humushoudend'),
 ('met klei en met weinig zandfractie. sporadisch fijn tot matig grof grind.',
      'met klei, zwak zandhoudend, zwak grindhoudend'),
 ('met klei en sporadisch zand','met klei, zwak zandhoudend'),
 ('met klei en sporadisch zand. leem is bruin.','bruin, met klei'),
 ('met leem (en met glauconietkorrels?)','met leem, glauconiethoudend'),
 ('met silt en met fijn grind.','met silt, grindhoudend'),
 ('met silt en met zand (donkerbruin)', 'donkerbruin, met silt, met zand'),
 ('met silt en zand','met silt, met zand'),
 ('met zand en grind. zand is fijn tot matig grof','met zand, met grind'),
 ('met zand en met humeus materiaal. donkerbruin.','donkerbruin, met zand, met humus'),
 ('met zand en met humeus materiaal. grijs to beige bruin.','beigebruin, met zand, met humus'),
 ('met zand. beige tot bruin.','lichtbruin, met zand'),
 ('mijnsteen','met mijnsteen'),
 ('plastisch en zwaar',''),
 ('siltige bijmenging','met silt'),
 ('slap',''),
 ('sporadisch grind','zwak grindhoudend'),
 ('sporadisch grind en zand','zwak grindhoudend, zwak zandhoudend'),
 ('sporadisch zand en sporadisch grind','zwak zandhoudend, zwak grindhoudend'),
 ('sterk grindig','sterk grindhoudend'),
 ('sterk kleiig','sterk kleihoudend'),
 ('sterk kleiïg','sterk kleihoudend'),
 ('sterk siltig','sterk silthoudend'),
 ('sterk zandig','sterk zandhoudend'),
 ('sterk zandig (sterk zandige klei)','sterk zandhoudend, kleihoudend'),
 ('sterk zandig met mogelijks twee kleilensjes','sterk zandhoudend, kleilenzen'),
 ('vast',''),
 ('weinig fijn grind. klei is vettig en lichtbruin.','lichtbruin, met fijn grind'),
 ('weinig fijn tot grof grind','met grind'),
 ('weinig kleiig','zwak kleihoudend'),
 ('weinig silt en weinig zand','zwak silthoudend', 'zwak zandhoudend'),
 ('weinig silt/klei','zwak silthoudend, zwak kleihoudend'),
 ('weinig tot sporadisch zand en sporadisch fijn grind. leem is bruin.',
  'bruin, zwak zandhoudend, zwak grindhoudend'),
 ('weinig zand','zwak zandhoudend'),
 ('weinig zand. lichtbruin','lichtbruin, zwak zandhoudend'),
 ('zandig','met zand'),
 ('zandige bijmenging','met zand'),
 ('zeer fijn tot matig fijn','fijn'),
 ('zeer grof (<355µm) met fijn tot zeer grof grind','zeer grof'),
 ('zeer grof tot uiterst grof (355 à 500 µm)','zeer grof'),
 ('zwak grindig','zwak grindhoudend'),
 ('zwak grindige bijmenging','zwak grindhoudend'),
 ('zwak siltig','zwak silthoudend'),
 ('zwak zandig','zwak zandhoudend'),
 ]


res2 = []
for s in  lithkey2:
    for r in replace2:
        if s==r[0]:
            s=r[1]
    res2.append(s)
res2 = set(res2)


#%%

lithkey3=[]
for line in lith['Lithology Description']:
    try:
        s = line.split(',')[2].lower().strip()
    except:
        s = ''
    lithkey3.append(s)
lithkey3 = set(lithkey3)      


replace2=[
 'beige met groenige schijn en zwarte korrels (glauconiet)','beige, glauconiethoudend'
 'beperkte hoeveelheid dus misschien enkel kleibrokken.',''
 'bestaat uit kwarts. wit/beige.','witbeige'
 'brokken leem','met leembrokken'
 'brokken roest','met roestbrokken'
 'brokken zand','met zandbrokken'
 'bruin groen geel','groengeel'
 'bruin.','bruin'
 'dominant beige schijn','beige'
 'dominant beige schijn (subtiel groen) en zwarte korrels (glauconiet)','beige, glauconiethoudend'
 'dominant bruin met groenige schijn en zwarte korrels (glauconiet)','bruin, glauconiethoudend'
 'dominant fijn','fijn'
 'dominant groene schijn','groen, glauconiethoudend'
 'dominant groene schijn en zwarte korrels (glauconiet)','groen, glauconiethoudend'
 'dominant groenige schijn en zwarte korrels (glauconiet)','groen, glauconiethoudend'
 'dominant matig grof','matig grof'
 'dominant medium','medium'
 'donkergrijs groen (<355µm)','donkergrijs'
 ('donkergrijs groen (<355µm). van 8.3 tot 8.6 m sporadish zeer grof grind (max 3cm). grind komt voor als platte korrels.',
      'donkergrijs')
 'donkergroen geel (> 250 µm)','donkergroen'
 'geel tot bruin.','geelbruin'
 'glauconiet','glauconiethoudend'
 'grijs to beige','grijsbeige'
 'grijs. vochtig.','grijs'
 'grind is fijn tot grof. spoelwater beduidend grijzer.','matig grof'
 'grindige bijmenging','grindhoudend'
 'groen-grijs','groengrijs'
 'groengeel (< 180µm)','fijn, groengeel'
 'groengeel (>125µm <180µm)','fijn, groengeel'
 'groengeel (>180µm <250µm)','matig grof, groengeel'
 'groengrijs (vanaf 10.5 bruingrijs) (180µm)','groengrijs'
 'grote steen',''
 'humeus materiaal','humushoudend'
 'klei is vettig en zwaar (groen/bruin).',''
 'kooltjes',''
 'laagjes grind','met laagjes grind'
 'laagjes klei','met laagjes klei'
 'laagjes leem','met laagjes leem'
 'laagjes zand','met laagjes zand'
 'leem','met leem'
 'lenzen klei','met kleilenzen'
 'lichtbruin (>180µm <250µm)','matig grof, lichtbruin'
 'lichtbruin geel (<125µm)','fijn, lightbruin'
 'lichtgrijs tot wit','lichtgrijs'
 'matig kleiïg','zwak kleihoudend'
 'matig tot sterk lemig.','leemhoudend'
 'matig tot sterk lemig. leembrokken tussen grind','leemhoudend'
 'matig tot sterk zandig. siltfractie aanwezig','zandhoudend, zwak silthoudend'
 'matig zandig.','matig zandhoudend'
 'matig zandig.  zand is zeer grof (>355µm)','matig zandhoudend'
 'matig zandig. zand is matig tot zeer grof (250 - 355 µm)','matig zandhoudend'
 'matig zandig. zand is uiterst grof (>710 µm).','matig zandhoudend'
 'matig zandig. zand is zeer tot uiterst grof (>355µm)','matig zandhoudend'
 'max 10 cm = steen)',''
 'met fijn grind en sporadisch stenen. overgangslaag naar breda?','met grind, zwak steenhoudend'
 'met fijn tot grof grind en met glauconietkorrels. leem is bruin.','bruin, met grind, glauconiethoudend'
 'met fijn tot grof grind en sporadisch stenen.','grindhoudend, zwak steenhoudend'
 'met fijn tot matig grof grind','medium'
 'met fijn tot matig grof grind.','medium'
 'met glauconietkorrels. leem is lichtbruin.','lichtbruin, glauconiethoudend'
 'met grof zand.','met grof zan'
 'met humeus materiaal','humushoudend'
 'met humeus materiaal en weinig leem. donkerbruin.','donkerbruin, humushoudend, zwak leemhoudend'
 'met humeus materiaal. droog.','humushoudend'
 'met humeus materiaal. vochtig.','humushoudend'
 'met leem en humeus materiaal','leemhoudend, humushoudend'
 'met leem en met zand','leemhoudend, zandhoudend'
 'met leem en sporadisch humeus materiaal','leemhoudend, zwak humushoudend'
 'met leem. droog.','leemhoudend'
 'met sporadisch fijn zand en fijn tot grof grind. spoelwater beduidend grijzer.','zwak grindhoudend'
 'met sporadisch stenen','zwak steenhoudend'
 'met stenen','steenhoudend'
 'met stenen en lemig. ongesorteerd. leem is zandig','steenhoudend, leemhoudend'
 'met stenen en sporadisch zand. ongesorteerd.','steenhoudend, zwak zandhoudend'
 'met stenen en weinig zand','steenhoudend, zwak zandhoudend'
 'met stenen en weinig zand.','steenhoudend, zwak zandhoudend'
 'met stenen en weinig zand. spoelwater roestige kleur','steenhoudend, zwak zandhoudend'
 'met stenen. ongesorteerd.','steenhoudend'
 'met weinig zand','zwak zandhoudend'
 'met zand','zanhoudend'
 'met zand (geel/bruin). sporadisch stenen.','geelbruin, zwak steenhoudend'
 'met zand (grijs). sporadisch stenen.','grijs, zandhoudend, zwak steenhoudend'
 ('met zand en met gerolde leem brokken (zeker geen duidelijke leemlaag). grind is ongesorteerd.',
 'zandhoudend, met leembrokken'),
 'met zand en met stenen. ongesorteerd.','ongesorteerd, met zand, steenhoudend'
 'met zand en sporadisch stenen','met zand, zwak steenhoudend'
 'met zand en sporadisch stenen en fijntjes. ongesorteerd.','met zand, zwak steenhoudend'
 'met zand en sporadisch stenen. grind is bruin.','bruin, met zand, zwak steenhoudend'
 'met zand en sporadisch stenen. grind is grijs tot bruin gekleurd.','grijsbruin, met zand, zwak steenhoudend'
 'met zand en stenen. ongesorteerd.','met zand, met stenen'
 'met zand. grind is roodbruin.','roodbruin, met zand'
 ('met zwarte korrels (glauconiet) en sporadisch humeus materiaal (bruinkool?).',
  'zwak humushoudend, glauconiethoudend'),
 'resten hout','met hout'
 'riet',''
 'roestig','roodbruin'
 'siltig','silthoudend'
 'sporadisch fijn tot grof grind','zwak grindhoudend'
 ('sporadisch fijn tot grof grind (kwarts) en sporadisch schelpfragmentjes van millimeter grootte. lichtgroene schijn. moeilijk penetreerbaar met puls. weinig doorlatend.',
     'zwak grindhoudend, met schelpfragmenten'),
 'sporadisch gravel','zwak grindhoudend'
 'sporadisch grind.','zwak grindhoudend'
 'sporadisch grind. met glauconietkorrels. zand is lichtbruin.','lichtbruin, zwak grindhoudend, glauconiethoudend'
 'sporadisch stenen','zwak steenhoudend'
 'sporadisch stenen. klei is vettig en lichtbruin.','lichbruin, zwak steenhoudend'
 'sporadisch stenen. spoelwater wordt bruin/beige','bruinbeige, zwak steenhoudend'
 'sporadisch zand','zwak zandhoudend'
 ('sporadisch zwarte glauconietkorrels. klei komt sporadisch voor als vettige kleibrokken (kleilenzen?).',
     'zwak kleihoudend, met kleibrokken, glauconiethoudend'),
 'sporen roest','zwak roesthoudend'
 'sporen veen','zwak veenhoudend'
 'stabilisatielaag',''
 'stabilisatiepuin',''
 'sterk grindig','sterk grindhoudend'
 'sterk humeus','sterk humuhoudend'
 'sterk kleiïg','sterk kleihoudend'
 'sterk lemig. leembrokken in grind.',
 'sterk siltig','sterk silthoudend'
 'sterk stenig','sterk steenhoudend'
 'sterk zandig','sterk zandhoudend'
 'uiterst roesthoudend',
 'uiterst zandig','uiterst zandhoudend'
 'veel glimmer','glimmerhoudend'
 'vochtig. aanvulling',''
 'weinig fijn grind. zand is grijs to wit.','grijswit, zwak grindhoudend'
 'weinig fijn tot grof grind. zand is geel tot groen.','geelgroen, zwak grindhoudend'
 'weinig fijn tot matig grof grind. licht gesorteerd.','grindhoudend',
 'weinig grind','zwak grindhoudend'
 'weinig leem','zwak leemhoudend'
 'weinig zand','zwak zandhoudend'
 'weinig zand.','zwak zandhoudend'
 'weinig zand. zeer permeabel.','zwak zandhoudend'
 'weinig zandig','zwak zandhoudend'
 'wortels','wortelhoudend
 'zandig','zandhoudend'
 'zwak grindig',
 'zwak humeus','zwak humushoudend'
 'zwak lemig. ongesorteerd.','zwak leemhoudend'
 'zwak siltig bruin','bruin, zwak silthoudend'
 'zwak stenig','zwak steenhoudend'
 'zwak zandig','zwak zandhoudend'
 'zwak zandig.','zwak zandhoudend'
 ('zwak zandig. op 3.5 tot 4.5m komen leembrokken voor in het grind. 1 brok is 8 cm herwerkt materiaal met fijn tot matig grof grind in.',
   'zwak zandhoudend, met leembrokken')
 'zwak zandig. zand is zeer tot uiterst grof (>355µm)','zwak zandhoudend'
]
    
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


