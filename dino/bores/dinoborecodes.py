grey1 = (0.2, 0.2, 0.2)
grey2 = (0.4, 0.4, 0.4)
grey3 = (0.6, 0.6, 0.6)
grey4 = (0.8, 0.8, 0.8)
grey5 = (0.9, 0.9, 0.9)

lithTags = [
    'lithology',
    'colorIntensity',
    'colorMain',
    'colorSecondary',
    'sandSorting',
    'gravelFrac',
    'gravelAdmix',
    'organicAdmix',
    'humusAdmix',
    'clayAdmix',
    'siltAdmix',
    'leemAdmix',
    'veenAdmix',
    'sandAdmix',
    'clasticAdmix',
    'coarseMatAdmix',
    'glaucFrac',
    'shellFrac',
    'authigenMin',
    'lithoLayerBoundary',
    'carbonateFracNEN5104',
    'plantType',
    'plantFrac',
    'lithoLayerTrend',
    'subLayerLithology',
    'subLayerThickness',
    'sedimentStructure',
    'lithostrat',
    ]

lithSandTags = [
    'sandMedianClass',
    ]
lithGravelTags = [
    'gravelMedianClass',
    ]

lithostrat = \
{'AA'  : 'Antropogeen',
 'AAOM': 'Antropogeen, omgewerkte grond',
 'AAOP': 'Antropogeen, opgebrachte grond',
 'BE': 'Formatie van Beegden',
 'BECA': 'Laagpakket van Caberg',
 'BEGR': 'Laagpakket van Gronsveld',
 'BEOM': 'Laagpakket van Oost Maarland',
 'BERT': 'Laagpakket van Rothem',
 'BR': 'Formatie van Breda',
 'BRHE': 'Laagpakket van Heksenberg',
 'BRKA': 'Laagpakket van Kakert',
 'BRVR': 'Laagpakket van Vrijherenberg',
 'BX': 'Formatie van Boxtel',
 'BXSC': 'Laagpakket van Schimmert',
 'BXSI': 'Laagpakket van Singraven',
 'BXWI': 'Laagpakket van Wierden',
 'HO': 'Formatie van Houthem',
 'IE': 'Formatie van Inden',
 'KI': 'Kiezelooliet Formatie',
 'LA': 'Formatie van Landen',
 'MT': 'Formatie van Maastricht',
 'NN': 'Onbekend',
 'RU': 'Formatie van Rupel',
 'RUBIBE': 'Laag van Berg',
 'RUBIKS': 'Laag van Kleine Spouwen',
 'RUBO': 'Laagpakket van Boom',
 'SY': 'Formatie van Stramproy',
 'TOGO': 'Laagpakket van Goudsberg',
 'TOKL': 'Laagpakket van Klimmen',
 'VI': 'Ville Formatie'}

lithostratColor = \
{'AA'  : 'gray',
 'AAOM': 'gray',
 'AAOP': 'lightgray',
 'BE':   'pink',
 'BECA': 'deeppink',
 'BEGR': 'fuchsia',
 'BEOM': 'violet',
 'BERT': 'darkviolet',
 'BR':   'green',
 'BRHE': 'rebeccapurple',
 'BRKA': 'stateblue',
 'BRVR': 'darkblue',
 'BX':   'peachpuff',
 'BXSC': 'seashell',
 'BXSI': 'coral',
 'BXWI': 'mistyrose',
 'HO':   'brown',
 'IE':   'maroon',
 'KI':   'peru',
 'LA':   'chocolate',
 'MT':   'lightgray',
 'NN':   'white',
 'RU':   'palegoldenrod',
 'RUBIBE': 'dodgerblue',
 'RUBIKS': 'skyblue',
 'RUBO': 'darkred',
 'SY':   'lemonchiffon',
 'TOGO': 'springgreen',
 'TOKL': 'forestgreen',
 'VI':   'thistle'}


lithology = \
{'AA' : 'antropogeen',
 'BRK': 'bruinkool',
 'DET': 'detritus',
 'GCZ': 'glauconietzand',
 'GIP': 'gips',
 'GM':  'geen monster',
 'HO':  'hout',
 'TLA': 'teelaarde',
 'GRA': 'gras',
 'HUM' : 'humus',

 'K': 'klei',
 'L': 'leem',
 'V': 'veen',
 'Z': 'zand',
 'G': 'grind',

 'KAS': 'kalksteen',
 'KLS': 'kleisteen',
 'MI': 'mijnsteen',
 'NBE': 'niet benoemd',
 'OER': 'ijzeroer',
 'PU': 'puin',
 'SLU': 'slurrie',
 'STK': 'steenkool',
 'STN': 'stenen',
 'VUS': 'vuursteen',
 'ZNS': 'zandsteen'}

shellFrac = \
{'SCH0': 'geen schelpen',
 'SCH1': 'spoor schelpen',
 'SCH2': 'weinig schelpen',
 'SCH3': 'veel schelpen',
 'SCHX': 'schelpen'}

organicAdmix = \
{'BRB2': 'weinig bruinkoolbrokken',
 'BRBX': 'met bruinkoolbrokken',
 'DTF2': 'weinig fijne detritus',
 'DTF3': 'veel fijne detritus',
 'DTR1': 'spoor detritus',
 'DTR2': 'weinig detritus',
 'DTR3': 'veel detritus',
 'HOK1': 'spoor houtskool',
 'HOK2': 'weinig houtskool',
 'HOK3': 'veel houtskool',
 'LIG1': 'spoor ligniet',
 'LIG3': 'veel ligniet',
 'LIGX': 'ligniet',
 'LEB2': 'weinig leembrokken',
 'LEBX': 'leembrokken',
 'ORM3': 'veel organisch materiaal',
 'SGSX': 'steenkoolgruis',
 'STB1': 'spoor steenkoolbrokjes',
 'STB3': 'veel steenkoolbrokjes',
 'STBX': 'met steenkoolbrokjes',
 'VEB1': 'spoor veenbrokjes',
 'VEB2': 'weinig veenbrokjes',
 'VEB3': 'veel veenbrokjes',
 'VEBX': 'veenbrokjes'}

humusAdmix = \
{'H1': 'zwak humeus',
 'H2': 'matig humeus',
 'H3': 'sterk humeus',
 'HX': 'humeus'}

humusAdmixColor = \
{'H1': grey1,
 'H2': grey2,
 'H3': grey3,
 'HX': grey2}

clayAdmix = \
{'K1': 'zwak kleiig',
 'K2': 'matig kleiig',
 'K3': 'sterk kleiig',
 'KM': 'mineraalarm',
 'KX': 'kleiig '}

clayAdmixColor = \
{'K1': grey1,
 'K3': grey2,
 'KM': grey3,
 'KX': grey2}

leemAdmix = \
{'L1': 'zwak lemig',
 'L2': 'matig lemig',
 'L3': 'sterk lemig',
 'L4': 'uiterst lemig',
 'LX': 'lemig'}

siltAdmix = \
{'S1': 'zwak siltig',
 'S2': 'matig siltig',
 'S3': 'sterk siltig',
 'S4': 'uiterst siltig',
 'SX': 'siltig'}

veenAdmix = \
{'V1': 'zwak venig',
 'V2': 'matig venig',
 'V3': 'sterk venig',
 'V4': 'uiterst venig',
 'VX': 'venig'}


siltAdmixColor = \
{'S1': grey1,
 'S2': grey2,
 'S3': grey3,
 'S4': grey4,
 'SX': grey3}


sandAdmix = \
{'Z1': 'zwak zandig',
 'Z2': 'matig zandig',
 'Z3': 'sterk zandig',
 'Z4': 'uiterst zandig',
 'ZX': 'zandig'}

sandAdmixColor = \
{'Z1': grey1,
 'Z2': grey2,
 'Z3': grey3,
 'Z4': grey4,
 'ZX': grey3}


gravelAdmix = \
{'G1': 'zwak grindig',
 'G2': 'matig grindig',
 'G3': 'sterk grindig',
 'GX': 'grindig'}

gravelAdmixColor = \
{'G1': grey1,
 'G2': grey2,
 'G3': grey3,
 'GX': grey2}


clasticAdmix = \
{'GKO1': 'spoor grove korrels',
 'GKO2': 'weinig grove korrels',
 'GRL1': 'spoor granuul',
 'KLB1': 'spoor kleibrokjes',
 'KLB2': 'weinig kleibrokjes',
 'KLB3': 'veel kleibrokjes',
 'KLBX': 'kleibrokjes',
 'LEB1': 'spoor leembrokjes',
 'LEB2': 'weinig leembrokjes',
 'LEB3': 'veel leembrokjes',
 'LEBX': 'leembrokjes'}

coarseMatAdmix = \
{'BKX': 'blokken',
 'PUX': 'met puin',
 'KE1': 'spoor keien',
 'KE2': 'weinig keien',
 'KEX': 'met keien',
 'ST1': 'spoor stenen',
 'ST2': 'weinig stenen',
 'ST3': 'veel stenen',
 'ST4': 'zeer veel stenen',
 'STX': 'met stenen'}

colorIntensity = \
{'DO': 'donker', 'LI': 'licht'}

colorMain = \
{'BL': 'blauw',
 'BE': 'beige',
 'BR': 'bruin',
 'GE': 'geel',
 'GN': 'groen',
 'GR': 'grijs',
 'OL': 'olijf',
 'OR': 'oranje',
 'RO': 'rood',
 'WI': 'wit',
 'ZW': 'zwart'}

colorSecondary = \
{'TBL': 'blauw',
 'TBE': 'beige',
 'TBR': 'bruin',
 'TGE': 'geel',
 'TGN': 'groen',
 'TGR': 'grijs',
 'TOL': 'olijf',
 'TOR': 'oranje',
 'TRO': 'rood',
 'TWI': 'wit',
 'TZW': 'zwart',}

glaucFrac = \
{'GC1': 'spoor glauconiet',
 'GC2': 'weinig glauconiet',
 'GC3': 'veel glauconiet',
 'GC4': 'zeer veel glauconiet',
 'GCX': 'glauconiet'}

lithoLayerBoundary = \
{'BSE': 'basis scherp'}

carbonateFracNEN5104 = \
{'CA1': 'kalkloos', 'CA2': 'kalkarm', 'CA3': 'kalkrijk', 'CAX': 'kalkhoudend'}

plantType = \
{'HOR1': 'spoor houtresten',
 'HOR2': 'weinig houtresten',
 'HOR3': 'veel houtresten',
 'HOU1': 'spoor hout',
 'HOU2': 'weinig hout',
 'HOU3': 'veel hout',
 'HOUX': 'met hout',
 'RIEX': 'met riet',
 'RIR2': 'weinig rietresten',
 'VEP1': 'spoor verspoelde plantenresten',
 'PLA1': 'weinig plantenhoundend',
 'PLA2': 'matig plantenhoudend',
 'PLA3': 'sterk plantendhoudend',
 'WOR1': 'spoor wortelresten',
 'WOR2': 'weinig wortelresten',
 'WOR3': 'veel wortelresten',
 'WOS1': 'spoor wortels',
 'WOS2': 'weinig wortels',
 'WOS3': 'veel wortels',
 'WOSX': 'met wortels'}

sandSorting = \
{'SMA': 'matige spreiding',
 'SMG': 'matig grote spreiding',
 'SMK': 'matig kleine spreiding',
 'STW': 'tweetoppige spreiding',
 'SZG': 'zeer grote spreiding',
 'SZK': 'zeer kleine spreiding'}

sandMedianClass = \
{'ZUFO': 'uiterst fijn (O)',
 'ZZFO': 'zeer fijn (O)',
 'ZMFO': 'matig fijn (O)',
 'ZFC':  'fijne categorie (O)',
 'ZMC':  'midden categorie (O)',
 'ZGC':  'grove  categorie (O)',
 'ZMGO': 'matig grof (O)',
 'ZZGO': 'zeer grof (O)',
 'ZUGO': 'uiterst grof (O)',
 'ZUF':  'uiterst fijn',
 'ZZF':  'zeer fijn',
 'ZMF':  'matig fijn',
 'ZMO':  'zandmediaan onduidelijk',
 'ZMG':  'matig grof',
 'ZZG':  'zeer grof',
 'ZUG':  'uiterst grof',
 'ZGX':  'grof',
 'ZFN':  'fijn',
 }

sandMedianClassWidth = \
{'ZUF':  0,
 'ZUFO': 0,
 'ZZF':  1,
 'ZZFO': 1,
 'ZMF':  2,
 'ZMFO': 2,
 'ZMO':  2,
 'ZFC':  2,
 'ZFN':  2,
 'ZMC':  2,
 'ZMG':  3,
 'ZMGO': 3,
 'ZGC':  4,
 'ZZG':  5,
 'ZZGO': 5,
 'ZUG':  6,
 'ZUGO': 6,
 }

sandMedianClassColor = \
{'ZUF' : 'goldenrod',
 'ZUFO': 'goldenrod',
 'ZZF':  'khaki',
 'ZZFO': 'khaki',
 'ZMF':  'palegoldenrod',
 'ZMFO': 'palegoldenrod',
 'ZMO':  'palegoldenrod',
 'ZFC':  'palegoldenrod',
 'ZMC':  'palegoldenrod',
 'ZMG':  'lightyellow',
 'ZMGO': 'lightyellow',
 'ZGC':  'gold',
 'ZZG':  'gold',
 'ZZGO': 'gold',
 'ZUG':  'yellow',
 'ZUGO': 'yellow',
 }

gravelMedianClass = \
{'GMF': 'matig fijn',
 'GMG': 'matig grof',
 'GZG': 'zeer grof',
 'GUG': 'uiterst grof',
 'GFN': 'fijn',
 'GGR': 'grof',
}

gravelMedianClassWidth = \
{'GFN': 0,
 'GMG': 1,
 'GGR': 2,
 'GZG': 3}

gravelMedianClassColor = \
{'GFN': 'navajowhite',
 'GMG': 'sandybrown',
 'GGR': 'orange',
 'GZG': 'darkorange'}

gravelFrac = \
{'FN1': 'spoor fijn grind',
 'FN2': 'weinig fijn grind',
 'FN3': 'veel fijn grind',
 'FN4': 'zeer veel fijn grind',
 'FN5': 'uiterst veel fijn grind',
 'FNX': 'met fijn grind',
 'MG1': 'spoor matig grof grind',
 'MG2': 'weinig matig grof grind',
 'MG3': 'veel matig grof grind',
 'MG4': 'zeer veel matig grof grind',
 'MG5': 'uiterst veel matig grof grind',
 'MGX': 'met matig grof grind',
 'ZG1': 'spoor zeer grof grind',
 'ZG2': 'weinig zeer grof grind',
 'ZG3': 'veel zeer grof grind',
 'ZG4': 'zeer veel zeer grof grind',
 'ZG5': 'uiterst veel zeer grof grind',
 'ZGX': 'met zeer grof grind'}

authigenMin = \
{'CCR1': 'spoor concreties',
 'CCRX': 'concreties',
 'FEC1': 'spoor ijzerconcreties',
 'FEC2': 'weinig ijzerconcreties',
 'FEC3': 'veel ijzerconcreties',
 'FECX': 'ijzerconcreties',
 'FEH1': 'spoor ijzeroxide huidjes',
 'FEH2': 'weinig ijzeroxide huidjes',
 'FEL1': 'spoor ijzerlagen',
 'FEL2': 'weinig ijzerlagen',
 'FEL3': 'veel ijzerlagen',
 'FEO1': 'spoor ijzeroxide',
 'FEO2': 'weinig ijzeroxide',
 'FEO3': 'veel ijzeroxide',
 'FEOX': 'ijzeroxide',
 'FFC2': 'weinig fosfaatconcreties',
 'FFCX': 'fosfaatconcreties',
 'GLH1': 'spoor glauconiethuidjes',
 'GLH2': 'weinig glauconiethuidjes',
 'GLH3': 'veel glauconiethuidjes',
 'GLHX': 'glauconiethuidjes',
 'GPSX': 'gips',
 'KAC2': 'weinig kalkconcreties',
 'MNC1': 'spoor mangaanconcreties',
 'MNC2': 'weinig mangaanconcreties',
 'MNC3': 'veel mangaanconcreties',
 'MNCX': 'mangaanconcreties',
 'PYR2': 'weinig pyriet',
 'PYRX': 'pyriet',
 'ROV3': 'veel roestvlekken',
 'ROVX': 'roestvlekken',
 'VKZX': 'verkiezeling',
 'ZAV1': 'spoor zandverkitting',
 'ZAV2': 'weinig zandverkitting',
 'ZAV3': 'veel zandverkitting'}


lithoLayerTrend = \
{'BAG': 'aan de basis grof',
 'BAGR': 'aan de basis grindig',
 'BAH': 'aan de basis humeus',
 'BAS': 'aan de basis siltig',
 'BAZ': 'aan de basis zandig',
 'CUA': 'naar boven toe grover',
 'CUM': 'toename korrelgrootte zand naar boven toe',
 'FUA': 'naar boven toe fijner',
 'FUG': 'afname grindpercentage naar boven toe',
 'FUZ': 'afname zandpercentage naar boven toe',
 'TOH': 'aan de top humeus',
 'TOZ': 'aan de top zandig'}

subLayerLithology = \
{'SLG1': 'met spoor grindlagen',
 'SLG2': 'met weinig grindlagen',
 'SLGX': 'met grindlagen',
 'SLK1': 'met spoor kleilagen',
 'SLK2': 'met weinig kleilagen',
 'SLK3': 'met veel kleilagen',
 'SLKX': 'met kleilagen',
 'SLL1': 'met spoor leemlagen',
 'SLL2': 'met weinig leemlagen',
 'SLL3': 'met veel leemlagen',
 'SLLX': 'met leemlagen',
 'SLV2': 'met weinig veenlagen',
 'SLZ1': 'met spoor zandlagen',
 'SLZ2': 'met weinig zandlagen',
 'SLZ3': 'met veel zandlagen',
 'SLZ4': 'met zeer veel zandlagen',
 'SLZX': 'met zandlagen'}

subLayerThickness = \
{'SLDC': 'dun',
 'SLDD': 'dik',
 'SLDM': 'zeer dun',
 'SLDW': 'met wisselende laagdikten'}

plantFrac = \
{'PL0': 'geen plantenresten',
 'PL1': 'spoor plantenresten',
 'PL2': 'weinig plantenresten',
 'PL3': 'veel plantenresten',
 'PLX': 'plantenresten'}

sedimentStructure = \
{'BIO': 'bioturbatie',
 'DWO': 'doorworteling',
 'GE1': 'zwak gelaagd',
 'GE2': 'weinig gelaagd',
 'GE3': 'sterk gelaagd',
 'GEX': 'gelaagd',
 'GMM': 'mm-gelaagdheid',
 'GRG': 'graafgangen',
 'GSC': 'scheve gelaagdheid',
 'GSP': 'spekkoek gelaagdheid',
 'LEG2': 'weinig grindlenzen',
 'LEK3': 'veel kleilenzen',
 'LEKX': 'kleilenzen',
 'LEL1': 'spoor leemlenzen',
 'LEL2': 'weinig leemlenzen',
 'LEL3': 'veel leemlenzen',
 'LELX': 'leemlenzen',
 'LEZ1': 'spoor zandlenzen',
 'LEZ2': 'weinig zandlenzen',
 'LEZX': 'zandlenzen',
 'STBR1': 'spoor bruinkoollagen',
 'STDE1': 'spoor detrituslagen',
 'STDE2': 'weinig detrituslagen',
 'STDE3': 'veel detrituslagen',
 'STGL1': 'spoor grindlagen',
 'STGL2': 'weinig grindlagen',
 'STGLX': 'grindlagen',
 'STKL1': 'spoor kleilagen',
 'STKL2': 'weinig kleilagen',
 'STKLX': 'kleilagen',
 'STLL1': 'spoor leemlagen',
 'STLL2': 'weinig leemlagen',
 'STLL3': 'veel leemlagen',
 'STVL2': 'weinig veenlagen',
 'STZL1': 'spoor zandlagen',
 'STZL2': 'weinig zandlagen',
 'STZLX': 'zandlagen'}
