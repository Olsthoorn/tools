"""Reading data files of the Royal Dutch Meteorological Institute KNMI

KNMI provides data files for all of its rain gauge and meteo stations for free, which are
downloadable either by hand or by script. See how-to here:

http://www.knmi.nl/kennis-en-datacentrum/achtergrond/data-ophalen-vanuit-een-script

Dowloading is done from:

http://www.knmi.nl/nederland-nu/klimatologie/daggegevens

This is done using the wget utility (which may be installed with homebrew)

# general way to request the data:
wget -O infile --post-data="variable=value&variable=value&...." "http://projects.knmi.nl/klimatologie/daggegevens/getdata_dag.cgi"


# for example if you want data for stations 235, 280 and 260, for variables VICL and PRCB
for a period defined by begin year, month day and end year month day hten the argument of
 the --post_data should be (note the double quotes):
--post-data="stns=235:280:260&vars=VICL:PRCP&byear=1970&bmonth=1&bday=1&eyear=2009&emonth=8&eday=18"

vars
List of desired variablesin arbitrary order, see selection page, separated by ':', for example 'TG:TN:EV24'.
Some convenience acronymes are defined to select related parameter groups

WIND = DDVEC:FG:FHX:FHX:FX wind
TEMP = TG:TN:TX:T10N temperatuur
SUNR = SQ:SP:Q Zonneschijnduur en globale straling
PRCP = DR:RH:EV24 neerslag en potentiële verdamping
PRES = PG:PGX:PGN druk op zeeniveau
VICL = VVN:VVX:NG zicht en bewolking
MSTR = UG:UX:UN luchtvochtigheid
ALL  = all variables
Default is ALL.

The header l line is in lines 1 through NSTN+NVAR+11 and starts with #
Data are in lines NSTN+NVAR+12 and beyond

So just for Gilze-Rijen, station ..

wget -O "gilze-rijen.dat" --post_data="stns=235&vars=PRCP&start=19900101&end=20101231" "http://projects.knmi.nl/klimatologie/daggegevens/getdata_dag.cgi"
"""
import requests
from pprint import pprint
import pandas as pd
import numpy as np
import logging
from coords import wgs2rd
import shelve
import matplotlib.pyplot as plt
import collections

import pdb

logger=logging.getLogger()

KNMI_URL = {'w' : "http://projects.knmi.nl/klimatologie/daggegevens/getdata_dag.cgi",
            'r' : "http://projects.knmi.nl/klimatologie/monv/reeksen/getdata_rr.cgi",
            'h' : "http://projects.knmi.nl/klimatologie/uurgegevens/getdata_uur.cgi"}

def date_parser(data, col='YYYYMMDD'):
    return [pd.datetime(d[:4] + '-' + d[4:6] + '-' + d[6:8])
                     for d in data[col]]



def parseKNMI(knmifname, fields=['RH', 'EV24'], to_mpd=True):
    '''return pd.DataFrame form knmi weather statio with given fields

    The files can be downloaed from (20190106)
    https://www.knmi.nl/nederland-nu/klimatologie/daggegevens
    You may want to check if this URL is still valid.

    parameters
    ----------
    knmifname: str
        name of knmi file for one of their weather stations.
        These names are usually calloed etmgeg_???.txt where
        ??? is the number of the weather station.
    fields: list of str
        list of the desired fields.

    The field 'YYYYMMDD' will be converted to timestamps and used
    as the index.
    Lines with Nan in any of the fields are dropped.

    @TO 20190106

    '''
    with open(knmifname, 'r') as f:
        skiprows = -1
        while True:
            skiprows += 1
            s = f.readline()
            if s.startswith('# STN'):
                header = s.replace(',',' ').split()[1:]
                break

        fields.insert(0, 'YYYYMMDD')
        usecols = []
        for fld in fields:
            usecols.append(header.index(fld))
        dtype = {1: str}

        #skiprows = None
        data = pd.read_csv(f, header=None, skip_blank_lines=False,
                           skiprows=skiprows, skipinitialspace=True,
                           usecols=usecols, dtype=dtype
                           )
        #                           date_parser=dateParser)
        data.columns = fields
        data.index = pd.to_datetime(
            ['-'.join([d[:4], d[4:6], d[6:]]) for d in data['YYYYMMDD']])

        # drop NaN's and the unused date column
        # note that we have to assign
        data = (data.dropna()).drop(columns='YYYYMMDD')

        if to_mpd:
            for fld in fields:
                if fld in ['RH', 'EV24']:
                    b = data.loc[:, fld]  < 0
                    data.loc[b, fld] = 0.25
                    data[fld] /= 10000.0 # original is in 0.1 mm units
    return data


def getWstations(loc="http://climexp.knmi.nl/KNMIData/list_dx.txt"):
    """return dict with meta data of KNMI stations.

    The URL was valid in March 2017.

    """
    r = requests.get(loc)
    #data = r.content.decode('utf-8').split('\n')
    data = r.text.split('\n')

    stations=dict()

    for i, line in enumerate(data[1:]):
        j = i%5
        if j==0:
            pass # '===='
        if j==1: # name and country we'll get it from station code
            pass
        elif j==2: # coordinates
            coords, meta = line.split(' <a href=')
            _, coords = coords.split(':')
            coords = coords.strip().replace(',','').replace('  ',' ')\
                .replace('  ',' ').replace('  ',' ').split(' ')
            meta = meta.split(' ')[0].replace('"', '')

            WGS = [float(c[:-1]) for c in coords]
            RD = WGS[:]
            RD[0], RD[1] = wgs2rd(WGS[1], WGS[0])
        elif j==3:
            code, name = line.split(':')[1].strip().split(' ')
        elif j==4:
            period = line.split(' ')[-1]
            stations[name] = {'code': code, 'lat': WGS[0], 'lon': WGS[1],
                    'x': RD[0], 'y': RD[1], 'z': RD[2],
                    'period': period, 'meta': meta}
    return stations


def getPstations(loc="http://climexp.knmi.nl/KNMIData/list_dx.txt"):
    """rReturn dict with meta data of KNMI precipitation stations.

    The default URL was valid in March 2017.

    TODO:
    ----
        get their coordinates from somewhere, pref. the KNMI site
        which was not possible on march 30

    """
    with open(loc, 'r') as rdr:
        lines = rdr.readlines()

    station=dict()
    for line in lines:
        try:
            L = line.replace('\t', '|', 2).replace(' t/m ','|',1).split('|')
            station[L[0]] = {'code': L[1], 'period': (L[2], L[3])}
        except:
            break
    return station


def getWdata(station=None, start=None,
                                        end=None, vars=None):
    '''Requests and saves KNMI data for specific KNMI weather stations.

    parameters:
    -----------
    station : int, e.d. 260 for De Bilt
        list of legal KNMI station code. See `getStations` to get the them.
    sDate: str
        start date like 'yyyyymmdd'
    eDate: str
        end date like 'yyyyymmdd'
    vars: [var, var, var ...], strings, must be legal variable names
        as defined in header of KNMI data files.
        Use `getVars()` to retrieve them.

        Acceptable vars and their group acronyms:
        WIND = DDVEC:FG:FHX:FHX:FX wind
        TEMP = TG:TN:TX:T10N temperatuur
        SUNR = SQ:SP:Q Zonneschijnduur en globale straling
        PRCP = DR:RH:EV24 neerslag en potentiële verdamping
        PRES = PG:PGX:PGN druk op zeeniveau
        VICL = VVN:VVX:NG zicht en bewolking
        MSTR = UG:UX:UN luchtvochtigheid
        ALL alle variabelen
        Default is ALL.

    Returns
    -------
        stationData as a pd.DataFrame

        stationData will also be shelved to file  'etmgeg' + str(station) + '.db'
    '''

    URL = 'http://projects.knmi.nl/klimatologie/daggegevens/getdata_dag.cgi'

    payload = {'start' : start, 'end': end, 'stns': [station], 'vars': vars}

    # post the request
    resp = requests.post(URL, data=payload)

    # get the data
    data = resp.text.split('\r\n')

    # find the header line
    i, got2 = 0, False
    for dline in data:
        if dline.startswith('# STN'):
            if got2:
                i += 1 # skip empty line
                break
            got2 = True
        i += 1

    # Get the headers for the data frame we make in the end
    columns = [c.strip() for c in dline.split(',')]
    columns[0] = columns[0].split(' ')[1]

    # Collect the data
    Data = []
    for d in data[i+1:]:
        Data.append([v.strip() for v in d.split(',')])

    stationData = pd.DataFrame(Data[:-1], columns=columns)

    fname = './etmgeg_' + str(station)
    with shelve.open(fname,  flag='n') as s:
        s[str(station)] = stationData
        print('Data for station {} saved to file {}'.format(station, fname))

    return stationData


def getPdata(station=260, start='19900101', end='20161231'):
    '''Requests and shelves precipitation from a KNMI regenstation.

    Saving will be on file 'neersl' + str(station) + '.db'

    parameters:
    -----------
    sDate: str
        start date like 'yyyyymmdd'
    eDate: str
        end date like 'yyyyymmdd'

    Returns
    -------
        stationData as a pd.DataFrame

        STN,YYYYMMDD, RD, SX,
        458,19731101, 0, 0, Aalsmeer
        458,19731102, 0, 0, Aalsmeer
        458,19731103, 0, 0, Aalsmeer
    '''

    URL = 'http://projects.knmi.nl/klimatologie/monv/reeksen/getdata_rr.cgi'

    payload = {'start' : start, 'end': end, 'stns': [station]}

    # post the request
    resp = requests.post(URL, data=payload)

    # get the data
    data = resp.text.split('\r\n')

    # find the header line
    i, got2 = 0, False
    for dline in data:
        if dline.startswith('STN'):
            if got2:
                i += 1 # skip empty line
                break
            got2 = True
        i += 1

    # Get the headers for the data frame we make in the end
    columns = [c.strip() for c in (dline + 'NAME').split(',')]

    # Collect the data
    Data = []
    for d in data[i+1:]:
        Data.append([v.strip() for v in d.split(',')])

    stationData = pd.DataFrame(Data[:-1], columns=columns)

    fname = './neersl_' + str(station)
    with shelve.open(fname,  flag='n') as s:
        s[str(station)] = stationData
        print('Precipidation data for station {} saved to file {}'
                                              .format(station, fname))

    return stationData


def data2iso(df):
    '''Return pd.DataFrame with \ 'YYYYMMDD' as index and precip and evap in m/d'

    Returns nothing and does nothing if 'YYYYMMDD' is still a column header.

    parameters
    ----------
        df : pd.DataFrame
            Meteo data as obtained with getWdata ()and getPdata()
    returns
    -------
        df : pd.DataFrame
            with YYYYMMDD as index and prec and evap in m/d.

    '''

    if 'YYYYMMDD' in df.columns:
        df = df.copy()
        df.index = df['YYYYMMDD']
        df = df.drop(columns=['YYYYMMDD'])

        for col in df.columns:
            if col in ['RH', 'EV24', 'RD']:
                tseries = df[col].copy().astype(float)
                tseries[tseries < 0] = 0.5
                df[col] = tseries / 10000  # from 0.1 mm/d to m/d
    else:
        print('Nothing to do the dataframe was already converted.')

    return df

def index2tstamp(df):
    '''Return index of df as pd.Timestamps

    parameters
    ----------
        df : pd.DataFrame
            data with 'YYYYMMDD' as (integer) index
    returns
    -------
        a list that can be used as a new index

    '''
    index = [str(s) for s in list(df.index)]
    ts = [pd.Timestamp('{}-{}-{} 08:00'.format(ds[0:4], ds[4:6], ds[6:8]))
                                                        for ds in index]
    return ts

#https://www.knmi.nl/nederland-nu/klimatologie/daggegevens

if __name__ == '__main__':

    P = parseKNMI('./data/uurgeg_240_2001-2010.txt', fields=['HH', 'P'], to_mpd=False)

    exit()

    data = parseKNMI('./data/etmgeg_260.txt', fields=['RH', 'EV24'])

    if False:
        station = getWstations(loc='KNMI_stations.txt')
        pprint(station)

    # Get overview of all stations
    if False:
        loc = "http://www.climexp.knmi.nl/KNMIData/list_dx.txt"
        #loc = "./data/KNMI_stations.txt"
        stations = getPstations(loc=loc)

    # Get data for a weather station
    if False:
        Wdf = getWdata(station=380, start='20180501',
                                        end='20180720', vars=None) # ['RH', 'EV24'] )

        W380 = data2iso(Wdf)
        W380.index = index2tstamp(W380)
        W380.plot()
        ax = plt.gca()
        ax.set_title('KNMI-station Maastricht (380)')
        ax.set_ylabel('m/d')
        ax.grid(True)


    # Get data for a precipitation station
    if True:
        name, station = 'De Bilt', 260
        #name, station = 'Buchten', 974
        Pdf = getPdata(station=station, start='20500101', end='20181231')
        Pdf = data2iso(Pdf)
        Pdf.index = index2tstamp(Pdf)
        Pdf.plot()
        ax = plt.gca()
        ax.set_title('KNMI-station {} ({})'.format(name, station))
        ax.set_ylabel('m/d')
        ax.grid(True)
        plt.gca().set_xlim((736824.7859375001, 736896.6411458333))


    # Same thing with kwargs
    if False:
        kwargs = {'start': '19900101', 'end' : '20161231', 'vars' : ['RH', 'EV24']}
        AWSpdf = getWdata(station=260, **kwargs)





