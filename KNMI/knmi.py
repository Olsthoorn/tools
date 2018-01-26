"""Reading data files of the Royal Dutch Meteorological Institute KNMI

KNMI provides data files for all of its raing gauge and meteo stations for free, which are
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
import urllib
from pprint import pprint
import subprocess
import pandas as pd
import numpy as np
import pdb

KNMI_URL = {'w' : "http://projects.knmi.nl/klimatologie/daggegevens/getdata_dag.cgi",
            'r' : "http://projects.knmi.nl/klimatologie/monv/reeksen/getdata_rr.cgi",
            'h' : "http://projects.knmi.nl/klimatologie/uurgegevens/getdata_uur.cgi"}


def getWeatherStations(loc="http://climexp.knmi.nl/KNMIData/list_dx.txt"):
    """return dict with meta data of KNMI stations.

    The default URL was valid in March 2017.

    TODO:
    ----
        convert LL to RD coordinates: coul be done using gdal !
        which stationa have weather data and which have only rain data?
        which have also hourly data?
        # we could simple check that be means of a request.
        # build this in, then a data base is not required.

    """
    if loc.startswith('http://'):
        with urllib.request.urlopen(loc) as rdr:
            lines = [line.decode("utf-8") for line in rdr.readlines()]
    else:
        with open(loc, 'r') as rdr:
            lines = rdr.readlines()

    station=dict()
    for i in range(2, len(lines), 5):
        try:
            name = " ".join(lines[i].split()[0:-1])
            _, N, S, NAP, *_  = lines[i + 1].split()
            _, _, code,* _    = lines[i + 2].split()
            *_, yearspan = lines[i + 3].split()
            syr, eyr = yearspan.split('-')
            station[name] = {'code': code, 'N': float(N[:-2]),
                   'S': float(S[:-2]), 'NAP': float(NAP[1:-2]),
                   'period': (syr, eyr)}
        except:
            break
    return station

def getPrecStations(loc="KNMI_prec_stations.txt"):
    """return dict with meta data of KNMI precipitation stations.

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


def getKNMIdata(fname, type='r', stations=[260], sDate='19900101', eDate='20161231', vars=['RH']):
    '''Requests KNMI data for specific KNMI stations.

    parameters:
    -----------
    fname = str, file to store the data
    type = str, oneof 'r', 'w', 'h' for rain stations, wheather stations or
        hourly data
    stations = [int, int, ...] list of legal KNMI station codes
        use getStations to get the them
    sDate: str, start date like 'yyyyymmdd'
    eDate: str, end   date like 'yyyyymmdd'
    vars: [var, var, var ...], strings, must be legal variable names
        as defined in header of KNMI data files.
        use getVars() to retrieve them.
    '''

    if not type in ['r', 'w', 'h']:
        raise ValueError("type not in ['r', 'w', 'h']")

    URL = KNMI_URL[type]

    stn=("stns" + (":{}"*len(stations)).format(*stations)).replace(':', '=', 1)
    vrs=("vars" + (":{}"*len(vars)).format(*vars)).replace(':', '=', 1)
    sD="start=" + sDate
    eD="end="   + eDate

    request = "&".join([stn, vrs, sD, eD])

    if False:
        iout = subprocess.call(" ".join(["/usr/local/bin/wget", "-O", fname, '--post-data="{}"'.format(request), URL]))
        print("iout = ", iout)
    print(["/usr/local/bin/wget", "-O", fname, '--post-data="{}"'.format(request), URL])
    print(" ".join(["/usr/local/bin/wget", "-O", fname, '--post-data="{}"'.format(request), URL]))

    vars = "PRCP"

    # TOTO: convert the RH field and to m/d or mm/d.
    return None

def readKNMIdata(fname, vars=['RH', 'EV24']):
    '''reads KNMI daily data from an existing file.

    parameters:
    -----------
    fname = str, file from which the data are read (downloaded from KNMI
    vars: [var, var, var ...], strings, must be legal variable names
        as defined in header of KNMI data files.
        use getVars() to retrieve them.
    '''

    with open(fname, 'r') as rdr:
        data = rdr.readlines()

    hdr_row = []
    for i, line in enumerate(data):
        if line.startswith('# STN'):
            hdr_row.append(i)
            if len(hdr_row)==2:
                break

    df1 = pd.read_csv(fname, skipinitialspace=True,
                      header=hdr_row[0], nrows=hdr_row[1]-hdr_row[0]-6)

    df2 = pd.read_csv(fname, index_col='YYYYMMDD', skipinitialspace=True,
                      header=hdr_row[1], dtype={'YYYYMMDD': str},
                      parse_dates=True)

    # cleanup the data
    # remove where index=NaT
    df2 = df2.drop(df2.index[pd.isnull(df2.index)], axis=0)
    # remove where RH or EV24==NaN
    df2 = df2.drop(df2.index[np.logical_or(pd.isnull(df2.RH), pd.isnull(df2.EV24))], inplace=False)
    L = np.logical_or(pd.isnull(df2.RH), pd.isnull(df2.EV24), pd.isnull(df2.index))
    df2 = df2.drop(df2.index[L], inplace=False)
    df2.loc[df2.index[df2.RH   < 0],   'RH'] = 0.5
    df2.loc[df2.index[df2.EV24 < 0], 'EV24'] = 0.5
    # all to m/d
    df2['RH']    /= 10000. # to m/d
    df2['EV24']  /= 10000. # to m/d
    df2['rch']    = (df2.RH - df2.EV24)  # m/2

    return df1, df2


    return None


def getVars(type='r'):
    '''returns a dict of the variables containedin a KNMI data file.

    parameters:
    -----------
        type str, indicates type of data, onof
            'r' daily rain data (rain station)
            'w' wheater station
            'h'  hourly data
    '''
    try:
        if type.startswith('r'):
            URL = "http://projects.knmi.nl/klimatologie/daggegevens/getdata_dag.cgi"
        elif type.startswith('r'):
            URL = "http://projects.knmi.nl/klimatologie/monv/reeksen/getdata_rr.cgi"
        elif type.startswith('h'):
            URL = "http://projects.knmi.nl/klimatologie/uurgegevens/getdata_uur.cgi"
        else:
            raise ValueError("input argument not one of 'r', 'w', 'h'")
    except:
        raise ValueError("input not a string that startswith one of 'r', 'w', 'h'")



    vars = dict()
    with urllib.request.urlopen(URL) as rdr:
        lines = [line.decode("utf-8") for line in rdr.readlines()]

    for line in lines:
        if '=' in line:
            try:
                line = line.replace('=', '::', 1).replace('#', '', 1).strip().split('::')
                vars[line[0].strip()] = line[-1].strip()
            except:
                break
    return vars


def __main__():
    station = getWeatherStations(loc='KNMI_stations.txt')
    pprint(station)
    #vars = getVars()
    #pprint(vars)


if False: #__name__ == "__main__":
    print("name = ", __name__)
    if False:
        wStations = getWeatherStations(loc='KNMI_stations.txt')
        pprint(wStations)
        pStations = getPrecStations()
        pprint(pStations)
        vars = getVars()
        pprint(vars)

    stations=[350]
    sDate='19500101'
    eDate='20170228'
    vrs=['RH', 'EV24']

    fname = "etmgeg_{}B.txt".format(stations[0])
    getKNMIdata(fname, type='w', stations=[350], sDate=sDate, eDate=eDate, vars=vrs)

    fname = '/Users/Theo/GRWMODELS/python/KNMI_data/data/etmgeg_350B.txt'
    pdb.set_trace()
    df1, df2 = readKNMIdata(fname, readKNMIdata(fname))
    print(df1)
    print(df2)




