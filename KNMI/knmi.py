"""Reading data files of the Royal Dutch Meteorological Institute KNMI.

KNMI provides data files for all of its rain gauge stations and meteo stations for free.
These data can be downloaded by hand or by script.

How to download KNMI data by script is described here (site last checkked on 20200516)
http://www.knmi.nl/kennis-en-datacentrum/achtergrond/data-ophalen-vanuit-een-script

However, we can just use the more convenient requests module.

Dowloading is done from: (site last checked on 20200516)
http://www.knmi.nl/nederland-nu/klimatologie/daggegevens

vars
Some convenience acronymes are defined to select related parameter groups

WIND = DDVEC:FG:FHX:FHX:FX             wind
TEMP = TG:TN:TX:T10N                   Temperatuur
SUNR = SQ:SP:Q                         Zonneschijnduur en globale straling
PRCP = DR:RH:EV24                      Neerslag en potentiële verdamping
PRES = PG:PGX:PGN                      Druk op zeeniveau
VICL = VVN:VVX:NG                      Zicht en bewolking
MSTR = UG:UX:UN                        luchtvochtigheid
ALL  =                                 All variables
Default is ALL.

The header line is in lines 1 through NSTN+NVAR+11 and starts with #
Data are in lines NSTN+NVAR+12 and beyond

See below how the data can be downloaded using requests.

@TO 2020-05-23
"""

import os
import requests
import pandas as pd
import numpy as np
import logging
#import matplotlib.pyplot as plt


logger=logging.getLogger()

# Uses 'w' for weather station, 'r' for rainstation and 'h' for hourly data
KNMI_URL = {'weer' : 'http://projects.knmi.nl/klimatologie/daggegevens/getdata_dag.cgi',
            'neerslag' : 'http://projects.knmi.nl/klimatologie/monv/reeksen/getdata_rr.cgi',
            'uur' : 'http://projects.knmi.nl/klimatologie/uurgegevens/getdata_uur.cgi'}

def datetime_index(data, datecol='YYYYMMDD', UTC=None):
    """Return datetime index for a pd dataframe parsed from column 'YYYYYDDMM'.

    Parameters
    ----------
        data: pd.DataFrame holding all the data including the column col
            the pd.DataFrame with the data
        col: str
            col to be converted into a pd.datetime index. The name of the col
            is used as the format yyyymmdd in the columns
        UTC: int
            UTC time in hours in UTC, marking the end of the prevous 24 hour
            period over which the values are reported. Mostly at 08:00 UTC.
            Wintertime is UTC +1h, summer time is UTC + 2h in the Netherlands
    """
    dt = np.asarray([d[:4] + '-' + d[4:6] + '-' + d[6:8]
                      for d in data[datecol].astype(str).values], dtype=np.datetime64)
    if UTC: # then add the UTC hours to the date
       return dt + np.timedelta64(int(UTC), 'h')
    else:
        return dt

def skiprows(fname, lookfor='STN', maxlines=100):
    """Return column names and rows to skip when reading into pd.DataFrame.

    Sections in KNMI data files generally start with '# STN' where STN is the
    station number.

    Parameters
    ----------
        filename: str
            name of file
        lookfor: str
            str to look for at the beginning of the line that was read
        times: int
            the number of time that lookfor must have occurred

    Returns
    -------
        columns: list of tokens find on the last line read (column headers)
        nrow: the number of lines processed (linenumber + 1)
    """
    irow, n = 0, -1
    with open(fname, 'r') as f:
        # Some files have commment lines starting with # others not
        # These files have a blank line after the header others not
        s = f.readline()
        hatch = s[0] == '#'
        irow += 1

        # Keep the last line with flookfor as header
        while irow < maxlines:
            s = f.readline()
            if hatch:
                s = s[2:]
            if s.startswith(lookfor):
                n = irow
                header = s
            irow += 1
    if n < 1:
        raise ValueError(
            f'"{lookfor}" was not found in {maxlines} lines of file {fname}')
    columns = header.replace(',',' ').split()

    # nskip is n + 1 if no blank line in file else it is n + 2
    nskip = n + 2 if hatch else n + 1
    return columns, nskip


def parseKNMI(knmifname, fields=['RH', 'EV24'], datecol='YYYYMMDD', tompd=None, UTC=None):
    """Return pd.DataFrame form knmi weather station with given fields.

    The knmi file was downlaoded from KNMI site see above

    Parameters
    ----------
    knmifname: str
        name of knmi file for one of their weather or precipitation stations.
    fields: list of str of None to get all fields
        list of the desired fields.
    tompd: bool
        if True, then fields in fields will be converted from 0.1 mm/d to mpd
    UTC: int
        number of hours to add to index to match exact time of value registration in UTC
        MET = UTC + 1, MEZT = UTC + 2

    The datecol will be converted to timestamps and used as the index.
    Lines with Nan in any of the fields are dropped.

    @TO 20190106, 20200512

    """
    columns, skip = skiprows(knmifname)

    data = pd.read_csv(knmifname, header=None, skip_blank_lines=False,
                       skiprows=skip, skipinitialspace=True)

    # set column headers. Some files (precipitationstaton files have an extra
    # field with the station name, without a specific header. So we add it
    try:
        data.columns = columns
    except ValueError:
        data.columns = columns + ['NAME']
    data.index = datetime_index(data, datecol=datecol, UTC=UTC)
    data.index.name = 'dateUTC'
    data = data.drop(columns=datecol) # remove the datacol, we have how datetime as index

    # See if you want to convert the columns to m/d
    if fields is not None and tompd:
        for fld in fields:
            b = data.loc[:, fld]  < 0
            data.loc[b, fld] = 0.25
            data[fld] /= 10000.0 # original is in 0.1 mm units
        return data[fields] # Only the specified fields!
    else:
        return data



def get_weather(stn=240, start='20100101', end='20191231', folder=''):
    """Return pd.DataFrame with recharge and makkink evapotranspiration for station.

    The data are downloaded and save in the current directory and then read into
    a pandas DataFrame, which is returned.

    Only the columns 'RH' and 'EV24' are used and the valuea are converted
    to m/d.

    The index is datetime, with time equal to UTC.

    Parameters
    ----------
        stn : int or str
            number of ond of the KNMI weather stations
        start: str in the form 'yyyymmdd'
            first day
        end: str of the form 'yyyymmdd'
            last day
        folder: str, default is '' (not None)
            folder name to store the file and look for it.
    @TO 2020-05-20
    """
    # This allows to get a list of all weather stations, including coordinaters
    what = 'weer'
    URL = KNMI_URL[what]
    stns  = [str(stn)]
    vars_ = 'PRCP'
    fields = ['RH', 'EV24']
    payload = {'start':start, 'end':end, 'vars':vars_, 'stns':':'.join(stns)}
    fname =os.path.join(folder, f"{what}{'+'.join(stns):}_{start}_{end}.txt")
    if os.path.isfile(fname): # file exists, don't download the data
        print(f"File <{fname}> exists, download was skipped.")
    else:
        r = requests.post(URL, data=payload)
        with open(fname, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=128):
                fd.write(chunk)
        print(f"File <{fname}> saved.")

    data = parseKNMI(fname, fields=fields, tompd=True, UTC=8)
    return data

# kaart neerslagstations KNMI:
# https://www.google.com/maps/d/viewer?mid=121t7Wgv0wTC7m4PZeDwk5I6jx8aaBZGP&ll=52.30459614052015%2C5.104167804120152&z=10

if __name__ == '__main__':

    # The follwing 4 requests allow to caputure lists of all available stations
    # However, not that these requests may take time, even though only a single day's data is requested
    # Also not that the weather stations and teh station for the hourly data are identical.
    # Furthermore, the # are missing for the header lines.
    # The precipitaton data are from 08:00 UTC on the previouw day to 08:00 UTC on the current day.
    # The precipidation station of tbe same name as a weather station has a differenct station number.

    # Get the precipitation and evaporation data for some stations
    if False:
        what = 'neerslag'
        URL = KNMI_URL[what]
        stns  = ['664', '458', '680', '678']
        start = '20030101'
        end   = '20030103'
        vars_ = 'PRCP'
        payload = {'start':start, 'end':end, 'vars':vars_, 'stns':':'.join(stns)}
        fname =f"{what}{'+'.join(stns):}_{start}_{end}.txt"
        r = requests.post(URL, data=payload)
        with open(fname, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=128):
                fd.write(chunk)

    if False:
        # This allows to get a list of all precipitaion stations. The obtained data do not
        # have the coordinates of the precipitation station on board.
        stns  = ['ALL']
        start = '20030101'
        end   = '20030101'
        vars_ = 'PRCP'
        payload = {'start':start, 'end':end, 'vars':vars_, 'stns':':'.join(stns)}
        fname =f"{what}{'+'.join(stns):}_{start}_{end}.txt"
        r = requests.post(URL, data=payload)
        with open(fname, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=128):
                fd.write(chunk)

    if False:
        # This allows to get a list of all weather stations, including coordinaters
        what = 'weer'
        URL = KNMI_URL[what]
        stns  = ['ALL']
        start = '20030101'
        end   = '20030101'
        vars_ = 'PRCP'
        payload = {'start':start, 'end':end, 'vars':vars_, 'stns':':'.join(stns)}
        fname =f"{what}{'+'.join(stns):}_{start}_{end}.txt"
        r = requests.post(URL, data=payload)
        with open(fname, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=128):
                fd.write(chunk)

    if False:
        # This allows to get a list of all hourly data stations
        what = 'uur'
        URL = KNMI_URL[what]
        stns  = ['ALL']
        start = '20030101'
        end   = '20030101'
        vars_ = 'PRCP'
        payload = {'start':start, 'end':end, 'vars':vars_, 'stns':':'.join(stns)}
        fname =f"{what}{'+'.join(stns):}_{start}_{end}.txt"
        r = requests.post(URL, data=payload)
        with open(fname, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=128):
                fd.write(chunk)


    if False: # ABCOUDE neerslagstation
        what = 'neerslag'
        URL = KNMI_URL[what]
        stns=['572'] # Abcoude
        start = '20100101'
        end = '20191231'
        vars_ = 'PRCP'
        payload = {'start':start, 'end':end, 'vars':vars_, 'stns':':'.join(stns)}
        fname =f"{what}{'+'.join(stns):}_{start}_{end}.txt"
        r = requests.post(URL, data=payload)
        with open(fname, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=128):
                fd.write(chunk)
            print('File ' + fname + ' saved')

        dataAbcoude = parseKNMI(fname, fields=['RD'], tompd=True, UTC=8)

    # Download weather station data Schiphol
    # Parse this into a pd.DataFrame
    # Retain only columns RH and EV24 and convert to m/d.
    if False: # Schiphol weerstation
        what = 'weer'
        URL = KNMI_URL[what]
        stns=['240'] # Schiphol
        start = '20100101'
        end = '20191231'
        vars_ = 'PRCP'
        payload = {'start':start, 'end':end, 'vars':vars_, 'stns':':'.join(stns)}
        fname =f"{what}{'+'.join(stns):}_{start}_{end}.txt"
        r = requests.post(URL, data=payload)
        with open(fname, 'wb') as fd:
            for chunk in r.iter_content(chunk_size=128):
                fd.write(chunk)
            print('File ' + fname + ' saved')

        data = parseKNMI(fname, fields=['RH', 'EV24'], tompd=True, UTC=8)


    # This is the way to get a data (240 is schiphol) directly from KNMI website
    data = get_weather(stn=240, start='20100101', end='20191231')