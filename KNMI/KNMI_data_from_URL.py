# KNMI weergegevens ophalen met script en weergeven

"""https://www.knmi.nl/kennis-en-datacentrum/achtergrond/data-ophalen-vanuit-een-script

# Variabelen opgeven in de payload:

WIND = DDVEC:FG:FHX:FHX:FX wind
TEMP = TG:TN:TX:T10N temperatuur
SUNR = SQ:SP:Q Zonneschijnduur en globale straling
PRCP = DR:RH:EV24 neerslag en potentiële verdamping
PRES = PG:PGX:PGN druk op zeeniveau
VICL = VVN:VVX:NG zicht en bewolking
MSTR = UG:UX:UN luchtvochtigheid
ALL alle variabelen
Default is ALL.
"""

# TO 20240714

# %%
import os
import requests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

print('os.getcwd()\n', os.getcwd())

month_names = 'jan feb mar apr mei juni juli aug sep okt nov dec'.split()

# %%
start = '20240601'
end = '20240712'
stn = '260'
vars = ['PRCP']


class Weather_stn():
    """KNMI Weatherstation class.
    for weather stations use what="daggegevens" in the URL for precipitaion stations use what='monv/reeksen'
    
    """

    def __init__(self, what='daggegevens', start='19600101', end='20240701', stn=None, vars=['PRCP'], fmt='csv', folders=None):
        """Return KNMI weerstation data.
        
        Parameters
        ----------
        what = str
            use 'daggegevens' for wheather station data
            use 'monv/reeksen' for rain station data
            
        you can use stn='260' for De Bilt.
        """
        legal_what = ['daggegevens', 'monv/reeksen']
        assert what in legal_what, f'what must be in {legal_what}'
        self.what = what
        if  self.what=='monv/reeksen':
            vars = None
        self.URL = os.path.join('https://www.daggegevens.knmi.nl/klimatologie/', what)
        self.start = start
        self.end = end
        assert stn is not None, "stn must not be None! Use 260 for De Bilt."
        self.stn = stn
        self.stns = [stn]
        self.vars = vars
        self.fmt = fmt
        
        self.folders = folders
                
        self.payload = {'start': self.start,
            'end': self.end,
            'stns': self.stns,
            'vars': self.vars,
            'fmt': 'csv'}

        self.result = requests.get(self.URL, params=self.payload)
        assert self.result.ok, f"Response is not ok, status_code = {self.result.status_code}"
        
    def outfile(self, start=None, end=None, path=None):
        """Return a suitable name for the outfile to be saved.
        
        Parameters
        ----------
        start: 'YYYYMMDD'
            start of data series
        end: 'YYYYMMDD'
            end of data series
        path: outfile            
        """
        if start is None:
            start = self.start
        if end is None:
            end = self.end
            
        outfile = f"KNMI_stn_{self.stn}_{':'.join(self.vars)}_{start}-{end}.txt"
        
        if os.path.isdir(path):
            outfile = os.path.join(path, outfile)
        
        return outfile
            

    @property
    def data(self):     
        """Return the weather station's data as a pd.DataFrame and keep it as self.df"""
        data = self.result.text.split('\n')
        
        headers = []
        for ln in data:
            if ln.startswith('#'):
                headers.append(ln)
            else:
                break
        self.columns = headers[-1][2:].replace(',', ' ').split()
        
        for ln in headers:
            data.pop(0)
            
        for i in range(len(data)):
            data[i] = data[i].replace(' ', '').split(',')
            
        if len(data[-1]) <=1:
            data.pop()
            
        if self.what == 'daggegevens':
            if self.vars[0] == 'PRCP':
                df = pd.DataFrame(data, columns=self.columns).astype(
                                  {'DR': float, 'RH': float, 'EV24': float})
            else:
                df = pd.DataFrame(data, columns=self.columns)
                
            df.index = [np.datetime64(f'{d[:4]}-{d[4:6]}-{d[6:]}') for d in df['YYYYMMDD']]
            df = df.drop(columns=['STN', 'YYYYMMDD'])
        
            if self.vars[0] == 'PRCP':
                df['EV24'] /= 10 # mm/d
                df['RH'  ] /= 10 # mm/d
                df['RH'  ][df['RH'] < 0] = 0.025 # mm/d
                df['RH'  ] = np.round(df['RH'], 1)
                df['DR'  ] /= 10 # h/d
            
            df.index += np.timedelta64(1, 'D') # End of 24h period in which the rain fell.
        
        elif self.what == 'monv/reeksen':
            df = pd.DataFrame(data, columns=self.columns)
            df = df.loc[df['RD'] != '']
            df.index = [np.datetime64(f'{d[:4]}-{d[4:6]}-{d[6:]}') for d in df['YYYYMMDD']]
            
            df.index += np.timedelta64(8, 'h') # End of 24 h period in which the rain fell
            
            df = df.drop(columns=['STN', 'YYYYMMDD'])
            df = df.astype(dtype = {'RD': float, 'SX': int})
            df.loc[:, 'RD'  ] = df['RD'] /10 # mm/d
            df.loc[df['RD'] < 0, 'RD'] = 0.025 # mm/d
            df.loc[:, 'RD'] = np.round(df['RD'], 1)
        
        _I = np.where(np.diff(df.index) > np.timedelta64(1, 'D'))[0]
        if len(_I) > 0:            
            self.df = df.iloc[_I[-1] + 1:]
        else:
            self.df = df
        return self.df
                                    
    def surplus_from_oct(self):
        """Return precipitation surplus okt till current month-day"""
        dt = datetime.today()
        years = np.unique(self.df.index.year.values)
        periods = [(np.datetime64(f'{y0}-10-01'),np.datetime64(f'{y1}-{dt.month:02d}-{dt.day:02d}'))
                   for y0, y1 in zip(years[:-1], years[1:])]
        rch = []
        for yr, (d1, d2) in zip(years[:-1], periods):
            idx = np.logical_and(self.df.index >= d1, self.df.index <= d2)
            rch.append([yr, d1, d2, np.round((self.df.loc[idx, 'RH'] - self.df.loc[idx, 'EV24']).sum(), 1)])
        
        return pd.DataFrame(rch, columns=['yr', 'd1', 'd2', 'rch'], index=years[:-1])
        
def datetime2knmi_date(dt):
    ddtt = dt.tolist()
    return f"{ddtt.year:4s}{ddtt.month:02s}{ddtt.day:02s}"

def get_deBilt():
    """Return 24h values for DR, RH and EV24 of DeBilt >= 1960-01-01 as pd.DataFrame and save in deBilt.csv.
    
    The data are directly requested from the KNMI server.
    
    The data are returned in hours/day for DR and in mm/d for RH and EV24. 
    
    Default values are used as input of class Weather_stn (weerstation) to get the total data set
    uptodate from 1960-01-01.
    """
    baseFile = "deBilt.csv"
    knmi = Weather_stn(what='daggegevens', start='19600101', end='20250101', stn='260', vars=['PRCP'], fmt='csv', folders=None)   
    df = knmi.data
    df.to_csv(baseFile)
    print(f"deBilt data 1960-01-01 to {np.datetime64('today')} were retrieved and saved to {baseFile}")
    print("The data are returned as a pd.DataFrame with 'DR' [h/d], RH [mm/d] and EV24 [mm/d].")
    return df

def get_rain_station(station_name='DeBilt', stn=260):
    """Return 24h values for DR, RH and EV24 of given KNMI rain station for the available period as pd.DataFrame and save in <station>.csv.
    
    The data are directly requested from the KNMI server.
    
    The data are returned in hours/day for DR and in mm/d for RH and EV24. 
    
    Default values are used as input of class Weather_stn (weerstation) to get the total data set
    uptodate from 1960-01-01.
    """

    baseFile =f"{station_name}.csv"
    knmi = Weather_stn(what='monv/reeksen', start='19600101', end='20250101', stn=stn, fmt='csv', folders=None)   
    df = knmi.data
    df.to_csv(baseFile)
    print(f"{station_name} data all available data before {np.datetime64('today')} were retrieved and saved to {baseFile}")
    print("The data are returned as a pd.DataFrame with 'DR' [h/d], RH [mm/d] and EV24 [mm/d].")
    return df

def rain_station_csv2df(csv_file, path=None):
    if path is not None:
        path = os.path.join(path, csv_file)
    assert os.path.isfile(path), f"Cant't find file <{path}>"
    df = pd.read_csv(path, index_col=0, parse_dates=True, dayfirst=False)
    return df

# %%
if __name__ == '__main__':
        
    # %% Neerslagstations KNMI anno 2021
    
    # Get the closest KNMI precipitation stations closest to Rondeellaan Hilversum
    knmi_nsl_stns_csv = 'Neerslagstations_mei2021_1d.csv'
    assert os.path.isfile(knmi_nsl_stns_csv), f"Can't find <{knmi_nsl_stns_csv}>"
    nsl_stns = pd.read_csv(knmi_nsl_stns_csv)
    
    xC, yC = 142.530, 469.420 # Centrum cirkel Rondeellaan Hilversum
    
    # Distance R to center of Rondeellaan Hilversum
    dx, dy = nsl_stns['STN_POS_X'] - xC , nsl_stns['STN_POS_Y'] - yC
    nsl_stns['R'] = np.round(np.sqrt(dx ** 2 + dy ** 2), decimals=1)
    
    # nearest 5 precip. stations
    idx_nearest = nsl_stns['R'].sort_values().index[:5]
    stns_nearest = nsl_stns.loc[idx_nearest]
    
    # Get the precipitation data for the nearest 5 rain stations:
    start = '19600101'
    end =   '20250101'

    for idx in stns_nearest.index:
        stn_meta = stns_nearest.loc[idx]
        csv_basename = stn_meta['LOCATIE'] + str(stn_meta['STN']) + '.csv'

        # Read the data fo the nearst rain stations
        knmi = Weather_stn(what='monv/reeksen', start=start, end=end, stn=stn_meta['STN'], folders=None)
        data = knmi.data  
        
        csv_out = csv_basename
        data.to_csv(csv_out)
        print(f'saved file <{csv_out}>')
     
    # %% Get deBilt weather station (stn = 260) data from 1960 till now
    stn = '260'
    vars = ['PRCP']
    baseFile = "De Bilt" + stn + ".csv"

    knmi = Weather_stn(what='daggegevens', start=start, end=end,  stn=stn, folders=None)    
    data = knmi.data
    data.to_csv(baseFile) # Save the data as a csv file

    # %% Plot precipitation - Makkink evapotranspiration for De Bilt
    fig, ax = plt.subplots()
    ax.set_title('deBilt RH - EV24')
    ax.set_ylabel('mm/d')
    ax.grid()
    ax.plot(data.index, data['RH'] - data['EV24'], label='RH - EV24 mm/d')
    ax.legend()

    # %% Verify correlation between rain stations
    stns = ["Laren593", "Eemnes596", "Soest595", "Spakenburg576", "De Bilt550", 'De Bilt260'] 
    
    rdf = {}
    for stn in stns:
        rdf[stn] = rain_station_csv2df(stn + '.csv')
        if stn == 'De Bilt260':# tackle weerstation De Bilt when used instead of rain station
            rdf[stn]['RD'] = rdf[stn]['RH']
            rdf[stn].index += np.timedelta64(8, 'h') # Shift 
        
    print("\nVerify rain station data series for gaps and missing values:\n")
    all_stns = pd.DataFrame(index=rdf[stns[0]].index)
    for stn in stns:
        all_stns[stn] = rdf[stn].loc[:, 'RD']
        unique_dts = np.unique(np.diff(rdf[stn].index) / np.timedelta64(1, 'D'))
        nans = np.isnan(rdf[stn]['RD']).sum()
        start, end = rdf[stn].index[[0, -1]]
        print(f"{stn:15s}: from {start} to {end} unique dts = {unique_dts}, nr of NaNs ={nans}")
        
    all_stns = all_stns.dropna(axis=0)
    print(f"\nCorrelation between rain stations between {str(all_stns.index[0])[:10]} and {str(all_stns.index[-1])[:10]}:\n")
    print(np.round(all_stns.corr(), decimals=2))
    
    print("""
          The correlation between De Bilt550 rain station and De Bil260 wheather station is bad, because there is an 8 hour difference
          between the 24 hour period over which the rain in a rain station and a weather station is recorded.
          The rain station always on 08:00h UTC and weather station at 00:00 h UTC.
          Not that the datetime for both the weather and rain stations is now the ennd of the 24h period
          over which it was measured. Hence the rain on Jan 1 is registered in this dataframe on 2 jan 00:00 h
          for the weather station and on 2 jan 08:00 UTC for the rain station.
    """)
    
    print("""
    In conclusion: the distance between rain stations matters. So for Hilversum
    prefer nearby Laren over the other KNMI more distant rain stations whenever possible.
    
    The data of stations nearby Hilversum are are shorter than that of De Bilt, except Spakenburg.
    
    Yet, Spakenburg has a higher correlation with Laren than De Bilt has, while Laren
    can is closest to Hilversum. Therefore we can generate the best approximation to Hilversum by taking Spakenburg from 1960 till start of Laren and Laren afterwards.
    
    This is our proxy for Hilversum rain.
    """)
    
    # Generate a proxy weather station for Hilversum
    stn = 'HilvProxy'
    rdf[stn] = rain_station_csv2df('Spakenburg576' + '.csv')
    rdf['Laren593']= rain_station_csv2df('Laren593' + '.csv')
    rdf[stn].loc[rdf[stn].index >= rdf['Laren593'].index[0]] = rdf['Laren593']
    rdf[stn]['EV24'] = rdf['De Bilt260']['EV24']
    rdf[stn] = rdf[stn].dropna(axis=0)
    rdf[stn]['RH'] = rdf[stn]['RD']
    rdf[stn].to_csv(stn + '.csv')
    
    print("""
    The rain data are valid for the day at 08:00 from 08:00 the previous day.
    The EV24 data, we use De Bilt, are valid on the day for whic it is given.
    In our recharge computation we combine the EV24 data of De Bilt of the previous day with
    the rain data on the reported day, which thus holds voor the current day at 08:00
    and the 24 hours before that.
    In our simulation we should in fact comopare with the groundwater head data recorded on
    the current day at 08:00 or intepolated to this date and time.
    
    For the recharge we should set the index on the reported day 08:00 h. It's regarded as
    the moment of reporting.
    """)
    
    # %% Plot rain surplus between october and july for every year since 2060
    
    rchFromOct = knmi.surplus_from_oct()
    
    today = datetime.today()
    
    fig, ax = plt.subplots()
    ax.set_title(f'deBilt RH - EV24, from {month_names[9]}-01 to {month_names[today.month - 1]}-{today.day}')
    ax.set_ylabel('mm')
    ax.grid()
    ax.plot(rchFromOct.index, rchFromOct['rch'], label=f'RH - EV24 (okt - {month_names[today.month - 1]}-{today.day})')
    y = rchFromOct.iloc[-1]['rch']
    ax.text(rchFromOct.index[-1], y, f"{y:.0f} mm", ha='center')
    ax.legend()
    
    plt.show()
    
# %%
