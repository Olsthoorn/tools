# KNMI weergegevens ophalen met script en weergeven

"""https://www.knmi.nl/kennis-en-datacentrum/achtergrond/data-ophalen-vanuit-een-script

# Vars to give in the request.payload:

Group parameters (vars)
-----------------------
WIND = DDVEC:FG:FHX:FHX:FX wind
TEMP = TG:TN:TX:T10N temperatuur
SUNR = SQ:SP:Q Zonneschijnduur en globale straling
PRCP = DR:RH:EV24 neerslag en potentiële verdamping
PRES = PG:PGX:PGN druk op zeeniveau
VICL = VVN:VVX:NG zicht en bewolking
MSTR = UG:UX:UN luchtvochtigheid
ALL  = alle variabelen

Default is ALL.
"""

# TO 20240714

# %%
import os
import numbers
import requests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from matplotlib.patches import Circle

print('os.getcwd()\n', os.getcwd())

month_names = 'jan feb mar apr mei juni juli aug sep okt nov dec'.split()

# %%
start = '20240601'
end = '20240712'
stn = '260'
vars = ['PRCP']


class Weather_stn():
    """KNMI Weather station class works for weather stations and precipitation stations.
    
    For weather stations use:      "daggegevens" in the URL
    For precipitaion stations use: 'monv/reeksen' in the URL
    """
    def __init__(self, what='weather',
                 start='19600101', end='20260701',
                 stns=260, vars=['PRCP'], fmt='csv', folder=None):
        """Return KNMI weerstation or precip station data.
        
        Parameters
        ----------
        what = str
            use 'weather' for wheather station data
            use 'preciptation' for rain station data
        startt: 'yyyymmdd'
            start date
        endtt: 'yyyymmdd'
            end date
        stns: int like 260 or sequence of ints
            You can use stn='260' for weather station De Bilt and '550' for precip station De Bilt.
        vars: sequence of str
            list of fields or fiedl groups (see on top of this file).
        fmt: str
            file format
        folders: object
            folder: dirs.weer
        """       
        # --- Choose 'weather' or 'precipitation'
        # --- Tricky: self.what differs from what!!
        if what == 'weather':
            self.what = 'daggegevens'
        elif what == 'precipitation':
            self.what = 'monv/reeksen'
        elif what == 'hourly':
            self.what = 'uurgegevens'
        else:
            raise ValueError(f"What must be 'weather' or 'precipitation' not '{what}'")
            
        if  what == 'precipitation':
            vars = None
        
        self.URL = os.path.join('https://www.daggegevens.knmi.nl/klimatologie/', self.what)
        
        self.start = start
        self.end = end
        
        # --- Prepare stns for payload
        if stns is None:
            raise ValueError("stns must not be None!\n" +
                    "Use 260 for De Bilt of a sequence of station numbers.")
        elif isinstance(stns, (str, numbers.Integral)):
            self.stns = str(stns)
        else:
            stns = ':'.join(map(str, stns))
            
        # --- Prepare vars for payload
        if vars is None:
            self.vars = vars
        else:
            self.vars = vars if isinstance(vars, str) else ':'.join(vars)
            
        # --- Not used, use .csv always
        self.fmt  = fmt
                
        # --- Folder where to store the data .csv file
        self.folder = folder
        if not os.path.isdir(folder):
            raise FileNotFoundError("Not a folder: '{folder}'")
                
        # --- Request's payload
        self.payload = {'start': self.start,
                        'end': self.end,
                        'stns': self.stns,
                        'vars': self.vars,
                        'fmt': 'csv'}

        # --- Generate outfile name (.csv file)
        if self.what == 'daggegevens':
            self.outfile = f"weer_{self.stns}_{start}_{end}.txt"
        elif self.what == 'monv/reeksen':
            self.outfile = f"prec_{self.stns}_{start}_{end}.txt"
        elif self.what == 'uurgegevens':
            self.outfile = f"uur_{self.stns}_{start}_{end}.txt"
            
        # === Get the data
        if os.path.isfile(os.path.join(self.folder, self.outfile)):
            # --- Data was already downloaded, skip
            print(f"Outfile {self.outfile} already exists in '{self.folder}', request skipped.")
            self.result = None
            
        else:
            # --- Issue the request for the data ----
            self.result = requests.get(self.URL, params=self.payload)
            
            # --- Verify the success of the request
            assert self.result.ok, f"Response is not ok, status_code = {self.result.status_code}"
            print("Request result ok.")
                    
            # --- Write the request reseult to a .csv file (outfile)
            with open(os.path.join(self.folder, self.outfile), 'w') as f:
                f.write(self.result.text)
                print(f"KNMI-file {os.path.basename(self.outfile)} written to {self.folder}.")
        
    @property
    def df(self):     
        """Return the weather station's data as a pd.DataFrame"""
        # --- split the entire requests result into lines
        if self.result is None:
            with open(os.path.join(self.folder, self.outfile), 'r') as f:
                data = f.readlines()
        else:
            data = self.result.text.split('\n')
        
        # --- Get header lines first
        headers = []
        for idx, ln in enumerate(data):
            if ln.startswith('#'):
                headers.append(ln[1:]) # --- cut off the #
            else:
                break
            
        # --- Store the header line with the data explications
        self.headers = headers[:-1]
            
        # --- Column headers are in the last line starting with #
        self.columns = [k.strip() for k in headers[-1].strip().split(',')]
        
        # --- Only keep the data lines after the headers
        data = data[idx:]
        
        # --- Drop last line if corrupt or incomplete
        if len(data[-1]) <= 1:
            data.pop()
                
        # --- Remove spaces and the split on ',' for each data line
        data = [d.strip().replace(' ', '').split(',') for d in data]
        
        # --- Convert data into DataFrame
        df = pd.DataFrame(data, columns=self.columns)

        # --- Remove lines with an empty field
        df = df.replace('', np.nan)
        df = df.astype(float)
        df = df.dropna()

        # --- Replace the index by column 'YYYYMMDD' after conversion to timestamp
        df.index  = pd.to_datetime(df['YYYYMMDD'], format='%Y%m%d')
        df.index.name = 'datetime'
        
        # --- No need for the these colunmns anymore
        df = df.drop(columns=['STN', 'YYYYMMDD'])

                      
        # === Daggegevens ===
        if self.what == 'daggegevens':
        
            # --- Convert the columns in group PRCP from tenths of mm/d to mm/d
            if self.vars[0] == 'PRCP':
                df['EV24'] /= 10 # mm/d
                df['RH'  ] /= 10 # mm/d
                df.loc[df['RH'] < 0, 'RH'] = 0.025 # mm/d
                df['RH'  ] = np.round(df['RH'], 1)
                df['DR'  ] /= 10 # h/d rainfall duration


        # === Uurgegevens ===
        if self.what == 'uurgegevens':

            # --- Add hour to the index
            df = df.astype({'HH': int})
            df.index += pd.to_timedelta(df['HH'], unit='h')
            
            # --- No need for the this colunmn anymore
            df = df.drop(columns=['HH'])
        
            # --- Convert the columns in group PRCP from tenth of mm to mm/d
            if self.vars[0] == 'PRCP':                
                df['RH'  ] /= 10 # mm/d
                df.loc[df['RH'] < 0, 'RH'] = 0.025 # mm/d
                df['RH'  ] = np.round(df['RH'], 1)
                df['DR'  ] /= 10 # h/d rainfall duration


        # === monv/reeksen (precipitation) ====
        elif self.what == 'monv/reeksen':
                        
            # --- Set the type of SX (sneeuwdek index)
            df = df.astype({'SX':int})
                        
            # --- Shift time index to end of 24 h period in which the rain fell
            df.index += np.timedelta64(8, 'h') # End of 24 h period in which the rain fell
                                    
            # --- Convert from tenths of mm/d to mm/d
            df['RD'] /= 10 # mm/d
            df.loc[df['RD'] < 0, 'RD'] = 0.025 # mm/d
            df.loc[:, 'RD'] = np.round(df['RD'], 1)
        
        # === Only retain the data after the last data gap
        # --- Get all  indices where data start missing
        _I = np.where(np.diff(df.index) > np.timedelta64(1, 'D'))[0]
        
        # --- If there are gaps, then keep the df after the last hole in the data.
        if len(_I) > 0:            
            df = df.iloc[_I[-1] + 1:]
        # --- Return the DataFrame
        return df
    

def datetime2knmi_date(dt):
    ddtt = dt.tolist()
    return f"{ddtt.year:4s}{ddtt.month:02s}{ddtt.day:02s}"


# === Namespace for project directories
class Dirs():
    def __init__(self):
        self.data = '/Users/Theo/Development/python/hydro_tools/tools/KNMI/data'


# %%
if __name__ == '__main__':
    # %%    
    # --- Data period:
    startt = '19600101'
    endt =   '20260701'
            
    dirs = Dirs()
      
    # %% --- Request data from KNMI or use file if it already exists.
          
    # --- Weather data for De Bilt (Weather station 260)
    DeBilt260 = Weather_stn(what='weather', start=start, end=endt, stns='260', vars=['PRCP'], fmt='csv', folder=dirs.data)
    
    # --- Return as pd.DataFrame
    dB260 = DeBilt260.df

    # --- Precipitation data for De Bilt (Rain station 550)
    DeBilt550 = Weather_stn(what='precipitation', start=startt, end=endt, stns='550', vars=['PRCP'], fmt='csv', folder=dirs.data)
    
    # --- Return as pd.DataFrame
    dB550 = DeBilt550.df
    
    # --- Hourly weather data (here just rain)
    # dBiltUur = Weather_stn(what='hourly', start=20260101, end=20260501, stns='260', vars=['PRCP'], fmt='csv', folder=dirs.data)
    # dB260h = dBiltUur.df
    
    # --- Hourly weather data (here just rain)
    dBiltUur = Weather_stn(what='hourly', start=20260101, end=20260501, stns='260', vars=['P', 'RH', 'FH', 'DD', 'U'], fmt='csv', folder=dirs.data)
    dB260h = dBiltUur.df


        
    # %% --- Show neerslagstations KNMI anno 2021
    
    # --- Let's define a project around a point
    # --- Rondeellaan Hilversum    
    xC, yC = 142.530, 469.420 # Centrum cirkel Rondeellaan Hilversum

    # Get the closest KNMI precipitation stations closest to Rondeellaan Hilversum
    
    # --- Meta data of the precipitation stations downloaded from KNMI
    knmi_nsl_stns_csv = 'KNMI_neerslagstations_mei2021_1d.csv'
    if not os.path.isfile(os.path.join(dirs.data, knmi_nsl_stns_csv)):
        raise FileNotFoundError(f"Can't find <{knmi_nsl_stns_csv}>")
    
    # --- Read the file
    nsl_stns = pd.read_csv(os.path.join(dirs.data, knmi_nsl_stns_csv))
    
    # --- Add distance R to center of Rondeellaan Hilversum to the DataFrame of the prec. stations
    dx, dy = nsl_stns['STN_POS_X'] - xC , nsl_stns['STN_POS_Y'] - yC
    nsl_stns['R'] = np.round(np.sqrt(dx ** 2 + dy ** 2), decimals=1)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set(title='Neerslagstations en project centrum met cirkel', xlabel='x [km]', ylabel='y km]')
    
    # --- The stations
    ax.plot(nsl_stns['STN_POS_X'], nsl_stns['STN_POS_Y'], 'ro')
    
    # --- Get the 5 nearest stations
    idx_nearest = nsl_stns['R'].sort_values().index[:5]
    stns_nearest = nsl_stns.loc[idx_nearest]
    
    # --- The 5 nearest sations
    ax.plot(stns_nearest['STN_POS_X'], stns_nearest['STN_POS_Y'], 'ro', mec='k', mfc='none' )
    ax.plot(xC, yC, 'k*', label='projectcentrum')
    
    for idx in idx_nearest:
        stn = stns_nearest.loc[idx]
        ax.text(stn['STN_POS_X'], stn['STN_POS_Y'], "<--" + stn['LOCATIE'],
                ha='left', va='center')
    
    # --- Circle / disk around project center
    ci = Circle((xC, yC), radius=13.5, ec='k', fc='gray', alpha=0.5)
    ax.add_patch(ci)
    
    ax.grid()
    ax.set_aspect(1.0)
    ax.legend()
    plt.show()
    
    
    # %% --- Get the precipitation data for the nearest 5 rain stations:    
    for idx in stns_nearest.index:
        stn_meta = stns_nearest.loc[idx]

        # --- Request the data and save it
        knmi = Weather_stn(what='precipitation',
                           start=startt,
                           end=endt,
                           stns=stn_meta['STN'],
                           folder=dirs.data)        
             

    # %% --- Precipitation - Makkink evapotranspiration for De Bilt
    DeBilt260 = Weather_stn(what='weather', start=startt, end=endt, stns='260', vars=['PRCP'], fmt='csv', folder=dirs.data)
    dB260 = DeBilt260.df

    fig, ax = plt.subplots()
    ax.set_title('deBilt RH - EV24')
    ax.set_ylabel('mm/d')

    ax.plot(dB260.index, dB260['RH'] - dB260['EV24'], label='RH - EV24 mm/d')
    
    ax.grid()
    ax.legend()


    # %% --- Verify correlation between rain stations
    stnNms = ["Laren593", "Eemnes596", "Soest595", "Spakenburg576", "De Bilt550"]
    
    stns = [int(k[-3:]) for k in stnNms]
    
    rdf = {}
    for stn in stns:
        rdf[stn]=Weather_stn(what='precipitation',
                             start=startt, end=endt, stns=stn, vars=['PRCP'], fmt='csv', folder=dirs.data).df

    # --- Tackle weerstation De Bilt separately
    deBilt260 = 260
    rdf[deBilt260]=Weather_stn(what='weather',
                start=startt, end=endt, stns=deBilt260, vars=['PRCP'], fmt='csv', folder=dirs.data).df

    rdf[deBilt260].loc[:, 'RD'] = rdf[deBilt260].loc[:, 'RH']
    # --- Shift index 1D and 8 hours to make the indexes compatible
    rdf[deBilt260].index += np.timedelta64(32, 'h') # Shift
    
    print("STN lines")
    for stn, df in rdf.items():
        print(f"{stn} {len(df)}") #  Pnp.unique(np.diff(df.index)))}")
        
    #  --- Correlations between raindata
    print("\nVerify rain station data series for gaps and missing values:\n")
    
    all_stns = pd.DataFrame()
    for stn in stns:
        all_stns[stn] = rdf[stn]['RD']
        
    all_stns[260] = rdf[260]['RH']
        
    all_stns = all_stns.dropna(axis=1, how='any')
        
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
    
    # %% --- Generate a proxy weather station for Hilversum
    
    def generate_hilvProxy():
        """Return precipitation proxy as a pd.DataFrame.
                
        Use global startt and endt 
        """
        stns = {"deBilt":260, "laren": 593, "spakenb": 576}
        
        proxy = Weather_stn(what='weather',
                            start=startt, end=endt,
                            stns=stns['deBilt'], folder=dirs.data).df
        proxy = proxy.drop(columns=['DR'])
        
        # --- Shift index over 1D + 8 hours so that EV24 is the accumulated
        #     value over 24 hours before the rain gauge recording.
        proxy.index += np.timedelta64(1, 'D') + np.timedelta64(8, 'h')
        
        spakenb = Weather_stn(what='precipitation',
                            start=startt, end=endt,
                            stns=stns['spakenb'], folder=dirs.data).df
        laren = Weather_stn(what='precipitation',
                            start=startt, end=endt,
                            stns=stns['laren'], folder=dirs.data).df

        # --- First index of spakenb is not in DeBilt remove it
        spakenb = spakenb.iloc[1:, :]
        proxy.loc[spakenb.index, 'RH'] = spakenb['RD']
        proxy.loc[laren.index,   'RH'] = laren['RD']
        proxy = proxy.dropna(axis=1, how='any')
        return proxy
    
    # --- Generate hilvProxy and save it
    hilvProxy = generate_hilvProxy()
    hilvProxy.to_csv(os.path.join(dirs.data, 'hilvProxy.csv'))
    hilvProxy['RH'].plot()
    
    print("""
    The rain data is accumulated from 08:00 on the previous day to 08:00 on the current dat.
    
    The EV24 data id accumulated over the current day.
    
    In our recharge computation we should take a day from 08:00 to 08:00 to
    match the actual rainfall period precisely.
    So we make sure that the date has the 08:00 time.
    
    For the evaporation we just use the EV24 of the previous day because
    the EV24 for the first 8 hours of any day is small.
    
    So we add 1D + 8h to the index of the weather station
    so that EV24 means EV24 accumulated over the previouw 24 hours.
    
    In our simulation we should in fact compare the groundwater head data recorded on the current day at 08:00 or intepolated to this date and time.
    
    For the recharge we should set the index on the reported day 08:00 h. It's regarded as     the moment of reporting.
    """)
    
    
    # %% --- Plot rain surplus between october and july for every year since 2060
    
    deBilt =Weather_stn(what='weather',
                start=startt, end=endt, stns=260, vars=['PRCP'], fmt='csv', folder=dirs.data).df
    
    fig, ax = plt.subplots()
    
    month1, month2 = 10, 5
    day = 1
    
    years = np.unique([dt.year for dt in hilvProxy.index])
    wettness = []
    for year1, year2 in zip(years[:-1], years[1:]):
        start = np.datetime64(f"{year1:04d}-{month1:02d}-{day:02d}") + np.timedelta64(8, 'h')
        end   = np.datetime64(f"{year2:04d}-{month2:02d}-{day:02d}") + np.timedelta64(8, 'h')
        wettness.append((year1, (hilvProxy.loc[start:end, 'RH'] - hilvProxy.loc[start:end, 'EV24']).sum()))
    
    wettness = np.array(wettness)
    ax.plot(wettness[:, 0], wettness[:, 1], '.-',
            label=f"RH - EV24 (okt - {month_names[month1 - 1]}-{month_names[month2 - 1]})")
    
    ax.set_title(f"deBilt RH - EV24, van {month_names[month1 -1]} tot {month_names[month2 - 1]}")
    ax.set_xlabel('year')
    ax.set_ylabel('mm')
    ax.grid()
    ax.legend()
    
    plt.show()
    
# %%
