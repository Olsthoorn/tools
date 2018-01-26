#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module to read and deal with groundwater head times series downloaded
from www.dinoloket.nl

Created on Tue Jan 24 22:18:31 2017

@author: Theo
"""

import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from collections import UserDict
import numpy as np
from datetime import datetime
import shapefile as sh

def ax_setup(**kwargs):
        ax     = kwargs.pop('ax', None)
        title  = kwargs.pop('title', None)
        xlabel = kwargs.pop('xlabel', 'x [m]')
        ylabel = kwargs.pop('ylabel', 'y [m]')
        xlim   = kwargs.pop('xlim', None)
        ylim   = kwargs.pop('ylim', None)
        grid   = kwargs.pop('grid', True)
        if ax is None:
            fig, ax = plt.subplots()

        if title  is not None: ax.set_title(title)
        if xlabel is not None: ax.set_xlabel(xlabel)
        if ylabel is not None: ax.set_ylabel(ylabel)
        if xlim   is not None: ax.set_xlim(xlim)
        if ylim   is not None: ax.set_ylim(ylim)
        if grid   is not None: ax.grid(grid)
        return ax, kwargs


class Piezometer:
    """Piezometer objects hold piezometer meta and time series data
       downloaded from www.dinolokel.nl

       Piezometer objects have properties obtained from their read mete data
       which can be accessed directly.

       Piezometers are best accessed through the Piezometers (plural) class
       which is a dict type of container with additional methods.

       parameters:
       -----------
       name : the name of the piezometer, = filename
       meta : pandas dataFrame with the meta information in the top of the
              original data file (dinoloket.nl)
       data:  pandas dataFrame with the actual data.
       returns:
       --------
       Piezometer instance
    """
    def __init__(self, name, meta, data):
        """returns an Piez object
        parameters:
        -----------
        name: the name of the piezometer
        meta: a pandas dataFrame with the meta data from the piezometer
        data: a pandas series or dataFrame with the time series of the piezometer
        """
        self.name = name
        self.meta = meta  # pandas dataFrame
        self.data = data  # pandas dataFrame

    def name(self):
        return self.name

    @property
    def x(self):
        return float(self.meta.iloc[0]['x'])
    @property
    def y(self):
        return float(self.meta.iloc[0]['y'])

    @property
    def loc(self):
        return (self.x, self.y)

    @property
    def mv(self):
        '''Returns ground surface elevation'''
        lastRow = len(self.meta)-1
        z = self.meta.ix[lastRow]['mv+nap']
        return z/100.
    @property
    def z0(self):
        '''Returns ground surface elevation.'''
        return self.mv
    @property
    def z1(self):
        '''Returns top of screen.'''
        return self.meta.ix[0]['scr_top']/100.
    @property
    def z2(self):
        '''Returns botom of screen.'''
        return self.meta.ix[0]['scr_bot']/100.
    @property
    def z(self):
        '''Returns tuple (ground elevation, top of screen, bottom of screen).'''
        return (self.z0, self.z1, self.z2)


    def plot(self, *args, **kwargs):
        """Plot the point on the map.
        parameters
        ----------
        args and kwargs
        key : str
            must be a valudi key in self.data.columns to select column to plot.
        ax : plt.Axes
            if present, ax is used, then name of piezom is used in label
            if not present, fig and ax are generated and label is used
            in title.
        xlim = (tstart, tend) tuple of two datetime objects
            if present is used to limit time of plot
        ylim =(min, max) tuple of 2 floats

            all other args and kwargs will be passed on to plt.plot
            if ax in kwargs then ax is used, else fig and ax are generated.

            if ax in kwargs then name of piezom is used inlabel
            else name of piezom goes to title.
        returns
        -------
        ax : plt.Axes object
        """
        key = kwargs.pop('key', 'nap')
        if not key in self.data.columns: key = 'mp'
        if not key in self.data.columns: raise KeyError(key)

        if 'ax' in kwargs:
            kwargs['label'] = self.name
        else:
            kwargs['title'] = self.name

        ax, kwargs = ax_setup(**kwargs)

        ax.plot(self.data.index, self.data[key], *args, **kwargs)
        return ax

    def graph(self, *args, **kwargs):
        """Make times series graph"""
        self.data.plot(*args, **kwargs)

    def __str__(self):
        return \
        '\nPiezometer: ' + self.name + \
        self.meta.__str__() + '\n' + \
        self.data.__str__() + '\n'

    def __len__(self):
        return len(self.data)

class Piezometers(UserDict):
    """Piezometer collection, a dict with added functionaility
       pertaining to groundwater piezometers.

       The collection holds Piezometer objects
    """

    def __init__(self, *args, **kwargs):
        """returns a collection of piezometers
        args
        ----
        args[0] : str
            topfolder: a directory in or below with dinoloket groundwater head
                 time series csv falues are stored.

                 The search will walk the entire path.
                 Head time series files are assumed to be in a subdirectory
                 calle "Grondwaterstanden_put", which is generated after
                 opening a dinoloket.nl zip.file.
        kwargs
        ------
        verbose: bool
            if true info during porcessing is print to the screen
        pickle : bool
            if true piezometers will be pickled
        nmax or n : int
            maximum number of piezometers to read
        pickle : str
            name of the file to pickle the piezometers to
        verbose: bool
            print more info when True
        returns:
        --------
        Piezometers instance, a dict like container of Piezometer objects
        with additional functonality.
        """
        self.data = {}

        if len(args) == 0 and len(kwargs) == 0:
            return  # empty Piezometers

        top_folder  = args[0]
        verbose     = kwargs.pop('verbose', False)
        pickleFname = kwargs.pop('pickle', None)
        nmax        = kwargs.pop('nmax', kwargs.pop('n', None))

        self.failed = list() # names of piezometer failed to be read in

        self.get_dinodataometers(top_folder,
                                   verbose=verbose, pickle_dump=False, nmax=nmax)
        self.cleanup(verbose=verbose)

        if pickleFname is not None and isinstance(pickle, 'str'):
            with open(pickleFname, 'wb') as fp:
                pickle.dump(self, fp)
                print('piezometers were pickled to file <{}>'.format(pickleFname))
                print('to read back use:')
                print('piezometeres = pickle.load({})'.pickleFname)
        return

    def __str__(self):
        return self.meta_to_list().__str__()


    def cleanup(self, verbose=True):
        """Cleanup current Piezometers instance (in place)

        1) The headers in the read Piezometers will be unified and made unique
        2) NaN values will be dropped
        parameters:
        -----------
        verbose: print info when True
        returns:
        --------
        None

        TO 20170410
        """

        self.clean__and_unify_headers() # all actions by reference

        if verbose: print("Removing Piezometer objects without meta info")
        self.data = {k:self.data[k]
            for k in self.data.keys() if len(self.data[k].meta)>0}

        if verbose:
            print("{:4d} non-empty piezometers loaded.".format(len(self.data)))

        if verbose:
            print("Dropping columns with all values NaN")
        for k in self.data.keys():
            self.data[k].data.dropna(axis=1, how='all', inplace=True)
            self.data[k].meta.dropna(axis=1, how='all', inplace=True)
            # Take care that we keep the data column and do not throw
            # away the labels where other columns have NaN
            self.data[k].data.dropna(axis=0, how='any', inplace=True)
            self.data[k].meta.dropna(axis=0, how='any', inplace=True)
            if verbose: print('piezometer DataFrame cleaned')

        return


    def columns(self):
            """returns the union of the column headers of the meta and data dataFrames
            over all Piezometers as a tuple of sets"""

            # set comprehension
            metacols = {col for k in self.data
                            for col in self.data[k].meta.columns}
            datacols = {col for k in self.data
                            for col in self.data[k].data.columns}
            return metacols, datacols



    def get_dinodataometers(self, folder,
                                   verbose=False, pickle_dump=False, nmax=None):
        '''Updates Piezometers by reading them from csv files in directory tree
        The enire directory tree below folder is walked. The groundwater
        head data time series are supposed to be the csv files in the
        subdirectory(s) of a directory "Grondwaterstanden_Put". The latter
        is automatically created when unpacking a zip file containing one or
        more piezometer time series obtained from www.dinoloket.nl.

        parameters:
        -----------
        folder  : top of the directory tree to start the search for the
                  csv files with grondwater head time series obtaied from
                  www.dinoloket.nl.
        returns:
        --------
        nothing. piezoms and failed are returned by reference

        @TO 20170126
        '''
        # recursively walk down the folder hierarchy, looking for dino csv files
        # They must be in a directory "Grondwaterstanden_Put"
        for root, dirs, files in os.walk(folder):
            for d in dirs:
                p = os.path.join(root, d)
                if d == 'Grondwaterstanden_Put':
                    print("Reading folder {}".format(p))
                    self.read_dino_csv_files(p, verbose, nmax)
                else:
                    self.get_dinodataometers(p, verbose, pickle_dump, nmax)

        if verbose:
            print("Total number of pie objects generated  = ", len(self.data))

        if pickle_dump:
            # after having read all the piezometers down topDir pickle them
            print('Dumping all piezometers to file as backup using pickle.dump.')
            with open('piez_bkp.pkl', 'wb') as fbkp:
                pickle.dump(self.data, fbkp)

                print('Making all headers upper case and removing empty columns.')
                for k in self.data:
                    self.data[k].meta.columns = [s.lower()
                            for s in self.data[k].meta.columns]
                    #piez[k].meta = piez[k].meta.dropna(axis=1, how='all')

                    self.data[k].data.columns = [s.lower()
                            for s in self.data[k].data.columns]
                    #piez[k].data = piez[k].data.dropna(axis=1, how='all')

                print('Dumping {:d} piezometers on file using pickle.dump:'.
                      format(len(self.data)))
            with open('piez.pkl', 'wb') as fpz:
                pickle.dump(self.data, fpz)
        return


    def read_dino_csv_files(self, folder, verbose=True, nmax=None):
        """Returns reads piezometer objects from csv files in current folder
           with csv files have been downloaded from www.dinoloket.nl
           parameters:
           -----------
           folder  : directory with csv files of groundwateter times series
           verbose : more info during processing
           nmax    : maximum number of csv file to be read from any folder
           returns:
           --------
           data are returned by reference
        """
        csvfiles = [f for f in os.listdir(folder) if f.endswith('_1.csv')]
        if nmax is not None:
            csvfiles = csvfiles[:nmax]

        if verbose:
            print("processing {:2} csv files in <{}>.".
                  format(len(csvfiles), folder))

        n = 0; m = 0
        for csvname in csvfiles:
            pzname = csvname.split('.')[0][:-2] # cut off _1.csv
            print(pzname)
            fullname = os.path.join(folder, csvname)
            headers   = get_hdr_line_nrs(fullname)
            nrowsMeta = headers[1][0] - headers[0][0] - 1
            if not headers[1][1][2].lower().startswith('peil'):
                print("Can't find PEILDATUM or PEIL DATUM TIJD in file ",fullname)
                break
            else:
                index_col1 = 2

            index_col0 = headers[0][1].index('DATUM MAAIVELD GEMETEN')

            try:
                # meta data
                df0 = pd.read_csv(fullname, header=headers[0][0],
                        index_col=index_col0, nrows=nrowsMeta, parse_dates=True, dayfirst=True)
                # the actual time series
                df1 = pd.read_csv(fullname, header=headers[1][0],
                        index_col=index_col1, parse_dates=True, dayfirst=True)
                # create a new piezom
                self.data[pzname] = Piezometer(pzname, df0, df1)
                n += 1
            except Exception as err:
                self.failed.append((fullname, err))
                print("Error in file. Skipping file:\n {:s}".format(fullname))
                m += 1

        if verbose:
            print("{:d} csv files processed in this directory, {:d} failed".
                  format(n, m))
        return

    def clean__and_unify_headers(self, verbose=False):
        """returns dataometers with cleaned and uniform headers

        clean up by making sure the header texts are unique and the
        same names are used in all data
        """
        not_ok = []
        for k in self.data:
            try:
                # list of meta columns
                Lm = list(self.data[k].meta.columns)
                # list of data columns
                Ld = list(self.data[k].data.columns)
                # clean up meta.columns
                self.data[k].meta.columns =  [h.lower()
                  .replace('(cm t.o.v.','')
                  .replace(')','')
                  .replace(' ','')
                  .replace('datummaaiveldgemeten','date_mv')
                  .replace('meetpunt',            'mp+')
                  .replace('maaiveld',            'mv+')
                  .replace('onderkant',           'bot_')
                  .replace('bovenkant',           'top_')
                  .replace('-coordinaat',         '')
                  .replace('filternummer',        'scr_nr')
                  .replace('locatie',             'location')
                  .replace('externeaanduiding',   'code')
                  .replace('startdatum',          'start')
                  .replace('einddatum',           'end')
                  .replace('filternap','scr_nap')
                  for h in Lm]
                if verbose: print('header of meta cleaned.')
                # clean up data.columns
                self.data[k].data.columns =  [h.lower()
                  .replace('(cm t.o.v.',     '')
                  .replace(')',                   '')
                  .replace('(',                   '')
                  .replace('stand',               '')
                  .replace(' ',                   '')
                  .replace('locatie',             'location')
                  .replace('filternummer',        'scr_nr')
                  .replace('opmerking',           'remark')
                  .replace('bijzonderheid',       'specialty')
                  for h in Ld]
                if verbose: print('header of data cleaned.')
                # convert time strings in meta to datetimes
                self.data[k].meta['start'] = pd.to_datetime(self.data[k].meta['startdatum'], dayfirst=True)
                self.data[k].meta['end']  = pd.to_datetime(self.data[k].meta['einddatum'] , dayfirst=True)
                self.data[k].meta['duration'] = self.data[k].meta['end'] - self.data[k].meta['start']
                if verbose: print('time strings in meta converted.')
            except:
                if verbose: print('somthing not ok in clean_and_unifiy_headers.')
                not_ok.append(k)

        # set columns "datummaaiveldgemeten" to index of meta
        for k in self.data:
            try:
                self.data[k].meta.set_index("start", drop=True, inplace=True)
            except:
                pass

        for k in self.data.keys():
            cols = list(self.data[k].data.columns)
            # shift column headers to the left, drop first
            # this is due to a bug in the csv files, that have one comma
            # less in the header than in the data.
            # This also works after TNO rectifies this error.
            try:
                cols.pop(cols.index('peildatum'))
                cols.append('dummy')
                self.data[k].data.columns = cols
            except:
                # no error, accept cols
                pass

        # convert given columns from cm to m
        for k in self.data:
            for c in self.data[k].meta.columns:
                if c.endswith('nap') or c.startswith('mp'):
                    self.data[k].meta[c] /= 100.
            for c in self.data[k].data.columns:
                if c == 'mv' or c == 'mp' or c == 'nap':
                    self.data[k].data[c] /=100.

        if verbose:
            print('The following piezometers could not be cleaned:')
            print(*not_ok)
        return


    def keys(self):
        return self.data.keys()


    def __getitem__(self, key):
        return self.data[key]


    def plotxy(self, *args, **kwargs):
        """Plots the piezometers on a map (x, y)
        kwargs
        ------
        ax : plt.Axes
            axes to plot on, generated if omitted
        n : int
            maximum number to be plotted, default None
        title : str default for new axes "locations of piezometers"
        xlabel : str
        ylabel : str
        xlim : (xmin, xmax)
            floats
        ylim : (ymin, ymax)
            floats
        grid :[True]|False
            whether to plot grid lines
        rotation : float
            rotation of labels

        """
        points = dict()
        for k in self.keys():
            #try:
            #    mv = self[k].meta['mv+nap'][0]
            #except:
            #    mv = np.NaN
            points[k] = {'x': self[k].meta['x'][0],
                         'y': self[k].meta['y'][0],
                         'name': self[k].name}


        n  = kwargs.pop('n', None)
        fs = kwargs.pop('fs', kwargs.pop('fontsize', 8))
        kwargs['title'] = 'Locations of piezometers'

        ax, kwargs = ax_setup(**kwargs)

        rotation = kwargs.pop('rotation', 45)

        for i, p in enumerate(points.keys()):
            if i == n:
                break

            ax.plot(points[p]['x'], points[p]['y'], '.', **kwargs)
            ax.text(points[p]['x'], points[p]['y'],
                    points[p]['name'], ha='left', va='bottom',
                    rotation=rotation, fontsize=fs)
        return ax


    def plot(self, **kwargs):
        '''plot all _piezometers with standnap in sets of n
        kwargs
        ------
            ax : plt.Axes
                used if in kwargs, else generate in new figure
            key: str
                column to print, if not specified 'nap' is used or 'mp'
                if 'nap' does not exist in self.data.keys
            n : int
                maximum number of piezometers to plot (default = 10)
            title = str
                title fo the graph, default 'head time series'
            xlabel: str
            ylabel: str
            xlim: (datetime, datetime)
            ylim: (float, float)
            grid: [True]|False
        returns
        -------
        ax : plt.Axes

        '''
        n = kwargs.pop('n', 10)

        if not 'ax' in kwargs: kwargs['title'] = 'piezometer head series'

        ax, kwargs = ax_setup(**kwargs)

        for i, p in enumerate(self):
            if i>=n:
                break
            self[p].plot(**kwargs)

        plt.legend(loc='best', fontsize='small')
        plt.show()

    def spans(self, **kwargs):
        """return tiem span of the piezometers, and plot if verbose==True
        """
        verbose = kwargs.pop('verbose', False)

        if verbose:
            kwargs['xlabel'] = 'time'
            kwargs['ylabel'] = 'nr'
            kwargs['title' ] = 'length of groundwater head series'

            ax, kwargs = ax_setup(**kwargs)

            for i, k in enumerate(self.data):
                ax.plot(k.meta[['start', 'end']], [i, i])
            plt.show()
        return [(k, k.meta[['start', 'end']]) for k in self.data]


    def apply(self, fun, verbose=False):
        """store in a dict the median head for all piezometers with column "standnap"""
        return [(k, k.data.fun()) for k in self.data]


    def remove_noscreens(self):
        """"
        remove piezometers without a known screen depth (or more or less safely
        assume they're in the first aquifer)
        """
        klist = []
        for k in self.data:
            if np.all(pd.isnull(self.data[k].meta[['top_scr', 'bot_scr']])):
                del(self.data[k])
                klist.append(k)
        return klist


    def meta_to_list(self):
        """ Join all meta data of all _piezometers in a single DataFrame
        """
        meta = pd.concat([self.data[k].meta for k in self.data])
        return meta


    def get_time_span(self, startDate, endDate):
        """returns piezometer dict that meets the specified time span
        parameters:
        -----------
        _piez : a dictionary with Piezometer objects
        startDate: date before which the series should start
        endDate  : date after which the series should end
        returns:
        --------
        dict with time span startdate, einddate
        """
        return {k:self.data[k] for k in self._piez
                if self.data[k].meta['start']<=startDate &
                   self.data[k].meta['eind' ]>=endDate}


    def get_piez_without_elev(self):
        """returns keys of piezometers in _piez withoutelevation data"""

        klist=[] # list of _piez keys with all nap==nan
        llist=[] # list of _piez keys with at most some nap==nan
        mlist=[] # list of _piez keys that have no columns nap (NAP unknown)
        for k in self.data:
            try:
                # _piezom has this column and all values nan
                if np.all(self.data[k].data['nap'].isnull()):
                    klist.append(k)
                else:
                    # _piezom has at most any values nan, not all
                    llist.append(k)
            except:
                # _piezom  does not have a data column "standnap"
                mlist.append(k)
        print("{:4d} piezometers with all NAP values nan".format(len(klist)))
        print("{:4d} piezometers with at least some not nan".format(len(llist)))
        print("{:4d} piezometers with no column `nap`, i.e. NAP unknown".format(len(mlist)))
        return klist, llist, mlist


    def get_elevation(self):
        """Add elevation to _piez if it has none"""
        klist, llist, mlist = self.get_piez_without_elev()

        needNAP = {k:(self.data[k].x, self.data[k].y) for k in mlist}

        print("Not yet implemented")
        return needNAP


    def toshape(self,shapefilename):
        '''write piezometrs to shapefile which stores meta data
        parameters
        ----------
        shapefileName: str
            name of shapefile to be saved
        returns
        -------
            None
        '''

        wr = sh.Writer(shapeType=sh.POINT)

        # turn keys into list, to keep their order
        keys = list(self.keys())

        # Get the keys common to all piezoms
        common_flds =set(self[keys[0]].meta.columns)
        for k in keys:
            common_flds= common_flds.intersection(self[k].meta.columns)

        # use initial record as example to generate fields for dbffile
        # turn xs of last meta line of first piezoem into a dict to set
        # types of shape fields
        dct  = dict(self[keys[0]].meta.iloc[-1])

        # generate fld as dict containing only fields common to meta of all piezometers
        fld = {f:dct[f] for f in common_flds}

        # add fiels start, end (from piezom.data) and piezom's name attribute
        fld['start'] = self[keys[0]].data.index[0]
        fld['end']   = self[keys[0]].data.index[-1]
        fld['name']  = self[keys[0]].name

        # Generate type code, decimals and size for dbf fields
        fldNms = list(fld.keys()) # use list to keep order of fields
        size = 12 # commong to all fieds (hard wired here)
        for f in fldNms:
            if isinstance(fld[f], float):
                fld[f] = {'fieldType':'N', 'decimals': 3, 'size':size}
            elif isinstance(fld[f], int):
                fld[f] = {'fieldType':'I', 'decimals' :0, 'size':size}
            elif isinstance(fld[f], datetime):
                fld[f] = {'fieldType':'D', 'decimals': 0, 'size':size}
            elif isinstance(fld[f], str):
                fld[f] = {'fieldType':'C', 'decimals': 0, 'size':size}
            else:
                fld[f] = {'fieldType':'C', 'decimals': 0, 'size':size}

            # add ths field to shapefile writer
            wr.field(f, fld[f]['fieldType'],
                        fld[f]['size'],
                        fld[f]['decimals'])

        # geneate record for every piezometer
        for k in keys:
            if not len(self[k]) == 0: # this is the len of self[k].data
                rec = dict(self[k].meta.iloc[-1])
                rec['name']  = self[k].name
                rec['start'] = self[k].data.index[0]
                rec['end']   = self[k].data.index[-1]

                wr.point(rec['x'], rec['y'])
                flds = []
                for f in fldNms: # fldNms only has fields common to all piezoms
                    flds.append(rec[f])
                wr.record(*flds)

        wr.save(shapefilename)
        return


    def get_borings(self, boreDir):
        ''' return list of borholes in boreDir corresponding to Piezometers
        parameters
        ----------
        boreDir : str
            path to directory which contains the files ending at ...1,4.xml
            downloade from dinoloket. These files contain the borehole
            descriptioins that correspond the these boreholes and can
            be plotted by functions in ...
        returns
        -------
        list of file names in boreDir corresponding to the time series in
        Piezometers (self)
        '''
        piezNames = [self[k].name[:-3] for k in self]
        boreNames = [f[:-8] for f in os.listdir(boreDir) if f.endswith('1.4.xml')]
        piezNames = set(piezNames)
        boreNames = set(boreNames)
        return piezNames.intersection(boreNames)


def get_hdr_line_nrs(csvfname,lookfor='locatie', number_of_times=2):
    '''returns linenumbers of header in the dino-loket piezometer csv file.

       This is necessary to readout the file using pd.read_csv()
       There are two header lines that both start with Locatie. Locatie
       may be in upper or lower case.
       Find those lines and return a list with the two numbers or
       an empyt list if not found.
       parameters:
       -----------
       csvfname: path to csv file containing piezometer time series from Dino
       returns:
       --------
       list [firstHeaderLine, secondHeaderLine, nrOfLinesToReadAfterFirst]

       @TO 20170126
    '''

    lineNr= 0
    headers=[]
    with open(csvfname, 'r') as fd:
        for line in fd:
            if line.lower().startswith(lookfor):
                headers.append([lineNr,
                        [hdr for hdr in line.upper().split(',')]
                         ])
                if len(headers)==2:
                    # append number of lines to read from Dino file read
                    # after first header
                    return headers
                    break
            if len(line) > 1: # empty lines don't count in pd.read_csv
                lineNr += 1
    return headers


if __name__ == "__main__":
    """Read data files from www.dinoloket.nl in Piezometer instances

    this examle and test shows how
    to generate a Piezometers instance (from dinoloket files)
    plot locations
    select only those piezometers whose measurements overlap period t1, t2
    plot locations
    plot time series
    given a directory with boreholes from dinoloekt
        get the locations for which there is a matching borehole
    generate a Piezometers object with only those piezometers for which
        we have a borehole lithology descripton
    plot locations of these piezometers
    plot graphs of these piezoemeters
    generate shapefile for these piezometers

    TO 171219
    """

    topDir = '/Users/Theo/GRWMODELS/python/td_analyse_WHK/data'
    topDir = './Veluwe_Eemvallei'

    # generate piezometers from the directory
    # With pickle dump true, the piezometers are pickled to file
    piezoms = Piezometers(topDir, pickle_dump=False, verbose=True, nmax=1000)

    print(piezoms.failed)

    piezoms.plotxy()

    piezoms.plot(n=30)

    # select the piezometer with data overlapping t1 to t2
    piez2 = Piezometers()  # empty Piezometeres
    t1 = datetime(1990, 1, 1) # start
    t2 = datetime(2000, 1, 1) # end
    for k in piezoms:
        try:
            if piezoms[k].data.index[0] <= t1 and piezoms[k].data.index[-1] >= t2:
                piez2[k] = piezoms[k]
        except:
            pass
    piez2.plotxy(rotation=30, fs=8)

    piezoms.toshape('myshape')  # write to shapefile

    # Given a directory with bore files from dinoloekt (*_1.4.xml)
    boreDir = 'boringenVanDinoloket/Boormonsterprofiel/'
    # Get the bore names that coincide with these piezom locations
    common = piez2.get_borings(boreDir)

    # Gegerate a new piezoms set with just those for which we have a boring
    piez3 = Piezometers()
    for k in piez2.keys():
        if piez2[k].meta['location'][0] in common:
            piez3[k] = piez2[k]

    piez3.toshape('piez3')  # generate a shape file
    piez3.plotxy()          # plot locations
    piez3.plot()            # plot curves



