# REead the i, x, y, dx, dy data from the xml-file exported by plotdigitzer app into recarray

import re
import numpy as np
import os
import matplotlib.pyplot as plt

dtype  = np.dtype([('i', int), ('x', float), ('y', float), ('dx', float), ('dy', float)])
dtype2 = np.dtype([('i', int),
                   ('x', float),  ( 'y', float),
                   ('dx', float), ('dy', float),
                   ('xw', float), ('yw', float)])

def fromPlotdigitizerXML(xml_fname, verbose=False):
    """Read xml data exported from plotdigitizerXML into recarray with [i, x, y, dx, dy].
    
    plotdigitizer is a free app (sourceforge). The free app can only export xml.
    
    Parameters
    ----------
    xml_filename : str
        path to xml file (.xml will be automatically)
    verbose: bool
        show lines read and intermediate results of RE
        
    @TO 20231223
    """
    
    if not xml_fname.endswith('.xml'):
        xml_fname += '.xml'
    assert os.path.isfile(
        xml_fname), "{} not found (should end with '.xml (lowercase)')".format(xml_fname)
    
    re_int  =  "'(\d+)'"
    re_float = "'([+-]*\d+\.\d+)'"
    
    re_point = re.compile(r"^<point.+n={0} x={1} y={1} dx={1} dy={1}".format(re_int, re_float))
    
    xy = [] # build list
    meta = {}
    with open(xml_fname) as f:
        for line in f:
            if line.startswith('<point'):
                if verbose: print(line)
                mo = re_point.search(line)
                if mo:
                    if verbose: print(mo)
                    xy.append((int(mo[1]), float(mo[2]), float(mo[3]), float(mo[4]), float(mo[5])))
                    if verbose: print(xy[-1])
                print(xy[-1])
            elif line.startswith('<image'):
                regex = re.compile("'(.+)'")
                mo = regex.search(line)
                meta['image_name'] = mo[1]
            elif line.startswith('<axesnames'):
                regex = re.compile("x='(.+)'\W+y='(.+)'")
                mo = regex.search(line)
                meta['xlabel'], meta['ylabel'] = mo[1], mo[2]
            elif line.startswith('<calibpoints'):
                s = r"minXaxisX={0} minXaxisY={0} maxXaxisX={0} maxXaxisY={0} minYaxisX={0} minYaxisY={0} maxYaxisX={0} maxYaxisY={0} aX1={0} aX2={0} aY1={0} aY2={0}".format(re_float)
                regex = re.compile(s)
                mo = regex.search(line)
                for i, p in zip([1, 2, 3, 4],
                                ['minXaxisX', 'minXaxisY', 'maxXaxisX', 'maxXaxisY']):
                    meta[p] = float(mo[i])
                for i, p in zip([5, 6, 7, 8],
                                ['minYaxisX', 'minYaxisY', 'maxYaxisX', 'maxYaxisY']):
                    meta[p] = float(mo[i])
                for i, p in zip([9, 10, 11, 12],
                                ['aX1', 'aX2', 'aY1', 'aY2']):
                    meta[p] = float(mo[i])
                print(meta)
            else:
                pass
                
        data = np.array(xy, dtype=dtype)
                
        data = converPxls2XY(data, meta)
    
    return data, meta
     
def converPxls2XY(data, meta):
        
        A = np.array([
            [meta['maxXaxisX'] - meta['minXaxisX'], -meta['maxYaxisX'] + meta['minYaxisX']],
            [meta['maxXaxisY'] - meta['minXaxisY'], -meta['maxYaxisY'] + meta['minYaxisY']]])
        
        B = np.array([[meta['minYaxisX'] - meta['minXaxisX']],
                      [meta['minYaxisY'] - meta['minXaxisY']]])
        
        # solve for lam and mu to get the interesetion of the two ax  vectors
        lam, mu = np.linalg.solve(A, B).T[0]
        
        # compute the intersection (pixels)
        
        # method 1 using lam
        C1 = np.array([[meta['minXaxisX']],
                      [meta['minYaxisY']]])         
        D1 = np.array([[meta['maxXaxisX'] - meta['minXaxisX']],
                       [meta['maxXaxisY'] - meta['minXaxisY']]])
        
        x01, y01 = (C1 + lam * D1).T[0]
        
        # method 2 using mu
        C2 = np.array([[meta['minYaxisX']],
                       [meta['minYaxisY']]])
        D2 = np.array([[meta['maxYaxisX'] - meta['minYaxisX']],
                       [meta['maxYaxisY'] - meta['minYaxisY']]])
        
        x02, y02 = (C2 + mu  * D2).T[0]
        
        assert np.isclose(x01, x02) and np.isclose(y01, y02), "x01 and x02 or y01 and y02 are not close!"
        
        # Save the compute intersection (pixel coordinates)
        meta['x0'], meta['y0'] = x01, y01

        # Compute the world coordinates of the  intersection        
        meta['xw0'] = meta['aX1'] + lam * (meta['aX2'] - meta['aX1'])
        meta['yw0'] = meta['aY1'] + lam * (meta['aY2'] - meta['aY1'])
        
        # compute the coordinates of a pixel using the compute pixel center of the axes
        # this give a new lam and mu
        
        Einv = np.linalg.inv(
            np.array([[meta['maxXaxisX']-meta['x0'], meta['maxYaxisX'] - meta['x0']],
                     [meta['maxXaxisY']-meta['y0'], meta['maxYaxisY'] - meta['y0']]])
        )

        data2 = np.zeros(len(data), dtype = dtype2)
        for fld in data.dtype.names:
            data2[fld] = data[fld]

        for i, (x, y) in enumerate(zip(data2['x'], data2['y'])):

            F = np.array([[x - meta['x0']],
                          [y - meta['y0']]])
        
            lam, mu = (Einv @ F).T[0]
        
            data2[i]['xw'] = meta['xw0'] + lam * (meta['aX2'] - meta['xw0'])
            data2[i]['yw'] = meta['yw0'] + mu  * (meta['aY2'] - meta['yw0'])
            
        return data2
     
            
if __name__ == '__main__':
    
    xml_fname = os.path.join('/Users/Theo/GRWMODELS/python/tools/coords/data',
                         'MoervaarDepressieDekzandrugMaldegemStekene')
    
    data, meta = fromPlotdigitizerXML(xml_fname + '.xml')
    
    fig, axs = plt.subplots(1, 2)
    fig.set_size_inches(14, 6)
    
    axs[0].set_title(meta['image_name'])
    axs[0].set_xlabel(meta['xlabel'] + ' pixels')
    axs[0].set_ylabel(meta['ylabel'] + ' pixels')
    axs[0].grid(True)
    axs[0].plot(data['x'], data['y'], label='world coordinates')

    
    axs[1].set_title(meta['image_name'])
    axs[1].set_xlabel(meta['xlabel'])
    axs[1].set_ylabel(meta['ylabel'])
    axs[1].grid(True)
    axs[1].plot(data['xw'], data['yw'], label='world coordinates')
    
    plt.show()
    
    print(data, meta)
         
            