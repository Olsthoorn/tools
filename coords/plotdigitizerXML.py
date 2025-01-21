# REead the i, x, y, dx, dy data from the xml-file exported by plotdigitzer app into recarray

import re
import numpy as np
import os
import matplotlib.pyplot as plt
from etc import newfig

dtype  = np.dtype([('i', int), ('xp', float), ('yp', float), ('xw', float), ('yw', float)])

def fromPlotdigitizerXML(xml_fname, verbose=False, chunkit=False, **kwargs):
    """Read xml data exported from plotdigitizerXML into recarray with [i, x, y, dx, dy].
    
    Notice: plotdigitizer is a free app (sourceforge). The free app can only export xml.
    
    Parameters
    ----------
    xml_filename : str
        path to xml file (.xml will be automatically)
    verbose: bool
        show lines read and intermediate results of RE
        
    @TO 20231223, 20250121
    """
    
    if not xml_fname.endswith('.xml'):
        xml_fname += '.xml'
    assert os.path.isfile(
        xml_fname), "{} not found (should end with '.xml (lowercase)')".format(xml_fname)
    
    re_int  =  r"'(\d+)'"
    re_float = r"'([+-]*\d+\.\d+)'"
    
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
                if verbose:
                    print(xy[-1])
            elif line.startswith('<image'):
                regex = re.compile("'(.+)'")
                mo = regex.search(line)
                meta['image_name'] = mo[1]
            elif line.startswith('<axesnames'):
                regex = re.compile(r"x='(.+)'\W+y='(.+)'")
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
                if verbose:
                    print(meta)
            else:
                pass
                
        data = np.array(xy, dtype=dtype)
        
        if chunkit:
            data = chunk_data(data, **kwargs)
    
    return data, meta

def chunk_data(data, fault_x=None, fault_width=100., xjump_min=0.):
    """Return list of lines, by splitting data based on xjump < -abs(xjump_min).
    
    The data are one stream of points are assumed to represent layer elevations.
    After each elevation, the xjump (x[n+1] - x[n]) will be negative because the
    stream of points jumps back to the start of the next elevation.
    To also deal with possible faults, xjump_min is set to some threshold value to
    prevent points at the fault to inadvertently signal a back jump.
    
    Parameters
    ----------
    data: recarray with dtype containing field ('xw', float)
        data to be split in chunks based on back-jump of xw coordinates
    fault_x: float or sequence of x values, one per vertical fault
        positon of vertical fault[s]
    fault_width: float, default 100. m
        width of fault (so reset x coordinates to exact fault location)
    xjump_min:
        minimum back jump [default = 0]
    """
    # set faults first
    if fault_x:
        if np.isscalar(fault_x):
            fault_x = [fault_x]
        for fx in fault_x:
            L = L = np.logical_and(data['xw'] > fx - fault_width / 2, data['xw'] < fx + fault_width / 2)
            data['xw'][L] = fx
    xjump = np.diff(data['xw'])
    chunks = []
    iFr = np.hstack((0, np.where(xjump < -abs(xjump_min))[0] + 1))
    iTo = np.hstack((iFr[1:], len(data)))
    for i1, i2 in zip(iFr, iTo):
        chunks.append(data[i1:i2])
    return chunks

def get_layertops(data, meta, dx=1.0, xm=False):
    """Return top of layers with a resolution dx.
    
    Notice that the layer tops will be at the xm locations.
    """
    xL, xR = meta['aX1'], meta['aX2']
    if isinstance(data, np.ndarray):
        data = chunk_data(data)        

    x = np.arange(xL, xR + dx, dx)    
    xm = 0.5 * (x[:-1] + x[1:])

    Z = np.zeros((len(data), len(xm)))    
    for iz, dat, in enumerate(data):
        Z[iz, :] = np.interp(xm, dat['xw'], dat['yw']) 
        if iz == 0:
            continue
        else:
            Z[iz] = np.fmin(Z[iz], Z[iz-1])
    return x, Z

def bridge_fault(Z, x=None, fault_x=None, verbose=True):
    """Return layer elevatrions that bridge the fault"""
    
    # note that Z.shape[-1] must be x.shape - 1, because Z are cell centers and the fault
    # is between to cells.
    assert fault_x is not None, "fault_x must be a float (x-value_), not None!"
    assert len(x) == Z.shape[-1] + 1, f"len(x)={len(x)} must equal Z.shape[-1] + 1={Z.shape[-1] + 1}"
    xm = 0.5 * (x[:-1] + x[1:])
    ixL, ixR = np.where(xm < fault_x)[0][-1], np.where(xm > fault_x)[0][0]
    xL, xR = xm[:ixL + 1], xm[ixR:]
    
    # z immediately left and right of fault
    zL, zR = Z[:, ixL], Z[:, ixR]
    
    # all Z to the left and right of fault
    ZL, ZR = Z[:, :ixL + 1], Z[:, ixR:]
    Iz_orig = np.arange(len(Z), dtype=float)
    
    # Layer indices floats, meaning fraction is in layer int(index)
    IzL = np.sort(np.hstack((Iz_orig, np.interp(-zR, -zL, Iz_orig))))[1:-1]
    IzR = np.sort(np.hstack((Iz_orig, np.interp(-zL, -zR, Iz_orig))))[1:-1]
    
    # Remove double at top and bottom
    IzL[[0, -1]] = [0, len(Z) - 1]
    IzR[[0, -1]] = [0, len(Z) - 1]
       
    # Create new Z array to the left and right of fault
    # bridging the fault by matching all layer left and right, making them ongoing.
    ZLnew = np.zeros((len(IzL), len(xL)))    
    for ilay, iz in enumerate(IzL):
        if iz == IzL[-1]:
            ZLnew[ilay] = ZL[int(iz)]
        else:
            ZLnew[ilay] = ZL[int(iz)] + (iz - int(iz)) * (ZL[int(iz) + 1] - ZL[int(iz)])
            
    ZRnew = np.zeros((len(IzR), len(xR)))    
    for ilay, iz in enumerate(IzR):
        if iz == IzR[-1]:
            ZRnew[ilay] = ZR[int(iz)]
        else:            
            ZRnew[ilay] = ZR[int(iz)] + (iz - int(iz)) * (ZR[int(iz) + 1] - ZR[int(iz)])
    
    if verbose:
        ax = newfig("New layers, bridging the fault", "x", "elevation")

        clrs = 'rbgkmcy' * 5
        for z, iz in zip(ZLnew, IzL):
            clr = clrs[int(iz)]
            lw = 2 if iz == int(iz) else 1
            ls = '-' if iz == int(iz) else '--' 
            ax.plot(xL, z, color=clr, ls=ls, lw=lw)
        for z, iz in zip(ZRnew, IzR):
            clr = clrs[int(iz)]
            lw = 2 if iz == int(iz) else 1
            ls = '-' if iz == int(iz) else '--' 
            ax.plot(xR, z, color=clr, ls=ls, lw=lw)

        ax.plot(xm[ixL] * np.ones(len(Z)), Z[:, ixL], 'k')
        ax.plot(xm[ixR] * np.ones(len(Z)), Z[:, ixR], 'k')

        print("IL: ", IzL)
        print("IR: ", IzR)
    
    # Property indices for each layer (or mother layer index)
    IzL = [int(i) for i in IzL]
    IzR = [int(i) for i in IzR]
    
    fault_ix = ixL + 1 # index of x (not xm) of fault
    Znew = np.hstack((ZLnew, ZRnew))
    return Znew, IzL, IzR, fault_ix
    
    

if __name__ == '__main__':
    
    dataPath = '/Users/Theo/GRWMODELS/python/tools/coords/data'
    
    fnames = [
              'digitize_test1.xml',
              'digitize_test2.xml',
              'digitize_test3.xml',
              'digitize_test4.xml',
              'BRO REGIS II  Verticale doorsnede 125380_401150.png.xml',
              'MoervaarDepressieDekzandrugMaldegemStekene.xml',
    ]
    
    faults_x = [None, None, 550, 550, 4190, None]
    
    for k, (fname, fault_x) in enumerate(zip(fnames, faults_x)):
        pname = os.path.join(dataPath, fname)
        
        if fname == fnames[0]:
            data, meta = fromPlotdigitizerXML(pname, verbose=False, chunkit=False)

            ax = newfig(fname + ', pixels', 'xp', 'yp')
            ax.invert_yaxis()
            ax.plot(data['xp'], data['yp'], label='pixels')
            ax.legend()

            ax = newfig(fname + ', wrold coordinates', 'xw', 'yw')
            ax.plot(data['xw'], data['yw'], label='world coordinates')
            ax.legend()
        else:
            kwargs = {'fault_x': fault_x, 'fault_width': 200., 'xjump_min': 100}
            data, meta = fromPlotdigitizerXML(pname, chunkit=True, **kwargs)
        
            x, Z = get_layertops(data, meta, dx=1.)
            xm = 0.5 * (x[:-1] + x[1:])
        
            ax = newfig(fname + ', world coordinates', 'xw', 'elevation')
            for z in Z:
                ax.plot(xm, z)
            
            if k in [2, 3, 4]:                
                Znew, IzL, IzR, fault_ix = bridge_fault(Z, x, fault_x=fault_x, verbose=True)

    print("__name__ = ", __name__)
    print("__file__ = ", __file__)
    
    plt.show()

         
            