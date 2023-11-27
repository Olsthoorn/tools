import numpy as np
from warnings import warn

def contourRange(H=None, NMax=25, Lay=None, fldname1='values', index=None):
    """Return a suitable range of values for contouring.

    The function tries to prevent unregular contour levels.
    You will not have exactly NMax contour values, but their number will bebetween 4 and NMax.
    
    USAGE:
      range = ContourRange(H, NMax, Lay, fldname1, index) 

    Parameters
    ----------
    H: may be a dict with a field holding 3D values, such as that
      obtained from readData or readMT3D.
      
    H may a 3D array. In that case the fldname1 and index should not be specified.

    NMax: int. (NMin, NMax) is allowed instead of NMax to limit the range of contours.
        set the number of contour levels.
    fldname: str
        The fldname/key must be specified if it is not the default: 'values' as
        is the case with the results of readBud and readBud6.
    index: int
        The index must be specified if the fldname1 refers to a dict holding
        more than a single 3D array, such as is the case for the field 'term'
        obtained from readBud and readBud6.

    USAGE:
     range = ContourRange(B, 'Psi')
     range = ContourRange(B, N, 'Psi')
     range = ContourRange(B, N, iLay, 'Psi')
            where N the number of desired contour values,
            iLay the layer number and
            'Psi' the field name.
     range = ContourRange(B,   'term')
     range = ContourRange(B,   'term', index)
     range = ContourRange(B, N,'term', index)
     range = ContourRange(B, N, iLay, 'term', index)
            same as before, but the budget struct B has many term cells each being
            a complete 3D array. In thata case, add the index as last argument so
            that the ContourRange will be based on B(:).term{index}(:,:,:).
     range = ContourRange(H)
     range = ContourRange(H, N)
     range = ContourRange(H, N, Lay)
     range = ContourRange({H, h0}, N); -- first subtract h0 from H
     range = ContourRange({H, h0}, N, iLay); -- first subtract h0 from H

     Example:
          A = peaks
          crange = ContourRange(A, 50)
          plt.contour(peaks, crange)
          plt.colorbar

     See also READDAT, READBUD, READMT3D, CONTOUR.

     TO 100529 110319 130221 130322

    #    Copyright 1999-2013 Theo Olsthoorn, TUDelft, Waternet 
    #    $Revision: 1 $  $Date: 2007/11/13 00:10:21 $
    """
    pass
    
    # %% Input arguments
    if H is None:
        raise ValueError('ContourRange: Not enough input data\n')

    if not isinstance(H, (dict, np.ndarray)):
        raise ValueError('H must be a dict or 3D np.ndarray, not {}'.format(type(H)))

    # %% Determine NMax and NMin
    if isinstance(NMax, int):
        NMin = 10
    elif isinstance(NMax, (tuple, list, np.ndarray)):
        NMin, NMax = NMax
    if NMin > NMax:
        NMin, NMax = NMax, NMin
    NMax = min(NMax, 40)
    NMin = max(4, NMin)

    # %% Use all layers or only specified ones
    if not 'Lay':
        if isinstance(H, dict):
            if isinstance(H[0], dict):
                Lay = np.arange(H[0][0][fldname1].shape[1])
            else:
                Lay = np.arange(H[0].shape[1])    
        else:
            if isinstance(H, dict):
                Lay = np.arange(H[0][fldname1][1])
            else:
                Lay = np.arange(H.shape[1])

    # %% If difference is desired:
    # in case H = [Var1, Var2] Var2 will be subtracted from Var1 where H is a
    # dict with field values like from readMT3D and Var2 an idential struct:
    # or an array what will be subtracted.::
    if isinstance(H, (tuple, list)):
        if isinstance(H[0], dict):
            if isinstance(H[1], dcit):
                try: 
                    Nt1 = len(H[0])
                    Nt2 = len(H[1])
                    for it in range(len(H[0])):
                        H[0][it][fldname1] = H[0][max(it, Nt1)][fldname1] - H[1][max(it, Nt2)][fldname2]
                except ME:
                    raise ValueError('{}: both dict in first argument must have compatible range'.format(ME.message))            
            elif isinstance(H[1], np.ndarray):
                try:
                    for it in range(len(H[0])):
                        H[0][it][fldname1] = H[0][it][fldname1] - H[1]            
                except ME:
                    print(ME.message)                
                    raise ValueError('{}: both struct and array in first argument have compatible range'.format(ME.message))
            else:
                raise ValueError('''{}: The first argument is a cell, it should have two compatible
                        structs and/or 3D numerical arrays.'''.format(mfilename))
        else:
            if isinstance(H[1], dict):
                try:
                    Nt2 = len(H[1])
                    for it in range(len(H[1])):
                        H[1][it][fldname2] = H - H[1][max(it , Nt2)][fldname2]
                except ME:
                    printf(ME.message)
                    raise ValueError('{}: array and struct in first argument have compatible range'.format(ME.message))
            elif isinstance(H[1], np.ndarray):
                try:
                    H[0] -= H[1]
                except ME:
                    raise ValueError('{}: both arrays in first argument have compatible range'.format(ME.message))
            else:
                raise ValueError('''{}: The first argument is a cell, it should have two compatible
                        structs and/or 3D numerical arrays.'''.format(mfilename))
        H = H[0]
    # %% Determine type of input
    if isinsrance(H, dict):
        m, M = +np.inf, -np.inf
        for i in range(len(H)):
            if isinstance(H[i][fldname1], dict):
                m = min(m, min( min( min(H[i][fldname1][index][:,:,Lay]))))
                M = max(M, max( max( max(H[i][fldname1][index][:,:,Lay]))))
            else:
                m = min(m, min( min( min(H[i][fldname1][:,:,Lay]))))
                M = max(M, max( max( max(H[i][fldname1][:,:,Lay]))))
    else:
        m = min( min( min(H[:,:,Lay])))
        M = max( max( max(H[:,:,Lay])))

    mm, MM =m, M

    # Compute contour range
    if m < 0:
        m = -10 ** np.round(np.log10(-m * 3), decimals=0)
    elif m > 0:
        m = +10 ** np.round(np.log10(+m / 3), decimals=0)
    else:
        m = 0

    if M<0:
        M = -10 ** np.round(np.log10(-M / 3), decimals=0)
    elif M > 0:
        M = +10 ** np.round(np.log10(+M * 3), decimals=0)
    else:
        M = 0

    dH    = 10 ** round(log10((M - m) / NMax))

    f = np.floor((M - m) / (MM - mm)) # always > 1
    
    if     f >= 5:   f = 5
    elif   f >= 2.5: f = 2.5
    elif   f >= 2  : f = 2
    else:  f = 1

    dH = (M - m) / NMax / f

    contour_range = np.arange(m, M + dH, dH)
    contour_range = contour_range[np.logical_and(contour_range >= mm, contour_range <= MM)]

    if not contour_range:
        warn(
        '''{}: Your values range for field <<{}>> is empty.\n',...
        'Hence, there are no contours to plot!\n',...
        'Most probably your computed data are uniform. Please check and resolve this.\n',...
        'You may more easily track the error after switching on\n',...
        'Debug>Stopif Errors/Warnings>always stop if error\n',...
        'from the menu bar of the editor, and then run again.'''.format(mfilename, fldname1))

    return contour_range
    
    
def get_orderly_levels(lo, hi, n):
    """Return a float array of orderly levels for contouring.
    
    Parameters
    ----------
    hi: float
        high values (max value)
    lo: float
        low value (mean value)
    n:  int
        desired (approximate) number of levels
        
    Returns
    -------
    Given a maximum and minimum values of a data set, the function
    returns an array of n orderly values that include the hi and lo
    values and which is suitable for a set of contouring levels.
    
    >>>tests = [[0, 45], [0, 123], [-45, 201], [-1, 1], [0.01, 0.6], [-0.009, 0.004], [-0.8, 0.9], [-335, -12]]
    >>>for test in tests:
    >>>   print('test: ({}) --> ticks: {}'.format(test, get_orderly_levels(*test, 10)))
    test: ([0, 45]) --> ticks: [ 0.  5. 10. 15. 20. 25. 30. 35. 40. 45. 50.]
    test: ([0, 123]) --> ticks: [  0.  20.  40.  60.  80. 100. 120. 140. 160. 180. 200.]
    test: ([-45, 201]) --> ticks: [-100.  -60.  -20.   20.   60.  100.  140.  180.  220.  260.  300.]
    test: ([-1, 1]) --> ticks: [-1.  -0.8 -0.6 -0.4 -0.2 -0.   0.2  0.4  0.6  0.8  1. ]
    test: ([0.01, 0.6]) --> ticks: [0.   0.06 0.12 0.18 0.24 0.3  0.36 0.42 0.48 0.54 0.6 ]
    test: ([-0.009, 0.004]) --> ticks: [-0.01  -0.008 -0.006 -0.004 -0.002  0.     0.002  0.004  0.006  0.008
      0.01 ]
    test: ([-0.8, 0.9]) --> ticks: [-1.  -0.8 -0.6 -0.4 -0.2 -0.   0.2  0.4  0.6  0.8  1. ]
    test: ([-335, -12]) --> ticks: [-400. -360. -320. -280. -240. -200. -160. -120.  -80.  -40.    0.]
    """
    s = 10 ** (np.floor(np.log10(hi - lo)))
    start = s * np.floor(lo / s)
    end   = s * np.ceil( hi / s)
    s = (end - start) / n
    return np.round(np.arange(start, end + s, s), decimals=6)
    
if __name__ == '__main__':
    print('Hello')
    
    tests = [[0, 45], [0, 123], [-45, 201], [-1, 1], [0.01, 0.6], [-0.009, 0.004], [-0.8, 0.9], [-335, -12]]
    for test in tests:
       print('test: ({}) --> ticks: {}'.format(test, get_orderly_levels(*test, 10)))
