"""Use of ginput to get coordinates from the screen image.

Spyder Editor


@TO 210116
"""

#%% generating a movie by grabbing frames
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import ginput
import scipy.linalg as la
from scipy.interpolate import interp1d
import os
import pdb

def convert(x=None, y=None, points=None):
    """Return points in the y domain given points in the from domain.

    This is an affine mapping of the from_domain to the to_domain.
    specify the three points of the to_domain that are going to be digitzed first.
    Then digitize the coordinates of the from_domain starting with the three that
    will be used for georeferencing.

    Then call with x the x the three reference coordinate pairss of the from domain
    and y the three reference coordinate pairs from the from_domain and points
    the rest of the digitized points in the from_domain.
    Returns the points mapped to the to_domain

    Conversion is done by Afine transformation. All points are like [[x1, y1], [x2, y2], ...]

    Parameters
    ----------
    x are three points in the from domain (like screen coordinates captured
                                           using ginput())
    y are three points in the to domain (world coordinates pertaining
                                         to the three previous points)
    points are points in the from domain that are to be converted (mapped)
    """
    x, y, points = np.array(x), np.array(y), np.array(points)
    X = np.vstack((x.T, np.ones((1, len(x)))))
    Y = np.vstack((y.T, np.ones((1, len(y)))))
    P = np.vstack((points.T, np.ones((1, len(points)))))
    M = Y @ la.inv(X)
    return (M @ P)[:2].T

def interpolate(x=None, points=None, kind='cubic'):
    """Return interpolated values given the x-axis points.

    Parameters
    ----------
    x: ndarray
        values at which interpolation is desired
    points: ndarray (n, 2) with given values
        data valurs
    """
    # Get the interpolator using method kind
    f = interp1d(*Pnts.T, kind=kind, bounds_error=False, fill_value='extrapolate')
    return np.vstack((x, f(x))).T


os.chdir('/Users/Theo/Instituten-Groepen-Overleggen/IHE/Syllabus/Syllabus2020/')

# The image from which to digitize the coordinates
fig = plt.figure(figsize=(20,30))
img=mpimg.imread('KalahariDeVries1984.png')
imgplot = plt.imshow(img)

# digitize the points, used backspace to delete and return to finish
pts = ginput(n=-1, timeout=-1)

# targed domain reference vectors (points)
y = np.array([[0, 0], [0, 450],  [600, 0]])


# First 3 points are the coordinates of the georeference points
# The rest are the data points
Pnts = convert(x=pts[:3], y=y, points=pts[3:])

fig, ax = plt.subplots()
fig.set_size_inches(12, 8)

ax.plot(*y.T, 'o', label="orginal points")
ax.plot(*Pnts.T, '.', label='converted points')
xp = np.linspace(0, 600)

# Interpolate between the digitized points to get a smooth curve
x = np.linspace(0, 650, 66)

kind='cubic' # Many alternatives, like linear squared

ax.plot(*interpolate(x, Pnts, kind=kind).T, '.-', label='interpolated, kind={}'.format(kind))

ax.legend()
