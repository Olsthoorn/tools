#!/usr/bin/env python3

# https://www.vbforums.com/showthread.php?481546-Parametric-Cubic-Spline-Tutorial

# @ TO 20220323

import matplotlib.pyplot as plt
import numpy as np

# %% example 1
s = np.linspace(0, 1, 101)

x = 26 * s ** 3 - 40 * s ** 2 + 15 * s - 1.0
y = -4 * s ** 2 + 3 * s

plt.plot(x, y)
plt.set_aspect(1.0)
plt.show

# %% Define newfig for easily creating new plots
def newfig(title="", xlabel="", ylabel="", xlim=None, ylim=None, aspect=None):
    _, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    if aspect: ax.set_aspect(aspect)
    ax.grid(True)
    return ax

# %% Getting hands dirty
ctr = np.array([(3 , 1), (2.5, 4), (0, 1), (-2.5, 4),(-3, 0), (-2.5, -4), (0, -1), (2.5, -4), (3, -1)])

ax = newfig("controle points", "x", "y")
ax.plot(*ctr.T)

# %% A. Drawing the B-spline Curve

from scipy.interpolate import splev
x, y = ctr.T
l = len(x)
t = np.linspace(0, 1, l-2)
t = np.hstack((0, 0, 0, t, 1, 1, 1))
tck=[t,[x,y],3]

u3=np.linspace(0,1,(max(l*2,70)),endpoint=True)

out = splev(u3,tck) 

ax = newfig("fitted B-spline curve", "", "",
            xlim=(min(x) - 1, max(x) + 1),
            ylim=(min(y) - 1, max(y) + 1))
ax.plot(*ctr.T, "--", marker='o', mfc='r', label="Control polygon")
ax.plot(out[0], out[1], 'b', linewidth=2.0, label='B-spline curve')
ax.legend()
plt.show()

# %%  B. interpolate the B-spline Curve
from scipy.interpolate import splprep

ctr = np.array([(3 , 1), (2.5, 4), (0, 1), (-2.5, 4),(-3, 0), (-2.5, -4), (0, -1), (2.5, -4), (3, -1)])

x, y = ctr.T

tck, u = splprep([x, y], k=3, s=0)

u=np.linspace(0, 1, 50)

out = splev(u,tck)

ax = newfig("fitted B-spline curve", "", "",
            xlim=(min(x) - 1, max(x) + 1),
            ylim=(min(y) - 1, max(y) + 1))
ax.plot(x, y, marker='o', mfc='r', label="Pi points")
ax.plot(out[0], out[1], 'b', linewidth=2.0, label='B-spline')
ax.legend()
plt.show()

# %%
import numpy as np
from scipy import interpolate

import matplotlib.pyplot as plt

#x = np.arange(0, 2*np.pi+np.pi/4, 2*np.pi/8)
#y = np.sin(x)

ctr =np.array( [(3 , 1), (2.5, 4), (0, 1), (-2.5, 4),
                (-3, 0), (-2.5, -4), (0, -1), (2.5, -4), (3, -1)])

x, y = ctr.T

tck, _ = interpolate.splprep([x,y],k=3,s=0)

u = np.linspace(0, 1, 50)
out = interpolate.splev(u,tck)

plt.figure()
plt.plot(x, y, 'ro', out[0], out[1], 'b')
plt.legend(['Points', 'Interpolated B-spline', 'True'],loc='best')
plt.axis([min(x)-1, max(x)+1, min(y)-1, max(y)+1])
plt.title('B-Spline interpolation')
plt.show()
# %%

def interp_spline(ctr, n):
    """Return a B-spline through the control points.
    
    Parameters
    ----------
    ctr: ndarray [m, 2]
        the control points.
    n: int
        number of evaluation point.
        
    Returns 
    -------
    out: ndaray (n, 2)
        points of the spline
    """
    import numpy as np
    from scipy.interpolate import splprep, splev

    tck, _ = splprep(ctr, k=3, s=0)

    u = np.linspace(0, 1, n)
    return splev(u,tck) 

ax = newfig("B-Spline interpolation", "x", "y",
            xlim=(min(x)-1, max(x)+1), ylim=(min(y)-1, max(y)+1))
plt.plot(*ctr.T, 'ro', label='control points')
plt.plot(out[0], out[1], 'b', label='interpolated B-spline')
plt.legend()
plt.show()
# %%
