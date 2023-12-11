# Dealing with photo's and videos for mflab

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

3 @TO 231123

# Stinkbug URL = 'https://raw.githubusercontent.com/matplotlib/matplotlib/main/doc/_static/stinkbug.png'


# %% matplotlib.pyplot.imshow
signatureImshow = "matplotlib.pyplot.imshow(X, cmap=None, norm=None, *, aspect=None, interpolation=None, alpha=None, vmin=None, vmax=None, origin=None, extent=None, interpolation_stage=None, filternorm=True, filterrad=4.0, resample=None, url=None, data=None, **kwargs)"

# %%
img = np.asarray(Image.open('stinkbug.png'))
print(repr(img))

# %% Shape of the image:
img.shape

# %% Plot the image
imgplot = plt.imshow(img)
plt.show()

# %% Pseudo colors of each R G and B layers (all similar, almost the same)
print("Plotting pseudo colors (one of R G B layers)")
for i, layer in zip(range(3), 'RGB'):  
    print('Next ={}'.format(i))
    plt.imshow(img[:, :, i])
    ax = plt.gca()
    ax.text(25, 25, "layer = {0}".format(layer))
    plt.show()
    
# %% Deifferent cmaps (viridis is the default)
for cmap in ['viridis', 'hot', 'gray']:
    plt.imshow(img[:, :, 0], cmap=cmap)
    ax = plt.gca()
    ax.text(25, 25, "cmap = {}".format(cmap))
    plt.show()

# %% Reset colormap to 'nipy_spectral'
imgplot = plt.imshow(img[:, : ,0])
imgplot.set_cmap('nipy_spectral')

ax = plt.gca()
ax.text(25, 25, "cmap = nipy_spectral")
plt.show()

# %% Color scale reference, using the colorbar
imgplot = plt.imshow(img[:, :, 0])
plt.colorbar()
plt.show()

# %%
# %% Examining a specific data range
plt.hist(img[:, :, 0].ravel(), bins=range(256), fc='k', ec='k')
plt.show()

# %% Using clim
clim = 120, 150
plt.imshow(img[:, :, 0], clim=clim)
ax = plt.gca()
ax.text(25, 25, "clim = {}".format(clim))
plt.show

# %% Array interpolation schemes
img = Image.open('stinkbug.png')
img.thumbnail((64, 64)) # resisze image in place

for interpolation in [None, 'bilinear', 'bicubic']:
    imgplot = plt.imshow(img, interpolation=interpolation)
    plt.text(5, 5, f"From 64x64 thumbnail: interpolation = {interpolation}")
    plt.show()

# %% Set xlim and ylim to scale image to real world: use extent argument
# extentfloats (left, right, bottom, top), optional
# The bounding box in data coordinates that the image will fill.
# The image is streched individually along x and y to fill the box.

img = np.asarray(Image.open('stinkbug.png'))
extent = -20, 20, 0, 40
imgplot = plt.imshow(img[:, :, 0], extent=extent)
plt.text(-15, 35, "Plotted to given exent {}".format(extent))
plt.show()

# %%
