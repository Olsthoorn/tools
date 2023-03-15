# Nederland coordinates

# %%
import numpy as np
import shapefile
import os
import matplotlib.pyplot as plt

# %%
resolution = ['1km', '10km' , '100km'] # km
#datafile = os.join('./data/Netherlands_shapefile/nl_{}.shp').format(resolution[1])
datafile = '/Users/Theo/GRWMODELS/python/tools/coords/data/Netherlands_shapefile/nl_{}.shp'.format(resolution[2])


sf = shapefile.Reader(datafile)

print(sf)    

sf.shapeTypeName

for shp in sf.shapes():
    X, Y = np.array(shp.points).T
    plt.plot(X, Y)
    
plt.show()
# %%

