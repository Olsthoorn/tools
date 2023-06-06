import numpy as np
import matplotlib.pyplot as plt
import ttim

ml = ttim.ModelMaq(kaq=[10, 30, 20], z=[0, -5, -10, -20, -25, -35], \
              c=[2000, 5000], Saq=[0.1, 1e-4, 2e-4], \
              Sll=[1e-4, 4e-4], phreatictop=True, \
              tmin=0.01, tmax=10)

screens = [0, 1]

w = ttim.well.Well(ml, xw=0, yw=0, rw=0.1, tsandQ=[0, 1000.], res=0, rc=None, layers=screens, wbstype='pumping', label='well 1')

#hwell = ttim.well.HeadWell(ml, xw=0, yw=0, rw=0.1,tsandh=[0, 1], res=0, #layers=0, label='hwell 1')

ml.solve()

print('Done')

ttim.Model3D()

ttim.ModelMaq()