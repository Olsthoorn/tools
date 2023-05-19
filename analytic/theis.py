import numpy as np
import matplotlib.pyplot as plt
from scipy.special import exp1

u = np.logspace(-6, 1)

plt.plot(1/u, exp1(u), label="Theis' well function (scipy.special.exp1)")
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.show()
plt.legend(loc="lower right")

print('Done')