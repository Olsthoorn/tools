#!/usr/bin/env python

# Test @ operator
# %%
import numpy as np

A = np.random.rand(5, 5)
B = np.random.rand(5)

C = A @ B
print("C=", C)
# %%