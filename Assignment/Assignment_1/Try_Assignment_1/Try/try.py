# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 14:10:27 2019

@author: Rajat_PC
"""
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
mu1, sigma1 = 5, 1
mu2, sigma2 = -4, 1
#mu3,sigma3=mu1+mu2,sigma1+sigma2
X1=np.random.normal(mu1, sigma1, 100)
X2=np.random.normal(mu2, sigma2, 100)
N=len(X1)
Z=np.arange(N)
Z=Z.astype(float)
mean1=X1.mean()
mean2=X2.mean()
for i in range(N):
        Z[i]=(X1[i]-mean1)*(X2[i]-mean2)