# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 00:27:05 2019

@author: Rajat_PC
"""

import numpy as np
import matplotlib.pyplot as plt

#CLT
N=10000

Mu=np.linspace(-10,10,N)
S=np.linspace(5,15,N)
X=np.empty([1,N])
Means=np.empty([N,1])
Sigma=np.empty([N,1])
var=np.empty([N,1])
#M = [[0 for x in range(1000)] for y in range(1000)]
M=np.random.uniform(0, 1, N)
Means[0]=M.mean()
Sigma[0]=M.var()
for i in range(1,N):
    newrow=np.random.uniform(0, 1, N)
    Means[i]=newrow.mean()
    Sigma[i]=newrow.var()
    
    
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111)
ax.set_title('CLT(10000 Samples)')
ax.set_xlabel('Means of 10000 Random Variables')
ax.set_ylabel('Probability ')
count, bins, ignored = plt.hist(Means, 30, density=True,label='Histogram')

plt.legend(loc='upper left')

plt.show()    
    
    
    

    
    




