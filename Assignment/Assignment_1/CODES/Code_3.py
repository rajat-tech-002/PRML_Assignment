# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 00:27:05 2019

@author: Rajat_PC
"""

import numpy as np
import matplotlib.pyplot as plt

#CLT
N=1000
Means_sample=np.empty([12,1])
for j in range(12):
    
 Means=np.empty([N,1])

#M = [[0 for x in range(1000)] for y in range(1000)]
 M=np.random.uniform(0, 1, N)
 Means[0]=M.mean()

 for i in range(1,N):
    newrow=np.random.uniform(0, 1, N)
    Means[i]=newrow.mean()
   
    
    
 fig = plt.figure(figsize=(8,5))
 ax = fig.add_subplot(111)
 ax.set_title('CLT(1000 Samples)')
 ax.set_xlabel('Bins')
 ax.set_ylabel('Histogram Counts ')
 count, bins, ignored = plt.hist(Means, 30, density=True,label='Histogram')

 plt.legend(loc='upper left')

 plt.show()    
    
    
    

    
    




