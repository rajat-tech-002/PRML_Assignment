# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 23:35:24 2019

@author: Rajat_PC
"""

# Box Muller Method
N=1000
import matplotlib.pyplot as plt
import math
import numpy as np
def Cov(X1,X2):
    M=len(X1)
    Z=np.arange(M)
    mean1=X1.mean()
    mean2=X2.mean()
    for i in range(M):
        Z[i]=(X1[i]-mean1)*(X2[i]-mean2)
    return Z.mean() 
U1= np.random.uniform(0, 1, N)
U2= np.random.uniform(0, 1, N)
Z0=np.empty([N,1])
Z1=np.empty([N,1])
for i in range(N):
    
 Z0[i]=np.sqrt(-2*math.log(U1[i],math.e)) * math.cos(2*math.pi*U2[i])
 Z1[i]=np.sqrt(-2*math.log(U1[i],math.e)) * math.sin(2*math.pi*U2[i])
 
 mu1,mu2= Z0.mean(), Z1.mean()
 stdev1, stdev2= Z0.var(), Z1.var()
 Covar=Cov(Z0,Z1)
 
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111)
ax.set_title('Univariate Normal Distribution(1000 Samples)')
ax.set_xlabel('Bins/Edes')
ax.set_ylabel('Probability ')
count, bins, ignored = plt.hist(Z0, 30, density=True,label='Histogram')
#var=1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (s - mu)**2 / (2 * sigma**2) )
#plt.plot(bins,var,linewidth=2, color='g')
#plt.plot(s,var,marker='o',color='r',linestyle='none')
#plt.plot(bins,norm.pdf(bins),'b-', lw=3,  label='Norm pdf')
plt.legend(loc='upper left')
plt.show()