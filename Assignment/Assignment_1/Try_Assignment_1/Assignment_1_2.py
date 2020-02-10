# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 00:20:06 2019

@author: Rajat_PC
"""

from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
mu1, sigma1 = 5, 1
mu2, sigma2 = -4, 1
mu3,sigma3=mu1+mu2,sigma1+sigma2
s1 =np.random.normal(mu1, sigma1, 1000)
s2=np.random.normal(mu2, sigma2, 1000)
s3=s1+s2
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111)
ax.set_title('Univariate Normal Distribution(1000 Samples)')
ax.set_xlabel('Bins/Edes')
ax.set_ylabel('Probability ')
count1, bins1, ignored1 = plt.hist(s1, 30, density=True,label='Histogram1')
count2, bins2, ignored2 = plt.hist(s2, 30, density=True,label='Histogram2')
var1=1/(sigma1 * np.sqrt(2 * np.pi)) *np.exp( - (s1 - mu1)**2 / (2 * sigma1**2) )
var2=1/(sigma2 * np.sqrt(2 * np.pi)) *np.exp( - (s2 - mu2)**2 / (2 * sigma2**2) )
var3=1/(sigma2 * np.sqrt(2 * np.pi)) *np.exp( - (s2 - mu2)**2 / (2 * sigma2**2) )
#plt.plot(bins,var,linewidth=2, color='g')
bins3=bins1+bins2
#plt.plot(s1,var1,marker='.',color='r',linestyle='none')
#plt.plot(s2,var2,marker='.',color='g',linestyle='none')
#plt.plot(s3,var3,marker='.',color='b',linestyle='none')

plt.plot(bins3,norm.pdf(bins3),'r-', lw=3,  label='Norm pdf')
#plt.plot(bins2,norm.pdf(bins2),'b-', lw=3,  label='Norm pdf')
#plt.legend(loc='upper left')

plt.show()


