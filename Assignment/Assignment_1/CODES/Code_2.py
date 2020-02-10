# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 12:57:00 2019

@author: Rajat_PC
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

def Cov(X1,X2):
    N=len(X1)
    Z=np.arange(N)
    mean1=X1.mean()
    mean2=X2.mean()
    for i in range(N):
        Z[i]=(X1[i]-mean1)*(X2[i]-mean2)
    return Z.mean()    
        
#X1 = np.linspace(-10,10,500) 
#X2= np.linspace(15,25,500)
mu1, sigma1 = -2, 2
mu2, sigma2 = 5, 16
mu=np.array([mu1, mu2])
#mu3,sigma3=mu1+mu2,sigma1+sigma2
x=np.random.normal(mu1, sigma1, 500)
y=np.random.normal(mu2, sigma2, 500)
Xnew=np.array([x,y])  # 2x100
X, Y = np.meshgrid(x,y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y
# Calculating Variance Covariance Matrix

M = [[0 for x in range(2)] for y in range(2)] 
for i in range(2):
    for j in range(2):
        M[i][j]=Cov(Xnew[i],Xnew[j])
        
# 2X2 -> M      
N=len(x)
Det_M=np.linalg.det(M)
M_inv=np.linalg.inv(M)
Bivar=np.arange(N)
Bivar=Bivar.astype(float)
for i in range(N):
    Xneww=np.array([x[i]-mu[0],y[i]-mu[1]])
    temp1=np.dot(Xneww,M_inv)
    temp2=np.dot(temp1,Xneww.transpose())
    Bivar[i]= (1/2*np.pi)* pow(Det_M,-0.5) * np.exp(-0.5 *temp2 ) 
    
 # Plotting bivariate

rv = multivariate_normal([mu1, mu2],M)
fig = plt.figure()
fig = plt.gcf()
fig.set_size_inches(8, 8)
ax = fig.gca(projection='3d')

#ax.plot_surface(X, Y, rv.pdf(pos),cmap='magma',linewidth=0)
#ax.set_xlabel('Normal Random Variable-1 X axis')
#ax.set_ylabel('Normal Random Variable-2 Y axis')
#ax.set_zlabel('Bivariate Probability Distribution- Z axis')

#plt.show()


###
##
#Plotting Histogram

hist, xedges, yedges = np.histogram2d(x, y, bins=30)


xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
xpos = xpos.flatten('F')
ypos = ypos.flatten('F')
zpos = np.zeros_like(xpos)


dx = 0.5 * np.ones_like(zpos)
dy = dx.copy()
dz = hist.flatten()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
ax.set_xlabel('Normal Random Variable-1 X axis')
ax.set_ylabel('Normal Random Variable-2 Y axis')
ax.set_zlabel('Bivariate Probability Distribution- Z axis')

plt.show()



        
        
     
        