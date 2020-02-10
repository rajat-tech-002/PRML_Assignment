from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
mu, sigma = 0, 1
s = np.random.normal(mu, sigma, 1000)

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(111)
ax.set_title('Univariate Normal Distribution(1000 Samples)')
ax.set_xlabel('Bins/Edes')
ax.set_ylabel('Probability ')
count, bins, ignored = plt.hist(s, 30, density=True,label='Histogram')
var=1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (s - mu)**2 / (2 * sigma**2) )
#plt.plot(bins,var,linewidth=2, color='g')
#plt.plot(s,var,marker='o',color='r',linestyle='none')
plt.plot(bins,norm.pdf(bins),'b-', lw=3,  label='Norm pdf')
plt.legend(loc='upper left')
plt.show()


