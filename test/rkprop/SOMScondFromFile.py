import numpy as np
import matplotlib.pyplot as plt

stats = np.loadtxt('condSOMS3D.csv',delimiter=',')
#print(stats)
#print(stats.shape)

Nvec = stats[0,:]
condS = stats[1,:]
condT = stats[3,:]

fitS = (1+np.log2(Nvec))**2
fitT = Nvec**(3/4)

fitS*=condS[-1]/fitS[-1]
fitT*=condT[-1]/fitT[-1]



plt.figure(1)
plt.loglog(Nvec,condS)
plt.loglog(Nvec,fitS,linestyle='dashed')
plt.loglog(Nvec,condT)
plt.loglog(Nvec,fitT,linestyle='dashed')
plt.legend(['condS','fitS','condT','fitT'])
plt.show()