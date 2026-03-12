import numpy as np
import matplotlib.pyplot as plt
N = np.array([2**14,2**16,2**18,2**20])

#nl = 128
t_ULV_32 = [1.06,4.56,22.22,103.3]
t_solve_32 =[0.1,0.3,1.41,5.18]

t_ULV_64 = [1.84,8.32,39.03,176.25]
t_solve_64 =[0.1,0.36,1.41,5.1]

t_ULV_128 = [4.09,18.68,85.49,388.10]
t_solve_128 =[0.1,0.36,1.67,8.65]

plt.figure(1)
plt.loglog(N,t_ULV_32)
plt.loglog(N,t_ULV_64)
plt.loglog(N,t_ULV_128)
NlogN = N*np.log(N)
plt.loglog(N,1.2*NlogN*(t_ULV_128[0]/NlogN[0]),linestyle='--')
plt.legend(['k=32','k=64','k=128','NlogN'])
plt.title("ULV factorization time: nl = 256")
plt.figure(2)
plt.loglog(N,t_solve_32)
plt.loglog(N,t_solve_64)
plt.loglog(N,t_solve_128)
plt.loglog(N,1.2*NlogN*(t_solve_128[0]/NlogN[0]),linestyle='--')
plt.legend(['k=32','k=64','k=128','NlogN'])
plt.title("ULV solve time: nl=256")

#nl=64
t_ULV_32 = [0.99,4.46,19.76,97.55]
t_solve_32 =[0.056,0.18,.89,4.28]

t_ULV_64 = [2.4,9.9,46.1,206.40]
t_solve_64 =[0.098,0.37,1.36,5.76]

t_ULV_128 = [4.32,18.8,88.25,423.93]
t_solve_128 =[0.14,0.46,1.95,8.82]

plt.figure(3)
plt.loglog(N,t_ULV_32)
plt.loglog(N,t_ULV_64)
plt.loglog(N,t_ULV_128)
NlogN = N*np.log(N)
plt.loglog(N,1.2*NlogN*(t_ULV_128[0]/NlogN[0]),linestyle='--')
plt.legend(['k=32','k=64','k=128','NlogN'])
plt.title("ULV factorization time: nl = 256")
plt.figure(4)
plt.loglog(N,t_solve_32)
plt.loglog(N,t_solve_64)
plt.loglog(N,t_solve_128)
plt.loglog(N,1.2*NlogN*(t_solve_128[0]/NlogN[0]),linestyle='--')
plt.legend(['k=32','k=64','k=128','NlogN'])
plt.title("ULV solve time: nl=256")
plt.show()