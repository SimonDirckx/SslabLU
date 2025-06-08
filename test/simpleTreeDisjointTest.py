import numpy as np
import matAssembly.HBS.simpleoctree.simpletree as simpletree
import matplotlib.pyplot as plt
from matAssembly.HBS.simpleoctree.examples.plotting_utils import *
n = 1000
r1=1.
r2=2.
circle1 = np.zeros(shape=(n,2))
circle2 = np.zeros(shape=(n,2))
thVec = np.linspace(0.,1.,n)
for i in range(n):
    th = thVec[i]
    z=np.array([np.cos(2*np.pi*th),np.sin(2*np.pi*th)])
    circle1[i,:]=r1*z
    circle2[i,:]=r2*z
XX = np.concatenate([circle1,circle2])
tree = simpletree.BalancedTree(XX,10)
fig,ax = plt.subplots()
box = 5
I_B = tree.get_box_inds(box)
I_N = tree.get_neigh_inds(box)
leaf_keys   = tree.leaf_keys
ax.scatter(XX[:,0], XX[:,1],s=1,color='tab:blue')

ax.scatter(XX[I_B,0],XX[I_B,1],s=1,color='black')
ax.scatter(XX[I_N,0],XX[I_N,1],s=1,color='tab:green')

##################################################################################
################################ Plotting utils   ################################

add_patches(ax,tree,leaf_keys,keys=True,\
            edgecolor='tab:gray',facecolor='none',linewidth=1.0,alpha=0.5)

add_patches(ax,tree,np.setdiff1d(tree.get_box_colleague_neigh(box),np.array([box])),\
                                 edgecolor='tab:green',text_label=False,fontsize=14)
add_patches(ax,tree,tree.get_box_coarse_neigh(box),edgecolor='tab:green',\
            text_label=False,fontsize=14)
add_patches(ax,tree,np.array([box]),edgecolor='black',\
            text_label=False,fontsize=14)

##################################################################################

plt.gca().set_aspect('equal', adjustable='box')
plt.show()
