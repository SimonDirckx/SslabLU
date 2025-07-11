import os

Horder = 2
for p in [8,10,12]:

    os.system("python glrTest.py --p %d --Horder %d --pickle %s" % (p, Horder, "glr_spectrum_p%d_horder%d.pkl" % (p,Horder)))

p = 8
for Horder in [0,1,2,3,4,5,6]:

    os.system("python glrTest.py --p %d --Horder %d --pickle %s" % (p, Horder, "glr_spectrum_p%d_horder%d.pkl" % (p,Horder)))
