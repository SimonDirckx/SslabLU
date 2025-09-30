import os

H = 1; p = 8 + 2
for ainv in [2,4,8,16]:
	os.system("python slab_example.py --H %5.15f --p %d --ndim 3 --kh 0 --ainv %d --pickle_loc %s" % \
		(H,p,ainv,"thinslab_H%2.5f_p%d_ainv%d.pkl" % (H,p,ainv)))

H = 0.25; p = 8 + 2
for ainv in [8,16,32,64]:
	os.system("python slab_example.py --H %5.15f --p %d --ndim 3 --kh 0 --ainv %d --pickle_loc %s" % \
		(H,p, ainv,"thinslab_H%2.5f_p%d_ainv%d.pkl" % (H,p,ainv)))

ainv = 64; p = 2
for nslabs in [1,2,4,8,16]:
	H = 1.0/nslabs
	os.system("python slab_example.py --H %5.15f --p %d --ndim 3 --kh 0 --ainv %d --pickle_loc %s" % \
		(H,p, ainv,"thinslab_H%2.5f_p%d_ainv%d.pkl" % (H,p,ainv)))	