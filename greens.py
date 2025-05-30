import numpy as np
from   scipy.special import j0
def get_known_greens(xx,kh,center=None):

    """
    Returns a Greens function on evaluated at given points.
    """

    xx_tmp = xx.copy()
    ndim   = xx_tmp.shape[-1]
    if (center is None):
        center = np.ones(ndim,)*10

    if (xx.shape[-1] == 2):

        xx_d0 = xx_tmp[:,0] - center[0]
        xx_d1 = xx_tmp[:,1] - center[1]
        ddsq  = np.multiply(xx_d0,xx_d0) + np.multiply(xx_d1,xx_d1)
        rr    = np.sqrt(ddsq)

        if (kh == 0):
            uu_exact = np.log(rr)
        else:
            uu_exact = j0(kh * rr)
    else:
        xx_d0 = xx_tmp[:,0] - center[0]
        xx_d1 = xx_tmp[:,1] - center[1]
        xx_d2 = xx_tmp[:,2] - center[2]
        ddsq  = np.multiply(xx_d0,xx_d0) + np.multiply(xx_d1,xx_d1) + np.multiply(xx_d2,xx_d2)
        rr    = np.sqrt(ddsq)

        if (kh == 0):
            uu_exact = 1 / rr
        else:
            uu_exact = np.sin(kh * rr) / rr
    
    return uu_exact