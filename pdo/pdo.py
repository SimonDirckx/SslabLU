import numpy as np
from   scipy.special import j0

def const(c=1):
    """
    Returns a function that generates a tensor of constants for a given set of locations.
    
    Parameters:
    c: The constant value to use. Defaults to 1.
    
    Returns:
    A function that takes `xxloc`, an n x d matrix of locations, and returns a tensor of constants c.
    """
    def const_func(xxloc):
        assert xxloc.shape[1] == 2 or xxloc.shape[1] == 3
        return c * np.ones(xxloc.shape[0],)
    return const_func

class PDO2d:
    """
    Represents a 2-dimensional Partial Differential Operator (PDO) with coefficients for the PDE terms.
    
    Parameters:
    c11, c22, c12: Coefficients for the second-order partial derivatives.
    c1, c2: Coefficients for the first-order derivatives.
    c: Coefficient for the zeroth-order term.

    The partial differential operator is given by:
    A [u](x) = - c11(x) [D11 u](x) - c22(x) [D22 u](x) 
    - 2 c12(x) [D12 u](x) + c1(x) [D1 u](x) + c2(x) [D2 u(x)] + c(x) [u](x)

    We assume the coefficients are smooth. 
    We also assume that the matrix [c11,c12; c12, c22] is positive definite in the domain to ensure ellipticity.
    """
    def __init__(self, c11, c22, c12=None, c1= None, c2 = None, c = None):
        self.c11, self.c22 = c11, c22
        self.c12 = c12
        self.c1, self.c2 = c1, c2
        self.c = c

class PDO3d:
    """
    Represents a 3-dimensional Partial Differential Operator (PDO) with coefficients for the PDE terms.
    
    Parameters:
    c11, c22, c33, c12, c13, c23: Coefficients for the second-order partial derivatives.
    c1, c2, c3: Coefficients for the first-order derivatives.
    c: Coefficient for the zeroth-order term.

    The partial differential operator is given by:
    A [u](x) = - c11(x) [D11 u](x) - c22(x) [D22 u](x) - c33(x) [D33 u](x)
    - 2 c12(x) [D12 u](x) - 2 c13(x) [D13 u](x) - 2 c23(x) [D23 u](x)
    + c1(x) [D1 u](x) + c2(x) [D2 u(x)] + c3(x) [D3 u(x)] + c(x) [u](x)

    We assume the coefficients are smooth. 
    We also assume that the matrix [c11,c12, c13; c12, c22 c23; c12, c23, c33] is positive definite in the domain to ensure ellipticity.
    These assumptions are from the paper by Hao, Martinsson on 3D HPS.
    See: https://users.oden.utexas.edu/~pgm/Pubs/2016_HPS_3D_final.pdf
    """
    def __init__(self, c11, c22, c33, c12=None, c13 = None, c23 = None, \
                 c1= None, c2 = None, c3 = None, c = None):
        self.c11, self.c22, self.c33 = c11, c22, c33
        self.c12, self.c13, self.c23 = c12, c13, c23
        self.c1, self.c2, self.c3    = c1, c2, c3
        self.c = c

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
    
    if (uu_exact.ndim == 1):
        uu_exact = uu_exact[:,np.newaxis]
    return uu_exact