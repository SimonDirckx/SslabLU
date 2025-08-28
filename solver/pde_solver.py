import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty
#from hps.pdo import get_known_greens
from solver.spectralmultidomain.hps.sparse_utils import SparseSolver

class AbstractPDESolver(metaclass=ABCMeta):
    """
    Abstract base class defining the interface for PDE solvers using the HPS framework.
    Subclasses must implement properties for geometry, indexing, and block matrices.
    """

    #################################################
    # Abstract properties defining essential data
    #################################################

    @abstractproperty
    def geom(self):
        """
        Geometry object containing domain bounds and, optionally, a parameterization map.
        Must have attribute `bounds` of shape (2, ndim).
        """
        pass

    @property
    def ndim(self):
        """
        Number of spatial dimensions, inferred from geometry bounds.
        """
        return self.geom.bounds.shape[-1]

    @abstractproperty
    def XX(self):
        """
        Flattened array of coordinates for all boundary (exterior) nodes:
        shape = (n_boundary_points, ndim).
        """
        pass

    @abstractproperty
    def p(self):
        """
        Polynomial degree used in each patch (number of Chebyshev nodes per direction).
        """
        pass

    @abstractproperty
    def Ji(self):
        """
        Index array for interior (duplicated interface) points in the global boundary ordering.
        """
        pass

    @abstractproperty
    def Jx(self):
        """
        Index array for unique exterior (non‐duplicated) boundary points.
        """
        pass

    @abstractproperty
    def npoints_dim(self):
        """
        Total number of Chebyshev points per dimension (npan_dim * p for each dimension).
        """
        pass

    #################################################
    # Abstract properties defining Schur complement blocks
    #################################################

    @abstractproperty
    def Aii(self):
        """
        Sparse matrix block coupling interior‐interior (duplicated interface) degrees of freedom.
        """
        pass

    @abstractproperty
    def Aix(self):
        """
        Sparse matrix block coupling interior (duplicated interface) to unique exterior DOFs.
        """
        pass

    @abstractproperty
    def Axx(self):
        """
        Sparse matrix block coupling unique exterior DOFs to themselves.
        """
        pass

    @abstractproperty
    def Axi(self):
        """
        Sparse matrix block coupling unique exterior DOFs to interior (duplicated interface) DOFs.
        """
        pass

    #################################################
    # Setup and retrieve a solver for the Aii block
    #################################################

    def setup_solver_Aii(self, solve_op=None, use_approx=False):
        """
        Initialize a linear solver for the Aii block. If solve_op is provided,
        use it directly. Otherwise, create a SparseSolver on Aii.

        Parameters:
        - solve_op:   Optionally, a pre‐constructed LinearOperator for Aii^{-1}.
        - use_approx: If True and PETSc is available, use an approximate iterative solver.
        """
        if solve_op is None:
            # Build a new SparseSolver on Aii; solver_Aii property will wrap it
            self.solve_op = SparseSolver(self.Aii, use_approx=use_approx).solve_op
        else:
            self.solve_op = solve_op

    @property
    def solver_Aii(self):
        """
        Return the LinearOperator that applies Aii^{-1}. If not yet set up,
        call setup_solver_Aii with default parameters.
        """
        if not hasattr(self, 'solve_op'):
            self.setup_solver_Aii()
        return self.solve_op

    #################################################
    # Helper functions: wavenumber and points‐per‐wavelength
    #################################################

    def get_nwaves_dim(self, kh):
        """
        Given wavenumber*characteristic_length kh (unitless),
        compute number of wavelengths across the domain in each dimension.

        Parameters:
        - kh: float or array-like, product of wavenumber k and reference length.

        Returns:
        - nwaves_dim: array-like, number of wavelengths per dimension.
        """
        # Number of wavelengths per unit length = kh / (2π)
        nwaves_unit = kh / (2 * np.pi)
        # Domain extents per dimension
        nunits_dim = self.geom.bounds[1] - self.geom.bounds[0]
        # Scale to full domain
        nwaves_dim = nwaves_unit * nunits_dim
        return nwaves_dim

    def get_ppw_dim(self, kh):
        """
        Compute points‐per‐wavelength in each dimension: 
        number of Chebyshev points divided by number of wavelengths.

        Parameters:
        - kh: float or array-like, product of wavenumber k and reference length.

        Returns:
        - ppw_ndim: array-like, points‐per‐wavelength per dimension.
        """
        ppw_ndim = self.npoints_dim / self.get_nwaves_dim(kh)
        return ppw_ndim

    #################################################
    # Solve the Dirichlet subproblem on interfaces
    #################################################

    def solve_dir(self, uu_dir, ff_body=None):
        """
        Solve Aii * u_interior = ff_body - Aix * u_boundary for interior interface DOFs.

        Parameters:
        - uu_dir:   Array of prescribed Dirichlet values at all unique exterior boundary DOFs.
                    Can be shape (n_boundary,) or (n_boundary, nrhs) for multiple RHS.
        - ff_body:  Optional interior forcing term at interface nodes,
                    shape (n_interior, nrhs). If None, assumed zero.

        Returns:
        - result: Solution at interior interface DOFs, shape (n_interior,) or (n_interior, nrhs).
        """
        # Ensure uu_dir has two dimensions: (n_boundary, nrhs)
        if uu_dir.ndim == 1:
            uu_tmp = uu_dir[:, np.newaxis]
        else:
            uu_tmp = uu_dir
        nrhs = uu_tmp.shape[-1]

        # If no body term is given, set ff_body to zero array
        if ff_body is None:
            ff_body = np.zeros((self.Ji.shape[0], nrhs))

        # Compute the right‐hand side: ff_body - Aix * uu_boundary
        rhs = ff_body - self.Aix @ uu_tmp
        # Solve using the Aii solver: Aii^{-1} rhs
        result = self.solver_Aii(rhs)

        # If single RHS and input was 1D, return a 1D array
        if uu_dir.ndim == 1:
            result = result.flatten()
        return result

    #################################################
    # Discretization verification via known Green's functions
    #################################################
    '''
    def verify_discretization(self, kh):
        """
        Test the Dirichlet solver by comparing to a known Green's function solution.

        1. Evaluate known Green's solution at all boundary nodes XX.
        2. Solve for interior DOFs using solve_dir, given boundary values.
        3. Compare to true interior values extracted from known solution.

        Parameters:
        - kh: float or array-like, product of wavenumber k and reference length.

        Returns:
        - relerr: float, relative ℓ₂ error between computed and true interior values.
        """
        # 1) Possibly map XX through a parameterization, if geometry defines one
        if hasattr(self.geom, 'parameter_map'):
            XX_mapped = self.geom.parameter_map(self.XX)
        else:
            XX_mapped = self.XX

        # 2) Evaluate known Green's function at boundary points, with a source placed far outside domain
        uu = get_known_greens(XX_mapped, kh, center=self.geom.bounds[1] + 10)
        # 3) Extract boundary values (unique exterior) and interior true values
        uu_sol = self.solve_dir(uu[self.Jx])
        uu_true = uu[self.Ji]

        # 4) Compute error and relative error
        err = np.linalg.norm(uu_sol - uu_true, ord=2)
        relerr = err / np.linalg.norm(uu_true, ord=2)
        return relerr
'''