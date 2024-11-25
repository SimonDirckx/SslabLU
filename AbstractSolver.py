import numpy as np
import scipy as sc
import scipy.sparse as sparse
import scipy.sparse.linalg as splinalg

import numpy as np
from   abc              import ABCMeta, abstractmethod, abstractproperty

'''
Abstract solver to be used in slabs
@internal
    geom        : Representation of the local geometry
	ndim        : Dimension of the underlying problem (2 or 3)
	XX          : Discretization points
	I_C         : Interior points
	I_X         : Boundary points
    A_CC & A_CX : Corresponding blocks in system mat
'''



class AbstractSolver(metaclass=ABCMeta):

	#################################################

	@abstractproperty
	def box_geom(self):
		pass

	@property
	def ndim(self):
		return self.box_geom.shape[-1]

	#################################################

	@abstractproperty
	def XX(self):
		pass

	@abstractproperty
	def p(self):
		pass

	@abstractproperty
	def I_C(self):
		pass

	@abstractproperty
	def I_X(self):
		pass

	@abstractproperty
	def npoints_dim(self):
		pass

	#################################################

	@abstractproperty
	def A_CC(self):
		pass

	@abstractproperty
	def A_CX(self):
		pass

	def setup_solver_CC(self,solve_op=None):
		if (solve_op is None):
			self.solve_op = SparseSolver(self.A_CC).solve_op
		else:
			self.solve_op = solve_op

	@property
	def solver_CC(self):
		if (not hasattr(self,'solve_op')):
			self.setup_solver_CC()
		return self.solve_op