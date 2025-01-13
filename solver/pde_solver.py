import numpy as np
from   abc              import ABCMeta, abstractmethod, abstractproperty
from solver import sparse_utils

class AbstractPDESolver(metaclass=ABCMeta):

	#################################################

	@abstractproperty
	def geom(self):
		pass

	@property
	def ndim(self):
		return self.geom.ndim

	#################################################

	@abstractproperty
	def XX(self):
		pass

	@abstractproperty
	def p(self):
		pass

	@abstractproperty
	def Ji(self):
		# index vector for interior points
		pass

	@abstractproperty
	def Jb(self):
		# index vector for exterior points
		pass

	@abstractproperty
	def npoints_dim(self):
		pass

	#################################################

	@abstractproperty
	def Aii(self):
		pass

	@abstractproperty
	def Aib(self):
		pass

	@abstractproperty
	def Abb(self):
		pass

	@abstractproperty
	def Abi(self):
		pass
	
	def constructSolverii(self):
		self.solver_ii = sparse_utils.SparseSolver(self.Aii).solve_op	
	
	def solver_ii(self):
		if self.solver_ii==None:
			self.constructSolverii()
			return sparse_utils.SparseSolver(self.Aii).solve_op
		else:
			return sparse_utils.SparseSolver(self.Aii).solve_op
	def solveDirichlet(self,b):
		if self.solver_ii==None:
			self.constructSolverii()
		return self.solver_ii@(self.Aib@b)