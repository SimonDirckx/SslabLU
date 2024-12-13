import numpy as np
from   abc              import ABCMeta, abstractmethod, abstractproperty

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