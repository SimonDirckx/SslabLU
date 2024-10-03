import numpy as np
from   scipy.sparse.linalg import LinearOperator
try:
	from petsc4py import PETSc
	petsc_imported = True
	print("PETSC IMPORTED")
except:
	petsc_imported = False

def setup_ksp(A):

	A = A.tocsr()

	ksp = PETSc.KSP().create()
	pA  = PETSc.Mat().createAIJ(A.shape, csr=(A.indptr,A.indices,A.data))
	ksp.setOperators(pA)

	ksp.setType('preonly')
	ksp.getPC().setType('lu')
	ksp.getPC().setFactorSolverType('mumps')
	ksp.setUp()
	return ksp

def get_vecsolve(ksp):

	def vecsolve(b):

		pb = PETSc.Vec().createWithArray(b)
		px = PETSc.Vec().createWithArray(np.zeros(b.shape))
		ksp.solve(pb,px)
		
		result = px.getArray()
		px.destroy(); pb.destroy()
		return result

	return vecsolve

def get_matsolve(ksp):

	def matsolve(B):
		try:
			pB = PETSc.Mat().createDense([B.shape[0],B.shape[1]],None,B)
			pX = PETSc.Mat().createDense([B.shape[0],B.shape[1]])
			pX.zeroEntries()

			self.LU_CC.matSolve(pB,pX)
			result = pX.getDenseArray()
			return result
		except:
			raise ValueError("on some petsc installations, matsolves cause issues")
	return matsolve

######################################################################################################

class PETScSolver:

	def __init__(self,A):

		v                 = np.random.rand(A.shape[0],)
		self.is_symmetric = np.linalg.norm(A @ v - A.conj().T @ v) < 1e-12
		self.N            = A.shape[0]

		if (not petsc_imported):
			raise ValueError("petsc not imported")

		self.ksp = setup_ksp(A)
		if (not self.is_symmetric):

			# on some installations of petsc, there are issues with transpose matsolve
			self.ksp_adj = setup_ksp(A.conj().T)

	@property
	def solve_op(self):

		if self.is_symmetric:

			return LinearOperator(shape=(self.N,self.N),\
				matvec =get_vecsolve(self.ksp),\
				rmatvec=get_vecsolve(self.ksp),\
				matmat =get_matsolve(self.ksp),\
				rmatmat=get_matsolve(self.ksp))
		else:

			return LinearOperator(shape=(self.N,self.N),\
				matvec =get_vecsolve(self.ksp),\
				rmatvec=get_vecsolve(self.ksp_adj),\
				matmat =get_matsolve(self.ksp),\
				rmatmat=get_matsolve(self.ksp_adj))
