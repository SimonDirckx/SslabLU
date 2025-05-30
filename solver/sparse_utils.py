import numpy as np
from   scipy.sparse.linalg import LinearOperator, splu
from   scipy.sparse        import csr_matrix, coo_matrix
try:
	from petsc4py import PETSc
	petsc_imported = True
except:
	petsc_imported = False

def petscdense_to_nparray(pM):
	M = pM.getValues(range(0,pM.getSize()[0]), \
		range(0,pM.getSize()[1]))
	return M

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
		pB = PETSc.Mat().createDense([B.shape[0],B.shape[1]],None,B)
		pX = PETSc.Mat().createDense([B.shape[0],B.shape[1]])
		pX.zeroEntries()

		ksp.matSolve(pB,pX)
		result = petscdense_to_nparray(pX)
		pX.destroy(); pB.destroy()
		return result
	return matsolve

######################################################################################################

class SparseSolver:

	def __init__(self,A):
		v                 = np.random.rand(A.shape[0],)
		self.is_symmetric = np.linalg.norm(A @ v - A.T @ v) < 1e-12
		self.N            = A.shape[0]

		self.use_petsc    = petsc_imported

		if (self.use_petsc):
			self.ksp = setup_ksp(A)
			if (not self.is_symmetric):
				# on some installations of petsc
				# there are issues with transpose matsolve
				self.ksp_adj = setup_ksp(A.T)
		else:
			self.ksp = splu(A.tocsc())

	@property
	def solve_op(self):

		if (self.use_petsc and self.is_symmetric):
			return LinearOperator(shape=(self.N,self.N),\
				matvec =get_vecsolve(self.ksp),\
				rmatvec=get_vecsolve(self.ksp),\
				matmat =get_matsolve(self.ksp),\
				rmatmat=get_matsolve(self.ksp))
			
		elif (self.use_petsc and not self.is_symmetric):
			return LinearOperator(shape=(self.N,self.N),\
				matvec =get_vecsolve(self.ksp),\
				rmatvec=get_vecsolve(self.ksp_adj),\
				matmat =get_matsolve(self.ksp),\
				rmatmat=get_matsolve(self.ksp_adj))

		elif (not self.use_petsc and self.is_symmetric):
			return LinearOperator(shape=(self.N,self.N),\
				matvec =lambda x: self.ksp.solve(x),\
				rmatvec=lambda x: self.ksp.solve(x),\
				matmat =lambda x: self.ksp.solve(x),\
				rmatmat=lambda x: self.ksp.solve(x))

		else:
			return LinearOperator(shape=(self.N,self.N),\
				matvec =lambda x: self.ksp.solve(x),\
				rmatvec=lambda x: self.ksp.solve(x,trans='T'),\
				matmat =lambda x: self.ksp.solve(x),\
				rmatmat=lambda x: self.ksp.solve(x,trans='T'))

class CSRBuilder:

	def __init__(self,M,N,nnz):

		self.M    = M
		self.N    = N

		self.row  = np.zeros(nnz,dtype=int)
		self.col  = np.zeros(nnz,dtype=int)
		self.data = np.zeros(nnz)
		self.acc  = 0

	def add_data(self,mat):

		mat   = mat.tocoo()
		ndata = mat.row.shape[0]
		assert self.acc + ndata < self.data.shape[0]

		self.row [self.acc : self.acc + ndata] = mat.row
		self.col [self.acc : self.acc + ndata] = mat.col
		self.data[self.acc : self.acc + ndata] = mat.data

		self.acc += ndata

	def tocsr(self):
		return coo_matrix((self.data[:self.acc],\
			(self.row[:self.acc],self.col[:self.acc]))).tocsr()