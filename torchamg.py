import torch

class TwoGrid:
    def __init__(self,A,p):
        self.A = A
        self.P = p
        self.R = p.T

    def Setup(self):
        A_c = self.R @ self.A @ self.P 
        self.dense_A_c = A_c.to_dense()

    def CoarseSolve(self,b):
        x = torch.linalg.solve(self.dense_A_c, b)
        return x

    def Solve(self,b):
        if len(b.shape) == 1:
            b = b.unsqueeze(1)

        Rb = self.R @ b
        x = self.CoarseSolve(Rb)
        out = self.P @ x

        return out

def make_system(A, x, b, formats=None):
    if formats is None:
        pass
    elif formats == ['csr']:
        if A.is_sparse_csr:
            pass
        elif A.is_sparse_bsr:
             A = A.to_sparse_csr()
        else:
            if A.is_sparse:
                pass
            else:
                # warn('implicit conversion to CSR', sparse.SparseEfficiencyWarning)
                A = A.to_sparse_csr()
    else:
        if A.is_sparse:
            pass
        else:
           A = A.to_sparse_csr()
    M, N = A.shape

    if M != N:
        raise ValueError('expected square matrix')

    if x.shape not in [(M,), (M, 1)]:
        raise ValueError('x has invalid dimensions')
    if b.shape not in [(M,), (M, 1)]:
        raise ValueError('b has invalid dimensions')

    if A.dtype != x.dtype or A.dtype != b.dtype:
        raise TypeError('arguments A, x, and b must have the same dtype')

    # if not x.flags.carray:
    #     raise ValueError('x must be contiguous in memory')

    x = torch.ravel(x)
    b = torch.ravel(b)

    return A, x, b

#雅可比迭代
def jacobi(A, x, b, iterations=1, omega=1.0):
    A, x, b = make_system(A, x, b, formats=['csr'])

    sweep = slice(None)
    (row_start, row_stop, row_step) = sweep.indices(A.shape[0])

    if (row_stop - row_start) * row_step <= 0:  # no work to do
        return

    temp = torch.empty_like(x)

    # Create uniform type, convert possibly complex scalars to length 1 arrays
    omega = torch.tensor(omega, dtype=A.dtype) #这里是将omega的type变成A的type

    if A.is_sparse_csr:
        for _iter in range(iterations):
            x = jacobi_solver(A.crow_indices(), A.col_indices(), A.values(), x, b, temp,
                            row_start, row_stop, row_step, omega)
    #分块矩阵的jacobi这里还没写, pytorch里的稀疏矩阵暂时还没找到储存blocksize的地方
    # else:
    #     R, C = A.blocksize
    #     if R != C:
    #         raise ValueError('BSR blocks must be square')
    #     row_start = int(row_start / R)
    #     row_stop = int(row_stop / R)
        # for _iter in range(iterations):
        #     amg_core.bsr_jacobi(A.indptr, A.indices, np.ravel(A.data),
        #                         x, b, temp, row_start, row_stop,
        #                         row_step, R, omega)
    return x

def jacobi_solver(indptr, indices, data, x, b, temp, row_start, row_stop, row_step, omega):
    one = 1.0
    omega2 = omega

    temp[range(row_start, row_stop, row_step)] = x[range(row_start, row_stop, row_step)].clone()
    
    for i in range(row_start, row_stop, row_step):
        with torch.no_grad():
            start = int(indptr[i])
            end = int(indptr[i+1])
        rsum = 0
        diag = 0

        for jj in range(start, end, 1):
            with torch.no_grad():
                j = int(indices[jj])
            if i==j:
                diag = data[jj].item()
                
            else:
                rsum = rsum + data[jj] * temp[j]

        if diag != 0:
            x[i] = (one-omega2) * temp[i] + omega2 * ((b[i]- rsum)/diag)
            
    return x

#官方算例，结果为5.835
A = poisson((10,10), format='csr')
A = torch.sparse_csr_tensor(A.indptr, A.indices, A.data, dtype=torch.float32)
# A = torch.sparse_bsr_tensor(A.indptr, A.indices, A.data, dtype=torch.float32)
x0 = torch.zeros(A.shape[0],1, dtype=torch.float32)
b = torch.ones(A.shape[0],1, dtype=torch.float32)
x = jacobi(A, x0, b, iterations=10, omega=1.0)
print(f'{torch.linalg.norm(b-sparse.mm(A, x0)):2.4}')

#计算gauss_seidel迭代
def gauss_seidel(A, x, b, iterations=1, sweep='forward'):
    A, x, b = make_system(A, x, b, formats=['csr', 'bsr'])


    if A.is_sparse_csr:
        blocksize = 1
    else:
        blocksie = 1
    #     # R, C = A.blocksize
    #     # if R != C:
    #     #     raise ValueError('BSR blocks must be square')
    #     # blocksize = R

    if sweep not in ('forward', 'backward', 'symmetric'):
        raise ValueError('valid sweep directions: "forward", "backward", and "symmetric"')

    if sweep == 'forward':
        row_start, row_stop, row_step = 0, int(len(x)/blocksize), 1
    elif sweep == 'backward':
        row_start, row_stop, row_step = int(len(x)/blocksize)-1, -1, -1
    elif sweep == 'symmetric':
        for _iter in range(iterations):
            gauss_seidel(A, x, b, iterations=1, sweep='forward')
            gauss_seidel(A, x, b, iterations=1, sweep='backward')

    if A.is_sparse_csr:
        for _iter in range(iterations):
            gauss_seidel_solver(A.crow_indices(), A.col_indices(), A.values(), x, b,
                            row_start, row_stop, row_step)
    # else:
    #     for _iter in range(iterations):
    #         amg_core.bsr_gauss_seidel(A.indptr, A.indices, np.ravel(A.data),
    #                                   x, b, row_start, row_stop, row_step, R)

def gauss_seidel_solver(indptr, indices, data, x, b, row_start, row_stop, row_step):
    for i in range(row_start, row_stop, row_step):
        with torch.no_grad():
            start = int(indptr[i])
            end = int(indptr[i+1])
        rsum = 0
        diag = 0

        for jj in range(start, end, 1):
            with torch.no_grad():
                j = int(indices[jj])
            if i==j:
                diag = data[jj].item()
                
            else:
                rsum = rsum + data[jj] * x[j]

        if diag != 0:
            x[i] = (b[i]- rsum)/diag
            
    return x

#官方算例的计算结果是4.007
A = poisson((10,10), format='csr')
# A.tobsr()
A = torch.sparse_csr_tensor(A.indptr, A.indices, A.data, dtype=torch.float32)
# A = torch.sparse_bsr_tensor(A.indptr, A.indices, A.data, dtype=torch.float32)
x0 = torch.zeros(A.shape[0],1, dtype=torch.float32)
b = torch.ones(A.shape[0],1, dtype=torch.float32)
x = gauss_seidel(A, x0, b, iterations=10)
print(f'{torch.linalg.norm(b-sparse.mm(A, x0)):2.4}')

