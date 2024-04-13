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


#雅可比迭代
def jacobi(A, x, b, diag, iterations=1, omega=1.0):
    temp = x
    # Create uniform type, convert possibly complex scalars to length 1 arrays
    omega = torch.tensor(omega, dtype=A.dtype) #这里是将omega的type变成A的type

    for _iter in range(iterations):
        y = jacobi_solver(A, diag, temp, b, omega)
        temp = y
    return y

def jacobi_solver(A, diag, x, b, omega):
    one = 1.0
    omega2 = omega
    I = torch.eye(A.shape[0]).to_sparse_coo().to('cuda:0')
    x = (one-omega2) * x + omega2 * sparse.mm((I - sparse.mm(diag, A)), x) + sparse.mm(diag, b)
            
    return x

#官方算例，结果为5.835
A = poisson((10,10), format='csr')
diag = torch.tensor(np.linalg.inv(np.diag(A.diagonal()))).to_sparse_coo().to('cuda:0')
A = torch.sparse_csr_tensor(A.indptr, A.indices, A.data).to('cuda:0')
A = A.to_sparse_coo()
# A = torch.sparse_bsr_tensor(A.indptr, A.indices, A.data, dtype=torch.float32)
x0 = torch.zeros(A.shape[0],1, dtype=A.dtype).to('cuda:0').requires_grad_(True)
b = torch.ones(A.shape[0],1, dtype=A.dtype).to('cuda:0')
x = jacobi(A, x0, b, diag, iterations=10, omega=1.0)
print('Jacobi计算结果:')
print(f'{torch.linalg.norm(b - sparse.mm(A, x)):2.4}')




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

# #官方算例的计算结果是4.007  
# A = poisson((10,10), format='csr')
# # A.tobsr()
# A = torch.sparse_csr_tensor(A.indptr, A.indices, A.data, dtype=torch.float32)
# # A = torch.sparse_bsr_tensor(A.indptr, A.indices, A.data, dtype=torch.float32)
# x0 = torch.zeros(A.shape[0],1, dtype=torch.float32)
# b = torch.ones(A.shape[0],1, dtype=torch.float32)
# x = gauss_seidel(A, x0, b, iterations=10)
# print('gauss-seidel计算结果:')
# print(f'{torch.linalg.norm(b-sparse.mm(A, x0)):2.4}')
