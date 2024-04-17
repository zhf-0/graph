import torch

class TwoGrid:
    def __init__(self, pre_smoother, post_smoother, smoothing_num, coarse_solver, coarse_num, dtype=torch.float64, device='cpu'):
        self.pre_smoother = pre_smoother
        self.post_smoother = post_smoother
        self.coarse_solver = coarse_solver
        self.smoothing_num = smoothing_num
        self.coarse_num = coarse_num
        self.dtype = dtype
        self.device = device

    def Setup(self, A, p):
        R = p.T
        A_c = R @ A @ p 

        self.pre_smoother.Setup(A)
        self.post_smoother.Setup(A)
        self.coarse_solver.Setup(A_c)
        
        self.P = p.to(self.device)
        self.R = R.to(self.device)
        self.A_c = A_c.to(self.device)
        self.A = A.to(self.device)
        # self.dense_A_c = A_c.to_dense().to(self.device)

    def CoarseSolve(self, b, x):
        for _ in range(self.coarse_num):
            x = self.coarse_solver.Solve(b, x)

        return x

    def Solve(self, b, x):
        if len(b.shape) == 1:
            b = b.unsqueeze(1)

        for _ in range(self.smoothing_num):
            x = self.pre_smoother.Solve(b, x)

        residual = b - self.A @ x

        coarse_b = self.R @ residual
        coarse_x = self.CoarseSolve(coarse_b, torch.zeros(coarse_b.shape,dtype=self.dtype,device=self.device) )
        x += self.P @ coarse_x

        for _ in range(self.smoothing_num):
            x = self.post_smoother.Solve(b, x)

        return x

def GetDiagVec(coo_A, dtype=torch.float64, device='cpu'):
    coo = coo_A.coalesce()
    row_vec, col_vec = coo.indices()
    val_vec = coo.values()
    nrow = coo.shape[0]
    
    diag = torch.zeros(nrow,dtype=dtype,device=device)
    mask = row_vec == col_vec
    diag[row_vec[mask]] = val_vec[mask]

    return diag


def GetInvDiagSpMat(coo_A, dtype=torch.float64, device='cpu'):
    diag = GetDiagVec(coo_A, dtype, device)
    invdiag = 1.0 / diag

    nrow = coo_A.shape[0]
    row_vec = torch.arange(nrow,device=device)
    col_vec = torch.arange(nrow,device=device)
    coo_invdiag = torch.sparse_coo_tensor(torch.stack((row_vec,col_vec)),invdiag, (nrow, nrow),dtype=dtype,device=device)

    return coo_invdiag

def CreateI(nrow, dtype=torch.float64, device='cpu'):
    row_vec = torch.arange(nrow,device=device)
    col_vec = torch.arange(nrow,device=device)
    val_vec = torch.ones(nrow,dtype=dtype,device=device)

    I = torch.sparse_coo_tensor(torch.stack((row_vec,col_vec)),val_vec, (nrow, nrow),dtype=dtype,device=device)

    return I 


class wJacoib:
    def __init__(self, weight=1.0, dtype=torch.float64, device='cpu'):
        self.weight = weight
        self.dtype = dtype
        self.device = device

    def Setup(self, A):
        invdiag = GetInvDiagSpMat(A,self.dtype,self.device)
        I = CreateI(A.shape[0],self.dtype,self.device)

        self.mat = I - self.weight * invdiag @ A
        self.A = A.to(self.device)
        self.invdiag = invdiag

    def Solve(self, b, x):
        x = self.mat @ x + self.weight * self.invdiag @ b
        return x



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
    x = (one-omega2) * x + omega2 * torch.sparse.mm((I - torch.sparse.mm(diag, A)), x) + torch.sparse.mm(diag, b)
            
    return x

#官方算例，结果为5.835
# def test1():
#     A = poisson((10,10), format='csr')
#     diag = torch.tensor(np.linalg.inv(np.diag(A.diagonal()))).to_sparse_coo().to('cuda:0')
#     A = torch.sparse_csr_tensor(A.indptr, A.indices, A.data).to('cuda:0')
#     A = A.to_sparse_coo()
#     # A = torch.sparse_bsr_tensor(A.indptr, A.indices, A.data, dtype=torch.float32)
#     x0 = torch.zeros(A.shape[0],1, dtype=A.dtype).to('cuda:0').requires_grad_(True)
#     b = torch.ones(A.shape[0],1, dtype=A.dtype).to('cuda:0')
#     x = jacobi(A, x0, b, diag, iterations=10, omega=1.0)
#     print('Jacobi计算结果:')
#     print(f'{torch.linalg.norm(b - sparse.mm(A, x)):2.4}')




# #计算gauss_seidel迭代
# def gauss_seidel(A, x, b, iterations=1, sweep='forward'):
#     A, x, b = make_system(A, x, b, formats=['csr', 'bsr'])


#     if A.is_sparse_csr:
#         blocksize = 1
#     else:
#         blocksie = 1
#     #     # R, C = A.blocksize
#     #     # if R != C:
#     #     #     raise ValueError('BSR blocks must be square')
#     #     # blocksize = R

#     if sweep not in ('forward', 'backward', 'symmetric'):
#         raise ValueError('valid sweep directions: "forward", "backward", and "symmetric"')

#     if sweep == 'forward':
#         row_start, row_stop, row_step = 0, int(len(x)/blocksize), 1
#     elif sweep == 'backward':
#         row_start, row_stop, row_step = int(len(x)/blocksize)-1, -1, -1
#     elif sweep == 'symmetric':
#         for _iter in range(iterations):
#             gauss_seidel(A, x, b, iterations=1, sweep='forward')
#             gauss_seidel(A, x, b, iterations=1, sweep='backward')

#     if A.is_sparse_csr:
#         for _iter in range(iterations):
#             gauss_seidel_solver(A.crow_indices(), A.col_indices(), A.values(), x, b,
#                             row_start, row_stop, row_step)
#     # else:
#     #     for _iter in range(iterations):
#     #         amg_core.bsr_gauss_seidel(A.indptr, A.indices, np.ravel(A.data),
#     #                                   x, b, row_start, row_stop, row_step, R)

# def gauss_seidel_solver(indptr, indices, data, x, b, row_start, row_stop, row_step):
#     for i in range(row_start, row_stop, row_step):
#         with torch.no_grad():
#             start = int(indptr[i])
#             end = int(indptr[i+1])
#         rsum = 0
#         diag = 0

#         for jj in range(start, end, 1):
#             with torch.no_grad():
#                 j = int(indices[jj])
#             if i==j:
#                 diag = data[jj].item()
                
#             else:
#                 rsum = rsum + data[jj] * x[j]

#         if diag != 0:
#             x[i] = (b[i]- rsum)/diag
            
#     return x

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
