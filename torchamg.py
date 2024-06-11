import torch
from torch_sparse import SparseTensor
import time

def GetInvDiagSpMat(csr_A, dtype=torch.float64, device='cpu'):
    diag = csr_A.get_diag()
    invdiag = 1.0 / diag

    nrow = csr_A.size(0)
    row_vec = torch.arange(nrow,device=device)
    col_vec = torch.arange(nrow,device=device)
    sp = SparseTensor(row=row_vec, col=col_vec, value=invdiag, sparse_sizes=(nrow,nrow))
    sp = sp.to(device)

    return sp

class wJacobi:
    def __init__(self, weight=1.0, dtype=torch.float64, device='cpu'):
        self.weight = weight
        self.dtype = dtype
        self.device = device

    def Setup(self, A):
        invdiag = GetInvDiagSpMat(A,self.dtype,self.device)

        # self.mat = I - self.weight * (invdiag @ A)
        self.mat = invdiag.matmul(A)
        self.A = A.to(self.device)
        self.invdiag = invdiag

    def Solve(self, b, x):
        res = x - self.weight * self.mat.matmul(x) + self.weight * self.invdiag.matmul(b)
        return res

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
        R = p.t()
        A_c = R.matmul(A).matmul(p) 

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
        # x = torch.linalg.solve(self.dense_A_c, b)
        return x

    def Solve(self, b, x):
        if len(b.shape) == 1:
            b = b.unsqueeze(1)

        for _ in range(self.smoothing_num):
            x = self.pre_smoother.Solve(b, x)

        residual = b - self.A.matmul(x)

        coarse_b = self.R.matmul(residual)
        coarse_x = self.CoarseSolve(coarse_b, torch.zeros(coarse_b.shape,dtype=self.dtype,device=self.device) )
        xx = x + self.P.matmul(coarse_x)

        for _ in range(self.smoothing_num):
            xx = self.post_smoother.Solve(b, xx)

        return xx
    def Multi_Solve(self, b, x, max_iter=100, threshold=1e-4):
        error = threshold+1
        iters = 0
        iter_time = 0
        while (error>threshold) and (iters<max_iter):
            t1 = time.perf_counter()
            x = self.Solve(b, x)
            t2 = time.perf_counter()
            iter_time+= t2-t1
            # mse error 
            error = torch.sqrt(torch.mean((self.A@x -b)**2))/torch.norm(b)
            iters+=1
        
        return x, iters, error, iter_time
    

class Level:
    def __init__(self,R=None,A=None,P=None,dtype=torch.float64,device='cpu'):
        self.R = R.to(device)
        self.A = A.to(device)
        self.P = P.to(device)
        self.dtype = dtype
        self.device = device

        self.pre_smoother = None
        self.pre_num_iter = 0
        
        self.post_smoother = None
        self.post_num_iter = 0

    def SetPreSmt(self,pre_smoother,num_iter):
        self.pre_smoother = pre_smoother
        self.pre_smoother.Setup(self.A)
        self.pre_num_iter = num_iter

    def SetPostSmt(self,post_smoother,num_iter):
        self.post_smoother = post_smoother
        self.post_smoother.Setup(self.A)
        self.post_num_iter = num_iter

class MultiGrid:
    def __init__(self, levels, coarse_solver, coarse_iter_num, dtype=torch.float64, device='cpu'):
        self.levels = levels
        self.coarse_solver = coarse_solver
        self.coarse_iter_num = coarse_iter_num
        self.dtype = dtype
        self.device = device

    def CoarseSolve(self, b, x):
        for _ in range(self.coarse_num):
            x = self.coarse_solver.Solve(b, x)
        # x = torch.linalg.solve(self.dense_A_c, b)
        return x

    def Iterate(self,b,x,lvl_idx=0):
        if lvl_idx == len(self.levels)-1:
            x = self.CoarseSolve(b,x)
            return x

        for _ in range(self.levels[lvl_idx].pre_num_iter):
            x = self.levels[lvl_idx].pre_smoother.Solve(b, x)

        residual = b - self.levels[lvl_idx].A.matmul(x)
        coarse_b = self.levels[lvl_idx].R.matmul(residual)
        coarse_x = torch.zeros(coarse_b.shape,dtype=self.dtype,device=self.device)

        coarse_x = self.Iterate(coarse_b,coarse_x,lvl_idx+1)
        x = x + self.levels[lvl_idx].P.matmul(coarse_x)

        for _ in range(self.levels[lvl_idx].post_num_iter):
            x = self.levels[lvl_idx].post_smoother.Solve(b, x)

        return x

    def Solve(self, b, x, max_iter=100, rtol=1e-3):
        pass




def CreateI(nrow, dtype=torch.float64, device='cpu'):
    row_vec = torch.arange(nrow,device=device)
    col_vec = torch.arange(nrow,device=device)
    val_vec = torch.ones(nrow,dtype=dtype,device=device)

    I = torch.sparse_coo_tensor(torch.stack((row_vec,col_vec)),val_vec, (nrow, nrow),dtype=dtype,device=device)
    I = I.to_sparse_csr()

    return I 


def GetTriMat(coo_A, dtype=torch.float64, device='cpu'):
    coo = coo_A.coalesce()
    row_vec, col_vec = coo.indices()
    val_vec = coo.values()
    nrow = coo.shape[0]
    
    mask = row_vec >= col_vec
    row_lowertri, col_lowertri = row_vec[mask], col_vec[mask]
    lowertri_val = torch.zeros(len(row_lowertri),dtype=dtype,device=device)
    lowertri_val = val_vec[mask]
    coo_lowertri = torch.sparse_coo_tensor(torch.stack((row_lowertri,col_lowertri)), lowertri_val, (nrow, nrow),dtype=dtype,device=device)

    mask = row_vec <= col_vec
    row_uppertri, col_uppertri = row_vec[mask], col_vec[mask]
    uppertri_val = torch.zeros(len(row_uppertri),dtype=dtype,device=device)
    uppertri_val = val_vec[mask]
    
    coo_uppertri = torch.sparse_coo_tensor(torch.stack((row_uppertri,col_uppertri)), uppertri_val, (nrow, nrow),dtype=dtype,device=device)
    return coo_lowertri, coo_uppertri

def GetInvLowerTriSpMat(coo_A, dtype=torch.float64, device='cpu'):
    diag = GetDiagVec(coo_A, dtype, device)
    lowtri, uptri = GetTriMat(coo_A, dtype, device=device)
    coo = lowtri.coalesce()
    row_vec, col_vec = coo.indices()
    val_vec = coo.values()
    nrow = coo_A.shape[0]
    invdiag = 1.0 / diag
    invlowtri_val = torch.zeros(len(val_vec),dtype=dtype,device=device)
    for i in np.arange(len(row_vec)-1, -1, -1):
        if row_vec[i] == col_vec[i]:
            invlowtri_val[i] = invdiag[row_vec[i]]
            current=i
        else:
            col_vec_temp = col_vec[:current+1]
            val_temp = val_vec[torch.where(col_vec_temp==col_vec_temp[i])]
            a = invdiag[int(col_vec[i])]
            
            invlowtri_val[i] = -torch.dot(val_temp[1:], invlowtri_val[i+1:current+1]) * invdiag[int(col_vec[i])]

    coo_invlowtri = torch.sparse_coo_tensor(torch.stack((row_vec,col_vec)),invlowtri_val, (nrow, nrow),dtype=dtype,device=device)

    coo = uptri.coalesce()
    row_vec, col_vec = coo.indices()
    val_vec = coo.values()
    mask = col_vec > row_vec
    coo_uptri = torch.sparse_coo_tensor(torch.stack((row_vec[mask],col_vec[mask])),val_vec[mask], (nrow, nrow),dtype=dtype,device=device)
    return coo_invlowtri, coo_uptri


class Conjugate_gradient:
    def __init__(self, dtype=torch.float64, device='cpu'):
        self.dtype = dtype
        self.device = device

    def Setup(self, A, b, x):
        self.A = A.to(self.device)
        self.b = b
        self.r = A @ x - b
        self.p = -self.r

    def Solve(self, b, x):
        print(self.A @ self.p)
        alpha = -(self.r.t() @ self.p) / (self.p.t() @ (self.A @ self.p))
        x = x + alpha * self.p
        self.r = self.A @ x - b
        beta = self.r.t() @ (self.A @ self.p) / (self.p.t() @ (self.A @ self.p))
        self.p = -self.r + beta * self.p
        return x

class Gauss_Seidel:
    def __init__(self, dtype=torch.float64, device='cpu'):
        self.dtype = dtype
        self.device = device

    def Setup(self, A):
        invlowtri, uptri = GetInvLowerTriSpMat(A ,self.dtype,self.device)
        I = CreateI(A.shape[0],self.dtype,self.device)
        self.mat = -invlowtri @ uptri
        self.A = A.to(self.device)
        self.invlowtri = invlowtri

    def Solve(self, b, x):
        x = self.mat @ x + self.invlowtri @ b
        return x



