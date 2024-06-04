import torch
from torch_sparse import SparseTensor
import torchamg
import scipy
import pyamg
import numpy as np

class TrainP(torch.nn.Module):
    def __init__(self,A,p_row,p_col,p_val,nrow,ncol,num):
        super().__init__()
        self.A = A
        self.p_row = p_row
        self.p_col = p_col
        self.p_val = torch.nn.Parameter(p_val,requires_grad=True)
        self.nrow = nrow
        self.ncol = ncol
        self.num = num

    def forward(self,b,x):
        sp_p = SparseTensor(rowptr=self.p_row, col=self.p_col, value=self.p_val, sparse_sizes=(self.nrow,self.ncol))
        dtype = torch.float64

        smoothing_num = 3
        coarse_num = 10
        pre_jacobi = torchamg.wJacobi(dtype=dtype)
        post_jacobi = torchamg.wJacobi(dtype=dtype)
        coarse_jacobi = torchamg.wJacobi(dtype=dtype)
        tg = torchamg.TwoGrid(pre_jacobi,post_jacobi,smoothing_num,coarse_jacobi,coarse_num,dtype)
        tg.Setup(self.A, sp_p)

        for _ in range(self.num):
            x = tg.Solve(b, x)
        return x

def GetData(mat_path,dtype=torch.float64,device='cpu'):
    csr_A = scipy.sparse.load_npz(mat_path)
    ml = pyamg.ruge_stuben_solver(csr_A, max_levels=2, keep=True)
    p = ml.levels[0].P

    sp_A = SparseTensor.from_scipy(csr_A)
    sp_A = sp_A.to(device)

    # change p into torch sparse matrix
    p_row_vec = p.indptr
    p_col_vec = p.indices
    p_val_vec = p.data
    p_row = torch.from_numpy(p_row_vec.astype(np.int64))
    p_col = torch.from_numpy(p_col_vec.astype(np.int64))
    p_val = torch.from_numpy(p_val_vec)
    
    p_row = p_row.to(device)
    p_col = p_col.to(device)
    p_val = p_val.to(dtype).to(device)

    return sp_A, p_row, p_col, p_val, p.shape[0], p.shape[1]

def Train():
    dtype = torch.float64
    # device = 'cuda:0'
    device = 'cpu'
    mat_path = 'poisson_tri10.npz'
    num_iter = 10

    A,p_row, p_col, p_val, nrow, ncol = GetData(mat_path,dtype,device)

    model = TrainP(A,p_row,p_col,p_val,nrow,ncol,num_iter)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    
    nrow = A.size(0)
    x = torch.zeros(nrow,1,dtype=dtype,device=device)
    b = torch.ones(nrow,1,dtype=dtype,device=device)

    optimizer.zero_grad()
    x = model(b,x)
    Ax = A.matmul(x)
    loss = torch.mean((Ax-b)**2)
    loss.backward() 
    optimizer.step()
    print(f"loss = {loss.item()}")
    
if __name__ == '__main__':
    Train()
