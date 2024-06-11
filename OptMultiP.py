import torch
from torch_sparse import SparseTensor
import torchamg
import scipy
import pyamg
import numpy as np

class TrainP(torch.nn.Module):
    def __init__(self,A,list_p,num):
        super().__init__()
        self.A = A
        self.list_p = list_p
        self.para_p_val_list = []

        self.num_level = len(list_p)
        assert self.num_level >= 2

        for i in range(self.num_level-1):
            tmp = torch.nn.Parameter(self.list_p[i][2],requires_grad=True)
            self.para_p_val_list.append(tmp)

        self.num = num
        self.levels = []

    def forward(self,b,x):
        dtype = torch.float64
        smoothing_num = 3

        P = SparseTensor(rowptr=self.list_p[0][0], col=self.list_p[0][1], value=self.para_p_val_list[0], sparse_sizes=(self.list_p[0][3],self.list_p[0][4]))
        R = P.t()
        level = torchamg.Level(R,self.A,P,device=self.device)
        pre_jacobi = torchamg.wJacobi(weight=2/3,dtype=dtype,device=self.device)
        post_jacobi = torchamg.wJacobi(weight=2/3,dtype=dtype,device=self.device)
        level.SetPreSmt(pre_jacobi,smoothing_num)
        level.SetPostSmt(post_jacobi,smoothing_num)
        self.levels.append(level)

        # set coarse solver
        coarse_num = 10
        coarse_jacobi = torchamg.wJacobi(weight=2/3,dtype=dtype,device=self.device)

        for i in range(1,self.num_level):
            A_c = self.levels[i-1].R.matmul(self.levels[i-1].A).matmul(self.levels[i-1].P)
            if i == (self.num_level - 1):
                level = torchamg.Level(A=A_c,device=self.device)
                coarse_jacobi.Setup(A_c)
            else:
                P = SparseTensor(rowptr=self.list_p[i][0], col=self.list_p[i][1], value=self.para_p_val_list[i], sparse_sizes=(self.list_p[i][3],self.list_p[i][4]))
                R = P.t()
                level = torchamg.Level(R,A_c,P,device=self.device)
                pre_jacobi = torchamg.wJacobi(weight=2/3,dtype=dtype,device=self.device)
                post_jacobi = torchamg.wJacobi(weight=2/3,dtype=dtype,device=self.device)
                level.SetPreSmt(pre_jacobi,smoothing_num)
                level.SetPostSmt(post_jacobi,smoothing_num)

            self.levels.append(level)


        mg = torchamg.MultiGrid(self.levels,coarse_jacobi,coarse_num,device=self.device)

        for _ in range(self.num):
            x = mg.Iterate(b, x)

        return x

def GetData(mat_path,dtype=torch.float64,device='cpu'):
    csr_A = scipy.sparse.load_npz(mat_path)
    ml = pyamg.ruge_stuben_solver(csr_A, keep=True)
    num_level = len(ml.levels)
    p = ml.levels[0].P

    sp_A = SparseTensor.from_scipy(csr_A)
    sp_A = sp_A.to(device)

    # change p into torch sparse matrix
    list_p = []
    for i in range(1,num_level-1):
        p = ml.levels[i].P
        p_row_vec = p.indptr
        p_col_vec = p.indices
        p_val_vec = p.data
        p_row = torch.from_numpy(p_row_vec.astype(np.int64))
        p_col = torch.from_numpy(p_col_vec.astype(np.int64))
        p_val = torch.from_numpy(p_val_vec)
        
        p_row = p_row.to(device)
        p_col = p_col.to(device)
        p_val = p_val.to(dtype).to(device)

        list_p.append( [p_row,p_col,p_val,p.shape[0],p.shape[1]] )

    return sp_A, list_p

def Train():
    dtype = torch.float64
    # device = 'cuda:0'
    device = 'cpu'
    mat_path = 'poisson_tri10.npz'
    num_iter = 10

    A,list_p = GetData(mat_path,dtype,device)

    model = TrainP(A,list_p,num_iter)

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
