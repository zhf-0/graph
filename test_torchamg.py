import torch
import torchamg
import pyamg
import numpy as np


def TestCG():
    '''
    A = [2, 1]
        [1, 2] 
    '''
    # test grad
    index = torch.tensor([ [0,0,1,1],[0,1,0,1] ])
    val = torch.tensor([2.0,1.0,1.0,2.0],dtype=torch.float64)
    coo = torch.sparse_coo_tensor(index, val, (2,2),dtype=torch.float64, requires_grad=True)

    b = torch.tensor([1.0,1.0],dtype=torch.float64).unsqueeze(1)
    b = b.clone().detach().requires_grad_(True)
    x = torch.tensor([0.0,0.0],dtype=torch.float64).unsqueeze(1)
    x = x.clone().detach().requires_grad_(True)
    cg = torchamg.Conjugate_gradient()
    cg.Setup(coo, b, x)
    # for i in range(10):
        # x = jac.Solve(b,x)
    out = cg.Solve(b,x)
    print('CG')
    out = out.mean()
    out.backward()
    print('out: ', out)
    print('x is leaf: ',x.is_leaf)
    print('x grad: ', x.grad)

    # test large matrix
    A_path='./IterMat/scipy_csr0.npz' 
    b_path='./IterMat/b0.npy'
    A_csr = np.load(A_path, mmap_mode='r')
    A_coo = torch.sparse_csr_tensor(crow_indices=A_csr['indptr'],
                                    col_indices=A_csr['indices'],
                                    values=A_csr['data']).to_sparse_coo()

    b = np.load(b_path, mmap_mode='r')
    b = torch.tensor(b,dtype=torch.float64).unsqueeze(1)
    x_torch = torch.zeros_like(b)
    cg = torchamg.Conjugate_gradient()
    cg.Setup(A_coo, b, x_torch)
    # for i in range(10):
        # x = jac.Solve(b,x)
    out = cg.Solve(b,x_torch)
    x_torch = out.detach().numpy().squeeze(1)

    ## pyamg part
    A = A_coo.to_dense().detach().numpy()
    b = b.detach().numpy()
    x0 = np.zeros_like(b)
    x_pyamg,info = pyamg.krylov.cg(A,b,x0)
    # print(x_pyamg)
    # print(b)
    print("torch: ",np.mean(np.abs(A@x_torch-b)))
    print("pyamg: ",np.mean(np.abs(A@x_pyamg-b)))
    print("MAE: ",np.mean(np.abs(x_torch-x_pyamg)))
    # print(np.isclose(out,xk,atol=1e-4))

def TestJacobi():

    ## test gradiant part
    index = torch.tensor([ [0,0,1,1],[0,1,0,1] ])
    val = torch.tensor([2.0,1.0,1.0,2.0],dtype=torch.float64)
    coo = torch.sparse_coo_tensor(index, val, (2,2),dtype=torch.float64 ,requires_grad=True)

    b = torch.tensor([1.0,1.0],dtype=torch.float64).unsqueeze(1)
    b = b.clone().detach().requires_grad_(True) 
    x_torch = torch.tensor([0.0,0.0],dtype=torch.float64).unsqueeze(1)
    x_torch = x_torch.clone().detach().requires_grad_(True) 

    jac = torchamg.wJacobi(weight=0.5,device='cpu')
    jac.Setup(coo)
    # for i in range(10):
    out = jac.Solve(b,x_torch)
    # for i in range(9):
    #     out = jac.Solve(b,out)
    out = out.mean()
    out.backward()
    print('out: ', out)
    print('x is leaf: ',x_torch.is_leaf)
    print('x grad: ', x_torch.grad)

    # # compare with pyamg 
    # A_path='./IterMat/scipy_csr0.npz' 
    # b_path='./IterMat/b0.npy'
    # A_csr = np.load(A_path, mmap_mode='r')
    # A_coo = torch.sparse_csr_tensor(crow_indices=A_csr['indptr'],
    #                                 col_indices=A_csr['indices'],
    #                                 values=A_csr['data']).to_sparse_coo()

    # b = np.load(b_path, mmap_mode='r')
    # b = torch.tensor(b,dtype=torch.float64).unsqueeze(1)
    # x_torch = torch.zeros_like(b)
    
    # jac = torchamg.wJacobi(weight=0.5,device='cpu')
    # jac.Setup(A_coo)
    # for i in range(10):
    #     x_torch = jac.Solve(b,x_torch)
    
    # # pyamg part
    # b = np.array(b).reshape(-1,1).astype(np.float64)
    # x_pyamg = np.zeros((100,1), dtype=np.float64)
    # x_temp = x_pyamg.copy()
    # Ap = A_csr['indptr']
    # Ap = Ap.astype(np.int32)
    # Aj = A_csr['indices']
    # Aj = Aj.astype(np.int32)
    # Ax=A_csr['data']
    # Ax = Ax.astype(np.float64)
    # omiga = np.array(0.5,dtype=np.float64)
    # pyamg.amg_core.jacobi(Ap=Ap, Aj=Aj, Ax=Ax,x=x_pyamg,b=b,omiga=omiga)# BUG
    # x_torch = x_torch.detach().numpy()
    # print(np.isclose(x_torch, x_pyamg, atol=1e-4).all())
    return x_torch

def TestGS():
    '''
    A = [2, 1]
        [1, 2] 
    '''
    index = torch.tensor([ [0,0,1,1],[0,1,0,1] ])
    val = torch.tensor([2.0,1.0,1.0,2.0],dtype=torch.float64)
    coo = torch.sparse_coo_tensor(index, val, (2,2),dtype=torch.float64,requires_grad=True )

    b = torch.tensor([1.0,1.0],dtype=torch.float64).unsqueeze(1)
    b = b.clone().detach().requires_grad_(True)
    x = torch.tensor([0.0,0.0],dtype=torch.float64).unsqueeze(1)
    x = x.clone().detach().requires_grad_(True)    
    jac = torchamg.Gauss_Seidel()
    jac.Setup(coo)
    out = jac.Solve(b,x)
    for i in range(9):
        out = jac.Solve(b,out)
    
    print('GS')
    out = out.mean()
    out.backward()
    print('out: ', out)
    print('x is leaf: ',x.is_leaf)
    print('x grad: ', x.grad)

    print(x)

def Testinvlow():
    index = torch.tensor([ [0,1,1,2,2,2,3,3,3,3],[0,0,1,0,1,2,0,1,2,3] ])
    val = torch.tensor([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0],dtype=torch.float64)
    coo = torch.sparse_coo_tensor(index, val, (4,4),dtype=torch.float64 )
    coo_low, coo_up = torchamg.GetTriMat(coo)
    coo_invlow, coo_up1 = torchamg.GetInvLowerTriSpMat(coo_low)
    print(coo_invlow.to_dense() @ coo_low.to_dense())

if __name__ == '__main__':
    # TestGS()
    TestJacobi()
