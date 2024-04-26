import torch
import torchamg
import pyamg


def TestCG():
    '''
    A = [2, 1]
        [1, 2] 
    '''
    index = torch.tensor([ [0,0,1,1],[0,1,0,1] ])
    val = torch.tensor([2.0,1.0,1.0,2.0],dtype=torch.float64)
    coo = torch.sparse_coo_tensor(index, val, (2,2),dtype=torch.float64 )

    b = torch.tensor([1.0,1.0],dtype=torch.float64).unsqueeze(1)
    x = torch.tensor([0.0,0.0],dtype=torch.float64).unsqueeze(1)
    
    jac = torchamg.Conjugate_gradient()
    jac.Setup(coo, b, x)
    # for i in range(10):
        # x = jac.Solve(b,x)
    x = jac.Solve(b,x)
    print('CG')
    print(x)

def TestJacobi():
    '''
    A = [2, 1]
        [1, 2] 
    '''
    index = torch.tensor([ [0,0,1,1],[0,1,0,1] ])
    val = torch.tensor([2.0,1.0,1.0,2.0],dtype=torch.float64)
    coo = torch.sparse_coo_tensor(index, val, (2,2),dtype=torch.float64 )

    b = torch.tensor([1.0,1.0],dtype=torch.float64).unsqueeze(1)
    x = torch.tensor([0.0,0.0],dtype=torch.float64).unsqueeze(1)
    
    jac = torchamg.wJacobi(weight=0.5)
    jac.Setup(coo)
    for i in range(10):
        x = jac.Solve(b,x)
    print('jacobi')
    print(x)

def TestGS():
    '''
    A = [2, 1]
        [1, 2] 
    '''
    index = torch.tensor([ [0,0,1,1],[0,1,0,1] ])
    val = torch.tensor([2.0,1.0,1.0,2.0],dtype=torch.float64)
    coo = torch.sparse_coo_tensor(index, val, (2,2),dtype=torch.float64 )

    b = torch.tensor([1.0,1.0],dtype=torch.float64).unsqueeze(1)
    x = torch.tensor([0.0,0.0],dtype=torch.float64).unsqueeze(1)
    
    jac = torchamg.Gauss_Seidel()
    jac.Setup(coo)
    for i in range(10):
        x = jac.Solve(b,x)
    print('GS')
    print(x)

def Testinvlow():
    index = torch.tensor([ [0,1,1,2,2,2,3,3,3,3],[0,0,1,0,1,2,0,1,2,3] ])
    val = torch.tensor([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0],dtype=torch.float64)
    coo = torch.sparse_coo_tensor(index, val, (4,4),dtype=torch.float64 )
    coo_low, coo_up = torchamg.GetTriMat(coo)
    coo_invlow, coo_up1 = torchamg.GetInvLowerTriSpMat(coo_low)
    print(coo_invlow.to_dense() @ coo_low.to_dense())

if __name__ == '__main__':
    TestJacobi()
    TestCG()
