import torch
from torch_sparse import SparseTensor
import numpy as np
import scipy
from scipy.sparse import coo_matrix


class SparseSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, b):
        '''
        A is a torch coo sparse matrix
        b is a tensor
        '''
        if A.ndim != 2 or (A.shape[0] != A.shape[1]):
            raise ValueError("A should be a square 2D matrix.")

        A = A.coalesce()
        A_idx = A.indices().to('cpu').numpy()
        A_val = A.values().to('cpu').numpy()
        sci_A = coo_matrix((A_val,(A_idx[0,:],A_idx[1,:]) ),shape=A.shape)
        sci_A = sci_A.tocsr()

        np_b = b.detach().cpu().numpy()
        # Solver the sparse system
        if np_b.ndim == 1:
            np_x = scipy.sparse.linalg.spsolve(sci_A, np_b)
        else:
            factorisedsolver = scipy.sparse.linalg.factorized(sci_A)
            np_x = factorisedsolver(np_b)

        # Transfer (dense) result back to PyTorch
        x = torch.as_tensor(np_x)
        # Not sure if the following is needed / helpful
        if A.requires_grad or b.requires_grad:
            x.requires_grad = True

        # Save context for backward pass
        ctx.save_for_backward(A, b, x)
        return x

    @staticmethod
    def backward(ctx, grad):
        # Recover context
        A, b, x = ctx.saved_tensors

        gradb = SparseSolve.apply(A.t(), grad)

        gradAidx = A.indices()
        mgradbselect = -gradb.index_select(0,gradAidx[0,:])
        xselect = x.index_select(0,gradAidx[1,:])
        mgbx = mgradbselect * xselect
        if x.dim() == 1:
            gradAvals = mgbx
        else:
            gradAvals = torch.sum( mgbx, dim=1 )
        gradA = torch.sparse_coo_tensor(gradAidx, gradAvals, A.shape)

        return gradA, gradb

sparsesolve = SparseSolve.apply

def TestSparseMatVec(Aref,bref):
    print('=============================================')
    print('test sparse matrix * vector solver')
    # Test matrix-vector solver
    A = Aref.detach().clone().requires_grad_(True)
    b = bref.detach().clone().requires_grad_(True)

    # Solve
    x = sparsesolve(A,b)

    # random scalar function to mimick a loss
    loss = x.sum()
    loss.backward()

    print('x',x)
    with torch.no_grad(): 
        print('allclose:',torch.allclose(A @ x, b))


if __name__ == '__main__':
    row_vec = torch.tensor([0, 0, 1, 2])
    col_vec = torch.tensor([0, 2, 1, 2])
    val_vec = torch.tensor([3.0, 4.0, 5.0, 6.0],dtype=torch.float64)

    Aref = torch.sparse_coo_tensor(torch.stack((row_vec,col_vec),0), val_vec, (3, 3))
    bref = torch.ones(3, dtype=torch.float64)
    
    TestSparseMatVec(Aref,bref)
