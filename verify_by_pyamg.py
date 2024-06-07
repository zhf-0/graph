import pyamg
from pyamg.relaxation.relaxation import jacobi
from pyamg.util.linalg import norm
from pyamg.multilevel import MultilevelSolver
from pyamg.strength import classical_strength_of_connection
from pyamg.classical.interpolate import direct_interpolation
from pyamg.classical.split import RS
from pyamg.relaxation.smoothing import change_smoothers


def PyAMGTG(A,P,x,b,n=1):
    R = P.T
    # C = classical_strength_of_connection(A)
    # splitting = RS(A)
    # P = direct_interpolation(A, C, splitting)
    # R = P.T

    levels = []
    levels.append(MultilevelSolver.Level())
    levels.append(MultilevelSolver.Level())
    levels[0].A = A
    # levels[0].C = C
    # levels[0].splitting = splitting
    levels[0].P = P
    levels[0].R = R

    levels[1].A = R @ A @ P 
    
    # choose coarse grid solver
    # coarse = ('jacobi', {'iterations' : 10,'omega':1.0,'withrho':False})
    coarse = 'splu'
    ml = MultilevelSolver(levels, coarse_solver=coarse)
    
    # choose smoother
    # smoother = ('jacobi', {'iterations' : 3,'omega':1.0,'withrho':False})
    smoother = ('jacobi', {'iterations' : 3,'omega':1.0})
    smoother = ('block_jacobi',{'iterations' : 3})
    smoother = ('gauss_seidel',{'iterations' : 3})
    change_smoothers(ml, smoother, smoother)

    residual = []
    accel = 'gmres'
    # final_x = ml.solve(b=b,x0=x,maxiter=n)
    final_x = ml.solve(b=b,x0=x,tol=1e-3,maxiter=n,residuals=residual)
    # final_x = ml.solve(b=b,x0=x,tol=1e-3,maxiter=n,accel=accel,residuals=residual)

    return final_x

def FilterP(p):
    '''
    change each row of matrix p
    '''
    return p

def MultiLevel(A,num_level,b,x,maxiter,rtol,min_coarse_num=100):
    assert num_level >= 2

    C = classical_strength_of_connection(A) 
    splitting = RS(A)
    P = direct_interpolation(A, C, splitting)
    P = FilterP(P)
    R = P.T.tocsr()

    levels = []
    levels.append(MultilevelSolver.Level())
    levels[0].A = A
    levels[0].P = P
    levels[0].R = R

    for i in range(1,num_level):
        levels.append(MultilevelSolver.Level())
        levels[i].A = levels[i-1].R @ levels[i-1].A @ levels[i-1].P 
        if levels[i].A.shape[0] > min_coarse_num:
            C = classical_strength_of_connection(levels[i].A)
            splitting = RS(levels[i].A)
            P = direct_interpolation(levels[i].A, C, splitting)
            P = FilterP(P)
            R = P.T.tocsr()
            levels[i].P = P
            levels[i].R = R
        else:
            break

            
    # choose coarse grid solver
    # coarse = ('jacobi', {'iterations' : 10,'omega':1.0,'withrho':False})
    coarse = 'splu'
    ml = MultilevelSolver(levels, coarse_solver=coarse)
    
    # choose smoother
    # smoother = ('jacobi', {'iterations' : 3,'omega':1.0,'withrho':False})
    smoother = ('jacobi', {'iterations' : 3,'omega':1.0})
    smoother = ('block_jacobi',{'iterations' : 3})
    smoother = ('gauss_seidel',{'iterations' : 3})
    change_smoothers(ml, smoother, smoother)

    residual = []
    accel = 'gmres'
    # final_x,info = ml.solve(b=b,x0=x,maxiter=n,return_info=True)
    final_x,info = ml.solve(b=b,x0=x,tol=rtol,maxiter=maxiter,residuals=residual,return_info=True)
    # final_x,info = ml.solve(b=b,x0=x,tol=rtol,maxiter=n,accel=accel,residuals=residual,return_info=True)

    return final_x,info

def TestMulti():
    import scipy
    import numpy as np

    A = scipy.sparse.load_npz('./MatData/scipy_csr1.npz')
    b = np.load('./MatData/b1.npy') 
    x = np.zeros(A.shape[0])
    maxiter = 100
    rtol = 1e-3
    num_level = 7

    x,info = MultiLevel(A,num_level,b,x,rtol,maxiter)
    if info == 0:
        print('success')
    else:
        print('fail')

if __name__ == '__main__':
    TestMulti()
