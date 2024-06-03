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

if __name__ == '__main__':
    pass
