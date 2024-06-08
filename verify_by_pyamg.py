import pyamg
from pyamg.relaxation.relaxation import jacobi
from pyamg.util.linalg import norm
from pyamg.multilevel import MultilevelSolver
from pyamg.strength import classical_strength_of_connection
from pyamg.classical.interpolate import direct_interpolation,classical_interpolation
from pyamg.classical.split import RS
from pyamg.relaxation.smoothing import change_smoothers
import numpy as np
import scipy
import scipy.sparse
import time
import matplotlib.pyplot as plt

def PyAMGTG(A,P,x,b,n=1, tol=1e-4):
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
    coarse = ('jacobi', {'iterations' : 10,'omega':1.0,'withrho':False})
    # coarse = 'splu'
    ml = MultilevelSolver(levels, coarse_solver=coarse)
    
    # choose smoother
    smoother = ('jacobi', {'iterations' : 5,'omega':1.0,'withrho':False})
    # smoother = ('jacobi', {'iterations' : 3,'omega':1.0})
    # smoother = ('block_jacobi',{'iterations' : 3})
    # smoother = ('gauss_seidel',{'iterations' : 3})
    change_smoothers(ml, smoother, smoother)

    residual = []
    accel = 'gmres'
    # final_x = ml.solve(b=b,x0=x,maxiter=n)
    final_x = ml.solve(b=b,x0=x,tol=tol,maxiter=n,residuals=residual)
    # final_x = ml.solve(b=b,x0=x,tol=1e-3,maxiter=n,accel=accel,residuals=residual)
    iter_step = len(residual)
    res = residual[-1]
    return final_x, iter_step, res
def GenerateP(A):
    ml = pyamg.ruge_stuben_solver(A, max_levels=2, keep=True)
    p = ml.levels[0].P

    return p

def spmat_set_1toRowMax_0Else(mat):
    mat = mat.tocsr()
    rows_start = mat.indptr[:-1]
    rows_end = mat.indptr[1:]

    new_data = []
    new_indices = []
    new_indptr = [0]

    for start, end in zip(rows_start, rows_end):
        if start < end:
            row_data = mat.data[start:end]
            row_indices = mat.indices[start:end]
            max_idx = np.argmax(row_data)
            new_data.append(1)
            new_indices.append(row_indices[max_idx])
        new_indptr.append(len(new_data))
    out_mat = scipy.sparse.csr_matrix((new_data, new_indices, new_indptr), shape=mat.shape)
    return out_mat

def FilterP(p):
    '''
    change each row of matrix p
    '''
    # p = spmat_set_1toRowMax_0Else(p)
    return p

def MultiLevel(A,num_level,b,x,maxiter,rtol,fn_filter,min_coarse_num=100):
    assert num_level >= 2

    C = classical_strength_of_connection(A) 
    splitting = RS(A)
    P = classical_interpolation(A, C, splitting)
    P = fn_filter(P)
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
            P = classical_interpolation(levels[i].A, C, splitting)
            P = fn_filter(P)
            R = P.T.tocsr()
            # autotune
            # if (R@levels[i].A@P).nnz/P.shape[1] >levels[i].A.nnz/A.shape[1]:
            #     P = fn_filter(P)
            #     R = P.T.tocsr()
            levels[i].P = P
            levels[i].R = R
        else:
            break

            
    # choose coarse grid solver
    coarse = ('jacobi', {'iterations' : 10,'omega':1.0,'withrho':False})
    # coarse = 'splu'
    ml = MultilevelSolver(levels, coarse_solver=coarse)
    
    # choose smoother
    # smoother = ('jacobi', {'iterations' : 3,'omega':1.0,'withrho':False})
    smoother = ('jacobi', {'iterations' : 3,'omega':1.0})
    # smoother = ('block_jacobi',{'iterations' : 3})
    # smoother = ('gauss_seidel',{'iterations' : 3})
    change_smoothers(ml, smoother, smoother)

    residual = []
    accel = 'gmres'
    # final_x = ml.solve(b=b,x0=x,maxiter=n)
    final_x = ml.solve(b=b,x0=x,tol=1e-3,maxiter=maxiter,residuals=residual)
    # final_x = ml.solve(b=b,x0=x,tol=1e-3,maxiter=n,accel=accel,residuals=residual)
    iter_step = len(residual)
    res = residual[-1]
    return final_x, iter_step, res
def GenerateP(A):
    ml = pyamg.ruge_stuben_solver(A, max_levels=2, keep=True)
    p = ml.levels[0].P

    return p

def spmat_set_1toRowMax_0Else(mat):
    mat = mat.tocsr()
    rows_start = mat.indptr[:-1]
    rows_end = mat.indptr[1:]

    new_data = []
    new_indices = []
    new_indptr = [0]

    for start, end in zip(rows_start, rows_end):
        if start < end:
            row_data = mat.data[start:end]
            row_indices = mat.indices[start:end]
            max_idx = np.argmax(row_data)
            new_data.append(1)
            new_indices.append(row_indices[max_idx])
        new_indptr.append(len(new_data))
    out_mat = scipy.sparse.csr_matrix((new_data, new_indices, new_indptr), shape=mat.shape)
    return out_mat


def run_multi():
    ori_iter_list = []
    opt_iter_list = []
    ori_time_list = []
    opt_time_list = []
    num = 10
    for i in range(num):
        A = scipy.sparse.load_npz(f'/work/graph/openfoam/oneraM/csr{i+1}.npz')
        
        # P_opt = scipy.sparse.load_npz(f"/work/get_optimal_P/result/p_opt{i+1}.npz")
        P = GenerateP(A)
        P_opt = spmat_set_1toRowMax_0Else(P)
        # np.random.seed(1)
        x = np.zeros(A.shape[0])
        b =  np.load(f"/work/graph/openfoam/oneraM/b{i+1}.npy")
        # b = np.ones(A.shape[0])
        b_norm = np.linalg.norm(b)
        t1 = time.time()
        _,steps, res = PyAMGTG(A,P,x,b,n=1000,tol=1e-3)
        t2 = time.time()
        ori_iter_list.append(steps)
        ori_time_list.append(t2-t1)
        print("+"*20)
        print(f"num: {i+1} P_ori iterations: {steps}, residual: {res/b_norm}, time used: {t2-t1:.4f}s")
        t3 = time.time()
        _,steps, res = PyAMGTG(A,P_opt,x,b,n=1000,tol=1e-3)
        t4 = time.time()
        opt_iter_list.append(steps)
        opt_time_list.append(t4-t3)
        print(f"num: {i+1} P_opt iterations: {steps}, residual: {res/b_norm}, time used: {t4-t3:.4f}s")
        print('-'*20)
    print(f"total num: {num}| P_ori mean iterations: {np.mean(ori_iter_list):.2f}, mean time used: {np.mean(ori_time_list):.4f}s")
    print(f"total num: {num}| P_opt mean iterations: {np.mean(opt_iter_list):.2f}, mean time used: {np.mean(opt_time_list):.4f}s")
    plt.plot(ori_iter_list,marker='o',label="P_ori")
    plt.plot(opt_iter_list,marker='o',label="P_opt")
    plt.xlabel("Mat id")
    plt.ylabel("Iterations")
    plt.legend()
    plt.savefig("./Iterations.png")
    plt.figure()
    plt.plot(ori_time_list,marker='o',label="P_ori")
    plt.plot(opt_time_list,marker='o',label="P_opt")
    plt.xlabel("Mat id")
    plt.ylabel("Time used")
    plt.legend()
    plt.savefig("./Time.png")
def run_one():
    A = scipy.sparse.load_npz(f'/work/get_optimal_P/data/poisson_tri.npz')

    P_opt = scipy.sparse.load_npz(f"/work/get_optimal_P/p_opt.npz")
    P = scipy.sparse.load_npz(f"/work/get_optimal_P/p.npz")
    P_opt_post = scipy.sparse.load_npz(f"/work/get_optimal_P/p_post.npz")
    # P = GenerateP(A)
    # P_opt = spmat_set_1toRowMax_0Else(P)
    # np.random.seed(1)
    x = np.zeros(A.shape[0])
    # b =  np.load(f"/work/graph/openfoam/matvec8000/b1.npy")
    b = np.ones(A.shape[0])
    b_norm = np.linalg.norm(b)
    _,steps, res = PyAMGTG(A,P,x,b,n=1000,tol=1e-3)
    print("+"*20)
    print(f"P_ori iterations: {steps}, residual: {res/b_norm}")
    _,steps, res = PyAMGTG(A,P_opt,x,b,n=1000,tol=1e-3)
    print(f"P_opt iterations: {steps}, residual: {res/b_norm}")
    _,steps, res = PyAMGTG(A,P_opt_post,x,b,n=1000,tol=1e-3)
    print(f"P_post iterations: {steps}, residual: {res/b_norm}")
    print('-'*20)
def run_multi_mg():
    ori_iter_list = []
    opt_iter_list = []
    ori_time_list = []
    opt_time_list = []
    num = 2
    fn_filter = FilterP
    fn_filter2 = spmat_set_1toRowMax_0Else
    rtol = 1e-3
    num_level = 7
    maxiter = 200
    for i in range(num):
        A = scipy.sparse.load_npz(f'/work/graph/openfoam/motor/csr{i+1}.npz')
        
        # P_opt = scipy.sparse.load_npz(f"/work/get_optimal_P/result/p_opt{i+1}.npz")
        P = GenerateP(A)
        P_opt = spmat_set_1toRowMax_0Else(P)
        # np.random.seed(1)
        x = np.zeros(A.shape[0])
        b =  np.load(f"/work/graph/openfoam/motor/b{i+1}.npy")
        # b = np.ones(A.shape[0])
        b_norm = np.linalg.norm(b)
        t1 = time.time()
        _,steps, res = MultiLevel(A,num_level,b,x,maxiter,rtol,fn_filter,min_coarse_num=100)
        t2 = time.time()
        ori_iter_list.append(steps)
        ori_time_list.append(t2-t1)
        print("+"*20)
        print(f"num: {i+1} P_ori  iterations: {steps}, residual: {res/b_norm}, time used: {t2-t1:.4f}s")
        t3 = time.time()
        _,steps, res = MultiLevel(A,num_level,b,x,maxiter,rtol,fn_filter2,min_coarse_num=100)
        t4 = time.time()
        opt_iter_list.append(steps)
        opt_time_list.append(t4-t3)
        print(f"num: {i+1} P_post iterations: {steps}, residual: {res/b_norm}, time used: {t4-t3:.4f}s")
        print('-'*20)
    print(f"total num: {num}| P_ori  mean iterations: {np.mean(ori_iter_list):.2f}, mean time used: {np.mean(ori_time_list):.4f}s")
    print(f"total num: {num}| P_post mean iterations: {np.mean(opt_iter_list):.2f}, mean time used: {np.mean(opt_time_list):.4f}s")
    plt.plot(ori_iter_list,marker='o',label="P_ori")
    plt.plot(opt_iter_list,marker='o',label="P_post")
    plt.xlabel("Mat id")
    plt.ylabel("Iterations")
    plt.legend()
    plt.savefig("./Iterations.png")
    plt.figure()
    plt.plot(ori_time_list,marker='o',label="P_ori")
    plt.plot(opt_time_list,marker='o',label="P_post")
    plt.xlabel("Mat id")
    plt.ylabel("Time used")
    plt.legend()
    plt.savefig("./Time.png")
if __name__ == '__main__':
    # run_one()
    # TestMulti()
    run_multi_mg()
