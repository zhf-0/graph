import pyamg
import numpy as np
import scipy.io as sio
import scipy.sparse as sparse

data = sio.loadmat('square.mat')
A = data['A'].tocsr()
# A = sparse.load_npz('./scipy_csr.npz')

ml = pyamg.ruge_stuben_solver(A, max_levels=2, max_coarse=1, CF='RS',keep=True)
print(ml)

# The CF splitting, 1 == C-node and 0 == F-node
splitting = ml.levels[0].splitting
C_nodes = splitting == 1
F_nodes = splitting == 0

p = ml.levels[0].P
print('P matrix')
print(p)

node_idx = np.arange(A.shape[0])
coarse_idx = node_idx[C_nodes]
print('coarse index =',coarse_idx)

# verify the index of coarse node
row_vec = p.indptr
col_vec = p.indices
val_vec = p.data
for i in range(p.shape[0]):
    begin_idx = row_vec[i]
    end_idx = row_vec[i+1]
    for j in range(begin_idx,end_idx):
        if abs(val_vec[j] - 1.0) < 10**(-10):
            coarse_idx_in_fine_mesh = coarse_idx[col_vec[j]]
            if i != coarse_idx_in_fine_mesh:
                print(f'row = {i}, col = {col_vec[j]}, coarse_idx_in_fine_mesh = {coarse_idx_in_fine_mesh}')


