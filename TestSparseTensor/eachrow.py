import torch
import torch.nn.functional as F

def t1():
    idx = [[0, 0, 1],[0, 1, 1]]
    val = [2.0,1.0,3.0]
    coo_A = torch.sparse_coo_tensor(idx, val, (2, 2), dtype = torch.float64,requires_grad = True)
    csr_A = coo_A.to_sparse_csr()
    row_vec = csr_A.crow_indices()
    val_vec = csr_A.values()
    for i in range(csr_A.shape[0]):
        # begin_idx = row_vec[i]
        # end_idx = row_vec[i+1]
        # val_vec[begin_idx:end_idx] = F.softmax(val_vec[begin_idx:end_idx])
        val_vec[row_vec[i]:row_vec[i+1]] = F.softmax(val_vec[row_vec[i]:row_vec[i+1]])

    print(csr_A)
    
    return csr_A

def t2():
    idx = [[0, 0, 1],[0, 1, 1]]
    val = [2.0,1.0,3.0]
    coo_A = torch.sparse_coo_tensor(idx, val, (2, 2), dtype = torch.float64,requires_grad = True)
    coo_A = coo_A.coalesce()
    row_vec,_ = coo_A.indices()
    val_vec = coo_A.values()
    new_vec = torch.zeros_like(val_vec)
    for i in range(coo_A.shape[0]):
        mask = row_vec == i
        new_vec[mask] = F.softmax(val_vec[mask],dim=0)

    new_A = torch.sparse_coo_tensor(idx, new_vec, (2, 2), dtype = torch.float64)
    print(new_A)
    
    return new_A

# A = t1()
A = t2()

b = torch.ones(2,1,dtype=torch.float64,requires_grad=True)
a = A @ b
print('result =',a)
s = a.sum()
s.backward()
