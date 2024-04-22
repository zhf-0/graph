import torch

idx = [[0, 0, 1],[0, 1, 1]]
val = [2.0,1.0,3.0]
coo_A = torch.sparse_coo_tensor(idx, val, (2, 2), dtype = torch.float64,requires_grad = True)
csr_A = coo_A.to_sparse_csr()
b = torch.ones(2,1,dtype=torch.float64,requires_grad=True)
a = csr_A @ b
print('result =',a)
s = a.sum()
s.backward()
