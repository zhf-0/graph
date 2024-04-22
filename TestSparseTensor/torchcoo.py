import torch
idx = [[0, 0, 1],[0, 1, 1]]
val = [2.0,1.0,3.0]
A = torch.sparse_coo_tensor(idx, val, (2, 2), dtype = torch.float64,requires_grad = True)
b = torch.ones(2,1,dtype=torch.float64,requires_grad=True)
a = A @ b
s = a.sum()
s.backward()
