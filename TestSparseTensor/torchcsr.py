import torch

def Ax():
    idx = [[0, 0, 1],[0, 1, 1]]
    val = [2.0,1.0,3.0]
    coo_A = torch.sparse_coo_tensor(idx, val, (2, 2), dtype = torch.float64,requires_grad = True)
    csr_A = coo_A.to_sparse_csr()
    b = torch.ones(2,1,dtype=torch.float64,requires_grad=True)
    a = csr_A @ b
    print('result =',a)
    s = a.sum()
    s.backward()

def AB():
    idx = [[0, 0, 1],[0, 1, 1]]
    val = [2.0,1.0,3.0]
    new_val = [1.0,1.0,1.0]
    coo_A = torch.sparse_coo_tensor(idx, val, (2, 2), dtype = torch.float64,requires_grad = True)
    csr_A = coo_A.to_sparse_csr()
    coo_B = torch.sparse_coo_tensor(idx, new_val, (2, 2), dtype = torch.float64,requires_grad = True)
    csr_B = coo_B.to_sparse_csr()

    b = torch.ones(2,1,dtype=torch.float64,requires_grad=True)
    a = csr_A @ csr_B
    c = a @ b
    print('result =',c)
    s = c.sum()
    s.backward()

def AminusB():
    idx = [[0, 0, 1],[0, 1, 1]]
    val = [2.0,1.0,3.0]
    new_val = [1.0,1.0,1.0]
    coo_A = torch.sparse_coo_tensor(idx, val, (2, 2), dtype = torch.float64,requires_grad = True)
    csr_A = coo_A.to_sparse_csr()
    coo_B = torch.sparse_coo_tensor(idx, new_val, (2, 2), dtype = torch.float64,requires_grad = True)
    csr_B = coo_B.to_sparse_csr()

    b = torch.ones(2,1,dtype=torch.float64,requires_grad=True)
    a = csr_A - csr_B
    c = a @ b
    print('result =',c)
    s = c.sum()
    s.backward()

def trans():
    idx = [[0, 0, 1],[0, 1, 1]]
    val = [2.0,1.0,3.0]
    coo_A = torch.sparse_coo_tensor(idx, val, (2, 2), dtype = torch.float64,requires_grad = True)
    csr_A = coo_A.to_sparse_csr()
    
    t = csr_A.t().to_sparse_csr()
    print(t.layout)
    b = torch.ones(2,1,dtype=torch.float64,requires_grad=True)
    c = t @ b
    print('result =',c)
    s = c.sum()
    s.backward()


# Ax()
# AB()
# AminusB()
trans()
