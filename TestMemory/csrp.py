import torch
import torch.nn.functional as F
from gpu_mem_track import MemTracker
gpu_tracker = MemTracker()

def cooP(single_mat_id):
    gpu_tracker.track()

    device = 'cuda:0'
    dtype = torch.float64

    extra_path = f'../GraphData/extra{single_mat_id}.dat'
    tensor_dict = torch.load(extra_path)

    gpu_tracker.track()

    coo_A = tensor_dict['coo_A'].to(device)
    coo_A = coo_A.coalesce()
    A_edge = coo_A.values().requires_grad_()

    p_index = tensor_dict['p_index'].to(device)
    edge_flag = tensor_dict['edge_flag'].to(device)

    # select edges belonging to matrix P
    p_edge = A_edge[edge_flag]

    # construct P and normalize each row of the matrix P
    p_size = tensor_dict['p_size'].to(device)
    p_row_index, _ = p_index
    new_p_val = torch.zeros(p_edge.shape[0],dtype=dtype,device=device)
    for i in range(p_size[0]):
        mask = p_row_index == i
        new_p_val[mask] = F.softmax(p_edge[mask],dim=0)

    p = torch.sparse_coo_tensor(p_index, new_p_val, (p_size[0],p_size[1]) )

    gpu_tracker.track()

def csrP(single_mat_id):
    gpu_tracker.track()

    device = 'cuda:0'
    dtype = torch.float64

    extra_path = f'../GraphData/extra{single_mat_id}.dat'
    tensor_dict = torch.load(extra_path)

    gpu_tracker.track()

    coo_A = tensor_dict['coo_A'].to(device)
    coo_A = coo_A.coalesce()
    A_edge = coo_A.values().requires_grad_()

    p_index = tensor_dict['p_index'].to(device)
    edge_flag = tensor_dict['edge_flag'].to(device)

    # select edges belonging to matrix P
    p_edge = A_edge[edge_flag]

    # construct P and normalize each row of the matrix P
    p_size = tensor_dict['p_size'].to(device)
    coo_p = torch.sparse_coo_tensor(p_index, p_edge, (p_size[0],p_size[1]))
    csr_p = coo_p.to_sparse_csr()

    p_row_vec = csr_p.crow_indices()
    p_col_vec = csr_p.col_indices()
    p_val_vec = csr_p.values()

    new_p_val = torch.zeros(p_edge.shape[0],dtype=dtype,device=device)
    for i in range(p_size[0]):
        begin_idx = p_row_vec[i]
        end_idx = p_row_vec[i+1]
        new_p_val[begin_idx:end_idx] = F.softmax(p_val_vec[begin_idx:end_idx],dim=0)

    p = torch.sparse_csr_tensor(p_row_vec, p_col_vec, new_p_val, size=(p_size[0],p_size[1]) )

    gpu_tracker.track()


# cooP(1)
csrP(1)
