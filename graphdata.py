import os
import numpy as np

import scipy.io as sio
import scipy.sparse as sparse
import pyamg

import torch
import torch_geometric.data as pygdat

class TransGraph():
    def __init__(self):
        pass

    def __call__(self,graph):
        pass
        # graph.edge_weight = graph.edge_weight.unsqueeze(1)
        # return graph


class GraphData(torch.utils.data.Dataset):
    def __init__(self, mat_idx_list,transform=None):
        self.mat_idx = mat_idx_list
        self.num = len(self.mat_idx)
            
        self.transform = transform
        self.root_path_mat = './MatData'
        self.root_path_graph = './GraphData'
            
        os.makedirs(self.root_path_graph,exist_ok=True)

        self.Process()

    def Process(self):
        print(f'begin to process {self.num} matrices')
        mat_template = self.root_path_mat+'/scipy_csr{}.npz'
        vec_template = self.root_path_mat+'/b{}.npy'
        graph_template = self.root_path_graph+'/graph{}.dat'
        extra_template = self.root_path_graph+'/extra{}.dat'
        for i,idx in enumerate(self.mat_idx):
            mat_path = mat_template.format(idx)
            vec_path = vec_template.format(idx)
            graph_path = graph_template.format(idx)
            extra_path = extra_template.format(idx)
            if not os.path.exists(graph_path):
                print(f'begin to deal with {i}-th matrix with index {idx}')
                scipy_csr = sparse.load_npz(mat_path)
                scipy_coo = scipy_csr.tocoo()

                # begin to construct the graph
                # ml = pyamg.ruge_stuben_solver(scipy_csr, max_levels=2, keep=True, CF=splitting)
                ml = pyamg.ruge_stuben_solver(scipy_csr, max_levels=2, keep=True)
                splitting = ml.levels[0].splitting
                coarse_node_encoding = np.zeros(splitting.shape[0])
                fine_node_encoding = np.zeros(splitting.shape[0])

                # node encoding 
                C_flag = splitting == 1
                F_flag = splitting == 0
                coarse_node_encoding[C_flag] = 1.0
                fine_node_encoding[F_flag] = 1.0
                node_encoding = np.stack([coarse_node_encoding, fine_node_encoding]).T
                x = torch.from_numpy(node_encoding)
                
                # the fine index of the coarse node
                node_idx = np.arange(scipy_csr.shape[0])
                coarse_idx = node_idx[C_flag]
                c2f = {}
                for k in range(coarse_idx.shape[0]):
                    c2f[k] = coarse_idx[k]

                # edge encoding
                p = ml.levels[0].P
                coo_p = p.tocoo()
                fine_coo_p_col = np.zeros(coo_p.nnz,dtype=np.int32)
                for k in range(coo_p.nnz):
                    fine_coo_p_col[k] = c2f[coo_p.col[k]]
                    
                p_edge_idx = np.core.records.fromarrays([coo_p.row, fine_coo_p_col], dtype='i,i')
                A_edge_idx = np.core.records.fromarrays([scipy_coo.row, scipy_coo.col], dtype='i,i')
                edge_flag = np.in1d(A_edge_idx, p_edge_idx, assume_unique=True)
                coarse_edge_encoding = np.zeros(scipy_coo.nnz)
                fine_edge_encoding = np.zeros(scipy_coo.nnz)
                coarse_edge_encoding[edge_flag] = 1.0
                fine_edge_encoding[~edge_flag] = 1.0
                edge_encoding = np.stack([scipy_coo.data,coarse_edge_encoding,fine_edge_encoding]).T
                edge_attr = torch.from_numpy(edge_encoding)

                # if there is a file saving the rhs vector, read it; otherwise create one
                if os.path.exists(vec_path):
                    b = np.load(vec_path)
                else:
                    b = np.ones(scipy_coo.shape[0])
                y = torch.from_numpy(b)


                # finaly cteate the graph
                np_row = scipy_coo.row
                np_col = scipy_coo.col
                torch_row = torch.from_numpy(np_row.astype(np.int64))
                torch_col = torch.from_numpy(np_col.astype(np.int64))
                edge_index = torch.stack((torch_row,torch_col),0)
                graph = pygdat.Data(x=x,edge_index = edge_index,edge_attr = edge_attr,y = y.reshape(-1,1))
                graph.mat_id = idx
                torch.save(graph,graph_path)
                
                # save variables into file for future training and testing
                tensor_dict = {}

                p_row = torch.from_numpy(coo_p.row.astype(np.int64))
                p_col = torch.from_numpy(coo_p.col.astype(np.int64))
                p_index = torch.stack((p_row,p_col),0)
                tensor_dict['p_index'] = p_index
                p_val = torch.from_numpy(coo_p.data)
                tensor_dict['p_val'] = p_val
                p_size = torch.tensor([coo_p.shape[0],coo_p.shape[1],coo_p.nnz])
                tensor_dict['p_size'] = p_size

                A_val = torch.from_numpy(scipy_coo.data)
                coo_A = torch.sparse_coo_tensor(edge_index, A_val, scipy_coo.shape)
                tensor_dict['coo_A'] = coo_A

                coarse_idx = torch.from_numpy(coarse_idx.astype(np.int64))
                tensor_dict['coarse_idx'] = coarse_idx

                edge_flag = torch.from_numpy(edge_flag)
                tensor_dict['edge_flag'] = edge_flag

                torch.save(tensor_dict,extra_path)
                


    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        graph = torch.load(os.path.join(self.root_path_graph,f'graph{idx}.dat'))
        
        if self.transform:
            graph = self.transform(graph)

        return graph




if __name__ == '__main__':
    mat_idx_list = [0]
    dataset = GraphData(mat_idx_list)
