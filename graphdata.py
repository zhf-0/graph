import os
import argparse 
import numpy as np
import matplotlib.pyplot as plt

import scipy.io as sio
import scipy.sparse as sparse
from scipy.sparse import coo_matrix
import pandas as pd 

import torch
import torch_geometric.data as pygdat
from torch_geometric.utils import degree

class TransUnsqu():
    def __init__(self):
        pass

    def __call__(self,graph):
        graph.edge_weight = graph.edge_weight.unsqueeze(1)
        graph.u = torch.tensor([[0.0]])
        return graph


class GraphData(torch.utils.data.Dataset):
    def __init__(self, transform=None, process=0):
        self.mat_idx = np.load('./idx.npy')
        self.num = len(self.mat_idx)
        self.labels = np.load('./y.npy')
            
        self.transform = transform
        self.root_path_mat = './MatData'
        self.root_path_graph = './GraphData'
            
        os.makedirs(self.root_path_graph,exist_ok=True)

        if process:
            self.Process()

    def Process(self):
        print('begin to process')
        mat_template = self.root_path_mat+'/scipy_csr{}.npz'
        graph_template = self.root_path_graph+'/graph{}.dat'
        for i,idx in enumerate(self.mat_idx):
            mat_path = mat_template.format(idx)
            graph_path = graph_template.format(idx)
            if not os.path.exists(graph_path):
                print(f'begin to deal with matrix {idx}')
                scipy_csr = sparse.load_npz(mat_path)
                scipy_coo = coo_matrix(scipy_csr)

                row, col = scipy_coo.nonzero()
                edge_weight = torch.from_numpy( np.abs(scipy_coo.data) )

                row = torch.from_numpy(row.astype(np.int64))
                col = torch.from_numpy(col.astype(np.int64))
                edge_weight = edge_weight.float().unsqueeze(1)

                x = degree(col,scipy_coo.shape[0],dtype=torch.float32).unsqueeze(1)
                edge_index = torch.stack((row,col),0)
                y = torch.from_numpy(self.labels[i,:].reshape(1,-1)).float()

                graph = pygdat.Data(x=x,edge_index = edge_index,edge_weight = edge_weight,y = y)
                graph.mat_id = idx
                graph.u = torch.tensor([[0.0]])
                torch.save(graph,graph_path)


    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        graph = torch.load(os.path.join(self.root_path_graph,'graph{}.dat'.format(idx)))
        
        if self.transform:
            graph = self.transform(graph)

        return graph




if __name__ == '__main__':
    dataset = GraphData(process=1)
