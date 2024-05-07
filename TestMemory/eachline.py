import os
import torch
from torch.nn import Sequential as Seq, Linear as Lin,  LayerNorm
from torch_scatter import scatter_mean, scatter_add
from torch import nn
from torch_geometric.nn import MetaLayer
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

from gpu_mem_track import MemTracker
gpu_tracker = MemTracker()

class TwoGrid:
    def __init__(self, pre_smoother, post_smoother, smoothing_num, coarse_solver, coarse_num, dtype=torch.float64, device='cpu'):
        self.pre_smoother = pre_smoother
        self.post_smoother = post_smoother
        self.coarse_solver = coarse_solver
        self.smoothing_num = smoothing_num
        self.coarse_num = coarse_num
        self.dtype = dtype
        self.device = device

    def Setup(self, A, p):
        R = p.t().to_sparse_csr()
        A_c = R @ A @ p 

        self.pre_smoother.Setup(A)
        self.post_smoother.Setup(A)
        self.coarse_solver.Setup(A_c)
        
        self.P = p.to(self.device)
        self.R = R.to(self.device)
        self.A_c = A_c.to(self.device)
        self.A = A.to(self.device)
        # self.dense_A_c = A_c.to_dense().to(self.device)

    def CoarseSolve(self, b, x):
        for _ in range(self.coarse_num):
            x = self.coarse_solver.Solve(b, x)

        return x

    def Solve(self, b, x):
        if len(b.shape) == 1:
            b = b.unsqueeze(1)

        for _ in range(self.smoothing_num):
            x = self.pre_smoother.Solve(b, x)

        residual = b - self.A @ x

        coarse_b = self.R @ residual
        coarse_x = self.CoarseSolve(coarse_b, torch.zeros(coarse_b.shape,dtype=self.dtype,device=self.device) )
        x += self.P @ coarse_x

        for _ in range(self.smoothing_num):
            x = self.post_smoother.Solve(b, x)

        return x

def GetDiagVec(csr_A, dtype=torch.float64, device='cpu'):
    coo = csr_A.to_sparse_coo().coalesce()
    row_vec, col_vec = coo.indices()
    val_vec = coo.values()
    nrow = coo.shape[0]
    
    diag = torch.zeros(nrow,dtype=dtype,device=device)
    mask = row_vec == col_vec
    diag[row_vec[mask]] = val_vec[mask]

    return diag


def GetInvDiagSpMat(csr_A, dtype=torch.float64, device='cpu'):
    diag = GetDiagVec(csr_A, dtype, device)
    invdiag = 1.0 / diag

    nrow = csr_A.shape[0]
    row_vec = torch.arange(nrow,device=device)
    col_vec = torch.arange(nrow,device=device)
    coo_invdiag = torch.sparse_coo_tensor(torch.stack((row_vec,col_vec)),invdiag, (nrow, nrow),dtype=dtype,device=device)
    csr_invdiag = coo_invdiag.to_sparse_csr()

    return csr_invdiag

def CreateI(nrow, dtype=torch.float64, device='cpu'):
    row_vec = torch.arange(nrow,device=device)
    col_vec = torch.arange(nrow,device=device)
    val_vec = torch.ones(nrow,dtype=dtype,device=device)

    I = torch.sparse_coo_tensor(torch.stack((row_vec,col_vec)),val_vec, (nrow, nrow),dtype=dtype,device=device)
    I = I.to_sparse_csr()

    return I 

class wJacobi:
    def __init__(self, weight=1.0, dtype=torch.float64, device='cpu'):
        self.weight = weight
        self.dtype = dtype
        self.device = device

    def Setup(self, A):
        invdiag = GetInvDiagSpMat(A,self.dtype,self.device)

        # self.mat = I - self.weight * (invdiag @ A)
        self.mat = self.weight * (invdiag @ A)
        self.A = A.to(self.device)
        self.invdiag = invdiag

    def Solve(self, b, x):
        x = x - self.mat @ x + self.weight * self.invdiag @ b
        return x


def CreateMLP(
    in_size,
    out_size,
    n_hidden,
    hidden_size,
    activation=nn.LeakyReLU,
    activate_last=False,
    layer_norm=False,
):
    arch = []
    l_in = in_size
    for l_idx in range(n_hidden):
        arch.append(Lin(l_in, hidden_size))
        arch.append(activation())
        l_in = hidden_size

    arch.append(Lin(l_in, out_size))

    if activate_last:
        arch.append(activation())

        if layer_norm:
            arch.append(LayerNorm(out_size))

    return Seq(*arch)

class EdgeModel(torch.nn.Module):
    def __init__(self,in_size, out_size, n_hidden, hidden_size, activation, activate_last, layer_norm):
        super().__init__()
        self.edge_mlp = CreateMLP(in_size, out_size, n_hidden, hidden_size, activation, activate_last, layer_norm)

    def forward(self, src, dest, edge_attr, u, batch):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]

        gpu_tracker.track()

        out = torch.cat([src, dest, edge_attr], 1)
        gpu_tracker.track()

        out = self.edge_mlp(out)
        
        gpu_tracker.track()
        return out

class NodeModel(torch.nn.Module):
    def __init__(self, x_size, in_size, out_size, n_hidden, hidden_size, activation, activate_last, layer_norm):
        super().__init__()
        # self.node_mlp = CreateMLP(in_size, out_size, n_hidden, hidden_size, activation, activate_last, layer_norm)
        self.node_mlp_1 = CreateMLP(in_size, out_size, n_hidden, hidden_size, activation, activate_last, layer_norm)
        self.node_mlp_2 = CreateMLP(x_size+out_size, out_size, n_hidden, hidden_size, activation, activate_last, layer_norm)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]

        gpu_tracker.track()

        # the equation is: x_i = x_i + Aggr(x_j, e_ij) 
        # official
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        out = self.node_mlp_2(out)

        # my personal new equation is: x_i = x_i + Aggr(e_ij) 
        # row, col = edge_index
        # out = scatter_mean(edge_attr, row, dim=0, dim_size=x.size(0))
        # out = torch.cat([x, out, u[batch]], dim=1)
        # out = self.node_mlp(out)

        gpu_tracker.track()
        return out

class GlobalModel(torch.nn.Module):
    def __init__(self,in_size, out_size, n_hidden, hidden_size, activation, activate_last, layer_norm):
        super().__init__()
        self.global_mlp = CreateMLP(in_size, out_size, n_hidden, hidden_size, activation, activate_last, layer_norm)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        
        # the equation is: u + node_attr 
        # official
        out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
        out = self.global_mlp(out)

        # my personal new equation is: u + node_attr + edge_attr
        # row, col = edge_index
        # e_batch = batch[row]
        # out = torch.cat(
        #     [
        #         u,
        #         scatter_mean(x, batch, dim=0),
        #         scatter_mean(edge_attr, e_batch, dim=0),
        #     ],
        #     dim=1,
        # )
        # out = self.global_mlp(out)

        return out


class GraphNet(torch.nn.Module):
    def __init__(self,middle_layer):
        super().__init__()
        e_in = 3
        n_in = 2
        g_in = 0
        
        e_mid = 4
        n_mid = 4
        g_mid = 0

        e_out = 1
        n_out = 1
        g_out = 0

        n_hidden = 1
        hidden_size = 16

        layers = []

        edge1 = EdgeModel(e_in+2*n_in+g_in,e_mid,n_hidden,hidden_size,nn.ReLU,False,False)
        node1 = NodeModel(n_in,n_in+e_mid+g_in,n_mid,n_hidden,hidden_size,nn.ReLU,False,False)
        layers.append(MetaLayer(edge_model = edge1,node_model=node1,global_model=None))

        for _ in range(middle_layer):
            edge = EdgeModel(e_mid+2*n_mid+g_mid,e_mid,n_hidden,hidden_size,nn.ReLU,False,False)
            node = NodeModel(n_mid,n_mid+e_mid+g_mid,n_mid,n_hidden,hidden_size,nn.ReLU,False,False)
            layers.append(MetaLayer(edge_model=edge,node_model=node,global_model=None))

        edge2 = EdgeModel(e_mid+2*n_mid+g_mid,e_out,n_hidden,hidden_size,nn.ReLU,False,False)
        node2 = NodeModel(n_mid,n_mid+e_out+g_mid,n_out,n_hidden,hidden_size,nn.ReLU,False,False)
        layers.append(MetaLayer(edge_model = edge2,node_model=node2,global_model=None))

        self.models = Seq(*layers)
        
    def forward(self,graph):
        x = graph.x
        edge_index  = graph.edge_index
        edge_attr = graph.edge_attr
        u = None
        batch = graph.batch

        gpu_tracker.track()

        for model in self.models:
            x, edge_attr, u = model(x, edge_index, edge_attr, u, batch)
            gpu_tracker.track()
        
        return edge_attr


def OptMatP(b, mat_id, edge_attr, batch, edge_batch, k, dtype, device):
    gpu_tracker.track()

    single_mat_id = mat_id[k]
    extra_path = f'../GraphData/extra{single_mat_id}.dat'
    tensor_dict = torch.load(extra_path)

    gpu_tracker.track()

    coo_A = tensor_dict['coo_A'].to(device)
    p_index = tensor_dict['p_index'].to(device)
    edge_flag = tensor_dict['edge_flag'].to(device)
    edge_mask = edge_batch == k

    # select edges of matrix k
    A_edge = edge_attr[edge_mask]

    # select edges belonging to matrix P
    p_edge = A_edge[edge_flag]

    # construct P and normalize each row of the matrix P
    p_size = tensor_dict['p_size'].to(device)
    coo_p = torch.sparse_coo_tensor(p_index, p_edge.unsqueeze(1), (p_size[0],p_size[1]))
    csr_p = coo_p.to_sparse_csr()

    p_row_vec = csr_p.crow_indices()
    p_col_vec = csr_p.col_indices()
    p_val_vec = csr_p.values()

    new_p_val = torch.zeros(p_edge.shape[0],dtype=dtype,device=device)
    for i in range(p_size[0]):
        begin_idx = p_row_vec[i]
        end_idx = p_row_vec[i+1]
        new_p_val[begin_idx:end_idx] = F.softmax(p_val_vec[begin_idx:end_idx],dim=0)

    csr_p = torch.sparse_csr_tensor(p_row_vec, p_col_vec, new_p_val, size=(p_size[0],p_size[1]) )
    csr_A = coo_A.to_sparse_csr()

    gpu_tracker.track()

    pre_jacobi = wJacobi(dtype=dtype,device=device)
    post_jacobi = wJacobi(dtype=dtype,device=device)
    coarse_jacobi = wJacobi(dtype=dtype,device=device)
    tg = TwoGrid(pre_jacobi,post_jacobi,3,coarse_jacobi,10,dtype,device)
    tg.Setup(csr_A,csr_p)

    node_mask = batch == k
    single_b = b[node_mask]
    x = torch.zeros(single_b.shape,dtype=dtype,device=device)
    x = tg.Solve(single_b, x)
    Ax = csr_A @ x

    gpu_tracker.track()

    return single_b, Ax


def OrigonalP(b, mat_id, batch, k, dtype, device):
    single_mat_id = mat_id[k]
    extra_path = f'../GraphData/extra{single_mat_id}.dat'
    tensor_dict = torch.load(extra_path)

    coo_A = tensor_dict['coo_A'].to(device)
    p_index = tensor_dict['p_index'].to(device)
    p_val = tensor_dict['p_val'].to(device)

    # construct P
    p_size = tensor_dict['p_size'].to(device)
    p = torch.sparse_coo_tensor(p_index, p_val, (p_size[0],p_size[1]) )

    csr_p = p.to_sparse_csr() 
    csr_A = coo_A.to_sparse_csr()

    pre_jacobi = wJacobi(dtype=dtype,device=device)
    post_jacobi = wJacobi(dtype=dtype,device=device)
    coarse_jacobi = wJacobi(dtype=dtype,device=device)
    tg = TwoGrid(pre_jacobi,post_jacobi,3,coarse_jacobi,10,dtype,device)
    tg.Setup(csr_A,csr_p)

    node_mask = batch == k
    single_b = b[node_mask]
    x = torch.zeros(single_b.shape,dtype=dtype,device=device)
    x = tg.Solve(single_b, x)
    Ax = csr_A @ x

    return single_b, Ax

    
class GraphWrap:
    def __init__(self, middle_layer, device, criterion, learning_rate, is_float=False):
        if is_float:
            self.model = GraphNet(middle_layer)
            self.dtype = torch.float32
        else:
            self.model = GraphNet(middle_layer).double()
            self.dtype = torch.float64

        self.model = self.model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.schedule = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10,gamma=0.2)


    def train(self, num_epochs,trainloader):
        gpu_tracker.track()
        print('begin to train')
        self.model.train()
        i = 0
        train_loss_list = []
        for epoch in range(num_epochs):
            ## training step
            for graphs in trainloader:
                graphs = graphs.to(self.device)

                gpu_tracker.track()

                out = self.model(graphs)

                gpu_tracker.track()

                batch = graphs.batch
                row_vec, _ = graphs.edge_index
                edge_batch = batch[row_vec]

                num_mat = len(graphs)
                loss = 0
                for k in range(num_mat):
                    b, Ax = OptMatP(graphs.y,graphs.mat_id,out,batch,edge_batch,k,self.dtype,self.device)
                    loss = loss + self.criterion(b,Ax)


                gpu_tracker.track()

                self.optimizer.zero_grad()

                gpu_tracker.track()

                loss.backward()

                gpu_tracker.track()

                ## update model params
                self.optimizer.step()

                gpu_tracker.track()

                train_running_loss = loss.item()
            
                print('Epoch: {:3} | Batch: {:3}| Loss: {:6.4f} '.format(epoch,i,train_running_loss))

                i = i + 1
                train_loss_list.append(train_running_loss)

            self.schedule.step()
        return  train_loss_list, i

    def test(self, testloader):
        print('begin to test')
        
        self.model.eval()
        for graphs in testloader:
            graphs = graphs.to(self.device)
            with torch.no_grad():
                out = self.model(graphs)
                
                batch = graphs.batch
                row_vec, _ = graphs.edge_index
                edge_batch = batch[row_vec]

                num_mat = len(graphs)
                for k in range(num_mat):
                    b0, Ax0 = OptMatP(graphs.y,graphs.mat_id,out,batch,edge_batch,k,self.dtype,self.device)
                    loss0 = self.criterion(b0,Ax0)
                    print(f'mat {graphs.mat_id[k]}: the MSE residual of the optimized P is {loss0}')

                    b1, Ax1 = OrigonalP(graphs.y,graphs.mat_id,batch,k,self.dtype,self.device)
                    loss1 = self.criterion(b1,Ax1)
                    print(f'mat {graphs.mat_id[k]}: the MSE residual of the origonal P is {loss1}')


gpu_tracker.track()

# graph0 = torch.load('../GraphData/graph0.dat')
graph0 = torch.load('../GraphData/graph2.dat')
dataset = [graph0]
loader = DataLoader(dataset,batch_size=1,shuffle=False)

gpu_tracker.track()

device = torch.device("cuda:0")
criterion = nn.MSELoss().to(device)
learning_rate = 0.001
middle_layer = 2

gpu_tracker.track()

model = GraphWrap(middle_layer,device,criterion,learning_rate)

gpu_tracker.track()

model.train(1,loader)

gpu_tracker.track()
