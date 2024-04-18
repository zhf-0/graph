import torch
from torch.nn import Sequential as Seq, Linear as Lin,  LayerNorm
from torch_scatter import scatter_mean, scatter_add
from torch import nn
from torch_geometric.nn import MetaLayer
import torch.nn.functional as F
import torchamg


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

        out = torch.cat([src, dest, edge_attr], 1)
        out = self.edge_mlp(out)
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
        edge_attr = graph.edge_weight
        u = None
        batch = graph.batch

        for model in self.models:
            x, edge_attr, u = model(x, edge_index, edge_attr, u, batch)
        
        return edge_attr


def OptMatP(b, mat_id, edge_attr, batch, edge_batch, k, dtype, device):
    single_mat_id = mat_id[k]
    extra_path = f'./GraphData/extra{single_mat_id}.dat'
    tensor_dict = torch.load(extra_path)

    coo_A = tensor_dict['coo_A'].to(device)
    p_index = tensor_dict['p_index'].to(device)
    edge_flag = tensor_dict['edge_flag'].to(device)
    edge_mask = edge_batch == k

    # select edges of matrix k
    A_edge = edge_attr[edge_mask]

    # select edges belonging to matrix P
    p_edge = A_edge[edge_flag]

    # construct P
    p_size = tensor_dict['p_size'].to(device)
    p = torch.sparse_coo_tensor(p_index, p_edge.squeeze(1), (p_size[0],p_size[1]) )

    pre_jacobi = torchamg.wJacobi(dtype=dtype,device=device)
    post_jacobi = torchamg.wJacobi(dtype=dtype,device=device)
    coarse_jacobi = torchamg.wJacobi(dtype=dtype,device=device)
    tg = torchamg.TwoGrid(pre_jacobi,post_jacobi,3,coarse_jacobi,10,dtype,device)
    tg.Setup(coo_A,p)

    node_mask = batch == k
    single_b = b[node_mask]
    x = torch.zeros(single_b.shape,dtype=dtype,device=device)
    x = tg.Solve(single_b, x)
    Ax = coo_A @ x

    return single_b, Ax


def OrigonalP(b, mat_id, batch, k, dtype, device):
    single_mat_id = mat_id[k]
    extra_path = f'./GraphData/extra{single_mat_id}.dat'
    tensor_dict = torch.load(extra_path)

    coo_A = tensor_dict['coo_A'].to(device)
    p_index = tensor_dict['p_index'].to(device)
    p_val = tensor_dict['p_val'].to(device)

    # construct P
    p_size = tensor_dict['p_size'].to(device)
    p = torch.sparse_coo_tensor(p_index, p_val, (p_size[0],p_size[1]) )

    pre_jacobi = torchamg.wJacobi(dtype=dtype,device=device)
    post_jacobi = torchamg.wJacobi(dtype=dtype,device=device)
    coarse_jacobi = torchamg.wJacobi(dtype=dtype,device=device)
    tg = torchamg.TwoGrid(pre_jacobi,post_jacobi,3,coarse_jacobi,10,dtype,device)
    tg.Setup(coo_A,p)

    node_mask = batch == k
    single_b = b[node_mask]
    x = torch.zeros(single_b.shape,dtype=dtype,device=device)
    x = tg.Solve(single_b, x)
    Ax = coo_A @ x

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
        print('begin to train')
        self.model.train()
        i = 0
        train_loss_list = []
        for epoch in range(num_epochs):
            ## training step
            for graphs in trainloader:
                graphs = graphs.to(self.device)
                out = self.model(graphs)

                batch = graphs.batch
                row_vec, _ = graphs.edge_index
                edge_batch = batch[row_vec]

                num_mat = len(graphs)
                loss = 0
                for k in range(num_mat):
                    b, Ax = OptMatP(graphs.y,graphs.mat_id,out,batch,edge_batch,k,self.dtype,self.device)
                    loss = loss + self.criterion(b,Ax)


                self.optimizer.zero_grad()
                loss.backward()

                ## update model params
                self.optimizer.step()

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
                
                




