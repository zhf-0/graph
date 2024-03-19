import torch
from torch.nn import Sequential as Seq, Linear as Lin,  LayerNorm
from torch_scatter import scatter_mean, scatter_add
from torch import nn
from torch_geometric.nn import MetaLayer
import torch.nn.functional as F

from torchmetrics.classification import MultilabelStatScores as Stat

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
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        out = self.edge_mlp(out)
        return out

class NodeModel(torch.nn.Module):
    def __init__(self,in_size, out_size, n_hidden, hidden_size, activation, activate_last, layer_norm):
        super().__init__()
        self.node_mlp = CreateMLP(in_size, out_size, n_hidden, hidden_size, activation, activate_last, layer_norm)

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.

        # official
        # x_i = x_i + Aggr(x_j, e_ij) 
        # row, col = edge_index
        # out = torch.cat([x[row], edge_attr], dim=1)
        # out = self.node_mlp_1(out)
        # out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        # out = torch.cat([x, out, u[batch]], dim=1)
        # out = self.node_mlp_2(out)

        # x_i = x_i + Aggr(e_ij)
        row, col = edge_index
        out = scatter_mean(edge_attr, row, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out, u[batch]], dim=1)
        out = self.node_mlp(out)
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
        
        # u + node_attr + edge_attr
        row, col = edge_index
        e_batch = batch[row]
        out = torch.cat(
            [
                u,
                scatter_mean(x, batch, dim=0),
                scatter_mean(edge_attr, e_batch, dim=0),
            ],
            dim=1,
        )
        out = self.global_mlp(out)

        # official
        # u + node_attr 
        # out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
        # out = self.global_mlp(out)
        return out


class GraphNet(torch.nn.Module):
    def __init__(self,class_num):
        super().__init__()
        e_in = 1
        n_in = 1
        g_in = 1

        e_out = 16
        n_out = 16
        g_out = 16

        n_hidden = 1
        hidden_size = 16

        edge1 = EdgeModel(e_in+2*n_in+g_in,e_out,n_hidden,hidden_size,nn.ReLU,False,False)
        node1 = NodeModel(n_in+e_out+g_in,n_out,n_hidden,hidden_size,nn.ReLU,False,False)
        global1 = GlobalModel(n_out+e_out+g_in,g_out,n_hidden,hidden_size,nn.ReLU,False,False)
        self.meta1 = MetaLayer(edge_model = edge1,node_model=node1,global_model=global1)

        edge2 = EdgeModel(e_out+2*n_out+g_out,e_out,n_hidden,hidden_size,nn.ReLU,False,False)
        node2 = NodeModel(n_out+e_out+g_out,n_out,n_hidden,hidden_size,nn.ReLU,False,False)
        global2 = GlobalModel(n_out+e_out+g_out,g_out,n_hidden,hidden_size,nn.ReLU,False,False)
        self.meta2 = MetaLayer(edge_model = edge2,node_model=node2,global_model=global2)
        
        edge3 = EdgeModel(e_out+2*n_out+g_out,e_out,n_hidden,hidden_size,nn.ReLU,False,False)
        node3 = NodeModel(n_out+e_out+g_out,n_out,n_hidden,hidden_size,nn.ReLU,False,False)
        # global3 = GlobalModel(n_out+e_out+g_out,g_out,n_hidden,hidden_size,nn.ReLU,False,False)
        global3 = GlobalModel(n_out+e_out+g_out,class_num,n_hidden,hidden_size,nn.ReLU,False,False)
        self.meta3 = MetaLayer(edge_model = edge3,node_model=node3,global_model=global3)

    def forward(self,graph):
        x, edge_attr, u = self.meta1(graph.x, graph.edge_index, graph.edge_weight, graph.u, graph.batch)

        x, edge_attr, u = self.meta2(x, graph.edge_index, edge_attr, u, graph.batch)

        x, edge_attr, u = self.meta3(x, graph.edge_index, edge_attr, u, graph.batch)
        
        return F.sigmoid(u)
    
class GraphWrap:
    def __init__(self, class_num, device, criterion, learning_rate, is_float=True):
        if is_float:
            self.model = GraphNet(class_num)
        else:
            self.model = GraphNet(class_num).double()

        self.model = self.model.to(device)
        self.device = device
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.schedule = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10,gamma=0.2)

        self.out_dim = class_num

    def train(self, num_epochs,trainloader):
        print('begin to train')
        self.model.train()
        i = 0
        train_loss_list = []
        for epoch in range(num_epochs):
            ## training step
            for graphs in trainloader:
                graphs = graphs.to(self.device)

                ## forward + backprop + loss
                out = self.model(graphs)
                loss = self.criterion(out, graphs.y)
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
        stat = Stat(num_labels=self.out_dim,average=None).to(self.device)
        final_stat = 0
        
        self.model.eval()
        total_num = len(testloader.dataset)
        batch_list = [] 
        bceloss_list = []
        for graphs in testloader:
            graphs = graphs.to(self.device)
            with torch.no_grad():
                out = self.model(graphs)
                
                bceloss = self.criterion(out, graphs.y)
                print('batch bce loss: {}'.format(bceloss.item()))
                batch_list.append( graphs.y.shape[0])
                bceloss_list.append(bceloss.item())
                
                s = stat(out,graphs.y.int())
                final_stat += s


        print(f'total num = {total_num}, batch sum = {sum(batch_list)}')
        final_bce = 0.0
        for i in range(len(batch_list)):
            final_bce += batch_list[i] * bceloss_list[i]

        final_bce = final_bce / sum(batch_list)
        print(f'bce loss: {final_bce}')

        final_stat = final_stat.to('cpu')
        print(f'Stat: {final_stat}')

        prec_list = []
        for i in range(final_stat.shape[0]):
            tmp = final_stat[i,0]+final_stat[i,1] 
            if  tmp == 0:
                prec_list.append(0.0)
            else:
                prec_list.append( final_stat[i,0]/tmp )

        print(f'the average precision: {sum(prec_list)/len(prec_list)}')
