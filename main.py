import json
import numpy as np
import random
import time
import os
# from matplotlib import pyplot as plt
# from matplotlib.pyplot import MultipleLocator
import torch
import torch.nn as nn
import torch_geometric.data as pygdat
from torch_geometric.loader import DataLoader

from graphdata import GraphData
from graphnet import GraphWrap


# def plot_loss(loss_list,num_batch,num_epoch,epoch_step,name):
#     x = list(range(num_batch))
#     fig, ax = plt.subplots(1,1)
#     ax.set_title("Classification",fontsize = 20)
#     ax.set_xlabel("epoch",fontsize = 20)
#     ax.set_ylabel("loss",fontsize = 20)
	
#     batch_per_epoch = num_batch//num_epoch
#     ticks = list(range(0,num_batch,batch_per_epoch*epoch_step))
#     labels = list(range(0,num_epoch,epoch_step))

#     ax.set_xticks(ticks)
#     ax.set_xticklabels(labels)
#     ax.plot(x,loss_list)

#     plt.show()
#     filename = name + "classify_loss.png"
#     fig.savefig(filename, bbox_inches='tight')
#     plt.close()


def setup_seed(seed):
    torch.manual_seed(seed)          
    torch.cuda.manual_seed(seed)     
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    num_epochs = 30
    epoch_step = 1
    learning_rate = 0.0001
    random_seed = 1
    setup_seed(random_seed)

    # mat_idx = list(np.load('./idx.npy'))
    # train_idx = list(np.load('./trainidx.npy'))
    # test_idx = list(np.load('./testidx.npy'))

    mat_idx = list(range(4))
    random.shuffle(mat_idx)
    ratio = 0.5
    mark = int(len(mat_idx)*ratio)
    train_idx = mat_idx[0:mark]
    test_idx = mat_idx[mark:]
    
    dataset = GraphData(mat_idx)

    train_set = torch.utils.data.Subset(dataset,train_idx)
    test_set = torch.utils.data.Subset(dataset,test_idx)
    train_loader = DataLoader(train_set,batch_size=1,shuffle=True,num_workers=2)
    test_loader = DataLoader(test_set,batch_size=1,shuffle=False)
    
    device = torch.device("cuda:0")
    # device = torch.device("cpu")
    criterion = nn.MSELoss().to(device)

    middle_layer = 2
    model = GraphWrap(middle_layer,device,criterion,learning_rate)

    loss_list, total_batch_num = model.train(num_epochs,train_loader)
    model.test(test_loader)

    # plot_loss(loss_list,total_batch_num,num_epochs,epoch_step,'graphnet')

def debug():
    random_seed = 1
    setup_seed(random_seed)

    mat_idx = list(range(4))
    dataset = GraphData(mat_idx)
    trainloader = DataLoader(dataset,batch_size=2,shuffle=True,num_workers=1)

    for graphs in trainloader:
        print('len grath = ', len(graphs))
        print(graphs.batch)
        idx = graphs.mat_id
        print(idx)

if __name__ == '__main__':
    main()
    # debug()
