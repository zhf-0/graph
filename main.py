import json
import numpy as np
import random
import time
import os
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
import torch
import torch.nn as nn
import torch_geometric.data as pygdat

from graphdata import GraphData, TransUnsqu
from gin import gin
from templateGCN import gcn
from graphnet import GraphWrap
from gat import GatWrap
from gcn23 import GCN23Warp


def plot_loss(loss_list,num_batch,num_epoch,epoch_step,name):
    x = list(range(num_batch))
    fig, ax = plt.subplots(1,1)
    ax.set_title("Classification",fontsize = 20)
    ax.set_xlabel("epoch",fontsize = 20)
    ax.set_ylabel("loss",fontsize = 20)
	
    batch_per_epoch = num_batch//num_epoch
    ticks = list(range(0,num_batch,batch_per_epoch*epoch_step))
    labels = list(range(0,num_epoch,epoch_step))

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.plot(x,loss_list)

    plt.show()
    filename = name + "classify_loss.png"
    fig.savefig(filename, bbox_inches='tight')
    plt.close()

def plot_acc(acc_list,num_batch,num_epoch,epoch_step,name):
    x = list(range(num_batch))
    fig, ax = plt.subplots(1,1)
    ax.set_title("Classification",fontsize = 20)
    ax.set_xlabel("epoch",fontsize = 20)
    ax.set_ylabel("acc",fontsize = 20)
	
    batch_per_epoch = num_batch//num_epoch
    ticks = list(range(0,num_batch,batch_per_epoch*epoch_step))
    labels = list(range(0,num_epoch,epoch_step))

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.plot(x,acc_list)

    plt.show()
    filename = name + "acc.png"
    fig.savefig(filename, bbox_inches='tight')
    plt.close()

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
    # random.shuffle(mat_idx)
    # ratio = 0.8
    # mark = int(len(mat_idx)*ratio)
    # train_idx = mat_idx[0:mark]
    # test_idx = mat_idx[mark:]

    train_idx = list(np.load('./trainidx.npy'))
    test_idx = list(np.load('./testidx.npy'))
    
    dataset = GraphData(process=1)

    train_set = torch.utils.data.Subset(dataset,train_idx)
    test_set = torch.utils.data.Subset(dataset,test_idx)
    train_loader = pygdat.DataLoader(train_set,batch_size=16,shuffle=True,num_workers=2)
    test_loader = pygdat.DataLoader(test_set,batch_size=1,shuffle=False)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    criterion = nn.BCELoss().to(device)

    out_dim = 9
    model = GraphWrap(out_dim,device,criterion,learning_rate)

    loss_list, total_batch_num = model.train(num_epochs,train_loader)
    model.test(test_loader)

    plot_loss(loss_list,total_batch_num,num_epochs,epoch_step,'graphnet')
    # plot_acc(acc_list,total_batch_num,num_epochs,epoch_step,'graphnet')


if __name__ == '__main__':
    main()
