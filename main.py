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
import wandb
import yaml
# def plot_loss(loss_list,num_batch,num_epoch,epoch_step,name):
#     x = list(range(num_batch))
#     fig, ax = plt.subplots(1,1)
#     ax.set_xlabel("epoch",fontsize = 20)
#     ax.set_ylabel("loss",fontsize = 20)
	
#     batch_per_epoch = num_batch//num_epoch
#     ticks = list(range(0,num_batch,batch_per_epoch*epoch_step))
#     labels = list(range(0,num_epoch,epoch_step))

#     ax.set_xticks(ticks)
#     ax.set_xticklabels(labels)
#     ax.plot(x,loss_list)

#     plt.show()
#     filename = name + "_loss.png"
#     fig.savefig(filename, bbox_inches='tight')
#     plt.close()


def setup_seed(seed):
    torch.manual_seed(seed)          
    torch.cuda.manual_seed(seed)     
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main(configFile="./config.yaml"):
    with open(configFile, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
        c_project = config["project"]
        c_data = config["data"]
        c_train = config["train"]
        c_model = config["model"]
    if c_project['use_wandb']:
        wandb.init(project=c_project["project_name"],
                    entity=None,
                    name=c_project["exp_name"],
                    config=config)
    setup_seed(c_project["seed"])

    ## data part
    mat_idx = list(range(c_data['mat_nums']))
    random.shuffle(mat_idx)
    mark = int(len(mat_idx)*c_data['train_ratio'])
    train_idx = mat_idx[0:mark]
    print("trian num: ", len(train_idx))
    test_idx = mat_idx[mark:]
    print("test num: ", len(test_idx))
    dataset = GraphData(mat_idx)
    batch_size = c_data["batch_size"]
    train_set = torch.utils.data.Subset(dataset,train_idx)
    test_set = torch.utils.data.Subset(dataset,test_idx)
    train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,num_workers=c_project["num_workers"])
    test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False)

    ## model part 
    device = torch.device(c_project["device"])
    if c_train["criterion"] =='mse':
        criterion = nn.MSELoss().to(device)
    elif c_train["criterion"] =='mae':
        criterion = nn.L1Loss().to(device)
    else:
        raise ValueError("no such criterion! ")

    model = GraphWrap(middle_layer=c_model['middle_layer'],
                      device=device,
                      criterion=criterion,
                      learning_rate=c_train['learning_rate'],
                      use_wandb=c_project["use_wandb"],
                      is_float=c_model['use_float'],
                      step_size=c_train['step_size'],
                      gamma=c_train['gamma'],
                      smoothing_num=c_model['smoothing_num'],
                      coarse_num=c_model['coarse_num'],
                      max_iter=c_model['max_iter'],
                      threshold=c_model['threshold']
                      )

    loss_list, total_batch_num = model.train(c_train['num_epochs'],train_loader)
    model.test(test_loader)

    # plot_loss(loss_list,total_batch_num,num_epochs,epoch_step,'graphnet')

def debug():
    random_seed = 1
    setup_seed(random_seed)

    mat_idx = list(range(8))
    dataset = GraphData(mat_idx)
    trainloader = DataLoader(dataset,batch_size=4,shuffle=True,num_workers=1)
    model = GraphWrap(2,"cuda",nn.MSELoss().to("cuda"),0.01,use_wandb=False)
    model.test(trainloader)
    # for graphs in trainloader:
    #     print('len grath = ', len(graphs))
    #     print(graphs.batch)
    #     idx = graphs.mat_id
    #     print(idx)

if __name__ == '__main__':
    main()
    # debug()
