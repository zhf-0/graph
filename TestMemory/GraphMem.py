import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

import sys
sys.path.append('../')
import graphnet

def TestRun():
    graph0 = torch.load('../GraphData/graph0.dat')
    dataset = [graph0]
    loader = DataLoader(dataset,batch_size=1,shuffle=False)

    device = torch.device("cuda:0")
    criterion = nn.MSELoss().to(device)
    learning_rate = 0.001
    middle_layer = 2
    model = graphnet.GraphWrap(middle_layer,device,criterion,learning_rate)
    model.train(2,loader)

def TestTorchInfo():
    from torchinfo import summary

    device = torch.device("cuda:0")
    # device = torch.device("cpu")
    criterion = nn.MSELoss().to(device)
    middle_layer = 2

    model = graphnet.GraphNet(middle_layer).to(device)
    graph0 = torch.load('../GraphData/graph0.dat')
    dataset = [graph0]
    loader = DataLoader(dataset,batch_size=1,shuffle=False)
    graph = next(iter(loader))
    summary(model,input_data=graph,batch_dim=1)

def TestSnap():
    # torch.cuda.memory._record_memory_history(max_entries=100000)
    torch.cuda.memory._record_memory_history(True)

    graph0 = torch.load('../GraphData/graph0.dat')
    dataset = [graph0]
    loader = DataLoader(dataset,batch_size=1,shuffle=False)

    device = torch.device("cuda:0")
    criterion = nn.MSELoss().to(device)
    learning_rate = 0.001
    middle_layer = 2
    model = graphnet.GraphWrap(middle_layer,device,criterion,learning_rate)
    model.train(1,loader)
    
    file_name = "visual_mem.pickle"
    torch.cuda.memory._dump_snapshot(file_name)
    torch.cuda.memory._record_memory_history(enabled=None)

def TestProfiler():
    from torch.profiler import profile, record_function, ProfilerActivity

    graph0 = torch.load('../GraphData/graph0.dat')
    dataset = [graph0]
    loader = DataLoader(dataset,batch_size=1,shuffle=False)

    device = torch.device("cuda:0")
    criterion = nn.MSELoss().to(device)
    learning_rate = 0.001
    middle_layer = 2
    model = graphnet.GraphWrap(middle_layer,device,criterion,learning_rate)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True, record_shapes=True) as prof:
        model.train(3,loader)


    print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))


def Train(model,graphs,criterion,optimizer,dtype,device):
    graphs = graphs.to(device)
    out = model(graphs)

    batch = graphs.batch
    row_vec, _ = graphs.edge_index
    edge_batch = batch[row_vec]

    num_mat = len(graphs)
    loss = 0
    for k in range(num_mat):
        b, Ax = graphnet.OptMatP(graphs.y,graphs.mat_id,out,batch,edge_batch,k,dtype,device)
        loss = loss + criterion(b,Ax)


    optimizer.zero_grad()
    loss.backward()

    ## update model params
    optimizer.step()

def TestProfilerTB():
    from torch.profiler import profile, record_function, ProfilerActivity

    graph0 = torch.load('../GraphData/graph0.dat')
    dataset = [graph0]
    loader = DataLoader(dataset,batch_size=1,shuffle=False)
    graph = next(iter(loader))

    dtype = torch.float64
    device = torch.device("cuda:0")
    criterion = nn.MSELoss().to(device)
    learning_rate = 0.001
    middle_layer = 2
    model = graphnet.GraphNet(middle_layer).double().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)
    schedule=torch.profiler.schedule(wait=0, warmup=0, active=3, repeat=1)
    # handler = torch.profiler.tensorboard_trace_handler('./log/')
    handler = torch.profiler.tensorboard_trace_handler('./log1/')
    prof = torch.profiler.profile( schedule=schedule,on_trace_ready=handler,record_shapes=True,profile_memory=True,with_stack=True)


    prof.start()
    for _ in range(3):
        prof.step()
        Train(model,graph,criterion,optimizer,dtype,device)
    prof.stop()



if __name__ == '__main__':
    # TestRun()
    # TestTorchInfo()
    # TestSnap()
    # TestProfiler()
    TestProfilerTB()
