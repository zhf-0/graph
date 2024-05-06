import torch

from gpu_mem_track import MemTracker
gpu_tracker = MemTracker()

gpu_tracker.track3()
# 模型初始化
linear1 = torch.nn.Linear(1024,1024, bias=False).cuda() # + 4194304
mem = torch.cuda.memory_allocated()
num = mem/(4*1024*1024)
print('linear1')
print(f'memory = {mem}, number of 1024*1024 = {num}')
gpu_tracker.track3()

linear2 = torch.nn.Linear(1024, 1, bias=False).cuda() # + 4096
mem = torch.cuda.memory_allocated()
num = mem/(4*1024*1024)
print('linear2')
print(f'memory = {mem}, number of 1024*1024 = {num}')
gpu_tracker.track3()

# 输入定义
inputs = torch.tensor([[1.0]*1024]*1024).cuda() # shape = (1024,1024) # + 4194304
mem = torch.cuda.memory_allocated()
num = mem/(4*1024*1024)
print('input')
print(f'memory = {mem}, number of 1024*1024 = {num}')
gpu_tracker.track3()

# 前向传播
loss = sum(linear2(linear1(inputs))) # shape = (1) # memory + 4194304 + 512
mem = torch.cuda.memory_allocated()
num = mem/(4*1024*1024)
print('forward')
print(f'memory = {mem}, number of 1024*1024 = {num}')
gpu_tracker.track3()

# 后向传播
loss.backward() # memory - 4194304 + 4194304 + 4096
mem = torch.cuda.memory_allocated()
num = mem/(4*1024*1024)
print('backward')
print(f'memory = {mem}, number of 1024*1024 = {num}')
gpu_tracker.track3()

# 再来一次~
loss = sum(linear2(linear1(inputs))) # shape = (1) # memory + 4194304  (512没了，因为loss的ref还在)
mem = torch.cuda.memory_allocated()
num = mem/(4*1024*1024)
print('forward 1')
print(f'memory = {mem}, number of 1024*1024 = {num}')
gpu_tracker.track3()

loss.backward() # memory - 4194304
mem = torch.cuda.memory_allocated()
num = mem/(4*1024*1024)
print('backward 1')
print(f'memory = {mem}, number of 1024*1024 = {num}')
gpu_tracker.track3()

# 再来一次~
loss = sum(linear2(linear1(inputs))) # shape = (1) # memory + 4194304  (512没了，因为loss的ref还在)
mem = torch.cuda.memory_allocated()
num = mem/(4*1024*1024)
print('forward 2')
print(f'memory = {mem}, number of 1024*1024 = {num}')
gpu_tracker.track3()

loss.backward() # memory - 4194304
mem = torch.cuda.memory_allocated()
num = mem/(4*1024*1024)
print('backward 2')
print(f'memory = {mem}, number of 1024*1024 = {num}')
gpu_tracker.track3()
