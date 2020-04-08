import os
import torch
import torch.distributed as dist
import numpy as np
import sys

try:
    import torch_ccl
except ImportError as e:
    print(e)
    print("can't import torch_ccl module")
    sys.exit()

def print0(str):
    if rank == 0: print(str)

def print_all(str):
    for i in range(size):
        dist.barrier()
        if i == rank: print(str)
    dist.barrier()

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'
os.environ['RANK'] = os.environ.get('PMI_RANK', -1)
os.environ['WORLD_SIZE'] = os.environ.get('PMI_SIZE', -1)

dist.init_process_group(backend='ccl')

rank = dist.get_rank()
size = dist.get_world_size()

print_all("rank = %d, size = %d" % (rank, size))

print_all("scatter using alltoall")
if rank == 1:
    x = [torch.ones([2]) * (r + 1) for r in range(size)]
else:
    x = [torch.zeros([0]) for _ in range(size)]
y = [torch.zeros([2]) if r == 1 else torch.zeros([0]) for r in range(size)]
print_all("x = %s" % x)
print_all("y = %s" % y)
dist.all_to_all(y, x)
print_all("y = %s" % y)

dist.barrier()

print_all("gather using alltoall")
x = [torch.ones([2])*(rank+1) if r == 1 else torch.zeros([0]) for r in range(size)]
if rank == 1:
    y = [torch.zeros([2]) for _ in range(size)]
else:
    y = [torch.zeros([0]) for _ in range(size)]
print_all("x = %s" % x)
print_all("y = %s" % y)
dist.all_to_all(y, x)
print_all("y = %s" % y)

dist.barrier()

print_all("scatter (root=1)")
x = [torch.ones([2])*(r+1) for r in range(size)] if rank == 1 else None
y = torch.zeros([2])
print_all("x = %s" % x)
print_all("y = %s" % y)
dist.scatter(y, x, src=1)
print_all("y = %s" % y)

dist.barrier()

print_all("gather (root=1)")
x = torch.ones([2]) * (rank + 1)
y = [torch.zeros([2]) for _ in range(size)] if rank == 1 else None
print_all("x = %s" % x)
print_all("y = %s" % y)
dist.gather(x, y, dst=1)
print_all("y = %s" % y)

dist.barrier()

print_all("send (rank 1 -> 0) using alltoall")
if rank == 1:
    x = [torch.ones([2]) if r == 0 else torch.zeros([0]) for r in range(size)]
else:
    x = [torch.zeros([0]) for _ in range(size)]
if rank == 0:
    y = [torch.zeros([2]) if r == 1 else torch.zeros([0]) for r in range(size)]
else:
    y = [torch.zeros([0]) for _ in range(size)]
print_all("x = %s" % x)
print_all("y = %s" % y)
dist.all_to_all(y, x)
print_all("y = %s" % y)

dist.barrier()

