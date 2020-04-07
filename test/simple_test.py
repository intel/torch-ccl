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

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'
os.environ['RANK'] = os.environ.get('PMI_RANK', -1)
os.environ['WORLD_SIZE'] = os.environ.get('PMI_SIZE', -1)

dist.init_process_group(backend='ccl')
rank = dist.get_rank()
size = dist.get_world_size()
print("rank = %d, size = %d" % (rank, size))
