import torch
import intel_extension_for_pytorch
import oneccl_bindings_for_pytorch
import torch.distributed as dist
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--device', '-dev', type=str, default='cpu', help='Device type to use: cpu, xpu')
args = parser.parse_args()

os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'

dist.init_process_group("ccl")
rank = dist.get_rank()
size = dist.get_world_size()

if args.device == 'xpu':
    device = "xpu:{}".format(rank)
else:
    device = 'cpu'

print("Barrier using device: ", args.device)
dist.barrier()
print("Finish")
