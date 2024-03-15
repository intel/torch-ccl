import torch
import intel_extension_for_pytorch
import oneccl_bindings_for_pytorch
import torch.distributed as dist
import os
import time

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dist_url', default='127.0.0.1', type=str, help='url used to set up distributed training')
parser.add_argument('--dist_port', default='29500', type=str, help='url port used to set up distributed training')
parser.add_argument('--sub_group', default=4, type=int, help='url port used to set up distributed training')
args = parser.parse_args()

if 'PMI_RANK' in os.environ.keys() and 'PMI_SIZE' in os.environ.keys():
    os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
    os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1)) # mpich set
elif 'PMIX_RANK' in os.environ.keys() and 'PALS_LOCAL_SIZE' in os.environ.keys():
    os.environ['RANK'] = os.environ.get('PMIX_RANK')
    os.environ['WORLD_SIZE'] = str(os.environ.get('PALS_LOCAL_SIZE', -1))

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'

init_method = 'tcp://' + args.dist_url + ':' + args.dist_port
dist.init_process_group(backend='ccl', init_method=init_method,
                        world_size=int(os.environ['WORLD_SIZE']), rank=int(os.environ['RANK']))

rank = dist.get_rank()
size = dist.get_world_size()
device = "xpu:{}".format(rank)
print('world_size:{}, global rank:{}'.format(size, rank))

shape = int(2048)
warm_shape = int(1)
warm = torch.ones(warm_shape).bfloat16().to(device)

input_shape = shape
input = torch.ones(input_shape).bfloat16().to(device)

#warm_up
dist.all_reduce(warm)

#sub_group=1(TP=12)
group1 = dist.new_group([0])
if rank ==0:
    dist.all_reduce(input, group=group1)

group_size = [[i+(size // args.sub_group)*j for j in range(args.sub_group)] for i in range(size // args.sub_group)]
sub_group = []

#construct sub group
for i in range(len(group_size)):
    sub_group.append(dist.new_group(group_size[i]))

for i in range(len(group_size)):
    if dist.get_rank() in group_size[i]:
        dist.all_reduce(input, group=sub_group[i])
