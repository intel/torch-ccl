import torch
import intel_extension_for_pytorch
import oneccl_bindings_for_pytorch
import torch.distributed as dist
import os


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--world_size', default=-1, type=int, help='number of gpu for distributed training')
parser.add_argument('--dist_url', default='127.0.0.1', type=str, help='url used to set up distributed training')
parser.add_argument('--dist_port', default='29800', type=str, help='url port used to set up distributed training')
args = parser.parse_args()

os.environ['RANK'] = str(os.environ.get('PMIX_RANK',0))
os.environ['WORLD_SIZE'] = str(args.world_size)
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'

init_method = 'tcp://' + args.dist_url + ':' + args.dist_port
dist.init_process_group(backend='ccl', init_method=init_method,
                        world_size=int(os.environ['WORLD_SIZE']), rank=int(os.environ['RANK']))

rank = dist.get_rank()
size = dist.get_world_size()
local_rank = os.environ['PALS_LOCAL_RANKID']
device = "xpu:{}".format(local_rank)
print('world_size:{}, global rank:{}, local_rank:{}'.format(size, rank, local_rank))

# allreduce is WA
data = torch.randn(2, dtype=torch.float32).to(device)
dist.all_reduce(data)

def send_tensor(buffer, recv_stage):
    if isinstance(buffer, torch.Tensor):
        type_tensor = torch.LongTensor(data=[0]).to(device)
        dist.send(type_tensor, recv_stage)
        send_shape = torch.LongTensor(data=buffer.size()).to(device)
        send_ndims = torch.LongTensor(data=[len(buffer.size())]).to(device)
        dist.send(send_ndims, recv_stage)
        dist.send(send_shape, recv_stage)
 
def recv_tensor(send_stage):
    type_tensor = torch.LongTensor(data=[0]).to(device)
    dist.recv(type_tensor, send_stage)
    recv_type = type_tensor.item()
 
    if recv_type == 0:
        recv_ndims = torch.LongTensor(data=[0]).to(device)
        dist.recv(recv_ndims, send_stage)
        recv_ndims = recv_ndims.item()
        recv_shape = torch.LongTensor([1] * recv_ndims).to(device)
        dist.recv(recv_shape, send_stage)
        print("recv_ndims", recv_ndims)
        print("recv_shape", recv_shape)
    else:
        print("----------------error-------------------")
size = dist.get_world_size()
device = "xpu:{}".format(local_rank)
 
data = torch.randn(1, dtype=torch.float32).to(device)
dist.all_reduce(data)
    
# rank1 -> rank3 -> rank15 -> rank23 -> rank8
if rank == 1:
    tensor = torch.ones(2048,3,256).xpu(device)
    send_tensor(tensor, 3)
if rank == 3:
    recv_tensor(1)
    tensor = torch.ones(2048,3,256).xpu(device)
    send_tensor(tensor, 15)
if rank == 15:
    recv_tensor(3)
    tensor = torch.ones(2048,3,256).xpu(device)
    send_tensor(tensor, 23)
if rank == 23:
    recv_tensor(15)
    tensor = torch.ones(2048,3,256).xpu(device)
    send_tensor(tensor, 8)
if rank == 8:
    recv_tensor(23)
