import torch
import intel_extension_for_pytorch
import oneccl_bindings_for_pytorch
import torch.distributed as dist
import os


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--world_size', default=-1, type=int, help='number of gpu for distributed training')
parser.add_argument('--dist_url', default='127.0.0.1', type=str, help='url used to set up distributed training')
parser.add_argument('--dist_port', default='29600', type=str, help='url port used to set up distributed training')
args = parser.parse_args()

os.environ['RANK'] = str(os.environ.get('PMIX_RANK',0))
os.environ['WORLD_SIZE'] = str(args.world_size)
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29600'

init_method = 'tcp://' + args.dist_url + ':' + args.dist_port
dist.init_process_group(backend='ccl', init_method=init_method,
                        world_size=int(os.environ['WORLD_SIZE']), rank=int(os.environ['RANK']))

rank = dist.get_rank()
print("-----global rank: ", rank)
size = dist.get_world_size()
local_rank = os.environ['PALS_LOCAL_RANKID']
device = "xpu:{}".format(local_rank)
print('world_size:{}, global rank:{}, local_rank:{}'.format(size, rank, local_rank))

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

# send/recv(rank0 -> rank12 -> rank24)
if rank <= 11:
    tensor = torch.ones(2048,3,256).xpu(device)
    send_tensor(tensor, rank+12)
elif rank >= 24 :
    recv_tensor(rank-12)
else:
    recv_tensor(rank-12)
    tensor = torch.ones(2048,3,256).xpu(device)
    send_tensor(tensor, rank+12)
print("-----finished send/recv-----")

# all_gather_base after p2p
torch.distributed.barrier()
world_size=36
device = "xpu:{}".format(local_rank)
rank_name_to_time = torch.zeros((world_size, 2),
                                 dtype=torch.float,
                                 device=device)

torch.distributed._all_gather_base(rank_name_to_time.view(-1),
                                   rank_name_to_time[rank, :].view(-1))
print("all_gather is done")
