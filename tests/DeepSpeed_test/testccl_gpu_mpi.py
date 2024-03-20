import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import os
import intel_extension_for_pytorch
import oneccl_bindings_for_pytorch
import argparse
import sys

rounds = 40
# input_file = "Example.csv"
input_file = "DeepSpeed.csv"

data_type = torch.bfloat16

os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'
dist.init_process_group("ccl")
rank = dist.get_rank()
world_size = dist.get_world_size()

def main():
    torch.xpu.set_device(rank)
    device = "xpu:{}".format(rank)
    ops, sizes, roots = read_file(input_file)
    test_ccl(ops, sizes, roots, device, rank, rounds)

def read_file(filename):
    ops = []
    sizes = []
    roots = []
    f = open(filename, "r")
    for line in f:
        op, size, root = line.strip().split(",")
        size = int(size)
        root = int(root)
        if root >= world_size:
            print("Invalid root {}".format(root))
            exit()
        ops.append(op)
        sizes.append(size)
        roots.append(root)
    f.close()
    return ops, sizes, roots

def test_ccl(ops, sizes,  roots, device, rank, rounds):
    input = []
    output = []
    print("Rank {}: starting to initialize tensors ...".format(rank))
    for i in range(0, len(sizes)):
        data = torch.randn(sizes[i], dtype = data_type)
        data = data.to(device)
        input.append(data)
        if ops[i] == 'allgather':
            tmp_output = []
            for j in range(0, world_size):
                data = torch.randn(sizes[i], dtype = data_type)
                data = data.to(device)
                tmp_output.append(data)
            output.append(tmp_output)
        else:
            output.append(data)
    print("Rank {}: tensors initialization finished!".format(rank), flush=True)
    for k in range(0, rounds):
        print("test round: ", k)
        for i in range(0, len(ops)):
            if ops[i] == 'reduce':
                print("Rank {}: reduce to {} w/ size {}".format(rank, roots[i], len(input[i])), flush=True)
                dist.reduce(input[i], roots[i], async_op=False)
            if ops[i] == 'allreduce':
                print("Rank {}: all_reduce w/ size {}".format(rank, len(input[i])), flush=True)
                dist.all_reduce(input[i], async_op=False)
            if ops[i] == 'allgather':
                print("Rank {}: all_gather w/ size {} & {} elements".format(rank, len(input[i]), len(output[i])), flush=True)
                dist.all_gather(output[i], input[i], async_op=False)
            if ops[i] == 'broadcast':
                print("Rank {}: broadcast from {} w/ size {}".format(rank, roots[i], len(input[i])), flush=True)
                dist.broadcast(input[i], roots[i], async_op=False)

            torch.xpu.synchronize()
        
if __name__ == '__main__':
    main()
    print("All tests finished!")
