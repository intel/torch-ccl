import torch
import torch.distributed as dist
from torch.multiprocessing import Process
import os
import intel_extension_for_pytorch

world_size = 12
rounds = 100
# input_file = "Example.csv"
input_file = "DeepSpeed.csv"

type = torch.bfloat16

def worker(given_rank):
    os.environ['MASTER_ADDR'] = '127.0.0.1' # xpu
    os.environ['MASTER_PORT'] = '29500' # xpu
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(given_rank)

    try:
        import oneccl_bindings_for_pytorch
    except ImportError:
        print("oneccl_bindings_for_pytorch not available!")
    dist.init_process_group(backend='ccl')

    rank = int(dist.get_rank())
    torch.xpu.set_device(rank)
    device = "xpu:{}".format(rank)

    ops, sizes, roots = read_file(input_file)
    test_ccl(ops, sizes, roots, device, rank, rounds)
 
 
def main():
    
    process_list = []
    for i in range(world_size):
        p = Process(target=worker, args=(i,))
        p.start()
        process_list.append(p)
 
    for p in process_list:
        p.join()

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
        data = torch.randn(sizes[i], dtype = type)
        data = data.to(device)
        input.append(data)
        if ops[i] == 'allgather':
            tmp_output = []
            for j in range(0, world_size):
                data = torch.randn(sizes[i], dtype = type)
                data = data.to(device)
                tmp_output.append(data)
            output.append(tmp_output)
        else:
            output.append(data)
    print("Rank {}: tensors initialization finished!".format(rank))
    for k in range(0, rounds):
        print("test round: ", k)
        for i in range(0, len(ops)):
            if ops[i] == 'reduce':
                print("Rank {}: reduce to {} w/ size {}".format(rank, roots[i], len(input[i])))
                dist.reduce(input[i], roots[i], async_op=False)
            if ops[i] == 'allreduce':
                print("Rank {}: all_reduce w/ size {}".format(rank, len(input[i])))
                dist.all_reduce(input[i], async_op=False)
            if ops[i] == 'allgather':
                print("Rank {}: all_gather w/ size {} & {} elements".format(rank, len(input[i]), len(output[i])))
                dist.all_gather(output[i], input[i], async_op=False)
            if ops[i] == 'broadcast':
                print("Rank {}: broadcast from {} w/ size {}".format(rank, roots[i], len(input[i])))
                dist.broadcast(input[i], roots[i], async_op=False)

            torch.xpu.synchronize()
        
if __name__ == '__main__':
    main()
    print("All tests finished!")
