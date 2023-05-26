import torch
import numpy as np
import time
import os
import argparse
import torch.distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument('--ptrace',
                    action='store_true',
                    default=False,
                    help='pytorch trace')
parser.add_argument('--warm', type=int, default=10, help='#warmup')
parser.add_argument('--iter', type=int, default=10, help='#iteration')
parser.add_argument('--size', type=int, default=25557032, help='number of f32/bf16 elements')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--broadcast', action='store_true', default=False)
parser.add_argument('--bf16', action='store_true', default=False)
parser.add_argument('--fixed',
                    action='store_true',
                    default=False,
                    help='fixed size')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if 'PMI_RANK' in os.environ.keys() and 'PMI_SIZE' in os.environ.keys():
    os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
    os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1)) # mpich set
elif 'PMIX_RANK' in os.environ.keys() and 'PALS_LOCAL_SIZE' in os.environ.keys():
    os.environ['RANK'] = os.environ.get('PMIX_RANK')
    os.environ['WORLD_SIZE'] = str(os.environ.get('PALS_LOCAL_SIZE', -1))
os.environ['MASTER_ADDR'] = '127.0.0.1'  # your master address
os.environ['MASTER_PORT'] = '29500'  # your master port

if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ.keys():
    local_rank = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
elif 'MPI_LOCALRANKID' in os.environ.keys():
    local_rank = os.environ['MPI_LOCALRANKID']
    if 'MPI_LOCALNRANKS' in os.environ.keys():
        os.environ['LOCAL_WORLD_SIZE'] = str(os.environ.get('MPI_LOCALNRANKS',-1))
else:
    local_rank = os.environ['PALS_LOCAL_RANKID']

local_rank = int(local_rank)
devid = local_rank

if not args.cuda:
    import intel_extension_for_pytorch
    try:
        import oneccl_bindings_for_pytorch
    except:
        import torch_ccl
    torch.xpu.set_device(devid)
    device = "xpu:{}".format(devid)
    dist.init_process_group(backend='ccl')
else:
    torch.cuda.set_device(devid)
    device = "cuda"
    dist.init_process_group(backend='nccl')

try:
    from horovod.torch import mpi_lib_v2 as mpi_lib
    if mpi_lib.ctenabled():
        mpi_lib = mpi_lib
except:
    mpi_lib = None

print(f'DDP local rank: {devid}')

if devid == 0:
    print(f'PyTorch DDP {"Broadcast" if args.broadcast else "AllReduce"} on {os.environ["WORLD_SIZE"]} {device} devices: ')

def _time():
    if args.cuda:
        torch.cuda.synchronize()
    else:
        torch.xpu.synchronize()
    return time.time()

if args.fixed:
    N = args.size
else:
    N = 1


with torch.autograd.profiler.profile(enabled=args.ptrace) as prof:
    while N <= args.size:
        for i in range(args.warm):
            data = torch.randn(N, dtype=torch.bfloat16 if args.bf16 else torch.float32).to(device)
            with torch.no_grad():
                if not args.broadcast:
                    dist.all_reduce(data)
                else:
                    dist.broadcast(data, 0)
        elapsed = []
        for i in range(args.iter):
            data = torch.randn(N, dtype=torch.bfloat16 if args.bf16 else torch.float32).to(device)
            t = _time()
            if mpi_lib:
                mpi_lib.ctpush("IPEX_ALLREDUCE")
            with torch.no_grad():
                if not args.broadcast:
                    dist.all_reduce(data)
                else:
                    dist.broadcast(data, 0)
            elapsed.append((_time() - t) * 1e6)
            if mpi_lib and mpi_lib.ctenabled():
                mpi_lib.ctpop()
        if devid == 0:
            print(
                f'{N*(2 if args.bf16 else 4):<10}{np.mean(elapsed):>10.1f}us ({np.min(elapsed):.1f}-{np.max(elapsed):.1f}) +-{1.96 * np.std(elapsed):.1f}'
            )
        if N == args.size:
            break
        N = 2 * N
        if N != args.size and N > args.size:
            N = args.size

if args.ptrace:
    prof.export_chrome_trace('rank' + str(hvd.rank()) + '_timeline.json')
dist.destroy_process_group()

