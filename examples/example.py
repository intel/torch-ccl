#
# Copyright (c) 2020, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the Intel Corporation nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#

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

print_all("all_to_all_single with empty tensors")
dist.all_to_all_single(torch.empty([0]), torch.empty([0]))

dist.barrier()

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
