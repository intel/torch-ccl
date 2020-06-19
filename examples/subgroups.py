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

import unittest
import os
import builtins
import numpy as np
import torch
from torch.autograd import Function
import torch.distributed as dist
from functools import partial, reduce

rank = -1
size = -1


def print0(str):
    if rank == 0: print(str)


def print_all(str):
    for i in range(size):
        dist.barrier()
        if i == rank: print(str)
    dist.barrier()


def env2int(env_list, default=-1):
    for e in env_list:
        val = int(os.environ.get(e, -1))
        if val >= 0: return val
    return default


def init_distributed(rank=-1, size=-1):
    global g1
    global g2
    global g3
    global group_list

    if rank == -1:
        rank = env2int(['PMI_RANK'], -1)
    if size == -1:
        size = env2int(['PMI_SIZE'], -1)

    if not os.environ.get('MASTER_ADDR', None):
        local_size = env2int(['MPI_LOCALNRANKS', 'OMPI_COMM_WORLD_LOCAL_SIZE', 'MV2_COMM_WORLD_LOCAL_SIZE'], 1)
        if local_size != size and backend != 'mpi':
            print(
                "Warning: Looks like distributed multinode run but MASTER_ADDR env not set, using '127.0.0.1' as default")
            print("If this run hangs, try exporting rank 0's hostname as MASTER_ADDR")
        os.environ['MASTER_ADDR'] = '127.0.0.1'
    if not os.environ.get('MASTER_PORT', None): os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = os.environ.get('PMI_RANK', -1)
    os.environ['WORLD_SIZE'] = os.environ.get('PMI_SIZE', -1)

    backend = os.environ.get('BACKEND', 'ccl')
    if backend == 'ccl':
        try:
            import torch_ccl
        except ImportError as e:
            print(e)
            print("can't import torch_ccl module")
            sys.exit()

    if size > 1:
        dist.init_process_group(backend, rank=rank, world_size=size)
        rank = dist.get_rank()
        size = dist.get_world_size()

        if size < 4:
            print("Number of ranks must be >=4")
            sys.exit()

        g1 = dist.new_group(ranks=[0, 1], backend=backend)
        g2 = dist.new_group(ranks=[2, 3], backend=backend)
        g3 = dist.new_group(ranks=[0, 2, 3], backend=backend)
        group_list = [dist.group.WORLD, g1, g2, g3]

    else:
        rank = 0
        size = 1

    print("rank = %d, size = %d, backend = %s" % (rank, size, backend))


class TestAllReduce(unittest.TestCase):
    def test_all_reduce_sum(self):
        self._test_all_reduce_sum(lambda t: t)

    def _test_all_reduce_sum(self, fn):
        for i in range(len(group_list)):
            g = group_list[i]

        print("group: rank: ", dist.get_rank(), " : size: ", dist.get_world_size())

        tests = simple_allreduce_tests(
            dist.get_rank(g),
            dist.get_world_size(g))

        for (inputs, outputs) in tests:
            tensors = [fn(input) for input in inputs]
            print("input: ", tensors[0])
            print("expected: ", outputs[0])
            dist.all_reduce(tensors[0], dist.ReduceOp.SUM, g)
            print("output: ", tensors[0])
            self.assertEqual(tensors[0], outputs[0])
            break

    def safeCoalesce(self, t):
        tc = t.coalesce()
        self.assertEqual(tc.to_dense(), t.to_dense())
        self.assertTrue(tc.is_coalesced())

        # Our code below doesn't work when nnz is 0, because
        # then it's a 0D tensor, not a 2D tensor.
        if t._nnz() == 0:
            self.assertEqual(t._indices(), tc._indices())
            self.assertEqual(t._values(), tc._values())
            return tc

        value_map = {}
        for idx, val in zip(t._indices().t(), t._values()):
            idx_tup = tuple(idx.tolist())
            if idx_tup in value_map:
                value_map[idx_tup] += val
            else:
                value_map[idx_tup] = val.clone() if isinstance(val, torch.Tensor) else val

        new_indices = sorted(list(value_map.keys()))
        new_values = [value_map[idx] for idx in new_indices]
        if t._values().ndimension() < 2:
            new_values = t._values().new(new_values)
        else:
            new_values = torch.stack(new_values)

        new_indices = t._indices().new(new_indices).t()
        tg = t.new(new_indices, new_values, t.size())

        self.assertEqual(tc._indices(), tg._indices())
        self.assertEqual(tc._values(), tg._values())

        if t.is_coalesced():
            self.assertEqual(tc._indices(), t._indices())
            self.assertEqual(tc._values(), t._values())

        return tg

    def assertEqual(self, x, y, prec=0, message='', allow_inf=False):
        def assertTensorsEqual(a, b):
            if a.numel() > 0:
                b = b.to(a)

                diff = a - b
                if a.is_floating_point():
                    # check that NaNs are in the same locations
                    nan_mask = torch.isnan(a)
                    self.assertTrue(torch.equal(nan_mask, torch.isnan(b)), message)
                    diff[nan_mask] = 0

                # TODO: implement abs on CharTensor (int8)
                if diff.is_signed() and diff.dtype != torch.int8:
                    diff = diff.abs()
                max_err = diff.max()
                self.assertLessEqual(max_err, prec, message)

        if x.is_sparse:
            x = self.safeCoalesce(x)
            y = self.safeCoalesce(y)
            assertTensorsEqual(x._indices(), y._indices())
            assertTensorsEqual(x._values(), y._values())
        else:
            assertTensorsEqual(x, y)


def simple_allreduce_tests(rank, world_size):
    tests = [
        (
            torch.tensor([rank + 1.0]),
            torch.tensor([float(world_size * (world_size + 1) / 2)]),
        )
    ]

    return tests


if __name__ == "__main__":
    init_distributed()
    unittest.main()