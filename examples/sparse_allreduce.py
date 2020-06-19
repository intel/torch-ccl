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

def env2int(env_list, default = -1):
    for e in env_list:
        val = int(os.environ.get(e, -1))
        if val >= 0: return val
    return default

def init_distributed(rank = -1, size = -1):

    if rank == -1:
        rank = env2int(['PMI_RANK'], -1)
    if size == -1:
        size = env2int(['PMI_SIZE'], -1)

    if not os.environ.get('MASTER_ADDR', None): os.environ['MASTER_ADDR'] = '127.0.0.1'
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

    dist.init_process_group(backend=backend)

    rank = dist.get_rank()
    size = dist.get_world_size()

    print_all("rank = %d, size = %d, backend = %s" % (rank, size, backend))

class TestSparse(unittest.TestCase, object):

    def test_sparse_all_reduce_sum(self):
        self._test_sparse_all_reduce_sum(lambda t: t)

    def _test_sparse_all_reduce_sum(self, fn):
        tests = simple_sparse_reduce_tests(
            dist.get_rank(),
            dist.get_world_size(),
            num_inputs=1)
        for (inputs, outputs) in tests:
            tensors = [fn(input) for input in inputs]
            print("input: ", tensors[0])
            print("expected: ", outputs[0])
            work = dist.all_reduce(tensors[0], dist.ReduceOp.SUM, dist.group.WORLD, True)
            work.wait()
            results = work.result()
            print("result_inplace: ", tensors[0])
            print("result_outofplace: ", results[0])
            # self.assertEqual(tensors[0], outputs[0])
            self.assertEqual(results[0], outputs[0])
            break;

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

    def assertTensorsEqual(self, a, b, message, prec):
        super().assertEqual(a.size(), b.size(), message)
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
            self.assertLessEqual(max_err, 0, message)

    def assertEqual(self, x, y, prec=None, message='', allow_inf=False):
        if x.is_sparse:
            x = self.safeCoalesce(x)
            y = self.safeCoalesce(y)
            self.assertTensorsEqual(x._indices(), y._indices(), message, prec)
            self.assertTensorsEqual(x._values(), y._values(), message, prec)


def simple_sparse_reduce_tests(rank, world_size, num_inputs=1):
    """
    Generate a number of basic test cases for sparse reduction.
    These cover tensors with a varying number of sparse dimensions and a varying
    number of dense dimensions. The only reduction operation we support is sum.
    """
    def generate(rank, world_size, sparse_dims=1, dense_dims=1):
        # First sparse dimension is [0..rank].
        # Subsequent dimensions are always 0, so we know there is
        # a non-empty intersection between any two sparse tensors.
        indices = [range(rank + 1)]
        shape = [world_size] + [2 for _ in range(dense_dims)]
        for _ in range(sparse_dims - 1):
            indices.append([0] * (rank + 1))
            shape.append(world_size)
        values = torch.ones([rank + 1] + [2 for _ in range(dense_dims)])
        #print("indices: ", indices)
        #print("values: ", values)
        #print("shape: ", shape)
        t = torch.sparse_coo_tensor(indices, values, shape)
        #print("sparse tensor: ", t)
        return torch.sparse_coo_tensor(indices, values, shape)

    def compute_sum(fn, world_size):
        return reduce(lambda a, b: a + b, [fn(rank, world_size) for rank in range(world_size)])

    print("rank: ", rank)
    print("size: ", world_size)
    print("num_inputs: ", num_inputs)

    return [
        (
            [
                fn(num_inputs * rank + i, num_inputs * world_size)
                for i in range(num_inputs)
            ],
            [
                compute_sum(fn, num_inputs * world_size)
                for i in range(num_inputs)
            ],
        )
        for fn in [
            partial(generate, sparse_dims=1, dense_dims=3)
        ]
    ]


if __name__ == "__main__":
    init_distributed(rank=-1, size=-1)
    unittest.main()
