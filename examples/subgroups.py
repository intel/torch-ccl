import unittest
import os
import builtins
import numpy as np
import torch
from torch.autograd import Function
import torch.distributed as dist
from functools import partial, reduce

try:
    import torch_ccl
except ImportError as e:
    print(e)
    torch_ccl = False

my_rank = -1
my_size = -1
my_local_rank = -1
my_local_size = -1

def env2int(env_list, default = -1):
    for e in env_list:
        val = int(os.environ.get(e, -1))
        if val >= 0: return val
    return default

def init_distributed(rank = -1, size = -1, backend=''):
    global g1
    global g2
    global g3

    # guess MPI ranks from env (works for IMPI, OMPI and MVAPICH2)
    num_mpi_ranks = env2int(['PMI_SIZE', 'OMPI_COMM_WORLD_SIZE', 'MV2_COMM_WORLD_SIZE', 'WORLD_SIZE'])
    if backend == '' and num_mpi_ranks > 1:
        if torch_ccl and env2int(['CCL_WORKER_COUNT']) > 0:
            backend = 'ccl'
        elif dist.is_mpi_available():
            backend = 'mpi'
        else:
            print("WARNING: MPI multi-process launch detected but PyTorch MPI backend not available.")
            backend = 'gloo'

    if backend != '':
        #guess Rank and size
        if rank == -1:
            rank = env2int(['PMI_RANK', 'OMPI_COMM_WORLD_RANK', 'MV2_COMM_WORLD_RANK', 'RANK'], 0)
        if size == -1:
            size = env2int(['PMI_SIZE', 'OMPI_COMM_WORLD_SIZE', 'MV2_COMM_WORLD_SIZE', 'WORLD_SIZE'], 1)
        if not os.environ.get('RANK', None) and rank != -1: os.environ['RANK'] = str(rank)
        if not os.environ.get('WORLD_SIZE', None) and size != -1: os.environ['WORLD_SIZE'] = str(size)
        if not os.environ.get('MASTER_PORT', None): os.environ['MASTER_PORT'] = '29500'
        if not os.environ.get('MASTER_ADDR', None):
            local_size = env2int(['MPI_LOCALNRANKS', 'OMPI_COMM_WORLD_LOCAL_SIZE', 'MV2_COMM_WORLD_LOCAL_SIZE'], 1)
            if local_size != size and backend != 'mpi':
                print("Warning: Looks like distributed multinode run but MASTER_ADDR env not set, using '127.0.0.1' as default")
                print("If this run hangs, try exporting rank 0's hostname as MASTER_ADDR")
            os.environ['MASTER_ADDR'] = '127.0.0.1'

    if size > 1:
        dist.init_process_group(backend, rank=rank, world_size=size)
        my_rank = dist.get_rank()
        my_size = dist.get_world_size()
        my_local_rank = env2int(['MPI_LOCALRANKID', 'OMPI_COMM_WORLD_LOCAL_RANK', 'MV2_COMM_WORLD_LOCAL_RANK'], 0)
        my_local_size = env2int(['MPI_LOCALNRANKS', 'OMPI_COMM_WORLD_LOCAL_SIZE', 'MV2_COMM_WORLD_LOCAL_SIZE'], 1)
        if my_rank == 0: print("Running on %d ranks using %s backend" % (my_size, backend))
        if backend == 'ccl':
            print("Using CCL_ATL_TRANSPORT=%s" % os.environ.get('CCL_ATL_TRANSPORT', '(default)'))
            print("Using CCL_ATL_SHM=%s" % os.environ.get('CCL_ATL_SHM', '(default)'))
            pass
    else:
        my_rank = 0
        my_size = 1
        my_local_rank = 0
        my_local_size = 1

    # create 2 subgroups
    g1 = dist.new_group(ranks=[0, 1], backend=backend)

    g2 = dist.new_group(ranks=[2, 3], backend=backend)

    g3 = dist.new_group(ranks=[0, 2, 3], backend=backend)

class TestAllReduce(unittest.TestCase):
    def test_all_reduce_sum(self):
        self._test_all_reduce_sum(lambda t: t)

    def _test_all_reduce_sum(self, fn):
        #print("_test_all_reduce_sum : Main group: rank: ", dist.get_rank(), " : size: ", dist.get_world_size())
        tests = simple_allreduce_tests(
            dist.get_rank(),
            dist.get_world_size())

        for (inputs, outputs) in tests:
            tensors = [fn(input) for input in inputs]
            print("_test_all_reduce_sum : input: ", tensors[0])
            print("_test_all_reduce_sum : expected: ", outputs[0])
            dist.all_reduce(tensors[0], dist.ReduceOp.SUM, dist.group.WORLD)
            print("_test_all_reduce_sum : output: ", tensors[0])
            self.assertEqual(tensors[0], outputs[0])
            break

        #print("_test_all_reduce_sum : g1: rank: ", dist.get_rank(g1), " : size: ", dist.get_world_size(g1))
        tests1 = simple_allreduce_tests(
            dist.get_rank(g1),
            dist.get_world_size(g1))

        for (inputs, outputs) in tests1:
            tensors = [fn(input) for input in inputs]
            print("_test_all_reduce_sum : input: ", tensors[0])
            print("_test_all_reduce_sum : expected: ", outputs[0])
            dist.all_reduce(tensors[0], dist.ReduceOp.SUM, g1)
            print("_test_all_reduce_sum : output: ", tensors[0])
            self.assertEqual(tensors[0], outputs[0])
            break


        #print("_test_all_reduce_sum : g2: rank: ", dist.get_rank(g2), " : size: ", dist.get_world_size(g2))
        tests2 = simple_allreduce_tests(
            dist.get_rank(g2),
            dist.get_world_size(g2))

        for (inputs, outputs) in tests2:
            tensors = [fn(input) for input in inputs]
            print("_test_all_reduce_sum : input: ", tensors[0])
            print("_test_all_reduce_sum : expected: ", outputs[0])
            dist.all_reduce(tensors[0], dist.ReduceOp.SUM, g2)
            print("_test_all_reduce_sum : output: ", tensors[0])
            self.assertEqual(tensors[0], outputs[0])
            break

        #print("_test_all_reduce_sum : g3: rank: ", dist.get_rank(g3), " : size: ", dist.get_world_size(g3))
        tests3 = simple_allreduce_tests(
            dist.get_rank(g3),
            dist.get_world_size(g3))

        for (inputs, outputs) in tests3:
            tensors = [fn(input) for input in inputs]
            print("_test_all_reduce_sum : input: ", tensors[0])
            print("_test_all_reduce_sum : expected: ", outputs[0])
            dist.all_reduce(tensors[0], dist.ReduceOp.SUM, g3)
            print("_test_all_reduce_sum : output: ", tensors[0])
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
    init_distributed(backend='ccl')
    unittest.main()