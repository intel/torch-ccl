import sys
import os
import torch

try:
    import intel_extension_for_pytorch
    xpu_is_avaliable = torch.xpu.is_available() if hasattr(torch, 'xpu') else False
except ImportError:
    # ignore the ipex
    xpu_is_avaliable = False
    pass

import oneccl_bindings_for_pytorch
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.common_distributed import MultiProcessTestCase, \
     simple_sparse_reduce_tests, \
     TEST_SKIPS, \
     TestSkip

import torch.distributed as c10d

import math
from functools import reduce, wraps
import operator

cpu_device = torch.device("cpu")


def skip_if_not_multixpu(func):
    """Multi-XPU tests requires at least 2 XPUS. Skip if this is not met."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if xpu_is_avaliable and torch.xpu.device_count() >= 2:
            return func(*args, **kwargs)
        message = "Need at least {} XPU devices".format(2)
        TEST_SKIPS["multi-gpu"] = TestSkip(75, message)
        sys.exit(TEST_SKIPS['multi-gpu'].exit_code)

    return wrapper


def skip_if_no_xpu(func):
    """ oneCCL xpu tests require at least 1 XPU. Skip if this is not met"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not xpu_is_avaliable:
            sys.exit(TEST_SKIPS["no_cuda"].exit_code)
        # if torch.xpu.device_count() < int(os.environ["WORLD_SIZE"]):
        #     message = "Need at least {} XPU devices".format(os.environ["WORLD_SIZE"])
        #     TEST_SKIPS["multi-gpu"] = TestSkip(75, message)
        #     sys.exit(TEST_SKIPS["multi-gpu"].exit_code)

        return func(*args, **kwargs)

    return wrapper


TEST_SKIPS["skip_test"] = TestSkip(80, "Skipped because test is not available.")


def skip_test(func):
    """ oneCCL skip test """
    @wraps(func)
    def wrapper(*args, **kwargs):
        sys.exit(TEST_SKIPS["skip_test"].exit_code)

    return wrapper


def simple_reduce_tests(rank, world_size):
    tests = [
        (
            c10d.ReduceOp.SUM,
            torch.tensor([rank + 1.0]),
            torch.tensor([float(world_size * (world_size + 1) / 2)]),
        ),
        (
            c10d.ReduceOp.PRODUCT,
            torch.tensor([rank + 1.0]),
            torch.tensor([float(math.factorial(world_size))]),
        ),
        (
            c10d.ReduceOp.MIN,
            torch.tensor([rank + 1.0]),
            torch.tensor([1.0]),
        ),
        (
            c10d.ReduceOp.MAX,
            torch.tensor([rank + 1.0]),
            torch.tensor([float(world_size)]),
        ),
    ]
    return tests

class ProcessGroupCCLTest(MultiProcessTestCase):

    def setUp(self):
        super(ProcessGroupCCLTest, self).setUp()
        self._spawn_processes()

    def test_broadcast_checks(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupCCL(store, self.rank, self.world_size)

        t1 = torch.zeros([1], dtype=torch.float32)
        t2 = torch.zeros([1], dtype=torch.float64)
        t3 = torch.zeros([2], dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "unexpected rank"):
            opts = c10d.BroadcastOptions()
            opts.rootRank = -1
            opts.rootTensor = 0
            pg.broadcast([t1], opts)


    def _test_broadcast_basics(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupCCL(store, self.rank, self.world_size)

        def broadcast(xs, rootRank, rootTensor):
            opts = c10d.BroadcastOptions()
            opts.rootRank = rootRank
            opts.rootTensor = rootTensor
            work = pg.broadcast(xs, opts)
            work.wait()

        # Every rank is root once
        for i in range(self.world_size):
            # Run with 1 input tensor
            x = fn(torch.tensor([self.rank]))
            broadcast([x], i, 0)
            # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
            self.assertEqualIgnoreType(torch.tensor([i]), x)

        x = fn(torch.tensor([self.rank + 1.0]))
        work = pg.broadcast(x, root=0)
        work.wait()
        self.assertEqual(torch.tensor([1.0]), x)

    def test_broadcast_basics(self):
        self._test_broadcast_basics(lambda t: t.clone())

    @skip_if_no_xpu
    def test_broadcast_basics_xpu(self):
        self._test_broadcast_basics(lambda t: t.clone().xpu())

    @skip_if_not_multixpu
    def test_broadcast_basics_multi_xpu(self):
        self._test_broadcast_basics(lambda t: t.clone().xpu("xpu:{}".format(self.rank)))


    def _test_broadcast_stress(self, inputs):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupCCL(store, self.rank, self.world_size)
        work_handles = [
            pg.broadcast(inputs[i], root=(i % self.world_size))
            for i in range(len(inputs))
        ]
        for i, work_handle in enumerate(work_handles):
            work_handle.wait()
            self.assertEqual(
                torch.tensor([
                    (i * self.world_size) + (i % self.world_size)
                ]),
                inputs[i],
                msg=("Mismatch in iteration %d" % i),
            )

    def test_broadcast_stress(self):
        inputs = [torch.tensor([i * self.world_size + self.rank]) for i in range(1000)]
        self._test_broadcast_stress(inputs)

    def _test_allreduce_basics(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupCCL(store, self.rank, self.world_size)

        # Single input tests
        tests = simple_reduce_tests(self.rank, self.world_size)
        for (op, input, output) in tests:
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            tensor = fn(input)
            work = pg.allreduce([tensor], opts)
            work.wait()

            # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
            self.assertEqualIgnoreType(output, tensor)

    def test_allreduce_basics(self):
        self._test_allreduce_basics(lambda t: t.clone())

    @skip_if_no_xpu
    def test_allreduce_basics_xpu(self):
        self._test_allreduce_basics(lambda t: t.clone().xpu())

    @skip_if_not_multixpu
    def test_allreduce_basics_multi_xpu(self):
        self._test_allreduce_basics(lambda t: t.clone().xpu("xpu:{}".format(self.rank)))
    
    def _test_allreduce_coalesced_basics(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupCCL(store, self.rank, self.world_size)
        tensors = [fn(torch.full((5 + i,), self.rank + 1 + i, dtype=torch.float)) for i in range(5)]
        opts = c10d.AllreduceCoalescedOptions()
        opts.reduceOp = c10d.ReduceOp.SUM
                
        pg.allreduce_coalesced(tensors, opts)
        for i, t in enumerate(tensors):
            self.assertEqual(t, torch.full_like(t, self.world_size * (i + (self.world_size + 1.) / 2.)))
    
    def test_allreduce_coalesced_basics(self):
        self._test_allreduce_coalesced_basics(lambda t: t.clone())
    
    @skip_if_no_xpu   
    def test_allreduce_coalesced_basics_xpu(self):
        self._test_allreduce_coalesced_basics(lambda t: t.clone().xpu())

    @skip_if_not_multixpu  
    def test_allreduce_coalesced_basics_multi_xpu(self):
        self._test_allreduce_coalesced_basics(lambda t: t.clone().xpu("xpu:{}".format(self.rank)))
    
    def _test_reduce_basics(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupCCL(store, self.rank, self.world_size)

        for (op, input, output) in simple_reduce_tests(self.rank, self.world_size):

            for root in range(self.world_size):
                opts = c10d.ReduceOptions()
                opts.reduceOp = op
                opts.rootRank = root
                tmp = fn(input)
                work = pg.reduce([tmp], opts)
                work.wait()
                #print(op, " ", input, " " , output)
                if root == self.rank:
                    self.assertEqual(output, tmp)

    @property
    def world_size(self):
        return 2

    def test_reduce_basics(self):
        self._test_reduce_basics(lambda t: t.clone())

    @skip_if_no_xpu
    def test_reduce_basics_xpu(self):
        self._test_reduce_basics(lambda t: t.clone().xpu())

    @skip_if_not_multixpu
    def test_reduce_basics_multi_xpu(self):
        self._test_reduce_basics(lambda t: t.clone().xpu("xpu:{}".format(self.rank)))

    def _test_gather_basics(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupCCL(store, self.rank, self.world_size)

        # Preallocate tensors for input/output
        input = [fn(torch.tensor([self.rank]))]
        outputs = [fn(torch.tensor([-1])) for _ in range(self.world_size)]

        # Take turns being the gather root and accumulate work items
        work = []
        for i in range(self.world_size):
            opts = c10d.GatherOptions()
            opts.rootRank = i
            if i == self.rank:
                work.append(pg.gather([outputs], input, opts))
            else:
                work.append(pg.gather([], input, opts))

        # Wait for work to complete
        expected = [torch.tensor([rank]) for rank in range(self.world_size)]
        for i in range(self.world_size):
            work[i].wait()
            if i == self.rank:
                self.assertEqual(expected, outputs)

    def test_gather_basics(self):
        self._test_gather_basics(lambda t: t.clone())

    @skip_if_no_xpu
    def test_gather_basics_xpu(self):
        self._test_gather_basics(lambda t: t.clone().xpu())

    @skip_if_not_multixpu
    def test_gather_basics_multi_xpu(self):
        self._test_gather_basics(lambda t: t.clone().xpu("xpu:{}".format(self.rank)))
   
    def test_allgather_base_ops(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupCCL(store, self.rank, self.world_size)

        def allgather_base(output_t, input_t):
            work = pg._allgather_base(output_t, input_t)
            work.wait()

        tensor = torch.tensor([self.rank])
        output_t = torch.empty((self.world_size), dtype=tensor.dtype)
        allgather_base(output_t, tensor)

    def test_reduce_scatter_base_ops(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupCCL(store, self.rank, self.world_size)

        def reduce_scatter_base(output_t, input_t):
            work = pg._reduce_scatter_base(output_t, input_t)
            work.wait()

        tensor = torch.arange(self.world_size)
        output_t = torch.tensor(self.rank, dtype=tensor.dtype)
        reduce_scatter_base(output_t, tensor)

    def _test_allgather_basics(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupCCL(store, self.rank, self.world_size)

        # Run with 1 input tensor per rank
        for n in range(self.world_size):
            input= [fn(torch.tensor([n * self.rank]))]
            output = [
                [
                    fn(torch.tensor([-1])) for _ in range(self.world_size)
                ]
            ]
            expected_output = [
                [
                    torch.tensor([i*n]) for i in range(self.world_size)
                ]
            ]
            work = pg.allgather(output, input)
            work.wait()
            self.assertEqual(expected_output, output)

    def test_allgather_basics(self):
        self._test_allgather_basics(lambda t: t.clone())

    @skip_if_no_xpu
    def test_allgather_basics_xpu(self):
        self._test_allgather_basics(lambda t: t.clone().xpu())

    @skip_if_not_multixpu
    def test_allgather_basics_multi_xpu(self):
        self._test_allgather_basics(lambda t: t.clone().xpu("xpu:{}".format(self.rank)))

    def _test_allgather_base_ops(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupCCL(store, self.rank, self.world_size)

        def allgather_base(output_t, input_t):
            work = pg._allgather_base(output_t, input_t)
            work.wait()

        tensor = fn(torch.tensor([self.rank]))
        output_t = fn(torch.empty((self.world_size), dtype=tensor.dtype))

        allgather_base(output_t, tensor)

        # Verification
        self.assertEqual(torch.arange(self.world_size), output_t)

    def test_allgather_base_ops(self):
        self._test_allgather_base_ops(lambda t: t.clone())

    @skip_if_no_xpu
    def test_allgather_base_ops_xpu(self):
        self._test_allgather_base_ops(lambda t: t.clone().xpu())

    @skip_if_not_multixpu
    def test_allgather_basics_multi_xpu(self):
        self._test_allgather_basics(lambda t: t.clone().xpu("xpu:{}".format(self.rank)))

    def _test_allgather_base_basics(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupCCL(store, self.rank, self.world_size)

        def allgather_base(output_t, input_t):
            work = pg._allgather_base(output_t, input_t)
            work.wait()

        # anticpate an error
        with self.assertRaisesRegex(
                RuntimeError,
                "output tensor size must be equal to world_size times input tensor size",
        ):
            tensor = fn(torch.tensor([self.rank]))
            output_t = fn(torch.empty((self.world_size + 1), dtype=tensor.dtype))
            # fails the check because output_t is not correctly sized
            allgather_base(output_t, tensor)

        # anticpate an error
        with self.assertRaisesRegex(
                RuntimeError, "Tensors are not equal in data type"
        ):
            tensor = fn(torch.tensor([self.rank], dtype=torch.float))
            output_t = fn(torch.empty((self.world_size + 1), dtype=torch.long))
            # fails the check because the dtype is different
            allgather_base(output_t, tensor)

    def test_allgather_base_basics(self):
        self._test_allgather_base_basics(lambda t: t.clone())

    @skip_if_no_xpu
    def test_allgather_base_basics_xpu(self):
        self._test_allgather_base_basics(lambda t: t.clone().xpu())

    @skip_if_not_multixpu
    def test_allgather_base_basics_multi_xpu(self):
        self._test_allgather_base_basics(lambda t: t.clone().xpu("xpu:{}".format(self.rank)))

    # alltoall_base
    def _test_alltoall_base_equal_split_helper(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupCCL(store, self.rank, self.world_size)
        group = list(range(0, self.world_size))
        rank = self.rank
        size = len(group)
        in_tensor = fn(torch.ones([size, size]) * rank)
        expected_tensor = torch.cat([torch.ones([1, size]) * i for i in group])
        out_tensor = fn(torch.ones([size, size]) * -1)
        in_splits = []
        out_splits = []
        work = pg.alltoall_base(out_tensor, in_tensor, out_splits, in_splits)
        work.wait()
        self.assertEqual(out_tensor.cpu(), expected_tensor.cpu())

    def _test_alltoall_base_unequal_split_helper(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupCCL(store, self.rank, self.world_size)
        group = list(range(0, self.world_size))
        rank = self.rank
        size = len(group)
        in_splits = [i + 1 for i in group]
        out_splits = [rank + 1 for _ in group]
        in_tensor = fn(torch.ones([sum(in_splits), size]) * rank)
        out_tensor = fn(torch.ones([(rank + 1) * size, size]))
        expected_tensor = fn(torch.cat([torch.ones([rank + 1, size]) * i for i in group]))

        work = pg.alltoall_base(
             out_tensor, in_tensor, out_splits, in_splits)
        work.wait()
        self.assertEqual(out_tensor.cpu(), expected_tensor.cpu())

    def test_allotall_equal_split_basics(self):
        self._test_alltoall_base_equal_split_helper(lambda t: t.clone())

    @skip_if_no_xpu
    def test_allotall_equal_split_basics_xpu(self):
        self._test_alltoall_base_equal_split_helper(lambda t: t.clone().xpu())

    @skip_if_not_multixpu
    def test_allotall_equal_split_basics_multi_xpu(self):
        self._test_alltoall_base_equal_split_helper(lambda t: t.clone().xpu("xpu:{}".format(self.rank)))

    def test_allotall_unequal_split_basics(self):
        self._test_alltoall_base_unequal_split_helper(lambda t: t.clone())

    @skip_if_no_xpu
    def test_allotall_unequal_split_basics_xpu(self):
        self._test_alltoall_base_unequal_split_helper(lambda t: t.clone().xpu())

    @skip_if_not_multixpu
    def test_allotall_unequal_split_basics_multi_xpu(self):
        self._test_alltoall_base_unequal_split_helper(lambda t: t.clone().xpu("xpu:{}".format(self.rank)))

    #alltoall
    def _test_all_to_all_helper(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupCCL(store, self.rank, self.world_size)
        group = list(range(0, self.world_size))
        rank = self.rank
        size = len(group)
        in_splits = [i + 1 for i in group]
        in_tensors = [
            torch.ones([in_splits[i], size]) * rank for i, _ in enumerate(group)
        ]
        out_tensors = [torch.ones([(rank + 1), size]) for _ in group]
        expected_tensors = [torch.ones([rank + 1, size]) * i for i in group]

        in_tensors = [fn(t) for t in in_tensors]
        out_tensors = [fn(t) for t in out_tensors]
        expected_tensors = [fn(t) for t in expected_tensors]

        work = pg.alltoall(out_tensors, in_tensors)
        work.wait()
        for t1, t2 in zip(out_tensors, expected_tensors):
            self.assertEqual(t1.cpu(), t2.cpu())

    def test_alltoall_basics(self):
        self._test_all_to_all_helper(lambda t: t.clone())

    @skip_if_no_xpu
    def test_alltoall_basics_xpu(self):
        self._test_all_to_all_helper(lambda t: t.clone().xpu())

    @skip_if_not_multixpu
    def test_alltoall_basics_multi_xpu(self):
        self._test_all_to_all_helper(lambda t: t.clone().xpu("xpu:{}".format(self.rank)))

    def _test_reduce_scatter_base_basics(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupCCL(store, self.rank, self.world_size)

        def reduce_scatter_base(output_t, input_t):
            work = pg._reduce_scatter_base(output_t, input_t)
            work.wait()
       
    def test_reduce_scatter_base_basics(self):
        self._test_reduce_scatter_base_basics(lambda t: t.clone())

    @skip_if_no_xpu
    def test_reduce_scatter_base_basics_xpu(self):
        self._test_reduce_scatter_base_basics(lambda t: t.clone().xpu())

    @skip_if_not_multixpu
    def test_reduce_scatter_base_basics_multi_xpu(self):
        self._test_reduce_scatter_base_basics(lambda t: t.clone().xpu("xpu:{}".format(self.rank)))

    def _test_reduce_scatter_base_ops(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupCCL(store, self.rank, self.world_size)

        def reduce_scatter_base(output_t, input_t):
            work = pg._reduce_scatter_base(output_t, input_t)
            work.wait()

        # reduce_scatter_base is GPU number agnostic.
        # Each rank contribute one tensor regardless of GPU counts
        output_t = fn(torch.empty([1]))
        tensor = fn(torch.arange(self.world_size, dtype=output_t.dtype))

        reduce_scatter_base(output_t, tensor)

        # Verification
        self.assertEqual(output_t[0], self.rank * self.world_size)

    def test_reduce_scatter_base(self):
        self._test_reduce_scatter_base_ops(lambda t: t.clone())

    @skip_if_no_xpu
    def test_reduce_scatter_base_xpu(self):
        self._test_reduce_scatter_base_ops(lambda t: t.clone().xpu())

    @skip_if_not_multixpu
    def test_reduce_scatter_base_multi_xpu(self):
        self._test_reduce_scatter_base_ops(lambda t: t.clone().xpu("xpu:{}".format(self.rank)))
    
    @skip_if_no_xpu
    def test_coalescing_manager(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            "ccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        device = torch.device("xpu:{}".format(self.rank))
        num_colls = 2
        size_per_coll = 8
        small_tensors = [
            torch.ones(size_per_coll, device=device) for _ in range(num_colls)
        ]
        
        with c10d._coalescing_manager(device=device):
            for i in range(num_colls):
                c10d.all_reduce(small_tensors[i])
        
        big_tensor = torch.ones(num_colls * size_per_coll, device=device)
        c10d.all_reduce(big_tensor)
        
        for i in range(num_colls):
            self.assertEqual(
                small_tensors[i], 
                big_tensor[i * size_per_coll : (i + 1) * size_per_coll]
            )
            
    @skip_if_no_xpu
    def test_allgather_into_tensor_coalesced(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            "ccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        device = torch.device("xpu:{}".format(self.rank))
        
        input_tensors = torch.ones(2, 2, device=device)
        output_tensors = [torch.zeros(2, 2, device=device) for _ in range(self.world_size)]
        
        with c10d._coalescing_manager(device=device):
            for i in range(self.world_size):
                c10d.all_gather_into_tensor(output_tensors[i], input_tensors[i])
        
        self.assertEqual(output_tensors[self.rank], input_tensors)

    @skip_if_no_xpu
    def test_reduce_scatter_tensor_coalesced(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            "ccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        device = torch.device("xpu:{}".format(self.rank))
        
        input_tensors = [torch.ones(2, 2, device=device) for _ in range(self.world_size)]
        output_tensors = torch.zeros(2, 2, device=device)
        
        with c10d._coalescing_manager(device=device):
            for i in range(self.world_size):
                c10d.reduce_scatter_tensor(output_tensors[i], input_tensors[i])
        
        self.assertEqual(output_tensors, input_tensors[self.rank] * self.world_size)
        
if __name__ == '__main__':
    run_tests()
