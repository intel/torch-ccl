import torch
import torch_ccl
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.common_distributed import MultiProcessTestCase, \
     simple_sparse_reduce_tests

import torch.distributed as c10d

import math
from functools import reduce
import operator

cpu_device = torch.device("cpu")

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
        self._fork_processes()

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

        x = torch.tensor([self.rank + 1.0])
        work = pg.broadcast(x, root=0)
        work.wait()
        self.assertEqual(torch.tensor([1.0]), x)

    def test_broadcast_basics(self):
        self._test_broadcast_basics(lambda t: t.clone())

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

    def test_reduce_basics(self):
        self._test_reduce_basics(lambda t: t.clone())
     
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

    
    # alltoall_base
    def _test_alltoall_base_equal_split_helper(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupCCL(store, self.rank, self.world_size)
        group = list(range(0, self.world_size))
        rank = self.rank
        size = len(group)
        in_tensor = fn(torch.ones([size, size]) * rank)
        expected_tensor = torch.cat([torch.ones([1, size]) * i for i in group])
        out_tensor = torch.ones([size, size]) * -1
        in_splits = []
        out_splits = []
        work = pg.alltoall_base(out_tensor, in_tensor, out_splits, in_splits)
        work.wait()
        self.assertEqual(out_tensor, expected_tensor)
   
    def _test_alltoall_base_unequal_split_helper(self, fn):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupCCL(store, self.rank, self.world_size)
        group = list(range(0, self.world_size))
        rank = self.rank
        size = len(group)
        in_splits = [i + 1 for i in group]
        out_splits = [rank + 1 for _ in group]
        in_tensor = torch.ones([sum(in_splits), size]) * rank
        out_tensor = torch.ones([(rank + 1) * size, size])
        expected_tensor = torch.cat([torch.ones([rank + 1, size]) * i for i in group])
        work = pg.alltoall_base(
             out_tensor, in_tensor, out_splits, in_splits)
        work.wait()
        self.assertEqual(out_tensor, expected_tensor)
        
    def test_allotall_equal_split_basics(self):
        self._test_alltoall_base_equal_split_helper(lambda t: t.clone())

    def test_allotall_unequal_split_basics(self):
        self._test_alltoall_base_unequal_split_helper(lambda t: t.clone())

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
        work = pg.alltoall(out_tensors, in_tensors)
        work.wait()
        for t1, t2 in zip(out_tensors, expected_tensors):
            self.assertEqual(t1, t2)
        
    def test_alltoall_basics(self):
        self._test_all_to_all_helper(lambda t: t.clone())

if __name__ == '__main__':
    run_tests()
