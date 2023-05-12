import sys
import os
import torch

try:
    import intel_extension_for_pytorch
    xpu_is_available = torch.xpu.is_available() if hasattr(torch, 'xpu') else False
except ImportError:
    # ignore the ipex
    xpu_is_available = False
    pass

import oneccl_bindings_for_pytorch
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.common_distributed import MultiProcessTestCase

import torch.distributed as dist

class ProcessGroupCCLTest(MultiProcessTestCase):

    def setUp(self):
        super(ProcessGroupCCLTest, self).setUp()
        self._spawn_processes()
      
    @property
    def world_size(self):
        return 6

    def _build_tensor(self, size, value=None, dtype=torch.float, device=None):
        if value is None:
            value = size
        if device is None:
            return torch.empty(size, size, size, dtype=dtype).fill_(value)
        else:
            return torch.empty(size, size, size, dtype=dtype).fill_(value).to(device)

    def _test_send_recv_withincard(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "ccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            )
        device = "xpu:{}".format(self.rank)

        # WA: allreduce
        # Ensure the process group has been fully initialized
        data = torch.zeros(1).to(device)
        dist.all_reduce(data)

        torch.xpu.set_device(device)
        tensor = self._build_tensor(self.rank + 1, device=device)

        # rank0 -> rank1
        src = 0
        dst = 1
        if self.rank == src:
            # Send
            dist.send(tensor, dst)
        elif self.rank == dst:
            # Recv
            expected_tensor = self._build_tensor(src + 1)
            output_tensor = self._build_tensor(
                src + 1, value=-1, device=device
            )
            dist.recv(output_tensor, src)
            self.assertEqual(output_tensor, expected_tensor)

    def test_send_recv_withincard(self):
        self._test_send_recv_withincard()

    def _test_send_recv_3rank(self):
        # cross-cards p2p: rank1 -> rank3 -> rank5
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "ccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            )
        device = "xpu:{}".format(self.rank)
        
        # WA: allreduce
        # Ensure the process group has been fully initialized
        data = torch.zeros(1).to(device)
        dist.all_reduce(data)

        torch.xpu.set_device(device)
        tensor = self._build_tensor(self.rank + 1, device=device)

        if self.rank == 1:
            dist.send(tensor, 3)
        if self.rank == 3:
            expected_tensor1 = self._build_tensor(1 + 1)
            output_tensor1 = self._build_tensor(
                1 + 1, value=-1, device=device
            )
            dist.recv(output_tensor1, 1)
            self.assertEqual(output_tensor1, expected_tensor1)

            # rank3 -> rank5
            dist.send(tensor, 5)
        if self.rank == 5:
            expected_tensor2 = self._build_tensor(3 + 1)
            output_tensor2 = self._build_tensor(
                3 + 1, value=-1, device=device
            )
            dist.recv(output_tensor2, 3)
            self.assertEqual(output_tensor2, expected_tensor2)

    def test_send_recv_3rank(self):
        self._test_send_recv_3rank()

    def _test_send_recv_crosscard(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "ccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            )
        device = "xpu:{}".format(self.rank)

        # WA: allreduce
        # Ensure the process group has been fully initialized
        data = torch.zeros(1).to(device)
        dist.all_reduce(data)

        torch.xpu.set_device(device)
        tensor = self._build_tensor(self.rank + 1, device=device)

        for src in range(0, self.world_size):
            if src == self.rank:
                # Send mode
                for dst in range(0, self.world_size):
                    if dst == self.rank:
                        continue
                    dist.send(tensor, dst)
            else:
                # Recv mode
                expected_tensor = self._build_tensor(src + 1)
                output_tensor = self._build_tensor(
                    src + 1, value=-1, device=device
                )
                dist.recv(output_tensor, src)
                self.assertEqual(output_tensor, expected_tensor)

    def test_send_recv_crosscard(self):
        self._test_send_recv_crosscard()

    def _test_send_recv_with_tag(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "ccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            )
        device = "xpu:{}".format(self.rank)

        # WA: allreduce
        # Ensure the process group has been fully initialized
        data = torch.zeros(1).to(device)
        dist.all_reduce(data)

        torch.xpu.set_device(device)
        tensor = self._build_tensor(10, value=self.rank, device=device)

        for dst in range(0, self.world_size):
            if dst == self.rank:
                # Recv mode
                for src in range(0, self.world_size):
                    if src == self.rank:
                        continue
                    output_tensor = self._build_tensor(10, value=-1, device=device)
                    dist.recv(output_tensor, src, tag=src)
                    self.assertTrue(output_tensor.eq(src).all())
            else:
                # Send mode
                dist.send(tensor, dst, tag=self.rank)

    def test_send_recv_with_tag(self):
        self._test_send_recv_with_tag()

if __name__ == '__main__':
    run_tests()
