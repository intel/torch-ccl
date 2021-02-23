# torch-ccl

This repository holds PyTorch bindings maintained by Intel for the Intel® oneAPI Collective Communications Library (oneCCL).


# Introduction

[PyTorch](https://github.com/pytorch/pytorch) is an open-source machine learning framework.

[Intel® oneCCL](https://github.com/oneapi-src/oneCCL) (collective commnications library) is a library for efficient distributed deep learning training implementing such collectives like allreduce, allgather, alltoall. For more information on oneCCL, please refer to the [oneCCL documentation](https://spec.oneapi.com/versions/latest/elements/oneCCL/source/index.html) and [oneCCL specification](https://spec.oneapi.com/versions/latest/elements/oneCCL/source/index.html).

`torch-ccl` module implements PyTorch C10D ProcessGroup API and can be dynamically loaded as external ProcessGroup and only works on Linux platform now.

# Pytorch API Align
We recommend Anaconda as Python package management system. The following is the corresponding branchs (tags) of torch-ccl and supported Pytorch.

   | ``torch`` | ``torch-ccl`` |  
   | :-----:| :---: |  
   |  ``master`` |  ``master``  |
   | [v1.7.1](https://github.com/pytorch/pytorch/tree/v1.7.1) |  [ccl_torch1.7](https://github.com/intel/torch-ccl/tree/ccl_torch1.7)   | 
   | [v1.6.0](https://github.com/pytorch/pytorch/tree/v1.6.0) |  [ccl_torch1.6](https://github.com/intel/torch-ccl/tree/ccl_torch1.6)   | 
   | [v1.5-rc3](https://github.com/pytorch/pytorch/tree/v1.5.0-rc3) |   [beta09](https://github.com/intel/torch-ccl/tree/beta09)   |

The usage details can be found in the README of corresponding branch. The following part is about the usage of v1.7 tag. if you want to use other version of torch-ccl please checkout to that branch(tag). For pytorch-1.5.0-rc3, the [#PR28068](https://github.com/pytorch/pytorch/pull/28068) and [#PR32361](https://github.com/pytorch/pytorch/pull/32361) are need to dynamicall register external ProcessGroup and enable ``alltoall`` collective communication primitive. The patch file about these two PRs is in ``patches`` directory and you can use it directly. 

# Requirements

Python 3.6 or later and a C++14 compiler

pytorch-v1.7.1.

# Installation

To install `torch-ccl`:

1. clone the `torch-ccl`.

```bash
   git clone https://github.com/intel/torch-ccl.git && cd torch-ccl 
   git submodule sync 
   git submodule update --init --recursive
```
2. Install torch-ccl

```bash
   python setup.py install
```


# Usage

example.py

```python

import torch.nn.parallel
import torch.distributed as dist
import torch_ccl

...

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'
os.environ['RANK'] = os.environ.get('PMI_RANK', -1)
os.environ['WORLD_SIZE'] = os.environ.get('PMI_SIZE', -1)

backend = 'ccl'
dist.init_process_group(backend, ...)
my_rank = dist.get_rank()
my_size = dist.get_world_size()
print("my rank = %d  my size = %d" % (my_rank, my_size))

...

model = torch.nn.parallel.DistributedDataParallel(model, ...)

...
```
(torch_ccl is installed along with the MPI toolset.)
```

$ source <torch_ccl_path>/env/setvars.sh

eg:
  $ torch_ccl_path=$(python -c "import torch; import torch_ccl; import os;  print(os.path.abspath(os.path.dirname(torch_ccl.__file__)))")
  $ source $torch_ccl_path/env/setvars.sh

$ mpirun -n <N> -ppn <PPN> -f <hostfile> python example.py
```


# Performance Debugging

For debugging performance of communication primitives PyTorch's [Autograd profiler](https://pytorch.org/docs/stable/autograd.html#profiler)
can be used to inspect time spent inside oneCCL calls.

Example:

profiling.py

```python

import torch.nn.parallel
import torch.distributed as dist
import torch_ccl
mport os

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'
os.environ['RANK'] = os.environ.get('PMI_RANK', -1)
os.environ['WORLD_SIZE'] = os.environ.get('PMI_SIZE', -1)

backend = 'ccl'
dist.init_process_group(backend, ...)
my_rank = dist.get_rank()
my_size = dist.get_world_size()
print("my rank = %d  my size = %d" % (my_rank, my_size))

x = torch.ones([2, 2])
y = torch.ones([4, 4])
with torch.autograd.profiler.profile(record_shapes=True) as prof:
    for _ in range(10):
        dist.all_reduce(x)
        dist.all_reduce(y)
dist.barrier()
print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total"))

```

```
$ mpirun -n 2 -l python profiling.py
```

```
[0] rank = 0, size = 2
[0] ------------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
[0] Name                            Self CPU total %  Self CPU total   CPU total %      CPU total        CPU time avg     Number of Calls  Input Shapes
[0] ------------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
[0] pg::allreduce                   37.70%           61.935us         37.70%           61.935us         6.194us          10               [[2, 2]]
[0] pg::allreduce                   23.40%           38.438us         23.40%           38.438us         3.844us          10               [[4, 4]]
[0] pg::wait::allreduce::sz:16      19.64%           32.258us         19.64%           32.258us         3.226us          10               []
[0] pg::wait::allreduce::sz:4       19.26%           31.634us         19.26%           31.634us         3.163us          10               []
[0] ------------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
[0] Self CPU time total: 164.265us
[0]
[1] rank = 1, size = 2
[1] ------------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
[1] Name                            Self CPU total %  Self CPU total   CPU total %      CPU total        CPU time avg     Number of Calls  Input Shapes
[1] ------------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
[1] pg::allreduce                   50.27%           62.730us         50.27%           62.730us         6.273us          10               [[2, 2]]
[1] pg::allreduce                   28.96%           36.133us         28.96%           36.133us         3.613us          10               [[4, 4]]
[1] pg::wait::allreduce::sz:4       13.83%           17.254us         13.83%           17.254us         1.725us          10               []
[1] pg::wait::allreduce::sz:16      6.95%            8.672us          6.95%            8.672us          0.867us          10               []
[1] ------------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
[1] Self CPU time total: 124.789us
[1]

```


# License
[BSD License](https://github.com/intel/torch-ccl/blob/master/LICENSE)
