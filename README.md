# torch-ccl

This repository holds PyTorch bindings maintained by Intel for the Intel® oneAPI Collective Communications Library (oneCCL).


# Introduction

[PyTorch](https://github.com/pytorch/pytorch) is an open-source machine learning framework.

[Intel® oneCCL](https://github.com/oneapi-src/oneCCL) (collective commnications library) is a library for efficient distributed deep learning training implementing such collectives like allreduce, allgather, alltoall. For more information on oneCCL, please refer to the [oneCCL documentation](https://oneapi-src.github.io/oneCCL).

`torch-ccl` module implements PyTorch C10D ProcessGroup API and can be dynamically loaded as external ProcessGroup.


# Requirements

PyTorch (1.5.0 or higher).

Intel® oneAPI Collective Communications Library (2021.1-beta05 or higher).


# Installation

To install `torch-ccl`:

1. Install PyTorch.

2. Install the `torch-ccl`.

```
$ python setup.py install
```

3. Source the oneCCL environment.

```
$ torch_ccl_path=$(python -c "import torch; import torch_ccl; import os;  print(os.path.abspath(os.path.dirname(torch_ccl.__file__)))")
$ source $torch_ccl_path/ccl/env/setvars.sh
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

```
$ torch_ccl_path=$(python -c "import torch; import torch_ccl; import os;  print(os.path.abspath(os.path.dirname(torch_ccl.__file__)))")
$ source $torch_ccl_path/ccl/env/setvars.sh
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
$ torch_ccl_path=$(python -c "import torch; import torch_ccl; import os;  print(os.path.abspath(os.path.dirname(torch_ccl.__file__)))")
$ source $torch_ccl_path/ccl/env/setvars.sh
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
