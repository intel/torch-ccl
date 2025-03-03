# Intel® oneCCL Bindings for PyTorch (formerly known as torch_ccl)

This repository holds PyTorch bindings maintained by Intel® for the Intel® oneAPI Collective Communications Library (oneCCL).

## Introduction

[PyTorch](https://github.com/pytorch/pytorch) is an open-source machine learning framework.

[Intel® oneCCL](https://github.com/oneapi-src/oneCCL) (collective communications library) is a library for efficient distributed deep learning training, implementing collectives like `allreduce`, `allgather`, `alltoall`. For more information on oneCCL, please refer to the [oneCCL documentation](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/oneccl/source/).

`oneccl_bindings_for_pytorch` module implements PyTorch C10D ProcessGroup API and can be dynamically loaded as external ProcessGroup and only works on Linux platform now.

## Capability

The table below shows which functions are available for use with CPU / Intel dGPU tensors.

|                  | CPU   | GPU   |
| :--------------- | :---: | :---: |
| `send`           | ×     | √     |
| `recv`           | ×     | √     |
| `broadcast`      | √     | √     |
| `all_reduce`     | √     | √     |
| `reduce`         | √     | √     |
| `all_gather`     | √     | √     |
| `gather`         | √     | √     |
| `scatter`        | √     | √     |
| `reduce_scatter` | √     | √     |
| `all_to_all`     | √     | √     |
| `barrier`        | √     | √     |


## PyTorch API Align

We recommend using Anaconda as Python package management system. The followings are the corresponding branches (tags) of `oneccl_bindings_for_pytorch` and supported PyTorch.

   | `torch`                                                         | `oneccl_bindings_for_pytorch`                                             |
   | :-------------------------------------------------------------: | :-----------------------------------------------------------------------: |
   | `master`                                                        |  `master`                                                                 |
   | [v2.6.0](https://github.com/pytorch/pytorch/tree/v2.6.0)        |  [ccl_torch2.6.0](https://github.com/intel/torch-ccl/tree/ccl_torch2.6.0+xpu)   |
   | [v2.5.0](https://github.com/pytorch/pytorch/tree/v2.5.0)        |  [ccl_torch2.5.0](https://github.com/intel/torch-ccl/tree/ccl_torch2.5.0+xpu)   |
   | [v2.3.1](https://github.com/pytorch/pytorch/tree/v2.3.1)        |  [ccl_torch2.3.100](https://github.com/intel/torch-ccl/tree/ccl_torch2.3.100+xpu)   |
   | [v2.1.0](https://github.com/pytorch/pytorch/tree/v2.1.0)        |  [ccl_torch2.1.400](https://github.com/intel/torch-ccl/tree/ccl_torch2.1.400+xpu)   |
   | [v2.1.0](https://github.com/pytorch/pytorch/tree/v2.1.0)        |  [ccl_torch2.1.300](https://github.com/intel/torch-ccl/tree/ccl_torch2.1.300+xpu)   |
   | [v2.1.0](https://github.com/pytorch/pytorch/tree/v2.1.0)        |  [ccl_torch2.1.200](https://github.com/intel/torch-ccl/tree/ccl_torch2.1.200+xpu)   |
   | [v2.1.0](https://github.com/pytorch/pytorch/tree/v2.1.0)        |  [ccl_torch2.1.100](https://github.com/intel/torch-ccl/tree/ccl_torch2.1.100+xpu)   |
   | [v2.0.1](https://github.com/pytorch/pytorch/tree/v2.0.1)        |  [ccl_torch2.0.100](https://github.com/intel/torch-ccl/tree/ccl_torch2.0.100)   |
   | [v1.13](https://github.com/pytorch/pytorch/tree/v1.13)          |  [ccl_torch1.13](https://github.com/intel/torch-ccl/tree/ccl_torch1.13)   |
   | [v1.12.1](https://github.com/pytorch/pytorch/tree/v1.12.1)      |  [ccl_torch1.12.100](https://github.com/intel/torch-ccl/tree/ccl_torch1.12.100)   |
   | [v1.12.0](https://github.com/pytorch/pytorch/tree/v1.12.0)      |  [ccl_torch1.12](https://github.com/intel/torch-ccl/tree/ccl_torch1.12)   |
   | [v1.11.0](https://github.com/pytorch/pytorch/tree/v1.11.0)      |  [ccl_torch1.11](https://github.com/intel/torch-ccl/tree/ccl_torch1.11)   |
   | [v1.10.0](https://github.com/pytorch/pytorch/tree/v1.10.0)      |  [ccl_torch1.10](https://github.com/intel/torch-ccl/tree/ccl_torch1.10)   |
   | [v1.9.0](https://github.com/pytorch/pytorch/tree/v1.9.0)        |  [ccl_torch1.9](https://github.com/intel/torch-ccl/tree/ccl_torch1.9)     |
   | [v1.8.1](https://github.com/pytorch/pytorch/tree/v1.8.1)        |  [ccl_torch1.8](https://github.com/intel/torch-ccl/tree/ccl_torch1.8)     |
   | [v1.7.1](https://github.com/pytorch/pytorch/tree/v1.7.1)        |  [ccl_torch1.7](https://github.com/intel/torch-ccl/tree/ccl_torch1.7)     |
   | [v1.6.0](https://github.com/pytorch/pytorch/tree/v1.6.0)        |  [ccl_torch1.6](https://github.com/intel/torch-ccl/tree/ccl_torch1.6)     |
   | [v1.5-rc3](https://github.com/pytorch/pytorch/tree/v1.5.0-rc3)  |  [beta09](https://github.com/intel/torch-ccl/tree/beta09)                 |

The usage details can be found in the README of corresponding branch.

## Requirements

- Python 3.8 or later and a C++17 compiler

- PyTorch v2.6.0

## Build Option List

The following build options are supported in Intel® oneCCL Bindings for PyTorch*.

| Build Option                        | Default Value  | Description                                                                                         |
| :---------------------------------- | :------------- | :-------------------------------------------------------------------------------------------------- |
| COMPUTE_BACKEND                     | N/A            | Set oneCCL `COMPUTE_BACKEND`, set to `dpcpp`  and use DPC++ compiler to enable support for Intel XPU |
| USE_SYSTEM_ONECCL                   | OFF            | Use oneCCL library in system                                                                        |
| CCL_PACKAGE_NAME                    | oneccl-bind-pt | Set wheel name                                                                                      |
| ONECCL_BINDINGS_FOR_PYTORCH_BACKEND | cpu            | Set backend                                                                                         |
| CCL_SHA_VERSION                     | False          | Add git head sha version into wheel name                                                            |

## Launch Option List

The following launch options are supported in Intel® oneCCL Bindings for PyTorch*.

| Launch Option                             | Default Value | Description                                                           |
| :--------------------------------------- | :------------ | :-------------------------------------------------------------------- |
| ONECCL_BINDINGS_FOR_PYTORCH_ENV_VERBOSE  | 0             | Set verbose level in oneccl_bindings_for_pytorch                      |
| ONECCL_BINDINGS_FOR_PYTORCH_ENV_WAIT_GDB | 0             | Set 1 to force the oneccl_bindings_for_pytorch wait for GDB attaching |
| TORCH_LLM_ALLREDUCE                      | 0             | Set 1 to enable this prototype feature for better scale-up performance. This is a prototype feature to provide better scale-up performance by enabling optimized collective algorithms in oneCCL and asynchronous execution in torch-ccl. This feature requires XeLink enabled for cross-cards communication.|
| CCL_BLOCKING_WAIT                        | 0             | Set 1 to enable this prototype feature, which is to control whether collectives execution on XPU is host blocking or non-blocking. |
| CCL_SAME_STREAM                          | 0             | Set 1 to enable this prototype feature, which is to allow using a computation stream as communication stream to minimize overhead for streams synchronization. |

## Installation

### Install from Source

1. clone the `oneccl_bindings_for_pytorch`.

   ```bash
   git clone https://github.com/intel/torch-ccl.git && cd torch-ccl
   git checkout ccl_torch2.6.0+xpu
   git submodule sync
   git submodule update --init --recursive
   ```

2. Install `oneccl_bindings_for_pytorch`

   ```bash
   # for CPU Backend Only
   python setup.py install
   # for XPU Backend: use DPC++ Compiler to enable support for Intel XPU
   # build with oneCCL from third party
   COMPUTE_BACKEND=dpcpp python setup.py install
   # build with oneCCL from basekit
   export INTELONEAPIROOT=${HOME}/intel/oneapi
   USE_SYSTEM_ONECCL=ON COMPUTE_BACKEND=dpcpp python setup.py install
   ```

### Install Prebuilt Wheel

Wheel files are available for the following Python versions. Please always use the latest release to get started.

| Extension Version | Python 3.6 | Python 3.7 | Python 3.8 | Python 3.9 | Python 3.10 | Python 3.11 | Python 3.12 | Python 3.13 |
| :---------------: | :--------: | :--------: | :--------: | :--------: | :---------: | :---------: | :---------: | :---------: |
| 2.6.0             |            |            |            | √          | √           | √           | √           | √           |
| 2.5.1             |            |            |            | √          | √           | √           | √           |             |
| 2.3.100           |            |            | √          | √          | √           | √           |             |             |
| 2.1.400           |            |            | √          | √          | √           | √           |             |             |
| 2.1.300           |            |            | √          | √          | √           | √           |             |             |
| 2.1.200           |            |            | √          | √          | √           | √           |             |             |
| 2.1.100           |            |            | √          | √          | √           | √           |             |             |
| 2.0.100           |            |            | √          | √          | √           | √           |             |             |
| 1.13              |            | √          | √          | √          | √           |             |             |             |
| 1.12.100          |            | √          | √          | √          | √           |             |             |             |
| 1.12.0            |            | √          | √          | √          | √           |             |             |             |
| 1.11.0            |            | √          | √          | √          | √           |             |             |             |
| 1.10.0            | √          | √          | √          | √          |             |             |             |             |

```bash
python -m pip install oneccl_bind_pt==2.6.0 --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

**Note:** Please set proxy or update URL address to https://pytorch-extension.intel.com/release-whl/stable/xpu/cn/ if you meet connection issue.

### Runtime Dynamic Linking

- If oneccl_bindings_for_pytorch is built without oneCCL and use oneCCL in system, dynamic link oneCCl from oneAPI basekit (recommended usage):

```bash
source $basekit_root/ccl/latest/env/vars.sh
```

Note: Make sure you have installed [basekit](https://www.intel.com/content/www/us/en/developer/tools/oneapi/toolkits.html#base-kit) when using Intel® oneCCL Bindings for Pytorch\* on Intel® GPUs.

- If oneccl_bindings_for_pytorch is built with oneCCL from third party or installed from prebuilt wheel:
Dynamic link oneCCL and Intel MPI libraries:

```bash
source $(python -c "import oneccl_bindings_for_pytorch as torch_ccl;print(torch_ccl.cwd)")/env/setvars.sh
```

Dynamic link oneCCL only (not including Intel MPI):

```bash
source $(python -c "import oneccl_bindings_for_pytorch as torch_ccl;print(torch_ccl.cwd)")/env/vars.sh
```

## Usage

**Note:** Please `import torch` and `import intel_extension_for_pytorch`, prior to `import oneccl_bindings_for_pytorch`.

example.py

```python

import torch
import intel_extension_for_pytorch
import oneccl_bindings_for_pytorch
import torch.nn.parallel
import torch.distributed as dist

...

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'
os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))

backend = 'ccl'
dist.init_process_group(backend, ...)
my_rank = dist.get_rank()
my_size = dist.get_world_size()
print("my rank = %d  my size = %d" % (my_rank, my_size))

...

model = torch.nn.parallel.DistributedDataParallel(model, ...)

...
```

(oneccl_bindings_for_pytorch is built without oneCCL, use oneCCL and MPI(if needed) in system)

```bash
source $basekit_root/ccl/latest/env/vars.sh
source $basekit_root/mpi/latest/env/vars.sh

mpirun -n <N> -ppn <PPN> -f <hostfile> python example.py
```

## Performance Debugging

For debugging performance of communication primitives PyTorch's [Autograd profiler](https://pytorch.org/docs/stable/autograd.html#profiler)
can be used to inspect time spent inside oneCCL calls.

Example:

profiling.py

```python

import torch.nn.parallel
import torch.distributed as dist
import oneccl_bindings_for_pytorch
import os

os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'
os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))

backend = 'ccl'
dist.init_process_group(backend)
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

```bash
mpirun -n 2 -l python profiling.py
```

```bash
[0] my rank = 0  my size = 2
[0] -----------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  --------------------
[0]                                                  Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls          Input Shapes
[0] -----------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  --------------------
[0]                oneccl_bindings_for_pytorch::allreduce        91.41%     297.900ms        91.41%     297.900ms      29.790ms            10              [[2, 2]]
[0]     oneccl_bindings_for_pytorch::wait::cpu::allreduce         8.24%      26.845ms         8.24%      26.845ms       2.684ms            10      [[2, 2], [2, 2]]
[0]     oneccl_bindings_for_pytorch::wait::cpu::allreduce         0.30%     973.651us         0.30%     973.651us      97.365us            10      [[4, 4], [4, 4]]
[0]                oneccl_bindings_for_pytorch::allreduce         0.06%     190.254us         0.06%     190.254us      19.025us            10              [[4, 4]]
[0] -----------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  --------------------
[0] Self CPU time total: 325.909ms
[0]
[1] my rank = 1  my size = 2
[1] -----------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  --------------------
[1]                                                  Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls          Input Shapes
[1] -----------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  --------------------
[1]                oneccl_bindings_for_pytorch::allreduce        96.03%     318.551ms        96.03%     318.551ms      31.855ms            10              [[2, 2]]
[1]     oneccl_bindings_for_pytorch::wait::cpu::allreduce         3.62%      12.019ms         3.62%      12.019ms       1.202ms            10      [[2, 2], [2, 2]]
[1]                oneccl_bindings_for_pytorch::allreduce         0.33%       1.082ms         0.33%       1.082ms     108.157us            10              [[4, 4]]
[1]     oneccl_bindings_for_pytorch::wait::cpu::allreduce         0.02%      56.505us         0.02%      56.505us       5.651us            10      [[4, 4], [4, 4]]
[1] -----------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  --------------------
[1] Self CPU time total: 331.708ms
[1]

```

## License

[BSD License](https://github.com/intel/torch-ccl/blob/master/LICENSE)
