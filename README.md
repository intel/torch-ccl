# torch-ccl

This repository holds PyTorch bindings maintained by Intel for the Intel® oneAPI Collective Communications Library (oneCCL).


# Introduction

[PyTorch](https://github.com/pytorch/pytorch) is an open-source machine learning framework.

[Intel oneCCL](https://github.com/oneapi-src/oneCCL) (collective commnications library) is a library for efficient distributed deep learning training implementing such collectives like allreduce, allgather, bcast. For more information on oneCCL, please refer to the [oneCCL documentation](https://oneapi-src.github.io/oneCCL).

`torch-ccl` module implements PyTorch C10D ProcessGroup API and can be dynamically loaded as external ProcessGroup.


# Requirements

PyTorch (1.6.0 or newer).

Intel® oneAPI Collective Communications Library (2021.1-beta05 or newer).


# Installation

To install `torch-ccl`:

1. Install PyTorch.

2. Install Intel oneCCL (please refer to [this page](https://oneapi-src.github.io/oneCCL/installation.html)).

3. Source the oneCCL environment.

```
$ source <ccl_install_path>/env/setvars.sh
```

4. Install the `torch-ccl` pip package.

```
$ pip setup.py install 
```


# Usage

example.py

```python

import torch.nn.parallel
import torch.distributed as dist
import torch_ccl

...

backend = 'ccl'
dist.init_process_group(backend, ...)
model = torch.nn.parallel.DistributedDataParallel(model, ...)

...
```

```
$ source <ccl_install_path>/env/setvars.sh
$ mpirun -n <N> -ppn <PPN> -f <hostfile> python example.py

```


# License
[BSD License](https://github.com/oneapi-src/oneCCL/blob/master/LICENSE)
