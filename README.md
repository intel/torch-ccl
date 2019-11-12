# pytorch-ccl

This repository holds PyTorch bindings maintained by Intel for the Intel® oneAPI Collective Communications Library (oneCCL).


# Introduction

PyTorch is an open-source machine learning [framework](https://github.com/pytorch/pytorch).

oneCCL is a library for efficient distributed deep learning training that implements collectives such as allreduce, allgather, and bcast. For more information on oneCCL, please refer to the [oneCCL documentation](https://github.com/intel/oneccl).

The `pytorch-ccl` module implements the PyTorch C10D ProcessGroup API and can be dynamically loaded as an external ProcessGroup.


# Requirements

PyTorch 1.3.x or newer (TODO - specify version with support of dynamic loading of external ProcessGroup)

Intel® oneAPI Collective Communications Library


# Installation

To install `pytoch-ccl`:

1. Install PyTorch.

2. Install oneCCL (please refer to [this page](https://github.com/intel/oneccl)).

3. Source the oneCCL environment.

```
$ source <ccl_install_path>/env/vars.sh
```

4. Install the `pytorch-ccl` pip package.

```
$ pip setup.py install 
```


# Usage

example.py

```python

import torch.nn.parallel
import torch.distributed as dist

...

backend = 'ccl'
dist.init_process_group(backend, ...)
model = torch.nn.parallel.DistributedDataParallel(model, ...)

...
```

```
$ source <ccl_install_path>/env/vars.sh
$ mpirun -n <N> -ppn <PPN> -f <hostfile> python example.py

```


# License
[BSD License](https://github.com/intel/pytorch-ccl/blob/master/LICENSE)
