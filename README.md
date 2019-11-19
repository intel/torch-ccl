# pytorch-ccl

This repository holds Intel-maintained PyTorch bindings for Intel oneCCL.


# Introduction

[PyTorch](https://github.com/pytorch/pytorch) is an open-source machine learning framework.

[Intel oneCCL](https://github.com/intel/oneccl)(collective commnication library) is a library for efficient distributed deep learning training implementing such collectives like allreduce, allgather, bcast. For more information on oneCCL, please refer to the [oneCCL documentation](https://intel.github.io/oneccl).

`pytorch-ccl` module implements PyTorch C10D ProcessGroup API and can be dynamically loaded as external ProcessGroup.


# Requirements

PyTorch 1.3.x or newer (TODO - specify version with support of dynamic loading of external ProcessGroup)

Intel oneCCL


# Installation

To install `pytoch-ccl`:

1. Install PyTorch.

2. Install Intel oneCCL (please refer to [this page](https://intel.github.io/oneccl/installation.html)).

3. Source Intel oneCCL environment.

```
$ source <ccl_install_path>/env/setvars.sh
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
$ source <ccl_install_path>/env/setvars.sh
$ mpirun -n <N> -ppn <PPN> -f <hostfile> python example.py

```

# License
[BSD License](https://github.com/intel/pytorch-ccl/blob/master/LICENSE)
