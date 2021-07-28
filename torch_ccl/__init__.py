import os
import sys
import warnings
import torch

cwd = os.path.dirname(os.path.abspath(__file__))

FI_PROVIDER_PATH = os.path.join(cwd, "lib/prov")
os.environ['FI_PROVIDER_PATH'] = FI_PROVIDER_PATH
if not os.path.exists(os.path.join(cwd, "version.py")):
    raise RuntimeError("torch_ccl is not installed!")
if not os.path.exists(os.path.join(cwd, "_C.py"))
    raise RuntimeError("Seems you are importing torch_ccl under it's root path, please change path and try again!")

from .version import __version__, git_version
from . import _C as ccl_lib

if hasattr(torch, 'xpu'):
    if torch.xpu.is_available():
        # try:
        # load the CCL/XPU library
        import ctypes
        my_c_library = ctypes.cdll.LoadLibrary(os.path.join(cwd, "lib/libtorch_ccl_xpu.so"))
        # except OSError:
        #     print("Cannot load xpu CCL. CCL doesn't work for XPU device")

__all__ = []
__all__ += [name for name in dir(ccl_lib)
            if name[0] != '_' and
            not name.endswith('Base')]


def is_available(tensors):
    devices = set()
    for tensor in tensors:
        if not tensor.is_contiguous():
            return False
        device = tensor.get_device()
        if device in devices:
            return False
        devices.add(device)

    return True

