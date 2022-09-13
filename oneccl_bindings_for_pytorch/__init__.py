import os
import sys
import warnings
import torch


cwd = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(cwd, "version.py")):
    raise RuntimeError("oneccl_bindings_for_pytorch is not installed!")


def set_env_default(env, key, value):
    new_value = env.get(key, value)
    env[key] = new_value


if os.environ.get("CCL_ROOT") is None:
    # set the default oneCCL and MPI library path
    set_env_default(os.environ, 'CCL_ROOT', cwd)

    FI_PROVIDER_PATH = os.path.join(cwd, "lib/prov")
    set_env_default(os.environ, 'FI_PROVIDER_PATH', FI_PROVIDER_PATH)


from .version import __version__, git_version
from . import _C as ccl_lib

if hasattr(torch, 'xpu'):
    try:
        # load the CCL/XPU library
        import ctypes
        my_c_library = ctypes.cdll.LoadLibrary(os.path.join(cwd, "lib/liboneccl_bindings_for_pytorch_xpu.so"))
    except OSError:
        print("Warning: Cannot load xpu CCL. CCL doesn't work for XPU device")

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

