import os
import sys
import warnings
from .version import __version__, git_version

cwd = cwd = os.path.dirname(os.path.abspath(__file__))

FI_PROVIDER_PATH = os.path.join(cwd, "lib/prov")
os.environ['FI_PROVIDER_PATH'] = FI_PROVIDER_PATH

from . import _C as ccl_lib

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

