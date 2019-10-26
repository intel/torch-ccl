import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

if "CCL_ROOT" not in os.environ:
    raise Exception("CCL_ROOT is not defined! Please run CCL variable setting shell script.")

ext_modules = [
    CppExtension(
        'ccl', ['ProcessGroupCCL.cpp'],
        include_dirs=['{}/include/'.format(os.getenv("CCL_ROOT"))],
        library_dirs=['{}/lib/'.format(os.getenv("CCL_ROOT"))],
        libraries=['ccl']
        )
]

setup(
    name='ccl',
    version='1.0',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension})
