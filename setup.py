import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

if "CCL_ROOT" not in os.environ:
    raise Exception("CCL_ROOT is not defined! Please run oneCCL variable setting shell script.")

ext_modules = [
    CppExtension(
        name="torch_ccl",
        sources=['src/ProcessGroupCCL.cpp'],
        include_dirs=['{}/include/'.format(os.getenv("CCL_ROOT"))],
        library_dirs=['{}/lib/'.format(os.getenv("CCL_ROOT"))],
        libraries=['ccl'],
        extra_compile_args=['-Wformat', '-Wformat-security', '-D_FORTIFY_SOURCE=2', '-fstack-protector'],
        extra_link_args=['-Wl,-z,noexecstack', '-Wl,-z,relro', '-Wl,-z,now']
        )
]

setup(
    name='torch-ccl',
    version='1.0',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension})
