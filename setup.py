# DEBUG build with debug
#
#   USE_SYSTEM_ONECCL=0
#     disables use of system-wide oneCCL (we will use our submoduled
#     copy in third_party/oneCCL)

import os
import sys
import pathlib
import shutil
from subprocess import check_call, check_output

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, library_paths
from setuptools import setup
from distutils.command.clean import clean
from tools.setup.cmake import CMakeExtension
from tools.setup.env import get_compiler

# Constant known variables used throughout this file
CWD = os.path.dirname(os.path.abspath(__file__))
ONECCL_BINDINGS_FOR_PYTORCH_PATH = os.path.join(CWD, "oneccl_bindings_for_pytorch")


def _check_env_flag(name, default=''):
    return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']


def check_file(f):
    if not os.path.exists(f):
        print("Could not find {}".format(f))
        print("Did you run 'git submodule update --init --recursive'?")
        sys.exit(1)


# all the work we need to do _before_ setup runs
def create_version():
    """Create the version string for torch-ccl"""
    package_name = os.getenv('CCL_PACKAGE_NAME', 'oneccl-bind-pt')
    version = open('version.txt', 'r').read().strip()
    sha = 'Unknown'

    try:
        sha = check_output(['git', 'rev-parse', 'HEAD'], cwd=CWD).decode('ascii').strip()
    except Exception:
        pass

    if os.getenv('CCL_SHA_VERSION', False):
        if sha != 'Unknown':
            version += '+' + sha[:7]

    if os.environ.get("COMPUTE_BACKEND") == "dpcpp":
        backend = "gpu"
    else:
        backend = os.environ.get("ONECCL_BINDINGS_FOR_PYTORCH_BACKEND", "cpu")

    if "+" not in version:
        version += '+' + backend

    print("Building {}-{}".format(package_name, version))

    version_path = os.path.join(CWD, 'oneccl_bindings_for_pytorch', 'version.py')
    with open(version_path, 'w') as f:
        f.write("__version__ = '{}'\n".format(version))
        f.write("git_version = {}\n".format(repr(sha)))

    return version, package_name


class BuildCMakeExt(BuildExtension):
    """
    Builds using cmake instead of the python setuptools implicit build
    """

    def run(self):
        """
        Perform build_cmake before doing the 'normal' stuff
        """
        cmake_extensions = [ext for ext in self.extensions if isinstance(ext, CMakeExtension)]
        for ext in cmake_extensions:
            self.build_cmake(ext)

        self.extensions = [ext for ext in self.extensions if not isinstance(ext, CMakeExtension)]
        super(BuildCMakeExt, self).run()
        build_py = self.get_finalized_command('build_py')
        build_py.data_files = build_py._get_data_files()
        build_py.run()

    def build_cmake(self, extension: CMakeExtension):
        """
        The steps required to build the extension
        """
        build_dir = pathlib.Path('.'.join([self.build_temp, extension.name]))

        build_dir.mkdir(parents=True, exist_ok=True)
        install_dir = ONECCL_BINDINGS_FOR_PYTORCH_PATH

        # Now that the necessary directories are created, build
        my_env = os.environ.copy()
        my_env["CMAKE_DISABLE_FIND_PACKAGE_MKL"] = "TRUE"
        build_type = 'Release'

        if _check_env_flag('DEBUG'):
            build_type = 'Debug'

        build_options = {
            'CMAKE_BUILD_TYPE': build_type,
            # The value cannot be easily obtained in CMakeLists.txt.
            'CMAKE_PREFIX_PATH': torch.utils.cmake_prefix_path,
            # skip the example and test code in oneCCL
            'BUILD_EXAMPLES': 'OFF',
            'BUILD_CONFIG': 'OFF',
            'BUILD_FT': 'OFF'
        }

        compute_backend = os.getenv('COMPUTE_BACKEND', 'n/a')
        runtime = 'gcc'
        if compute_backend == 'dpcpp':
            runtime = 'dpcpp'
            build_options['COMPUTE_BACKEND'] = compute_backend
            if "DPCPP_GCC_INSTALL_DIR" in my_env:
                exist_cflags = "CFLAGS" in my_env
                cflags = ""
                if exist_cflags:
                    cflags = my_env["CFLAGS"]
                my_env["CFLAGS"] = f"--gcc-install-dir={my_env['DPCPP_GCC_INSTALL_DIR']} {cflags}"
                exist_cxxflags = "CXXFLAGS" in my_env
                cxxflags = ""
                if exist_cxxflags:
                    cxxflags = my_env["CXXFLAGS"]
                my_env["CXXFLAGS"] = f"--gcc-install-dir={my_env['DPCPP_GCC_INSTALL_DIR']} {cxxflags}"
                exist_ldflags = "LDFLAGS" in my_env
                ldflags = ""
                if exist_ldflags:
                    ldflags = my_env["LDFLAGS"]
                my_env["LDFLAGS"] = f"--gcc-install-dir={my_env['DPCPP_GCC_INSTALL_DIR']} -fuse-ld=lld -lrt -lpthread {ldflags}"

        cc, cxx = get_compiler(runtime)
        build_options['CMAKE_C_COMPILER'] = cc
        build_options['CMAKE_CXX_COMPILER'] = cxx

        extension.generate(build_options, my_env, build_dir, install_dir)

        if compute_backend == 'dpcpp':
            if "DPCPP_GCC_INSTALL_DIR" in my_env:
                if exist_cflags:
                    my_env["CFLAGS"] = cflags
                else:
                    del my_env["CFLAGS"]
                if exist_cxxflags:
                    my_env["CXXFLAGS"] = cxxflags
                else:
                    del my_env["CXXFLAGS"]
                if exist_ldflags:
                    my_env["LDFLAGS"] = ldflags
                else:
                    del my_env["LDFLAGS"]

        build_args = ['-j', str(os.cpu_count())]
        check_call(['make', 'oneccl_bindings_for_pytorch'] + build_args, cwd=str(build_dir))
        if compute_backend == 'dpcpp':
            check_call(['make', 'oneccl_bindings_for_pytorch_xpu'] + build_args, cwd=str(build_dir))
        check_call(['make', 'install'], cwd=str(build_dir))


class Clean(clean):
    def run(self):
        import glob
        import re

        with open('.gitignore', 'r') as f:
            ignores = f.read()
            pat = re.compile(r'^#( BEGIN NOT-CLEAN-FILES )?')
            for wildcard in filter(None, ignores.split('\n')):
                match = pat.match(wildcard)
                if match:
                    if match.group(1):
                        # Marker is found and stop reading .gitignore.
                        break
                    # Ignore lines which begin with '#'.
                else:
                    for filename in glob.glob(wildcard):
                        try:
                            os.remove(filename)
                        except OSError:
                            shutil.rmtree(filename, ignore_errors=True)

        clean.run(self)


def get_python_c_module():
    main_compile_args = []
    main_libraries = ['oneccl_bindings_for_pytorch']
    main_link_args = []
    main_sources = ["oneccl_bindings_for_pytorch/csrc/_C.cpp", "oneccl_bindings_for_pytorch/csrc/init.cpp"]
    lib_path = os.path.join(ONECCL_BINDINGS_FOR_PYTORCH_PATH, "lib")
    library_dirs = [lib_path]
    include_path = os.path.join(CWD, "src")
    include_dirs = [include_path]
    extra_link_args = []
    extra_compile_args = [
        '-Wall',
        '-Wextra',
        '-Wno-strict-overflow',
        '-Wno-unused-parameter',
        '-Wno-missing-field-initializers',
        '-Wno-write-strings',
        '-Wno-unknown-pragmas',
        # This is required for Python 2 declarations that are deprecated in 3.
        '-Wno-deprecated-declarations',
        # Python 2.6 requires -fno-strict-aliasing, see
        # http://legacy.python.org/dev/peps/pep-3123/
        # We also depend on it in our code (even Python 3).
        '-fno-strict-aliasing',
        # Clang has an unfixed bug leading to spurious missing
        # braces warnings, see
        # https://bugs.llvm.org/show_bug.cgi?id=21629
        '-Wno-missing-braces',
    ]

    def make_relative_rpath(path):
        ret = []
        ret.append('-Wl,-rpath,$ORIGIN/' + path)
        if os.getenv('COMPUTE_BACKEND', 'n/a') == 'dpcpp':
            ret.append('-Wl,-rpath,$ORIGIN/../../../')
            ret.append('-Wl,--disable-new-dtags')
        return ret

    _c_module = CppExtension("oneccl_bindings_for_pytorch._C",
                             libraries=main_libraries,
                             sources=main_sources,
                             language='c',
                             extra_compile_args=main_compile_args + extra_compile_args,
                             include_dirs=include_dirs,
                             library_dirs=library_dirs,
                             extra_link_args=extra_link_args + main_link_args + make_relative_rpath('lib'))

    return _c_module


if __name__ == '__main__':
    version, package_name = create_version()
    c_module = get_python_c_module()
    cmake_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CMakeLists.txt")
    modules = [CMakeExtension("liboneccl_bindings_for_pytorch", cmake_file), c_module]
    setup(
        name=package_name,
        version=version,
        ext_modules=modules,
        packages=['oneccl_bindings_for_pytorch'],
        package_data={
            'oneccl_bindings_for_pytorch': [
                '*.py',
                '*/*.h',
                '*/*.hpp',
                'lib/*.so*',
                'opt/mpi/lib/*.so*',
                'bin/*',
                'opt/mpi/bin/*',
                'env/*',
                'etc/*',
                'opt/mpi/etc/*',
                'examples/*',
                'include/native_device_api/*.h*',
                'include/native_device_api/l0/*.h*',
                'include/*.h*',
                'opt/mpi/include/*.h*',
                'lib/lib*',
                'opt/mpi/libfabric/lib/lib*',
                'lib/prov/lib*',
                'lib/ccl/kernels/*',
                'opt/mpi/libfabric/lib/prov/lib*',
                'licensing/*',
                'modulefiles/*',
            ]},
        cmdclass={
            'build_ext': BuildCMakeExt,
            'clean': Clean,
        }
    )
