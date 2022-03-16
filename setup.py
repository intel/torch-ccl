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
TORCH_CCL_PATH = os.path.join(CWD, "torch_ccl")

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
    cwd = os.path.dirname(os.path.abspath(__file__))
    package_name = os.getenv('CCL_PACKAGE_NAME', 'oneccl-bind-pt')
    version = open('version.txt', 'r').read().strip()
    sha = 'Unknown'

    try:
        sha = check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
    except Exception:
        pass

    if os.getenv('CCL_SHA_VERSION', False):
        if sha != 'Unknown':
            version += '+' + sha[:7]

    print("Building {}-{}".format(package_name, version))

    version_path = os.path.join(cwd, 'torch_ccl', 'version.py')
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
            try:
                # temp patch the oneCCL code
                check_call(["git", "apply", "./patches/Update_Internal_oneCCL.patch"], cwd=CWD)
            except:
                # ignore patch fail
                pass
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
        install_dir = TORCH_CCL_PATH

        # Now that the necessary directories are created, build
        my_env = os.environ.copy()
        build_type = 'Release'

        if _check_env_flag('DEBUG'):
            build_type = 'Debug'

        build_options = {
            'CMAKE_BUILD_TYPE' : build_type,
            # The value cannot be easily obtained in CMakeLists.txt.
            'CMAKE_PREFIX_PATH': torch.utils.cmake_prefix_path,
            'PYTORCH_LIBRARY_DIRS': CMakeExtension.convert_cmake_dirs(library_paths()),
            # skip the example and test code in oneCCL
            'BUILD_EXAMPLES': 'OFF',
            'BUILD_CONFIG': 'OFF',
            'BUILD_UT': 'OFF',
            'BUILD_FT': 'OFF'
        }

        runtime = 'gcc'
        if 'COMPUTE_BACKEND' in os.environ:
            if os.environ['COMPUTE_BACKEND'] == 'dpcpp_level_zero':
                runtime = 'dpcpp'
                build_options['COMPUTE_BACKEND'] = os.environ['COMPUTE_BACKEND']
                import intel_extension_for_pytorch
                build_options['CMAKE_PREFIX_PATH'] += ";" + intel_extension_for_pytorch.cmake_prefix_path

        cc, cxx = get_compiler(runtime)
        build_options['CMAKE_C_COMPILER'] = cc
        build_options['CMAKE_CXX_COMPILER'] = cxx

        extension.generate(build_options, my_env, build_dir, install_dir)

        build_args = ['-j', str(os.cpu_count())]
        check_call(['make', 'torch_ccl'] + build_args, cwd=str(build_dir))
        if 'COMPUTE_BACKEND' in os.environ:
            if os.environ['COMPUTE_BACKEND'] == 'dpcpp_level_zero':
                check_call(['make', 'torch_ccl_xpu'] + build_args, cwd=str(build_dir))
        check_call(['make', 'install'], cwd=str(build_dir))


class Clean(clean):
    def run(self):
        import glob
        import re
        try:
            check_call(["git", "reset", "--hard"], cwd=os.path.join(CWD, "third_party/oneCCL"))
        except Exception as e:
            print("=" * 64 + "\nWARNNING!\n" + "=" * 64)
            print(e)
            print("=" * 64)

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
    main_libraries = ['torch_ccl']
    main_link_args = []
    main_sources = ["torch_ccl/csrc/_C.cpp", "torch_ccl/csrc/init.cpp"]
    lib_path = os.path.join(TORCH_CCL_PATH, "lib")
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
        return '-Wl,-rpath,$ORIGIN/' + path

    _c_module = CppExtension("torch_ccl._C",
                             libraries=main_libraries,
                             sources=main_sources,
                             language='c',
                             extra_compile_args=main_compile_args + extra_compile_args,
                             include_dirs=include_dirs,
                             library_dirs=library_dirs,
                             extra_link_args=extra_link_args + main_link_args + [make_relative_rpath('lib')])

    return _c_module


if __name__ == '__main__':
    version, package_name = create_version()
    c_module = get_python_c_module()
    cmake_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CMakeLists.txt")
    modules = [CMakeExtension("libtorch_ccl", cmake_file), c_module]
    setup(
        name=package_name,
        version=version,
        ext_modules=modules,
        packages=['torch_ccl'],
        package_data={
            'torch_ccl': [
                '*.py',
                '*/*.h',
                '*/*.hpp',
                'lib/*.so*',
                'bin/*',
                'env/*',
                'etc/*',
                'examples/*',
                'include/native_device_api/*.h*',
                'include/native_device_api/l0/*.h*',
                'include/*.h*',
                'lib/lib*',
                'lib/prov/lib*',
                'lib/kernels/*',
                'licensing/*',
                'modulefiles/*',
            ]},
        cmdclass={
            'build_ext': BuildCMakeExt,
            'clean': Clean,
        }
    )
