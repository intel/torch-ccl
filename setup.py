# DEBUG build with debug
import os
import sys
import pathlib
import shutil
import multiprocessing
from subprocess import check_call, check_output

from torch.utils.cpp_extension import include_paths, library_paths
from setuptools import setup, Extension, distutils
from setuptools.command.build_ext import build_ext
from distutils.command.clean import clean
from tools.setup.cmake import CMakeExtension

# Constant known variables used throughout this file
CWD = os.path.dirname(os.path.abspath(__file__))
TORCH_CCL_PATH = os.path.join(CWD, "torch_ccl")

def check_file(f):
    if not os.path.exists(f):
        print("Could not find {}".format(f))
        print("Did you run 'git submodule update --init --recursive'?")
        sys.exit(1)


# all the work we need to do _before_ setup runs
def create_version():
    """Create the version string for torch-ccl"""
    cwd = os.path.dirname(os.path.abspath(__file__))
    package_name = os.getenv('CCL_PACKAGE_NAME', 'torch-ccl')
    version = open('version.txt', 'r').read().strip()
    sha = 'Unknown'

    try:
        sha = check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
    except Exception:
        pass

    if os.getenv('PYTORCH_BUILD_VERSION'):
        assert os.getenv('PYTORCH_BUILD_NUMBER') is not None
        build_number = int(os.getenv('PYTORCH_BUILD_NUMBER'))
        version = os.getenv('PYTORCH_BUILD_VERSION')
        if build_number > 1:
            version += '.post' + str(build_number)
    elif sha != 'Unknown':
        version += '+' + sha[:7]

    print("Building {}-{}".format(package_name, version))

    version_path = os.path.join(cwd, 'torch_ccl', 'version.py')
    with open(version_path, 'w') as f:
        f.write("__version__ = '{}'\n".format(version))
        f.write("git_version = {}\n".format(repr(sha)))

    return version


class BuildCMakeExt(build_ext):
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
        install_dir = TORCH_CCL_PATH

        # Now that the necessary directories are created, build
        my_env = os.environ.copy()

        build_options = {
            # The value cannot be easily obtained in CMakeLists.txt.
            'PYTHON_INCLUDE_DIRS': str(distutils.sysconfig.get_python_inc()),
            'PYTORCH_INCLUDE_DIRS': CMakeExtension.convert_cmake_dirs(include_paths()),
            'PYTORCH_LIBRARY_DIRS': CMakeExtension.convert_cmake_dirs(library_paths()),
        }

        extension.generate(build_options, my_env, build_dir, install_dir)

        max_jobs = os.getenv('MAX_JOBS', str(multiprocessing.cpu_count()))
        build_args = ['-j', max_jobs]
        check_call(['make', 'torch_ccl'] + build_args, cwd=str(build_dir), env=my_env)
        check_call(['make', 'install'], cwd=str(build_dir), env=my_env)


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
    main_libraries = ['torch_ccl']
    main_link_args = []
    main_sources = ["torch_ccl/csrc/_C.cpp"]
    lib_path = os.path.join(TORCH_CCL_PATH, "lib")
    library_dirs = [lib_path]
    include_path = os.path.join(CWD, "src")
    include_dirs = include_paths()
    include_dirs.append(include_path)
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

    _c_module = Extension("torch_ccl._C",
                          libraries=main_libraries,
                          sources=main_sources,
                          language='c',
                          extra_compile_args=main_compile_args + extra_compile_args,
                          include_dirs=include_dirs,
                          library_dirs=library_dirs,
                          extra_link_args=extra_link_args + main_link_args + [make_relative_rpath('lib')])

    return _c_module


if __name__ == '__main__':
    version = create_version()
    c_module = get_python_c_module()
    cmake_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CMakeLists.txt")
    modules = [CMakeExtension("libtorch_ccl", cmake_file), c_module]
    setup(
        name='torch_ccl',
        version=version,
        ext_modules=modules,
        packages=['torch_ccl'],
        install_requires=['torch'],
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
                'licensing/*',
                'modulefiles/*',
            ]},
        cmdclass={
            'build_ext': BuildCMakeExt,
            'clean': Clean,
        }
    )
