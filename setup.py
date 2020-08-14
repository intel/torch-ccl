#
import os
import sys
import platform
import multiprocessing

from subprocess import check_call
from setuptools import setup, Extension, distutils
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

from distutils.spawn import find_executable
from distutils.version import LooseVersion

version = '1.0.1'

try:
    import torch
except ImportError as e:
    print('Unable to import torch. Error:')
    print('\t', e)
    print('You need to install pytorch first.')
    sys.exit(1)

pytorch_dir = os.path.abspath(os.path.dirname(torch.__file__))

if 'DEBUG' in os.environ:
    build_type = 'Debug'
elif 'PROFILE' in os.environ:
    build_type = 'Profile'
else:
    build_type = 'Release'

class TorchCCLExtension(Extension):
    def __init__(self, name, project_dir=os.path.dirname(__file__)):
        Extension.__init__(self, name, sources=[])
        self.project_dir = os.path.abspath(project_dir)
        self.build_dir = os.path.join(project_dir, 'build')

class TorchCCLBuildExt(build_ext):
    def run(self):
        # Prefer cmake3 over cmake on CentOS.
        cmake = find_executable('cmake3') or find_executable('cmake')
        if cmake == None:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                    ", ".join(e.name for e in self.extensions))
        self.cmake = cmake

        if platform.system() == "Windows":
            raise RuntimeError("Windows is not supported!!!")

        for ext in self.extensions:
            self.build_extension(ext)
        build_py = self.get_finalized_command('build_py')
        build_py.data_files = build_py._get_data_files()
        build_py.run()

    def build_extension(self, ext):
        ext_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        ccl_install_path = os.path.join("..", "ccl")
        cmake_args = [
                '-DPYTORCH_INSTALL_DIR=' + pytorch_dir,
                '-DCMAKE_BUILD_TYPE=' + build_type,
                '-DCMAKE_INSTALL_PREFIX=' + ccl_install_path,
                '-DPYTHON_EXECUTABLE=' + sys.executable,
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + ext_dir,
            ]
        build_args = ['-j', str(multiprocessing.cpu_count())]

        if not os.path.exists(ext.build_dir):
            os.mkdir(ext.build_dir)

        env = os.environ.copy()
        check_call([self.cmake, ext.project_dir] + cmake_args,
                cwd=ext.build_dir, env=env)
        check_call(['make','install'] + build_args, cwd=ext.build_dir, env=env)

cmdclass = {
    'build_ext' : TorchCCLBuildExt,
}

setup(
    name='torch_ccl',
    version = version,
    description=('Intel Torch-ccl'),
    package_dir={"ccl": "ccl"},
    packages=['ccl'],
    package_data ={
        'ccl':[
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
    zip_safe = False,
    ext_modules=[TorchCCLExtension('torch_ccl')],
    cmdclass=cmdclass,
)
