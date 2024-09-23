import os
import shutil
import platform


IS_LINUX = (platform.system() == 'Linux')

BUILD_DIR = 'build'


def get_compiler(runtime):
    if runtime == 'dpcpp':
        c_compiler = 'icx'
        cpp_compiler = 'icpx'
    else:
        c_compiler = os.environ.get('CC', 'cc')
        cpp_compiler = os.environ.get('CXX', 'c++')

    cc = shutil.which(c_compiler)
    cpp = shutil.which(cpp_compiler)
    if cpp is None or cc is None:
        raise RuntimeError("couldn't find the compiler '{}' or '{}'".format(c_compiler, cpp_compiler))
    return cc, cpp


def check_env_flag(name, env=os.environ, default=''):
    return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']
