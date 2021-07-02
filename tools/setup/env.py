import os
import platform

from . import which

IS_LINUX = (platform.system() == 'Linux')

BUILD_DIR = 'build'


def get_compiler(runtime):
    if runtime == 'dpcpp':
        cc = which('clang')
        cpp = which('clang++')
    else:
        cc = which('cc')
        cpp = which('c++')
    return cc, cpp


def check_env_flag(name, env=os.environ, default=''):
    return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']
