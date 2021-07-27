import os
import shutil
import platform


IS_LINUX = (platform.system() == 'Linux')

BUILD_DIR = 'build'


def get_compiler(runtime):
    if runtime == 'dpcpp':
        cc = shutil.which('clang')
        cpp = shutil.which('clang++')
    else:
        cc = shutil.which('cc')
        cpp = shutil.which('c++')
    return cc, cpp


def check_env_flag(name, env=os.environ, default=''):
    return os.getenv(name, default).upper() in ['ON', '1', 'YES', 'TRUE', 'Y']
