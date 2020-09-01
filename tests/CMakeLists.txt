#
# Copyright (c) 2020, Intel Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the Intel Corporation nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#

cmake_minimum_required (VERSION 2.8)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(GCC_MIN_SUPPORTED "8.3.0")
if (NOT ${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" OR (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" AND ${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS ${GCC_MIN_SUPPORTED}))
    message(FATAL_ERROR "Use GNU compiler 8.3.0 or higher")
endif()

if (NOT DEFINED ENV{CCL_ROOT})
    message(FATAL_ERROR "CCL_ROOT is not defined! Please run oneCCL variable setting shell script.")
endif()

execute_process(
  COMMAND python -c "import sysconfig; print(sysconfig.get_config_var(\"LIBDIR\"), end='')"
  OUTPUT_VARIABLE PYTHON_LIB_PATH
)

execute_process(
  COMMAND bash -c "ls ${PYTHON_LIB_PATH} | grep '^libpython.*.so$' | head -n 1"
  OUTPUT_VARIABLE PYTHON_LIB_NAME
)

string(REPLACE "lib" "" PYTHON_LIB_NAME "${PYTHON_LIB_NAME}")
string(REPLACE ".so" "" PYTHON_LIB_NAME "${PYTHON_LIB_NAME}")
string(REPLACE "\n" "" PYTHON_LIB_NAME "${PYTHON_LIB_NAME}")

execute_process(
  COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
  OUTPUT_VARIABLE TORCH_PATH
)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_PATH})
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -g -fopenmp")

link_directories(${TORCH_PATH}/lib $ENV{CCL_ROOT}/lib ${PYTHON_LIB_PATH})
set(CMAKE_INSTALL_RPATH ${PYTHON_LIB_PATH})
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_definitions(-DPROCESS_GROUP_CCL_TEST)

# show build info
message(STATUS "C compiler : ${CMAKE_C_COMPILER}")
message(STATUS "CXX compiler : ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_CXX_FLAGS : ${CMAKE_CXX_FLAGS}")
message(STATUS "TORCH_LIBRARIES : ${TORCH_LIBRARIES}")
message(STATUS "TORCH_INCLUDE_DIRS : ${TORCH_INCLUDE_DIRS}")
message(STATUS "TORCH_CXX_FLAGS : ${TORCH_CXX_FLAGS}")
message(STATUS "CCL_ROOT : $ENV{CCL_ROOT}")
message(STATUS "PYTHON_LIB_PATH : ${PYTHON_LIB_PATH}")
message(STATUS "PYTHON_LIB_NAME : ${PYTHON_LIB_NAME}")

set(test_src
    ${CMAKE_CURRENT_SOURCE_DIR}/../tests/ProcessGroupCCLTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/ProcessGroupCCL.cpp)
set(test_name ProcessGroupCCLTest)
add_executable(${test_name} "${test_src}")
set_property(TARGET ${test_name} PROPERTY CXX_STANDARD 14)
target_include_directories(${test_name} PRIVATE ${TORCH_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../src $ENV{CCL_ROOT}/include)
target_link_libraries(${test_name} ${PYTHON_LIB_NAME} shm c10 torch torch_cpu torch_python pthread ccl)
target_compile_options(${test_name} PRIVATE -Wno-error)
