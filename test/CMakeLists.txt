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
  COMMAND python -c "import sys; import os; print(os.path.split(sys.executable)[0], end='')"
  OUTPUT_VARIABLE PYTHON_LIB_PATH
)
set(PYTHON_LIB_PATH "${PYTHON_LIB_PATH}/../lib")

execute_process(
  COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
  OUTPUT_VARIABLE TORCH_PATH
)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_PATH})
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -g")

link_directories(${TORCH_PATH}/lib $ENV{CCL_ROOT}/lib ${PYTHON_LIB_PATH})
set(CMAKE_INSTALL_RPATH ${PYTHON_LIB_PATH})
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_definitions(-DPROCESS_GROUP_CCL_TEST)

#show build info
message(STATUS "C compiler : ${CMAKE_C_COMPILER}")
message(STATUS "CXX compiler : ${CMAKE_CXX_COMPILER}")
message(STATUS "TORCH_LIBRARIES : ${TORCH_LIBRARIES}")
message(STATUS "TORCH_INCLUDE_DIRS : ${TORCH_INCLUDE_DIRS}")
message(STATUS "TORCH_CXX_FLAGS : ${TORCH_CXX_FLAGS}")
message(STATUS "CCL_ROOT : $ENV{CCL_ROOT}")
message(STATUS "PYTHON_LIB_PATH : ${PYTHON_LIB_PATH}")

set(test_src
    ${CMAKE_CURRENT_SOURCE_DIR}/../test/ProcessGroupCCLTest.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../src/ProcessGroupCCL.cpp)
set(test_name ProcessGroupCCLTest)
add_executable(${test_name} "${test_src}")
set_property(TARGET ${test_name} PROPERTY CXX_STANDARD 14)
target_include_directories(${test_name} PRIVATE ${TORCH_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../src $ENV{CCL_ROOT}/include)
target_link_libraries(${test_name} python3.6m shm c10 torch torch_cpu torch_python pthread ccl)
target_compile_options(${test_name} PRIVATE -Wno-error)
