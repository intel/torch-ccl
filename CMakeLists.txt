cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

set(LINUX TRUE)
set(CMAKE_INSTALL_MESSAGE NEVER)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(PLUGIN_NAME torch_ccl)

PROJECT(${PLUGIN_NAME} C CXX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=cpp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat-security")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstack-protector")
set(CMAKE_CXX_STANDARD 14)

#Set installed PyTorch dir firstly
if(DEFINED PYTORCH_INSTALL_DIR)
  include_directories(${PYTORCH_INSTALL_DIR}/include)
  include_directories(${PYTORCH_INSTALL_DIR}/include/torch/csrc/api/include)
else()
  message(FATAL_ERROR "Plese set -DPYTORCH_INSTALL_DIR to Pytorch install path firstly")
endif()

add_subdirectory(third_party/oneCCL)
add_subdirectory(third_party/pybind11)
pybind11_add_module(${PLUGIN_NAME} SHARED src/ProcessGroupCCL.cpp)
include_directories(${PROJECT_SOURCE_DIR}/build/third_party/pybind11/include)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/third_party/oneCCL/include)

add_dependencies(${PLUGIN_NAME} ccl)
add_dependencies(${PLUGIN_NAME} pybind11)


link_directories(${PYTORCH_INSTALL_DIR}/lib)
target_link_libraries(${PLUGIN_NAME} PUBLIC ${PYTORCH_INSTALL_DIR}/lib/libtorch_python.so)
target_link_libraries(${PLUGIN_NAME} PUBLIC ${PYTORCH_INSTALL_DIR}/lib/libtorch_cpu.so)
target_link_libraries(${PLUGIN_NAME} PUBLIC ${PYTORCH_INSTALL_DIR}/lib/libc10.so)
target_link_libraries(${PLUGIN_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/build/third_party/oneCCL/src/libccl.a)
target_compile_options(${PLUGIN_NAME} PRIVATE "-DC10_BUILD_MAIN_LIB")
