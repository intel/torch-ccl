cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 17)

project(oneccl_bindings_for_pytorch C CXX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=cpp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat-security")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstack-protector")
# Since 2016 Debian start using RUNPATH instead of normally RPATH, which gave the annoy effect that
# allow LD_LIBRARY_PATH to override dynamic linking path. Depends on intention of linking priority,
# change below for best outcome: disable, using RPATH, enable, using RUNPATH
if (ENABLE_LINKER_RUNPATH)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--enable-new-dtags")
else()
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--disable-new-dtags")
endif()

set(LINUX TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INSTALL_MESSAGE NEVER)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)

set(RPATH_VALUE $ORIGIN)
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${RPATH_VALUE}")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY true)

option(USE_SYSTEM_ONECCL "Use oneCCL library in system" OFF)

option(BUILD_NO_ONECCL_PACKAGE "Build with oneCCL excluded" OFF)

# Find the Torch lib
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Find OneCCL Lib
set(DEPENDS_LIB)
find_package(oneCCL REQUIRED)
list(APPEND DEPENDS_LIB oneCCL torch)

SET(LIB_NAME "oneccl_bindings_for_pytorch")

add_subdirectory(./src)

function (print_configuration_summary)
    get_directory_property(CMAKE_COMPILE_DEFINITIONS DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS)

    message(STATUS "")
    message(STATUS "******** Summary ********")
    message(STATUS "General:")
    message(STATUS "  CMake version         : ${CMAKE_VERSION}")
    message(STATUS "  CMake command         : ${CMAKE_COMMAND}")
    message(STATUS "  System                : ${CMAKE_SYSTEM_NAME}")
    message(STATUS "  Target name           : ${LIB_NAME}")
    message(STATUS "  Install path          : ${CMAKE_INSTALL_PREFIX}")
    message(STATUS "  Build type            : ${CMAKE_BUILD_TYPE}")
    message(STATUS "  C++ compiler          : ${CMAKE_CXX_COMPILER}")
    message(STATUS "  C++ compiler id       : ${CMAKE_CXX_COMPILER_ID}")
    message(STATUS "  C++ compiler version  : ${CMAKE_CXX_COMPILER_VERSION}")
    message(STATUS "  CXX flags             : ${CMAKE_CXX_FLAGS}")
    message(STATUS "  Compile flags         : ${IPEX_COMPILE_FLAGS}")
    message(STATUS "  Compile definitions   : ${CMAKE_COMPILE_DEFINITIONS}")
    message(STATUS "  Linker options        : ${CMAKE_SHARED_LINKER_FLAGS}")
    get_target_property(LINK_LIBRARIES oneccl_bindings_for_pytorch LINK_LIBRARIES)
    message(STATUS "  Linker libraries        : ${LINK_LIBRARIES}")
    get_target_property(LINK_DIRECTORS oneccl_bindings_for_pytorch LINK_DIRECTORIES)
    message(STATUS "  Linker directors        : ${LINK_DIRECTORS}")

    message(STATUS "")
endfunction()

print_configuration_summary()

