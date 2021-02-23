cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 14)

project(torch_ccl C CXX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=cpp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat-security")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstack-protector")

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

# Find OneCCL Lib
set(DEPENDS_LIB)
find_package(oneCCL REQUIRED)
list(APPEND DEPENDS_LIB ${ONECCL_LIBRARIES})

SET(LIB_NAME "torch_ccl")

#workaroud patch for oneCCL link issue
find_package(Patch)
if(Patch_FOUND)
  message("Patch found: ${Patch_EXECUTABLE}")
else()
  message(FATAL_ERROR, "Did not find patch")
endif()
set (patch_file "${CMAKE_CURRENT_SOURCE_DIR}/patches/Fix_the_link_path_issue_in_oneCCL.patch")
execute_process(COMMAND ${Patch_EXECUTABLE} -p1 -i ${patch_file} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

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
    get_target_property(LINK_LIBRARIES torch_ccl LINK_LIBRARIES)
    message(STATUS "  Linker directors        : ${LINK_LIBRARIES}")
    get_target_property(LINK_DIRECTORS torch_ccl LINK_DIRECTORIES)
    message(STATUS "  Linker libraries        : ${LINK_DIRECTORS}")

    message(STATUS "")
endfunction()

print_configuration_summary()

