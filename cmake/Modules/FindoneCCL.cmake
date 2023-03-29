# - Try to find oneCCL
#
# The following are set after configuration is done:
#  ONECCL_FOUND          : set to true if oneCCL is found.
#  ONECCL_INCLUDE_DIRS   : path to oneCCL include dir.
#  ONECCL_LIBRARIES      : list of libraries for oneCCL
#
# and the following imported targets:
#
#   oneCCL

IF (NOT ONECCL_FOUND)
SET(ONECCL_FOUND OFF)
SET(ONECCL_LIBRARIES)
SET(ONECCL_INCLUDE_DIRS)

IF (USE_SYSTEM_ONECCL)
    # Find the oneCCL in oneapi bundle
    set(oneapi_root_hint)
    if(DEFINED INTELONEAPIROOT)
        set(oneapi_root_hint ${INTELONEAPIROOT})
    elseif(DEFINED ENV{INTELONEAPIROOT})
        set(oneapi_root_hint $ENV{INTELONEAPIROOT})
    endif()

    IF(COMPUTE_BACKEND STREQUAL "dpcpp")
        SET(CCL_CONFIGURATION "cpu_gpu_dpcpp")
    ELSE()
        SET(CCL_CONFIGURATION "cpu_icc")
    ENDIF()

    find_package(oneCCL CONFIG REQUIRED PATHS ${oneapi_root_hint}/ccl/latest)

ELSE()
    SET(ONECCL_ROOT "${PROJECT_SOURCE_DIR}/third_party/oneCCL")

    IF(BUILD_NO_ONECCL_PACKAGE)
        ADD_SUBDIRECTORY(${ONECCL_ROOT} oneCCL EXCLUDE_FROM_ALL)
    ELSE()
        ADD_SUBDIRECTORY(${ONECCL_ROOT})
    ENDIF()

    IF(NOT TARGET ccl)
        MESSAGE(FATAL_ERROR "Failed to find oneCCL target")
    ENDIF()
    add_library(oneCCL ALIAS ccl)
ENDIF()

GET_TARGET_PROPERTY(INCLUDE_DIRS oneCCL INCLUDE_DIRECTORIES)
SET(ONECCL_INCLUDE_DIRS ${INCLUDE_DIRS})
SET(ONECCL_LIBRARIES oneCCL)

IF (NOT USE_SYSTEM_ONECCL)
  find_package_handle_standard_args(oneCCL FOUND_VAR ONECCL_FOUND REQUIRED_VARS ONECCL_LIBRARIES ONECCL_INCLUDE_DIRS)
ENDIF()

ENDIF(NOT ONECCL_FOUND)
