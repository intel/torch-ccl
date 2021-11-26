# - Try to find oneCCL
#
# The following are set after configuration is done:
#  ONECCL_FOUND          : set to true if oneCCL is found.
#  ONECCL_INCLUDE_DIR    : path to oneCCL include dir.
#  ONECCL_LIBRARIES      : list of libraries for oneCCL
#
# The following variables are used:
#  ONECCL_USE_NATIVE_ARCH : Whether native CPU instructions should be used in ONECCL. This should be turned off for
#  general packaging to avoid incompatible CPU instructions. Default: OFF.

IF (NOT ONECCL_FOUND)
SET(ONECCL_FOUND OFF)

SET(ONECCL_LIBRARIES)
SET(ONECCL_INCLUDE_DIR)

IF (USE_SYSTEM_ONECCL)
    # Find the oneCCL in oneapi bundle
    set(oneapi_root_hint)
    if(DEFINED INTELONEAPIROOT)
        set(oneapi_root_hint ${INTELONEAPIROOT})
    elseif(DEFINED ENV{INTELONEAPIROOT})
        set(oneapi_root_hint $ENV{INTELONEAPIROOT})
    endif()

    IF(COMPUTE_BACKEND STREQUAL "dpcpp_level_zero")
        SET(CCL_CONFIGURATION "cpu_gpu_dpcpp")
    ELSE()
        SET(CCL_CONFIGURATION "cpu_icc")
    ENDIF()

    find_path(ONECCL_INCLUDE_DIR
            NAMES oneapi/ccl.hpp
            HINTS ${oneapi_root_hint}/ccl/latest
            PATH_SUFFIXES include/${CCL_CONFIGURATION})
    find_library(ONECCL_LIBRARIES
            NAMES ccl
            HINTS ${oneapi_root_hint}/ccl/latest
            PATH_SUFFIXES lib/${CCL_CONFIGURATION})

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(oneCCL
            REQUIRED_VARS
            ONECCL_INCLUDE_DIR
            ONECCL_LIBRARIES
            HANDLE_COMPONENTS
            )
ENDIF(USE_SYSTEM_ONECCL)

IF (NOT ONECCL_FOUND)
	IF (USE_DEV_ONECCL)
		SET(ONECCL_ROOT "${PROJECT_SOURCE_DIR}/third_party/Internal_oneCCL")
	ELSE()
    		SET(ONECCL_ROOT "${PROJECT_SOURCE_DIR}/third_party/oneCCL")
	ENDIF()
	

    ADD_SUBDIRECTORY(${ONECCL_ROOT} )
    IF(NOT TARGET ccl)
        MESSAGE(FATAL_ERROR "Failed to find oneCCL target")
    ENDIF()
    GET_TARGET_PROPERTY(INCLUDE_DIRS ccl INCLUDE_DIRECTORIES)
    SET(ONECCL_INCLUDE_DIR ${INCLUDE_DIRS})
#    add_library(mpi SHARED IMPORTED)
#    set_target_properties(mpi PROPERTIES IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/third_party/oneCCL/mpi/lib/libmpi.so)
#
#
#    add_library(fabric SHARED IMPORTED)
#    set_target_properties(fabric PROPERTIES IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/third_party/oneCCL/ofi/lib/libfabric.so)
#
#    SET(ONECCL_LIBRARIES ccl fabric mpi)
    SET(ONECCL_LIBRARIES ccl)
ENDIF(NOT ONECCL_FOUND)

ENDIF(NOT ONECCL_FOUND)
