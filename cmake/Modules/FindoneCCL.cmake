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

    IF(COMPUTE_BACKEND STREQUAL "dpcpp")
        SET(CCL_CONFIGURATION "cpu_gpu_dpcpp")
    ELSE()
        SET(CCL_CONFIGURATION "cpu_icc")
    ENDIF()

    find_package(oneCCL CONFIG REQUIRED PATHS ${INTELONEAPIROOT}/ccl/latest)

    SET(ONECCL_LIBRARIES oneCCL)
    SET(ONECCL_FOUND TRUE)
ENDIF(USE_SYSTEM_ONECCL)

IF (NOT ONECCL_FOUND)

    SET(ONECCL_ROOT "${PROJECT_SOURCE_DIR}/third_party/oneCCL")

    ADD_SUBDIRECTORY(${ONECCL_ROOT})
    IF(NOT TARGET ccl)
        MESSAGE(FATAL_ERROR "Failed to find oneCCL target")
    ENDIF()
    GET_TARGET_PROPERTY(INCLUDE_DIRS ccl INCLUDE_DIRECTORIES)
    SET(ONECCL_INCLUDE_DIR ${INCLUDE_DIRS})
    SET(ONECCL_LIBRARIES ccl)

ENDIF(NOT ONECCL_FOUND)

ENDIF(NOT ONECCL_FOUND)
