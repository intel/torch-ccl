find_package(TorchIpex REQUIRED)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-fsycl DPCPP_SUPPORTED)
if(NOT DPCPP_SUPPORTED)
    message(FATAL_ERROR "the compiler not support intel dpcpp")
else()
    set(CCL_DPCPP_SRCS dpcpp_ccl.cpp)

    set_source_files_properties(${CCL_DPCPP_SRCS} PROPERTIES COMPILE_DEFINITIONS "USE_DPCPP;__STRICT_ANSI__")
    set_source_files_properties(${CCL_DPCPP_SRCS} PROPERTIES COMPILE_FLAGS "-fsycl")

    add_library(torch_ccl_xpu SHARED ${CCL_DPCPP_SRCS})

    target_include_directories(torch_ccl_xpu PUBLIC ${ONECCL_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ${IPEX_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../)

    target_link_directories(torch_ccl_xpu PUBLIC ${IPEX_LIBRARY_DIRS})
    target_link_directories(torch_ccl_xpu PUBLIC ${PYTORCH_LIBRARY_DIRS})


    target_link_libraries(torch_ccl_xpu PUBLIC ${ONECCL_LIBRARIES})
    target_link_libraries(torch_ccl_xpu PUBLIC ${TORCH_LIBRARIES})
    target_link_libraries(torch_ccl_xpu PUBLIC torch_ccl)
    target_link_libraries(torch_ccl_xpu PUBLIC torch_ipex)
    target_link_libraries(torch_ccl_xpu PUBLIC c10d)

    set_target_properties(torch_ccl_xpu PROPERTIES LINK_FLAGS "-Wl,-rpath,${CMAKE_INSTALL_RPATH}")

    install(TARGETS torch_ccl_xpu LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")
endif()
