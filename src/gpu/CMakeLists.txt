find_package(IPEX REQUIRED)

set(CCL_DPCPP_SRCS dpcpp_ccl.cpp)

set_source_files_properties(${CCL_DPCPP_SRCS} PROPERTIES COMPILE_DEFINITIONS "USE_DPCPP;__STRICT_ANSI__")
set_source_files_properties(${CCL_DPCPP_SRCS} PROPERTIES COMPILE_FLAGS "-fsycl")

add_library(oneccl_bindings_for_pytorch_xpu SHARED ${CCL_DPCPP_SRCS})

target_link_libraries(oneccl_bindings_for_pytorch_xpu PUBLIC ${DEPENDS_LIB})
target_link_libraries(oneccl_bindings_for_pytorch_xpu PUBLIC oneccl_bindings_for_pytorch)
target_link_libraries(oneccl_bindings_for_pytorch_xpu PUBLIC intel-ext-pt-gpu)

set_target_properties(oneccl_bindings_for_pytorch_xpu PROPERTIES LINK_FLAGS "-Wl,-rpath,${CMAKE_INSTALL_RPATH}")

install(TARGETS oneccl_bindings_for_pytorch_xpu LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")

