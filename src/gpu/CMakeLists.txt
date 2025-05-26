set(CCL_DPCPP_SRCS dpcpp_ccl.cpp ze_exception.hpp allreduce.h sycl_misc.hpp runtime.hpp cxxopts.hpp)

set_source_files_properties(${CCL_DPCPP_SRCS} PROPERTIES COMPILE_DEFINITIONS "USE_DPCPP;__STRICT_ANSI__")
set_source_files_properties(${CCL_DPCPP_SRCS} PROPERTIES COMPILE_FLAGS -fsycl)

add_library(oneccl_bindings_for_pytorch_xpu SHARED ${CCL_DPCPP_SRCS})

target_link_libraries(oneccl_bindings_for_pytorch_xpu PUBLIC ${DEPENDS_LIB})
target_link_libraries(oneccl_bindings_for_pytorch_xpu PUBLIC oneccl_bindings_for_pytorch)

foreach(RPATH ${CMAKE_INSTALL_RPATH})
    set_target_properties(oneccl_bindings_for_pytorch_xpu PROPERTIES LINK_FLAGS "-Wl,-rpath,${RPATH}")
endforeach()
set_target_properties(oneccl_bindings_for_pytorch_xpu PROPERTIES LINK_FLAGS "-Wl,--disable-new-dtags")

install(TARGETS oneccl_bindings_for_pytorch_xpu LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")

