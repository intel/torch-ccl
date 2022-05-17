find_package(IntelExtensionPytorch REQUIRED)

set(CCL_DPCPP_SRCS dpcpp_ccl.cpp)

set_source_files_properties(${CCL_DPCPP_SRCS} PROPERTIES COMPILE_DEFINITIONS "USE_DPCPP;__STRICT_ANSI__")
set_source_files_properties(${CCL_DPCPP_SRCS} PROPERTIES COMPILE_FLAGS "-fsycl")

add_library(torch_ccl_xpu SHARED ${CCL_DPCPP_SRCS})

target_link_libraries(torch_ccl_xpu PUBLIC ${DEPENDS_LIB})
target_link_libraries(torch_ccl_xpu PUBLIC torch_ccl)
target_link_libraries(torch_ccl_xpu PUBLIC intel-ext-pt-gpu)

set_target_properties(torch_ccl_xpu PROPERTIES LINK_FLAGS "-Wl,-rpath,${CMAKE_INSTALL_RPATH}")

install(TARGETS torch_ccl_xpu LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")

