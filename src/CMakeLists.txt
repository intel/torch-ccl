set(CCL_SRCS ProcessGroupCCL.cpp dispatch_stub.cpp utils.cpp ccl_comm_collector.cpp env.cpp)
set(CCL_CPU_SRCS cpu/cpu_ccl.cpp)
add_library(torch_ccl SHARED ${CCL_SRCS} ${CCL_CPU_SRCS})
set_target_properties(torch_ccl PROPERTIES OUTPUT_NAME ${LIB_NAME})
set_target_properties(torch_ccl PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_options(torch_ccl PUBLIC -Wall
        -Wno-sign-compare
        -Wno-unused-function)

if(COMPUTE_BACKEND STREQUAL "dpcpp")
    target_compile_options(torch_ccl PUBLIC -fsycl)
    add_subdirectory(./gpu)
endif()

target_include_directories(torch_ccl PUBLIC ${ONECCL_INCLUDE_DIR} ${TORCH_INCLUDE_DIRS} ./)


target_link_libraries(torch_ccl PUBLIC ${DEPENDS_LIB})
target_link_libraries(torch_ccl PUBLIC ${TORCH_LIBRARIES})

set_target_properties(torch_ccl PROPERTIES LINK_FLAGS "-Wl,-rpath,${CMAKE_INSTALL_RPATH}")

install(TARGETS torch_ccl LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")
