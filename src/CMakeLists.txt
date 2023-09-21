set(CCL_SRCS ProcessGroupCCL.cpp dispatch_stub.cpp utils.cpp ccl_comm_collector.cpp env.cpp)
set(CCL_CPU_SRCS cpu/cpu_ccl.cpp)
add_library(oneccl_bindings_for_pytorch SHARED ${CCL_SRCS} ${CCL_CPU_SRCS})
set_target_properties(oneccl_bindings_for_pytorch PROPERTIES OUTPUT_NAME ${LIB_NAME})
set_target_properties(oneccl_bindings_for_pytorch PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_options(oneccl_bindings_for_pytorch PUBLIC -Wall
        -Wno-sign-compare
        -Wno-unused-function
        -fsycl)

target_link_options(oneccl_bindings_for_pytorch PUBLIC -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device pvc -options -vc-codegen")

if(COMPUTE_BACKEND STREQUAL "dpcpp")
    add_subdirectory(./gpu)
    add_definitions (-DUSE_GPU)
endif()

target_include_directories(oneccl_bindings_for_pytorch PUBLIC ./)

target_link_libraries(oneccl_bindings_for_pytorch PUBLIC ${DEPENDS_LIB})

set_target_properties(oneccl_bindings_for_pytorch PROPERTIES LINK_FLAGS "-Wl,-rpath,${CMAKE_INSTALL_RPATH}")

install(TARGETS oneccl_bindings_for_pytorch LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")
