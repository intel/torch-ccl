#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <system_error>

#include <mpi.h>
#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>

#include "cxxopts.hpp"
#include "ze_exception.hpp"
#include "sycl_misc.hpp"

#include <iostream>
#include <stdlib.h>

#define ATOMIC_RELAXED

struct exchange_contents {
  // first 4-byte is file descriptor for drmbuf or gem object
  union {
    ze_ipc_mem_handle_t ipc_handle;
    int fd = -1;
  };
  size_t offset = 0;
  int pid = -1;
};

#define sysCheck(x) \
  if (x == -1) {  \
    throw std::system_error(  \
        std::make_error_code(std::errc(errno)));  \
  }

template <typename data_type, uint32_t max_rank = 8, uint32_t max_buffer = 1024 /*KB*/>
class allreducer {
public:
    allreducer(){
        initialized = false;
    }

    void init(sycl::queue& queue, uint32_t rank_in, uint32_t world_in){
        rank = rank_in;
        world = world_in;
        // temporal buffer used for allreduce temporal use only.
        void* local_buffer = sycl::malloc_device(world * max_buffer * 1024 + 1024, queue);
        uint32_t* ptr = (uint32_t *)local_buffer + world * max_buffer * 1024 / sizeof(uint32_t);
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range { 1024 / sizeof(uint32_t) }, ([=](sycl::id<1> index) {
                    if (index % 32 != 16) {
                        ptr[index] = 0;
                    } else {
                        ptr[index] = world_in;
                    }
                }));
        });
        local_sync_buffer = (void *)ptr;
        queue.wait();
        // XXX: gain access to remote pointers
        exchange_peer_ipc_mem(queue, local_buffer);

        char *tmp_str = getenv("DISABLE_WORK");
        if (tmp_str) {
            disable_work = atoi(tmp_str);
        }
        tmp_str = getenv("DISABLE_CHECK");
        if (tmp_str) {
            disable_check = atoi(tmp_str);
        }
        tmp_str = getenv("DISABLE_SYNC");
        if (tmp_str) {
            disable_sync = atoi(tmp_str);
        }

        initialized = true;
    }

    void allreduce(sycl::queue& queue, void* inout_buffer, uint32_t size){
        assert(initialized == true);
        void* temp_buffer[max_rank];
        for (uint32_t i = 0; i < world; i++){
            temp_buffer[i] = buffer[i];
        }
        void* temp_sync_buffer[max_rank];
        for (uint32_t i = 0; i < world; i++){
            temp_sync_buffer[i] = sync_buffer[i];
        }
        void* temp_ready_buffer[max_rank];
        for (uint32_t i = 0; i < world; i++){
            temp_ready_buffer[i] = ready_buffer[i];
        }
        uint32_t temp_rank = rank;
        uint32_t temp_world = world;

        char* temp_local_sync_buffer = (char*)local_sync_buffer;

        if (disable_check == 0) {
            queue.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range { temp_world }, ([=](sycl::id<1> index) {
                        // atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);
                        if (index != temp_rank) {
                            int * temp_sync_ptr = (int *)(temp_local_sync_buffer + 128 * index + 64);

                            auto v =
        #ifdef ATOMIC_RELAXED
                                            sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                                                    sycl::memory_scope::system,
                                                                    sycl::access::address_space::global_space>(temp_sync_ptr[0]);
        #else
                                            sycl::atomic_ref<int, sycl::memory_order::acq_rel,
                                                                    sycl::memory_scope::system,
                                                                    sycl::access::address_space::global_space>(temp_sync_ptr[0]);
        #endif
                            // spining to wait all peers updating the count.
                            int count = v.load();
                            while (count < 1){
                                count = v.load();
                            }
                            v.store(0);
                        }
                    }));
            });
        }
        if (disable_work == 0) {
            queue.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range { size }, ([=](sycl::id<1> index) {
                        // copy input data to temp buffer on all peers
                        for (uint32_t i = 0; i < temp_world; i++) {
                            data_type * peer_ptr = (data_type*)temp_buffer[i];
                            peer_ptr[temp_rank * max_buffer * 1024 / sizeof(data_type) + index] = ((data_type*)inout_buffer)[index];
                        }
                    }));
            });
        }

        if (disable_sync == 0) {
            queue.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range { temp_world }, ([=](sycl::id<1> index) {
                        if (index != temp_rank) {
                            int * peer_sync_ptr = (int*)temp_sync_buffer[index];
                            auto v =
    #ifdef ATOMIC_RELAXED
                                        sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                                                sycl::memory_scope::system,
                                                                sycl::access::address_space::global_space>(peer_sync_ptr[0]);
    #else
                                        sycl::atomic_ref<int, sycl::memory_order::acq_rel,
                                                                sycl::memory_scope::system,
                                                                sycl::access::address_space::global_space>(peer_sync_ptr[0]);
    #endif
                                    v.store(1);

                            int * temp_sync_ptr = (int *)(temp_local_sync_buffer + 128 * index);
                            auto w =
        #ifdef ATOMIC_RELAXED
                                            sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                                                    sycl::memory_scope::system,
                                                                    sycl::access::address_space::global_space>(temp_sync_ptr[0]);
        #else
                                            sycl::atomic_ref<int, sycl::memory_order::acq_rel,
                                                                    sycl::memory_scope::system,
                                                                    sycl::access::address_space::global_space>(temp_sync_ptr[0]);
        #endif
                            // spining to wait all peers updating the count.
                            int count = w.load();
                            while (count < 1) {
                                count = w.load();
                            }
                            // reset the count for next allreduce
                            w.store(0);
                        }
                    }));
            });
        }

        if (disable_work == 0) {
            queue.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range { size }, ([=](sycl::id<1> index) {
                        // perform local sum
                        data_type * ptr = (data_type *)temp_buffer[temp_rank];
                        data_type sum = 0;
                        for (uint32_t i = 0; i < temp_world; i++){
                            sum += ptr[i * max_buffer * 1024 / sizeof(data_type) + index];
                        }
                        ((data_type*)inout_buffer)[index] = sum;
                    }));
            });
        }

        if (disable_check == 0) {
            queue.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(sycl::range { temp_world }, ([=](sycl::id<1> index) {
                        // remote atomice based sync
                        if (index != temp_rank) {
                            int * peer_sync_ptr = (int*)temp_ready_buffer[index];
                            auto v =
    #ifdef ATOMIC_RELAXED
                                        sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                                                sycl::memory_scope::system,
                                                                sycl::access::address_space::global_space>(peer_sync_ptr[0]);
    #else
                                        sycl::atomic_ref<int, sycl::memory_order::acq_rel,
                                                                sycl::memory_scope::system,
                                                                sycl::access::address_space::global_space>(peer_sync_ptr[0]);
    #endif
                                    v.store(1);
                        }
                    }));
            });
        }
    }

    void allreduce_old(sycl::queue& queue, void* inout_buffer, uint32_t size){
        assert(initialized == true);        
        void* temp_buffer[max_rank];
        for (uint32_t i = 0; i < world; i++){
            temp_buffer[i] = buffer[i];
        }
        void* temp_sync_buffer[max_rank];
        for (uint32_t i = 0; i < world; i++){
            temp_sync_buffer[i] = sync_buffer[i];
        }
        uint32_t temp_rank = rank;
        uint32_t temp_world = world;
        queue.submit([&](sycl::handler& cgh) {    
            cgh.parallel_for(sycl::range { size }, ([=](sycl::id<1> index) {
                    // copy input data to temp buffer on all peers
                    for (uint32_t i = 0; i < temp_world; i++){
                        data_type * peer_ptr = (data_type*)temp_buffer[i];
                        peer_ptr[temp_rank * max_buffer * 1024 / sizeof(data_type) + index] = ((data_type*)inout_buffer)[index];
                    }
                    atomic_fence(sycl::memory_order::release, sycl::memory_scope::system);                    

                    // remote atomice based sync
                    for (uint32_t i = 0; i < temp_world; i++){
                        int * peer_sync_ptr = (int*)temp_sync_buffer[i];
                        auto v =
#ifdef ATOMIC_RELAXED
                                     sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                                              sycl::memory_scope::system,
                                                              sycl::access::address_space::global_space>(peer_sync_ptr[0]);
#else
                                     sycl::atomic_ref<int, sycl::memory_order::acq_rel,
                                                              sycl::memory_scope::system,
                                                              sycl::access::address_space::global_space>(peer_sync_ptr[0]);
#endif
                                 v.fetch_add(1);

                    }
                    int * sync_ptr = (int *)temp_sync_buffer[temp_rank];
                    auto v =
#ifdef ATOMIC_RELAXED
                                     sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                                              sycl::memory_scope::system,
                                                              sycl::access::address_space::global_space>(sync_ptr[0]);
#else
                                     sycl::atomic_ref<int, sycl::memory_order::acq_rel,
                                                              sycl::memory_scope::system,
                                                              sycl::access::address_space::global_space>(sync_ptr[0]);
#endif
                    // spining to wait all peers updating the count.
                    int count = v.load();
                    while (count < temp_world * size){
                        count = v.load();
                    }
                    // perform local sum
                    data_type * ptr = (data_type *)temp_buffer[temp_rank];
                    data_type sum = 0;
                    for (uint32_t i = 0; i < temp_world; i++){
                        sum += ptr[i * max_buffer * 1024 / sizeof(data_type) + index];
                    }
                    ((data_type*)inout_buffer)[index] = sum;
                    // reset the count for next allreduce
                    v.store(0);
                }));
        });
    }

    void release(sycl::queue& queue){        
        // Clean up, close/put ipc handles, free memory, etc.
        auto l0_ctx = sycl::get_native<
            sycl::backend::ext_oneapi_level_zero>(queue.get_context());
        for (uint32_t i = 0; i < world; i++){
            if (i != rank){
                zeCheck(zeMemCloseIpcHandle(l0_ctx, (char *)buffer[i] - offset[i]));
            }                
        }     
        
        sycl::free(buffer[rank], queue);
        initialized = false;
    }

    void allreduce_2steps(sycl::queue& queue, void* inout_buffer, uint32_t size){
        assert(initialized == true);
        void* temp_buffer[max_rank];
        for (uint32_t i = 0; i < world; i++){
            temp_buffer[i] = buffer[i];
        }
        uint32_t temp_rank = rank;
        uint32_t temp_world = world;
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for(sycl::range { size }, ([=](sycl::id<1> index) {
                    for (uint32_t i = 0; i < temp_world; i++){
                        data_type * peer_ptr = (data_type*)temp_buffer[i];
                        peer_ptr[temp_rank * max_buffer * 1024 / sizeof(data_type) + index] = ((data_type*)inout_buffer)[index];
                    }
                }));
        });
        queue.wait();
        MPI_Barrier(MPI_COMM_WORLD);
        queue.submit([&](sycl::handler& cgh) {    
            cgh.parallel_for(sycl::range { size }, ([=](sycl::id<1> index) {
                data_type * ptr = (data_type *)temp_buffer[temp_rank];
                data_type sum = 0;
                for (uint32_t i = 0; i < temp_world; i++){
                    sum += ptr[i * max_buffer * 1024 / sizeof(data_type) + index];
                }
                ((data_type*)inout_buffer)[index] = sum;
            }));
        });          
    }

private:
    void exchange_peer_ipc_mem(sycl::queue& queue, void* ptr) {
        // Step 1: Get base address of the pointer
        sycl::context ctx = queue.get_context();
        auto l0_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);

        void *base_addr;
        size_t base_size;
        zeCheck(zeMemGetAddressRange(l0_ctx, ptr, &base_addr, &base_size));

        // Step 2: Get IPC mem handle from base address
        alignas(64) exchange_contents send_buf;
        alignas(64) exchange_contents recv_buf[world];

        // fill in the exchange info
        zeCheck(zeMemGetIpcHandle(l0_ctx, base_addr, &send_buf.ipc_handle));
        send_buf.offset = (char*)ptr - (char*)base_addr;
        send_buf.pid = getpid();

        // Step 3: Exchange the handles and offsets
        memset(recv_buf, 0, sizeof(recv_buf));
        // Overkill if we don't really needs all peer's handles
        MPI_Allgather(
            &send_buf, sizeof(send_buf), MPI_BYTE, recv_buf, sizeof(send_buf), MPI_BYTE, MPI_COMM_WORLD);

        
        for (uint32_t i = 0; i < world; i++){
            // Step 4: Prepare pid file descriptor of next process
            auto* peer = recv_buf + i;
            auto pid_fd = syscall(__NR_pidfd_open, peer->pid, 0);
            sysCheck(pid_fd);
            //
            // Step 5: Duplicate GEM object handle to local process
            // and overwrite original file descriptor number
            //
            peer->fd = syscall(__NR_pidfd_getfd, pid_fd, peer->fd, 0);
            sysCheck(peer->fd);

            // Step 6: Open IPC handle of remote peer
            auto l0_device
                = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_device());
            void* peer_base;

            zeCheck(zeMemOpenIpcHandle(
                    l0_ctx, l0_device, peer->ipc_handle, ZE_IPC_MEMORY_FLAG_BIAS_CACHED, &peer_base));
            buffer[i] = (char*)peer_base + peer->offset;
            sync_buffer[i] = (char*)peer_base + peer->offset + world * max_buffer * 1024 + rank * 128;
            ready_buffer[i] = (char*)peer_base + peer->offset + world * max_buffer * 1024 + rank * 128 + 64;
            offset[i] = peer->offset;
            ipc_handle[i] = send_buf.ipc_handle;
        }    
    }
    
    bool initialized;
    void* buffer[max_rank];
    void* local_sync_buffer;
    void* sync_buffer[max_rank];
    void* ready_buffer[max_rank];
    size_t offset[max_rank];
    ze_ipc_mem_handle_t ipc_handle[max_rank];
    int rank, world;
    int disable_check = 0;
    int disable_work = 0;
    int disable_sync = 0;

};
// template <typename data_type>
// inline data_type *alloc_device_and_init(size_t size,
//         std::function<void(data_type *data, size_t elements)> init_func,
//         sycl::queue &queue, sycl::device &device, sycl::context &context) {
//     auto host_ptr = static_cast<data_type *>(malloc(size * sizeof(data_type)));

//     for (size_t i = 0; i < size; ++i) {
//         init_func(host_ptr, i);
//     }

//     auto device_ptr = static_cast<data_type *>(aligned_alloc_device(
//             DEVICE_MEM_ALIGNMENT, size * sizeof(data_type), device, context));

//     queue.memcpy((void *)device_ptr, (void *)host_ptr, size * sizeof(data_type))
//             .wait();

//     free(host_ptr);

//     return device_ptr;
// }

// template <typename data_type>
// inline data_type *alloc_host_and_copy(
//         data_type *device_ptr, size_t size, sycl::queue &queue) {
//     auto host_ptr = static_cast<data_type *>(malloc(size * sizeof(data_type)));

//     queue.memcpy(host_ptr, device_ptr, size * sizeof(data_type)).wait();
//     return host_ptr;
// }