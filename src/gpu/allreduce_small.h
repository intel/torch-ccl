#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <system_error>

#include <mpi.h>
#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>
#include <ext/intel/esimd.hpp>

#include "cxxopts.hpp"
#include "ze_exception.hpp"
//#include "sycl_misc.hpp"
#include "allreduce.h"

#define SIMD 128
#define SMALL_SIMD_ATOMIC 16
//#define MAX_RANK 8
#define UNROLL_COUNT MAX_RANK
#define UNROLL_SIZE 2
#define TRIPLE_BUFFER 3
#define SMALL_SYNC_BYTE (SMALL_SIMD_ATOMIC * sizeof(int) * 2)
#define ALIGNMENT_BYTE 224
#define SMALL_MAX_COUNT (EU_COUNT_PER_RANK*1024)
#define LOOP_COUNT_LIMIT (1000000)
#define DEBUG_DATA_SIZE 16
#define DEBUG_THREAD_COUNT 2
#define DEBUG_DUMP_TO_DEDICATED_OFFSET 1
#define DEBUG 0

const int kernel_inner_loop = 1;

/*
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
*/

template <typename data_type, uint32_t max_rank = 8, uint32_t max_buffer = 1024 /*KB*/>
class allreducer_small {
public:
    allreducer_small(){
        initialized = false;
        buffer_index = 0;
        size_per_buffer = 0;
    }

    void init(sycl::queue& queue, uint32_t rank_in, uint32_t world_in){
        using namespace __ESIMD_NS;
        using namespace __ESIMD_ENS;
        rank = rank_in;
        world = world_in;
        // temporal buffer used for allreduce temporal use only.
        data_size_per_buffer = ((SMALL_MAX_COUNT + SIMD * UNROLL_SIZE * kernel_inner_loop - 1) / (SIMD * UNROLL_SIZE * kernel_inner_loop)) * SIMD * UNROLL_SIZE * kernel_inner_loop;
        data_size_per_buffer = ((data_size_per_buffer * sizeof(data_type) + ALIGNMENT_BYTE - 1) / ALIGNMENT_BYTE) * ALIGNMENT_BYTE / sizeof(data_type); //aligned size
        size_per_buffer = data_size_per_buffer * sizeof(data_type) + SMALL_SYNC_BYTE;
        int size_per_buffer_kernel = size_per_buffer;
        //printf("DEBUG rank%d: init-malloc_shared\n", rank);
        void* local_triple_buffer = sycl::malloc_device(size_per_buffer * TRIPLE_BUFFER, queue);

        uint32_t total_threads_needed = (SMALL_SYNC_BYTE /sizeof(data_type) + SIMD - 1) / SIMD;
        int wg_size = 1;
        uint32_t buffer_offset_to_sync = data_size_per_buffer;
        //printf("DEBUG rank%d: init-kernel\n", rank);
        sycl::event e;
        //initialize the sync buffers to 0.
        //format of the triple buffer: count_sync_count_sync_count_sync
        //There are three sync buffers in triple buffer.
        e = queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<class copy_kernel2>(
                sycl::nd_range<1>({ total_threads_needed }, wg_size), [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL{
                  int idx = item.get_global_id(0);
                  simd<data_type, SIMD> grf; //4 registers allocated.
                  uint32_t index = idx * SIMD;

                  // init buffer
                  grf = 0;
                  data_type * ptr = (data_type*)local_triple_buffer;
                  //init the sync buffer in triple buffer
                  lsc_block_store<data_type, SIMD, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                      (ptr + index + buffer_offset_to_sync, grf);
                  lsc_block_store<data_type, SIMD, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                      (ptr + index + buffer_offset_to_sync + size_per_buffer_kernel / sizeof(data_type), grf);
                  lsc_block_store<data_type, SIMD, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                      (ptr + index + buffer_offset_to_sync + 2 * size_per_buffer_kernel / sizeof(data_type), grf);
                  fence<memory_kind::global, fence_flush_op::none, fence_scope::system>();
                });
        });
        e.wait();

        // XXX: gain access to remote pointers
        //printf("DEBUG rank%d: init-exchange_peer_ipc_mem\n", rank);
        exchange_peer_ipc_mem(queue, local_triple_buffer);
        initialized = true;
        //printf("DEBUG rank%d: init-end\n", rank);

    }
    void allreduce(sycl::queue& queue, void* inout_buffer, uint32_t size){
        using namespace __ESIMD_NS;
        using namespace __ESIMD_ENS;

        //gpu_timer<1> gtimer;
        //cpu_timer<1> ctimer;
        //sycl::event e;
        //ctimer.start(0);
        buffer_index = (buffer_index + 1) % 3;

        assert(initialized == true);        
        void* temp_buffer[max_rank];
        for (uint32_t i = 0; i < world; i++){
            temp_buffer[i] = buffer[i];
            //printf("DEBUG rank%dr%d temp_buffer base addr=0x%x%x\n", rank, i, (((uint64_t)temp_buffer[i])>>32), (uint64_t)temp_buffer[i]&0xffffffff);
        }
        void* temp_sync_buffer[max_rank];
        for (uint32_t i = 0; i < world; i++){
            temp_sync_buffer[i] = sync_buffer[i];
            //printf("DEBUG rank%dr%d temp_sync_buffer base addr=0x%x%x\n", rank, i, (((uint64_t)temp_sync_buffer[i]) >> 32), (uint64_t)temp_sync_buffer[i] & 0xffffffff);
        }
        uint32_t temp_rank = rank;
        uint32_t temp_world = world;
        uint32_t total_threads_needed = (size + SIMD * UNROLL_SIZE * kernel_inner_loop - 1) / (SIMD * UNROLL_SIZE * kernel_inner_loop); //ceiling
        //printf("rank%d iter%d required gpu hw thread count = %d\n", rank, index_to_triple_buffer, total_threads_needed);
        int wg_size = 1;
        int size_per_buffer_kernel = size_per_buffer;
        uint32_t wait = total_threads_needed * 15 / 2048 + 1;
        int data_size_per_buffer_kernel = data_size_per_buffer;
        int buffer_index_kernel = buffer_index; //index to the triple temp buffer
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<class Allreduce_kernel>(
                sycl::nd_range<1>({ total_threads_needed }, wg_size), [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                    int idx = item.get_global_id(0);
                    /////////////////////////////////////////////////////////////////////////////////
                    //ESIMD kernel
                    uint offset = idx * SIMD * UNROLL_SIZE * kernel_inner_loop;
                    simd<data_type, max_rank * SIMD * UNROLL_SIZE> buffer; //64 registers
                    simd<ushort, SMALL_SIMD_ATOMIC> ramp;
#if DEBUG
                    simd<int, DEBUG_DATA_SIZE> debug = -99;
#endif

                    //to do:
                    //O3 compiler optimization: not much difference after the change.
                    //tune the fence: good perf improvements
                    //tune the cacheability for each IO message: no noticeable improvement
                    //tune the thread size: not much improvements
                    //tune the polling freq
#pragma unroll
                    for (uint32_t i = 0; i < SMALL_SIMD_ATOMIC; i++)
                    {
                        ramp[i] = i * sizeof(int);
                    }

                    //do copy from input buffer to temp buffer.
                    for (int i = 0; i < kernel_inner_loop; i++) {
#pragma unroll
                        for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++) {
                            buffer.template select<SIMD, 1>(unroll_i * SIMD) = lsc_block_load<data_type, SIMD, lsc_data_size::default_size, cache_hint::cached, cache_hint::cached>
                                ((data_type *)inout_buffer + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE);
                        }

                        //use the temp buffer for the current rank to copy the data to.
                        data_type * local_temp_ptr = (data_type*)temp_buffer[temp_rank];
                        local_temp_ptr += (buffer_index_kernel * size_per_buffer_kernel / sizeof(data_type)); //point to the correct buffer inside the triple buffer

#pragma unroll
                        for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++) {
                            lsc_block_store<data_type, SIMD, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                                ((data_type *)local_temp_ptr + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE, buffer.template select<SIMD, 1>(unroll_i * SIMD));
                        }
                    }
                    fence<memory_kind::global, fence_flush_op::none, fence_scope::gpus>();

                    //since each threads are copying small chunks of data to temp buffer, all the threads needs to sync globally using atomics within this rank
                    simd_mask<SMALL_SIMD_ATOMIC> pred;
                    simd<int, SMALL_SIMD_ATOMIC>status0;
                    pred = false;
                    pred[0] = true;
                    //pred[14] = pred[15] = false;

                    //sync locally within local GPU first.
                    int * local_sync_ptr = (int*)temp_sync_buffer[temp_rank]; //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                    local_sync_ptr += (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                    status0 = lsc_atomic_update<atomic_op::inc, int, SMALL_SIMD_ATOMIC, lsc_data_size::default_size, cache_hint::none, cache_hint::none>
                        (local_sync_ptr, ramp, pred);

                    //wait for all the local TG to sync. Then sync the other remote GPUs
                    {
#if DEBUG
                        int loop_counter = 0;
                        debug[0] = 1111;
                        debug[1] = loop_counter;
                        debug[2] = temp_rank;
                        debug[3] = buffer_index_kernel;
#endif
                        while (status0[0] != total_threads_needed)
                        {
                            volatile int i = 0;
                            while (i++ < wait);
                            status0 = lsc_atomic_update<atomic_op::load, int, SMALL_SIMD_ATOMIC, lsc_data_size::default_size, cache_hint::none, cache_hint::none>
                                (local_sync_ptr, ramp, pred);

#if DEBUG
                            if (loop_counter > LOOP_COUNT_LIMIT)
                                break;
                            else
                                loop_counter++;
                            debug[0] = 1111;
                            debug[1] = loop_counter;
                            debug[2] = temp_rank;
                            debug[3] = buffer_index_kernel;
#endif

                        }
                    }

                    //once the local level sync is done, atomically write its counter to other remote gpus' atomic counter
                    pred = false;
                    pred[1] = true; //use different lane for the remote gpu sync
                    if (idx == 0) //one thread in the local gpu notifies the remote gpu of its status.
                    {
                        status0 = total_threads_needed;
                        for (int i = 0; i < temp_world; i++)
                        {
                            int * sync_ptr = (int*)temp_sync_buffer[i]; //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                            sync_ptr += (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                            lsc_atomic_update<atomic_op::add, int, SMALL_SIMD_ATOMIC, lsc_data_size::default_size, cache_hint::none, cache_hint::none>
                                (sync_ptr, ramp, status0, pred);
                        }
                    }

                    //once all the local TGs are sync, do fence so that other GPU can see.
                    //lsc_fence<lsc_memory_kind::untyped_global, lsc_fence_op::none, lsc_scope::gpus>();

                    //wait for completion of the atomic sync
                    status0 = lsc_atomic_update<atomic_op::load, int, SMALL_SIMD_ATOMIC, lsc_data_size::default_size, cache_hint::none, cache_hint::none>
                        (local_sync_ptr, ramp, pred);
#if DEBUG
                    int loop_counter = 0;

                    debug[4] = 2222;
                    debug[5] = loop_counter;
                    debug[6] = temp_rank;
                    debug[7] = buffer_index_kernel;
#endif
                    while (status0[1] != total_threads_needed * temp_world)
                    {
                        status0 = lsc_atomic_update<atomic_op::load, int, SMALL_SIMD_ATOMIC, lsc_data_size::default_size, cache_hint::none, cache_hint::none>
                            (local_sync_ptr, ramp, pred);
#if DEBUG

                        if (loop_counter > LOOP_COUNT_LIMIT)
                            break;
                        else
                            loop_counter++;

                        debug[4] = 2222;
                        debug[5] = loop_counter;
                        debug[6] = temp_rank;
                        debug[7] = buffer_index_kernel;
#endif
                    }

                    //reset the sync counter for the next allreduce session. Each rank reset's its own buffer
                    if (idx == 0) //one thread in the local gpu notifies the remote gpu of its status.
                    {
                        int buffer_index_to_reset = (buffer_index_kernel + 2) % 3;
                        status0 = 0;
                        pred = true;
                        local_sync_ptr = (int*)temp_sync_buffer[temp_rank]; //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                        local_sync_ptr += (buffer_index_to_reset * size_per_buffer_kernel / sizeof(int));
                        lsc_atomic_update<atomic_op::store, int, SMALL_SIMD_ATOMIC, lsc_data_size::default_size, cache_hint::none, cache_hint::none>
                            (local_sync_ptr, ramp, status0, pred); //reset the first half of sync buffer
                    }

                    //at this point, all the threads are done copying data from input buffer to temp buffer.
                    //do All reduce
                    simd<data_type, SIMD * UNROLL_SIZE> result;
                    for (int i = 0; i < kernel_inner_loop; i++)
                    {
                        if (temp_world == max_rank)
                        {
                            int * peer_ptr0 = ((int*)temp_buffer[0]) + (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                            int * peer_ptr1 = ((int*)temp_buffer[1]) + (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                            int * peer_ptr2 = ((int*)temp_buffer[2]) + (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                            int * peer_ptr3 = ((int*)temp_buffer[3]) + (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                            int * peer_ptr4 = ((int*)temp_buffer[4]) + (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                            int * peer_ptr5 = ((int*)temp_buffer[5]) + (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                            int * peer_ptr6 = ((int*)temp_buffer[6]) + (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                            int * peer_ptr7 = ((int*)temp_buffer[7]) + (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));

#pragma unroll
                            for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++) 
                            {
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 0 * SIMD * UNROLL_SIZE) = lsc_block_load<data_type, SIMD, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                                    ((data_type *)peer_ptr0 + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE);
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 1 * SIMD * UNROLL_SIZE) = lsc_block_load<data_type, SIMD, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                                    ((data_type *)peer_ptr1 + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE);
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 2 * SIMD * UNROLL_SIZE) = lsc_block_load<data_type, SIMD, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                                    ((data_type *)peer_ptr2 + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE);
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 3 * SIMD * UNROLL_SIZE) = lsc_block_load<data_type, SIMD, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                                    ((data_type *)peer_ptr3 + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE);
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 4 * SIMD * UNROLL_SIZE) = lsc_block_load<data_type, SIMD, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                                    ((data_type *)peer_ptr4 + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE);
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 5 * SIMD * UNROLL_SIZE) = lsc_block_load<data_type, SIMD, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                                    ((data_type *)peer_ptr5 + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE);
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 6 * SIMD * UNROLL_SIZE) = lsc_block_load<data_type, SIMD, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                                    ((data_type *)peer_ptr6 + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE);
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 7 * SIMD * UNROLL_SIZE) = lsc_block_load<data_type, SIMD, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                                    ((data_type *)peer_ptr7 + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE);
                            }
                            //do the actual reduction
                            result = 0;
#pragma unroll
                            for (int r = 0; r < max_rank; r++)
                            {
                                //result += buffer.template select<SIMD * UNROLL_SIZE, 1>(r * SIMD * UNROLL_SIZE);
                                result = result + buffer.template select<SIMD * UNROLL_SIZE, 1>(r * SIMD * UNROLL_SIZE);
                            }

                        }
                        else
                        {
                            for (int r = 0; r < temp_world; r++)
                            {
                                int * peer_ptr = ((int*)temp_buffer[r]) + (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
#pragma unroll
                                for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++)
                                {
                                    buffer.template select<SIMD, 1>(unroll_i * SIMD + r * SIMD * UNROLL_SIZE) = lsc_block_load<data_type, SIMD, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                                        ((data_type *)peer_ptr + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE);

                                }
                            }
                            //do the actual reduction
                            result = 0;
                            for (int r = 0; r < temp_world; r++)
                            {
                                //result += buffer.template select<SIMD * UNROLL_SIZE, 1>(r * SIMD * UNROLL_SIZE);
                                result = result + buffer.template select<SIMD * UNROLL_SIZE, 1>(r * SIMD * UNROLL_SIZE);
                            }
                        }

                        //write out the results
#pragma unroll
                        for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++) {
                            lsc_block_store<data_type, SIMD, lsc_data_size::default_size, cache_hint::write_back, cache_hint::write_back>
                                ((data_type *)inout_buffer + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE, result.template select<SIMD, 1>(unroll_i * SIMD));
                        }

  
#if DEBUG
                        //write out the debug dmp
                        if (idx == 0 || idx == 1)
                        {
#if DEBUG_DUMP_TO_DEDICATED_OFFSET
                            lsc_block_store<int, DEBUG_DATA_SIZE, lsc_data_size::default_size, cache_hint::uncached, cache_hint::write_back>
                                ((int *)inout_buffer + (SIMD * UNROLL_SIZE) * total_threads_needed / (sizeof(int) / sizeof(data_type)) + idx * DEBUG_DATA_SIZE, debug.template select<DEBUG_DATA_SIZE, 1>(0));
#else
                            lsc_block_store<int, DEBUG_DATA_SIZE, lsc_data_size::default_size, cache_hint::uncached, cache_hint::write_back>
                                ((int *)inout_buffer + offset / (sizeof(int) / sizeof(sycl::half)), debug.template select<DEBUG_DATA_SIZE, 1>(0));
#endif
                        }
                        //lsc_fence<lsc_memory_kind::untyped_global, lsc_fence_op::none, lsc_scope::system>();
#endif                        
                    }
                    //14us-29ms upto here.

                });
        });
        //e.wait();
#if DEBUG
        //printf("rank%d iter%d required gpu hw thread count = %d. DONE\n", rank, index_to_triple_buffer, total_threads_needed);
#endif
        //gtimer.record(0, e);
        //ctimer.stop(0);

        //order the prints by delaying by rank ID
        //for (int i = 0; i < temp_rank * 1024 * 128; i++)
        //{
        //    if (i * temp_rank == 0x7fffffff)
        //        printf(" ");
        //}
        //std::cout << "rank" << temp_rank << " iter" << index_to_triple_buffer << ": kernel us= " << gtimer.get_us(0);
        //std::cout << " host us= " << ctimer.get_us(0) << "\n";

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

    void work_only(sycl::queue& queue, void* inout_buffer, uint32_t size){
        using namespace __ESIMD_NS;
        using namespace __ESIMD_ENS;

        //gpu_timer<1> gtimer;
        //cpu_timer<1> ctimer;
        //sycl::event e;
        //ctimer.start(0);
        buffer_index = (buffer_index + 1) % 3;

        assert(initialized == true);        
        void* temp_buffer[max_rank];
        for (uint32_t i = 0; i < world; i++){
            temp_buffer[i] = buffer[i];
            //printf("DEBUG rank%dr%d temp_buffer base addr=0x%x%x\n", rank, i, (((uint64_t)temp_buffer[i])>>32), (uint64_t)temp_buffer[i]&0xffffffff);
        }
        void* temp_sync_buffer[max_rank];
        for (uint32_t i = 0; i < world; i++){
            temp_sync_buffer[i] = sync_buffer[i];
            //printf("DEBUG rank%dr%d temp_sync_buffer base addr=0x%x%x\n", rank, i, (((uint64_t)temp_sync_buffer[i]) >> 32), (uint64_t)temp_sync_buffer[i] & 0xffffffff);
        }
        uint32_t temp_rank = rank;
        uint32_t temp_world = world;
        uint32_t total_threads_needed = (size + SIMD * UNROLL_SIZE * kernel_inner_loop - 1) / (SIMD * UNROLL_SIZE * kernel_inner_loop); //ceiling
        //printf("rank%d iter%d required gpu hw thread count = %d\n", rank, index_to_triple_buffer, total_threads_needed);
        int wg_size = 1;
        int size_per_buffer_kernel = size_per_buffer;
        int data_size_per_buffer_kernel = data_size_per_buffer;
        int buffer_index_kernel = buffer_index; //index to the triple temp buffer
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<class Workonly_kernel>(
                sycl::nd_range<1>({ total_threads_needed }, wg_size), [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                    int idx = item.get_global_id(0);
                    /////////////////////////////////////////////////////////////////////////////////
                    //ESIMD kernel
                    uint offset = idx * SIMD * UNROLL_SIZE * kernel_inner_loop;
                    simd<data_type, max_rank * SIMD * UNROLL_SIZE> buffer; //64 registers
                    simd<ushort, SMALL_SIMD_ATOMIC> ramp;
#if DEBUG
                    simd<int, DEBUG_DATA_SIZE> debug = -99;
#endif

                    //to do:
                    //O3 compiler optimization: not much difference after the change.
                    //tune the fence: good perf improvements
                    //tune the cacheability for each IO message: no noticeable improvement
                    //tune the thread size: not much improvements
                    //tune the polling freq
#pragma unroll
                    for (uint32_t i = 0; i < SMALL_SIMD_ATOMIC; i++)
                    {
                        ramp[i] = i * sizeof(int);
                    }

                    //do copy from input buffer to temp buffer.
                    for (int i = 0; i < kernel_inner_loop; i++) {
#pragma unroll
                        for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++) {
                            buffer.template select<SIMD, 1>(unroll_i * SIMD) = lsc_block_load<data_type, SIMD, lsc_data_size::default_size, cache_hint::cached, cache_hint::cached>
                                ((data_type *)inout_buffer + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE);
                        }

                        //use the temp buffer for the current rank to copy the data to.
                        data_type * local_temp_ptr = (data_type*)temp_buffer[temp_rank];
                        local_temp_ptr += (buffer_index_kernel * size_per_buffer_kernel / sizeof(data_type)); //point to the correct buffer inside the triple buffer

#pragma unroll
                        for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++) {
                            lsc_block_store<data_type, SIMD, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                                ((data_type *)local_temp_ptr + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE, buffer.template select<SIMD, 1>(unroll_i * SIMD));
                        }
                    }
                    fence<memory_kind::global, fence_flush_op::none, fence_scope::gpus>();

                    //since each threads are copying small chunks of data to temp buffer, all the threads needs to sync globally using atomics within this rank
                    simd_mask<SMALL_SIMD_ATOMIC> pred;
                    simd<int, SMALL_SIMD_ATOMIC>status0;
                    pred = false;
                    pred[0] = true;
                    //pred[14] = pred[15] = false;

                    //sync locally within local GPU first.
                    int * local_sync_ptr = (int*)temp_sync_buffer[temp_rank]; //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                    local_sync_ptr += (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                    status0 = lsc_atomic_update<atomic_op::inc, int, SMALL_SIMD_ATOMIC, lsc_data_size::default_size, cache_hint::none, cache_hint::none>
                        (local_sync_ptr, ramp, pred);

                    //wait for all the local TG to sync. Then sync the other remote GPUs
                    {
#if DEBUG
                        int loop_counter = 0;
                        debug[0] = 1111;
                        debug[1] = loop_counter;
                        debug[2] = temp_rank;
                        debug[3] = buffer_index_kernel;
#endif
                        while (status0[0] != total_threads_needed)
                        {
                            status0 = lsc_atomic_update<atomic_op::load, int, SMALL_SIMD_ATOMIC, lsc_data_size::default_size, cache_hint::none, cache_hint::none>
                                (local_sync_ptr, ramp, pred);

#if DEBUG
                            if (loop_counter > LOOP_COUNT_LIMIT)
                                break;
                            else
                                loop_counter++;
                            debug[0] = 1111;
                            debug[1] = loop_counter;
                            debug[2] = temp_rank;
                            debug[3] = buffer_index_kernel;
#endif

                        }
                    }

                    /*
                    //once the local level sync is done, atomically write its counter to other remote gpus' atomic counter
                    pred = false;
                    pred[1] = true; //use different lane for the remote gpu sync
                    if (idx == 0) //one thread in the local gpu notifies the remote gpu of its status.
                    {
                        status0 = total_threads_needed;
                        for (int i = 0; i < temp_world; i++)
                        {
                            int * sync_ptr = (int*)temp_sync_buffer[i]; //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                            sync_ptr += (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                            lsc_atomic_update<atomic_op::add, int, SIMD_ATOMIC, lsc_data_size::default_size, cache_hint::none, cache_hint::none>
                                (sync_ptr, ramp, status0, pred);
                        }
                    }

                    //once all the local TGs are sync, do fence so that other GPU can see.
                    //lsc_fence<lsc_memory_kind::untyped_global, lsc_fence_op::none, lsc_scope::gpus>();

                    //wait for completion of the atomic sync
                    status0 = lsc_atomic_update<atomic_op::load, int, SIMD_ATOMIC, lsc_data_size::default_size, cache_hint::none, cache_hint::none>
                        (local_sync_ptr, ramp, pred);
#if DEBUG
                    int loop_counter = 0;

                    debug[4] = 2222;
                    debug[5] = loop_counter;
                    debug[6] = temp_rank;
                    debug[7] = buffer_index_kernel;
#endif
                    while (status0[1] != total_threads_needed * temp_world)
                    {
                        status0 = lsc_atomic_update<atomic_op::load, int, SIMD_ATOMIC, lsc_data_size::default_size, cache_hint::none, cache_hint::none>
                            (local_sync_ptr, ramp, pred);
#if DEBUG

                        if (loop_counter > LOOP_COUNT_LIMIT)
                            break;
                        else
                            loop_counter++;

                        debug[4] = 2222;
                        debug[5] = loop_counter;
                        debug[6] = temp_rank;
                        debug[7] = buffer_index_kernel;
#endif
                    }
                    */

                    //reset the sync counter for the next allreduce session. Each rank reset's its own buffer
                    if (idx == 0) //one thread in the local gpu notifies the remote gpu of its status.
                    {
                        int buffer_index_to_reset = (buffer_index_kernel + 2) % 3;
                        status0 = 0;
                        pred = true;
                        local_sync_ptr = (int*)temp_sync_buffer[temp_rank]; //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                        local_sync_ptr += (buffer_index_to_reset * size_per_buffer_kernel / sizeof(int));
                        lsc_atomic_update<atomic_op::store, int, SMALL_SIMD_ATOMIC, lsc_data_size::default_size, cache_hint::none, cache_hint::none>
                            (local_sync_ptr, ramp, status0, pred); //reset the first half of sync buffer
                    }
                    
                    //at this point, all the threads are done copying data from input buffer to temp buffer.
                    //do All reduce
                    simd<data_type, SIMD * UNROLL_SIZE> result;
                    for (int i = 0; i < kernel_inner_loop; i++)
                    {
                        if (temp_world == max_rank)
                        {
                            int * peer_ptr0 = ((int*)temp_buffer[0]) + (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                            int * peer_ptr1 = ((int*)temp_buffer[1]) + (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                            int * peer_ptr2 = ((int*)temp_buffer[2]) + (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                            int * peer_ptr3 = ((int*)temp_buffer[3]) + (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                            int * peer_ptr4 = ((int*)temp_buffer[4]) + (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                            int * peer_ptr5 = ((int*)temp_buffer[5]) + (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                            int * peer_ptr6 = ((int*)temp_buffer[6]) + (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                            int * peer_ptr7 = ((int*)temp_buffer[7]) + (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));

#pragma unroll
                            for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++) 
                            {
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 0 * SIMD * UNROLL_SIZE) = lsc_block_load<data_type, SIMD, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                                    ((data_type *)peer_ptr0 + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE);
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 1 * SIMD * UNROLL_SIZE) = lsc_block_load<data_type, SIMD, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                                    ((data_type *)peer_ptr1 + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE);
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 2 * SIMD * UNROLL_SIZE) = lsc_block_load<data_type, SIMD, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                                    ((data_type *)peer_ptr2 + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE);
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 3 * SIMD * UNROLL_SIZE) = lsc_block_load<data_type, SIMD, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                                    ((data_type *)peer_ptr3 + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE);
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 4 * SIMD * UNROLL_SIZE) = lsc_block_load<data_type, SIMD, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                                    ((data_type *)peer_ptr4 + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE);
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 5 * SIMD * UNROLL_SIZE) = lsc_block_load<data_type, SIMD, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                                    ((data_type *)peer_ptr5 + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE);
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 6 * SIMD * UNROLL_SIZE) = lsc_block_load<data_type, SIMD, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                                    ((data_type *)peer_ptr6 + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE);
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 7 * SIMD * UNROLL_SIZE) = lsc_block_load<data_type, SIMD, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                                    ((data_type *)peer_ptr7 + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE);
                            }
                            //do the actual reduction
                            result = 0;
#pragma unroll
                            for (int r = 0; r < max_rank; r++)
                            {
                                //result += buffer.template select<SIMD * UNROLL_SIZE, 1>(r * SIMD * UNROLL_SIZE);
                                result = result + buffer.template select<SIMD * UNROLL_SIZE, 1>(r * SIMD * UNROLL_SIZE);
                            }

                        }
                        else
                        {
                            for (int r = 0; r < temp_world; r++)
                            {
                                int * peer_ptr = ((int*)temp_buffer[r]) + (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
#pragma unroll
                                for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++)
                                {
                                    buffer.template select<SIMD, 1>(unroll_i * SIMD + r * SIMD * UNROLL_SIZE) = lsc_block_load<data_type, SIMD, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                                        ((data_type *)peer_ptr + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE);

                                }
                            }
                            //do the actual reduction
                            result = 0;
                            for (int r = 0; r < temp_world; r++)
                            {
                                //result += buffer.template select<SIMD * UNROLL_SIZE, 1>(r * SIMD * UNROLL_SIZE);
                                result = result + buffer.template select<SIMD * UNROLL_SIZE, 1>(r * SIMD * UNROLL_SIZE);
                            }
                        }

                        //write out the results
#pragma unroll
                        for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++) {
                            lsc_block_store<data_type, SIMD, lsc_data_size::default_size, cache_hint::write_back, cache_hint::write_back>
                                ((data_type *)inout_buffer + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE, result.template select<SIMD, 1>(unroll_i * SIMD));
                        }

  
#if DEBUG
                        //write out the debug dmp
                        if (idx == 0 || idx == 1)
                        {
#if DEBUG_DUMP_TO_DEDICATED_OFFSET
                            lsc_block_store<int, DEBUG_DATA_SIZE, lsc_data_size::default_size, cache_hint::uncached, cache_hint::write_back>
                                ((int *)inout_buffer + (SIMD * UNROLL_SIZE) * total_threads_needed / (sizeof(int) / sizeof(data_type)) + idx * DEBUG_DATA_SIZE, debug.template select<DEBUG_DATA_SIZE, 1>(0));
#else
                            lsc_block_store<int, DEBUG_DATA_SIZE, lsc_data_size::default_size, cache_hint::uncached, cache_hint::write_back>
                                ((int *)inout_buffer + offset / (sizeof(int) / sizeof(sycl::half)), debug.template select<DEBUG_DATA_SIZE, 1>(0));
#endif
                        }
                        //lsc_fence<lsc_memory_kind::untyped_global, lsc_fence_op::none, lsc_scope::system>();
#endif                        
                    }
                    //14us-29ms upto here.

                });
        });
        //e.wait();
#if DEBUG
        //printf("rank%d iter%d required gpu hw thread count = %d. DONE\n", rank, index_to_triple_buffer, total_threads_needed);
#endif
        //gtimer.record(0, e);
        //ctimer.stop(0);

        //order the prints by delaying by rank ID
        //for (int i = 0; i < temp_rank * 1024 * 128; i++)
        //{
        //    if (i * temp_rank == 0x7fffffff)
        //        printf(" ");
        //}
        //std::cout << "rank" << temp_rank << " iter" << index_to_triple_buffer << ": kernel us= " << gtimer.get_us(0);
        //std::cout << " host us= " << ctimer.get_us(0) << "\n";

    }

    void sync_only(sycl::queue& queue, void* inout_buffer, uint32_t size){
        using namespace __ESIMD_NS;
        using namespace __ESIMD_ENS;

        //gpu_timer<1> gtimer;
        //cpu_timer<1> ctimer;
        //sycl::event e;
        //ctimer.start(0);
        buffer_index = (buffer_index + 1) % 3;

        assert(initialized == true);        
        void* temp_buffer[max_rank];
        for (uint32_t i = 0; i < world; i++){
            temp_buffer[i] = buffer[i];
            //printf("DEBUG rank%dr%d temp_buffer base addr=0x%x%x\n", rank, i, (((uint64_t)temp_buffer[i])>>32), (uint64_t)temp_buffer[i]&0xffffffff);
        }
        void* temp_sync_buffer[max_rank];
        for (uint32_t i = 0; i < world; i++){
            temp_sync_buffer[i] = sync_buffer[i];
            //printf("DEBUG rank%dr%d temp_sync_buffer base addr=0x%x%x\n", rank, i, (((uint64_t)temp_sync_buffer[i]) >> 32), (uint64_t)temp_sync_buffer[i] & 0xffffffff);
        }
        uint32_t temp_rank = rank;
        uint32_t temp_world = world;
        uint32_t total_threads_needed = (size + SIMD * UNROLL_SIZE * kernel_inner_loop - 1) / (SIMD * UNROLL_SIZE * kernel_inner_loop); //ceiling
        //printf("rank%d iter%d required gpu hw thread count = %d\n", rank, index_to_triple_buffer, total_threads_needed);
        int wg_size = 1;
        int size_per_buffer_kernel = size_per_buffer;
        int data_size_per_buffer_kernel = data_size_per_buffer;
        int buffer_index_kernel = buffer_index; //index to the triple temp buffer
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<class Synconly_kernel>(
                sycl::nd_range<1>({ total_threads_needed }, wg_size), [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                    int idx = item.get_global_id(0);
                    /////////////////////////////////////////////////////////////////////////////////
                    //ESIMD kernel
                    uint offset = idx * SIMD * UNROLL_SIZE * kernel_inner_loop;
                    simd<data_type, max_rank * SIMD * UNROLL_SIZE> buffer; //64 registers
                    simd<ushort, SMALL_SIMD_ATOMIC> ramp;
#if DEBUG
                    simd<int, DEBUG_DATA_SIZE> debug = -99;
#endif

                    //to do:
                    //O3 compiler optimization: not much difference after the change.
                    //tune the fence: good perf improvements
                    //tune the cacheability for each IO message: no noticeable improvement
                    //tune the thread size: not much improvements
                    //tune the polling freq
#pragma unroll
                    for (uint32_t i = 0; i < SMALL_SIMD_ATOMIC; i++)
                    {
                        ramp[i] = i * sizeof(int);
                    }

                    /*
                    //do copy from input buffer to temp buffer.
                    for (int i = 0; i < kernel_inner_loop; i++) {
#pragma unroll
                        for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++) {
                            buffer.template select<SIMD, 1>(unroll_i * SIMD) = lsc_block_load<data_type, SIMD, lsc_data_size::default_size, cache_hint::cached, cache_hint::cached>
                                ((data_type *)inout_buffer + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE);
                        }

                        //use the temp buffer for the current rank to copy the data to.
                        data_type * local_temp_ptr = (data_type*)temp_buffer[temp_rank];
                        local_temp_ptr += (buffer_index_kernel * size_per_buffer_kernel / sizeof(data_type)); //point to the correct buffer inside the triple buffer

#pragma unroll
                        for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++) {
                            lsc_block_store<data_type, SIMD, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                                ((data_type *)local_temp_ptr + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE, buffer.template select<SIMD, 1>(unroll_i * SIMD));
                        }
                    }
                    lsc_fence<lsc_memory_kind::untyped_global, lsc_fence_op::none, lsc_scope::gpus>();

                    //since each threads are copying small chunks of data to temp buffer, all the threads needs to sync globally using atomics within this rank
                    simd_mask<SIMD_ATOMIC> pred;
                    simd<int, SIMD_ATOMIC> status0;
                    pred = false;
                    pred[0] = true;
                    //pred[14] = pred[15] = false;

                    //sync locally within local GPU first.
                    int * local_sync_ptr = (int*)temp_sync_buffer[temp_rank]; //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                    local_sync_ptr += (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                    status0 = lsc_atomic_update<atomic_op::inc, int, SIMD_ATOMIC, lsc_data_size::default_size, cache_hint::none, cache_hint::none>
                        (local_sync_ptr, ramp, pred);

                    //wait for all the local TG to sync. Then sync the other remote GPUs
                    {
#if DEBUG
                        int loop_counter = 0;
                        debug[0] = 1111;
                        debug[1] = loop_counter;
                        debug[2] = temp_rank;
                        debug[3] = buffer_index_kernel;
#endif
                        while (status0[0] != total_threads_needed)
                        {
                            status0 = lsc_atomic_update<atomic_op::load, int, SIMD_ATOMIC, lsc_data_size::default_size, cache_hint::none, cache_hint::none>
                                (local_sync_ptr, ramp, pred);

#if DEBUG
                            if (loop_counter > LOOP_COUNT_LIMIT)
                                break;
                            else
                                loop_counter++;
                            debug[0] = 1111;
                            debug[1] = loop_counter;
                            debug[2] = temp_rank;
                            debug[3] = buffer_index_kernel;
#endif

                        }
                    }
                    */

                    simd_mask<SMALL_SIMD_ATOMIC> pred;
                    simd<int, SMALL_SIMD_ATOMIC> status0;

                    //once the local level sync is done, atomically write its counter to other remote gpus' atomic counter
                    pred = false;
                    pred[1] = true; //use different lane for the remote gpu sync
                    if (idx == 0) //one thread in the local gpu notifies the remote gpu of its status.
                    {
                        status0 = total_threads_needed;
                        for (int i = 0; i < temp_world; i++)
                        {
                            int * sync_ptr = (int*)temp_sync_buffer[i]; //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                            sync_ptr += (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                            lsc_atomic_update<atomic_op::add, int, SMALL_SIMD_ATOMIC, lsc_data_size::default_size, cache_hint::none, cache_hint::none>
                                (sync_ptr, ramp, status0, pred);
                        }
                    }

                    int * local_sync_ptr = (int*)temp_sync_buffer[temp_rank]; //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                    local_sync_ptr += (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));

                    //once all the local TGs are sync, do fence so that other GPU can see.
                    //lsc_fence<lsc_memory_kind::untyped_global, lsc_fence_op::none, lsc_scope::gpus>();

                    //wait for completion of the atomic sync
                    status0 = lsc_atomic_update<atomic_op::load, int, SMALL_SIMD_ATOMIC, lsc_data_size::default_size, cache_hint::none, cache_hint::none>
                        (local_sync_ptr, ramp, pred);
#if DEBUG
                    int loop_counter = 0;

                    debug[4] = 2222;
                    debug[5] = loop_counter;
                    debug[6] = temp_rank;
                    debug[7] = buffer_index_kernel;
#endif
                    while (status0[1] != total_threads_needed * temp_world)
                    {
                        status0 = lsc_atomic_update<atomic_op::load, int, SMALL_SIMD_ATOMIC, lsc_data_size::default_size, cache_hint::none, cache_hint::none>
                            (local_sync_ptr, ramp, pred);
#if DEBUG

                        if (loop_counter > LOOP_COUNT_LIMIT)
                            break;
                        else
                            loop_counter++;

                        debug[4] = 2222;
                        debug[5] = loop_counter;
                        debug[6] = temp_rank;
                        debug[7] = buffer_index_kernel;
#endif
                    }
                    

                    //reset the sync counter for the next allreduce session. Each rank reset's its own buffer
                    if (idx == 0) //one thread in the local gpu notifies the remote gpu of its status.
                    {
                        int buffer_index_to_reset = (buffer_index_kernel + 2) % 3;
                        status0 = 0;
                        pred = true;
                        local_sync_ptr = (int*)temp_sync_buffer[temp_rank]; //the buffer might be located in remote GPU. But during the atomics, local L2 should be utilized.
                        local_sync_ptr += (buffer_index_to_reset * size_per_buffer_kernel / sizeof(int));
                        lsc_atomic_update<atomic_op::store, int, SMALL_SIMD_ATOMIC, lsc_data_size::default_size, cache_hint::none, cache_hint::none>
                            (local_sync_ptr, ramp, status0, pred); //reset the first half of sync buffer
                    }

                    /*
                    //at this point, all the threads are done copying data from input buffer to temp buffer.
                    //do All reduce
                    simd<data_type, SIMD * UNROLL_SIZE> result;
                    for (int i = 0; i < kernel_inner_loop; i++)
                    {
                        if (temp_world == max_rank)
                        {
                            int * peer_ptr0 = ((int*)temp_buffer[0]) + (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                            int * peer_ptr1 = ((int*)temp_buffer[1]) + (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                            int * peer_ptr2 = ((int*)temp_buffer[2]) + (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                            int * peer_ptr3 = ((int*)temp_buffer[3]) + (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                            int * peer_ptr4 = ((int*)temp_buffer[4]) + (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                            int * peer_ptr5 = ((int*)temp_buffer[5]) + (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                            int * peer_ptr6 = ((int*)temp_buffer[6]) + (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
                            int * peer_ptr7 = ((int*)temp_buffer[7]) + (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));

#pragma unroll
                            for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++) 
                            {
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 0 * SIMD * UNROLL_SIZE) = lsc_block_load<data_type, SIMD, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                                    ((data_type *)peer_ptr0 + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE);
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 1 * SIMD * UNROLL_SIZE) = lsc_block_load<data_type, SIMD, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                                    ((data_type *)peer_ptr1 + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE);
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 2 * SIMD * UNROLL_SIZE) = lsc_block_load<data_type, SIMD, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                                    ((data_type *)peer_ptr2 + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE);
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 3 * SIMD * UNROLL_SIZE) = lsc_block_load<data_type, SIMD, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                                    ((data_type *)peer_ptr3 + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE);
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 4 * SIMD * UNROLL_SIZE) = lsc_block_load<data_type, SIMD, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                                    ((data_type *)peer_ptr4 + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE);
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 5 * SIMD * UNROLL_SIZE) = lsc_block_load<data_type, SIMD, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                                    ((data_type *)peer_ptr5 + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE);
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 6 * SIMD * UNROLL_SIZE) = lsc_block_load<data_type, SIMD, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                                    ((data_type *)peer_ptr6 + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE);
                                buffer.template select<SIMD, 1>(unroll_i * SIMD + 7 * SIMD * UNROLL_SIZE) = lsc_block_load<data_type, SIMD, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                                    ((data_type *)peer_ptr7 + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE);
                            }
                            //do the actual reduction
                            result = 0;
#pragma unroll
                            for (int r = 0; r < max_rank; r++)
                            {
                                //result += buffer.template select<SIMD * UNROLL_SIZE, 1>(r * SIMD * UNROLL_SIZE);
                                result = result + buffer.template select<SIMD * UNROLL_SIZE, 1>(r * SIMD * UNROLL_SIZE);
                            }

                        }
                        else
                        {
                            for (int r = 0; r < temp_world; r++)
                            {
                                int * peer_ptr = ((int*)temp_buffer[r]) + (buffer_index_kernel * size_per_buffer_kernel / sizeof(int));
#pragma unroll
                                for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++)
                                {
                                    buffer.template select<SIMD, 1>(unroll_i * SIMD + r * SIMD * UNROLL_SIZE) = lsc_block_load<data_type, SIMD, lsc_data_size::default_size, cache_hint::uncached, cache_hint::uncached>
                                        ((data_type *)peer_ptr + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE);

                                }
                            }
                            //do the actual reduction
                            result = 0;
                            for (int r = 0; r < temp_world; r++)
                            {
                                //result += buffer.template select<SIMD * UNROLL_SIZE, 1>(r * SIMD * UNROLL_SIZE);
                                result = result + buffer.template select<SIMD * UNROLL_SIZE, 1>(r * SIMD * UNROLL_SIZE);
                            }
                        }

                        //write out the results
#pragma unroll
                        for (int unroll_i = 0; unroll_i < UNROLL_SIZE; unroll_i++) {
                            lsc_block_store<data_type, SIMD, lsc_data_size::default_size, cache_hint::write_back, cache_hint::write_back>
                                ((data_type *)inout_buffer + offset + unroll_i * SIMD + i * SIMD * UNROLL_SIZE, result.template select<SIMD, 1>(unroll_i * SIMD));
                        }

  
#if DEBUG
                        //write out the debug dmp
                        if (idx == 0 || idx == 1)
                        {
#if DEBUG_DUMP_TO_DEDICATED_OFFSET
                            lsc_block_store<int, DEBUG_DATA_SIZE, lsc_data_size::default_size, cache_hint::uncached, cache_hint::write_back>
                                ((int *)inout_buffer + (SIMD * UNROLL_SIZE) * total_threads_needed / (sizeof(int) / sizeof(data_type)) + idx * DEBUG_DATA_SIZE, debug.template select<DEBUG_DATA_SIZE, 1>(0));
#else
                            lsc_block_store<int, DEBUG_DATA_SIZE, lsc_data_size::default_size, cache_hint::uncached, cache_hint::write_back>
                                ((int *)inout_buffer + offset / (sizeof(int) / sizeof(sycl::half)), debug.template select<DEBUG_DATA_SIZE, 1>(0));
#endif
                        }
                        //lsc_fence<lsc_memory_kind::untyped_global, lsc_fence_op::none, lsc_scope::system>();
#endif                        
                    }
                    //14us-29ms upto here.
                    */

                });
        });
        //e.wait();
#if DEBUG
        //printf("rank%d iter%d required gpu hw thread count = %d. DONE\n", rank, index_to_triple_buffer, total_threads_needed);
#endif
        //gtimer.record(0, e);
        //ctimer.stop(0);

        //order the prints by delaying by rank ID
        //for (int i = 0; i < temp_rank * 1024 * 128; i++)
        //{
        //    if (i * temp_rank == 0x7fffffff)
        //        printf(" ");
        //}
        //std::cout << "rank" << temp_rank << " iter" << index_to_triple_buffer << ": kernel us= " << gtimer.get_us(0);
        //std::cout << " host us= " << ctimer.get_us(0) << "\n";

    }


private:
    void exchange_peer_ipc_mem(sycl::queue& queue, void* ptr) {
        // Step 1: Get base address of the pointer
        sycl::context ctx = queue.get_context();
        auto l0_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);

        void *base_addr;
        size_t base_size;
        //printf("DEBUG rank%d: init-exchange_peer_ipc_mem-zeMemGetAddressRange\n", rank);
        zeCheck(zeMemGetAddressRange(l0_ctx, ptr, &base_addr, &base_size));

        // Step 2: Get IPC mem handle from base address
        alignas(64) exchange_contents send_buf;
        alignas(64) exchange_contents recv_buf[world];

        // fill in the exchange info
        // printf("DEBUG rank%d: init-exchange_peer_ipc_mem-zeMemGetIpcHandle\n", rank);
        zeCheck(zeMemGetIpcHandle(l0_ctx, base_addr, &send_buf.ipc_handle));
        send_buf.offset = (char*)ptr - (char*)base_addr;
        send_buf.pid = getpid();

        // Step 3: Exchange the handles and offsets
        //printf("DEBUG rank%d: init-exchange_peer_ipc_mem-memset\n", rank);
        memset(recv_buf, 0, sizeof(recv_buf));
        // Overkill if we don't really needs all peer's handles
        //printf("DEBUG rank%d: init-exchange_peer_ipc_mem-MPI_Allgather\n", rank);
        un_allgather(&send_buf, recv_buf, rank, world);

        //printf("DEBUG rank%d: init-exchange_peer_ipc_mem-forloop\n", rank);
        for (uint32_t i = 0; i < world; i++){
            // Step 4: Prepare pid file descriptor of next process
            auto* peer = recv_buf + i;

            // Step 6: Open IPC handle of remote peer
            //printf("DEBUG rank%d: init-exchange_peer_ipc_mem-forloop%d-get_native\n", rank, i);
            auto l0_device
                = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(queue.get_device());
            void* peer_base;

            //printf("DEBUG rank%d: init-exchange_peer_ipc_mem-forloop%d-zeMemOpenIpcHandle\n", rank, i);
            zeCheck(zeMemOpenIpcHandle(
                    l0_ctx, l0_device, peer->ipc_handle, ZE_IPC_MEMORY_FLAG_BIAS_CACHED, &peer_base));
            buffer[i] = (char*)peer_base + peer->offset;
            sync_buffer[i] = (char*)peer_base + peer->offset + data_size_per_buffer * sizeof(data_type);
            offset[i] = peer->offset;
            ipc_handle[i] = send_buf.ipc_handle;
            //printf("DEBUG rank%d: init-exchange_peer_ipc_mem-forloop%d-end\n", rank, i);
        }
        //printf("DEBUG rank%d: init-exchange_peer_ipc_mem-end\n", rank);
    }


    
    bool initialized;
    void* buffer[max_rank];
    void* sync_buffer[max_rank];
    size_t offset[max_rank];
    ze_ipc_mem_handle_t ipc_handle[max_rank];
    int rank, world;
    int buffer_index;
    int size_per_buffer;
    int data_size_per_buffer;
};
