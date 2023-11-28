#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <system_error>

#include <mpi.h>
#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>

#include "ze_exception.hpp"
#include "sycl_misc.hpp"

#define MAGIC_NUM 15
#define MAX_RANK 8
#define MAX_BUFFER 4096
#define OPERATE_SIZE 14336

size_t buffer_base_size = MAX_BUFFER * 1024 * MAX_RANK;
size_t check_size = 4096;

int world = -1;
int rank = -1;

int use_tmp_buffer;

void* buffer[MAX_RANK];
void* sync_buffer[MAX_RANK];
void* ready_buffer[MAX_RANK];

void exchange_mem(sycl::queue& queue, void* ptr);

struct exchange_contents {
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
  
int main(int argc, char* argv[]) {
  if (argc > 1) {
    use_tmp_buffer = 1;
  }

  size_t buffer_size = buffer_base_size + 1024 * 32768;

  auto ret = MPI_Init(&argc, &argv);
  if (ret == MPI_ERR_OTHER) {
    std::cout<<"MPI init error"<<std::endl;
    return -1;
  }

  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  zeCheck(zeInit(0));

  // rank 0, device 0, subdevice 0
  // rank 1, device 0, subdevice 1
  // rank 2, device 1, subdevice 0
  // ...
  auto queue = currentQueue(rank / 2, rank & 1);
  std::cout << "Buffer size: " << buffer_size << std::endl;

  void* operate_buffer = sycl::malloc_device(buffer_size, queue);

  sycl::event e;

  uint32_t temp_rank = rank;

  uint32_t* ptr = (uint32_t *)operate_buffer + buffer_base_size / sizeof(uint32_t);
  e = queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::range { 1024 / sizeof(uint32_t) }, ([=](sycl::id<1> index) {
      ptr[index] = (uint32_t)temp_rank;
    }));
  });
  queue.wait();

  exchange_mem(queue, operate_buffer);

  MPI_Finalize();
}

void exchange_mem(sycl::queue& queue, void* ptr) {
  // Step 1: Get base address of the pointer
  sycl::context ctx = queue.get_context();
  auto l0_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);

  void *base_addr;
  size_t base_size;
  zeCheck(zeMemGetAddressRange(l0_ctx, ptr, &base_addr, &base_size));
  
  std::cout << "Base size: " << base_size << std::endl;
  std::cout << "Buffer base size: " << buffer_base_size << std::endl;
  std::cout << "Actual buffer size: " << (buffer_base_size + 1024) << std::endl;
  std::cout << "Local: " << base_addr << "+" << (char*)ptr - (char*)base_addr << " ~ " << (void *)((char *)base_addr + base_size) << std::endl;

  // Step 2: Get IPC mem handle from base address
  alignas(64) exchange_contents send_buf;
  alignas(64) exchange_contents recv_buf[world];

  // fill in the exchange info
  zeCheck(zeMemGetIpcHandle(l0_ctx, base_addr, &send_buf.ipc_handle));
  send_buf.offset = (char*)ptr - (char*)base_addr;
  send_buf.pid = getpid();

  int* host_buffer = (int *)(malloc(1024)); 
  void* tmp_buffer = sycl::malloc_device(1024, queue);

  void* sync_addr = NULL;
  sync_addr = (void *)((char*)base_addr + send_buf.offset + buffer_base_size);
  std::cout << "Sync buffer content at " << sync_addr << ": ";
  queue.memcpy(host_buffer, sync_addr, 1024);
  queue.wait();
  for (int i = 0; i < 256; i += 16) {
    std::cout << &host_buffer[i] << ": " << host_buffer[i] << std::endl;
  }

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
    sync_buffer[i] = (char*)peer_base + peer->offset + buffer_base_size + rank * 128;
    ready_buffer[i] = (char*)peer_base + peer->offset + buffer_base_size + rank * 128 + 64;

    char* end = (char*)peer_base + peer->offset + base_size;

    std::cout << "Rank " << i << ": " << peer_base << "+" << peer->offset << " ~ " << (void *)end << std::endl;

    sync_addr = (void *)((char*)peer_base + peer->offset + buffer_base_size);
    //sync_addr = (void *)((char*)base_addr + send_buf.offset + buffer_base_size);

    if (use_tmp_buffer == 0) {    
      std::cout << "Copy sync buffer (mapped from rank " << i << ") at " << sync_addr << " to host" << std::endl;
      queue.memcpy(host_buffer, sync_addr, 1024);
    } else {
      std::cout << "Copy sync buffer (mapped from rank " << i << ") at " << sync_addr << " to temp buffer & then to host" << std::endl;
      queue.memcpy(tmp_buffer, sync_addr, 1024);
      queue.memcpy(host_buffer, tmp_buffer, 1024);
    }
    queue.wait();
   
    std::cout << "Sync buffer content at " << sync_addr << std::endl;
    for (int i = 0; i < 256; i += 16) {
      std::cout << &host_buffer[i] << ": " << host_buffer[i] << std::endl;
    }
  }    
}

