#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <system_error>

#include <mpi.h>
#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>

#include "cxxopts.hpp"
#include "ze_exception.hpp"
#include "allreduce.h"


static size_t align_up(size_t size, size_t align_sz) {
    return ((size + align_sz -1) / align_sz) * align_sz;
}

void *mmap_host(size_t map_size, int dma_buf_fd) {
  auto page_size = getpagesize();
  map_size = align_up(map_size, page_size);
  return mmap(nullptr, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, dma_buf_fd, 0);
}

template <typename T>
bool checkResults(T *ptr, T c, size_t count) {
  for (int i = 0; i < count; ++ i) {
    if (*ptr != c) return false;
  }
  return true;
}

int main(int argc, char* argv[]) {
  // parse command line options
  cxxopts::Options opts(
      "Fill remote GPU memory",
      "Exchange IPC handle to next rank (wrap around), and fill received buffer");

  opts.allow_unrecognised_options();
  opts.add_options()
    ("c,count", "Data content count", cxxopts::value<size_t>()->default_value("8192"))
    ("t,type", "Data content type", cxxopts::value<std::string>()->default_value("fp16"))
    ;

  auto parsed_opts = opts.parse(argc, argv);
  auto count = parsed_opts["count"].as<size_t>();
  auto dtype = parsed_opts["type"].as<std::string>();

  size_t alloc_size = 0;

  if (dtype == "fp16")
    alloc_size = count * sizeof(sycl::half);
  else if (dtype == "float")
    alloc_size = count * sizeof(float);

  // init section
  auto ret = MPI_Init(&argc, &argv);
  if (ret == MPI_ERR_OTHER) {
    std::cout<<"MPI init error"<<std::endl;
    return -1;
  }

  zeCheck(zeInit(0));
  int rank, world;

  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  

  // rank 0, device 0, subdevice 0
  // rank 1, device 0, subdevice 1
  // rank 2, device 1, subdevice 0
  // ...
  auto queue = currentQueue(rank / 2, rank & 1);
  allreducer<float> ar;
  ar.init(queue, rank, world);
  // temporal buffer used for allreduce temporal use only.
  void* buffer = sycl::malloc_device(alloc_size, queue); 
  queue.fill<float>((float *)buffer, (float)(rank), count); 
  queue.wait();
//   void* temp_buffer = NULL;
  ar.allreduce(queue, buffer, count);
  // avoid race condition
  queue.wait();
  MPI_Barrier(MPI_COMM_WORLD);

  // Check buffer contents
  void* host_buf = sycl::malloc_host(alloc_size, queue);
  queue.memcpy(host_buf, buffer, alloc_size);
  queue.wait();

  // Or we map the device to host
//   int dma_buf = 0;
//   memcpy(&dma_buf, &ipc_handle, sizeof(int));
//   void *host_buf = mmap_host(alloc_size, dma_buf);

  bool check = false;
  int32_t sum = world * (world - 1) / 2;
  if (dtype == "fp16"){
    check = checkResults((sycl::half *)host_buf, (sycl::half)sum, count);
    std::cout<<"world:"<<world<<"\nrank:" <<rank <<"\nvalue:"<<((sycl::half *)host_buf)[0]<<std::endl;
  } else {
    check = checkResults((float*)host_buf, (float)sum, count);
    std::cout<<"world:"<<world<<"\nrank:" <<rank <<"\nvalue:"<<((float *)host_buf)[0]<<std::endl;
  }
    
  
  if (check)
    std::cout<<"Successfully fill remote buffer"<<std::endl;
  else
    std::cout<<"Error occured when fill remote buffer"<<std::endl;

  // Clean up, close/put ipc handles, free memory, etc.
  ar.release(queue);
  // zeCheck(zeMemPutIpcHandle(l0_ctx, ipc_handle)); /* the API is added after v1.6 */
  sycl::free(buffer, queue);
  // sycl::free(host_buf, queue);
  munmap(host_buf, alloc_size);
}
