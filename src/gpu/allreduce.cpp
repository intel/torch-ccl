#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <system_error>

#include <mpi.h>
#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>

#include "ze_exception.hpp"
#include "allreduce.h"
#include <chrono>

#define REPEAT 10

int work_only = -1;
int sync_only = -1;

int get_work_only(int init_value = 0) {
  int tmp_work_only = init_value;
  char *tmp_str = getenv("TORCH_CCL_WORK_ONLY");
  if (tmp_str) {
    tmp_work_only = atoi(tmp_str);
  }
  work_only = tmp_work_only;
  return tmp_work_only;
}

int get_sync_only(int init_value = 0) {
  int tmp_sync_only = init_value;
  char *tmp_str = getenv("TORCH_CCL_SYNC_ONLY");
  if (tmp_str) {
    tmp_sync_only = atoi(tmp_str);
  }
  sync_only = tmp_sync_only;
  return tmp_sync_only;
}

void act(allreducer<sycl::half>& ar, sycl::queue& queue, void* inout_buffer, uint32_t size);

int main(int argc, char* argv[]) {
  // init section
  auto ret = MPI_Init(&argc, &argv);
  if (ret == MPI_ERR_OTHER) {
    std::cout<<"MPI init error"<<std::endl;
    return -1;
  }

  if (work_only == -1) {
    get_work_only(0);
  }
  if (sync_only == -1) {
    get_sync_only(0);
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
  allreducer<sycl::half> ar;
  ar.init(queue, rank, world);

  sycl::half* small_buffer = (sycl::half*)sycl::malloc_device(14336 * sizeof(sycl::half), queue);
  sycl::half* large_buffer = (sycl::half*)sycl::malloc_device(14336 * 32 * sizeof(sycl::half), queue);

  for (int i = 0; i < 140; i++) {
    act(ar, queue, large_buffer, 14336 * 32);
  }
  for (int i = 0; i < 31; i++) {
    for (int j = 0; j < 140; j++) {
      act(ar, queue, small_buffer, 14336);
    }
  }
  queue.wait();

  uint64_t host_time[REPEAT];
  uint64_t full_time[REPEAT];

  for (int k = 0; k < REPEAT; k++) {
    MPI_Barrier(MPI_COMM_WORLD);
    uint64_t start = int64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count());

    for (int i = 0; i < 140; i++) {
      act(ar, queue, large_buffer, 14336 * 32);
    }
    for (int i = 0; i < 31; i++) {
      for (int j = 0; j < 140; j++) {
        act(ar, queue, small_buffer, 14336);
      }
    }
    uint64_t host_end = int64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count());
    queue.wait();
    uint64_t full_end = int64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count());
    host_time[k] = host_end - start;
    full_time[k] = full_end - start;
  }

  uint64_t total_host_time = 0;
  uint64_t total_full_time = 0;
  for (int k = 0; k < REPEAT; k++) {
    total_host_time += host_time[k];
    total_full_time += full_time[k];
  }

  total_host_time /= REPEAT;
  total_full_time /= REPEAT;

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

  std::cout << "Average full time: " << total_full_time << std::endl;
  std::cout << "Average host time (for reference): " << total_host_time << std::endl;
  for (int k = 0; k < REPEAT; k++) {
    std::cout << "  Full time on round " << k << ": " << full_time[k] << std::endl;
    std::cout << "  Host time on round " << k << " (for reference): " << host_time[k] << std::endl;
  }
}

void act(allreducer<sycl::half>& ar, sycl::queue& queue, void* inout_buffer, uint32_t size) {
  if (work_only != 0) {
    ar.work_only(queue, inout_buffer, size);
    return;
  }
  if (sync_only != 0) {
    ar.sync_only(queue, inout_buffer, size);
    return;
  }
  ar.allreduce(queue, inout_buffer, size);
}
