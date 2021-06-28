/*
 * Copyright (c) 2020-2021, Intel Corporation
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the Intel Corporation nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <sys/types.h>
#include <unistd.h>
#include <map>
#include <ATen/record_function.h>
#include <ccl_comm_collector.h>
#include "ProcessGroupCCL.hpp"
#include "dispatch_stub.h"
#include "env.h"


namespace c10d
{

using torch_ccl::DispatchStub;
using torch_ccl::call_with_lock;
using torch_ccl::format_tensors_param;

namespace {

static std::once_flag cclInitOnceFlag;

void checkRank(int rank, int size)
{
  if (!((rank >= 0) && (rank < size))) {
    throw std::invalid_argument("unexpected rank");
  }
}

} // namespace

const int64_t ProcessGroupCCL::OP_TIMEOUT_MILLIS = 10 * 1000;
std::mutex ProcessGroupCCL::globalMutex;

void ProcessGroupCCL::cclFini()
{
  DispatchStub::reset_all();
  std::unique_lock<std::mutex> globalLock(globalMutex);
}

void ProcessGroupCCL::cclInitOnce()
{
  std::call_once(cclInitOnceFlag, []() {
    if (std::atexit(ProcessGroupCCL::cclFini))
    {
        throw std::runtime_error("failed to register the CCL exit handler");
    }
  });
}

c10::intrusive_ptr<ProcessGroup> ProcessGroupCCL::createProcessGroupCCL(
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    const std::chrono::duration<float>& op_time_out)
{
  return c10::make_intrusive<ProcessGroupCCL>(store, rank, size, op_time_out);
}

ProcessGroupCCL::ProcessGroupCCL(const c10::intrusive_ptr<Store>& store, int rank, int size, const std::chrono::duration<float>& op_time_out)
    : ProcessGroup(rank, size), store_(store), op_timeout_millis(op_time_out),
      ccl_member_(std::make_unique<torch_ccl::CCLCommCollector>())
{
#ifdef NDEBUG
    TORCH_CHECK(!torch_ccl_wait_gdb(), "Cannot force torch ccl wait for gdb attaching in release version");
#else
  if (torch_ccl_wait_gdb()) {
    volatile int gwf = 0;
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    fflush(stdout);
    while (0 == gwf)
      sleep(5);
  }
#endif
}

ProcessGroupCCL::~ProcessGroupCCL()
{
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts)
{
  std::vector<c10::IValue> tensor_param;
  format_tensors_param(tensor_param, tensors);
  RECORD_FUNCTION("torch_ccl::broadcast", tensor_param);

  checkRank(opts.rootRank, getSize());
  auto work = DispatchStub::broadcast(tensors, opts, *this);

  work->run();
  return work;
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupCCL::allreduce(
  std::vector<at::Tensor>& tensors,
  const AllreduceOptions& opts)
{
  std::vector<c10::IValue> tensor_param;
  format_tensors_param(tensor_param, tensors);
  RECORD_FUNCTION("torch_ccl::allreduce", tensor_param);

  auto work = DispatchStub::allreduce(tensors, opts, *this);
  work->run();
  return work;
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupCCL::allreduce_coalesced(
    std::vector<at::Tensor>& /* unused */,
    const AllreduceCoalescedOptions& /* unused */)
{
  TORCH_CHECK(false, "ProcessGroupCCL does not support allreduce_coalesced");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupCCL::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts)
{
  std::vector<c10::IValue> tensor_param;
  format_tensors_param(tensor_param, tensors);
  RECORD_FUNCTION("torch_ccl::reduce", tensor_param);

  checkRank(opts.rootRank, getSize());
  auto work = DispatchStub::reduce(tensors, opts, *this);
  work->run();
  return work;
}


c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupCCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts)
{
  std::vector<c10::IValue> tensor_param;
  format_tensors_param(tensor_param, inputTensors);
  format_tensors_param(tensor_param, outputTensors);
  RECORD_FUNCTION("torch_ccl::allgather", tensor_param);

  auto work = DispatchStub::allgather(outputTensors, inputTensors, opts, *this);
  work->run();
  return work;
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupCCL::_allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& /* unused */)
{
  TORCH_CHECK(false, "ProcessGroupCCL does not support _allgather_base");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupCCL::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllgatherOptions& /* unused */)
{
  TORCH_CHECK(false, "ProcessGroupCCL does not support allgather_coalesced");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupCCL::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts)
{
  std::vector<c10::IValue> tensor_param;
  format_tensors_param(tensor_param, inputTensors);
  format_tensors_param(tensor_param, outputTensors);
  RECORD_FUNCTION("torch_ccl::gather", tensor_param);

  auto work = DispatchStub::gather(outputTensors, inputTensors, opts, *this);
  work->run();
  return work;
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupCCL::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts)
{
  std::vector<c10::IValue> tensor_param;
  format_tensors_param(tensor_param, inputTensors);
  format_tensors_param(tensor_param, outputTensors);
  RECORD_FUNCTION("torch_ccl::scatter", tensor_param);

  auto work = DispatchStub::scatter(outputTensors, inputTensors, opts, *this);
  work->run();
  return work;
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupCCL::reduce_scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ReduceScatterOptions& /* unused */)
{
  TORCH_CHECK(false, "ProcessGroupCCL does not support reduce_scatter");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupCCL::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& opts)
{
  std::vector<c10::IValue> tensor_param;
  format_tensors_param(tensor_param, inputTensor);
  format_tensors_param(tensor_param, outputTensor);
  RECORD_FUNCTION("torch_ccl::alltoall_base", tensor_param);

  auto work = DispatchStub::alltoall_base(outputTensor, inputTensor, outputSplitSizes, inputSplitSizes, opts, *this);
  work->run();
  return work;
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupCCL::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllToAllOptions& opts)
{
  std::vector<c10::IValue> tensor_param;
  format_tensors_param(tensor_param, inputTensors);
  format_tensors_param(tensor_param, outputTensors);
  RECORD_FUNCTION("torch_ccl::alltoall", tensor_param);

  auto work = DispatchStub::alltoall(outputTensors, inputTensors, opts, *this);
  work->run();
  return work;
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupCCL::send(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */,
    int /* unused */)
{
  TORCH_CHECK(false, "ProcessGroupCCL does not support send");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupCCL::recv(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */,
    int /* unused */)
{
  TORCH_CHECK(false, "ProcessGroupCCL does not support recv");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupCCL::recvAnysource(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */)
{
  TORCH_CHECK(false, "ProcessGroupCCL does not support recvAnysource");
}

c10::intrusive_ptr<ProcessGroup::Work> ProcessGroupCCL::barrier(
    const BarrierOptions& opts)
{
 return DispatchStub::barrier(opts, *this);
}

} // namespace c10d
