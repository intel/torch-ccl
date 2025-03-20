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

namespace ops {
// pytorch 2.2 above
#if TORCH_VERSION_MAJOR > 1 && TORCH_VERSION_MINOR > 1
std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<C10D_Work>> broadcast_xpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t root_rank,
    int64_t root_tensor,
    bool asyncOp,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::XPU)
          ->broadcast(
              tensor_vec,
              BroadcastOptions{
                  root_rank, root_tensor, std::chrono::milliseconds(timeout), asyncOp});

  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<C10D_Work>>(
      std::move(tensor_vec), work);
}
#else
std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<C10D_Work>> broadcast_xpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t root_rank,
    int64_t root_tensor,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::XPU)
          ->broadcast(
              tensor_vec,
              BroadcastOptions{
                  root_rank, root_tensor, std::chrono::milliseconds(timeout)});

  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<C10D_Work>>(
      std::move(tensor_vec), work);
}
#endif

TORCH_LIBRARY_IMPL(c10d, XPU, m) {
  m.impl("broadcast_", broadcast_xpu_);
}

#if TORCH_VERSION_MAJOR > 1 && TORCH_VERSION_MINOR >= 1
// TODO: Enable sparse all_reduce https://github.com/pytorch/pytorch/pull/103916
std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<C10D_Work>> allreduce_xpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    const c10::optional<at::Tensor>& sparse_indices,
    bool asyncOp,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::XPU)
            ->allreduce(
              tensor_vec,
              c10d::AllreduceOptions{
                  *reduce_op.get(), std::chrono::milliseconds(timeout), asyncOp});

  // Return input tensors as output tensors to make inplace allreduce look like
  // a functional API, so that make_fx can correctly build the dependencies in
  // the graph later.
  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<C10D_Work>>(
      std::move(tensor_vec), work);
}
#else
// Compatible with PyTorch2.0
std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<C10D_Work>> allreduce_xpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::XPU)
            ->allreduce(
              tensor_vec,
              c10d::AllreduceOptions{
                  *reduce_op.get(), std::chrono::milliseconds(timeout)});

  // Return input tensors as output tensors to make inplace allreduce look like
  // a functional API, so that make_fx can correctly build the dependencies in
  // the graph later.
  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<C10D_Work>>(
      std::move(tensor_vec), work);
}
#endif

TORCH_LIBRARY_IMPL(c10d, XPU, m) {
  m.impl("allreduce_", allreduce_xpu_);
}

c10::intrusive_ptr<C10D_Work> allreduce_coalesced_xpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    bool asyncOp,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  AllreduceCoalescedOptions opts = AllreduceCoalescedOptions{};
  opts.reduceOp = *reduce_op.get();
  opts.timeout = std::chrono::milliseconds(timeout);
  opts.asyncOp = asyncOp;

  return process_group->getBackend(c10::DeviceType::XPU)
      ->allreduce_coalesced(tensor_vec, opts);
}

TORCH_LIBRARY_IMPL(c10d, XPU, m) {
  m.impl("allreduce_coalesced_", allreduce_coalesced_xpu_);
}

c10::intrusive_ptr<C10D_Work> reduce_xpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    int64_t root_rank,
    int64_t root_tensor,
    bool asyncOp,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  return process_group->getBackend(c10::DeviceType::XPU)
      ->reduce(
          tensor_vec,
          ReduceOptions{
              *reduce_op.get(),
              root_rank,
              root_tensor,
              std::chrono::milliseconds(timeout), asyncOp});
}

TORCH_LIBRARY_IMPL(c10d, XPU, m) {
  m.impl("reduce_", reduce_xpu_);
}

std::tuple<std::vector<std::vector<at::Tensor>>, c10::intrusive_ptr<C10D_Work>>
allgather_xpu_(
    const std::vector<std::vector<at::Tensor>>& output_tensors,
    at::TensorList input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    bool asyncOp,
    int64_t timeout) {
  auto input_tensors_vec = input_tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::XPU)
          ->allgather(
              const_cast<std::vector<std::vector<at::Tensor>>&>(output_tensors),
              input_tensors_vec,
              AllgatherOptions{std::chrono::milliseconds(timeout), asyncOp});

  // Copy output tensors (not storage) so that this can be used in a functional
  // manner
  return std::
      tuple<std::vector<std::vector<at::Tensor>>, c10::intrusive_ptr<C10D_Work>>(
          output_tensors, work);
}

TORCH_LIBRARY_IMPL(c10d, XPU, m) {
  m.impl("allgather_", allgather_xpu_);
}

// pytorch 2.2 above
#if TORCH_VERSION_MAJOR > 1 && TORCH_VERSION_MINOR > 1
std::tuple<at::Tensor, c10::intrusive_ptr<C10D_Work>> _allgather_base_xpu_(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    bool asyncOp,
    int64_t timeout) {
  auto work = process_group->getBackend(c10::DeviceType::XPU)
                  ->_allgather_base(output_tensor,
                                    input_tensor,
                                    AllgatherOptions{std::chrono::milliseconds(timeout), asyncOp});
  return std::tuple<at::Tensor, c10::intrusive_ptr<C10D_Work>>(output_tensor, work);
}
#else
std::tuple<at::Tensor, c10::intrusive_ptr<C10D_Work>> _allgather_base_xpu_(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const c10::intrusive_ptr<ProcessGroup>& process_group) {
  auto work = process_group->getBackend(c10::DeviceType::XPU)
                  ->_allgather_base(output_tensor, input_tensor);

  return std::tuple<at::Tensor, c10::intrusive_ptr<C10D_Work>>(output_tensor, work);
}
#endif

TORCH_LIBRARY_IMPL(c10d, XPU, m) {
  m.impl("_allgather_base_", _allgather_base_xpu_);
}

c10::intrusive_ptr<c10d::Work> allgather_into_tensor_coalesced_xpu_(
    at::TensorList outputs,
    at::TensorList inputs,  
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    bool asyncOp) {

  auto output_vec = outputs.vec(); 
  auto input_vec = inputs.vec();
  AllgatherOptions opts = AllgatherOptions{};
  opts.asyncOp = asyncOp;
  return process_group->getBackend(c10::DeviceType::XPU)
            ->allgather_into_tensor_coalesced(output_vec, input_vec, opts);
}

TORCH_LIBRARY_IMPL(c10d, XPU, m) {
  m.impl("allgather_into_tensor_coalesced_", allgather_into_tensor_coalesced_xpu_);
}

c10::intrusive_ptr<C10D_Work> allgather_coalesced_xpu_(
    const std::vector<std::vector<at::Tensor>>& output_lists,
    const at::TensorList& input_list,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    bool asyncOp) {
  auto input_list_vec = input_list.vec();
  AllgatherOptions opts = AllgatherOptions{};
  opts.asyncOp = asyncOp;
  return process_group->getBackend(c10::DeviceType::XPU)
      ->allgather_coalesced(
          const_cast<std::vector<std::vector<at::Tensor>>&>(output_lists),
          input_list_vec,
          opts);
}

TORCH_LIBRARY_IMPL(c10d, XPU, m) {
  m.impl("allgather_coalesced_", allgather_coalesced_xpu_);
}

c10::intrusive_ptr<C10D_Work> gather_xpu_(
    const std::vector<std::vector<at::Tensor>>& output_tensors,
    const at::TensorList& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t root_rank,
    bool asyncOp,
    int64_t timeout) {
  auto input_tensors_vec = input_tensors.vec();
  return process_group->getBackend(c10::DeviceType::XPU)
      ->gather(
          const_cast<std::vector<std::vector<at::Tensor>>&>(output_tensors),
          input_tensors_vec,
          GatherOptions{root_rank, std::chrono::milliseconds(timeout), asyncOp});
}

TORCH_LIBRARY_IMPL(c10d, XPU, m) {
  m.impl("gather_", gather_xpu_);
}

// pytorch 2.2 above
#if TORCH_VERSION_MAJOR > 1 && TORCH_VERSION_MINOR > 1
std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<C10D_Work>> scatter_xpu_(
    const at::TensorList& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t root_rank,
    bool asyncOp,
    int64_t timeout) {
  auto output_tensors_vec = output_tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::XPU)
          ->scatter(
              output_tensors_vec,
              const_cast<std::vector<std::vector<at::Tensor>>&>(input_tensors),
              ScatterOptions{root_rank, std::chrono::milliseconds(timeout), asyncOp});

  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<C10D_Work>>(
      std::move(output_tensors_vec), work);
}
#else
std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<C10D_Work>> scatter_xpu_(
    const at::TensorList& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t root_rank,
    int64_t timeout) {
  auto output_tensors_vec = output_tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::XPU)
          ->scatter(
              output_tensors_vec,
              const_cast<std::vector<std::vector<at::Tensor>>&>(input_tensors),
              ScatterOptions{root_rank, std::chrono::milliseconds(timeout)});

  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<C10D_Work>>(
      std::move(output_tensors_vec), work);
}
#endif

TORCH_LIBRARY_IMPL(c10d, XPU, m) {
  m.impl("scatter_", scatter_xpu_);
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<C10D_Work>>
reduce_scatter_xpu_(
    const at::TensorList& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    bool asyncOp,
    int64_t timeout) {
  auto output_tensors_vec = output_tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::XPU)
          ->reduce_scatter(
              output_tensors_vec,
              const_cast<std::vector<std::vector<at::Tensor>>&>(input_tensors),
              ReduceScatterOptions{
                  *reduce_op.get(), std::chrono::milliseconds(timeout), asyncOp});

  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<C10D_Work>>(
      output_tensors_vec, work);
}

TORCH_LIBRARY_IMPL(c10d, XPU, m) {
  m.impl("reduce_scatter_", reduce_scatter_xpu_);
}

// pytorch 2.2 above
#if TORCH_VERSION_MAJOR > 1 && TORCH_VERSION_MINOR > 1
std::tuple<at::Tensor, c10::intrusive_ptr<C10D_Work>> _reduce_scatter_base_xpu_(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    bool asyncOp,
    int64_t timeout) {
  auto work =
      process_group->getBackend(c10::DeviceType::XPU)
          ->_reduce_scatter_base(
              output_tensor,
              input_tensor,
              ReduceScatterOptions{
                  *reduce_op.get(), std::chrono::milliseconds(timeout), asyncOp});

  return std::tuple<at::Tensor, c10::intrusive_ptr<C10D_Work>>(output_tensor, work);
}
#else
std::tuple<at::Tensor, c10::intrusive_ptr<C10D_Work>> _reduce_scatter_base_xpu_(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    int64_t timeout) {
  auto work =
      process_group->getBackend(c10::DeviceType::XPU)
          ->_reduce_scatter_base(
              output_tensor,
              input_tensor,
              ReduceScatterOptions{
                  *reduce_op.get(), std::chrono::milliseconds(timeout)});

  return std::tuple<at::Tensor, c10::intrusive_ptr<C10D_Work>>(output_tensor, work);
}
#endif

TORCH_LIBRARY_IMPL(c10d, XPU, m) {
  m.impl("_reduce_scatter_base_", _reduce_scatter_base_xpu_);
}

c10::intrusive_ptr<C10D_Work> reduce_scatter_tensor_coalesced_xpu_(
    at::TensorList outputs,
    at::TensorList inputs,  
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    bool asyncOp,
    int64_t timeout) {
  auto output_vec = outputs.vec();
  auto input_vec = inputs.vec();
  return process_group->getBackend(c10::DeviceType::XPU)
    ->reduce_scatter_tensor_coalesced(
      output_vec,
      input_vec,
      ReduceScatterOptions{
        *reduce_op.get(), std::chrono::milliseconds(timeout), asyncOp});
}

TORCH_LIBRARY_IMPL(c10d, XPU, m) {
  m.impl("reduce_scatter_tensor_coalesced_", reduce_scatter_tensor_coalesced_xpu_);
}

c10::intrusive_ptr<C10D_Work> alltoall_base_xpu_(
    at::Tensor& output,
    at::Tensor& input,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    std::vector<int64_t> output_split_sizes,
    std::vector<int64_t> input_split_sizes,
    bool asyncOp,
    int64_t timeout) {
  return process_group->getBackend(c10::DeviceType::XPU)
      ->alltoall_base(
          output,
          input,
          output_split_sizes,
          input_split_sizes,
          AllToAllOptions{std::chrono::milliseconds(timeout), asyncOp});
}

TORCH_LIBRARY_IMPL(c10d, XPU, m) {
  m.impl("alltoall_base_", alltoall_base_xpu_);
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<C10D_Work>> alltoall_xpu_(
    const at::TensorList& output_tensors,
    const at::TensorList& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    bool asyncOp,
    int64_t timeout) {
  auto output_tensors_vec = output_tensors.vec();
  auto input_tensors_vec = input_tensors.vec();
  auto work = process_group->getBackend(c10::DeviceType::XPU)
                  ->alltoall(
                      output_tensors_vec,
                      input_tensors_vec,
                      AllToAllOptions{std::chrono::milliseconds(timeout), asyncOp});
  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<C10D_Work>>(
      std::move(output_tensors_vec), work);
}

TORCH_LIBRARY_IMPL(c10d, XPU, m) {
  m.impl("alltoall_", alltoall_xpu_);
}

c10::intrusive_ptr<C10D_Work> send_xpu(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t dstRank,
    int64_t tag) {
  auto tensor_vec = tensors.vec();
  return process_group->getBackend(c10::DeviceType::XPU)
      ->send(tensor_vec, static_cast<int>(dstRank), static_cast<int>(tag));
}

TORCH_LIBRARY_IMPL(c10d, XPU, m) {
  m.impl("send", send_xpu);
}

c10::intrusive_ptr<C10D_Work> recv_xpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t srcRank,
    int64_t tag) {
  auto tensor_vec = tensors.vec();
  return process_group->getBackend(c10::DeviceType::XPU)
      ->recv(tensor_vec, static_cast<int>(srcRank), static_cast<int>(tag));
}

TORCH_LIBRARY_IMPL(c10d, XPU, m) {
  m.impl("recv_", recv_xpu_);
}

c10::intrusive_ptr<C10D_Work> recv_any_source_xpu_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t tag) {
  auto tensor_vec = tensors.vec();
  return process_group->getBackend(c10::DeviceType::XPU)
      ->recvAnysource(tensor_vec, static_cast<int>(tag));
}

TORCH_LIBRARY_IMPL(c10d, XPU, m) {
  m.impl("recv_any_source_", recv_any_source_xpu_);
}

c10::intrusive_ptr<Work> barrier_xpu(
    at::Tensor /* unused */,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const std::vector<int64_t>& device_ids,
    bool asyncOp,
    int64_t timeout) {
  BarrierOptions opts = BarrierOptions{};
  opts.device_ids = device_ids;
  opts.timeout = std::chrono::milliseconds(timeout);
  opts.asyncOp = asyncOp;
  return process_group->getBackend(c10::DeviceType::XPU)
      ->barrier(opts);
}

TORCH_LIBRARY_IMPL(c10d, XPU, m) {
  m.impl("barrier", barrier_xpu);
}
} // namespace ops


using oneccl_bindings_for_pytorch::DispatchStub;
using oneccl_bindings_for_pytorch::call_with_lock;
using oneccl_bindings_for_pytorch::format_tensors_param;

namespace {

static std::once_flag cclInitOnceFlag;

void checkRank(int rank, int size)
{
  if (!((rank >= 0) && (rank < size))) {
    throw std::invalid_argument("unexpected rank");
  }
}

} // namespace


c10::intrusive_ptr<c10::ivalue::Future> createFutureAsOutput(
        const std::vector<std::vector<at::Tensor>>& outputTensors) {
  if (outputTensors.size() > 1) {
    return c10::make_intrusive<c10::ivalue::Future>(
            c10::ListType::create(c10::ListType::create(c10::TensorType::get())));
  }
  return c10::make_intrusive<c10::ivalue::Future>(
          c10::ListType::create(c10::TensorType::get()));
}

void returnFutureWithOutput(
        c10::intrusive_ptr<c10::ivalue::Future>& future,
        const std::vector<std::vector<at::Tensor>>& outputTensors) {
  if (outputTensors.size() == 0) {
    future->markCompleted(c10::IValue(std::vector<at::Tensor>()));
    return;
  }
  if (outputTensors.size() > 1) {
    future->markCompleted(c10::IValue(outputTensors));
    return;
  }
  future->markCompleted(c10::IValue(outputTensors[0]));
}

bool parseTorchCCLEnvVarFlag(const char* envVarName, bool default_val) {
    char* stringValue = std::getenv(envVarName);
    int val;
    if (stringValue != nullptr) {
      try {
        val = std::stoi(stringValue);
      } catch (std::exception& e) {
        TORCH_CHECK(false,
            "Invalid value for environment variable: " + std::string(envVarName));
      }
    } else {
      return default_val;
    }

    if (val == 1) return true; else return false;
}

int getOneCCLEnvVar(std::string envVarName) {
    char* stringValue = std::getenv(envVarName.c_str());
    if (stringValue != nullptr) {
      try {
        int val = std::stoi(stringValue);
        return val;
      } catch (std::exception& e) {
        TORCH_CHECK(false,
            "Invalid value for environment variable: " + std::string(envVarName));
      }
    } else {
       return -1;
    }
}

void setOneCCLEnvVar(std::string envVarName, int val) {
    setenv(envVarName.c_str(), std::to_string(val).c_str(), val);
}

void setOneCCLEnvVar(std::string envVarName, std::string val) {
    setenv(envVarName.c_str(), val.c_str(), 1);
}

bool with_mpirun() {
    return (getenv("MPI_LOCALRANKID") || getenv("MPI_LOCALNRANKS") || getenv("PMI_RANK") ||
            getenv("PMI_SIZE") || getenv("PMIX_RANK"))
               ? true
               : false;
}

ProcessGroupCCL::AsyncWorkCCL::AsyncWorkCCL(std::vector<std::vector<at::Tensor>> outputTensors,
                                            int rank,
                                            c10d::OpType opType,
                                            const char* profilingTitle,
                                            const c10::optional<std::vector<at::Tensor>>& inputTensors)
// Profiler: Pass nullptr as profilingTitle to parent constructor to
// replace default profiler implementation with async version that reports
// correct timestamps for work that is asynchronously executed.
        : C10D_Work(rank, opType, nullptr, inputTensors),
          outputTensors_(std::move(outputTensors)),
          future_(createFutureAsOutput(outputTensors)
          ) {
  if (profilingTitle != nullptr) {
//    recordAsyncWorkProfilingInfo(profilingTitle, inputTensors);
    // TODO: for cpu async profiling repot.
  }
}

c10::intrusive_ptr<c10::ivalue::Future> ProcessGroupCCL::AsyncWorkCCL::getFuture() {
  return future_;
}

std::vector<at::Tensor> ProcessGroupCCL::AsyncWorkCCL::result() {
  TORCH_CHECK(
          isCompleted(),
          "Work needs to be completed before calling result(). "
          "Should call wait() before result().");
  TORCH_CHECK(
          outputTensors_.size() <= 1,
          "work result does not support list of lists, use .getFuture() and value()");
  return outputTensors_.size() == 0 ? std::vector<at::Tensor>()
                                    : outputTensors_.at(0);
}

void ProcessGroupCCL::AsyncWorkCCL::finishAsyncWorkCCLError(std::exception_ptr eptr) {
  future_->setError(eptr);
  finish(eptr);
}

void ProcessGroupCCL::AsyncWorkCCL::finishAsyncWorkCCL() {
  returnFutureWithOutput(future_, outputTensors_);
  finish();
}

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
#if TORCH_VERSION_MAJOR > 1
c10::intrusive_ptr<Backend> ProcessGroupCCL::createProcessGroupCCL(
#else
c10::intrusive_ptr<ProcessGroup> ProcessGroupCCL::createProcessGroupCCL(
#endif
    const c10::intrusive_ptr<Store>& store,
    int rank,
    int size,
    std::chrono::milliseconds op_time_out)
{
  return c10::make_intrusive<ProcessGroupCCL>(store, rank, size, op_time_out);
}

ProcessGroupCCL::ProcessGroupCCL(const c10::intrusive_ptr<Store>& store, int rank, int size, std::chrono::milliseconds op_time_out)
#if TORCH_VERSION_MAJOR > 1
    : Backend(rank, size), store_(store), timeout(op_time_out),
#else
    : ProcessGroup(rank, size), store_(store), timeout(op_time_out),
#endif
      ccl_member_(std::make_unique<oneccl_bindings_for_pytorch::CCLCommCollector>())
{
  torch_llm_allreduce_ = parseTorchCCLEnvVarFlag(TORCH_LLM_ALLREDUCE, torch_llm_allreduce_);
  // Hide CCL_SKIP_SCHEDULER/CCL_ENABLE_SYCL_KERNELS/CCL_SYCL_ESIMD by TORCH_LLM_ALLREDUCE
  if (torch_llm_allreduce_) {
#if CCL_MINOR_VERSION < 14
      setOneCCLEnvVar("CCL_SKIP_SCHEDULER", 1); // for basekit 2024.1
      setOneCCLEnvVar("CCL_ENABLE_SYCL_KERNELS", 1); // for basekit 2024.2
      setOneCCLEnvVar("CCL_SYCL_ESIMD", 1); // for basekit 2024.2
      blockingWait_ = false;
#endif
      useSameStream_ = true;
  }
  useSameStream_ = parseTorchCCLEnvVarFlag(CCL_SAME_STREAM, useSameStream_);
  blockingWait_ = parseTorchCCLEnvVarFlag(CCL_BLOCKING_WAIT, blockingWait_);

  // Set these 3 variables to follow oneCCL specs, which is required to enable use drmfd mode of ze exchange mechanism.
  if (!with_mpirun()) {
    // If it's launched by 'torchrun', LOCAL_RANK and LOCAL_WORLD_SIZE were set.
    int local_rank = getOneCCLEnvVar("LOCAL_RANK");
    int local_world_size = getOneCCLEnvVar("LOCAL_WORLD_SIZE");

    // If these 2 variables were not set, it's launched by multi-processing package. In that case we'll
    // use rank and size.
    if (local_rank == -1 || local_world_size == -1) {
        local_rank = rank;
        local_world_size = size;
    }
    
    setOneCCLEnvVar("CCL_PROCESS_LAUNCHER", "none");
    setOneCCLEnvVar("CCL_LOCAL_RANK", local_rank);
    setOneCCLEnvVar("CCL_LOCAL_SIZE", local_world_size);
  }

#ifdef NDEBUG
    TORCH_CHECK(!oneccl_bindings_for_pytorch_wait_gdb(), "Cannot force torch ccl wait for gdb attaching in release version");
#else
  if (oneccl_bindings_for_pytorch_wait_gdb()) {
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

void ProcessGroupCCL::startCoalescing() {
    // TODO: GroupStart
    // Currently oneccl dost not support group execution like NCCL, just mark here.
    coalescedDevices_.clear();
    is_coalescing_ = true;
}

c10::intrusive_ptr<Work> ProcessGroupCCL::endCoalescing() {
    // TODO: GroupEnd
    is_coalescing_ = false;
    auto work = DispatchStub::end_coalescing(*this);
    return work;
}

c10::intrusive_ptr<C10D_Work> ProcessGroupCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts)
{
  std::vector<c10::IValue> tensor_param;
  format_tensors_param(tensor_param, tensors);
  RECORD_FUNCTION("oneccl_bindings_for_pytorch::broadcast", tensor_param);

  checkRank(opts.rootRank, getSize());
  auto work = DispatchStub::broadcast(tensors, opts, *this);

  return work;
}

c10::intrusive_ptr<C10D_Work> ProcessGroupCCL::allreduce(
  std::vector<at::Tensor>& tensors,
  const AllreduceOptions& opts)
{
  std::vector<c10::IValue> tensor_param;
  format_tensors_param(tensor_param, tensors);
  RECORD_FUNCTION("oneccl_bindings_for_pytorch::allreduce", tensor_param);

  auto work = DispatchStub::allreduce(tensors, opts, *this);
  return work;
}

c10::intrusive_ptr<C10D_Work> ProcessGroupCCL::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts)
{
  std::vector<c10::IValue> tensor_param;
  format_tensors_param(tensor_param, tensors);
  RECORD_FUNCTION("oneccl_bindings_for_pytorch::allreduce_coalesced", tensor_param);

  auto work = DispatchStub::allreduce_coalesced(tensors, opts, *this);
  return work;
}

c10::intrusive_ptr<C10D_Work> ProcessGroupCCL::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts)
{
  std::vector<c10::IValue> tensor_param;
  format_tensors_param(tensor_param, tensors);
  RECORD_FUNCTION("oneccl_bindings_for_pytorch::reduce", tensor_param);

  checkRank(opts.rootRank, getSize());
  auto work = DispatchStub::reduce(tensors, opts, *this);
  return work;
}


c10::intrusive_ptr<C10D_Work> ProcessGroupCCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts)
{
  std::vector<c10::IValue> tensor_param;
  format_tensors_param(tensor_param, inputTensors);
  format_tensors_param(tensor_param, outputTensors);
  RECORD_FUNCTION("oneccl_bindings_for_pytorch::allgather", tensor_param);

  auto work = DispatchStub::allgather(outputTensors, inputTensors, opts, *this);
  return work;
}

c10::intrusive_ptr<C10D_Work> ProcessGroupCCL::_allgather_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      const AllgatherOptions& opts)
{
  std::vector<c10::IValue> tensor_param;
  format_tensors_param(tensor_param, inputTensor);
  format_tensors_param(tensor_param, outputTensor);
  RECORD_FUNCTION("oneccl_bindings_for_pytorch::_allgather_base", tensor_param);
  auto work = DispatchStub::_allgather_base(outputTensor, inputTensor, opts, *this);
  return work;
}

c10::intrusive_ptr<C10D_Work> ProcessGroupCCL::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllgatherOptions& /* unused */)
{
  TORCH_CHECK(false, "ProcessGroupCCL does not support allgather_coalesced");
}

c10::intrusive_ptr<Work> ProcessGroupCCL::allgather_into_tensor_coalesced(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts)
{
  std::vector<c10::IValue> tensor_param;
  format_tensors_param(tensor_param, inputTensors);
  format_tensors_param(tensor_param, outputTensors);
  RECORD_FUNCTION("oneccl_bindings_for_pytorch::allgather_into_tensor_coalesced", tensor_param);

  auto work = DispatchStub::allgather_into_tensor_coalesced(outputTensors, inputTensors, opts, *this);
  return work;
}

c10::intrusive_ptr<C10D_Work> ProcessGroupCCL::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts)
{
  std::vector<c10::IValue> tensor_param;
  format_tensors_param(tensor_param, inputTensors);
  format_tensors_param(tensor_param, outputTensors);
  RECORD_FUNCTION("oneccl_bindings_for_pytorch::gather", tensor_param);

  auto work = DispatchStub::gather(outputTensors, inputTensors, opts, *this);
  return work;
}

c10::intrusive_ptr<C10D_Work> ProcessGroupCCL::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts)
{
  std::vector<c10::IValue> tensor_param;
  format_tensors_param(tensor_param, inputTensors);
  format_tensors_param(tensor_param, outputTensors);
  RECORD_FUNCTION("oneccl_bindings_for_pytorch::scatter", tensor_param);

  auto work = DispatchStub::scatter(outputTensors, inputTensors, opts, *this);
  return work;
}

c10::intrusive_ptr<C10D_Work> ProcessGroupCCL::reduce_scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ReduceScatterOptions& opts)
{
  std::vector<c10::IValue> tensor_param;
  format_tensors_param(tensor_param, inputTensors);
  format_tensors_param(tensor_param, outputTensors);
  RECORD_FUNCTION("oneccl_bindings_for_pytorch::reduce_scatter", tensor_param);

  auto work = DispatchStub::reduce_scatter(outputTensors, inputTensors, opts, *this);
  return work;
}

c10::intrusive_ptr<C10D_Work> ProcessGroupCCL::_reduce_scatter_base(
     at::Tensor& outputTensor,
     at::Tensor& inputTensor,
     const ReduceScatterOptions& opts)
{
     std::vector<c10::IValue> tensor_param;
     format_tensors_param(tensor_param, inputTensor);
     format_tensors_param(tensor_param, outputTensor);
     RECORD_FUNCTION("oneccl_bindings_for_pytorch::_reduce_scatter_base", tensor_param);
     auto work = DispatchStub::_reduce_scatter_base(outputTensor, inputTensor, opts, *this);
     return work;
}

c10::intrusive_ptr<Work> ProcessGroupCCL::reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const ReduceScatterOptions& opts)
{
  std::vector<c10::IValue> tensor_param;
  format_tensors_param(tensor_param, inputTensors);
  format_tensors_param(tensor_param, outputTensors);
  RECORD_FUNCTION("oneccl_bindings_for_pytorch::reduce_scatter_tensor_coalesced", tensor_param);
  
  auto work = DispatchStub::reduce_scatter_tensor_coalesced(outputTensors, inputTensors, opts, *this);
  return work;
}

c10::intrusive_ptr<C10D_Work> ProcessGroupCCL::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& opts)
{
  std::vector<c10::IValue> tensor_param;
  format_tensors_param(tensor_param, inputTensor);
  format_tensors_param(tensor_param, outputTensor);
  RECORD_FUNCTION("oneccl_bindings_for_pytorch::alltoall_base", tensor_param);

  auto work = DispatchStub::alltoall_base(outputTensor, inputTensor, outputSplitSizes, inputSplitSizes, opts, *this);
  return work;
}

c10::intrusive_ptr<C10D_Work> ProcessGroupCCL::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllToAllOptions& opts)
{
  std::vector<c10::IValue> tensor_param;
  format_tensors_param(tensor_param, inputTensors);
  format_tensors_param(tensor_param, outputTensors);
  RECORD_FUNCTION("oneccl_bindings_for_pytorch::alltoall", tensor_param);

  auto work = DispatchStub::alltoall(outputTensors, inputTensors, opts, *this);
  return work;
}

c10::intrusive_ptr<C10D_Work> ProcessGroupCCL::send(
    std::vector<at::Tensor>& tensors,
    int dstRank,
    int tag)
{
  std::vector<c10::IValue> tensor_param;
  format_tensors_param(tensor_param, tensors);
  RECORD_FUNCTION("oneccl_bindings_for_pytorch::send", tensor_param);

  auto work = DispatchStub::send(tensors, dstRank, tag, *this);
  return work;
}

c10::intrusive_ptr<C10D_Work> ProcessGroupCCL::recv(
    std::vector<at::Tensor>& tensors,
    int srcRank,
    int tag)
{
  std::vector<c10::IValue> tensor_param;
  format_tensors_param(tensor_param, tensors);
  RECORD_FUNCTION("oneccl_bindings_for_pytorch::recv", tensor_param);

  auto work = DispatchStub::recv(tensors, srcRank, tag, *this);
  return work;
}

c10::intrusive_ptr<C10D_Work> ProcessGroupCCL::recvAnysource(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */)
{
  TORCH_CHECK(false, "ProcessGroupCCL does not support recvAnysource");
}

c10::intrusive_ptr<C10D_Work> ProcessGroupCCL::barrier(
    const BarrierOptions& opts)
{
 return DispatchStub::barrier(opts, *this);
}

} // namespace c10d
