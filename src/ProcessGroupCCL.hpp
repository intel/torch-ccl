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

#pragma once


#include <exception>
#include <memory>
#include <mutex>
#include <vector>

#include <torch/version.h>
#if  TORCH_VERSION_MAJOR > 1 || TORCH_VERSION_MINOR >= 13
  #if TORCH_VERSION_MAJOR > 1
  #include <torch/csrc/distributed/c10d/Backend.hpp>
  #include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
  #else
  #include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
  #endif 
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#else
#include <c10d/ProcessGroup.hpp>
#include <c10d/Store.hpp>
#include <c10d/Types.hpp>
#include <c10d/Utils.hpp>
#endif


namespace oneccl_bindings_for_pytorch {
struct CCLCommCollector;

static inline void format_tensors_param(std::vector<c10::IValue>& param, const at::Tensor& tensor) {
  param.emplace_back(tensor);
}

template <typename T>
static inline void format_tensors_param(std::vector<c10::IValue>& param, const std::vector<T>& vec) {
  for (const auto& elem : vec) {
    format_tensors_param(param, elem);
  }
}
}

namespace c10d {

#if TORCH_VERSION_MAJOR > 1 || TORCH_VERSION_MINOR >= 13
using C10D_Work = c10d::Work;
#else
using C10D_Work = c10d::ProcessGroup::Work;
#endif

// WorkCCL is the state associated with a CCL operarion.
//
// ProcessGroupCCL implements CCL bindings for c10d.
//
// All functions on this class are expected to be called in the same
// order across processes in the group.
//
// All collective functions provided by this class are scheduled
// for asynchronous execution by CCL.
constexpr const char* CCL_BACKEND_NAME = "ccl";

// Environment variable which controls whether wait() and synchronize() are blocking or
// non-blocking.
constexpr const char* CCL_BLOCKING_WAIT = "CCL_BLOCKING_WAIT";

// Environment variable which controls whether or not use default stream as
// communication stream for collectives
constexpr const char* CCL_SAME_STREAM = "CCL_SAME_STREAM";

constexpr const char* TORCH_LLM_ALLREDUCE = "TORCH_LLM_ALLREDUCE";

// inline constexpr CoalActive = 0x01, CoalColl = 0x02, CoalP2P = 0x04;

#if TORCH_VERSION_MAJOR > 1
using Baseclass = Backend;
#else
using Baseclass = ProcessGroup;
#endif
class ProcessGroupCCL : public Baseclass 
{
public:
  class AsyncWorkCCL : public C10D_Work {
  public:
    AsyncWorkCCL(std::vector<std::vector<at::Tensor>> outputTensors,
                 int rank = -1,
                 c10d::OpType opType = OpType::UNKNOWN,
                 const char* profilingTitle = nullptr,
                 const c10::optional<std::vector<at::Tensor>>& inputTensors = c10::nullopt);

    virtual void run() = 0;

    c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

    std::vector<at::Tensor> result() override;

    virtual void finishAsyncWorkCCL();

    void finishAsyncWorkCCLError(std::exception_ptr eptr);

  public:
    std::string debugName;
    // Clone of blockingWait_ from ProcessGroupCCL.
  #if CCL_MINOR_VERSION < 14
    bool blockingWait_ = true;
  #else
    bool blockingWait_ = false;
  #endif
    // Clone of useSameStream_ from ProcessGroupCCL.
    bool useSameStream_ = false;

  protected:
    friend class ProcessGroupCCL;
    const std::vector<std::vector<at::Tensor>> outputTensors_;
    // The future returned by getFuture.
    c10::intrusive_ptr<at::ivalue::Future> future_;
  };

  explicit ProcessGroupCCL(const c10::intrusive_ptr<Store>& store,
                           int rank,
                           int size,
                           std::chrono::milliseconds);
  virtual ~ProcessGroupCCL();

#if TORCH_VERSION_MINOR >= 11
  const std::string getBackendName() const override {
    return std::string(CCL_BACKEND_NAME);
  }
#endif

  void startCoalescing() override;

  c10::intrusive_ptr<Work> endCoalescing() override;

  c10::intrusive_ptr<C10D_Work> broadcast(
      std::vector<at::Tensor>& data,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  c10::intrusive_ptr<C10D_Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<C10D_Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<C10D_Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  c10::intrusive_ptr<C10D_Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<C10D_Work> _allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<C10D_Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<C10D_Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override;

  c10::intrusive_ptr<C10D_Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override;

  c10::intrusive_ptr<C10D_Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;
  
  c10::intrusive_ptr<C10D_Work> _reduce_scatter_base(
          at::Tensor& outputBuffer,
          at::Tensor& inputBuffer,
          const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  c10::intrusive_ptr<C10D_Work> alltoall_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<C10D_Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<C10D_Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;

  c10::intrusive_ptr<C10D_Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

  c10::intrusive_ptr<C10D_Work> recvAnysource(
      std::vector<at::Tensor>& tensor,
      int tag) override;

  c10::intrusive_ptr<C10D_Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

  void groupStart();

  void groupEnd();

  // create a new ProcessGroupCCL and initialize CCL if not initialized
  #if TORCH_VERSION_MAJOR > 1
  static c10::intrusive_ptr<Backend> createProcessGroupCCL(
  #else
  static c10::intrusive_ptr<ProcessGroup> createProcessGroupCCL(
  #endif
      const c10::intrusive_ptr<Store>& store,
      int rank = -1,
      int size = -1,
      std::chrono::milliseconds op_time_out = kNoTimeout);
  static const int64_t OP_TIMEOUT_MILLIS;
 public:

  static void cclInitOnce();
  static void cclFini();

  // Store that is used to exchange information between processes.
  c10::intrusive_ptr<Store> store_;

  std::chrono::milliseconds timeout;

  std::unique_ptr<oneccl_bindings_for_pytorch::CCLCommCollector> ccl_member_;

  static std::mutex globalMutex;

  // Whether or not wait() and synchronize() are blocking operations that wait
  // for the operation to complete.
#if CCL_MINOR_VERSION < 14
  bool blockingWait_ = true;
#else
  bool blockingWait_ = false;
#endif

  // Environment variable which controls whether to keep same stream
  // for collectives and compute
  bool useSameStream_ = false;

  bool torch_llm_allreduce_ = false;

  // Flag to denote if a coalescing groupStart/groupEnd block is active
  bool is_coalescing_ = false;

  // Stores device indexes for all collectives run inside a coalescing block
  std::vector<at::Device> coalescedDevices_;

  // The number of active groupStart() calls. This counter will be increased
  // by 1 when groupStart() is called and decreased by 1 when group_end()
  // is called.
  static thread_local uint64_t cclActiveGroupCounter_;
};

} // namespace c10d
