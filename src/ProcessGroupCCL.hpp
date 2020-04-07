/*
 * Copyright (c) 2020, Intel Corporation
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

#include <deque>
#include <exception>
#include <memory>
#include <mutex>

#ifndef PROCESS_GROUP_CCL_TEST
#include <pybind11/chrono.h>
#endif

#include <thread>
#include <vector>

#include <c10d/ProcessGroup.hpp>
#include <c10d/Store.hpp>
#include <c10d/Types.hpp>
#include <c10d/Utils.hpp>
#include <ccl.hpp>

#ifndef PROCESS_GROUP_CCL_TEST
#include <torch/extension.h>
#endif

#define USE_VECTOR_ALLGATHERV
//#define USE_CACHE

namespace c10d
{

// WorkCCL is the state associated with a CCL operarion.
//
// ProcessGroupCCL implements CCL bindings for c10d.
//
// All functions on this class are expected to be called in the same
// order across processes in the group.
//
// All collective functions provided by this class is scheduled for asynchronous execution by CCL.
//
// Also note that ProcessGroupCCL only supports a single Tensor operation. In
// other words, the size of the input Tensor vector should always be 1.
//

class ProcessGroupCCL : public ProcessGroup
{

public:

  class WorkCCL : public ProcessGroup::Work
  {
  public:

      WorkCCL() {}
      WorkCCL(std::shared_ptr<ccl::request> req,
              const std::vector<at::Tensor>& tensors) :
          req(req),
          tensors(tensors)
      {}

      template<class ...Args>
      WorkCCL(std::shared_ptr<ccl::request> req,
              Args&& ...args) :
          req(req),
          tensors(std::forward<Args>(args)...)
      {}


      virtual ~WorkCCL();

      bool isCompleted() override;
      bool isSuccess() const override;
      bool wait() override;
      void abort() override;

  protected:
      std::shared_ptr<ccl::request> req;

      /*
          keep copy of tensors to incrememt tensor reference counters
          while CCL operation is in progress
      */
      std::vector<at::Tensor> tensors;

      friend class ProcessGroupCCL;
  };

  explicit ProcessGroupCCL(int rank = -1, int size = -1);
  virtual ~ProcessGroupCCL();

  std::shared_ptr<ProcessGroup::Work> broadcast(
      std::vector<at::Tensor>& data,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override;

  std::shared_ptr<ProcessGroup::Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> allgather_coalesced(
      std::vector<std::vector<at::Tensor>>& outputTensorLists,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override;

  std::shared_ptr<ProcessGroup::Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override;

  std::shared_ptr<ProcessGroup::Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  std::shared_ptr<ProcessGroup::Work> alltoall_base(
      at::Tensor& outputTensor,
      at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  std::shared_ptr<ProcessGroup::Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  std::shared_ptr<ProcessGroup::Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag) override;

  std::shared_ptr<ProcessGroup::Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag) override;

  std::shared_ptr<ProcessGroup::Work> recvAnysource(
      std::vector<at::Tensor>& tensor,
      int tag) override;

  std::shared_ptr<ProcessGroup::Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

  // create a new ProcessGroupCCL and initialize CCL if not initialized
  static std::shared_ptr<ProcessGroup> createProcessGroupCCL(
      const std::shared_ptr<Store>& store,
      int rank,
      int size,
      const std::chrono::duration<float>& timeout);

#ifndef PROCESS_GROUP_CCL_TEST
  static void ProcessGroupCCLConstructor() __attribute__((constructor))
  {
      py::object register_backend =
          py::module::import("torch.distributed").attr("Backend").attr("register_backend");
      register_backend("ccl", py::cpp_function(createProcessGroupCCL));
  }
#endif

 protected:

  static void cclInitOnce();
  static void cclFini();

  ccl::communicator* comm;

#ifdef USE_VECTOR_ALLGATHERV
  static thread_local std::vector<void*> agRecvBuffers;
#endif
};

} // namespace c10d
