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

#include "utils.h"
#include "ProcessGroupCCL.hpp"

namespace oneccl_bindings_for_pytorch {

using namespace c10d;

constexpr typename std::underlying_type<c10::DeviceType>::type to_int(c10::DeviceType dev_type) noexcept {
  return static_cast<typename std::underlying_type<c10::DeviceType>::type>(dev_type);
}

class DispatchStub {

public:
  DispatchStub() {}

  static c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> allreduce(std::vector<at::Tensor>& tensors,
                                                                  const AllreduceOptions& opts,
                                                                  ProcessGroupCCL& pg_ccl);

  static c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> reduce(std::vector<at::Tensor>& tensors,
                                                               const ReduceOptions& opts,
                                                               ProcessGroupCCL& pg_ccl);

  static c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> broadcast(std::vector<at::Tensor>& tensors,
                                                                  const BroadcastOptions& opts,
                                                                  ProcessGroupCCL& pg_ccl);

  static c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> allgather(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                                  std::vector<at::Tensor>& inputTensors,
                                                                  const AllgatherOptions& opts,
                                                                  ProcessGroupCCL& pg_ccl);

  static c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> _allgather_base(
                                                                at::Tensor& outputBuffer,
                                                                at::Tensor& inputBuffer,
                                                                const AllgatherOptions& opts,
                                                                ProcessGroupCCL& pg_ccl);

  static c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> gather(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                               std::vector<at::Tensor>& inputTensors,
                                                               const GatherOptions& opts,
                                                               ProcessGroupCCL& pg_ccl);

  static c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> scatter(std::vector<at::Tensor>& outputTensors,
                                                             std::vector<std::vector<at::Tensor>>& inputTensors,
                                                             const ScatterOptions& opts,
                                                             ProcessGroupCCL& pg_ccl);

  static c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> _reduce_scatter_base(at::Tensor & outputTensor,
                                                                  at::Tensor& inputTensor,
                                                                  const ReduceScatterOptions& opts,
                                                                  ProcessGroupCCL& pg_ccl);

  static c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> alltoall_base(at::Tensor& outputTensor,
                                                                      at::Tensor& inputTensor,
                                                                      std::vector<int64_t>& outputSplitSizes,
                                                                      std::vector<int64_t>& inputSplitSizes,
                                                                      const AllToAllOptions& opts,
                                                                      ProcessGroupCCL& pg_ccl);
  
  static c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> alltoall(std::vector<at::Tensor>& outputTensors,
                                                                 std::vector<at::Tensor>& inputTensors,
                                                                 const AllToAllOptions& opts,
                                                                 ProcessGroupCCL& pg_ccl);
                                                                
  static c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> send(std::vector<at::Tensor>& tensors,
                                                                int dstRank,
                                                                int tag,
                                                                ProcessGroupCCL& pg_ccl);

  static c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> recv(std::vector<at::Tensor>& tensors,
                                                                int srcRank,
                                                                int tag,
                                                                ProcessGroupCCL& pg_ccl);  

  static c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> barrier(const BarrierOptions& opts,
                                                                ProcessGroupCCL& pg_ccl);

  static void reset_all();

  virtual ~DispatchStub() {};

  static void register_ccl_stub(c10::DeviceType devoce_type, DispatchStub* stub);

  static DispatchStub* get_ccl_stub(c10::DeviceType dev_type);

  virtual void reset() {};

  virtual c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> reduce_(std::vector<at::Tensor>& tensors,
                                                                 const ReduceOptions& opts,
                                                                 ProcessGroupCCL& pg_ccl) {
    fail(tensors[0].device().type(), "reduce");
    return c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL>();
  }

  virtual c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> allgather_(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                                    std::vector<at::Tensor>& inputTensors,
                                                                    const AllgatherOptions& opts,
                                                                    ProcessGroupCCL& pg_ccl) {

    fail(inputTensors[0].device().type(), "allgather");
    return c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL>();
  }

  virtual c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> _allgather_base_(at::Tensor& outputTensor,
                                                                        at::Tensor& inputTensor,
                                                                        const AllgatherOptions& opts,
                                                                        ProcessGroupCCL& pg_ccl)  {

      fail(inputTensor.device().type(), "_allgather_base");
      return c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL>();
  }

  virtual c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> gather_(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                                 std::vector<at::Tensor>& inputTensors,
                                                                 const GatherOptions& opts,
                                                                 ProcessGroupCCL& pg_ccl) {
    fail(inputTensors[0].device().type(), "gather");
    return c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL>();
  }

  virtual c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> scatter_(std::vector<at::Tensor>& outputTensors,
                                                                  std::vector<std::vector<at::Tensor>>& inputTensors,
                                                                  const ScatterOptions& opts,
                                                                  ProcessGroupCCL& pg_ccl){
    fail(outputTensors[0].device().type(), "scatter");
    return c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL>();
  }
  
  virtual c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> _reduce_scatter_base_(at::Tensor& outputTensor,
                                                                    at::Tensor& inputTensor,
                                                                    const ReduceScatterOptions& opts,
                                                                    ProcessGroupCCL& pg_ccl) {

    fail(inputTensor.device().type(), "_reduce_scatter_base");
    return c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL>();
  }

  virtual c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> broadcast_(std::vector<at::Tensor>& tensors,
                                                                    const BroadcastOptions& opts,
                                                                    ProcessGroupCCL& pg_ccl) {
    fail(tensors[0].device().type(), "broadcast");
    return c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL>();
  }

  virtual c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> alltoall_base_(at::Tensor& outputTensor,
                                                                        at::Tensor& inputTensor,
                                                                        std::vector<int64_t>& outputSplitSizes,
                                                                        std::vector<int64_t>& inputSplitSizes,
                                                                        const AllToAllOptions& opts,
                                                                        ProcessGroupCCL& pg_ccl) {
    fail(inputTensor[0].device().type(), "alltoall_base");
    return c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL>();
  }

  virtual c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> alltoall_(std::vector<at::Tensor>& outputTensors,
                                                                   std::vector<at::Tensor>& inputTensors,
                                                                   const AllToAllOptions& opts,
                                                                   ProcessGroupCCL& pg_ccl) {
    fail(inputTensors[0].device().type(), "alltoall");
    return c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL>();
  }

  virtual c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> barrier_(const BarrierOptions& opts,
                                                                  ProcessGroupCCL& pg_ccl) {
    fail(c10::DeviceType::XPU, "barrier");
    return c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL>();
  }

  virtual c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> allreduce_(std::vector<at::Tensor>& tensors,
                                                                    const AllreduceOptions& opts,
                                                                    ProcessGroupCCL& pg_ccl) {
    fail(tensors[0].device().type(), "allreduce");
    return c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL>();
  }

  virtual c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> send_(std::vector<at::Tensor>& tensors,
                                                                int dstRank,
                                                                int tag,
                                                                ProcessGroupCCL& pg_ccl) {
    fail(tensors[0].device().type(),"send");
    return c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL>();                                                            
  }

  virtual c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> recv_(std::vector<at::Tensor>& tensors,
                                                                int srcRank,
                                                                int tag,
                                                                ProcessGroupCCL& pg_ccl) {
    fail(tensors[0].device().type(),"recv");
    return c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL>();                                                            
  }
private:
  static void fail(c10::DeviceType dev_type, const std::string method) {
    TORCH_CHECK(false, "oneccl_bindings_for_pytorch: ", method, " isn't implementd on backend [", dev_type, "].");
  }
};

}
