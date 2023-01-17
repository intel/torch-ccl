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

#include <chrono>
#include "env.h"
#include "dispatch_stub.h"


namespace oneccl_bindings_for_pytorch {

using namespace c10d;

static DispatchStub default_stubs;
constexpr DispatchStub* default_stubs_addr = &default_stubs;
constexpr auto num_dev_type = static_cast<std::underlying_type<c10::DeviceType>::type>(c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES);

static void format_tensors_size(std::ostream& os, const at::Tensor& tensor) {
  os << "(" << tensor.device() << ", " << tensor.sizes() << ")";
}

template <typename T>
static void format_tensors_size(std::ostream& os, const std::vector<T>& vec) {
  int i = 0;
  os << "[";
  for (const auto& elem : vec) {
    if (i++ > 0)
      os << ", ";
    format_tensors_size(os, elem);
  }
  os << "]";
}

static void format_pg_rank_with_number(std::ostream& os, const ProcessGroupCCL& pg_ccl, const int64_t& curr_ccl_primitive_number) {
  os << "[" <<pg_ccl.getRank() << "/" <<  pg_ccl.getSize() << "/" << curr_ccl_primitive_number << "]";
}

static void format_time_elapsed(std::ostream& os, const std::chrono::microseconds& duration) {
  os << " Time elapsed (ms): " << duration.count() / 1000.0F;
}

class DebugCCLStub final: public DispatchStub {

public:

  DebugCCLStub(c10::DeviceType dev_type, DispatchStub* stub) : dev_type(dev_type), hdlr(stub), ccl_primitive_number(0) {}

  ~DebugCCLStub() {}

protected:

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> allreduce_(std::vector<at::Tensor>& tensors,
                                                            const AllreduceOptions& opts,
                                                            ProcessGroupCCL& pg_ccl) override {
    std::stringstream os;
    os << "oneccl_bindings_for_pytorch::" << dev_type << "::allreduce: ";
    format_pg_rank_with_number(os, pg_ccl, ccl_primitive_number++);
    os << " ";
    format_tensors_size(os, tensors);
    std::cout << os.str() << std::endl;

    auto workStartTime_ = std::chrono::steady_clock::now();
    auto work = hdlr->allreduce_(tensors, opts, pg_ccl);
    auto currentTimepoint = std::chrono::steady_clock::now();
    auto timeElapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(
        currentTimepoint - workStartTime_);
    format_time_elapsed(os, timeElapsed);
    std::cout << os.str() << std::endl;
    return work;
  }

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> reduce_(std::vector<at::Tensor>& tensors,
                                                         const ReduceOptions& opts,
                                                         ProcessGroupCCL& pg_ccl) override {
    std::stringstream os;
    os << "oneccl_bindings_for_pytorch::" << dev_type << "::reduce: ";
    format_pg_rank_with_number(os, pg_ccl, ccl_primitive_number++);
    os << " ";
    format_tensors_size(os, tensors);
    std::cout << os.str() << std::endl;

    auto workStartTime_ = std::chrono::steady_clock::now();
    auto work = hdlr->reduce_(tensors, opts, pg_ccl);
    auto currentTimepoint = std::chrono::steady_clock::now();
    auto timeElapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(
        currentTimepoint - workStartTime_);
    format_time_elapsed(os, timeElapsed);
    std::cout << os.str() << std::endl;
    return work;
  }

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> _reduce_scatter_base_(at::Tensor& outputTensor,
                                                                          at::Tensor& inputTensor,
                                                                          const ReduceScatterOptions& opts,
                                                                          ProcessGroupCCL& pg_ccl) override {
    std::stringstream os;
    os << "oneccl_bindings_for_pytorch::" << dev_type << "::_reduce_scatter_base: ";
    format_pg_rank_with_number(os, pg_ccl, ccl_primitive_number++);
    os << " input ";
    format_tensors_size(os, inputTensor);
    os << " output ";
    format_tensors_size(os, outputTensor);
    std::cout << os.str() << std::endl;

    auto workStartTime_ = std::chrono::steady_clock::now();
    auto work = hdlr->_reduce_scatter_base_(outputTensor, inputTensor, opts, pg_ccl);
    auto currentTimepoint = std::chrono::steady_clock::now();
    auto timeElapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(
        currentTimepoint - workStartTime_);
    format_time_elapsed(os, timeElapsed);
    std::cout << os.str() << std::endl;
    return work;
  }

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> allgather_(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                            std::vector<at::Tensor>& inputTensors,
                                                            const AllgatherOptions& opts,
                                                            ProcessGroupCCL& pg_ccl) override {
    std::stringstream os;
    os << "oneccl_bindings_for_pytorch::" << dev_type << "::allgather: ";
    format_pg_rank_with_number(os, pg_ccl, ccl_primitive_number++);
    os << " input ";
    format_tensors_size(os, inputTensors);
    os << " output ";
    format_tensors_size(os, outputTensors);
    std::cout << os.str() << std::endl;

    auto workStartTime_ = std::chrono::steady_clock::now();
    auto work = hdlr->allgather_(outputTensors, inputTensors, opts, pg_ccl);
    auto currentTimepoint = std::chrono::steady_clock::now();
    auto timeElapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(
        currentTimepoint - workStartTime_);
    format_time_elapsed(os, timeElapsed);
    std::cout << os.str() << std::endl;
    return work;
  }

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> gather_(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                         std::vector<at::Tensor>& inputTensors,
                                                         const GatherOptions& opts,
                                                         ProcessGroupCCL& pg_ccl) override {
    std::stringstream os;
    os << "oneccl_bindings_for_pytorch::" << dev_type << "::gather: ";
    format_pg_rank_with_number(os, pg_ccl, ccl_primitive_number++);
    os << " input ";
    format_tensors_size(os, inputTensors);
    os << " output ";
    format_tensors_size(os, outputTensors);
    std::cout << os.str() << std::endl;

    auto workStartTime_ = std::chrono::steady_clock::now();
    auto work = hdlr->gather_(outputTensors, inputTensors, opts, pg_ccl);
    auto currentTimepoint = std::chrono::steady_clock::now();
    auto timeElapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(
        currentTimepoint - workStartTime_);
    format_time_elapsed(os, timeElapsed);
    std::cout << os.str() << std::endl;
    return work;
  }

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> _allgather_base_(at::Tensor& outputTensor,
                                                                at::Tensor& inputTensor,
                                                                const AllgatherOptions& opts,
                                                                ProcessGroupCCL& pg_ccl) override {
    std::stringstream os;
    os << "oneccl_bindings_for_pytorch::" << dev_type << "::_allgather_base: ";
    format_pg_rank_with_number(os, pg_ccl, ccl_primitive_number++);
    os << " input ";
    format_tensors_size(os, inputTensor);
    os << " output ";
    format_tensors_size(os, outputTensor);
    std::cout << os.str() << std::endl;

    auto workStartTime_ = std::chrono::steady_clock::now();
    auto work = hdlr->_allgather_base_(outputTensor, inputTensor, opts, pg_ccl);
    auto currentTimepoint = std::chrono::steady_clock::now();
    auto timeElapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(
        currentTimepoint - workStartTime_);
    format_time_elapsed(os, timeElapsed);
    std::cout << os.str() << std::endl;
    return work;
  }

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> scatter_(std::vector<at::Tensor>& outputTensors,
                                                          std::vector<std::vector<at::Tensor>>& inputTensors,
                                                          const ScatterOptions& opts,
                                                          ProcessGroupCCL& pg_ccl) override {
    std::stringstream os;
    os << "oneccl_bindings_for_pytorch::" << dev_type << "::scatter: ";
    format_pg_rank_with_number(os, pg_ccl, ccl_primitive_number++);
    os << " input ";
    format_tensors_size(os, inputTensors);
    os << " output ";
    format_tensors_size(os, outputTensors);
    std::cout << os.str() << std::endl;

    auto workStartTime_ = std::chrono::steady_clock::now();
    auto work = hdlr->scatter_(outputTensors, inputTensors, opts, pg_ccl);
    auto currentTimepoint = std::chrono::steady_clock::now();
    auto timeElapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(
        currentTimepoint - workStartTime_);
    format_time_elapsed(os, timeElapsed);
    std::cout << os.str() << std::endl;
    return work;
  }

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> broadcast_(std::vector<at::Tensor>& tensors,
                                                            const BroadcastOptions& opts,
                                                            ProcessGroupCCL& pg_ccl) override {
    std::stringstream os;
    os << "oneccl_bindings_for_pytorch::" << dev_type << "::broadcast: ";
    format_pg_rank_with_number(os, pg_ccl, ccl_primitive_number++);
    os << " ";
    format_tensors_size(os, tensors);
    std::cout << os.str() << std::endl;

    auto workStartTime_ = std::chrono::steady_clock::now();
    auto work = hdlr->broadcast_(tensors, opts, pg_ccl);
    auto currentTimepoint = std::chrono::steady_clock::now();
    auto timeElapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(
        currentTimepoint - workStartTime_);
    format_time_elapsed(os, timeElapsed);
    std::cout << os.str() << std::endl;
    return work;
  }
  
  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> _reduce_scatter_base_(at::Tensor& outputTensor,
                                                                at::Tensor& inputTensor,
                                                                const ReduceScatterOptions& opts,
                                                                ProcessGroupCCL& pg_ccl) {
      c10::DeviceType dev_type = inputTensor.device().type();
      return get_ccl_stub(dev_type)->_reduce_scatter_base_(outputTensor, inputTensor, opts, pg_ccl);
  }

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> alltoall_base_(at::Tensor& outputTensor,
                                                                at::Tensor& inputTensor,
                                                                std::vector<int64_t>& outputSplitSizes,
                                                                std::vector<int64_t>& inputSplitSizes,
                                                                const AllToAllOptions& opts,
                                                                ProcessGroupCCL& pg_ccl) override {
    std::stringstream os;
    os << "oneccl_bindings_for_pytorch::" << dev_type << "::alltoall_base: ";
    format_pg_rank_with_number(os, pg_ccl, ccl_primitive_number++);
    os << " input ";
    format_tensors_size(os, inputTensor);
    os << " output ";
    format_tensors_size(os, outputTensor);
    os << " inputSplitSizes [" << inputSplitSizes << "]";
    os << " outputSplitSizes [" << outputSplitSizes << "]";
    std::cout << os.str() << std::endl;

    auto workStartTime_ = std::chrono::steady_clock::now();
    auto work = hdlr->alltoall_base_(outputTensor, inputTensor, outputSplitSizes, inputSplitSizes, opts, pg_ccl);
    auto currentTimepoint = std::chrono::steady_clock::now();
    auto timeElapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(
        currentTimepoint - workStartTime_);
    format_time_elapsed(os, timeElapsed);
    std::cout << os.str() << std::endl;
    return work;
  }

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> alltoall_(std::vector<at::Tensor>& outputTensors,
                                                           std::vector<at::Tensor>& inputTensors,
                                                           const AllToAllOptions& opts,
                                                           ProcessGroupCCL& pg_ccl) override {
    std::stringstream os;
    os << "oneccl_bindings_for_pytorch::" << dev_type << "::alltoall: ";
    format_pg_rank_with_number(os, pg_ccl, ccl_primitive_number++);
    os << " inputs ";
    format_tensors_size(os, inputTensors);
    os << " outputs ";
    format_tensors_size(os, outputTensors);
    std::cout << os.str() << std::endl;

    auto workStartTime_ = std::chrono::steady_clock::now();
    auto work = hdlr->alltoall_(outputTensors, inputTensors, opts, pg_ccl);
    auto currentTimepoint = std::chrono::steady_clock::now();
    auto timeElapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(
        currentTimepoint - workStartTime_);
    format_time_elapsed(os, timeElapsed);
    std::cout << os.str() << std::endl;
    return work;
  }

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> barrier_(const BarrierOptions& opts,
                                                          ProcessGroupCCL& pg_ccl) override {
    std::stringstream os;
    os << "oneccl_bindings_for_pytorch::" << dev_type << "::barrier: ";
    format_pg_rank_with_number(os, pg_ccl, ccl_primitive_number++);
    std::cout << os.str() << std::endl;

    auto workStartTime_ = std::chrono::steady_clock::now();
    auto work = hdlr->barrier_(opts, pg_ccl);
    auto currentTimepoint = std::chrono::steady_clock::now();
    auto timeElapsed =
      std::chrono::duration_cast<std::chrono::microseconds>(
        currentTimepoint - workStartTime_);
    format_time_elapsed(os, timeElapsed);
    std::cout << os.str() << std::endl;
    return work;
  }
private:
  c10::DeviceType dev_type;
  DispatchStub* hdlr;
  int64_t ccl_primitive_number;
};



std::vector<DispatchStub*>& get_dispatch_stub(){
  static std::vector<DispatchStub*> dispatch_stubs;
  return dispatch_stubs;
}

void DispatchStub::register_ccl_stub(c10::DeviceType dev_type, DispatchStub* stub) {
  static std::once_flag dispatch_once_flag;
  std::vector<DispatchStub*>& dispatch_stubs = get_dispatch_stub();
  std::call_once(dispatch_once_flag, [&]() {
    dispatch_stubs.resize(num_dev_type, default_stubs_addr);
  });

  auto stub_idx = to_int(dev_type);
  TORCH_CHECK(stub_idx < dispatch_stubs.size(), "unknown device type [", dev_type, "].");
  TORCH_CHECK(dispatch_stubs[stub_idx] == default_stubs_addr, "device type [", dev_type, "] ccl stub has already been registered.");

  if (oneccl_bindings_for_pytorch_verbose())
    stub = new DebugCCLStub(dev_type, stub);

  dispatch_stubs[stub_idx] = stub;
}

DispatchStub* DispatchStub::get_ccl_stub(c10::DeviceType dev_type) {
  auto stub_idx = to_int(dev_type);
  auto dispatch_stubs = get_dispatch_stub();
  TORCH_CHECK(stub_idx < dispatch_stubs.size(), "unknown device type [", dev_type, "].");
  return dispatch_stubs[stub_idx];
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> DispatchStub::allreduce(std::vector<at::Tensor>& tensors,
                                                                       const AllreduceOptions& opts,
                                                                       ProcessGroupCCL& pg_ccl) {
  checkSameType(tensors[0], tensors);
  c10::DeviceType dev_type = tensors[0].device().type();
  return get_ccl_stub(dev_type)->allreduce_(tensors, opts, pg_ccl);
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> DispatchStub::reduce(std::vector<at::Tensor>& tensors,
                                                             const ReduceOptions& opts,
                                                             ProcessGroupCCL& pg_ccl) {
  checkSameType(tensors[0], tensors);
  c10::DeviceType dev_type = tensors[0].device().type();
  return get_ccl_stub(dev_type)->reduce_(tensors, opts, pg_ccl);
}


c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> DispatchStub::_reduce_scatter_base(at::Tensor& outputTensor,
                                                                      at::Tensor& inputTensor,
                                                                      const ReduceScatterOptions& opts,
                                                                      ProcessGroupCCL& pg_ccl) {
  checkSameType(outputTensor, {outputTensor, inputTensor});
  c10::DeviceType dev_type = outputTensor.device().type();
  return get_ccl_stub(dev_type)->_reduce_scatter_base_(outputTensor, inputTensor, opts, pg_ccl);
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> DispatchStub::broadcast(std::vector<at::Tensor>& tensors,
                                                                const BroadcastOptions& opts,
                                                                ProcessGroupCCL& pg_ccl) {
  checkSameType(tensors[0], tensors);
  c10::DeviceType dev_type = tensors[0].device().type();
  return get_ccl_stub(dev_type)->broadcast_(tensors, opts, pg_ccl);
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> DispatchStub::allgather(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                                std::vector<at::Tensor>& inputTensors,
                                                                const AllgatherOptions& opts,
                                                                ProcessGroupCCL& pg_ccl) {
  checkSameType(inputTensors[0], inputTensors);
  checkSameType(inputTensors[0], outputTensors);
  c10::DeviceType dev_type = inputTensors[0].device().type();
  return get_ccl_stub(dev_type)->allgather_(outputTensors, inputTensors, opts, pg_ccl);
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> DispatchStub::_allgather_base(
                                                                at::Tensor& outputTensor,
                                                                at::Tensor& inputTensor,
                                                                const AllgatherOptions& opts,
                                                                ProcessGroupCCL& pg_ccl) {
  checkSameType(inputTensor, std::vector{outputTensor});
  c10::DeviceType dev_type = inputTensor.device().type();
  return get_ccl_stub(dev_type)->_allgather_base_(outputTensor, inputTensor, opts, pg_ccl);
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> DispatchStub::gather(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                             std::vector<at::Tensor>& inputTensors,
                                                             const GatherOptions& opts,
                                                             ProcessGroupCCL& pg_ccl) {
  checkSameType(inputTensors[0], inputTensors);
  checkSameType(inputTensors[0], outputTensors);
  c10::DeviceType dev_type = inputTensors[0].device().type();
  return get_ccl_stub(dev_type)->gather_(outputTensors, inputTensors, opts, pg_ccl);
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> DispatchStub::scatter(std::vector<at::Tensor>& outputTensors,
                                                              std::vector<std::vector<at::Tensor>>& inputTensors,
                                                              const ScatterOptions& opts,
                                                              ProcessGroupCCL& pg_ccl){
  checkSameType(outputTensors[0], inputTensors);
  checkSameType(outputTensors[0], outputTensors);
  c10::DeviceType dev_type = outputTensors[0].device().type();
  return get_ccl_stub(dev_type)->scatter_(outputTensors, inputTensors, opts, pg_ccl);
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> DispatchStub::_reduce_scatter_base(at::Tensor& outputTensor,
                                                                at::Tensor& inputTensor,
                                                                const ReduceScatterOptions& opts,
                                                                ProcessGroupCCL& pg_ccl) {
  c10::DeviceType dev_type = inputTensor.device().type();
  return get_ccl_stub(dev_type)->_reduce_scatter_base_(outputTensor, inputTensor, opts, pg_ccl);
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> DispatchStub::alltoall_base(at::Tensor& outputTensor,
                                                                    at::Tensor& inputTensor,
                                                                    std::vector<int64_t>& outputSplitSizes,
                                                                    std::vector<int64_t>& inputSplitSizes,
                                                                    const AllToAllOptions& opts,
                                                                    ProcessGroupCCL& pg_ccl) {
  checkSameType(inputTensor, {outputTensor});
  c10::DeviceType dev_type = inputTensor.device().type();
  return get_ccl_stub(dev_type)->alltoall_base_(outputTensor, inputTensor, outputSplitSizes, inputSplitSizes, opts, pg_ccl);
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> DispatchStub::alltoall(std::vector<at::Tensor>& outputTensors,
                                                               std::vector<at::Tensor>& inputTensors,
                                                               const AllToAllOptions& opts,
                                                               ProcessGroupCCL& pg_ccl) {
  checkSameType(inputTensors[0], inputTensors);
  checkSameType(inputTensors[0], outputTensors);
  c10::DeviceType dev_type = inputTensors[0].device().type();
  return get_ccl_stub(dev_type)->alltoall_(outputTensors, inputTensors, opts, pg_ccl);
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> DispatchStub::barrier(const BarrierOptions& opts,
                                                              ProcessGroupCCL& pg_ccl) {
  c10::DeviceType dev_type = c10::DeviceType::CPU;
  return get_ccl_stub(dev_type)->barrier_(opts, pg_ccl);
}

void DispatchStub::reset_all() {
  auto dispatch_stubs = get_dispatch_stub();
  for(auto stub: dispatch_stubs) {
    stub->reset();
  }
}

}


