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

#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <ProcessGroupCCL.hpp>
#include <dispatch_stub.h>
#include <ATen/record_function.h>
#include "../utils.h"

namespace oneccl_bindings_for_pytorch
{

namespace {

void checkSameSizeAndType(const at::Tensor& tensor,
                          const std::vector<at::Tensor>& tensors) __attribute__((unused));

void checkSameSizeAndType(const at::Tensor& tensor,
                          const std::vector<at::Tensor>& tensors)
{
  for (size_t i = 0; i < tensors.size(); ++i)
  {
    TORCH_CHECK((tensors[i].numel() == tensor.numel()) &&
                (tensors[i].scalar_type() == tensor.scalar_type()),
                "tensors are not equal in size or data type");

    checkSingleTensorHelper(tensors[i]);
  }
}

Comms& get_ccl_comms(c10d::ProcessGroupCCL& pg, const std::string& devices_key, const std::vector<at::Device>& devices, c10d::OpType op_type = OpType::UNKNOWN, int p2pRank = 0, bool isSendRecvSelf = false) {
  // Sanity check
  if (devices_key.empty()) {
    throw std::runtime_error(
            "Not able to create/get the CCL Communicator since "
            "the devices are empty ");
  }

  TORCH_CHECK(devices.size() == 1, "CPU device size must be 1");

  auto cached_comms = pg.ccl_member_->get_comms(devices_key);
  if (cached_comms) {
    return *cached_comms;
  }

  ccl::vector_class<ccl::communicator> cpu_comms;
  auto kvs = pg.ccl_member_->get_kvs(pg.getRank(), *pg.store_);
  cpu_comms.emplace_back(
    call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&](){
      CCL_CHECK(return ccl::create_communicator(pg.getSize(), pg.getRank(), kvs););
      })
  );
  std::shared_ptr<Comms> cpu_comms_ptr = std::make_shared<Comms>(cpu_comms);
  pg.ccl_member_->add_comms(devices_key, cpu_comms_ptr);

  return *cpu_comms_ptr.get();
}

template <typename RunF, typename CommType, typename InputType, typename OutputType, typename attr_t>
class CPUWorkCCL : public CollectiveAsyncWorkCCL<RunF, CommType, InputType, OutputType, attr_t> {
public:
  CPUWorkCCL(const std::vector<InputType>& inputs,
                         const std::vector<OutputType>& outputs,
                         const RunF f,
                         CommType& comms,
                         attr_t& attr,
                         std::chrono::milliseconds timeout,
                         int rank,
                         c10d::OpType opType,
                         const char* profilingTitle,
                         const c10::optional<std::vector<at::Tensor>>& inputTensors) :
                         CollectiveAsyncWorkCCL<RunF, CommType, InputType, OutputType, attr_t>(
                                 inputs, outputs, f, comms, attr, timeout, rank, opType, profilingTitle, inputTensors) {}

};

} //namespace anonymous


class VanillaCPU final: public DispatchStub {
public:

  VanillaCPU() {
    stop_=false;
    workerThread_ = std::thread(&VanillaCPU::runLoop, this);
  }

  ~VanillaCPU() {destroy();}


protected:

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> allreduce_(std::vector<at::Tensor>& tensors,
                                                            const AllreduceOptions& opts,
                                                            ProcessGroupCCL& pg) override;

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> allreduce_coalesced_(std::vector<at::Tensor>& tensors,
                                                                    const AllreduceOptions& opts,
                                                                    ProcessGroupCCL& pg) override;


  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> reduce_(std::vector<at::Tensor>& tensors,
                                                         const ReduceOptions& opts,
                                                         ProcessGroupCCL& pg) override;

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> _reduce_scatter_base_(at::Tensor& outputTensor,
                                                                          at::Tensor& inputTensor,
                                                                          const ReduceScatterOptions& opts,
                                                                          ProcessGroupCCL& pg_ccl) override;

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> broadcast_(std::vector<at::Tensor>& tensors,
                                                            const BroadcastOptions& opts,
                                                            ProcessGroupCCL& pg) override;

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> allgather_(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                            std::vector<at::Tensor>& inputTensors,
                                                            const AllgatherOptions& opts,
                                                            ProcessGroupCCL& pg) override;

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> _allgather_base_(at::Tensor& outputTensor,
                                                                     at::Tensor& inputTensor,
                                                                     const AllgatherOptions& opts,
                                                                     ProcessGroupCCL& pg_ccl) override;

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> gather_(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                            std::vector<at::Tensor>& inputTensors,
                                                            const GatherOptions& opts,
                                                            ProcessGroupCCL& pg) override;
  
  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> alltoall_base_(at::Tensor& outputTensor,
                                                               at::Tensor& inputTensor,
                                                               std::vector<int64_t>& outputSplitSizes,
                                                               std::vector<int64_t>& inputSplitSizes,
                                                               const AllToAllOptions& opts,
                                                               ProcessGroupCCL& pg) override;

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> alltoall_(std::vector<at::Tensor>& outputTensors,
                                                             std::vector<at::Tensor>& inputTensors,
                                                             const AllToAllOptions& opts,
                                                             ProcessGroupCCL& pg) override;

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> barrier_(const BarrierOptions& opts,
                                                                ProcessGroupCCL& pg) override;
  void destroy();
  void reset() override {}
  void runLoop();
  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> enqueue(c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> & work);
private:
  bool stop_;
  std::mutex pgMutex_;
  std::thread workerThread_;

  std::deque<c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL>> queue_;
  std::condition_variable queueProduceCV_;
  std::condition_variable queueConsumeCV_;

};

struct RegisterCPUPMethods {
  RegisterCPUPMethods() {
    static VanillaCPU methods;
    DispatchStub::register_ccl_stub(c10::DeviceType::CPU, &methods);
  }
};

class callback_context {
public:
  virtual void run_completion_hook(
      const void* indBuf, size_t indCount, ccl::datatype indDatatype,
      const void* valBuf, size_t valCount, ccl::datatype valDatatype) = 0;
};

template<typename RunF>
class cpu_completion_callback final : public callback_context {
public:
  cpu_completion_callback(RunF cb): f(cb) {};
  void run_completion_hook(
      const void* indBuf, size_t indCount, ccl::datatype indDatatype,
      const void* valBuf, size_t valCount, ccl::datatype valDatatype) override {
    actual_run(indBuf, indCount, indDatatype, valBuf, valCount, valDatatype);
  }

private:
    void actual_run(
        const void* indBuf, size_t indCount, ccl::datatype indDatatype,
        const void* valBuf, size_t valCount, ccl::datatype valDatatype) {
      f(indBuf, indCount, indDatatype, valBuf, valCount, valDatatype);
    }

  RunF f;
};

template <typename RunF>
std::shared_ptr<cpu_completion_callback<RunF>> make_cpu_callback(RunF f) {
  std::shared_ptr<cpu_completion_callback<RunF>> ret_ptr;
  ret_ptr.reset(new cpu_completion_callback<RunF>(f));
  return ret_ptr;
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> VanillaCPU::enqueue(c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> & work) {
  work->run();
  std::unique_lock<std::mutex> lock(pgMutex_);
  queue_.push_back(work);
  lock.unlock();
  queueProduceCV_.notify_one();
  return work;
}

void VanillaCPU::destroy() {
  std::unique_lock<std::mutex> lock(pgMutex_);
  queueConsumeCV_.wait(lock, [&] { return queue_.empty(); });

  // Queue is empty, signal stop
  stop_ = true;

  // Release lock to allow threads to terminate
  lock.unlock();
  queueProduceCV_.notify_all();

  // Join the single worker thread
  workerThread_.join();
}

void VanillaCPU::runLoop() {
  std::unique_lock<std::mutex> lock(pgMutex_);
  while (!stop_) {
    if (queue_.empty()) {
      queueProduceCV_.wait(lock);
      continue;
    }

    auto work = std::move(queue_.front());

    queue_.pop_front();

    lock.unlock();
    queueConsumeCV_.notify_one();

    try {
      work->synchronize();
      work->finishAsyncWorkCCL();

    } catch (...) {
      work->finishAsyncWorkCCLError(std::current_exception());
    }

    lock.lock();
  }
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> VanillaCPU::allreduce_(std::vector<at::Tensor>& tensors,
                                                                      const AllreduceOptions& opts,
                                                                      ProcessGroupCCL& pg) {
  checkSingleTensor(tensors);

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  work = collective<get_ccl_comms, CPUWorkCCL>(
          pg,
          tensors,
          tensors,
          [=](at::Tensor input,
              at::Tensor output,
              ccl::allreduce_attr attr,
              ccl::communicator& comm){
              ccl::event ret_evt;
              call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&](){
                  CCL_CHECK(ret_evt = ccl::allreduce(input.data_ptr(),
                                                     output.data_ptr(),
                                                     (size_t) input.numel(),
                                                     cclDatatypes.at(input.scalar_type()),
                                                     cclOps.at(opts.reduceOp),
                                                     comm,
                                                     attr););
              });
              return ret_evt;
          },
          c10d::OpType::ALLREDUCE,
          "oneccl_bindings_for_pytorch::cpu_work::allreduce");
  work->debugName = std::string("cpu::allreduce");
  enqueue(work);
  return work;
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> VanillaCPU::allreduce_coalesced_(std::vector<at::Tensor>& tensors,
                                                                const AllreduceOptions& opts,
                                                                ProcessGroupCCL& pg) {
  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  work = collective<get_ccl_comms, CPUWorkCCL>(
          pg,
          tensors,
          tensors,
          [=](at::Tensor input,
              at::Tensor output,
              ccl::allreduce_attr attr,
              ccl::communicator& comm){
              ccl::event ret_evt;
              call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&](){
                  CCL_CHECK(ret_evt = ccl::allreduce(input.data_ptr(),
                                                     output.data_ptr(),
                                                     (size_t) input.numel(),
                                                     cclDatatypes.at(input.scalar_type()),
                                                     cclOps.at(opts.reduceOp),
                                                     comm,
                                                     attr););
              });
              return ret_evt;
          },
          c10d::OpType::ALLREDUCE,
          "oneccl_bindings_for_pytorch::cpu_work::allreduce_coalesced_");
  work->debugName = std::string("cpu::allreduce_coalesced");
  enqueue(work);
  return work;
}


c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> VanillaCPU::reduce_(std::vector<at::Tensor>& tensors,
                                                                   const ReduceOptions& opts,
                                                                   ProcessGroupCCL& pg) {
  checkSingleTensor(tensors);

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  work = collective<get_ccl_comms, CPUWorkCCL>(
          pg,
          tensors,
          tensors,
          [=](at::Tensor input,
              at::Tensor output,
              ccl::reduce_attr attr,
              ccl::communicator& comm) {
              ccl::event ret_evt;
              call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&](){
                  CCL_CHECK(ret_evt = ccl::reduce(input.data_ptr(),
                                                  output.data_ptr(),
                                                  (size_t)input.numel(),
                                                  cclDatatypes.at(input.scalar_type()),
                                                  cclOps.at(opts.reduceOp),
                                                  (int)opts.rootRank,
                                                  comm,
                                                  attr););
              });
              return ret_evt;
          },
          c10d::OpType::REDUCE,
          "oneccl_bindings_for_pytorch::cpu_work::reduce");

  work->debugName = std::string("cpu::reduce");
  enqueue(work);
  return work;
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> VanillaCPU::broadcast_(std::vector<at::Tensor>& tensors,
                                                                      const BroadcastOptions &opts,
                                                                      ProcessGroupCCL& pg) {
  checkSingleTensor(tensors);

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  work = collective<get_ccl_comms, CPUWorkCCL>(
          pg,
          tensors,
          tensors,
          [=](at::Tensor input,
              at::Tensor /*output*/,
              ccl::broadcast_attr attr,
              ccl::communicator& comm) {
              ccl::event ret_evt;
              call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&](){
                  CCL_CHECK(ret_evt = ccl::broadcast(input.data_ptr(),
                                                     (size_t) input.numel(),
                                                     cclDatatypes.at(input.scalar_type()),
                                                     (size_t) opts.rootRank,
                                                     comm));
              });
              return ret_evt;
          },
          c10d::OpType::BROADCAST,
          "oneccl_bindings_for_pytorch::cpu_work::broadcast");

  work->debugName = std::string("cpu::broadcast");
  enqueue(work);
  return work;
}


c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> VanillaCPU::allgather_(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                                      std::vector<at::Tensor>& inputTensors,
                                                                      const AllgatherOptions& opts,
                                                                      ProcessGroupCCL& pg) {
  checkSingleTensor(inputTensors);
  TORCH_CHECK(static_cast<size_t>(pg.getSize()) == outputTensors[0].size(),
              "allgather: number of output tensors should equal to the world size");

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  int size = pg.getSize();
  int rank = pg.getRank();
  work = collective<get_ccl_comms, CPUWorkCCL>(
    pg,
    inputTensors,
    outputTensors,
    [=](at::Tensor input,
        const std::vector<at::Tensor>& outputs,
        ccl::allgatherv_attr attr,
        ccl::communicator& comm) {
        ccl::event ret_evt;
        std::vector<size_t> recvCounts(size, 0);

        auto flatRes = computeLengthsAndCheckFlat(outputs, recvCounts);

        TORCH_CHECK((size_t)input.numel() == recvCounts[rank],
                    "allgather: send and recv count doesn't match");

        if (flatRes.isFlat) {
          void* recvBuf = flatRes.firstTensor.data_ptr();

          call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&](){
              CCL_CHECK(ret_evt = ccl::allgatherv(input.data_ptr(),
                                                  (size_t) input.numel(),
                                                  recvBuf,
                                                  recvCounts,
                                                  cclDatatypes.at(input.scalar_type()),
                                                  comm,
                                                  attr););
          });
        }
        else {
          std::vector<void*> recvBufs;
          std::transform(outputs.begin(), outputs.end(),
                         std::back_inserter(recvBufs),
                         [](const at::Tensor& t) { return t.data_ptr(); } );

          call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&](){
              CCL_CHECK(ret_evt = ccl::allgatherv(input.data_ptr(),
                                                  (size_t) input.numel(),
                                                  recvBufs,
                                                  recvCounts,
                                                  cclDatatypes.at(input.scalar_type()),
                                                  comm,
                                                  attr););

          });
        }

        return ret_evt;
    },
    c10d::OpType::ALLGATHER,
    "oneccl_bindings_for_pytorch::cpu_work::allgather");

  work->debugName = std::string("cpu::allgather");
  enqueue(work);
  return work;
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> VanillaCPU::_allgather_base_(at::Tensor& outputTensor,
                                                                               at::Tensor& inputTensor,
                                                                               const AllgatherOptions& opts,
                                                                               ProcessGroupCCL& pg_ccl) {
  const int world_size = pg_ccl.getSize();
  if (inputTensor.numel() * world_size != outputTensor.numel()) {
    TORCH_CHECK(false, "output tensor size must be equal to world_size times input tensor size");
  }

  // just a wrapper to fit the collective interface
  auto inputs = std::vector<at::Tensor> {inputTensor};
  auto outputs = std::vector<at::Tensor> {outputTensor};

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  work = collective<get_ccl_comms, CPUWorkCCL>(
          pg_ccl,
          inputs,
          outputs,
          [=](at::Tensor input,
              at::Tensor output,
              ccl::allgatherv_attr attr,
              ccl::communicator& comm) {
            RECORD_FUNCTION("oneccl_bindings_for_pytorch::cpu::_allgather_base", std::vector<c10::IValue>({input}));

            std::vector<size_t> recvCounts(world_size, input.numel());

            ccl::event ret_evt;
            call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&]() {
              CCL_CHECK(ret_evt = ccl::allgatherv(input.data_ptr(),
                                                  (size_t) input.numel(),
                                                  output.data_ptr(),
                                                  recvCounts,
                                                  cclDatatypes.at(input.scalar_type()),
                                                  comm,
                                                  attr));
            });
            return ret_evt;
          },
          c10d::OpType::_ALLGATHER_BASE);
  work->debugName = std::string("cpu::_allgather_base");
  enqueue(work);
  return work;
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> VanillaCPU::gather_(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                                      std::vector<at::Tensor>& inputTensors,
                                                                      const GatherOptions& opts,
                                                                      ProcessGroupCCL& pg) {
  checkSingleTensor(inputTensors);
  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  auto grp_size = pg.getSize();
  auto rank = pg.getRank();

  if (rank != opts.rootRank)
  {
      TORCH_CHECK(outputTensors.size() == 0,
          "gather: number of output tensors should be 0 "
          "for non-root");
  }
  else
  {
      TORCH_CHECK(outputTensors.size() == 1,
          "gather: multi-GPU collective is not supported");

      TORCH_CHECK(static_cast<size_t>(grp_size) == outputTensors[0].size(),
          "gather: number of output tensors should equal "
          "to the world size");
  }
  work = collective<get_ccl_comms, CPUWorkCCL>(
      pg,
      inputTensors,
      outputTensors,
      [=](at::Tensor input,
          const std::vector<at::Tensor>& outputs,
          ccl::alltoallv_attr attr,
          ccl::communicator& comm) {

      std::vector<size_t> sendCounts(grp_size, 0);
      std::vector<size_t> recvCounts(grp_size, 0);
      sendCounts[opts.rootRank] = input.numel();

      at::Tensor flatOutput;
      int64_t flatRecvCount = 0;
       bool isOutputFlat = false;

      if (rank == opts.rootRank)
      {
          isOutputFlat =
              computeLengthsAndCheckAndGetFlat(outputs,
                                               recvCounts, flatOutput, flatRecvCount);
          TORCH_CHECK(sendCounts[rank] == recvCounts[rank],
              "gather: send and recv count doesn't match");
      }
      else
      {
          flatOutput = at::empty({0}, input.options());
      }
      
      ccl::event ret_evt;
      call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&](){
          CCL_CHECK(ret_evt = ccl::alltoallv(input.data_ptr(),
                                             sendCounts,
                                             flatOutput.data_ptr(),
                                             recvCounts,
                                             cclDatatypes.at(flatOutput.scalar_type()),
                                             comm,
                                             attr););
      });


      if (rank == opts.rootRank)
      {
          if (!isOutputFlat)
          {
              ret_evt.wait();
              auto flatOutputSplits =
                  flatOutput.split_with_sizes(c10::IntArrayRef((int64_t*)recvCounts.data(),
                                              recvCounts.size()), 0);

              for (int i = 0; i < grp_size; i++)
              {
                  outputs[i].view({-1}).copy_(flatOutputSplits[i]);
              }
          }
       }

       return ret_evt;
  },
  c10d::OpType::GATHER,
  "oneccl_bindings_for_pytorch::cpu_work::gather");
    
  work->debugName = std::string("cpu::gather");
  enqueue(work);
  return work;
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> VanillaCPU::_reduce_scatter_base_(at::Tensor& outputTensor,
                                                                      at::Tensor& inputTensor,
                                                                      const ReduceScatterOptions& opts,
                                                                      ProcessGroupCCL& pg) {

  checkSingleTensorHelper(inputTensor);
  checkSingleTensorHelper(outputTensor);
  int size = pg.getSize();
  if (inputTensor.dtype() != outputTensor.dtype()) {
    TORCH_CHECK(false, "output tensor must have the same type as input tensor");
  }

  if (outputTensor.numel() * size != inputTensor.numel()) {
    TORCH_CHECK(
        false,
        "input tensor size must be equal to world_size times output tensor size");
  }
  std::vector<at::Tensor> inputs{inputTensor};
  std::vector<at::Tensor> outputs{outputTensor};

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  work = collective<get_ccl_comms, CPUWorkCCL>(
    pg,
    inputs,
    outputs,
    [=](at::Tensor input,
        at::Tensor output,
        ccl::reduce_scatter_attr attr,
        ccl::communicator& comm) {
        ccl::event ret_evt;

        call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&](){
            CCL_CHECK(ret_evt = ccl::reduce_scatter(input.data_ptr(),                                         
                                                output.data_ptr(),
                                                size_t(input.numel()/size),
                                                cclDatatypes.at(input.scalar_type()),
                                                cclOps.at(opts.reduceOp),
                                                comm,
                                                attr););
        });

        return ret_evt;
      },  
    c10d::OpType::_REDUCE_SCATTER_BASE,
    "oneccl_bindings_for_pytorch::cpu_work::_reduce_scatter_base");

  work->debugName = std::string("cpu::_reduce_scatter_base");
  enqueue(work);
  return work;
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> VanillaCPU::alltoall_base_(at::Tensor& outputTensor,
                                                             at::Tensor& inputTensor,
                                                             std::vector<int64_t>& outputSplitSizes,
                                                             std::vector<int64_t>& inputSplitSizes,
                                                             const AllToAllOptions& opts,
                                                             ProcessGroupCCL& pg){
  checkSingleTensorHelper(inputTensor);
  checkSingleTensorHelper(outputTensor);

  std::vector<at::Tensor> inputs{inputTensor};
  std::vector<at::Tensor> outputs{outputTensor};
  auto grp_size = pg.getSize();
  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  if (outputSplitSizes.size() == 0 && inputSplitSizes.size() == 0){
    TORCH_CHECK(outputTensor.numel() == inputTensor.numel() &&
        outputTensor.scalar_type() == inputTensor.scalar_type(),
        "alltoall_base: tensors are not equal in size or data type");

    TORCH_CHECK(outputTensor.size(0) % grp_size == 0,
        "alltoall_base: tensor's dim 0 does not divide equally across group size");
    work = collective<get_ccl_comms, CPUWorkCCL>(
      pg,
      inputs,
      outputs,
      [=](at::Tensor input,
          at::Tensor output,
          ccl::alltoall_attr attr,
          ccl::communicator& comm) {
            ccl::event ret_evt;
            call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&](){
                CCL_CHECK(ret_evt = ccl::alltoall(input.data_ptr(),
                                                  output.data_ptr(),
                                                  (size_t)output.numel() / comm.size(),
                                                  cclDatatypes.at(output.scalar_type()),
                                                  comm,
                                                  attr););
            });
            return ret_evt;
          },
      c10d::OpType::ALLTOALL_BASE,
      "oneccl_bindings_for_pytorch::cpu_work::alltoall_base");

  }
  else{
    // Need alltoallv
    work = collective<get_ccl_comms, CPUWorkCCL>(
      pg,
      inputs,
      outputs,
      [=](at::Tensor input,
          at::Tensor output,
          ccl::alltoallv_attr attr,
          ccl::communicator& comm) {
            ccl::event ret_evt;
            c10d::checkSplitSizes(inputSplitSizes, input, grp_size);
            c10d::checkSplitSizes(outputSplitSizes, output, grp_size);

            std::vector<size_t> sendCounts(grp_size);
            std::vector<size_t> recvCounts(grp_size);
            bool inputSplitsEqual = inputSplitSizes.size() == 0;
            bool outputSplitsEqual = outputSplitSizes.size() == 0;

            size_t inLen = input.numel();
            size_t outLen = output.numel();
            if (inLen) inLen /= (inputSplitsEqual ? grp_size : input.size(0));
            if (outLen) outLen /= (outputSplitsEqual ? grp_size : output.size(0));

            for (int i = 0; i < grp_size; i++)
            {
                sendCounts[i] = (inputSplitsEqual ? inLen : inputSplitSizes[i] * inLen);
                recvCounts[i] = (outputSplitsEqual ? outLen : outputSplitSizes[i] * outLen);
            }

            call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&](){
                CCL_CHECK(ret_evt = ccl::alltoallv(input.data_ptr(),
                                                   sendCounts,
                                                   output.data_ptr(),
                                                   recvCounts,
                                                   cclDatatypes.at(output.scalar_type()),
                                                   comm,
                                                   attr););
            });
            return ret_evt;
    },
    c10d::OpType::ALLTOALL_BASE,
    "oneccl_bindings_for_pytorch::cpu_work::alltoall_base");
  }

  work->debugName = std::string("cpu::alltoall_base");
  enqueue(work);
  return work;
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> VanillaCPU::alltoall_(std::vector<at::Tensor>& outputTensors,
                                                             std::vector<at::Tensor>& inputTensors,
                                                             const AllToAllOptions& opts,
                                                             ProcessGroupCCL& pg){
  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  auto grp_size = pg.getSize();

  TORCH_CHECK(inputTensors.size() == (size_t)grp_size,
      "alltoall: number of input tensors are not equal to group size");

  TORCH_CHECK(outputTensors.size() == (size_t)grp_size,
      "alltoall: number of output tensors are not equal to group size");
  
  std::vector<std::vector<at::Tensor>> outputTensors_list = {outputTensors};
  std::vector<std::vector<at::Tensor>> inputTensors_list = {inputTensors};
  work = collective<get_ccl_comms, CPUWorkCCL>(
      pg,
      inputTensors_list,
      outputTensors_list,
      [=](std::vector<at::Tensor> inputs,
          std::vector<at::Tensor> outputs,
          ccl::alltoallv_attr attr,
          ccl::communicator& comm) {

      std::vector<size_t> sendCounts(grp_size);
      std::vector<size_t> recvCounts(grp_size);

      at::Tensor flatInput;
      at::Tensor flatOutput;

      int64_t flatSendCount;
      int64_t flatRecvCount;

      bool isInputFlat =
          computeLengthsAndCheckAndGetFlat(inputs, sendCounts, flatInput, flatSendCount);

      bool isOutputFlat =
          computeLengthsAndCheckAndGetFlat(outputs, recvCounts, flatOutput, flatRecvCount);

      if (!isInputFlat)
      {
          auto flatInputSplits =
              flatInput.split_with_sizes(c10::IntArrayRef((int64_t*)sendCounts.data(),
                                         sendCounts.size()), 0);

          for (int i = 0; i < grp_size; i++)
          {
              flatInputSplits[i].copy_(inputs[i].view({-1}));
          }
      }

      ccl::event ret_evt;

      call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&](){
          CCL_CHECK(ret_evt = ccl::alltoallv(flatInput.data_ptr(),
                                             sendCounts,
                                             flatOutput.data_ptr(),
                                             recvCounts,
                                             cclDatatypes.at(flatOutput.scalar_type()),
                                             comm,
                                             attr););
      });

      if (!isOutputFlat)
      {
         ret_evt.wait();

         auto flatOutputSplits =
             flatOutput.split_with_sizes(c10::IntArrayRef((int64_t*)recvCounts.data(),
                                         recvCounts.size()), 0);

         for (int i = 0; i < grp_size; i++)
         {
             outputs[i].view({-1}).copy_(flatOutputSplits[i]);
         }
      }
      return ret_evt;
  },
  c10d::OpType::ALLTOALL,
  "oneccl_bindings_for_pytorch::cpu_work::alltoall");
 
  work->debugName = std::string("cpu::alltoall");
  enqueue(work);
  return work;
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> VanillaCPU::barrier_(const BarrierOptions& opts,
                                                                   ProcessGroupCCL& pg) {

  c10::intrusive_ptr<AsyncBarrierWork> work = c10::make_intrusive<AsyncBarrierWork>();

  if (pg.ccl_member_->ccl_comms.size() == 0) {
    std::vector<at::Device> cpu_devices{at::Device("cpu")};
    const auto key = get_key_from_devs(cpu_devices);
    get_ccl_comms(pg, key, cpu_devices);
  }

  auto& comms_map = pg.ccl_member_->ccl_comms;
  for(auto iter = comms_map.begin(); iter != comms_map.end(); iter++){
      for(size_t i =0 ; i < iter->second->comms.size(); i++){
         work->getEvents().emplace_back(
                 call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&](){
                   if (i < iter->second->streams.size()) {
                     CCL_CHECK(return ccl::barrier(iter->second->comms[i],
                                                   iter->second->streams[i]););
                   } else {
                     CCL_CHECK(return ccl::barrier(iter->second->comms[i]););
                   }
                 })
                 );
     }
  }
  return work;
}


RegisterCPUPMethods cpu_register;

} // namespace c10d
