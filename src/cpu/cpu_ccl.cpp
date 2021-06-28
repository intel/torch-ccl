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

#include <ProcessGroupCCL.hpp>
#include <dispatch_stub.h>
#include <ATen/record_function.h>
#include "../utils.h"


namespace torch_ccl
{

namespace {

enum class
SparseResultMode : std::uint8_t
{
  DIRECT,
  OOP,
  COPY
};

static ccl::sparse_coalesce_mode sparseCoalesceMode;
static SparseResultMode sparseResultMode;

// Type mapping

std::map<ccl::datatype, at::ScalarType> ptDatatypes =
  {
    {ccl::datatype::uint8, at::kByte},
    {ccl::datatype::int32, at::kInt},
    {ccl::datatype::bfloat16, at::kBFloat16},
    {ccl::datatype::float32, at::kFloat},
    {ccl::datatype::float64, at::kDouble},
    {ccl::datatype::int64, at::kLong}
  };

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

Comms& get_ccl_comms(c10d::ProcessGroupCCL& pg, const std::string& devices_key, const std::vector<at::Device>& devices) {
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

} //namespace anonymous


class VanillaCPU final: public DispatchStub {
public:

  VanillaCPU() {}

  ~VanillaCPU() {}

protected:

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> allreduce_(std::vector<at::Tensor>& tensors,
                                                            const AllreduceOptions& opts,
                                                            ProcessGroupCCL& pg) override;


  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> reduce_(std::vector<at::Tensor>& tensors,
                                                         const ReduceOptions& opts,
                                                         ProcessGroupCCL& pg) override;

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> broadcast_(std::vector<at::Tensor>& tensors,
                                                            const BroadcastOptions& opts,
                                                            ProcessGroupCCL& pg) override;

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> allgather_(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                            std::vector<at::Tensor>& inputTensors,
                                                            const AllgatherOptions& opts,
                                                            ProcessGroupCCL& pg) override;

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

  void reset() override {}

private:
};

struct RegisterCPUPMethods {
  RegisterCPUPMethods() {
    static VanillaCPU methods;
    sparseCoalesceMode = ccl::sparse_coalesce_mode::regular;
    const char* sparseCoalesceModeEnv = getenv("CCL_SPARSE_COALESCE_MODE");
    if (sparseCoalesceModeEnv)
    {
      sparseCoalesceMode = ccl::sparse_coalesce_mode(atoi(sparseCoalesceModeEnv));
    }

    sparseResultMode = SparseResultMode::DIRECT;
    const char* sparseResultModeEnv = getenv("CCL_SPARSE_RESULT_MODE");
    if (sparseResultModeEnv)
    {
      sparseResultMode = (SparseResultMode)atoi(sparseResultModeEnv);
    }

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

void sparseAllreduceCompletionFn(
  const void* indBuf, size_t indCount, ccl::datatype indDatatype,
  const void* valBuf, size_t valCount, ccl::datatype valDatatype,
  const void* fnCtx)
{
  TORCH_CHECK(fnCtx, "null fn ctx");

  callback_context* work_cts = (callback_context*)fnCtx;
  work_cts->run_completion_hook(indBuf, indCount, indDatatype, valBuf, valCount, valDatatype);
  return ;
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> VanillaCPU::allreduce_(std::vector<at::Tensor>& tensors,
                                                                      const AllreduceOptions& opts,
                                                                      ProcessGroupCCL& pg) {
  checkSingleTensor(tensors);
  const auto& layout = tensors[0].layout();

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  if (layout == c10::kStrided) {
    work = collective<get_ccl_comms>(
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
            });

    work->debugName = std::string("cpu::allreduce");

  } else if (layout == c10::kSparse) {
    work = collective<get_ccl_comms>(
      pg,
      tensors,
      tensors,
      [=](at::Tensor input,
          at::Tensor output,
          ccl::sparse_allreduce_attr attr,
          ccl::communicator& comm){
            TORCH_CHECK(input.sparse_dim() == 1, "allreduce: only single sparse_dim is supported");

            ccl::event ret_evt;
            auto indices = input._indices();
            auto values = input._values();

            auto cpu_cb_ptr = make_cpu_callback(
              [=](const void* indBuf, size_t indCount, ccl::datatype indDatatype, const void* valBuf, size_t valCount, ccl::datatype valDatatype) {
                  const auto valueShape = input.sizes().slice(input.sparse_dim());
                  auto resultValueShape = std::vector<int64_t>({(int64_t)indCount});
                  std::copy(valueShape.begin(), valueShape.end(), std::back_inserter(resultValueShape));

                  auto rawIndices = at::from_blob((void*)indBuf,
                                                  {1, (long int)indCount},
                                                  ptDatatypes.at(indDatatype));

                  auto rawValues = at::from_blob((void*)valBuf,
                                                 resultValueShape,
                                                 ptDatatypes.at(valDatatype));

                  auto indices = at::empty({1, (long int)indCount}, input._indices().options());

                  auto values = at::empty(resultValueShape, input._values().options());

                  indices.copy_(rawIndices);
                  values.copy_(rawValues);

                  auto resultTensor = at::_sparse_coo_tensor_unsafe(indices, values, input.sizes(), input.options());

                  output.copy_(resultTensor);
              });
            attr.set<ccl::sparse_allreduce_attr_id::completion_fn>(static_cast<ccl::sparse_allreduce_completion_fn>(sparseAllreduceCompletionFn));
            attr.set<ccl::sparse_allreduce_attr_id::fn_ctx>(static_cast<const void*>(cpu_cb_ptr.get()));
            attr.set<ccl::sparse_allreduce_attr_id::coalesce_mode>(sparseCoalesceMode);


          call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&](){
            CCL_CHECK(ret_evt = ccl::preview::sparse_allreduce(indices.data_ptr(),
                                                  (size_t)indices.numel(),
                                                  values.data_ptr(),
                                                  (size_t)values.numel(),
                                                  nullptr, 0, nullptr, 0,
                                                  cclDatatypes.at(indices.scalar_type()),
                                                  cclDatatypes.at(values.scalar_type()),
                                                  cclOps.at(opts.reduceOp),
                                                  comm,
                                                  attr););
          });

          return std::make_tuple(std::move(ret_evt), cpu_cb_ptr);
      });

    work->debugName = std::string("cpu::sparse_allreduce");
  }

  return work;
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> VanillaCPU::reduce_(std::vector<at::Tensor>& tensors,
                                                                   const ReduceOptions& opts,
                                                                   ProcessGroupCCL& pg) {
  checkSingleTensor(tensors);

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  work = collective<get_ccl_comms>(
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
          });

  work->debugName = std::string("cpu::reduce");

  return work;
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> VanillaCPU::broadcast_(std::vector<at::Tensor>& tensors,
                                                                      const BroadcastOptions &opts,
                                                                      ProcessGroupCCL& pg) {
  checkSingleTensor(tensors);

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  work = collective<get_ccl_comms>(
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
          });

  work->debugName = std::string("cpu::broadcast");

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
  work = collective<get_ccl_comms>(
    pg,
    inputTensors,
    outputTensors,
    [=](at::Tensor input,
        std::vector<at::Tensor>& outputs,
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
    });

  work->debugName = std::string("cpu::allgather");

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
  work = collective<get_ccl_comms>(
      pg,
      inputTensors,
      outputTensors,
      [=](at::Tensor input,
          std::vector<at::Tensor>& outputs,
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
  });
    
  work->debugName = std::string("cpu::gather");

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
    work = collective<get_ccl_comms>(
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
          });

  }
  else{
    // Need alltoallv
    work = collective<get_ccl_comms>(
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
    });
  }

  work->debugName = std::string("cpu::alltoall_base");

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
  work = collective<get_ccl_comms>(
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
  });
 
  work->debugName = std::string("cpu::alltoall_base");
  return work;
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> VanillaCPU::barrier_(const BarrierOptions& opts,
                                                                   ProcessGroupCCL& pg) {

  c10::intrusive_ptr<AsyncBarrierWork> work = c10::make_intrusive<AsyncBarrierWork>();
  auto& comms_map = pg.ccl_member_->ccl_comms;
  for(auto iter = comms_map.begin(); iter != comms_map.end(); iter++){
      for(size_t i =0 ; i < iter->second->comms.size(); i++){
         work->getEvents().emplace_back(
                 call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&](){
                  CCL_CHECK(return ccl::barrier(iter->second->comms[i]););
                 })
                 );
     }
  }
  return work; 
}


RegisterCPUPMethods cpu_register;

} // namespace c10d
