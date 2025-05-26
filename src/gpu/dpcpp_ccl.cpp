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

#include <ATen/record_function.h>
#include <ProcessGroupCCL.hpp>
#include <dispatch_stub.h>

#include <sycl/sycl.hpp>
//#include "allreduce.h"
#include "allreduce_small.h"

// pytorch 2.3 above
#if TORCH_VERSION_MAJOR > 1 && TORCH_VERSION_MINOR > 2
#include <c10/xpu/XPUStream.h>
#endif

int disable_allreduce = -1;
int work_only = -1;
int sync_only = -1;

//allreducer<sycl::ext::oneapi::bfloat16, 8, 4096> gpu_allreducer_bf16;
allreducer<sycl::half, 8, 4096> gpu_allreducer_fp16;
allreducer_small<sycl::half, 8, 4096> gpu_allreducer_small_fp16;
allreducer_small<sycl::half, 8, 4096> gpu_allreducer_small_fp32;
allreducer_small<sycl::half, 8, 4096> gpu_allreducer_small_bf16;


int get_disable_allreduce(int init_value = 0) {
  int tmp_disable_allreduce = init_value;
  char *tmp_str = getenv("TORCH_CCL_DISABLE_ALLREDUCE");
  if (tmp_str) {
    tmp_disable_allreduce = atoi(tmp_str);
  }
  disable_allreduce = tmp_disable_allreduce;
  return tmp_disable_allreduce;
}


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

inline sycl::queue get_sycl_queue(const c10::Stream& stream) {
// pytorch 2.3 above
#if TORCH_VERSION_MAJOR > 1 && TORCH_VERSION_MINOR > 2
    return at::xpu::XPUStream(stream).queue();
#else
    return xpu::get_queue_from_stream(stream);
#endif

}

namespace
{
    int use_llm_allreduce =  0;
    int last_use_llm_allreduce = 0;
    bool support_fp64 = false;
    int eu_count = 0;

    std::once_flag allreducer_initialize_flag;
    void init_llm_allreducer(c10d::ProcessGroupCCL& pg_ccl, const std::vector<at::Device>& devices){
        int total_rank_size = pg_ccl.getSize();
        int local_base_rank = pg_ccl.getRank();

        c10::impl::VirtualGuardImpl impl(devices[0].type());
        c10::Stream stream = impl.getStream(devices[0]);
        auto q = get_sycl_queue(stream);
        
        gpu_allreducer_fp16.init(q, local_base_rank, total_rank_size);
        gpu_allreducer_small_fp16.init(q, local_base_rank, total_rank_size);

        // Get device properity.
        auto dev = q.get_device();
        support_fp64 = dev.has(sycl::aspect::fp64);
        eu_count = dev.has(sycl::aspect::ext_intel_gpu_eu_count) ? dev.get_info<sycl::ext::intel::info::device::gpu_eu_count>():512;
        // std::cout << "support_fp64: "<<support_fp64<<", eu_count: " << eu_count << std::endl;
    }

    c10::Stream llm_torch_stream(c10::Stream::DEFAULT, c10::Device(c10::DeviceType::XPU, 0));
    void check_llm_allreduce_env(c10d::ProcessGroupCCL& pg_ccl, const std::vector<at::Device>& devices) {
        char *use_llm_allreduce_str = getenv("TORCH_LLM_ALLREDUCE_DEBUG");
        if (use_llm_allreduce_str) {
            last_use_llm_allreduce = use_llm_allreduce;
            use_llm_allreduce = atoi(use_llm_allreduce_str);
        }
        if (use_llm_allreduce != 0) {
            // Initialize stream.
            c10::impl::VirtualGuardImpl impl(devices[0].type());
            llm_torch_stream = impl.getStream(devices[0]);
        }
    }

    int get_local_size() {
        // Currently, we only support intel mpi and mpich as launcher for LLM allreduce
        // Try to get intel mpi environment firstly.
        char *impi_local_size_str = getenv("MPI_LOCALNRANKS");
        int local_world_size = 0;
        if (impi_local_size_str) {
            local_world_size = atoi(impi_local_size_str);
        } else {
            // Try mpich environment.
            char *mpich_local_size_str = getenv("PALS_LOCAL_SIZE");
            if (mpich_local_size_str) {
                local_world_size = atoi(mpich_local_size_str);
            }
        }
        return local_world_size;
    }

    bool llm_allreduce_available(const at::Tensor& input, const int world_size, const int local_world_size, const c10d::AllreduceOptions& opts) {
        if (support_fp64 &&
            use_llm_allreduce !=0 &&
            opts.reduceOp == c10d::ReduceOp::SUM &&
            input.scalar_type() == at::kHalf &&
            world_size <= 8 &&
            local_world_size > 0 &&
            local_world_size <= world_size) {
            return true;
        }
        return false;
    }
} // namespace


namespace oneccl_bindings_for_pytorch
{

namespace {
// [Sync Streams] Helper that lets the input ccl::stream to wait for the current
// stream. oneCCL communications run on ccl::stream, but input tensors are
// allocated on different streams (i.e., current streams). Communications on
// ccl::stream cannot start before pending input tensor ops on current streams
// finish. Otherwise, ops on two streams might read/write same tensors
// concurrently.
//
// The synchronization above alone is not enough. We also need to make sure
// input tensors are not freed before their usages on ccl::stream finish. This
// can be achieved by calling aten::record_stream,
// which remembers the usage stream (ccl::stream), creates an event on the usage
// stream when GC attempts to free the input tensor, and delays GC until that
// event is done.
void sync_streams(
        const std::vector<at::Device>& devices,
        const std::vector<c10::Stream>& ccl_torch_streams) {
  for (const auto i : c10::irange(devices.size())) {
    c10::impl::VirtualGuardImpl impl(devices[i].type());
    c10::Stream stream = impl.getStream(devices[i]);
    c10::Event evt(at::kXPU);
    evt.record(stream);
    c10::Stream ccl_torch_stream = ccl_torch_streams[i];
    evt.block(ccl_torch_stream);
  }
}

void record_tensor(const at::Tensor& tensor, at::Stream stream) {
  tensor.record_stream(stream);
}

void record_tensor(const std::vector<at::Tensor>& tensors, at::Stream stream) {
  for (auto& tensor : tensors) {
    tensor.record_stream(stream);
  }
}

// Check that all `tensors' have the same device and type and shape and
// are distributed across distinct GPUs if these are GPU tensors.
c10::DeviceType check_tensors_properties(const std::vector<at::Tensor>& tensors) {
  if (tensors.size() == 0) {
    throw std::runtime_error("Tensor list must be nonempty");
  }
  c10::Device device = tensors.front().device();
  c10::impl::VirtualGuardImpl impl(device.type());
  auto device_count = impl.deviceCount();
  if (tensors.size() > static_cast<size_t>(device_count)) {
    throw std::runtime_error(
      "Tensor list mustn't be larger than the number of available GPUs");
  }

  const auto& first = tensors.front();
  auto dev_type = first.device().type();

  // Set for ensuring that tensors are on separate devices.
  std::unordered_set<decltype(first.get_device())> usedDevices;
  usedDevices.reserve(tensors.size());

  for (const auto& t : tensors) {
    if (t.is_sparse()) {
      throw std::runtime_error("Tensors must be dense");
    }
    if (t.scalar_type() != first.scalar_type()) {
      throw std::runtime_error("Tensors must have identical type");
    }
    if (t.sizes() != first.sizes()) {
      throw std::runtime_error("Tensors must have identical size");
    }
    if (!t.is_contiguous()) {
      throw std::runtime_error("Tensors must be contiguous");
    }
    if (dev_type != t.device().type()) {
      throw std::runtime_error("Tensors must be on the same device type");
    }
    const auto inserted = usedDevices.insert(t.get_device()).second;
    if (!inserted) {
      throw std::runtime_error("Tensors must be on distinct devices");
    }
  }

  return dev_type;
}

Comms& get_ccl_comms(c10d::ProcessGroupCCL& pg_ccl, const std::string& devices_key, const std::vector<at::Device>& devices, c10d::OpType op_type = OpType::UNKNOWN, int p2pRank = 0, bool isSendRecvSelf = false) {

  RECORD_FUNCTION("oneccl_bindings_for_pytorch::xpu::get_ccl_comms", std::vector<c10::IValue>());
  // Sanity check
  if (devices_key.empty()) {
    throw std::runtime_error(
            "Not able to create/get the CCL Communicator since "
            "the devices are empty ");
  }

  if (devices.size() != 1) {
    throw std::runtime_error("Torch CCL only support one device per process now");
  }

  if (pg_ccl.useSameStream_) {
      c10::impl::VirtualGuardImpl impl(devices[0].type());
      c10::Stream current_stream = impl.getStream(devices[0]);
      auto cached_comms = pg_ccl.ccl_member_->get_comms(devices_key + "_" + std::to_string(current_stream.id()));
      if (cached_comms) {
          return *cached_comms;
      }
  } else {
     auto cached_comms = pg_ccl.ccl_member_->get_comms(devices_key); // stream is not in cache key
     if (cached_comms && use_llm_allreduce == last_use_llm_allreduce) {
         return *cached_comms;
     }
  }

  bool batchP2P = pg_ccl.cclActiveGroupCounter_ > 0;
  bool singleP2POp = c10d::isP2POp(op_type, batchP2P);

  ccl::vector_class<ccl::pair_class<int, ccl::device>> devs_rank;
  ccl::vector_class<ccl::stream> ccl_streams;
  ccl_streams.reserve(devices.size());
  std::vector<c10::Stream> torch_streams;
  torch_streams.reserve(devices.size());

  if (disable_allreduce == -1) {
    get_disable_allreduce(0);
  }
  if (work_only == -1) {
    get_work_only(0);
  }
  if (sync_only == -1) {
    get_sync_only(0);
  }
  // Create the stream and rank dev mapping

  // for (const auto i : c10::irange(pg_ccl.cclActiveGroupCounter_)) {
  //   (void)i;
  //   // comms have not been initiated yet, so can only check in blocking-way
  //   ccl::group_end();
  // }
  // GPU world size and GPU local rank
  // Only support the symmetric distributed communication
  int total_rank_size, local_base_rank;
  for (size_t i = 0; i < devices.size(); i++) {

    if (!singleP2POp) {
      total_rank_size = pg_ccl.getSize() * devices.size();
      local_base_rank = pg_ccl.getRank() * devices.size();
    } else if (isSendRecvSelf) {
      total_rank_size = 1;
      local_base_rank = 0;
    } else {
      total_rank_size = 2;
      local_base_rank = p2pRank;
    }

    c10::impl::VirtualGuardImpl impl(devices[i].type());
    if (pg_ccl.useSameStream_) {
        c10::Stream stream = impl.getStream(devices[i]);
        torch_streams.push_back(stream);
        auto q = get_sycl_queue(stream);
        ccl_streams.push_back(ccl::create_stream(q));

        int rank = local_base_rank + i;
        devs_rank.emplace_back(rank, ccl::create_device(q.get_device()));
    } else {
        // XPU doesn't support prioritized stream.
        c10::Stream stream = impl.getStreamFromGlobalPool(devices[i], /*isHighPriority=*/false);
        torch_streams.push_back(stream);
        auto q = get_sycl_queue(stream);
        ccl_streams.push_back(ccl::create_stream(q));

        int rank = local_base_rank + i;
        devs_rank.emplace_back(rank, ccl::create_device(q.get_device()));
    }

  }

  // The IPEX use default global context.
  // TODO: add get default global context API in IPEX.
  c10::impl::VirtualGuardImpl impl(devices[0].type());
  c10::Stream dpcpp_stream = impl.getStream(devices[0]);
  auto q = get_sycl_queue(dpcpp_stream);
  auto ctx = ccl::create_context(q.get_context());
  // Create ccl::communicators
  int init_rank = pg_ccl.getRank();
  if(singleP2POp) init_rank = p2pRank;
  auto dpcpp_comms = ccl::create_communicators(total_rank_size, devs_rank, ctx, 
    pg_ccl.ccl_member_->get_kvs(init_rank, *pg_ccl.store_, singleP2POp, devices_key, p2pRank));

  // Initialize allreducer only if use_llm_allreduce is set to nonzero.
  if (use_llm_allreduce != 0){
    std::call_once(allreducer_initialize_flag, init_llm_allreducer, pg_ccl, devices);
    std::cout << "Allreduce goes to LLM path." << std::endl;
  }

  // for (const auto i : c10::irange(pg_ccl.cclActiveGroupCounter_)) {
  //   (void)i;
  //   ccl::group_start();
  // }
  // Store the comms to cache
  std::shared_ptr<Comms> dpcpp_comms_ptr = std::make_shared<Comms>(dpcpp_comms, ccl_streams, torch_streams);

  if (pg_ccl.useSameStream_) {
      auto torch_streams = dpcpp_comms_ptr->torch_streams;
      // Add stream id to cache. Then if new stream comes, new communicator will be created.
      pg_ccl.ccl_member_->add_comms(devices_key + "_" + std::to_string(torch_streams[0].id()), dpcpp_comms_ptr);
  } else {
      pg_ccl.ccl_member_->add_comms(devices_key, dpcpp_comms_ptr);
  }

  return *dpcpp_comms_ptr.get();
}

template <typename RunF, typename CommType, typename InputType, typename OutputType, typename attr_t>
class XPUWorkCCL : public CollectiveAsyncWorkCCL<RunF, CommType, InputType, OutputType, attr_t> {
public:
  XPUWorkCCL(const std::vector<InputType>& inputs,
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
                     inputs, outputs, f, comms, attr, timeout, rank, opType, profilingTitle, inputTensors), 
                     is_coalescing_end(inputs.empty() && outputs.empty()) {}

  void run() override {
    // Return immediately if current op is coalescing_end since inputs and outputs are empty. 
    if (is_coalescing_end) {
      return;
    }

    if (!this->useSameStream_ && use_llm_allreduce == 0) {
      const auto devices = get_device_list(this->inputs);

      // add SYCL running dependency computation -> communication.
      sync_streams(devices, this->comms.torch_streams);

      for (const auto i : c10::irange(this->inputs.size())) {
        // Both `inputs' and `outputs' are created on a worker stream and used in
        // different ncclStreams.  Hence, both must record the ncclStream to
        // prevent being freed before the collective finishes.
        //
        // We only record `inputs' here, and leave recording `outputs' to `fn' for
        // operations where `inputs' and `outputs' are not the same.
        //
        // See [Sync Streams].
        // Now torch-ccl only supports one device per process, so there is only one 
        // torch stream in comms.torch_streams.
        record_tensor(this->inputs[i], this->comms.torch_streams[0]);
      }
    }

    CollectiveAsyncWorkCCL<RunF, CommType, InputType, OutputType, attr_t>::run();
  };

  // No explicitly synchronization.
  virtual ~XPUWorkCCL() {}

  // Waiting on the work's on XPU backend
  bool wait(std::chrono::milliseconds timeout) override {
    this->synchronizeInternalForXPU(timeout);
    // Check for errors and throw appropriate exception.
    this->checkAndThrowException();
    return true;
  }

  void synchronizeInternalForXPU(std::chrono::milliseconds timeout) {
      if (this->blockingWait_) {
          this->synchronizeInternal(kNoTimeout);
      } else if (!this->useSameStream_) {
          // sync from ccl event to torch stream if communication stream
          // is not as computation stream(or default stream)
          for(int i = 0; i < this->rets.size(); i++) {
              ccl::event& req = this->rets[i];
              const auto devices = get_device_list(this->inputs);
              for (const auto i : c10::irange(devices.size())) {
                c10::impl::VirtualGuardImpl impl(devices[i].type());
                c10::Stream stream = impl.getStream(devices[i]);
                auto torch_queue = get_sycl_queue(stream);
                torch_queue.ext_oneapi_submit_barrier({req.get_native()});
              }
          }
      }
  }

  void synchronize() override {
    this->synchronizeInternalForXPU(kNoTimeout);
  }

  void finishAsyncWorkCCL() override {
    if (!this->useSameStream_ && use_llm_allreduce == 0) {
      c10::MultiStreamGuard streams_guard(this->comms.torch_streams);
    }
    // under the stream guard. Mark the Future completing.
    this->AsyncWorkCCL::finishAsyncWorkCCL();
  }
private:
    bool is_coalescing_end;
};

template <typename RunF, typename CommType, typename InputType, typename OutputType, typename attr_t>
class XPUP2PAsyncWork : public P2PAsyncWork<RunF, CommType, InputType, OutputType, attr_t> {

public:
  XPUP2PAsyncWork(const std::vector<InputType>& inputs,
             const std::vector<OutputType>& outputs,
             int peer,
             const RunF f,
             CommType& comms,
             attr_t& attr,
             std::chrono::milliseconds timeout,
             int rank,
             c10d::OpType opType,
             const char* profilingTitle,
             const c10::optional<std::vector<at::Tensor>>& inputTensors) :
             P2PAsyncWork<RunF, CommType, InputType, OutputType, attr_t>(
                     inputs, outputs, peer, f, comms, attr, timeout, rank, opType, profilingTitle, inputTensors) {}

  void run() override {
    const auto devices = get_device_list(this->inputs);
    // add SYCL running dependency computation -> communication.
    sync_streams(devices, this->comms.torch_streams);
    for (const auto i : c10::irange(this->inputs.size())) {
        // Both `inputs' and `outputs' are created on a worker stream and used in
        // different ncclStreams.  Hence, both must record the ncclStream to
        // prevent being freed before the collective finishes.
        //
        // We only record `inputs' here, and leave recording `outputs' to `fn' for
        // operations where `inputs' and `outputs' are not the same.
        //
        // See [Sync Streams].
        record_tensor(this->inputs[i], this->comms.torch_streams[i]);
    }

    P2PAsyncWork<RunF, CommType, InputType, OutputType, attr_t>::run();
  };

  // No explicitly synchronization.
  virtual ~XPUP2PAsyncWork() {}

  // Waiting on the work's on XPU backend
  bool wait(std::chrono::milliseconds timeout) override {
    this->synchronizeInternal(timeout);
    // Check for errors and throw appropriate exception.
    this->checkAndThrowException();
    return true;
  }

  void finishAsyncWorkCCL() override {
    c10::MultiStreamGuard streams_guard(this->comms.torch_streams);
    // under the stream guard. Mark the Future completing.
    this->AsyncWorkCCL::finishAsyncWorkCCL();
  }
private:

};

} //namespace anonymous

class XPUCCLStubs final: public DispatchStub {

public:

  XPUCCLStubs() {
    stop_=false;
    workerThread_ = std::thread(&XPUCCLStubs::runLoop, this);
  }

  ~XPUCCLStubs() {destroy();}

protected:

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> allreduce_(std::vector<at::Tensor>& tensors,
                                                            const AllreduceOptions& opts,
                                                            ProcessGroupCCL& pg_ccl) override;

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> allreduce_coalesced_(std::vector<at::Tensor>& tensors,
                                                                    const AllreduceOptions& opts,
                                                                    ProcessGroupCCL& pg_ccl) override;


  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> reduce_(std::vector<at::Tensor>& tensors,
                                                         const ReduceOptions& opts,
                                                         ProcessGroupCCL& pg_ccl) override;

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> reduce_scatter_(std::vector<at::Tensor>& outputTensors,
                                                                          std::vector<std::vector<at::Tensor>>& inputTensors,
                                                                          const ReduceScatterOptions& opts,
                                                                          ProcessGroupCCL& pg_ccl) override;

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> _reduce_scatter_base_(at::Tensor& outputTensor,
                                                                          at::Tensor& inputTensor,
                                                                          const ReduceScatterOptions& opts,
                                                                          ProcessGroupCCL& pg_ccl) override;

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL>  reduce_scatter_tensor_coalesced_(std::vector<at::Tensor>& outputTensors,
                                                                        std::vector<at::Tensor>& inputTensors,
                                                                        const ReduceScatterOptions& opts,
                                                                        ProcessGroupCCL& pg_ccl) override;

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> broadcast_(std::vector<at::Tensor>& tensors,
                                                            const BroadcastOptions& opts,
                                                            ProcessGroupCCL& pg_ccl) override;

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> allgather_(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                            std::vector<at::Tensor>& inputTensors,
                                                            const AllgatherOptions& opts,
                                                            ProcessGroupCCL& pg_ccl) override;

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> _allgather_base_(at::Tensor& outputTensor,
                                                                     at::Tensor& inputTensor,
                                                                     const AllgatherOptions& opts,
                                                                     ProcessGroupCCL& pg_ccl) override;

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> allgather_into_tensor_coalesced_(std::vector<at::Tensor>& outputTensors,
                                                                        std::vector<at::Tensor>& inputTensors,
                                                                        const AllgatherOptions& opts,
                                                                        ProcessGroupCCL& pg_ccl) override;

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> gather_(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                         std::vector<at::Tensor>& inputTensors,
                                                         const GatherOptions& opts,
                                                         ProcessGroupCCL& pg) override;

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> scatter_(std::vector<at::Tensor>& outputTensors,
                                                             std::vector<std::vector<at::Tensor>>& inputTensors,
                                                             const ScatterOptions& opts,
                                                             ProcessGroupCCL& pg_ccl) override;

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> alltoall_(std::vector<at::Tensor>& outputTensors,
                                                           std::vector<at::Tensor>& inputTensors,
                                                           const AllToAllOptions& opts,
                                                           ProcessGroupCCL& pg) override;

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> alltoall_base_(at::Tensor& outputTensor,
                                                                at::Tensor& inputTensor,
                                                                std::vector<int64_t>& outputSplitSizes,
                                                                std::vector<int64_t>& inputSplitSizes,
                                                                const AllToAllOptions& opts,
                                                                ProcessGroupCCL& pg) override;

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> send_(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag,
      ProcessGroupCCL& pg) override;

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> recv_(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag,
      ProcessGroupCCL& pg) override;

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> barrier_(const BarrierOptions& opts,
                                                                ProcessGroupCCL& pg) override;

  virtual c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> end_coalescing_(ProcessGroupCCL& pg_ccl) override;

  void destroy();
  void reset() override {}
  void runLoop();
  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> execute(c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> & work);
private:
  bool stop_;
  std::mutex pgMutex_;
  std::thread workerThread_;
  std::deque<c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL>> queue_;
  std::condition_variable queueProduceCV_;
  std::condition_variable queueConsumeCV_;

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> allreduce_impl(std::vector<at::Tensor>& tensors,
                                                            const AllreduceOptions& opts,
                                                            ProcessGroupCCL& pg_ccl);
  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> _reduce_oop(at::Tensor& outputTensor,
                                                         at::Tensor& inputTensor,
                                                         const ReduceOptions& opts,
                                                         ProcessGroupCCL& pg_ccl);
};

struct RegisterXPUMethods {
  RegisterXPUMethods() {
    static XPUCCLStubs methods;
    DispatchStub::register_ccl_stub(c10::DeviceType::XPU, &methods);
  }
};

void checkGPUTensor(const at::Tensor& tensor)
{
//  TORCH_CHECK(!is_block_format(tensor), "ccl doesn't support block format tensor");
}

void checkGPUTensor(const std::vector<at::Tensor>& tensors)
{
  checkGPUTensor(tensors[0]);
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> XPUCCLStubs::execute(c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> & work){
  try {
    work->run();
  } catch (...) {
    work->finishAsyncWorkCCLError(std::current_exception());
    return work;
  }
  // mark the work finished asynchronizely.
  work->finishAsyncWorkCCL();

  // Track the work internal
  std::unique_lock<std::mutex> lock(pgMutex_);
  queue_.push_back(work);
  lock.unlock();
  queueProduceCV_.notify_one();

  return work;
}

void XPUCCLStubs::destroy() {
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

void XPUCCLStubs::runLoop() {
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
//      work->finishAsyncWorkCCL();

    } catch (...) {
//      work->finishAsyncWorkCCLError(std::current_exception());
    }

    lock.lock();
  }
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> XPUCCLStubs::allreduce_impl(std::vector<at::Tensor>& tensors,
                                                                       const AllreduceOptions& opts,
                                                                       ProcessGroupCCL& pg_ccl) {
  checkGPUTensor(tensors);
  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  const int world_size = pg_ccl.getSize();
  const int local_world_size = get_local_size();
  const auto devices = get_device_list(tensors);
  check_llm_allreduce_env(pg_ccl, devices);

  work = collective<get_ccl_comms, XPUWorkCCL>(
    pg_ccl,
    tensors,
    tensors,
    [=](at::Tensor input,
        at::Tensor output,
        ccl::allreduce_attr attr,
        ccl::communicator& comm,
        ccl::stream& stream) {

      ccl::event ret_evt;
      if (disable_allreduce != 0 || world_size == 1) {
        sycl::event sycl_evt;
	sycl_evt = stream.get_native().ext_oneapi_submit_barrier();
        return ccl::event::create_from_native(sycl_evt);
      }

      if (llm_allreduce_available(input, world_size, local_world_size, opts)) {
        auto q = get_sycl_queue(llm_torch_stream);
        /*
        if (sync_only != 0) {
        gpu_allreducer_fp16.sync_only(stream.get_native(), input.data_ptr(), (size_t)input.numel());  
        return ret_evt;
        }
        if (work_only != 0) {
        gpu_allreducer_fp16.work_only(stream.get_native(), input.data_ptr(), (size_t)input.numel());  
        return ret_evt;
        }
        */
        // Now SMALL_MAX_COUNT is 448K(448 * 1024).
        if ((size_t)input.numel() <= SMALL_MAX_COUNT) {
            gpu_allreducer_small_fp16.allreduce(q, input.data_ptr(), (size_t)input.numel());
        } else {
            gpu_allreducer_fp16.allreduce(q, input.data_ptr(), (size_t)input.numel());
        }
        // printf("Use LLM allreduce.\n");
        return ret_evt;
    }

    call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&](){
        CCL_CHECK(ret_evt = ccl::allreduce(input.data_ptr(),
                                            output.data_ptr(),
                                            (size_t) input.numel(),
                                            cclDatatypes.at(input.scalar_type()),
                                            cclOps.at(opts.reduceOp),
                                            comm,
                                            stream,
                                            attr));
      });
    // printf("Use One CCL allreduce.\n");
    return ret_evt;
  },
  c10d::OpType::ALLREDUCE);

  work->debugName = std::string("xpu::allreduce");
  execute(work);

  return work;
}


c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> XPUCCLStubs::allreduce_(std::vector<at::Tensor>& tensors,
                                                                       const AllreduceOptions& opts,
                                                                       ProcessGroupCCL& pg_ccl) {
    RECORD_FUNCTION("oneccl_bindings_for_pytorch::xpu::allreduce", std::vector<c10::IValue>({tensors[0]}));
    return allreduce_impl(tensors, opts, pg_ccl);
}


c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> XPUCCLStubs::allreduce_coalesced_(std::vector<at::Tensor>& tensors,
                                                                       const AllreduceOptions& opts,
                                                                      ProcessGroupCCL& pg_ccl) {
    std::vector<c10::IValue> params(tensors.size());
    std::transform(tensors.begin(), tensors.end(), params.begin(), 
        [](const at::Tensor& t) {return static_cast<c10::IValue>(t);});
    RECORD_FUNCTION("oneccl_bindings_for_pytorch::xpu::allreduce_coalesced", params);
    return allreduce_impl(tensors, opts, pg_ccl);
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> XPUCCLStubs::reduce_(std::vector<at::Tensor>& tensors,
                                                                    const ReduceOptions& opts,
                                                                    ProcessGroupCCL& pg_ccl) {
  checkGPUTensor(tensors);
  const int root = opts.rootRank * tensors.size() + opts.rootTensor;
  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  work = collective<get_ccl_comms, XPUWorkCCL>(
    pg_ccl,
    tensors,
    tensors,
    [=](at::Tensor input,
        at::Tensor output,
        ccl::reduce_attr attr,
        ccl::communicator& comm,
        ccl::stream& stream) {
      RECORD_FUNCTION("oneccl_bindings_for_pytorch::xpu::reduce", std::vector<c10::IValue>{input});

      ccl::event ret_evt;
      call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&]() {
        CCL_CHECK(ret_evt = ccl::reduce(input.data_ptr(),
                                output.data_ptr(),
                                (size_t) input.numel(),
                                cclDatatypes.at(input.scalar_type()),
                                cclOps.at(opts.reduceOp),
                                root,
                                comm,
                                stream));
      });
      return ret_evt;

  },
    c10d::OpType::REDUCE);

  work->debugName = std::string("xpu::reduce");
  execute(work);

  return work;
}

// _reduce_oop implements an out-of-place reduce procedure.
c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> XPUCCLStubs::_reduce_oop(at::Tensor& outputTensor,
                                                        at::Tensor& inputTensor,
                                                        const ReduceOptions& opts,
                                                        ProcessGroupCCL& pg_ccl) {
  const int root = opts.rootRank + opts.rootTensor;
  std::vector<at::Tensor> inputTensors{inputTensor};
  std::vector<at::Tensor> outputTensors{outputTensor};
  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  work = collective<get_ccl_comms, XPUWorkCCL>(
    pg_ccl,
    inputTensors,
    outputTensors,
    [=](at::Tensor input,
        at::Tensor output,
        ccl::reduce_attr attr,
        ccl::communicator& comm,
        ccl::stream& stream) {
      RECORD_FUNCTION("oneccl_bindings_for_pytorch::xpu::reduce_oop", std::vector<c10::IValue>{input});

      ccl::event ret_evt;
      call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&]() {
        CCL_CHECK(ret_evt = ccl::reduce(input.data_ptr(),
                                output.data_ptr(),
                                (size_t) input.numel(),
                                cclDatatypes.at(input.scalar_type()),
                                cclOps.at(opts.reduceOp),
                                root,
                                comm,
                                stream));
      });
      return ret_evt;

  },
    c10d::OpType::REDUCE);

  work->debugName = std::string("xpu::_reduce_oop");
  execute(work);

  return work;
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> XPUCCLStubs::reduce_scatter_(std::vector<at::Tensor>& outputTensors,
                                                                        std::vector<std::vector<at::Tensor>>& inputTensors,
                                                                        const ReduceScatterOptions& opts,
                                                                        ProcessGroupCCL& pg_ccl) {
  checkSingleTensor(outputTensors);
  auto outputTensor = outputTensors.back();
  auto inputTensors_ = inputTensors.back();
  bool same_size = check_same_size(inputTensors_);
  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  if (same_size) {
    auto inputFlattened = newLikeFlat(inputTensors_);
    for (const auto j : c10::irange(inputTensors_.size())) {
        inputFlattened[j].copy_(inputTensors_[j], true);
    }
    std::vector<at::Tensor> flattendInputTensors{inputFlattened};

    work = collective<get_ccl_comms, XPUWorkCCL>(
            pg_ccl,
            flattendInputTensors,
            outputTensors,
            [=](at::Tensor input,
                at::Tensor output,
                ccl::reduce_attr attr,
                ccl::communicator& comm,
                ccl::stream& stream) {
                RECORD_FUNCTION("oneccl_bindings_for_pytorch::xpu::reduce_scatter", std::vector<c10::IValue>{input});

                ccl::event ret_evt;
                call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&]() {
                CCL_CHECK(ret_evt = ccl::reduce_scatter(input.data_ptr(),
                                                        output.data_ptr(),
                                                        (size_t) output.numel(),
                                                        cclDatatypes.at(input.scalar_type()),
                                                        cclOps.at(opts.reduceOp),
                                                        comm,
                                                        stream));
                });
                return ret_evt;

            },
            c10d::OpType::REDUCE_SCATTER);

    work->debugName = std::string("xpu::reduce_scatter");
    execute(work);
    return work;
  } else {
    // Use multiple reduce to simulate reduce_scatter.
    // Currently one-ccl doest support grouped primitives, we'll add coalescing when it supports.
    // todo: After oneCCL support non-p2p op
    // std::vector<c10::intrusive_ptr<Work>> works;
    // pg_ccl.startCoalescing();
    const auto num_reduces = inputTensors_.size();
    for (const int i : c10::irange(num_reduces)) {
      auto& input = inputTensors_[i];
      auto& output = (i == pg_ccl.getRank()) ? outputTensor : input;
      auto reduceOpts = ReduceOptions{
          opts.reduceOp,
          static_cast<int64_t>(i),
          static_cast<int64_t>(0),
          opts.timeout};
      work = _reduce_oop(output, input, reduceOpts, pg_ccl);
      // works.push_back(work);
    }
    return work;
    // todo: After oneCCL support non-p2p op
    // auto work = pg_ccl.endCoalescing();
    // return c10::static_intrusive_pointer_cast<ProcessGroupCCL::AsyncWorkCCL>(work);
  }
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> XPUCCLStubs::_reduce_scatter_base_(at::Tensor& outputTensor,
                                                                        at::Tensor& inputTensor,
                                                                        const ReduceScatterOptions& opts,
                                                                        ProcessGroupCCL& pg_ccl) {

  checkGPUTensor({outputTensor, inputTensor});
  const int world_size = pg_ccl.getSize();
  if (inputTensor.numel() != outputTensor.numel() * world_size) {
    TORCH_CHECK(
            false,
            "input tensor must be the same size as output size times world size");
  }

  // just a wrapper to fit the collective interface
  auto inputs = std::vector<at::Tensor> {inputTensor};
  auto outputs = std::vector<at::Tensor> {outputTensor};

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  work = collective<get_ccl_comms, XPUWorkCCL>(
          pg_ccl,
          inputs,
          outputs,
          [=](at::Tensor input,
              at::Tensor output,
              ccl::reduce_attr attr,
              ccl::communicator& comm,
              ccl::stream& stream) {
            RECORD_FUNCTION("oneccl_bindings_for_pytorch::xpu::_reduce_scatter_base", std::vector<c10::IValue>{input});

            ccl::event ret_evt;
            call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&]() {
              CCL_CHECK(ret_evt = ccl::reduce_scatter(input.data_ptr(),
                                                      output.data_ptr(),
                                                      (size_t) output.numel(),
                                                      cclDatatypes.at(input.scalar_type()),
                                                      cclOps.at(opts.reduceOp),
                                                      comm,
                                                      stream));
            });
            return ret_evt;

          },
          c10d::OpType::_REDUCE_SCATTER_BASE);

  work->debugName = std::string("xpu::_reduce_scatter_base");
  execute(work);

  return work;
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> XPUCCLStubs::reduce_scatter_tensor_coalesced_(std::vector<at::Tensor>& outputTensors,
                                                                    std::vector<at::Tensor>& inputTensors,
                                                                    const ReduceScatterOptions& opts,
                                                                    ProcessGroupCCL& pg_ccl) {
  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  work = collective<get_ccl_comms, XPUWorkCCL>(
          pg_ccl,
          inputTensors,
          outputTensors,
          [=](at::Tensor input,
              at::Tensor output,
              ccl::allgatherv_attr attr,
              ccl::communicator& comm,
              ccl::stream& stream) {
            RECORD_FUNCTION("oneccl_bindings_for_pytorch::xpu::reduce_scatter_tensor_coalesced_", std::vector<c10::IValue>({input}));

            ccl::event ret_evt;
            call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&]() {
              CCL_CHECK(ret_evt = ccl::reduce_scatter(input.data_ptr(),
                                                      output.data_ptr(),
                                                      (size_t) output.numel(),
                                                      cclDatatypes.at(input.scalar_type()),
                                                      cclOps.at(opts.reduceOp),
                                                      comm,
                                                      stream));
            });
            return ret_evt;
          },
          c10d::OpType::COALESCED);

  work->debugName = std::string("xpu::reduce_scatter_tensor_coalesced_");
  execute(work);

  return work;
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> XPUCCLStubs::broadcast_(std::vector<at::Tensor>& tensors,
                                                                       const BroadcastOptions &opts,
                                                                       ProcessGroupCCL& pg_ccl) {
  checkGPUTensor(tensors);
  const int root = opts.rootRank * tensors.size() + opts.rootTensor;
  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  work = collective<get_ccl_comms, XPUWorkCCL>(
    pg_ccl,
    tensors,
    tensors,
    [=](at::Tensor input,
        at::Tensor output,
        ccl::broadcast_attr attr,
        ccl::communicator& comm,
        ccl::stream& stream) {
      RECORD_FUNCTION("oneccl_bindings_for_pytorch::xpu::broadcast", std::vector<c10::IValue>({input}));

      ccl::event ret_evt;
      call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&](){
          CCL_CHECK(ret_evt = ccl::broadcast(input.data_ptr(),
                                             (size_t) input.numel(),
                                             cclDatatypes.at(input.scalar_type()),
                                             root,
                                             comm,
                                             stream,
                                             attr));
      });
      return ret_evt;
    },
    c10d::OpType::BROADCAST);


  work->debugName = std::string("xpu::broadcast");
  execute(work);

  return work;
}


c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> XPUCCLStubs::allgather_(std::vector<std::vector<at::Tensor>>& outputTensors,
                                                                       std::vector<at::Tensor>& inputTensors,
                                                                       const AllgatherOptions& opts,
                                                                       ProcessGroupCCL& pg_ccl) {
  const int rank = pg_ccl.getRank();
  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  work = collective<get_ccl_comms, XPUWorkCCL>(
    pg_ccl,
    inputTensors,
    outputTensors,
    [=](at::Tensor input,
        const std::vector<at::Tensor>& outputs,
        ccl::allgatherv_attr attr,
        ccl::communicator& comm,
        ccl::stream& stream) {
      RECORD_FUNCTION("oneccl_bindings_for_pytorch::xpu::allgather", std::vector<c10::IValue>({input}));

      ccl::event ret_evt;
      std::vector<size_t> recvCounts(outputs.size(), 0);
      std::transform(outputs.begin(), outputs.end(), recvCounts.begin(),
                     [](const at::Tensor& t) {
                          return t.numel();
                     });

      TORCH_CHECK((size_t)input.numel() == recvCounts[rank], "allgather: send and recv count doesn't match");
      std::vector<void*> recvBufs(outputs.size(), nullptr);
      std::transform(outputs.begin(), outputs.end(), recvBufs.begin(),
                     [](const at::Tensor& t) {
                        return t.data_ptr();
                     });

      call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&]() {
        CCL_CHECK(ret_evt = ccl::allgatherv(input.data_ptr(),
                                  (size_t) input.numel(),
                                  recvBufs,
                                  recvCounts,
                                  cclDatatypes.at(input.scalar_type()),
                                  comm,
                                  stream));
      });

      return ret_evt;
    },
    c10d::OpType::ALLGATHER);

  work->debugName = std::string("xpu::allgather");
  execute(work);

  return work;
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> XPUCCLStubs::_allgather_base_(at::Tensor& outputTensor,
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
  work = collective<get_ccl_comms, XPUWorkCCL>(
          pg_ccl,
          inputs,
          outputs,
          [=](at::Tensor input,
              at::Tensor output,
              ccl::allgatherv_attr attr,
              ccl::communicator& comm,
              ccl::stream& stream) {
            RECORD_FUNCTION("oneccl_bindings_for_pytorch::xpu::_allgather_base_", std::vector<c10::IValue>({input}));

            std::vector<size_t> recvCounts(world_size, input.numel());

            ccl::event ret_evt;
            call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&]() {
              CCL_CHECK(ret_evt = ccl::allgatherv(input.data_ptr(),
                                         (size_t) input.numel(),
                                         output.data_ptr(),
                                         recvCounts,
                                         cclDatatypes.at(input.scalar_type()),
                                         comm,
                                         stream));
            });
            return ret_evt;
          },
          c10d::OpType::_ALLGATHER_BASE);

  work->debugName = std::string("xpu::_allgather_base_");
  execute(work);

  return work;
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> XPUCCLStubs::allgather_into_tensor_coalesced_(std::vector<at::Tensor>& outputTensors,
                                                                    std::vector<at::Tensor>& inputTensors,
                                                                    const AllgatherOptions& opts,
                                                                    ProcessGroupCCL& pg_ccl) {
  const int world_size = pg_ccl.getSize();

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  work = collective<get_ccl_comms, XPUWorkCCL>(
          pg_ccl,
          inputTensors,
          outputTensors,
          [=](at::Tensor input,
              at::Tensor output,
              ccl::allgatherv_attr attr,
              ccl::communicator& comm,
              ccl::stream& stream) {
            if (input.numel() * world_size != output.numel()) {
                TORCH_CHECK(false, "output tensor size must be equal to world_size times input tensor size");
            }

            RECORD_FUNCTION("oneccl_bindings_for_pytorch::xpu::allgather_into_tensor_coalesced_", std::vector<c10::IValue>({input}));

            std::vector<size_t> recvCounts(world_size, input.numel());
            ccl::event ret_evt;
            call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&]() {
              CCL_CHECK(ret_evt = ccl::allgatherv(input.data_ptr(),
                                         (size_t) input.numel(),
                                         output.data_ptr(),
                                         recvCounts,
                                         cclDatatypes.at(input.scalar_type()),
                                         comm,
                                         stream));
            });
            return ret_evt;
          },
          c10d::OpType::COALESCED);

  work->debugName = std::string("xpu::allgather_into_tensor_coalesced_");
  execute(work);

  return work;
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> XPUCCLStubs::gather_(std::vector<std::vector<at::Tensor>>& outputTensors,
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
  work = collective<get_ccl_comms, XPUWorkCCL>(
          pg,
          inputTensors,
          outputTensors,
          [=](at::Tensor input,
              const std::vector<at::Tensor>& outputs,
              ccl::alltoallv_attr attr,
              ccl::communicator& comm,
              ccl::stream& stream) {

                ccl::event ret_evt;
                size_t count = input.numel();
                auto type = cclDatatypes.at(input.scalar_type());
                int root = opts.rootRank;

                if (rank == root) {
                    for (const auto r: c10::irange(grp_size)) {
                        if (r != root) {
                            // do receive
                            CCL_CHECK(ret_evt = ccl::recv(outputs[r].data_ptr(), count, type, r, comm, stream));
                        } else {
                            // on its own rank, simply copy from the input
                            outputs[r].copy_(input);
                        }
                    }
                } else {
                    // do send
                    CCL_CHECK(ret_evt = ccl::send(input.data_ptr(), count, type, root, comm, stream));
                }
                
                return ret_evt;
          },
          c10d::OpType::GATHER);

  work->debugName = std::string("xpu::gather");
  execute(work);

  return work;
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> XPUCCLStubs::scatter_(std::vector<at::Tensor>& outputTensors,
                                                                    std::vector<std::vector<at::Tensor>>& inputTensors,
                                                                    const ScatterOptions& opts,
                                                                    ProcessGroupCCL& pg) {
  static auto invalidArgument = [](const std::string& msg) {
    C10_THROW_ERROR(ValueError, "ProcessGroupCCL::scatter: " + msg);
  };

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  auto grp_size = pg.getSize();
  auto rank = pg.getRank();

  if (rank == opts.rootRank) {
    if (inputTensors.size() != 1) {
      std::stringstream ss;
      ss << "requires a single-element input list containing a list with " << grp_size << " tensors.";
      invalidArgument(ss.str());
    } else if (inputTensors[0].size() != static_cast<size_t>(grp_size)) {
      std::stringstream ss;
      ss << "Incorrect input list size " << inputTensors[0].size()
        << ". Input list size should be " << grp_size
        << ", same as size of the process group.";
        invalidArgument(ss.str());
    }
  }
  else {
    TORCH_CHECK(inputTensors.size() == 0,
                "scatter: number of input tensors should be 0 "
                "for non-root");
  }
  work = collective<get_ccl_comms, XPUWorkCCL>(
          pg,
          inputTensors,
          outputTensors,
          [=](const std::vector<at::Tensor>& inputs,
              at::Tensor output,
              ccl::alltoallv_attr attr,
              ccl::communicator& comm,
              ccl::stream& stream) {
                ccl::event ret_evt;
                int root = opts.rootRank;
                if (rank == root) {
                    for (const auto r: c10::irange(grp_size)) {
                        if (r != root) {
                            // do send
                            size_t send_count = inputs[r].numel();
                            auto send_type = cclDatatypes.at(inputs[r].scalar_type());
                            CCL_CHECK(ret_evt = ccl::send(inputs[r].data_ptr(), send_count, send_type, r, comm, stream));
                        } else {
                            // on its own rank, simply copy from the input
                            output.copy_(inputs[r]);
                        }
                    }
                } else {
                    // do receive
                    size_t recv_count = output.numel();
                    auto recv_type = cclDatatypes.at(output.scalar_type());
                    CCL_CHECK(ret_evt = ccl::recv(output.data_ptr(), recv_count, recv_type, root, comm, stream));
                }
                
                return ret_evt;
          },
          c10d::OpType::SCATTER);

  work->debugName = std::string("xpu::scatter");
  execute(work);

  return work;
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> XPUCCLStubs::alltoall_base_(at::Tensor& outputTensor,
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
    work = collective<get_ccl_comms, XPUWorkCCL>(
            pg,
            inputs,
            outputs,
            [=](at::Tensor input,
                at::Tensor output,
                ccl::alltoall_attr attr,
                ccl::communicator& comm,
                ccl::stream& stream) {
                ccl::event ret_evt;

                call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&](){
                    CCL_CHECK(ret_evt = ccl::alltoall(input.data_ptr(),
                                                      output.data_ptr(),
                                                      (size_t)output.numel() / comm.size(),
                                                      cclDatatypes.at(output.scalar_type()),
                                                      comm,
                                                      stream,
                                                      attr));
                });

                return ret_evt;
            },
            c10d::OpType::ALLTOALL_BASE);
  }
  else{
    // Need alltoallv
    work = collective<get_ccl_comms, XPUWorkCCL>(
            pg,
            inputs,
            outputs,
            [=](at::Tensor input,
                at::Tensor output,
                ccl::alltoallv_attr attr,
                ccl::communicator& comm,
                ccl::stream& stream) {
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
                                                       stream,
                                                       attr));
                });
                return ret_evt;
            },
            c10d::OpType::ALLTOALL_BASE);
  }

  work->debugName = std::string("xpu::alltoall_base");
  execute(work);

  return work;
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> XPUCCLStubs::alltoall_(std::vector<at::Tensor>& outputTensors,
                                                                      std::vector<at::Tensor>& inputTensors,
                                                                      const AllToAllOptions& opts,
                                                                      ProcessGroupCCL& pg){
  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  auto grp_size = pg.getSize();

  std::vector<std::vector<at::Tensor>> outputTensors_list = {outputTensors};
  std::vector<std::vector<at::Tensor>> inputTensors_list = {inputTensors};
  work = collective<get_ccl_comms, XPUWorkCCL>(
          pg,
          inputTensors_list,
          outputTensors_list,
          [=](std::vector<at::Tensor> inputs,
              std::vector<at::Tensor> outputs,
              ccl::alltoallv_attr attr,
              ccl::communicator& comm,
              ccl::stream& stream,
              c10::Stream& torch_stream) {

              c10::OptionalStreamGuard stream_guard(torch_stream);

              at::Tensor flatInput;
              at::Tensor flatOutput;

              std::vector<size_t> sendCounts(grp_size);
              std::vector<size_t> recvCounts(grp_size);

              int64_t flatSendCount;
              int64_t flatRecvCount;

              bool isInputFlat =
                      computeLengthsAndCheckAndGetFlat(inputTensors, sendCounts, flatInput, flatSendCount);

              bool isOutputFlat =
                      computeLengthsAndCheckAndGetFlat(outputTensors, recvCounts, flatOutput, flatRecvCount);

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
                                                     stream));
              });

              if (!isOutputFlat) {
                ret_evt.wait();
                auto flatOutputSplits =
                        flatOutput.split_with_sizes(c10::IntArrayRef((int64_t*)recvCounts.data(),
                                                                     recvCounts.size()), 0);

                for (int i = 0; i < grp_size; i++)
                {
                  outputs[i].view({-1}).copy_(flatOutputSplits[i]);
                }
              }

	      torch_stream.synchronize();
              return ret_evt;
          },
          c10d::OpType::ALLTOALL);

  work->debugName = std::string("xpu::alltoall");
  execute(work);

  return work;
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> XPUCCLStubs::send_(std::vector<at::Tensor>& tensors,
                                                                       int dstRank,
                                                                       int /* unused */,
                                                                       ProcessGroupCCL& pg_ccl) {

  checkGPUTensor(tensors);
  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  work = pointToPoint<get_ccl_comms, XPUP2PAsyncWork>(
    pg_ccl,
    tensors,
    tensors,
    [=](at::Tensor input,
        int dst,
        ccl::pt2pt_attr attr,
        ccl::communicator& comm,
        ccl::stream& stream) {
      RECORD_FUNCTION("oneccl_bindings_for_pytorch::xpu::send", std::vector<c10::IValue>({input}));

      ccl::event ret_evt;
      call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&](){
          CCL_CHECK(ret_evt = ccl::send(input.data_ptr(),
                                             (size_t) input.numel(),
                                             cclDatatypes.at(input.scalar_type()),
                                             dst,
                                             comm,
                                             stream,
                                             attr));
      });
      return ret_evt;
  },
  dstRank,
  c10d::OpType::SEND);

  work->debugName = std::string("xpu::send");
  execute(work);

  return work;
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> XPUCCLStubs::recv_(std::vector<at::Tensor>& tensors,
                                                                       int srcRank,
                                                                       int /* unused */,
                                                                       ProcessGroupCCL& pg_ccl) {

  checkGPUTensor(tensors);
  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  work = pointToPoint<get_ccl_comms, XPUP2PAsyncWork>(
    pg_ccl,
    tensors,
    tensors,
    [=](at::Tensor output,
        int src,
        ccl::pt2pt_attr attr,
        ccl::communicator& comm,
        ccl::stream& stream) {
      RECORD_FUNCTION("oneccl_bindings_for_pytorch::xpu::recv", std::vector<c10::IValue>({output}));

      ccl::event ret_evt;
      call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&](){
          CCL_CHECK(ret_evt = ccl::recv(output.data_ptr(),
                                             (size_t) output.numel(),
                                             cclDatatypes.at(output.scalar_type()),
                                             src,
                                             comm,
                                             stream,
                                             attr));
      });
      return ret_evt;
  },
  srcRank,
  c10d::OpType::RECV);

  work->debugName = std::string("xpu::recv");
  execute(work);

  return work;
}

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> XPUCCLStubs::barrier_(const BarrierOptions& opts,
                                                                   ProcessGroupCCL& pg) {

  c10::intrusive_ptr<AsyncBarrierWork> work = c10::make_intrusive<AsyncBarrierWork>();

  if (pg.ccl_member_->ccl_comms.size() == 0) {
    std::vector<at::Device> xpu_devices{at::Device(at::kXPU)};
    const auto key = get_key_from_devs(xpu_devices);
    get_ccl_comms(pg, key, xpu_devices);
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

c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> XPUCCLStubs::end_coalescing_(ProcessGroupCCL& pg_ccl) {
  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  // Create empty tensors.
  std::vector<at::Tensor> tensors;
  work = collective<get_ccl_comms, XPUWorkCCL>(
    pg_ccl,
    tensors,
    tensors,
    [=](at::Tensor input,
        at::Tensor output,
        ccl::allreduce_attr attr, // This is a silly attr, which will not be used. It could be any attr type of ccl.
        ccl::communicator& comm,
        ccl::stream& stream) {
      RECORD_FUNCTION("oneccl_bindings_for_pytorch::xpu::coalescing", std::vector<c10::IValue>{});
      ccl::event ret_evt;
      return ret_evt;
    },
    c10d::OpType::COALESCED);

  work->debugName = std::string("xpu::end_coalescing");
  execute(work);

  return work;    
}

RegisterXPUMethods xpu_register;

}
