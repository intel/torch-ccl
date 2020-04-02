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

#include "ProcessGroupCCL.hpp"

#include <map>

namespace c10d
{

#define CCL_CHECK(cmd)                                               \
  do {                                                               \
    try {                                                            \
        cmd;                                                         \
    }                                                                \
    catch (ccl::ccl_error& e) {                                      \
        throw std::runtime_error("CCL error in: " +                  \
            std::string(__FILE__) + ":" + std::to_string(__LINE__) + \
            ", with error message: " + e.what());                    \
    }                                                                \
    catch (...) {                                                    \
        throw std::runtime_error("unknown error in: " +              \
            std::string(__FILE__) + ":" + std::to_string(__LINE__)); \
    }                                                                \
  } while (0)

namespace {

// Op mapping
std::map<ReduceOp, ccl::reduction> cclOps =
{
    {ReduceOp::MIN, ccl::reduction::min},
    {ReduceOp::MAX, ccl::reduction::max},
    {ReduceOp::SUM, ccl::reduction::sum},
    {ReduceOp::PRODUCT, ccl::reduction::prod},
};

// Type mapping
std::map<at::ScalarType, ccl::data_type> cclDatatypes =
{
    {at::kByte, ccl::data_type::dt_char},
    {at::kChar, ccl::data_type::dt_char},
    {at::kDouble, ccl::data_type::dt_double},
    {at::kBFloat16, ccl::data_type::dt_bfp16},
    {at::kFloat, ccl::data_type::dt_float},
    {at::kInt, ccl::data_type::dt_int},
    {at::kLong, ccl::data_type::dt_int64}
};

static std::once_flag cclInitOnceFlag;
static std::mutex globalMutex;
static ccl::communicator_t globalComm;
static ccl::coll_attr collAttr;
static ccl::coll_attr collAttrAg;

// Checking the input tensor's validity
void checkSingleTensorHelper(const at::Tensor& tensor)
{
    if (!tensor.is_contiguous())
    {
        throw std::runtime_error("input tensor has to be contiguous");
    }

    if (tensor.is_sparse())
    {
        throw std::runtime_error("input tensor has to be dense");
    }

    if (tensor.is_cuda())
    {
        throw std::runtime_error("CUDA tensor detected and CCL doesn't support CUDA buffers");
    }

    if (tensor.numel() < 0)
    {
        throw std::runtime_error("input tensor numel should be non-negative");
    }
}

void checkRank(int rank, int size)
{
    if (rank < 0 || rank >= size)
    {
        throw std::runtime_error("unexpected rank");
    }
}

void checkSingleTensor(const std::vector<at::Tensor>& tensors)
{
    if (tensors.size() != 1)
    {
        throw std::runtime_error(
            "CCL process group does not support tensors count " + std::to_string(tensors.size()));
    }
    checkSingleTensorHelper(tensors[0]);
}

void checkSameSizeAndType(const at::Tensor& tensor,
                          const std::vector<at::Tensor>& tensors)
{
    for (size_t i = 0; i < tensors.size(); ++i)
    {
        if ((tensors[i].numel() != tensor.numel()) ||
            (tensors[i].type() != tensor.type()))
        {
            throw std::runtime_error("tensors are not equal in size or data type");
        }
        checkSingleTensorHelper(tensors[i]);
    }
}

void checkSplitSizes(
    const std::vector<int64_t>& split_sizes,
    const at::Tensor& tensor,
    int group_size)
{
    if (split_sizes.size() == 0)
    {
        TORCH_CHECK(
            tensor.size(0) % group_size == 0,
            "Tensor's dim 0 does not divide equally across group size");
    }
    else
    {
        TORCH_CHECK(
            split_sizes.size() == (size_t)group_size,
            "Number of tensor splits not equal to group size");
        int sum = std::accumulate(split_sizes.begin(), split_sizes.end(), 0);
        TORCH_CHECK(
            sum == tensor.size(0), "Split sizes doesn't match total dim 0 size");
    }
}

/*int64_t computeLengthsAndOffsets(
    const std::vector<int64_t>& split_sizes,
    const at::Tensor& tensor,
    std::vector<int>* lengths,
    std::vector<int>* offsets)
{
    int64_t group_size = lengths->size();
    bool equal_splits = false;
    int64_t dim0_size = tensor.size(0);
    int64_t row_size = (dim0_size ? tensor.numel() / dim0_size : 1);
    int64_t split_size = 0;
    int64_t offset = 0;

    if (split_sizes.size() == 0)
    {
        equal_splits = true;
        split_size = tensor.size(0) / group_size;
    }

    for (int i = 0; i < group_size; i++)
    {
        int64_t length = row_size * (equal_splits ? split_size : split_sizes[i]);
        TORCH_INTERNAL_ASSERT(
            length <= std::numeric_limits<int>::max() &&
                offset <= std::numeric_limits<int>::max(),
            "Length or offset larger than INT_MAX not supported");
        (*lengths)[i] = length;
        (*offsets)[i] = offset;
        offset += length;
    }

    return offset;
}

int64_t computeLengthsAndOffsets(
    const std::vector<at::Tensor>& tensors,
    std::vector<int>* lengths,
    std::vector<int>* offsets)
{
    int64_t group_size = lengths->size();
    int64_t offset = 0;

    for (int i = 0; i < group_size; i++)
    {
        int64_t length = tensors[i].numel();
        TORCH_INTERNAL_ASSERT(
            length <= std::numeric_limits<int>::max() &&
                offset <= std::numeric_limits<int>::max(),
            "Length or offset larger than INT_MAX not supported");
        (*lengths)[i] = length;
        (*offsets)[i] = offset;
        offset += length;
    }

    return offset;
}*/

} // namespace

ProcessGroupCCL::WorkCCL::~WorkCCL()
{
    if (req)
    {
        std::cerr << "attempted destruction of WorkCCL before work has completed, "
                  << "terminating the program."
                  << std::endl;
        std::terminate();
    }
}

bool ProcessGroupCCL::WorkCCL::isCompleted()
{
    if (!req)
    {
        return true;
    }

    bool flag = false;

    std::unique_lock<std::mutex> globalLock(globalMutex);
    CCL_CHECK(flag = req->test());

    if (flag)
    {
        req.reset();
        tensors.clear();
    }

    return flag;
}

bool ProcessGroupCCL::WorkCCL::isSuccess() const
{
    if (req)
    {
        throw std::runtime_error(
            "invalid call to WorkCCL::isSuccess before work has completed");
    }
    return true;
}

bool ProcessGroupCCL::WorkCCL::wait()
{
    if (!req)
    {
        return true;
    }

    std::unique_lock<std::mutex> globalLock(globalMutex);
    CCL_CHECK(req->wait());
    req.reset();
    tensors.clear();

    // Always return true, because abort API is not implemented.
    return true;
}

void ProcessGroupCCL::WorkCCL::abort()
{
    TORCH_CHECK(false, "ProcessGroupCCL::WorkCCL::abort not implemented.")
}

#ifdef USE_VECTOR_ALLGATHERV
thread_local std::vector<void*> ProcessGroupCCL::agRecvBuffers;
#endif

void ProcessGroupCCL::cclFini()
{
    std::unique_lock<std::mutex> globalLock(globalMutex);
    globalComm.reset();
}

void ProcessGroupCCL::cclInitOnce()
{
    std::call_once(cclInitOnceFlag, []() {

#ifdef USE_CACHE
      /* to enable collective caching */
      collAttr.to_cache = 1;
      collAttrAg.to_cache = 1;
#endif

#ifdef USE_VECTOR_ALLGATHERV
      /* to enable allgatherv with recv buffers vector */
      collAttrAg.vector_buf = 1;
#endif

      CCL_CHECK(globalComm = ccl::environment::instance().create_communicator());

#ifdef USE_VECTOR_ALLGATHERV
      agRecvBuffers.reserve(globalComm->size());
#endif

      if (std::atexit(ProcessGroupCCL::cclFini))
      {
          throw std::runtime_error("failed to register the CCL exit handler");
      }
  });
}

std::shared_ptr<ProcessGroup> ProcessGroupCCL::createProcessGroupCCL(
    const std::shared_ptr<Store>& store,
    int rank,
    int size,
    const std::chrono::duration<float>& timeout)
{
    cclInitOnce();

    if ((rank != -1) && (rank < 0 || (size_t)rank != globalComm->rank()))
    {
        throw std::runtime_error("unexpected rank " + std::to_string(rank) +
                                 ", CCL rank " + std::to_string(globalComm->rank()));
    }

    if ((size != -1) && (size <= 0 || (size_t)size != globalComm->size()))
    {
        throw std::runtime_error("unexpected size " + std::to_string(size) +
                                 ", CCL size " + std::to_string(globalComm->size()));
    }

    return std::make_shared<ProcessGroupCCL>(rank, size);
}

ProcessGroupCCL::ProcessGroupCCL(int rank, int size)
    : ProcessGroup(globalComm->rank(),
                   globalComm->size()),
      comm(globalComm.get())
{}

ProcessGroupCCL::~ProcessGroupCCL() {}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts)
{
    checkSingleTensor(tensors);
    checkRank(opts.rootRank, getSize());

#ifdef USE_CACHE
    collAttr.match_id = tensorName.c_str();
#endif

    std::shared_ptr<ccl::request> req;

    std::unique_lock<std::mutex> globalLock(globalMutex);
    CCL_CHECK(req = comm->bcast(tensors[0].data_ptr(),
                                (size_t)tensors[0].numel(),
                                cclDatatypes.at(tensors[0].scalar_type()),
                                (size_t)opts.rootRank,
                                &collAttr));

    return std::make_shared<ProcessGroupCCL::WorkCCL>(req, tensors);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts)
{
    checkSingleTensor(tensors);

#ifdef USE_CACHE
    collAttr.match_id = tensorName.c_str();
#endif

    std::shared_ptr<ccl::request> req;

    std::unique_lock<std::mutex> globalLock(globalMutex);
    CCL_CHECK(req = comm->allreduce(tensors[0].data_ptr(),
                                    tensors[0].data_ptr(),
                                    (size_t)tensors[0].numel(),
                                    cclDatatypes.at(tensors[0].scalar_type()),
                                    cclOps.at(opts.reduceOp),
                                    &collAttr));

    return std::make_shared<ProcessGroupCCL::WorkCCL>(req, tensors);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::allreduce_coalesced(
    std::vector<at::Tensor>& /* unused */,
    const AllreduceCoalescedOptions& /* unused */)
{
    throw std::runtime_error("ProcessGroupCCL does not support allreduce_coalesced");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts)
{
    checkSingleTensor(tensors);
    checkRank(opts.rootRank, getSize());

#ifdef USE_CACHE
    collAttr.match_id = tensorName.c_str();
#endif

    std::shared_ptr<ccl::request> req;

    std::unique_lock<std::mutex> globalLock(globalMutex);
    CCL_CHECK(req = comm->reduce(tensors[0].data_ptr(),
                                 tensors[0].data_ptr(),
                                 (size_t)tensors[0].numel(),
                                 cclDatatypes.at(tensors[0].scalar_type()),
                                 cclOps.at(opts.reduceOp),
                                 (size_t)opts.rootRank,
                                 &collAttr));

    return std::make_shared<ProcessGroupCCL::WorkCCL>(req, tensors);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts)
{
    checkSingleTensor(inputTensors);

    if (outputTensors.size() != 1)
    {
        throw std::runtime_error(
            "ProcessGroupCCL/allgather supports a single tensor op only");
    }

    if (comm->size() != outputTensors[0].size())
    {
        throw std::runtime_error(
            "ProcessGroupCCL/allgather: number of output tensors should equal "
            "to the world size");
    }

    checkSameSizeAndType(inputTensors[0], outputTensors[0]);

#ifdef USE_CACHE
    collAttrAg.match_id = tensorName.c_str();
#endif

#ifdef USE_VECTOR_ALLGATHERV
    agRecvBuffers.clear();
    std::transform(outputTensors[0].begin(), outputTensors[0].end(),
                   std::back_inserter(agRecvBuffers), [](const at::Tensor& t) { return t.data_ptr(); } );
#else
    auto flatOutputTensor = newLikeFlat(outputTensors[0]);
#endif
    std::vector<size_t> recvCounts(comm->size(), inputTensors[0].numel());

    std::shared_ptr<ccl::request> req;

    std::unique_lock<std::mutex> globalLock(globalMutex);
    CCL_CHECK(req = comm->allgatherv(inputTensors[0].data_ptr(),
                                     (size_t)inputTensors[0].numel(),
#ifdef USE_VECTOR_ALLGATHERV
                                     agRecvBuffers.data(),
#else
                                     flatOutputTensor.data_ptr(),
#endif
                                     (size_t*)recvCounts.data(),
                                     cclDatatypes.at(inputTensors[0].scalar_type()),
                                     &collAttrAg));

#ifdef USE_VECTOR_ALLGATHERV
    auto agTensors = std::vector<at::Tensor>(outputTensors[0].begin(), outputTensors[0].end());
    agTensors.emplace_back(inputTensors[0]);
#else
    req->wait();
    for (size_t i = 0; i < outputTensors[0].size(); ++i)
    {
        outputTensors[0][i].copy_(flatOutputTensor[i]);
    }
    std::vector<at::Tensor> agTensors;
#endif

    return std::make_shared<ProcessGroupCCL::WorkCCL>(req, std::move(agTensors));
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& /* unused */)
{
    throw std::runtime_error("ProcessGroupCCL does not support allgather_base");
}


std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllgatherOptions& /* unused */)
{
    throw std::runtime_error("ProcessGroupCCL does not support allgather_coalesced");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::gather(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const GatherOptions& /* unused */)
{
    throw std::runtime_error("ProcessGroupCCL does not support gather");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ScatterOptions& /* unused */)
{
    throw std::runtime_error("ProcessGroupCCL does not support scatter");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::reduce_scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ReduceScatterOptions& /* unused */)
{
    throw std::runtime_error("ProcessGroupCCL does not support reduce_scatter");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& opts)
{
    checkSingleTensorHelper(inputTensor);
    checkSingleTensorHelper(outputTensor);

    if (outputSplitSizes.size() == 0 && inputSplitSizes.size() == 0)
    {
        // We can use alltoall
        TORCH_CHECK(
            outputTensor.numel() == inputTensor.numel() &&
                outputTensor.type() == inputTensor.type(),
            "Tensors are not equal in size or data type");
        TORCH_CHECK(
            outputTensor.size(0) % size_ == 0,
            "Tensor's dim 0 does not divide equally across group size");

        std::shared_ptr<ccl::request> req;

        std::unique_lock<std::mutex> globalLock(globalMutex);
        CCL_CHECK(req = comm->alltoall(inputTensor.data_ptr(),
                                       outputTensor.data_ptr(),
                                       (size_t)outputTensor.numel() / comm->size(),
                                       cclDatatypes.at(outputTensor.scalar_type()),
                                       &collAttr));

        auto a2aTensors = std::vector<at::Tensor> { inputTensor, outputTensor };
        return std::make_shared<ProcessGroupCCL::WorkCCL>(req, std::move(a2aTensors));
    }
    else
    {
        // Need alltoallv
        checkSplitSizes(inputSplitSizes, inputTensor, size_);
        checkSplitSizes(outputSplitSizes, outputTensor, size_);

        std::vector<size_t> send_counts(size_);
        std::vector<size_t> recv_counts(size_);

        size_t inLen = inputSplitSizes.size() == 0 ? inputTensor.numel() / size_ : 0;
        size_t outLen = outputSplitSizes.size() == 0 ? outputTensor.numel() / size_ : 0;

        for (int i = 0; i < size_; i++)
        {
            send_counts[i] = (inLen > 0 ? inLen : inputSplitSizes[i]);
            recv_counts[i] = (outLen > 0 ? outLen : outputSplitSizes[i]);
        }

        std::shared_ptr<ccl::request> req;

        std::unique_lock<std::mutex> globalLock(globalMutex);
        CCL_CHECK(req = comm->alltoallv(inputTensor.data_ptr(),
                                        send_counts.data(),
                                        outputTensor.data_ptr(),
                                        recv_counts.data(),
                                        cclDatatypes.at(outputTensor.scalar_type()),
                                        &collAttr));

        auto a2aTensors = std::vector<at::Tensor> { inputTensor, outputTensor };
        return std::make_shared<ProcessGroupCCL::WorkCCL>(req, std::move(a2aTensors));
    }
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllToAllOptions& opts)
{
    TORCH_CHECK(
        inputTensors.size() == (size_t)size_,
        "Number of input tensors are not equal to group size");
    TORCH_CHECK(
        outputTensors.size() == (size_t)size_,
        "Number of output tensors are not equal to group size");

    /* TODO */
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::send(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */,
    int /* unused */)
{
    throw std::runtime_error("ProcessGroupCCL does not support send");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::recv(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */,
    int /* unused */)
{
    throw std::runtime_error("ProcessGroupCCL does not support recv");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::recvAnysource(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */)
{
    throw std::runtime_error("ProcessGroupCCL does not support recvAnysource");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::barrier(
    const BarrierOptions& opts)
{
    std::unique_lock<std::mutex> globalLock(globalMutex);
    CCL_CHECK(comm->barrier());
    return std::make_shared<ProcessGroupCCL::WorkCCL>();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("createProcessGroupCCL", &ProcessGroupCCL::createProcessGroupCCL);
}

} // namespace c10d
