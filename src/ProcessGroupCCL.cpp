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

#include <map>

#include <omp.h>

#include <ATen/record_function.h>

#include "ProcessGroupCCL.hpp"

#ifndef gettid
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#define gettid() syscall(SYS_gettid)
#endif

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
#if CCL_MAJOR_VERSION == 0 && CCL_MINOR_VERSION < 6
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
#else
std::map<at::ScalarType, ccl::datatype> cclDatatypes =
{
    {at::kByte, ccl::datatype::dt_char},
    {at::kChar, ccl::datatype::dt_char},
    {at::kDouble, ccl::datatype::dt_double},
    {at::kBFloat16, ccl::datatype::dt_bfp16},
    {at::kFloat, ccl::datatype::dt_float},
    {at::kInt, ccl::datatype::dt_int},
    {at::kLong, ccl::datatype::dt_int64}
};

std::map<ccl_datatype_t, at::ScalarType> ptDatatypes =
{
    {ccl_dtype_char, at::kByte},
    {ccl_dtype_int, at::kInt},
    {ccl_dtype_bfp16, at::kBFloat16},
    {ccl_dtype_float, at::kFloat},
    {ccl_dtype_double, at::kDouble},
    {ccl_dtype_int64, at::kLong}
};
#endif

enum class
SparseResultMode : std::uint8_t
{
    DIRECT,
    OOP,
    COPY
};

std::ostream& operator << (std::ostream& os, const SparseResultMode& mode)
{
   os << static_cast<std::underlying_type<SparseResultMode>::type>(mode);
   return os;
}

static std::once_flag cclInitOnceFlag;
static std::mutex globalMutex;
static ccl::communicator_t globalComm;
static ccl_sparse_coalesce_mode_t sparseCoalesceMode;
static SparseResultMode sparseResultMode;

// Checking the input tensor's validity
void checkSingleTensorHelper(const at::Tensor& tensor)
{
    TORCH_CHECK(tensor.is_sparse() || tensor.is_contiguous(), "input dense tensor has to be contiguous");
    TORCH_CHECK(!tensor.is_cuda(), "CUDA tensor detected and CCL doesn't support CUDA buffers");
    TORCH_CHECK(tensor.numel() >= 0, "input tensor numel should be non-negative");
}

void checkRank(int rank, int size)
{
    TORCH_CHECK((rank >= 0) && (rank < size), "unexpected rank");
}

void checkSingleTensor(const std::vector<at::Tensor>& tensors)
{
    TORCH_CHECK(tensors.size() == 1,
        "CCL process group does not support tensors count " + std::to_string(tensors.size()));

    checkSingleTensorHelper(tensors[0]);
}

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

void checkSameType(const at::Tensor& tensor,
                   const std::vector<at::Tensor>& tensors)
{
    for (size_t i = 0; i < tensors.size(); ++i)
    {
        TORCH_CHECK(tensors[i].scalar_type() == tensor.scalar_type(),
            "tensors are not equal in data type");

        checkSingleTensorHelper(tensors[i]);
    }
}

void checkSplitSizes(
    const std::vector<int64_t>& split_sizes,
    const at::Tensor& tensor,
    int groupSize)
{
    if (split_sizes.size() == 0)
    {
        TORCH_CHECK(tensor.size(0) % groupSize == 0,
            "tensor's dim 0 does not divide equally across group size");
    }
    else
    {
        TORCH_CHECK(split_sizes.size() == (size_t)groupSize,
            "number of tensor splits not equal to group size");

        int sum = std::accumulate(split_sizes.begin(), split_sizes.end(), 0);

        TORCH_CHECK(sum == tensor.size(0),
            "split sizes doesn't match total dim 0 size");
    }
}

typedef struct
{
    bool isFlat;
    int64_t size;
    at::Tensor firstTensor;
} FlatCheckResult;

FlatCheckResult computeLengthsAndCheckFlat(
    const std::vector<at::Tensor>& tensors,
    std::vector<size_t>& lengths)
{
    int64_t groupSize = lengths.size();
    auto firstTensor = tensors[0];
    int64_t offset = 0;
    auto firstLength = firstTensor.numel();
    auto storage = firstTensor.storage();
    auto firstStorageOffset = firstTensor.storage_offset();
    bool isFlat = true;

    for (int i = 0; i < groupSize; i++)
    {
        auto& curTensor = tensors[i];
        int64_t length = curTensor.numel();

        if (firstLength == 0 && length != 0)
        {
            firstLength = length;
            firstTensor = curTensor;
            storage = curTensor.storage();
            firstStorageOffset = curTensor.storage_offset();
        }

        lengths[i] = length;

        if (isFlat && length != 0 &&
            (!storage.is_alias_of(curTensor.storage()) ||
             curTensor.storage_offset() != firstStorageOffset + offset))
            isFlat = false;

        offset += length;
    }

    return FlatCheckResult{isFlat, offset, firstTensor};
}

bool computeLengthsAndCheckAndGetFlat(
    const std::vector<at::Tensor>& tensors,
    std::vector<size_t>& lengths,
    at::Tensor& flatTensor,
    int64_t& flatLength)
{
    auto flatRes = computeLengthsAndCheckFlat(tensors, lengths);

    flatLength = flatRes.size;

    if (flatRes.isFlat)
    {
        flatTensor = flatRes.firstTensor;
    }
    else
    {
        flatTensor = at::empty({flatRes.size}, flatRes.firstTensor.options());
    }

    return flatRes.isFlat;
}

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

    RECORD_FUNCTION(std::string("pg::wait::") + debugName, std::vector<c10::IValue>());

    std::unique_lock<std::mutex> globalLock(globalMutex);
    CCL_CHECK(req->wait());
    req.reset();
    tensors.clear();

    // Always return true, because abort API is not implemented.
    return true;
}

void ProcessGroupCCL::WorkCCL::abort()
{
    TORCH_CHECK(false, "ProcessGroupCCL::WorkCCL::abort not implemented");
}

void ProcessGroupCCL::cclFini()
{
    std::unique_lock<std::mutex> globalLock(globalMutex);
    globalComm.reset();
}

void ProcessGroupCCL::cclInitOnce()
{
    std::call_once(cclInitOnceFlag, []() {

      CCL_CHECK(globalComm = ccl::environment::instance().create_communicator());

      if (std::atexit(ProcessGroupCCL::cclFini))
      {
          throw std::runtime_error("failed to register the CCL exit handler");
      }

      sparseCoalesceMode = ccl_sparse_coalesce_regular;
      const char* sparseCoalesceModeEnv = getenv("CCL_SPARSE_COALESCE_MODE");
      if (sparseCoalesceModeEnv)
      {
          sparseCoalesceMode = (ccl_sparse_coalesce_mode_t)(atoi(sparseCoalesceModeEnv));
      }

      sparseResultMode = SparseResultMode::DIRECT;
      const char* sparseResultModeEnv = getenv("CCL_SPARSE_RESULT_MODE");
      if (sparseResultModeEnv)
      {
          sparseResultMode = (SparseResultMode)atoi(sparseResultModeEnv);
      }

      if (globalComm->rank() == 0)
      {
          printf("sparse options: coalesce mode %d, result mode %d\n",
                 sparseCoalesceMode, (int)sparseResultMode);
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

    TORCH_CHECK(((rank == -1) || (size_t)rank == globalComm->rank()),
        "unexpected rank " + std::to_string(rank) +
        ", CCL rank " + std::to_string(globalComm->rank()));

    TORCH_CHECK(((size == -1) || (size_t)size == globalComm->size()),
        "unexpected size " + std::to_string(size) +
        ", CCL size " + std::to_string(globalComm->size()));

    return std::make_shared<ProcessGroupCCL>(rank, size);
}

ProcessGroupCCL::ProcessGroupCCL(int rank, int size, const std::vector<int> ranks)
    : ProcessGroup(ranks.empty() ? globalComm->rank() : rank,
                   ranks.empty() ? globalComm->size() : size),
      collAttrAg({})
{
    std::unique_lock<std::mutex> globalLock(globalMutex);

    if (ranks.empty())
    {
        CCL_CHECK(comm = ccl::environment::instance().create_communicator());
    }
    else
    {
        int color;
        commAttr = ccl::environment::instance().create_host_comm_attr();

        if (std::find(ranks.begin(), ranks.end(), rank) != ranks.end())
        {
            color = 1;
        }
        else
        {
            color = 0;
        }

        commAttr->set_value<ccl_host_color>(color);
        CCL_CHECK(comm = ccl::environment::instance().create_communicator(commAttr));

        if (commAttr->get_value<ccl_host_color>() == 0)
        {
            comm.reset();
        }
    }
}

ProcessGroupCCL::~ProcessGroupCCL()
{
    std::unique_lock<std::mutex> globalLock(globalMutex);
    comm.reset();
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts)
{
    RECORD_FUNCTION("pg::bcast", std::vector<c10::IValue>({tensors[0]}));

    checkSingleTensor(tensors);
    checkRank(opts.rootRank, getSize());

    std::shared_ptr<ccl::request> req;

    {
        std::unique_lock<std::mutex> globalLock(globalMutex);
        CCL_CHECK(req = comm->bcast(tensors[0].data_ptr(),
                                    (size_t)tensors[0].numel(),
                                    cclDatatypes.at(tensors[0].scalar_type()),
                                    (size_t)opts.rootRank));
    }

    std::string debugName = std::string("bcast::sz:") + std::to_string(tensors[0].numel());

    return std::make_shared<ProcessGroupCCL::WorkCCL>(req, tensors, std::move(debugName));
}

class SparseAllreduceWork : public ProcessGroupCCL::WorkCCL
{
public:

     SparseAllreduceWork(const std::vector<at::Tensor>& tensors,
                         std::string&& debugName) :
          ProcessGroupCCL::WorkCCL(tensors, std::move(debugName))
      {}

      std::vector<at::Tensor>& getOutputTensors()
      {
          return outputTensors;
      }

      std::vector<at::Tensor> result() const override
      {
          TORCH_CHECK(outputTensors.size() == 1, "unexpected result size");
          return outputTensors;
      }

private:
      std::vector<at::Tensor> outputTensors;
};

ccl_status_t sparseAllreduceCompletionFn(
    const void* indBuf, size_t indCount, ccl_datatype_t indDatatype,
    const void* valBuf, size_t valCount, ccl_datatype_t valDatatype,
    const void* fnCtx)
{
    TORCH_CHECK(fnCtx, "null fn ctx");

    /*printf("sparseAllreduceCompletionFn: "
           "indices buf %p, count %zu, dt %d, "
           "values buf %p, count %zu, dt %d, "
           "fn_ctx %p\n",
           indBuf, indCount, indDatatype,
           valBuf, valCount, valDatatype,
           fnCtx); fflush(stdout);*/

    SparseAllreduceWork* work = (SparseAllreduceWork*)(fnCtx);

    std::vector<at::Tensor>& inputTensors = work->getTensors();
    std::vector<at::Tensor>& outputTensors = work->getOutputTensors();

    TORCH_CHECK(inputTensors.size() == 1, "unexpected inputTensors size");
    TORCH_CHECK(outputTensors.size() == 0, "unexpected outputTensors size");

    outputTensors.reserve(inputTensors.size());

    at::Tensor& inputTensor = inputTensors[0];

    TORCH_CHECK(inputTensor.sparse_dim() == 1, "unexpected sparse_dim");

    const auto valueShape = inputTensor.sizes().slice(inputTensor.sparse_dim());
    auto resultValueShape = std::vector<int64_t>({(int64_t)indCount});
    std::copy(valueShape.begin(), valueShape.end(), std::back_inserter(resultValueShape));

    auto rawIndices = at::from_blob((void*)indBuf,
                                    {1, (long int)indCount},
                                    ptDatatypes.at(indDatatype));

    auto rawValues = at::from_blob((void*)valBuf,
                                   resultValueShape,
                                   ptDatatypes.at(valDatatype));

    auto indices = at::empty({1, (long int)indCount}, inputTensor._indices().options());

    auto values = at::empty(resultValueShape,
                            inputTensor._values().options());

    indices.copy_(rawIndices);
    values.copy_(rawValues);

    /*int64_t* indPtr = indices.data_ptr<int64_t>();
    for (size_t idx = 0; idx < indCount; idx++)
    {
        printf("indices[%zu] = %ld\n", idx, indPtr[idx]);
    }

    float* valPtr = values.data_ptr<float>();
    for (size_t idx = 0; idx < valCount; idx++)
    {
        printf("values[%zu] = %f\n", idx, valPtr[idx]);
    }*/

    auto resultTensor =
        at::_sparse_coo_tensor_unsafe(indices,
                                      values,
                                      inputTensor.sizes(),
                                      inputTensor.options());

    if (sparseCoalesceMode != ccl_sparse_coalesce_disable)
        resultTensor._coalesced_(true);

    if (sparseResultMode == SparseResultMode::COPY)
    {
        /* propagate result using 2 ways - inputTensors and outputTensors */
        for (size_t i = 0; i < inputTensors.size(); i++)
        {
            inputTensors[i].copy_(resultTensor);
            if (resultTensor.is_sparse())
            {
                outputTensors.push_back(resultTensor.clone());
            }
            else
            {
                outputTensors.push_back(resultTensor.clone(at::MemoryFormat::Contiguous));
            }
        }
    }
    else if (sparseResultMode == SparseResultMode::OOP)
    {
        /* propagate result using 1 way - outputTensors */
        TORCH_CHECK(resultTensor.layout() == c10::kSparse, "unexpected tensor layout");
        outputTensors.push_back(resultTensor);
    }
    else
    {
        TORCH_CHECK(0, "unexpected sparseResultMode ", sparseResultMode);
    }

    return ccl_status_success;
}

ccl_status_t sparseAllreduceAllocFn(
    size_t indCount, ccl_datatype_t indDatatype,
    size_t valCount, ccl_datatype_t valDatatype,
    const void* fnCtx, void** outIndBuf, void** outValBuf)
{
    TORCH_CHECK(fnCtx, "fnCtx");
    TORCH_CHECK(outIndBuf, "outIndBuf");
    TORCH_CHECK(outValBuf, "outValBuf");

    TORCH_CHECK(sparseResultMode == SparseResultMode::DIRECT,
                "unexpected sparseResultMode ", sparseResultMode);

    SparseAllreduceWork* work = (SparseAllreduceWork*)(fnCtx);

    std::vector<at::Tensor>& inputTensors = work->getTensors();
    std::vector<at::Tensor>& outputTensors = work->getOutputTensors();

    TORCH_CHECK(inputTensors.size() == 1, "unexpected inputTensors size");
    TORCH_CHECK(outputTensors.size() == 0, "unexpected outputTensors size");

    outputTensors.reserve(inputTensors.size());

    at::Tensor& inputTensor = inputTensors[0];

    TORCH_CHECK(inputTensor.sparse_dim() == 1, "unexpected sparse_dim");

    const auto valueShape = inputTensor.sizes().slice(inputTensor.sparse_dim());
    auto resultValueShape = std::vector<int64_t>({(int64_t)indCount});
    std::copy(valueShape.begin(), valueShape.end(), std::back_inserter(resultValueShape));

    auto indices = at::empty({1, (long int)indCount}, inputTensor._indices().options());
    auto values = at::empty(resultValueShape,
                            inputTensor._values().options());

    auto resultTensor =
        at::_sparse_coo_tensor_unsafe(indices,
                                      values,
                                      inputTensor.sizes(),
                                      inputTensor.options());

    if (sparseCoalesceMode != ccl_sparse_coalesce_disable)
        resultTensor._coalesced_(true);

    /* propagate result using 1 way - outputTensors */
    TORCH_CHECK(resultTensor.layout() == c10::kSparse, "unexpected tensor layout");
    outputTensors.push_back(resultTensor);

    *outIndBuf = resultTensor._indices().data_ptr();
    *outValBuf = resultTensor._values().data_ptr();

    TORCH_CHECK(*outIndBuf, "result outIndBuf");
    TORCH_CHECK(*outValBuf, "result outValBuf");

    return ccl_status_success;
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts)
{
    const auto& layout = tensors[0].layout();
    std::string funcName = (layout == c10::kStrided) ? "pg::allreduce" : "pg::sparse_allreduce";
    RECORD_FUNCTION(funcName, std::vector<c10::IValue>({tensors[0]}));

    checkSingleTensor(tensors);

    std::shared_ptr<ccl::request> req;
    std::shared_ptr<ProcessGroupCCL::WorkCCL> work;

    {
        std::unique_lock<std::mutex> globalLock(globalMutex);

        if (layout == c10::kStrided)
        {
            std::string debugName = std::string("allreduce::sz:") + std::to_string(tensors[0].numel());
            work = std::make_shared<ProcessGroupCCL::WorkCCL>(tensors, std::move(debugName));
            CCL_CHECK(req = comm->allreduce(tensors[0].data_ptr(),
                                            tensors[0].data_ptr(),
                                            (size_t)tensors[0].numel(),
                                            cclDatatypes.at(tensors[0].scalar_type()),
                                            cclOps.at(opts.reduceOp)));
        }
        else if (layout == c10::kSparse)
        {
            at::Tensor input = tensors[0];
            TORCH_CHECK(input.sparse_dim() == 1, "allreduce: only single sparse_dim is supported");

            auto indices = input._indices();
            auto values = input._values();

            std::string debugName =
                std::string("sparse_allreduce::ind_sz:") + std::to_string(indices.numel()) +
                std::string(":val_sz:") + std::to_string(values.numel());

            work = std::make_shared<SparseAllreduceWork>(tensors, std::move(debugName));

            ccl::coll_attr attr{};

            if (sparseResultMode == SparseResultMode::DIRECT)
                attr.sparse_allreduce_alloc_fn = sparseAllreduceAllocFn;
            else
                attr.sparse_allreduce_completion_fn = sparseAllreduceCompletionFn;
            attr.sparse_allreduce_fn_ctx = work.get();
            attr.sparse_coalesce_mode = sparseCoalesceMode;

            CCL_CHECK(req = comm->sparse_allreduce(indices.data_ptr(),
                                                   (size_t)indices.numel(),
                                                   values.data_ptr(),
                                                   (size_t)values.numel(),
                                                   nullptr, 0, nullptr, 0,
                                                   cclDatatypes.at(indices.scalar_type()),
                                                   cclDatatypes.at(values.scalar_type()),
                                                   cclOps.at(opts.reduceOp),
                                                   &attr));
        }
    }
    work->setRequest(req);
    return work;
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::allreduce_coalesced(
    std::vector<at::Tensor>& /* unused */,
    const AllreduceCoalescedOptions& /* unused */)
{
    TORCH_CHECK(false, "ProcessGroupCCL does not support allreduce_coalesced");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts)
{
    RECORD_FUNCTION("pg::reduce", std::vector<c10::IValue>({tensors[0]}));

    checkSingleTensor(tensors);
    checkRank(opts.rootRank, getSize());

    std::shared_ptr<ccl::request> req;

    {
        std::unique_lock<std::mutex> globalLock(globalMutex);
        CCL_CHECK(req = comm->reduce(tensors[0].data_ptr(),
                                     tensors[0].data_ptr(),
                                     (size_t)tensors[0].numel(),
                                     cclDatatypes.at(tensors[0].scalar_type()),
                                     cclOps.at(opts.reduceOp),
                                     (size_t)opts.rootRank));
    }

    std::string debugName = std::string("reduce::sz:") + std::to_string(tensors[0].numel());

    return std::make_shared<ProcessGroupCCL::WorkCCL>(req, tensors, std::move(debugName));
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts)
{
    RECORD_FUNCTION("pg::allgather", std::vector<c10::IValue>({inputTensors[0]}));

    checkSingleTensor(inputTensors);

    TORCH_CHECK(outputTensors.size() == 1,
        "allgather: multi-GPU collective is not supported");

    TORCH_CHECK(comm->size() == outputTensors[0].size(),
        "allgather: number of output tensors should equal to the world size");

    checkSameType(inputTensors[0], outputTensors[0]);

    void* recvBuf = nullptr;
    std::vector<void*> recvBufs;
    std::vector<size_t> recvCounts(size_, 0);

    auto flatRes = computeLengthsAndCheckFlat(outputTensors[0], recvCounts);

    TORCH_CHECK((size_t)inputTensors[0].numel() == recvCounts[rank_],
        "allgather: send and recv count doesn't match");

    if (flatRes.isFlat)
    {
        recvBuf = flatRes.firstTensor.data_ptr();
    }
    else
    {
        std::transform(outputTensors[0].begin(), outputTensors[0].end(),
                       std::back_inserter(recvBufs),
                       [](const at::Tensor& t) { return t.data_ptr(); } );
        recvBuf = recvBufs.data();
    }

    std::shared_ptr<ccl::request> req;

    {
        std::unique_lock<std::mutex> globalLock(globalMutex);
        collAttrAg.vector_buf = flatRes.isFlat ? 0 : 1;
        CCL_CHECK(req = comm->allgatherv(inputTensors[0].data_ptr(),
                                         (size_t)inputTensors[0].numel(),
                                         recvBuf,
                                         (size_t*)recvCounts.data(),
                                         cclDatatypes.at(inputTensors[0].scalar_type()),
                                         &collAttrAg));
    }

    std::vector<at::Tensor> agTensors;

    if (flatRes.isFlat)
        agTensors.emplace_back(flatRes.firstTensor);
    else
        agTensors.assign(outputTensors[0].begin(), outputTensors[0].end());
    agTensors.emplace_back(inputTensors[0]);

    std::string debugName = std::string("allgather::sz:") + std::to_string(inputTensors[0].numel());

    return std::make_shared<ProcessGroupCCL::WorkCCL>(req, std::move(agTensors), std::move(debugName));
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::allgather_base(
      at::Tensor& outputBuffer,
      at::Tensor& inputBuffer,
      const AllgatherOptions& /* unused */)
{
    TORCH_CHECK(false, "ProcessGroupCCL does not support allgather_base");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::allgather_coalesced(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const AllgatherOptions& /* unused */)
{
    TORCH_CHECK(false, "ProcessGroupCCL does not support allgather_coalesced");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::gather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const GatherOptions& opts)
{
    RECORD_FUNCTION("pg::gather", std::vector<c10::IValue>({inputTensors[0]}));

    checkSingleTensor(inputTensors);

    if (rank_ != opts.rootRank)
    {
        TORCH_CHECK(outputTensors.size() == 0,
            "gather: number of output tensors should be 0 "
            "for non-root");
    }
    else
    {
        TORCH_CHECK(outputTensors.size() == 1,
            "gather: multi-GPU collective is not supported");

        TORCH_CHECK(static_cast<size_t>(size_) == outputTensors[0].size(),
            "gather: number of output tensors should equal "
            "to the world size");

        checkSameType(inputTensors[0], outputTensors[0]);
    }

    std::vector<size_t> sendCounts(size_, 0);
    std::vector<size_t> recvCounts(size_, 0);
    sendCounts[opts.rootRank] = inputTensors[0].numel();

    at::Tensor flatOutput;
    int64_t flatRecvCount = 0;
    bool isOutputFlat = false;

    if (rank_ == opts.rootRank)
    {
        isOutputFlat =
            computeLengthsAndCheckAndGetFlat(outputTensors[0],
                                             recvCounts, flatOutput, flatRecvCount);
        TORCH_CHECK(sendCounts[rank_] == recvCounts[rank_],
            "gather: send and recv count doesn't match");
    }
    else
    {
        flatOutput = at::empty({0}, inputTensors[0].options());
    }


    std::shared_ptr<ccl::request> req;

    {
        std::unique_lock<std::mutex> globalLock(globalMutex);
        CCL_CHECK(req = comm->alltoallv(inputTensors[0].data_ptr(),
                                        sendCounts.data(),
                                        flatOutput.data_ptr(),
                                        recvCounts.data(),
                                        cclDatatypes.at(flatOutput.scalar_type())));
    }

    std::vector<at::Tensor> gatherTensors;

    if (rank_ == opts.rootRank)
    {
        if (!isOutputFlat)
        {
            req->wait();

            auto flatOutputSplits =
                flatOutput.split_with_sizes(c10::IntArrayRef((int64_t*)recvCounts.data(),
                                            recvCounts.size()), 0);

            for (int i = 0; i < size_; i++)
            {
                outputTensors[0][i].view({-1}).copy_(flatOutputSplits[i]);
            }
        }
        else
        {
            gatherTensors.emplace_back(flatOutput);
            gatherTensors.emplace_back(inputTensors[0]);
        }
    }
    else
    {
        gatherTensors.emplace_back(inputTensors[0]);
    }

    std::string debugName = std::string("gather::sz:") + std::to_string(inputTensors[0].numel());

    return std::make_shared<ProcessGroupCCL::WorkCCL>(req, std::move(gatherTensors), std::move(debugName));
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::scatter(
    std::vector<at::Tensor>& outputTensors,
    std::vector<std::vector<at::Tensor>>& inputTensors,
    const ScatterOptions& opts)
{
    RECORD_FUNCTION("pg::scatter", std::vector<c10::IValue>({outputTensors}));

    checkSingleTensor(outputTensors);

    if (rank_ != opts.rootRank)
    {
        TORCH_CHECK(inputTensors.size() == 0,
            "scatter: number of input tensors should be 0 "
            "for non-root");
    }
    else
    {
        TORCH_CHECK(inputTensors.size() == 1,
            "scatter: multi-GPU collective is not supported");

        TORCH_CHECK(static_cast<size_t>(size_) == inputTensors[0].size(),
            "scatter: number of input tensors should equal "
            "to the world size");

        checkSameType(outputTensors[0], inputTensors[0]);
    }

    std::vector<size_t> sendCounts(size_, 0);
    std::vector<size_t> recvCounts(size_, 0);
    recvCounts[opts.rootRank] = outputTensors[0].numel();

    at::Tensor flatInput;
    int64_t flatSendCount = 0;

    if (rank_ == opts.rootRank)
    {
        bool isInputFlat =
            computeLengthsAndCheckAndGetFlat(inputTensors[0],
                                             sendCounts, flatInput, flatSendCount);

        if (!isInputFlat)
        {
            auto flatInputSplits =
                flatInput.split_with_sizes(c10::IntArrayRef((int64_t*)sendCounts.data(),
                                           sendCounts.size()), 0);

            for (int i = 0; i < size_; i++)
            {
                flatInputSplits[i].copy_(inputTensors[0][i].view({-1}));
            }
        }
        TORCH_CHECK(recvCounts[rank_] == sendCounts[rank_],
            "scatter: send and recv count doesn't match");
    }
    else
    {
        flatInput = at::empty({0}, outputTensors[0].options());
    }

    std::shared_ptr<ccl::request> req;

    {
        std::unique_lock<std::mutex> globalLock(globalMutex);
        CCL_CHECK(req = comm->alltoallv(flatInput.data_ptr(),
                                        sendCounts.data(),
                                        outputTensors[0].data_ptr(),
                                        recvCounts.data(),
                                        cclDatatypes.at(flatInput.scalar_type())));
    }

    std::vector<at::Tensor> scatterTensors;
    scatterTensors.emplace_back(outputTensors[0]);
    if (rank_ == opts.rootRank)
        scatterTensors.emplace_back(flatInput);

    std::string debugName = std::string("scatter::sz:") + std::to_string(outputTensors[0].numel());

    return std::make_shared<ProcessGroupCCL::WorkCCL>(req, std::move(scatterTensors), std::move(debugName));
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::reduce_scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ReduceScatterOptions& /* unused */)
{
    TORCH_CHECK(false, "ProcessGroupCCL does not support reduce_scatter");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::alltoall_base(
    at::Tensor& outputTensor,
    at::Tensor& inputTensor,
    std::vector<int64_t>& outputSplitSizes,
    std::vector<int64_t>& inputSplitSizes,
    const AllToAllOptions& opts)
{
    RECORD_FUNCTION("pg::alltoall_base", std::vector<c10::IValue>({inputTensor, outputTensor}));

    checkSingleTensorHelper(inputTensor);
    checkSingleTensorHelper(outputTensor);

    std::shared_ptr<ccl::request> req;

    if (outputSplitSizes.size() == 0 && inputSplitSizes.size() == 0)
    {
        // We can use alltoall
        TORCH_CHECK(outputTensor.numel() == inputTensor.numel() &&
                    outputTensor.scalar_type() == inputTensor.scalar_type(),
                    "alltoall_base: tensors are not equal in size or data type");

        TORCH_CHECK(outputTensor.size(0) % size_ == 0,
            "alltoall_base: tensor's dim 0 does not divide equally across group size");

        {
            std::unique_lock<std::mutex> globalLock(globalMutex);
            CCL_CHECK(req = comm->alltoall(inputTensor.data_ptr(),
                                           outputTensor.data_ptr(),
                                           (size_t)outputTensor.numel() / comm->size(),
                                           cclDatatypes.at(outputTensor.scalar_type())));
        }
    }
    else
    {
        // Need alltoallv
        checkSplitSizes(inputSplitSizes, inputTensor, size_);
        checkSplitSizes(outputSplitSizes, outputTensor, size_);

        std::vector<size_t> sendCounts(size_);
        std::vector<size_t> recvCounts(size_);

        // inLen or outLen can be 0 so we need explicit flag
        bool inputSplitsEqual = inputSplitSizes.size() == 0;
        bool outputSplitsEqual = outputSplitSizes.size() == 0;

        size_t inLen = inputTensor.numel();
        size_t outLen = outputTensor.numel();
        if (inLen) inLen /= (inputSplitsEqual ? size_ : inputTensor.size(0));
        if (outLen) outLen /= (outputSplitsEqual ? size_ : outputTensor.size(0));

        for (int i = 0; i < size_; i++)
        {
            sendCounts[i] = (inputSplitsEqual ? inLen : inputSplitSizes[i] * inLen);
            recvCounts[i] = (outputSplitsEqual ? outLen : outputSplitSizes[i] * outLen);
        }

        {
            std::unique_lock<std::mutex> globalLock(globalMutex);
            CCL_CHECK(req = comm->alltoallv(inputTensor.data_ptr(),
                                            sendCounts.data(),
                                            outputTensor.data_ptr(),
                                            recvCounts.data(),
                                            cclDatatypes.at(outputTensor.scalar_type())));
        }
    }

    auto a2aTensors = std::vector<at::Tensor> { inputTensor, outputTensor };
    std::string debugName = std::string("alltoall_base::sz:") +
        std::to_string((inputTensor.numel() + outputTensor.numel()) / (2 * size_));

    return std::make_shared<ProcessGroupCCL::WorkCCL>(req, std::move(a2aTensors), std::move(debugName));
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllToAllOptions& opts)
{
    RECORD_FUNCTION("pg::alltoall", std::vector<c10::IValue>());

    TORCH_CHECK(inputTensors.size() == (size_t)size_,
        "alltoall: number of input tensors are not equal to group size");

    TORCH_CHECK(outputTensors.size() == (size_t)size_,
        "alltoall: number of output tensors are not equal to group size");

    checkSameType(outputTensors[0], inputTensors);
    checkSameType(inputTensors[0], outputTensors);

    std::vector<size_t> sendCounts(size_);
    std::vector<size_t> recvCounts(size_);

    at::Tensor flatInput;
    at::Tensor flatOutput;

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

        for (int i = 0; i < size_; i++)
        {
            flatInputSplits[i].copy_(inputTensors[i].view({-1}));
        }
    }

    std::shared_ptr<ccl::request> req;

    {
        std::unique_lock<std::mutex> globalLock(globalMutex);
        CCL_CHECK(req = comm->alltoallv(flatInput.data_ptr(),
                                        sendCounts.data(),
                                        flatOutput.data_ptr(),
                                        recvCounts.data(),
                                        cclDatatypes.at(flatOutput.scalar_type())));
    }

    std::vector<at::Tensor> a2aTensors;

    if (!isOutputFlat)
    {
        req->wait();

        auto flatOutputSplits =
            flatOutput.split_with_sizes(c10::IntArrayRef((int64_t*)recvCounts.data(),
                                        recvCounts.size()), 0);

        for (int i = 0; i < size_; i++)
        {
            outputTensors[i].view({-1}).copy_(flatOutputSplits[i]);
        }
    }
    else
    {
        a2aTensors.emplace_back(flatOutput);
        a2aTensors.emplace_back(flatInput);
    }

    std::string debugName = std::string("alltoall::sz:") +
        std::to_string((flatSendCount + flatRecvCount) / (2 * size_));

    return std::make_shared<ProcessGroupCCL::WorkCCL>(req, std::move(a2aTensors), std::move(debugName));
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::send(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */,
    int /* unused */)
{
    TORCH_CHECK(false, "ProcessGroupCCL does not support send");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::recv(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */,
    int /* unused */)
{
    TORCH_CHECK(false, "ProcessGroupCCL does not support recv");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::recvAnysource(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */)
{
    TORCH_CHECK(false, "ProcessGroupCCL does not support recvAnysource");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::barrier(
    const BarrierOptions& opts)
{
    RECORD_FUNCTION("pg::barrier", std::vector<c10::IValue>());

    std::unique_lock<std::mutex> globalLock(globalMutex);
    CCL_CHECK(comm->barrier());

    return std::make_shared<ProcessGroupCCL::WorkCCL>();
}

#ifndef PROCESS_GROUP_CCL_TEST
PYBIND11_MODULE(torch_ccl, m)
{
    m.def("createProcessGroupCCL", &ProcessGroupCCL::createProcessGroupCCL);
}
#endif

} // namespace c10d
