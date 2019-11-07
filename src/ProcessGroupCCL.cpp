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
        std::string err = "CCL error in: " + std::string(__FILE__) + \
            ":" + std::to_string(__LINE__) +                         \
            ", with error message: " + e.what();                     \
        throw std::runtime_error(err);                               \
    }                                                                \
    catch (...) {                                                    \
        std::string err = "unknown error in: " +                     \
            std::string(__FILE__) + ":" + std::to_string(__LINE__);  \
        throw std::runtime_error(err);                               \
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

} // namespace c10d

ProcessGroupCCL::WorkCCL::~WorkCCL()
{
    if (req.get())
    {
        std::cerr << "attempted destruction of WorkCCL before work has completed, "
                  << "terminating the program."
                  << std::endl;
        std::terminate();
    }
}

bool ProcessGroupCCL::WorkCCL::isCompleted()
{
    if (!req.get())
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
    if (req.get())
    {
        throw std::runtime_error(
            "invalid call to WorkCCL::isSuccess before work has completed");
    }
    return true;
}

void ProcessGroupCCL::WorkCCL::wait()
{
    if (!req.get())
    {
        return;
    }

    std::unique_lock<std::mutex> globalLock(globalMutex);
    CCL_CHECK(req->wait());
    req.reset();
    tensors.clear();
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

#ifdef USE_VECTOR_ALLGATHERV
      /* to enable efficient allgatherv with recv buffers vector */
      setenv("CCL_ALLGATHERV_IOV", "1", 1);
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
    const std::string& groupName)
{
  cclInitOnce();

  if (rank != -1 && (rank < 0 || (size_t)rank != globalComm->rank()))
  {
      throw std::runtime_error("unexpected rank " + std::to_string(rank) +
                               ", CCL rank " + std::to_string(globalComm->rank()));
  }

  if (size != -1 && (size <= 0 || (size_t)size != globalComm->size()))
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

    std::shared_ptr<ccl::request> req;

    std::unique_lock<std::mutex> globalLock(globalMutex);
    CCL_CHECK(req = comm->bcast(tensors[0].data_ptr(),
                                (size_t)tensors[0].numel(),
                                cclDatatypes.at(tensors[0].type().scalarType()),
                                (size_t)opts.rootRank,
                                &collAttr));

    return std::make_shared<ProcessGroupCCL::WorkCCL>(req, tensors);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts)
{
    checkSingleTensor(tensors);

    std::shared_ptr<ccl::request> req;

    std::unique_lock<std::mutex> globalLock(globalMutex);
    CCL_CHECK(req = comm->allreduce(tensors[0].data_ptr(),
                                    tensors[0].data_ptr(),
                                    (size_t)tensors[0].numel(),
                                    cclDatatypes.at(tensors[0].type().scalarType()),
                                    cclOps.at(opts.reduceOp),
                                    &collAttr));

    return std::make_shared<ProcessGroupCCL::WorkCCL>(req, tensors);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts)
{
    throw std::runtime_error("ProcessGroupCCL does not support allreduce_coalesced");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts)
{
    checkSingleTensor(tensors);
    checkRank(opts.rootRank, getSize());

    std::shared_ptr<ccl::request> req;

    std::unique_lock<std::mutex> globalLock(globalMutex);
    CCL_CHECK(req = comm->reduce(tensors[0].data_ptr(),
                                 tensors[0].data_ptr(),
                                 (size_t)tensors[0].numel(),
                                 cclDatatypes.at(tensors[0].type().scalarType()),
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
                                     cclDatatypes.at(inputTensors[0].type().scalarType()),
                                     &collAttr));

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

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::barrier(
    const BarrierOptions& opts)
{
    std::unique_lock<std::mutex> globalLock(globalMutex);
    CCL_CHECK(comm->barrier());
    return std::make_shared<ProcessGroupCCL::WorkCCL>();
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

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllToAllOptions& opts)
{
    checkSingleTensor(outputTensors);
    checkSingleTensor(inputTensors);
    //checkSameSizeAndType(inputTensors[0], outputTensors[0]);

    std::shared_ptr<ccl::request> req;

    std::unique_lock<std::mutex> globalLock(globalMutex);
    CCL_CHECK(req = comm->alltoall(inputTensors[0].data_ptr(),
                                   outputTensors[0].data_ptr(),
                                   (size_t)outputTensors[0].numel() / comm->size(),
                                   cclDatatypes.at(outputTensors[0].type().scalarType()),
                                   &collAttr));

    auto a2aTensors = std::vector<at::Tensor> { inputTensors[0], outputTensors[0] };
    return std::make_shared<ProcessGroupCCL::WorkCCL>(req, std::move(a2aTensors));
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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("ProcessGroupCCL", &ProcessGroupCCL::createProcessGroupCCL, "ProcessGroupCCL");
}

} // namespace c10d
