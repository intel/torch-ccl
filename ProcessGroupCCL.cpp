#include "ProcessGroupCCL.hpp"

#include <map>

namespace c10d {

#define USE_VECTOR_ALLGATHERV

#define CCL_CHECK(cmd)                                                   \
  do {                                                                   \
    try {                                                                \
        cmd;                                                             \
    }                                                                    \
    catch (ccl::ccl_error& e) {                                          \
        std::string err = "CCL error in: " + std::string(__FILE__) +     \
            ":" + std::to_string(__LINE__) +                             \
            ", with error message: " + e.what();                         \
        fprintf(stderr, "\n%s\n", err.c_str());                          \
        throw std::runtime_error(err);                                   \
    }                                                                    \
    catch (...) {                                                        \
        std::string err = "unknown error in: " + std::string(__FILE__) + \
            ":" + std::to_string(__LINE__);                              \
        fprintf(stderr, "\n%s\n", err.c_str());                          \
        throw std::runtime_error(err);                                   \
    }                                                                    \
  } while (0)

namespace {

// Op mapping
std::map<ReduceOp, ccl::reduction> cclOp = {
    {ReduceOp::MIN, ccl::reduction::min},
    {ReduceOp::MAX, ccl::reduction::max},
    {ReduceOp::SUM, ccl::reduction::sum},
    {ReduceOp::PRODUCT, ccl::reduction::prod},
};

// Type mapping
std::map<at::ScalarType, ccl::data_type> cclDatatype = {
    {at::kByte, ccl::data_type::dt_char},
    {at::kChar, ccl::data_type::dt_char},
    {at::kDouble, ccl::data_type::dt_double},
    {at::kBFloat16, ccl::data_type::dt_bfp16},
    {at::kFloat, ccl::data_type::dt_float},
    {at::kInt, ccl::data_type::dt_int},
    {at::kLong, ccl::data_type::dt_int64}
};

// Checking the input tensor's validity
void checkSingleTensorHelper(const at::Tensor& tensor) {
  if (!tensor.is_contiguous()) {
      throw std::runtime_error("input tensor has to be contiguous");
  }
  if (tensor.is_sparse()) {
      throw std::runtime_error("input tensor has to be dense");
  }
  if (tensor.is_cuda()) {
      throw std::runtime_error(
          "CUDA tensor detected and CCL doesn't support CUDA buffers");
  }
  if (tensor.numel() < 0) {
      throw std::runtime_error("input tensor numel should be non-negative");
  }
}

void checkRank(int rank, int size) {
  if (rank < 0 || rank >= size) {
      throw std::runtime_error("unexpected rank");
  }
}

void checkSingleTensor(const std::vector<at::Tensor>& tensors) {
  if (tensors.size() != 1) {
    throw std::runtime_error(
        "CCL process group does not support multi-GPU collectives");
  }
  checkSingleTensorHelper(tensors[0]);
}

void checkSameSizeAndType(
    const at::Tensor& tensor,
    const std::vector<at::Tensor>& tensors) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    if ((tensors[i].numel() != tensor.numel()) ||
        (tensors[i].type() != tensor.type())) {
      throw std::runtime_error("Tensors are not equal in size or data type");
    }
    checkSingleTensorHelper(tensors[i]);
  }
}

} // namespace

ProcessGroupCCL::WorkCCL::~WorkCCL() {
  if (request_.get()) {
    std::cerr
        << "Attempted destruction of WorkCCL before work has completed, "
        << "terminating the program." << std::endl;
    std::terminate();
  }
}

bool ProcessGroupCCL::WorkCCL::isCompleted() {
  if (!request_.get()) {
    return true;
  }

  bool flag = false;

  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  CCL_CHECK(flag = request_->test());

  if (flag) {
    request_.reset();
    tensors_.clear();
  }

  return flag;
}

bool ProcessGroupCCL::WorkCCL::isSuccess() const {
  if (request_.get()) {
    throw std::runtime_error(
        "Invalid call to WorkCCL::isSuccess before work has completed");
  }
  return true;
}

void ProcessGroupCCL::WorkCCL::wait() {
  if (!request_.get()) {
    return;
  }

  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  CCL_CHECK(request_->wait());
  request_.reset();
  tensors_.clear();
}

std::mutex ProcessGroupCCL::pgGlobalMutex_;
std::once_flag ProcessGroupCCL::onceFlagInitCCL;
ccl::environment* ProcessGroupCCL::cclEnv_ = nullptr;
ccl::communicator_t ProcessGroupCCL::cclGlobalComm_;
ccl::coll_attr ProcessGroupCCL::collAttr_;

void ProcessGroupCCL::cclExit() {
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  cclGlobalComm_.reset();
}

void ProcessGroupCCL::initCCLOnce() {
  // Initialize CCL environment
  std::call_once(onceFlagInitCCL, []() {

#ifdef USE_VECTOR_ALLGATHERV
    /* to enable efficient allgatherv with recv buffers vector */
    setenv("CCL_VECTOR_ALLGATHERV", "1", 1);
#endif

    CCL_CHECK(cclEnv_ = &ccl::environment::instance());
    CCL_CHECK(cclGlobalComm_ = cclEnv_->create_communicator());
    /* TODO: user buffers can differ from call to call so disabling caching for now
     To enable caching we need to have additional context from user like tensor_name */
    collAttr_.to_cache = 0;
    if (std::atexit(ProcessGroupCCL::cclExit)) {
      throw std::runtime_error("Fail to register the CCL exit handler");
    }
  });
}

std::shared_ptr<ProcessGroup>
ProcessGroupCCL::createProcessGroupCCL(const std::shared_ptr<Store>& store,
                                       int rank,
                                       int size,
                                       const std::string& groupName) {
  // Once initialization
  initCCLOnce();

  /* TODO:
     clarify what scenarious we need to support in CCL with custom process groups
     currently only cclGlobalComm_ is used
  */
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);

  if (rank != -1 && (rank < 0 || (size_t)rank != ProcessGroupCCL::cclGlobalComm_->rank()))
  {
      printf("rank %d, ccl_rank %zu\n", rank, ProcessGroupCCL::cclGlobalComm_->rank());
      throw std::runtime_error("unexpected rank");
  }

  if (size != -1 && (size <= 0 || (size_t)size != ProcessGroupCCL::cclGlobalComm_->size()))
  {
      printf("size %d, ccl_size %zu\n", size, ProcessGroupCCL::cclGlobalComm_->size());
      throw std::runtime_error("unexpected size");
  }

  globalLock.unlock();

  return std::make_shared<ProcessGroupCCL>(rank, size);
}

ProcessGroupCCL::ProcessGroupCCL(int rank, int size)
    : ProcessGroup(ProcessGroupCCL::cclGlobalComm_->rank(),
                   ProcessGroupCCL::cclGlobalComm_->size()),
      pgComm_(ProcessGroupCCL::cclGlobalComm_.get()) {}

ProcessGroupCCL::~ProcessGroupCCL() {}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::broadcast(
    std::vector<at::Tensor>& tensors,
    const BroadcastOptions& opts) {
  checkSingleTensor(tensors);
  checkRank(opts.rootRank, getSize());

  std::shared_ptr<ccl::request> req;

  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  CCL_CHECK(req = pgComm_->bcast(
    tensors[0].data_ptr(),
    (size_t)tensors[0].numel(),
    cclDatatype.at(tensors[0].type().scalarType()),
    (size_t)opts.rootRank,
    &collAttr_));

  return std::make_shared<ProcessGroupCCL::WorkCCL>(req, tensors);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::allreduce(
    std::vector<at::Tensor>& tensors,
    const AllreduceOptions& opts) {
  checkSingleTensor(tensors);

  std::shared_ptr<ccl::request> req;

  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  CCL_CHECK(req = pgComm_->allreduce(
    tensors[0].data_ptr(),
    tensors[0].data_ptr(),
    (size_t)tensors[0].numel(),
    cclDatatype.at(tensors[0].type().scalarType()),
    cclOp.at(opts.reduceOp),
    &collAttr_));

  return std::make_shared<ProcessGroupCCL::WorkCCL>(req, tensors);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::allreduce_coalesced(
    std::vector<at::Tensor>& tensors,
    const AllreduceCoalescedOptions& opts) {
  throw std::runtime_error(
      "allreduce_coalesced is currently not supported with CCL");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::reduce(
    std::vector<at::Tensor>& tensors,
    const ReduceOptions& opts) {
  checkSingleTensor(tensors);
  checkRank(opts.rootRank, getSize());

  std::shared_ptr<ccl::request> req;

  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  CCL_CHECK(req = pgComm_->reduce(
    tensors[0].data_ptr(),
    tensors[0].data_ptr(),
    (size_t)tensors[0].numel(),
    cclDatatype.at(tensors[0].type().scalarType()),
    cclOp.at(opts.reduceOp),
    (size_t)opts.rootRank,
    &collAttr_));

  return std::make_shared<ProcessGroupCCL::WorkCCL>(req, tensors);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::allgather(
    std::vector<std::vector<at::Tensor>>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllgatherOptions& opts) {
  checkSingleTensor(inputTensors);

  if (outputTensors.size() != 1) {
    throw std::runtime_error(
        "CCL process group only supports a single "
        "tensor op");
  }

  if (static_cast<size_t>(size_) != outputTensors[0].size()) {
    throw std::runtime_error(
        "allgather: number of output tensors should equal "
        "to the world size");
  }

  checkSameSizeAndType(inputTensors[0], outputTensors[0]);

  std::shared_ptr<ccl::request> req;

#ifdef USE_VECTOR_ALLGATHERV
  std::vector<void*> recvBuffers;
  for (size_t idx = 0; idx < outputTensors[0].size(); idx++)
    recvBuffers.push_back(outputTensors[0][idx].data_ptr());
#else
  auto flatOutputTensor = newLikeFlat(outputTensors[0]);
#endif
  std::vector<size_t> recvCounts(size_, inputTensors[0].numel());

  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  CCL_CHECK(req = pgComm_->allgatherv(
    inputTensors[0].data_ptr(),
    (size_t)inputTensors[0].numel(),
#ifdef USE_VECTOR_ALLGATHERV
    recvBuffers.data(),
#else
    flatOutputTensor.data_ptr(),
#endif
    (size_t*)recvCounts.data(),
    cclDatatype.at(inputTensors[0].type().scalarType()),
    &collAttr_));

#ifndef USE_VECTOR_ALLGATHERV
  req->wait();
  for (size_t i = 0; i < outputTensors[0].size(); ++i) {
    outputTensors[0][i].copy_(flatOutputTensor[i]);
  }
#endif

  auto ag_tensors = std::vector<at::Tensor>(outputTensors[0].begin(), outputTensors[0].end());
  ag_tensors.emplace_back(inputTensors[0]);
  return std::make_shared<ProcessGroupCCL::WorkCCL>(req, ag_tensors);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::barrier(
    const BarrierOptions& opts) {
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  CCL_CHECK(pgComm_->barrier());
  return std::make_shared<ProcessGroupCCL::WorkCCL>();
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::gather(
    std::vector<std::vector<at::Tensor>>& /* unused */,
    std::vector<at::Tensor>& /* unused */,
    const GatherOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupCCL does not support gather");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ScatterOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupCCL does not support scatter");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::reduce_scatter(
    std::vector<at::Tensor>& /* unused */,
    std::vector<std::vector<at::Tensor>>& /* unused */,
    const ReduceScatterOptions& /* unused */) {
  throw std::runtime_error("ProcessGroupCCL does not support reduce_scatter");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::alltoall(
    std::vector<at::Tensor>& outputTensors,
    std::vector<at::Tensor>& inputTensors,
    const AllToAllOptions& opts) {
  checkSingleTensor(outputTensors);
  checkSingleTensor(inputTensors);
  //checkSameSizeAndType(inputTensors[0], outputTensors[0]);

  std::shared_ptr<ccl::request> req;
  std::unique_lock<std::mutex> globalLock(pgGlobalMutex_);
  CCL_CHECK(req = pgComm_->alltoall(
    inputTensors[0].data_ptr(),
    outputTensors[0].data_ptr(),
    (size_t)outputTensors[0].numel() / size_,
    cclDatatype.at(outputTensors[0].type().scalarType()),
    &collAttr_));

  auto a2a_tensors = std::vector<at::Tensor> { inputTensors[0], outputTensors[0] };
  return std::make_shared<ProcessGroupCCL::WorkCCL>(req, a2a_tensors);
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::send(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */,
    int /* unused */) {
  throw std::runtime_error("ProcessGroupCCL does not support send");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::recv(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */,
    int /* unused */) {
  throw std::runtime_error("ProcessGroupCCL does not support recv");
}

std::shared_ptr<ProcessGroup::Work> ProcessGroupCCL::recvAnysource(
    std::vector<at::Tensor>& /* unused */,
    int /* unused */) {
  throw std::runtime_error("ProcessGroupCCL does not support recvAnysource");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ProcessGroupCCL", &ProcessGroupCCL::createProcessGroupCCL, "ProcessGroupCCL");
}

} // namespace c10d
