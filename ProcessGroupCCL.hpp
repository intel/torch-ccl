#pragma once

#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include <c10d/ProcessGroup.hpp>
#include <c10d/Store.hpp>
#include <c10d/Types.hpp>
#include <c10d/Utils.hpp>
#include <ccl.hpp>
#include <torch/extension.h>

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

/* TODO: remove this after alltoall upstream into public PyTorch */
struct AllToAllOptions
{
    std::chrono::milliseconds timeout = kUnsetTimeout;
};

class ProcessGroupCCL : public ProcessGroup,
                        public std::enable_shared_from_this<ProcessGroupCCL>
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
      void wait() override;

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

  std::shared_ptr<ProcessGroup::Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

  // Unsupported Ops
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

  std::shared_ptr<ProcessGroup::Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) /* override */;

  std::shared_ptr<ProcessGroup::Work> send(
      std::vector<at::Tensor>& tensors,
      int dstRank,
      int tag);

  std::shared_ptr<ProcessGroup::Work> recv(
      std::vector<at::Tensor>& tensors,
      int srcRank,
      int tag);

  std::shared_ptr<ProcessGroup::Work> recvAnysource(
      std::vector<at::Tensor>& tensor,
      int tag);

  // create a new ProcessGroupCCL and initialize CCL if not initialized
  static std::shared_ptr<ProcessGroup> createProcessGroupCCL(
      const std::shared_ptr<Store>& store,
      int rank,
      int size,
      const std::string& groupName = "");

 protected:

  static void cclInitOnce();
  static void cclFini();

  ccl::communicator* comm;
};

} // namespace c10d
