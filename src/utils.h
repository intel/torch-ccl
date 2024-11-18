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

#include <unistd.h>
#include <thread>
#include <ATen/detail/FunctionTraits.h>
#include <ATen/record_function.h>

#include <torch/version.h>
#if TORCH_VERSION_MAJOR > 1 || TORCH_VERSION_MINOR >= 13
#include <torch/csrc/distributed/c10d/Types.hpp>
#else
#include <c10d/Types.hpp>
#endif
#include <torch/csrc/distributed/c10d/Utils.hpp>

#include <ccl_comm_collector.h>
#include "ProcessGroupCCL.hpp"


constexpr uint64_t kSynchronizeBusyWaitMicro = 10; // 50us

#define CCL_CHECK(cmd)                                               \
  do {                                                               \
    try {                                                            \
        cmd;                                                         \
    }                                                                \
    catch (ccl::exception& e) {                                      \
      e.what();                                                      \
      throw e;                                                       \
    }                                                                \
  }while(0)


#define CCL_DISPATCH_INTEGRAL_FLOATS_TYPES(TYPE, NAME, ...)                          \
  [&] {                                                                      \
    const auto& the_type = TYPE;                                             \
    /* don't use TYPE again in case it is an expensive or side-effect op */  \
    at::ScalarType _st = ::detail::scalar_type(the_type);                    \
    switch (_st) {                                                           \
      /*AT_PRIVATE_CASE_TYPE(at::ScalarType::Char, char, __VA_ARGS__)  */    \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Int, int, __VA_ARGS__)      \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Long, int64_t, __VA_ARGS__) \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Float, float, __VA_ARGS__)  \
      AT_PRIVATE_CASE_TYPE(NAME, at::ScalarType::Double, double, __VA_ARGS__) \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

namespace oneccl_bindings_for_pytorch {

using c10d::ProcessGroupCCL;

extern std::map<c10d::ReduceOp, ccl::reduction> cclOps;
extern std::map<at::ScalarType, ccl::datatype> cclDatatypes;

// Get the deviceList String from the list of devices
std::string get_key_from_devs(const std::vector<at::Device>& devices);

std::string get_key_send_recv(int myRank, int peer);

// Get the list of devices from list of tensors
std::vector<at::Device> get_device_list(const std::vector<at::Tensor>& tensors);
std::vector<at::Device> get_device_list(const std::vector<std::vector<at::Tensor>>& tensors);
bool check_same_size(const std::vector<at::Tensor>& tensors);

inline at::Tensor newLikeFlat(std::vector<at::Tensor>& tensors) {
  if (tensors.empty()) {
    TORCH_CHECK(false, "Received an empty list");
  }
  auto& t = tensors[0];
  at::DeviceGuard deviceGuard(t.device());
  std::vector<int64_t> sizes{static_cast<int64_t>(tensors.size())};
  sizes.insert(sizes.end(), t.sizes().begin(), t.sizes().end());
  return at::empty(sizes, t.options());
}

template <typename ccl_fn_type>
decltype(auto) call_with_lock(std::mutex& lock, ccl_fn_type fn) {
  std::unique_lock<std::mutex> globalLock(lock);
  return fn();
}

class AsyncBarrierWork: public ProcessGroupCCL::AsyncWorkCCL {
public:
  AsyncBarrierWork():AsyncWorkCCL({}){}

   ~AsyncBarrierWork()
  {
    if (!events.empty()) {
      std::cerr << "attempted destruction of WorkCCL before work has completed, "
                << "terminating the program."
                << std::endl;
      std::terminate();
    }
  }

  std::vector<ccl::event>& getEvents(){
    return events;
  }

  bool isCompleted() override
  {
    for(auto& event : events) {
      bool flag;

      try {
        call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&]() {
            CCL_CHECK(flag = event.test());
        });
      } catch (...) {
        finishAsyncWorkCCLError(std::current_exception());
        return true;
      }

      if (!flag) {
        return false;
      }
    }
    return true;
  }

   bool isSuccess() const override
  {
    throw std::runtime_error("invalid call to ::isSuccess.");
  }

  bool wait(std::chrono::milliseconds timeout) override
  {
    for(auto& event : events) {
      call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&]() {
          CCL_CHECK(event.wait());
      });
    }
    events.clear();
    return true;
  }

  void abort() override
  {
    TORCH_CHECK(false, "ProcessGroupCCL::WorkCCL::abort not implemented");
  }
  
  void run() override{
    TORCH_CHECK(false, "AsyncBarrierWork::run not implemented");
  }

private:
  std::vector<ccl::event> events;

};

c10::intrusive_ptr<c10::ivalue::Future> createFutureAsOutput(
        const std::vector<std::vector<at::Tensor>>& outputTensors);

c10::intrusive_ptr<c10::ivalue::Future> createFutureAsOutput(
        const std::vector<at::Tensor>& outputTensors);

template <typename> struct is_tuple: std::false_type {};

template <typename ...T> struct is_tuple<std::tuple<T...>>: std::true_type {};

template <typename> struct is_vector: std::false_type {};

template <typename T> struct is_vector<std::vector<T>>: std::true_type {};

template <typename RunF, typename CommType, typename InputType, typename OutputType, typename attr_t>
class CollectiveAsyncWorkCCL : public ProcessGroupCCL::AsyncWorkCCL {
public:
  using traits = function_traits<RunF>;
  static constexpr int num_params = traits::arity;
  using ret_t = typename traits::result_type;

  template<typename T = OutputType>
  CollectiveAsyncWorkCCL(const std::vector<InputType>& inputs,
                   const std::vector<OutputType>& outputs,
                   const RunF f,
                   CommType& comms,
                   attr_t& attr,
                   std::chrono::milliseconds timeout,
                   int rank,
                   c10d::OpType opType,
                   const char* profilingTitle,
                   const c10::optional<std::vector<at::Tensor>>& inputTensors,
                   typename std::enable_if<is_vector<T>::value, int>::type* = 0) :
                   AsyncWorkCCL(outputs, rank, opType, profilingTitle, inputTensors),
                   f(f), comms(comms), attr(attr), inputs(inputs), opTimeout_(timeout) {}

  template<typename T = OutputType>
  CollectiveAsyncWorkCCL(const std::vector<InputType>& inputs,
                         const std::vector<OutputType>& outputs,
                         const RunF f,
                         CommType& comms,
                         attr_t& attr,
                         std::chrono::milliseconds timeout,
                         int rank,
                         c10d::OpType opType,
                         const char* profilingTitle,
                         const c10::optional<std::vector<at::Tensor>>& inputTensors,
                         typename std::enable_if<!is_vector<T>::value, int>::type* = 0) :
                         AsyncWorkCCL({outputs}, rank, opType, profilingTitle, inputTensors),
                         f(f), comms(comms), attr(attr), inputs(inputs), opTimeout_(timeout) {}

  void run() override {
    if constexpr (num_params == 6) {
        workStartTime_ = std::chrono::steady_clock::now();
        run_wrap_();
    }
    else{
        using Indices = std::make_index_sequence<num_params - 4>;
        workStartTime_ = std::chrono::steady_clock::now();
        run_wrap_(Indices{});
    }
  };

  virtual ~CollectiveAsyncWorkCCL()
  {
#if 0
    if (!rets.empty()) {
      std::cerr << "attempted destruction of WorkCCL before work has completed, "
                << "waiting the request."
                << std::endl;
      synchronize();
    }
#endif
  }

  bool isCompleted() override {
    for(auto& ret : rets) {
      bool flag;
      ccl::event& req = get_event_from_ret_<ret_t>(ret);

      try {
        call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&]() {
            req.wait();
            flag = true;
        });
      } catch (...) {
        finishAsyncWorkCCLError(std::current_exception());
        return true;
      }

      if (!flag) {
        return false;
      }
    }
    return true;
  }

  bool timedOut(std::chrono::milliseconds timeout) const {
    auto currentTimepoint = std::chrono::steady_clock::now();
    return (
            std::chrono::duration_cast<std::chrono::milliseconds>(
                    currentTimepoint - workStartTime_) >= timeout);
  }

  void checkAndThrowException() {
    // Throw an exception, only if we have a valid exception.
    if (exception()) {
      std::rethrow_exception(exception());
    }
  }

  void synchronizeInternal(std::chrono::milliseconds timeout) {
    // Wait for the operation to complete.
    std::chrono::milliseconds workTimeout =
            timeout == kNoTimeout ? this->opTimeout_ : timeout;
    while (!isCompleted()) {
      if (timedOut(workTimeout)) {

        auto currentTimepoint = std::chrono::steady_clock::now();
        auto timeElapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                        currentTimepoint - workStartTime_);
        std::string exceptionMsg = c10::str(
                "[Rank ",
                rank_,
                "] ",
                "Caught collective operation timeout: ",
                " ran for ",
                timeElapsed.count(),
                " milliseconds before timing out.");
        TORCH_CHECK(false, exceptionMsg);
      }
      std::this_thread::sleep_for(
              std::chrono::microseconds (kSynchronizeBusyWaitMicro));
    }
  }

  void synchronize() override {
    synchronizeInternal(kNoTimeout);
  }

protected:
  std::vector<ccl::event>& get_ccl_event()
  {
    return cclEvents_;
  }

  template <typename T = OutputType, std::size_t...INDEX>
  typename std::enable_if<is_vector<T>::value, void>::type run_wrap_(std::index_sequence<INDEX...>) {
    if (rets.empty()) {
      auto& outputs = outputTensors_;
      for (size_t i = 0; i < inputs.size(); i++) {
        CCL_CHECK(rets.push_back(f(inputs[i], outputs[i], attr, comms.comms[i], comms.streams[i + INDEX]...)));
      }
    }
    else {
      // add warning for re run the ccl work
    }
  }

  template <typename T = OutputType, std::size_t...INDEX>
  typename std::enable_if<!is_vector<T>::value, void>::type run_wrap_(std::index_sequence<INDEX...>) {
    if (rets.empty()) {
      auto& outputs = outputTensors_[0];
      // Since torch-ccl only supports one device per process now, the size of comms.comms must be 0.
      if (comms.comms.size() !=1) {
        throw std::runtime_error("Torch CCL only support one device per process now");
      }

      // Some primitives have empty input(scatter), so we get the size after checking size of input and output.
      auto size = inputs.empty()?outputs.size():inputs.size();
      for (size_t i = 0; i < size; i++) {
        CCL_CHECK(rets.push_back(f(inputs[i], outputs[i], attr, comms.comms[0], comms.streams[0 + INDEX]...)));
      }
    }
    else {
      // add warning for re run the ccl work
    }
  }

    template <typename T = OutputType>
    typename std::enable_if<is_vector<T>::value, void>::type run_wrap_() {
    if (rets.empty()) {
      auto& outputs = outputTensors_;
      for (size_t i = 0; i < inputs.size(); i++) {
        CCL_CHECK(rets.push_back(f(inputs[i], outputs[i], attr, comms.comms[i], comms.streams[i], comms.torch_streams[i])));
      }
    }
    else {
      // add warning for re run the ccl work
    }
  }

  template <typename T = OutputType>
  typename std::enable_if<!is_vector<T>::value, void>::type run_wrap_() {
    if (rets.empty()) {
      auto& outputs = outputTensors_[0];
      // Some primitives have empty input(scatter), so we get the size after checking size of input and output.
      auto size = inputs.empty()?outputs.size():inputs.size();
      for (size_t i = 0; i < size; i++) {
        CCL_CHECK(rets.push_back(f(inputs[i], outputs[i], attr, comms.comms[i], comms.streams[i], comms.torch_streams[i])));
      }
    }
    else {
      // add warning for re run the ccl work
    }
  }


  template <typename R, std::enable_if_t<is_tuple<R>::value, bool> = true>
  ccl::event& get_event_from_ret_(R& ret)
  {
      return std::get<0>(ret);
  }

  template <typename R, std::enable_if_t<std::is_same<R, ccl::event>::value, bool> = true>
  ccl::event& get_event_from_ret_(R& ret)
  {
    return ret;
  }

  RunF f;
  CommType& comms;
  attr_t attr;
  // Keep the reference to the tensor.
  std::vector<InputType> inputs;
  std::chrono::milliseconds opTimeout_;
  // Keep the reference to the returned value. E.G: the callback functor.
  std::vector<ret_t> rets;
  std::vector<ccl::event> cclEvents_;
  std::chrono::time_point<std::chrono::steady_clock> workStartTime_;
};

template <typename RunF, typename CommType, typename InputType, typename OutputType, typename attr_t>
class P2PAsyncWork : public ProcessGroupCCL::AsyncWorkCCL {

public:
  using traits = function_traits<RunF>;
  static constexpr int num_params = traits::arity;
  using ret_t = typename traits::result_type;

  template<typename T = OutputType>
  P2PAsyncWork(const std::vector<InputType>& inputs,
                   const std::vector<OutputType>& outputs,
                   int peer,
                   const RunF f,
                   CommType& comms,
                   attr_t& attr,
                   std::chrono::milliseconds timeout,
                   int rank,
                   c10d::OpType opType,
                   const char* profilingTitle,
                   const c10::optional<std::vector<at::Tensor>>& inputTensors,
                   typename std::enable_if<is_vector<T>::value, int>::type* = 0) :
                   AsyncWorkCCL(outputs, rank, opType, profilingTitle, inputTensors),
                   f(f), comms(comms), attr(attr), inputs(inputs), peer(peer), opTimeout_(timeout) {}

  template<typename T = OutputType>
  P2PAsyncWork(const std::vector<InputType>& inputs,
                         const std::vector<OutputType>& outputs,
                         int peer,
                         const RunF f,
                         CommType& comms,
                         attr_t& attr,
                         std::chrono::milliseconds timeout,
                         int rank,
                         c10d::OpType opType,
                         const char* profilingTitle,
                         const c10::optional<std::vector<at::Tensor>>& inputTensors,
                         typename std::enable_if<!is_vector<T>::value, int>::type* = 0) :
                         AsyncWorkCCL({outputs}, rank, opType, profilingTitle, inputTensors),
                         f(f), comms(comms), attr(attr), inputs(inputs), peer(peer), opTimeout_(timeout) {}


  void run() override {
    if constexpr (num_params == 6) {
        workStartTime_ = std::chrono::steady_clock::now();
        run_wrap_();
    }
    else{
        using Indices = std::make_index_sequence<num_params - 4>;
        workStartTime_ = std::chrono::steady_clock::now();
        run_wrap_(Indices{});
    }
  };

  virtual ~P2PAsyncWork()
  {
#if 0
    if (!rets.empty()) {
      std::cerr << "attempted destruction of WorkCCL before work has completed, "
                << "waiting the request."
                << std::endl;
      synchronize();
    }
#endif
  }

  bool isCompleted() override {
    for(auto& ret : rets) {
      bool flag;
      ccl::event& req = get_event_from_ret_<ret_t>(ret);

      try {
        call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&]() {
            req.wait();
            flag = true;
        });
      } catch (...) {
        finishAsyncWorkCCLError(std::current_exception());
        return true;
      }

      if (!flag) {
        return false;
      }
    }
    return true;
  }

  bool timedOut(std::chrono::milliseconds timeout) const {
    auto currentTimepoint = std::chrono::steady_clock::now();
    return (
            std::chrono::duration_cast<std::chrono::milliseconds>(
                    currentTimepoint - workStartTime_) >= timeout);
  }

  void checkAndThrowException() {
    // Throw an exception, only if we have a valid exception.
    if (exception()) {
      std::rethrow_exception(exception());
    }
  }

  void synchronizeInternal(std::chrono::milliseconds timeout) {
    // Wait for the operation to complete.
    std::chrono::milliseconds workTimeout =
            timeout == kNoTimeout ? this->opTimeout_ : timeout;
    while (!isCompleted()) {
      if (timedOut(workTimeout)) {

        auto currentTimepoint = std::chrono::steady_clock::now();
        auto timeElapsed =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                        currentTimepoint - workStartTime_);
        std::string exceptionMsg = c10::str(
                "[Rank ",
                rank_,
                "] ",
                "Caught collective operation timeout: ",
                " ran for ",
                timeElapsed.count(),
                " milliseconds before timing out.");
        TORCH_CHECK(false, exceptionMsg);
      }
      std::this_thread::sleep_for(
              std::chrono::microseconds (kSynchronizeBusyWaitMicro));
    }
  }

  void synchronize() override {
    synchronizeInternal(kNoTimeout);
  }

protected:
  std::vector<ccl::event>& get_ccl_event()
  {
    return cclEvents_;
  }

  template <typename T = OutputType, std::size_t...INDEX>
  typename std::enable_if<is_vector<T>::value, void>::type run_wrap_(std::index_sequence<INDEX...>) {
    if (rets.empty()) {
      for (size_t i = 0; i < inputs.size(); i++) {
        CCL_CHECK(rets.push_back(f(inputs[i], peer, attr, comms.comms[i], comms.streams[i + INDEX]...)));
      }
    }
    else {
      // add warning for re run the ccl work
    }
  }

  template <typename T = OutputType, std::size_t...INDEX>
  typename std::enable_if<!is_vector<T>::value, void>::type run_wrap_(std::index_sequence<INDEX...>) {
    if (rets.empty()) {
      for (size_t i = 0; i < inputs.size(); i++) {
        CCL_CHECK(rets.push_back(f(inputs[i], peer, attr, comms.comms[i], comms.streams[i + INDEX]...)));
      }
    }
    else {
      // add warning for re run the ccl work
    }
  }

    template <typename T = OutputType>
    typename std::enable_if<is_vector<T>::value, void>::type run_wrap_() {
    if (rets.empty()) {
      for (size_t i = 0; i < inputs.size(); i++) {
        CCL_CHECK(rets.push_back(f(inputs[i], peer, attr, comms.comms[i], comms.streams[i], comms.torch_streams[i])));
      }
    }
    else {
      // add warning for re run the ccl work
    }
  }

  template <typename T = OutputType>
  typename std::enable_if<!is_vector<T>::value, void>::type run_wrap_() {
    if (rets.empty()) {
      for (size_t i = 0; i < inputs.size(); i++) {
        CCL_CHECK(rets.push_back(f(inputs[i], peer, attr, comms.comms[i], comms.streams[i], comms.torch_streams[i])));
      }
    }
    else {
      // add warning for re run the ccl work
    }
  }


  template <typename R, std::enable_if_t<is_tuple<R>::value, bool> = true>
  ccl::event& get_event_from_ret_(R& ret)
  {
      return std::get<0>(ret);
  }

  template <typename R, std::enable_if_t<std::is_same<R, ccl::event>::value, bool> = true>
  ccl::event& get_event_from_ret_(R& ret)
  {
    return ret;
  }

  RunF f;
  CommType& comms;
  attr_t attr;
  // Keep the reference to the tensor.
  std::vector<InputType> inputs;
  int peer;
  std::chrono::milliseconds opTimeout_;
  // Keep the reference to the returned value. E.G: the callback functor.
  std::vector<ret_t> rets;
  std::vector<ccl::event> cclEvents_;
  std::chrono::time_point<std::chrono::steady_clock> workStartTime_;
};

template <template<typename, typename, typename, typename, typename> class WorkCCL,
          typename RunF, typename CommType, typename InputType, typename OutputType, typename attr_t>
c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> make_work_ccl(const std::vector<InputType>& inputs,
                                                                const std::vector<OutputType>& outputs,
                                                                RunF f,
                                                                CommType& comms,
                                                                attr_t& attr,
                                                                std::chrono::milliseconds timeout,
                                                                int rank,
                                                                c10d::OpType op_type,
                                                                const char* prof_title) {

  c10::intrusive_ptr<WorkCCL<RunF, CommType, InputType, OutputType, attr_t>> ret_ptr =
      c10::make_intrusive<WorkCCL<RunF, CommType, InputType, OutputType, attr_t>>(inputs, outputs, f, comms, attr, timeout,
              rank, op_type, prof_title, c10::nullopt);
  return ret_ptr;
}

template <template<typename, typename, typename, typename, typename> class WorkP2P,
          typename RunF, typename CommType, typename InputType, typename OutputType, typename attr_t>
c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> make_work_p2p(const std::vector<InputType>& inputs,
                                                                const std::vector<OutputType>& outputs,
                                                                int peer,
                                                                RunF f,
                                                                CommType& comms,
                                                                attr_t& attr,
                                                                std::chrono::milliseconds timeout,
                                                                int rank,
                                                                c10d::OpType op_type,
                                                                const char* prof_title) {

  c10::intrusive_ptr<WorkP2P<RunF, CommType, InputType, OutputType, attr_t>> ret_ptr =
      c10::make_intrusive<WorkP2P<RunF, CommType, InputType, OutputType, attr_t>>(inputs, outputs, peer, f, comms, attr, timeout,
              rank, op_type, prof_title, c10::nullopt);
  return ret_ptr;
}

template <Comms& (*get_ccl_fn)(c10d::ProcessGroupCCL& pg_ccl, const std::string& devices_key, const std::vector<at::Device>& devices, c10d::OpType op_type, int p2pRank, bool isSendRecvSelf),
        template<typename, typename, typename, typename, typename> class WorkCCL,
        typename fn, typename pre_process, typename post_process, typename input_t, typename output_t>
c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> collective(
  ProcessGroupCCL& pg_ccl,
  std::vector<input_t>& inputs,
  std::vector<output_t>& outputs,
  fn fun,
  pre_process pre,
  post_process post,
  c10d::OpType op_type,
  const char* prof_title = nullptr) {
  using traits = function_traits<fn>;
  using attr_t = typename traits::template arg<2>::type;
  attr_t attr = ccl::create_operation_attr<attr_t>();

  std::vector<at::Device> devices;
  if (inputs.empty() && outputs.empty()) {
    // Coalesced op has empty tensors for both input and output. 
    // Use 'coalescedDevices_' as devices, which should have same set of devices across collectives
    devices = {pg_ccl.coalescedDevices_[0]};
  } else {
    if (!inputs.empty()) {
      devices = get_device_list(inputs);
    } else {
      devices = get_device_list(outputs);
    }
  }

  const auto key = get_key_from_devs(devices);
  auto& comms = get_ccl_fn(pg_ccl, key, devices, c10d::OpType::UNKNOWN, 0, false);

  if (pg_ccl.is_coalescing_) {
    pg_ccl.coalescedDevices_.push_back(devices[0]);
  }

  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  work = make_work_ccl<WorkCCL>(inputs, outputs, fun, comms, attr, pg_ccl.timeout, pg_ccl.getRank(), op_type, prof_title);

  // Set appropriate work parameters.
  work->blockingWait_ = pg_ccl.blockingWait_;
  work->useSameStream_ = pg_ccl.useSameStream_;
  return work;
}

template <Comms& (*get_ccl_fn)(c10d::ProcessGroupCCL& pg_ccl, const std::string& devices_key, const std::vector<at::Device>& devices, c10d::OpType op_type, int p2pRank, bool isSendRecvSelf),
        template<typename, typename, typename, typename, typename> class WorkP2P,
        typename fn, typename pre_process, typename post_process, typename input_t, typename output_t>
c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> pointToPoint(
  ProcessGroupCCL& pg_ccl,
  std::vector<input_t>& inputs,
  std::vector<output_t>& outputs,
  fn fun,
  int peer,
  c10d::OpType op_type,
  pre_process pre,
  post_process post,
  const char* prof_title = nullptr) {

  using traits = function_traits<fn>;
  using attr_t = typename traits::template arg<2>::type;
  attr_t attr = ccl::create_operation_attr<attr_t>();
  const auto devices = get_device_list(inputs);
  std::string key;
  int p2pRank = 0, p2pTargetRank = 0;
  bool isSendRecvSelf = false;

  int rank_ = pg_ccl.getRank();
  bool batchP2P = pg_ccl.cclActiveGroupCounter_ > 0;

  if (batchP2P) {
    // Because sub-group processes is not supported now, we treat p2p like collective and put into batchp2p
    // For batch P2P, we need to treat it like a collective when selecting
    // communicator, because other ranks can call into this batch other than my
    // rank and my peer
    key = get_key_from_devs(devices);
    p2pRank = rank_;
    p2pTargetRank = peer;
  } else {
    // TODO: single P2P
    // For single P2P, preserve the old two-rank behavior (to avoid perf diff)
    key = get_key_send_recv(rank_, peer);
    p2pRank = rank_ <= peer ? 0 : 1;
    isSendRecvSelf = rank_ == peer;
    p2pTargetRank = isSendRecvSelf ? 0 : 1 - p2pRank;
  }

  auto& comms = get_ccl_fn(pg_ccl, key, devices, op_type, p2pRank, isSendRecvSelf);
  if (pg_ccl.is_coalescing_) {
    pg_ccl.coalescedDevices_.push_back(devices[0]);
  }
  c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> work;

  work = make_work_p2p<WorkP2P>(inputs, outputs, p2pTargetRank, fun, comms, attr, pg_ccl.timeout, pg_ccl.getRank(), op_type, prof_title);

  return work;
}

template <Comms& (*get_ccl_fn)(c10d::ProcessGroupCCL& pg_ccl, const std::string& devices_key, const std::vector<at::Device>& devices, c10d::OpType op_type, int p2pRank, bool isSendRecvSelf),
        template<typename, typename, typename, typename, typename> class WorkCCL, typename fn, typename input_t, typename output_t>
c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> collective(
  ProcessGroupCCL& pg_ccl,
  std::vector<input_t>& inputs,
  std::vector<output_t>& outputs,
  fn fun,
  c10d::OpType op_type,
  const char* prof_title = nullptr) {
  return collective<get_ccl_fn, WorkCCL>(
    pg_ccl,
    inputs,
    outputs,
    fun,
    [](std::vector<ccl::stream>&) {},
    [](std::vector<ccl::stream>&) {},
    op_type,
    prof_title);
}


template <Comms& (*get_ccl_fn)(c10d::ProcessGroupCCL& pg_ccl, const std::string& devices_key, const std::vector<at::Device>& devices, c10d::OpType op_type, int p2pRank, bool isSendRecvSelf),
        template<typename, typename, typename, typename, typename> class WorkP2P, typename fn, typename input_t, typename output_t>
c10::intrusive_ptr<ProcessGroupCCL::AsyncWorkCCL> pointToPoint(
    ProcessGroupCCL& pg_ccl,
    std::vector<input_t>& input,
    std::vector<output_t>& output,
    fn fun,
    int peer,
    c10d::OpType op_type,
    const char* prof_title = nullptr) {

  return pointToPoint<get_ccl_fn, WorkP2P>(
      pg_ccl,
      input,
      output,
      fun,
      peer,
      op_type,
      [](std::vector<ccl::stream>&) {},
      [](std::vector<ccl::stream>&) {},
      prof_title);
}

typedef struct
{
  bool isFlat;
  int64_t size;
  at::Tensor firstTensor;
} FlatCheckResult;

FlatCheckResult computeLengthsAndCheckFlat(
        const std::vector<at::Tensor>& tensors,
        std::vector<size_t>& lengths);

bool computeLengthsAndCheckAndGetFlat(
        const std::vector<at::Tensor>& tensors,
        std::vector<size_t>& lengths,
        at::Tensor& flatTensor,
        int64_t& flatLength);

void checkSingleTensorHelper(const at::Tensor& tensor);

void checkSingleTensor(const std::vector<at::Tensor>& tensors);

void checkSameType(const at::Tensor& tensor, const std::vector<at::Tensor>& tensors);

void checkSameType(const at::Tensor& tensor,
                   const std::vector<std::vector<at::Tensor>>& tensors);

}
