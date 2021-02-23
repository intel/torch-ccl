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

#include "ProcessGroupCCL.hpp"
#include <ATen/detail/FunctionTraits.h>
#include <ATen/record_function.h>
#include <c10d/Types.hpp>

#define CCL_CHECK(cmd)                                               \
  do {                                                               \
    try {                                                            \
        cmd;                                                         \
    }                                                                \
    catch (std::runtime_error& e) {                                  \
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
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Long, int64_t, __VA_ARGS__)       \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)        \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Double, double, __VA_ARGS__)      \
      default:                                                               \
        AT_ERROR(#NAME, " not implemented for '", toString(_st), "'");       \
    }                                                                        \
  }()

namespace torch_ccl {

using c10d::ProcessGroupCCL;

extern std::map<c10d::ReduceOp, ccl::reduction> cclOps;

// Get the deviceList String from the list of devices
std::string get_key_from_devs(const std::vector<at::Device>& devices);

// Get the list of devices from list of tensors
std::vector<at::Device> get_device_list(const std::vector<at::Tensor>& tensors);
std::vector<at::Device> get_device_list(const std::vector<std::vector<at::Tensor>>& tensors);

template <typename ccl_fn_type>
decltype(auto) call_with_lock(std::mutex& lock, ccl_fn_type fn) {
  std::unique_lock<std::mutex> globalLock(lock);
  return fn();
}

class AsyncBarrierWork: public ProcessGroupCCL::AsyncWorkCCL {
public:
  AsyncBarrierWork(){}

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

      call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&]() {
        CCL_CHECK(flag = event.test());
      });

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

template <typename> struct is_tuple: std::false_type {};

template <typename ...T> struct is_tuple<std::tuple<T...>>: std::true_type {};

template <typename RunF, typename CommType, typename InputType, typename OutputType, typename attr_t>
class AsyncWorkCCLWrap : public ProcessGroupCCL::AsyncWorkCCL {
public:
  using traits = function_traits<RunF>;
  static constexpr int num_params = traits::arity;
  using ret_t = typename traits::result_type;

  AsyncWorkCCLWrap(const std::vector<InputType>& inputs,
                   const std::vector<OutputType>& outputs,
                   const RunF f,
                   CommType& comms,
                   attr_t& attr) : AsyncWorkCCL(), f(f), comms(comms), attr(attr), inputs(inputs), outputs(outputs) {}

  void run() override {
    using Indices = std::make_index_sequence<num_params - 4>;
      run_wrap_(Indices{});
  };

  ~AsyncWorkCCLWrap()
  {
    if (!rets.empty()) {
      std::cerr << "attempted destruction of WorkCCL before work has completed, "
                << "terminating the program."
                << std::endl;
      std::terminate();
    }
  }

  bool isCompleted() override
  {
    for(auto& ret : rets) {
      bool flag;
      ccl::event& req = _get_event_from_ret<ret_t>(ret);
      call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&]() {
          CCL_CHECK(flag = req.test());
      });
      if (!flag) {
        return false;
      }
    }
    // all request has been finished
    return true;
  }

  bool isSuccess() const override
  {
    throw std::runtime_error("invalid call to ::isSuccess.");
  }

  bool wait(std::chrono::milliseconds timeout) override
  {
    RECORD_FUNCTION(std::string("torch_ccl::wait::") + debugName, std::vector<c10::IValue>());
    for(auto& ret : rets) {
      ccl::event& evt = _get_event_from_ret<ret_t>(ret);
      call_with_lock(c10d::ProcessGroupCCL::globalMutex, [&]() {
          CCL_CHECK(evt.wait());
      });
    }
    rets.clear();
    // Always return true, because abort API is not implemented.
    return true;
  }

  void abort() override
  {
    TORCH_CHECK(false, "ProcessGroupCCL::WorkCCL::abort not implemented");
  }

  std::vector<at::Tensor> result() override
  {
    return result_wrap_<OutputType>();
  }

private:

  template <std::size_t...INDEX>
  void run_wrap_(std::index_sequence<INDEX...>) {
    if (rets.empty()) {
      for (size_t i = 0; i < inputs.size(); i++) {
        CCL_CHECK(rets.push_back(f(inputs[i], outputs[i], attr, comms.comms[i], comms.streams[i + INDEX]...)));
      }
    }
    else {
      // add warning for re run the ccl work
    }
  }

  template<typename T, std::enable_if_t<!std::is_same<T, at::Tensor>::value, bool> = true>
  std::vector<at::Tensor> result_wrap_()
  {
    AT_ERROR("NOT implemented for the non std::vector<at::Tensor> return");
  }

  template<typename T, std::enable_if_t<std::is_same<T, at::Tensor>::value, bool> = true>
  std::vector<at::Tensor> result_wrap_()
  {
    TORCH_CHECK(outputs.size() == 1, "unexpected result size");
    return outputs;
  }

  template <typename R, std::enable_if_t<is_tuple<R>::value, bool> = true>
  ccl::event& _get_event_from_ret(R& ret)
  {
      return std::get<0>(ret);
  }

  template <typename R, std::enable_if_t<std::is_same<R, ccl::event>::value, bool> = true>
  ccl::event& _get_event_from_ret(R& ret)
  {
    return ret;
  }

  RunF f;
  CommType& comms;
  attr_t attr;
  /*
      keep copy of tensors to increment tensor reference counters
      while CCL operation is in progress
  */
  std::vector<InputType> inputs;
  std::vector<OutputType> outputs;
  std::vector<ret_t> rets;
};

template <typename RunF, typename CommType, typename InputType, typename OutputType, typename attr_t>
std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> make_work_ccl(const std::vector<InputType>& inputs,
                                                             const std::vector<OutputType>& outputs,
                                                             RunF f,
                                                             CommType& comms,
                                                             attr_t& attr) {
  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> ret_ptr;
  ret_ptr.reset(new AsyncWorkCCLWrap<RunF, CommType, InputType, OutputType, attr_t>(inputs, outputs, f, comms, attr));
  return ret_ptr;
}

template <Comms& (*get_ccl_fn)(c10d::ProcessGroupCCL& pg_ccl, const std::string& devices_key, const std::vector<at::Device>& devices),
          typename fn, typename pre_process, typename post_process, typename input_t, typename output_t>
std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> collective(
  ProcessGroupCCL& pg_ccl,
  std::vector<input_t>& inputs,
  std::vector<output_t>& outputs,
  fn fun,
  pre_process pre,
  post_process post) {
  using traits = function_traits<fn>;
  using attr_t = typename traits::template arg<2>::type;
  attr_t attr = ccl::create_operation_attr<attr_t>();

  const auto devices = get_device_list(inputs);
  const auto key = get_key_from_devs(devices);
  auto& comms = get_ccl_fn(pg_ccl, key, devices);

  std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> work;
  work = make_work_ccl(inputs, outputs, fun, comms, attr);

  return work;
}

template <Comms& (*get_ccl_fn)(c10d::ProcessGroupCCL& pg_ccl, const std::string& devices_key, const std::vector<at::Device>& devices),
          typename fn, typename input_t, typename output_t>
std::shared_ptr<ProcessGroupCCL::AsyncWorkCCL> collective(
  ProcessGroupCCL& pg_ccl,
  std::vector<input_t>& inputs,
  std::vector<output_t>& outputs,
  fn fun) {
  return collective<get_ccl_fn>(
    pg_ccl,
    inputs,
    outputs,
    fun,
    [](std::vector<ccl::stream>&) {},
    [](std::vector<ccl::stream>&) {});
}

}
