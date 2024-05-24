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

#include "init.h"
#include <torch/extension.h>

#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#include <chrono>
#include <pybind11/chrono.h>
#include <pybind11/cast.h>

#include <torch/version.h>
#if TORCH_VERSION_MAJOR > 1 || TORCH_VERSION_MINOR >= 13
#if TORCH_VERSION_MAJOR > 1
#include <torch/csrc/distributed/c10d/Backend.hpp>
#else
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#endif
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#else
#include <c10d/ProcessGroup.hpp>
#include <c10d/Store.hpp>
#include <c10d/Types.hpp>
#include <c10d/Utils.hpp>
#endif

#include <ProcessGroupCCL.hpp>

namespace py = pybind11;


namespace {

// This is a intrusive helper from pytorch.
template <typename T>
class IntrusivePtrNoGilDestructor {
  c10::intrusive_ptr<T> impl_;

public:
  IntrusivePtrNoGilDestructor() = default;
  IntrusivePtrNoGilDestructor(const IntrusivePtrNoGilDestructor&) = default;
  IntrusivePtrNoGilDestructor(IntrusivePtrNoGilDestructor&&) = default;
  IntrusivePtrNoGilDestructor& operator=(const IntrusivePtrNoGilDestructor&) =
  default;
  IntrusivePtrNoGilDestructor& operator=(IntrusivePtrNoGilDestructor&&) =
  default;
  /* implicit */ IntrusivePtrNoGilDestructor(c10::intrusive_ptr<T> impl)
          : impl_(std::move(impl)) {}
  // This ctor is very important; see
  // https://github.com/pybind/pybind11/issues/2957
  explicit IntrusivePtrNoGilDestructor(T* impl)
          : impl_(c10::intrusive_ptr<T>::unsafe_steal_from_new(impl)) {}
  ~IntrusivePtrNoGilDestructor() {
    if (impl_) {
      if (PyGILState_Check()) {
        pybind11::gil_scoped_release release;
        impl_.reset();
      } else {
        impl_.reset();
      }
    }
  }
  T& operator*() const noexcept {
    return *impl_;
  }
  T* operator->() const noexcept {
    return impl_.get();
  }
  C10_NODISCARD T* get() const noexcept {
    return impl_.get();
  }
  void reset() noexcept {
    impl_.reset();
  }
  operator bool() const noexcept {
    return impl_;
  }
};

} // anonymous namespace

PYBIND11_DECLARE_HOLDER_TYPE(T, IntrusivePtrNoGilDestructor<T>, true);

template <typename T>
using intrusive_ptr_no_gil_destructor_class_ =
py::class_<T, IntrusivePtrNoGilDestructor<T>>;

TORCH_CCL_CPP_API void torch_ccl_python_init(pybind11::module &m) {
  c10d::ProcessGroupCCL::cclInitOnce();
  py::object module = py::module::import("torch.distributed");
  py::object register_backend = module.attr("Backend").attr("register_backend");
  #if TORCH_VERSION_MAJOR > 1 
  auto backend = py::module::import("torch._C._distributed_c10d").attr("Backend");
  #else 
  auto backend = module.attr("ProcessGroup");
  #endif
  register_backend("ccl", py::cpp_function(&c10d::ProcessGroupCCL::createProcessGroupCCL,
                                           py::arg("store"),
                                           py::arg("rank"),
                                           py::arg("size"),
                                           py::arg("timeout") = std::chrono::milliseconds(
                                                   ::c10d::ProcessGroupCCL::OP_TIMEOUT_MILLIS)),
		                           false, std::vector<std::string>{"xpu", "cpu"});
  
  auto processGroupCCL = intrusive_ptr_no_gil_destructor_class_<::c10d::ProcessGroupCCL>(
          module, "ProcessGroupCCL", backend);

  processGroupCCL.def(
    py::init([](const c10::intrusive_ptr<::c10d::Store>& store,
                int rank,
                int size,
                std::chrono::milliseconds timeout) {
      return c10::make_intrusive<::c10d::ProcessGroupCCL>(store, rank, size, timeout);
    }),
    py::arg("store"),
    py::arg("rank"),
    py::arg("size"),
    py::arg("timeout") = std::chrono::milliseconds(10 * 1000));

}
