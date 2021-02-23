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

#include <init.h>
#include <pybind11/chrono.h>
#include <oneapi/ccl/config.h>
#include "ProcessGroupCCL.hpp"

namespace py = pybind11;

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

TORCH_CCL_CPP_API void torch_ccl_python_init(pybind11::module &m) {

  c10d::ProcessGroupCCL::cclInitOnce();

  m.def("oneCCL_spec_version", []() {
    std::string oneCCL_spec_ver;
    std::ostringstream os(oneCCL_spec_ver);
    os << ONECCL_SPEC_VERSION;
    return oneCCL_spec_ver;
  });

  m.def("oneCCL_version", []() {
    std::string oneCCL_ver;
    std::ostringstream os(oneCCL_ver);
    os << CCL_MAJOR_VERSION << CCL_MINOR_VERSION << CCL_UPDATE_VERSION;
    return oneCCL_ver;
  });

  py::object module = py::module::import("torch.distributed");
  py::object register_backend = module.attr("Backend").attr("register_backend");

  register_backend("ccl", py::cpp_function(&c10d::ProcessGroupCCL::createProcessGroupCCL,
                                           py::arg("store"),
                                           py::arg("rank"),
                                           py::arg("size"),
                                           py::arg("timeout") = std::chrono::milliseconds(
                                                   ::c10d::ProcessGroupCCL::OP_TIMEOUT_MILLIS)));

  auto processGroup = module.attr("ProcessGroup");
  auto processGroupCCL = shared_ptr_class_<::c10d::ProcessGroupCCL>(
          module, "ProcessGroupCCL", processGroup);

  processGroupCCL.def(
    py::init([](const std::shared_ptr<::c10d::Store>& store,
                int rank,
                int size,
                std::chrono::milliseconds timeout) {
      return std::make_shared<::c10d::ProcessGroupCCL>(store, rank, size, timeout);
    }),
    py::arg("store"),
    py::arg("rank"),
    py::arg("size"),
    py::arg("timeout") = std::chrono::milliseconds(10 * 1000));

}
