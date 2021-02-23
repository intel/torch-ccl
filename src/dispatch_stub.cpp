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

#include "dispatch_stub.h"

namespace torch_ccl {

static DispatchStub default_stubs;
constexpr DispatchStub* default_stubs_addr = &default_stubs;

DispatchStub* DispatchStub::stubs_[to_int(c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES)] = {
  /*[0 ... (to_int(c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES) - 1)] = default_stubs_addr*/
};

void DispatchStub::register_ccl_stub(c10::DeviceType dev_type, DispatchStub* stub) {
  static std::once_flag dispatch_once_flag;
  std::call_once(dispatch_once_flag, []() {
    for(size_t i = 0; i < static_cast<int>(c10::DeviceType::COMPILE_TIME_MAX_DEVICE_TYPES); i++) {
      stubs_[i] = default_stubs_addr;
    }
  });
  stubs_[to_int(dev_type)] = stub;
}

}
