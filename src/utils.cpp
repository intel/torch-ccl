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

#include "utils.h"

namespace oneccl_bindings_for_pytorch {

// Op mapping
using c10d::ReduceOp;
std::map<c10d::ReduceOp, ccl::reduction> cclOps =
  {
    {ReduceOp::MIN, ccl::reduction::min},
    {ReduceOp::MAX, ccl::reduction::max},
    {ReduceOp::SUM, ccl::reduction::sum},
    {ReduceOp::PRODUCT, ccl::reduction::prod},
  };

std::map<at::ScalarType, ccl::datatype> cclDatatypes =
  {
    {at::kByte, ccl::datatype::uint8},
    {at::kChar, ccl::datatype::int8},
    {at::kShort, ccl::datatype::int16},
    {at::kInt, ccl::datatype::int32},
    {at::kLong, ccl::datatype::int64},
    {at::kHalf, ccl::datatype::float16},
    {at::kFloat, ccl::datatype::float32},
    {at::kDouble, ccl::datatype::float64},
    {at::kBFloat16, ccl::datatype::bfloat16},
  };

// Get the key from the list of devices
std::string get_key_from_devs(const std::vector<at::Device>& devices) {
  std::string key = DeviceTypeName(devices[0].type(), /* lower case */ true) + ":";
  for (auto& device : devices) {
    key.append(std::to_string(device.index()) + ",");
  }
  return key;
}

// Get the list of devices from list of tensors
std::vector<at::Device> get_device_list(const std::vector<at::Tensor>& tensors) {
  std::vector<at::Device> res;
  res.reserve(tensors.size());
  for (auto& tensor : tensors) {
    res.push_back(tensor.device());
  }
  return res;
}

std::vector<at::Device> get_device_list(const std::vector<std::vector<at::Tensor> >& tensors) {
  std::vector<at::Device> res;
  res.reserve(tensors.size());
  for (auto& tensor : tensors) {
    res.push_back(tensor[0].device());
  }
  return res;
}

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

void checkSingleTensorHelper(const at::Tensor& tensor)
{
  TORCH_CHECK(tensor.is_sparse() || tensor.is_contiguous(), "input dense tensor has to be contiguous");
  TORCH_CHECK(!tensor.is_cuda(), "CUDA tensor detected and CCL doesn't support CUDA buffers");
  TORCH_CHECK(tensor.numel() >= 0, "input tensor numel should be non-negative");
}

void checkSingleTensor(const std::vector<at::Tensor>& tensors)
{
  TORCH_CHECK(tensors.size() == 1,
              "CCL process group does not support tensors count " + std::to_string(tensors.size()));

  checkSingleTensorHelper(tensors[0]);
}


void checkSameType(const at::Tensor& tensor,
                   const std::vector<at::Tensor>& tensors)
{
  for (size_t i = 0; i < tensors.size(); ++i)
  {
    TORCH_CHECK(tensors[i].scalar_type() == tensor.scalar_type(),
                "Tensors are not equal in data type");
    TORCH_CHECK(tensors[i].device().type() == tensor.device().type(),
                "Tensors are not in same device type. Expect: ", tensor.device().type(),
                " But got: ", tensors[i].device().type());

    checkSingleTensorHelper(tensors[i]);
  }
}

void checkSameType(const at::Tensor& tensor,
                   const std::vector<std::vector<at::Tensor>>& tensors)
{
  for (size_t i = 0; i < tensors.size(); ++i)
  {
    checkSameType(tensor, tensors[i]);
  }
}

}
