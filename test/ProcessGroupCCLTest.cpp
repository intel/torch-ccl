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

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>

#include <c10d/HashStore.hpp>

#include "ProcessGroupCCL.hpp"

std::shared_ptr<c10d::ProcessGroup> createProcessGroup()
{
    auto store = std::make_shared<c10d::HashStore>();
    std::chrono::duration<float> timeout(1);
    return c10d::ProcessGroupCCL::createProcessGroupCCL(store, -1, -1, timeout);
}

void waitWork(std::shared_ptr<c10d::ProcessGroup> pg,
              std::vector<std::shared_ptr<c10d::ProcessGroup::Work>> works)
{
    for (auto& work : works)
    {
        try
        {
            work->wait();
        }
        catch (const std::exception& ex)
        {
            std::cerr << "Exception received: " << ex.what() << std::endl;
            exit(0);
        }
    }
}

void testAllgatherFlat(int iter = 1000)
{
    auto pg = createProcessGroup();

    std::vector<std::vector<at::Tensor>> allInputTensors(iter);
    std::vector<std::vector<std::vector<at::Tensor>>> allOutputTensors(iter);

    auto rank = pg->getRank();
    auto worldSize = pg->getSize();

    const int perRankSize = 256;

    std::vector<int64_t> recvCounts(worldSize);
    int64_t totalRecvCount = 0;

    for (auto i = 0; i < worldSize; ++i)
    {
        recvCounts[i] = perRankSize + i;
        totalRecvCount += recvCounts[i];
    }

    // Generate inputs
    for (auto i = 0; i < iter; ++i)
    {
        auto tensorOne = at::ones({recvCounts[rank]}) * i * rank;
        allInputTensors[i] = std::vector<at::Tensor>({tensorOne});

        auto tensorZero = at::zeros({totalRecvCount});

        auto tensorZeroChunks =
            tensorZero.split_with_sizes(c10::IntArrayRef((int64_t*)recvCounts.data(), recvCounts.size()), 0);
        allOutputTensors[i] = std::vector<std::vector<at::Tensor>>(1);
        allOutputTensors[i][0].resize(worldSize);
        for (auto j = 0; j < worldSize; ++j)
        {
            allOutputTensors[i][0][j] = tensorZeroChunks[j];
        }
    }

    std::vector<std::shared_ptr<::c10d::ProcessGroup::Work>> works;
    for (size_t i = 0; i < allInputTensors.size(); ++i)
    {
        // Kick off work
        std::shared_ptr<::c10d::ProcessGroup::Work> work =
            pg->allgather(allOutputTensors[i], allInputTensors[i]);
        works.push_back(std::move(work));
    }

    waitWork(pg, works);

    // Verify outputs
    for (int i = 0; i < iter; ++i)
    {
        for (int j = 0; j < worldSize; ++j)
        {
            const auto expected = i * j;
            auto data = allOutputTensors[i][0][j].data_ptr<float>();
            for (auto k = 0; k < allOutputTensors[i][0][j].numel(); ++k)
            {
                if (data[k] != expected)
                {
                    printf("testAllgatherFlat: unexpected result: got %f, expected %f\n",
                        data[k], (float)expected);
                    throw std::runtime_error("BOOM!");
                }
            }
        }
    }

    pg->barrier()->wait();

    if (rank == 0)
        printf("testAllgatherFlat: passed\n");
}

void testAllgatherNotFlat(int iter = 1000)
{
    auto pg = createProcessGroup();

    std::vector<std::vector<at::Tensor>> allInputTensors(iter);
    std::vector<std::vector<std::vector<at::Tensor>>> allOutputTensors(iter);

    auto rank = pg->getRank();
    auto worldSize = pg->getSize();

    // Generate inputs
    for (auto i = 0; i < iter; ++i)
    {
        auto tensor = at::ones({16, 16}) * i * rank;
        allInputTensors[i] = std::vector<at::Tensor>({tensor});
        allOutputTensors[i] = std::vector<std::vector<at::Tensor>>(1);
        allOutputTensors[i][0].resize(worldSize);
        for (auto j = 0; j < worldSize; ++j)
        {
            allOutputTensors[i][0][j] = at::zeros({16, 16});
        }
    }

    std::vector<std::shared_ptr<::c10d::ProcessGroup::Work>> works;
    for (size_t i = 0; i < allInputTensors.size(); ++i)
    {
        // Kick off work
        std::shared_ptr<::c10d::ProcessGroup::Work> work =
            pg->allgather(allOutputTensors[i], allInputTensors[i]);
        works.push_back(std::move(work));
    }

    waitWork(pg, works);

    // Verify outputs
    for (int i = 0; i < iter; ++i)
    {
        for (int j = 0; j < worldSize; ++j)
        {
            const auto expected = i * j;
            auto data = allOutputTensors[i][0][j].data_ptr<float>();
            for (auto k = 0; k < allOutputTensors[i][0][j].numel(); ++k)
            {
                if (data[k] != expected)
                {
                    printf("testAllgatherNotFlat: unexpected result: got %f, expected %f\n",
                        data[k], (float)expected);
                    throw std::runtime_error("BOOM!");
                }
            }
        }
    }

    pg->barrier()->wait();

    if (rank == 0)
        printf("testAllgatherNotFlat: passed\n");
}

void testAllreduce(int iter = 1000)
{
    auto pg = createProcessGroup();

    auto rank = pg->getRank();
    auto worldSize = pg->getSize();

    // Generate inputs
    std::vector<std::vector<at::Tensor>> allTensors(iter);

    for (auto i = 0; i < iter; ++i)
    {
        auto tensor = at::ones({16, 16}) * i;
        allTensors[i] = std::vector<at::Tensor>({tensor});
    }

    std::vector<std::shared_ptr<::c10d::ProcessGroup::Work>> works;
    for (auto& tensors : allTensors)
    {
        // Kick off work
        std::shared_ptr<::c10d::ProcessGroup::Work> work = pg->allreduce(tensors);
        works.push_back(std::move(work));
    }

    waitWork(pg, works);

    // Verify outputs
    for (int i = 0; i < iter; ++i)
    {
        const auto expected = worldSize * i;
        auto data = allTensors[i][0].data_ptr<float>();
        for (auto j = 0; j < allTensors[i][0].numel(); ++j)
        {
            if (data[j] != expected)
            {
                printf("testAllreduce: unexpected result: got %f, expected %f\n",
                    data[j], (float)expected);
                throw std::runtime_error("BOOM!");
            }
        }
    }

    pg->barrier()->wait();

    if (rank == 0)
        printf("testAllreduce: passed\n");
}

void testAlltoallBase(int iter = 1000)
{
    auto pg = createProcessGroup();

    std::vector<at::Tensor> allInputTensors(iter);
    std::vector<at::Tensor> allOutputTensors(iter);

    auto rank = pg->getRank();
    auto worldSize = pg->getSize();

    const int perRankSize = 256;

    // Generate inputs
    for (auto i = 0; i < iter; ++i)
    {
        auto tensorOne = at::ones({worldSize * perRankSize}) * i * rank;
        auto tensorZero = at::zeros({worldSize * perRankSize});

        allInputTensors[i] = tensorOne;
        allOutputTensors[i] = tensorZero;
    }

    std::vector<int64_t> splitSizes;
    std::vector<std::shared_ptr<::c10d::ProcessGroup::Work>> works;

    for (size_t i = 0; i < allInputTensors.size(); ++i)
    {
        // Kick off work
        std::shared_ptr<::c10d::ProcessGroup::Work> work =
            pg->alltoall_base(allOutputTensors[i], allInputTensors[i],
                              splitSizes, splitSizes);
        works.push_back(std::move(work));
    }

    waitWork(pg, works);

    // Verify outputs
    for (int i = 0; i < iter; ++i)
    {
        for (int j = 0; j < worldSize; ++j)
        {
            for (auto k = 0; k < perRankSize; ++k)
            {
                const auto expected = i * j;
                auto data = allOutputTensors[i].data_ptr<float>();

                size_t idx = j * perRankSize + k;

                if (data[idx] != expected)
                {
                    printf("testAlltoall: unexpected result: got %f, expected %f, iter %d, rank %d, elem %d\n",
                        data[idx], (float)expected, i, j, k);
                    throw std::runtime_error("BOOM!");
                }
            }
        }
    }

    pg->barrier()->wait();

    if (rank == 0)
        printf("testAlltoall: passed\n");
}

void testAlltoallFlat(int iter = 1000)
{
    auto pg = createProcessGroup();

    std::vector<std::vector<at::Tensor>> allInputTensors(iter);
    std::vector<std::vector<at::Tensor>> allOutputTensors(iter);

    auto rank = pg->getRank();
    auto worldSize = pg->getSize();

    const int perRankSize = 256;

    // Generate inputs
    for (auto i = 0; i < iter; ++i)
    {
        auto tensorOne = at::ones({worldSize * perRankSize}) * i * rank;
        auto tensorZero = at::zeros({worldSize * perRankSize});

        auto tensorOneChunks = tensorOne.split(perRankSize);
        auto tensorZeroChunks = tensorZero.split(perRankSize);

        allInputTensors[i].resize(worldSize);
        allOutputTensors[i].resize(worldSize);

        for (auto j = 0; j < worldSize; ++j)
        {
            allInputTensors[i][j] = tensorOneChunks[j];
            allOutputTensors[i][j] = tensorZeroChunks[j];
        }
    }

    std::vector<std::shared_ptr<::c10d::ProcessGroup::Work>> works;
    for (size_t i = 0; i < allInputTensors.size(); ++i)
    {
        // Kick off work
        std::shared_ptr<::c10d::ProcessGroup::Work> work =
            pg->alltoall(allOutputTensors[i], allInputTensors[i]);
        works.push_back(std::move(work));
    }

    waitWork(pg, works);

    // Verify outputs
    for (int i = 0; i < iter; ++i)
    {
        for (int j = 0; j < worldSize; ++j)
        {
            const auto expected = i * j;
            auto data = allOutputTensors[i][j].data_ptr<float>();
            for (auto k = 0; k < allOutputTensors[i][j].numel(); ++k)
            {
                if (data[k] != expected)
                {
                    printf("testAlltoallFlat: unexpected result: got %f, "
                        "expected %f, iter %d, rank %d, elem %d\n",
                        data[k], (float)expected, i, j, k);
                    throw std::runtime_error("BOOM!");
                }
            }
        }
    }

    pg->barrier()->wait();

    if (rank == 0)
        printf("testAlltoallFlat: passed\n");
}

void testAlltoallNonFlat(int iter = 1000)
{
    auto pg = createProcessGroup();

    std::vector<std::vector<at::Tensor>> allInputTensors(iter);
    std::vector<std::vector<at::Tensor>> allOutputTensors(iter);

    auto rank = pg->getRank();
    auto worldSize = pg->getSize();

    // Generate inputs
    for (auto i = 0; i < iter; ++i)
    {
        auto tensorOne = at::ones({16, 16}) * i * rank;
        auto tensorZero = at::zeros({16, 16});

        allInputTensors[i].resize(worldSize);
        allOutputTensors[i].resize(worldSize);

        for (auto j = 0; j < worldSize; ++j)
        {
            allInputTensors[i][j] = tensorOne.clone();
            allOutputTensors[i][j] = tensorZero.clone();
        }
    }

    std::vector<std::shared_ptr<::c10d::ProcessGroup::Work>> works;
    for (size_t i = 0; i < allInputTensors.size(); ++i)
    {
        // Kick off work
        std::shared_ptr<::c10d::ProcessGroup::Work> work =
            pg->alltoall(allOutputTensors[i], allInputTensors[i]);
        works.push_back(std::move(work));
    }

    waitWork(pg, works);

    // Verify outputs
    for (int i = 0; i < iter; ++i)
    {
        for (int j = 0; j < worldSize; ++j)
        {
            const auto expected = i * j;
            auto data = allOutputTensors[i][j].data_ptr<float>();
            for (auto k = 0; k < allOutputTensors[i][j].numel(); ++k)
            {
                if (data[k] != expected)
                {
                    printf("testAlltoallNonFlat: unexpected result: got %f, "
                        "expected %f, iter %d, rank %d, elem %d\n",
                        data[k], (float)expected, i, j, k);
                    throw std::runtime_error("BOOM!");
                }
            }
        }
    }

    pg->barrier()->wait();

    if (rank == 0)
        printf("testAlltoallNonFlat: passed\n");
}

void testBarrier(int iter = 1000)
{
    auto pg = createProcessGroup();

    auto rank = pg->getRank();

    std::vector<std::shared_ptr<::c10d::ProcessGroup::Work>> works;
    for (auto i = 0; i < iter; ++i)
    {
        // Kick off work
        std::shared_ptr<::c10d::ProcessGroup::Work> work = pg->barrier();
        works.push_back(std::move(work));
    }

    waitWork(pg, works);

    pg->barrier()->wait();

    if (rank == 0)
        printf("testBarrier: passed\n");
}

void testBroadcast(int iter = 1000)
{
    auto pg = createProcessGroup();

    auto rank = pg->getRank();

    // Generate inputs
    std::vector<std::vector<at::Tensor>> allTensors(iter);

    for (auto i = 0; i < iter; ++i)
    {
        if (pg->getRank() == 0)
        {
            auto tensor = at::ones({16, 16}) * i;
            allTensors[i] = std::vector<at::Tensor>({tensor});
        }
        else
        {
            auto tensor = at::zeros({16, 16});
            allTensors[i] = std::vector<at::Tensor>({tensor});
        }
    }

    std::vector<std::shared_ptr<::c10d::ProcessGroup::Work>> works;
    for (auto& tensors : allTensors)
    {
        // Kick off work
        std::shared_ptr<::c10d::ProcessGroup::Work> work = pg->broadcast(tensors);
        works.push_back(std::move(work));
    }

    waitWork(pg, works);

    // Verify outputs
    for (int i = 0; i < iter; ++i)
    {
        const auto expected = i;
        auto data = allTensors[i][0].data_ptr<float>();
        for (auto j = 0; j < allTensors[i][0].numel(); ++j)
        {
            if (data[j] != expected)
            {
                printf("testBroadcast: unexpected result: got %f, expected %f\n",
                    data[j], (float)expected);
                throw std::runtime_error("BOOM!");
            }
        }
    }

    pg->barrier()->wait();

    if (rank == 0)
        printf("testBroadcast: passed\n");
}

void testGather(int iter = 1000)
{
    auto pg = createProcessGroup();

    auto rank = pg->getRank();
    auto worldSize = pg->getSize();

    int rootRank = 0;

    // Generate inputs
    std::vector<std::vector<at::Tensor>> allInputTensors(iter);
    std::vector<std::vector<std::vector<at::Tensor>>> allOutputTensors(iter);

    for (auto i = 0; i < iter; ++i)
    {
        auto tensor = at::ones({16, 16}) * i * rank;
        allInputTensors[i] = std::vector<at::Tensor>({tensor});

        if (rank == rootRank)
        {
            allOutputTensors[i] = std::vector<std::vector<at::Tensor>>(1);
            allOutputTensors[i][0].resize(worldSize);
            for (auto j = 0; j < worldSize; ++j)
            {
                allOutputTensors[i][0][j] = at::zeros({16, 16});
            }
        }
    }

    c10d::GatherOptions gatherOptions;
    gatherOptions.rootRank = rootRank;

    std::vector<std::shared_ptr<::c10d::ProcessGroup::Work>> works;

    for (size_t i = 0; i < allInputTensors.size(); ++i)
    {
        // Kick off work
        std::shared_ptr<::c10d::ProcessGroup::Work> work =
            pg->gather(allOutputTensors[i], allInputTensors[i], gatherOptions);
        works.push_back(std::move(work));
    }

    waitWork(pg, works);

    if (rank == 0)
    {
        // Verify outputs
        for (int i = 0; i < iter; ++i)
        {
            for (int j = 0; j < worldSize; ++j)
            {
                const auto expected = i * j;
                auto data = allOutputTensors[i][0][j].data_ptr<float>();
                for (auto k = 0; k < allOutputTensors[i][0][j].numel(); ++k)
                {
                    if (data[k] != expected)
                    {
                        printf("testGather: unexpected result: got %f, expected %f\n",
                            data[k], (float)expected);
                        throw std::runtime_error("BOOM!");
                    }
                }
            }
        }
    }

    pg->barrier()->wait();

    if (rank == 0)
        printf("testGather: passed\n");
}

void testReduce(int iter = 1000)
{
    auto pg = createProcessGroup();

    auto rank = pg->getRank();

    // Generate inputs
    std::vector<std::vector<at::Tensor>> allTensors(iter);

    for (auto i = 0; i < iter; ++i)
    {
        auto tensor = at::ones({16, 16}) * i;
        allTensors[i] = std::vector<at::Tensor>({tensor});
    }

    std::vector<std::shared_ptr<::c10d::ProcessGroup::Work>> works;
    for (auto& tensors : allTensors)
    {
        // Kick off work
        std::shared_ptr<::c10d::ProcessGroup::Work> work = pg->reduce(tensors);
        works.push_back(std::move(work));
    }

    waitWork(pg, works);

    // Get the world size
    auto worldSize = pg->getSize();

    if (pg->getRank() == 0)
    {
        // Verify outputs
        for (int i = 0; i < iter; ++i)
        {
            const auto expected = worldSize * i;
            auto data = allTensors[i][0].data_ptr<float>();
            for (auto j = 0; j < allTensors[i][0].numel(); ++j)
            {
                if (data[j] != expected)
                {
                    printf("testReduce: unexpected result: got %f, expected %f\n",
                        data[j], (float)expected);
                    throw std::runtime_error("BOOM!");
                }
            }
        }
    }

    pg->barrier()->wait();

    if (rank == 0)
        printf("testReduce: passed\n");
}

void testScatter(int iter = 1000)
{
    auto pg = createProcessGroup();

    auto rank = pg->getRank();
    auto worldSize = pg->getSize();

    int rootRank = 0;

    // Generate inputs
    std::vector<std::vector<std::vector<at::Tensor>>> allInputTensors(iter);
    std::vector<std::vector<at::Tensor>> allOutputTensors(iter);

    for (auto i = 0; i < iter; ++i)
    {
        auto tensor = at::zeros({16, 16});
        allOutputTensors[i] = std::vector<at::Tensor>({tensor});

        if (rank == rootRank)
        {
            tensor = at::ones({16, 16}) * i;

            allInputTensors[i] = std::vector<std::vector<at::Tensor>>(1);
            allInputTensors[i][0].resize(worldSize);
            for (auto j = 0; j < worldSize; ++j)
            {
                allInputTensors[i][0][j] = tensor.clone();
            }
        }        
    }

    c10d::ScatterOptions scatterOptions;
    scatterOptions.rootRank = rootRank;

    std::vector<std::shared_ptr<::c10d::ProcessGroup::Work>> works;
    for (size_t i = 0; i < allInputTensors.size(); ++i)
    {
        // Kick off work
        std::shared_ptr<::c10d::ProcessGroup::Work> work =
            pg->scatter(allOutputTensors[i], allInputTensors[i], scatterOptions);
        works.push_back(std::move(work));
    }

    waitWork(pg, works);

    // Verify outputs
    for (int i = 0; i < iter; ++i)
    {
        for (int j = 0; j < worldSize; ++j)
        {
            const auto expected = i;
            auto data = allOutputTensors[i][0].data_ptr<float>();
            for (auto k = 0; k < allOutputTensors[i][0].numel(); ++k)
            {
                if (data[k] != expected)
                {
                    printf("testScatter: unexpected result: got %f, expected %f\n",
                        data[k], (float)expected);
                    throw std::runtime_error("BOOM!");
                }
            }
        }
    }

    pg->barrier()->wait();

    if (rank == 0)
        printf("testScatter: passed\n");
}

int main(int argc, char** argv)
{
    testAllgatherFlat();
    testAllgatherNotFlat();
    testAllreduce();
    testAlltoallBase();
    testAlltoallFlat();
    testAlltoallNonFlat();
    testBarrier();
    testBroadcast();
    testGather();
    testReduce();
    testScatter();

    std::cout << "Test successful" << std::endl;

    return EXIT_SUCCESS;
}
