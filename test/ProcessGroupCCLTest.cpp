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

void waitWork(
    std::shared_ptr<c10d::ProcessGroup> pg,
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

void testAllgather(int iter = 10000)
{
    auto pg = createProcessGroup();

    std::vector<std::vector<at::Tensor>> allTensors(iter);
    std::vector<std::vector<std::vector<at::Tensor>>> allOutputTensors(iter);

    auto worldSize = pg->getSize();
    auto rank = pg->getRank();

    // Generate inputs
    for (auto i = 0; i < iter; ++i)
    {
        auto tensor = at::ones({16, 16}) * i * rank;
        allTensors[i] = std::vector<at::Tensor>({tensor});
        allOutputTensors[i] = std::vector<std::vector<at::Tensor>>(1);
        allOutputTensors[i][0].resize(worldSize);
        for (auto j = 0; j < worldSize; ++j)
        {
            allOutputTensors[i][0][j] = at::zeros({16, 16});
        }
    }

    std::vector<std::shared_ptr<::c10d::ProcessGroup::Work>> works;
    for (size_t i = 0; i < allTensors.size(); ++i)
    {
        // Kick off work
        std::shared_ptr<::c10d::ProcessGroup::Work> work =
            pg->allgather(allOutputTensors[i], allTensors[i]);
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
                    printf("testAllgather: unexpected result: got %f, expected %f\n",
                        data[k], (float)expected);
                    throw std::runtime_error("BOOM!");
                }
            }
        }
    }
}

void testAllreduce(int iter = 10000)
{
    auto pg = createProcessGroup();

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

    auto worldSize = pg->getSize();

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
}

void testAlltoallBase(int iter = 10000)
{
    auto pg = createProcessGroup();

    std::vector<at::Tensor> allTensors(iter);
    std::vector<at::Tensor> allOutputTensors(iter);

    auto worldSize = pg->getSize();
    auto rank = pg->getRank();

    const int per_rank_size = 16;

    // Generate inputs
    for (auto i = 0; i < iter; ++i)
    {
        auto tensorOne = at::ones({worldSize * per_rank_size}) * i * rank;
        auto tensorZero = at::zeros({worldSize * per_rank_size});

        allTensors[i] = tensorOne;
        allOutputTensors[i] = tensorZero;
    }

    std::vector<int64_t> splitSizes;
    std::vector<std::shared_ptr<::c10d::ProcessGroup::Work>> works;

    for (size_t i = 0; i < allTensors.size(); ++i)
    {
        // Kick off work
        std::shared_ptr<::c10d::ProcessGroup::Work> work =
            pg->alltoall_base(allOutputTensors[i], allTensors[i], splitSizes, splitSizes);
        works.push_back(std::move(work));
    }

    waitWork(pg, works);

    // Verify outputs
    for (int i = 0; i < iter; ++i)
    {
        for (int j = 0; j < worldSize; ++j)
        {
            for (auto k = 0; k < per_rank_size; ++k)
            {
                const auto expected = i * j;
                auto data = allOutputTensors[i].data_ptr<float>();

                size_t idx = j * per_rank_size + k;

                if (data[idx] != expected)
                {
                    printf("testAlltoall: unexpected result: got %f, expected %f, iter %d, rank %d, elem %d\n",
                        data[idx], (float)expected, i, j, k);
                    throw std::runtime_error("BOOM!");
                }
            }
        }
    }
}

void testAlltoall(int iter = 10000)
{
    auto pg = createProcessGroup();

    std::vector<std::vector<at::Tensor>> allTensors(iter);
    std::vector<std::vector<at::Tensor>> allOutputTensors(iter);

    auto worldSize = pg->getSize();
    auto rank = pg->getRank();

    // Generate inputs
    for (auto i = 0; i < iter; ++i)
    {
        auto tensorOne = at::ones({16, 16}) * i * rank;
        auto tensorZero = at::zeros({16, 16});

        allTensors[i].resize(worldSize);
        allOutputTensors[i].resize(worldSize);

        for (auto j = 0; j < worldSize; ++j)
        {
            allTensors[i][j] = tensorOne;
            allOutputTensors[i][j] = tensorZero;
        }
    }

    std::vector<std::shared_ptr<::c10d::ProcessGroup::Work>> works;
    for (size_t i = 0; i < allTensors.size(); ++i)
    {
        // Kick off work
        std::shared_ptr<::c10d::ProcessGroup::Work> work =
            pg->alltoall(allOutputTensors[i], allTensors[i]);
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
                    printf("testAlltoall: unexpected result: got %f, expected %f, iter %d, rank %d, elem %d\n",
                        data[k], (float)expected, i, j, k);
                    throw std::runtime_error("BOOM!");
                }
            }
        }
    }
}

void testBarrier(int iter = 10000)
{
    auto pg = createProcessGroup();

    std::vector<std::shared_ptr<::c10d::ProcessGroup::Work>> works;
    for (auto i = 0; i < iter; ++i)
    {
        // Kick off work
        std::shared_ptr<::c10d::ProcessGroup::Work> work = pg->barrier();
        works.push_back(std::move(work));
    }

    waitWork(pg, works);

    pg->barrier()->wait();
}

void testBroadcast(int iter = 10000)
{
    auto pg = createProcessGroup();

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
}

void testGather(int iter = 10000)
{
    auto pg = createProcessGroup();

    auto worldSize = pg->getSize();
    auto rank = pg->getRank();

    int rootRank = 0;

    // Generate inputs
    std::vector<std::vector<at::Tensor>> allTensors(iter);
    std::vector<std::vector<std::vector<at::Tensor>>> allOutputTensors(iter);

    for (auto i = 0; i < iter; ++i)
    {
        auto tensor = at::ones({16, 16}) * i * rank;
        allTensors[i] = std::vector<at::Tensor>({tensor});

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

    for (size_t i = 0; i < allTensors.size(); ++i)
    {
        // Kick off work
        std::shared_ptr<::c10d::ProcessGroup::Work> work =
            pg->gather(allOutputTensors[i], allTensors[i], gatherOptions);
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
}

void testReduce(int iter = 10000)
{
    auto pg = createProcessGroup();

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
}

void testScatter(int iter = 10000)
{
    auto pg = createProcessGroup();

    auto worldSize = pg->getSize();
    auto rank = pg->getRank();

    int rootRank = 0;

    // Generate inputs
    std::vector<std::vector<std::vector<at::Tensor>>> allTensors(iter);
    std::vector<std::vector<at::Tensor>> allOutputTensors(iter);

    for (auto i = 0; i < iter; ++i)
    {
        auto tensor = at::zeros({16, 16});
        allOutputTensors[i] = std::vector<at::Tensor>({tensor});

        if (rank == rootRank)
        {
            tensor = at::ones({16, 16}) * i;

            allTensors[i] = std::vector<std::vector<at::Tensor>>(1);
            allTensors[i][0].resize(worldSize);
            for (auto j = 0; j < worldSize; ++j)
            {
                allTensors[i][0][j] = tensor;
            }
        }        
    }

    c10d::ScatterOptions scatterOptions;
    scatterOptions.rootRank = rootRank;

    std::vector<std::shared_ptr<::c10d::ProcessGroup::Work>> works;
    for (size_t i = 0; i < allTensors.size(); ++i)
    {
        // Kick off work
        std::shared_ptr<::c10d::ProcessGroup::Work> work =
            pg->scatter(allOutputTensors[i], allTensors[i], scatterOptions);
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
}

int main(int argc, char** argv)
{
    //testAllgather();
    //testAllreduce();
    testAlltoallBase();
    //testAlltoall();
    //testBarrier();
    //testBroadcast();
    //testGather();
    //testReduce();
    //testScatter();

    std::cout << "Test successful" << std::endl;

    return EXIT_SUCCESS;
}
