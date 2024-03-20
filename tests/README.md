# Intel® oneCCL Bindings for PyTorch* unit tests

These tests validate the functionality and performance for collective communication primitives in Intel® oneCCL Bindings for PyTorch*.

## functionality validation of collective communication primitives
To start the test_c10d_ccl.py test, run: 

```bash
python test_c10d_ccl.py
```

## functionality validation of point-to-point communication primitives
For within-card and cross-cards p2p test, run:

```bash
python test_c10d_p2p.py
```

For cross-nodes p2p test, run:

```bash
# Mpich
mpiexec -host nodeA,nodeB -np 24 -ppn 12 python -u test_p2p_crossnodes.py --dist_url $NODE_IP --world_size 24
```

## functionality validation of barrier
For cpu barrier, run:

```bash
mpirun -np 2 python test_barrier.py
```

For xpu barrier (built with "COMPUTE_BACKEND=dpcpp"), run:

```bash
mpirun -np 2 python test_barrier.py --device xpu
```

## broadcast/allreduce profiling
To start the test_allreduce.py test, run:

```bash
mpirun -np 12 -ppn 12 python ddp_allreduce.py --warm 10 --iter 20 --fixed
```

## DeepSpeed test
cpu test:
```bash
python testccl_cpu.py
```

gpu test (runs on 1 node 6 cards 12 tiles):
```bash
python testccl_gpu.py --world_size 12
```
gpu test for scale-out (runs on 2nodes and 24 ranks):
```bash
mpirun -np 24 -ppn 12 python testccl_gpu_mpi.py
```

Note this unit test is a stress test with a long time to start. You may need to wait ~5min to get the log "starting to initialize tensors ...".

## allreduce of LLM path
This test case goes to special path for allreduce operation on xpu device if launched rank(-np) <= 8. Run:
```bash
mpirun -np 2 python test_llm_allreduce.py
```
If you want to disable this path and use oneCCL allreduce instead, set TORCH_CCL_GPU_ALLREDUCE to 0. Run:
```bash
TORCH_CCL_GPU_ALLREDUCE=0 mpirun -np 2 python test_llm_allreduce.py
## Test Functionality of FSDP
```bash
export CCL_ZE_IPC_EXCHANGE=sockets # for pytorch multiprocessing launch
python test_fsdp.py
```

## subgroup tests ds_subgroup_allreduce.py
# for OAM (sub_group=2/4)
```bash
mpirun -np 8 -ppn 8 python -u ds_subgroup_allreduce.py --sub_group=2
mpirun -np 8 -ppn 8 python -u ds_subgroup_allreduce.py --sub_group=4
```
# for Aurora System(TP=2/3/4/6)
```bash
mpirun -np 12 -ppn 12 python -u ds_subgroup_allreduce.py --sub_group=2
mpirun -np 12 -ppn 12 python -u ds_subgroup_allreduce.py --sub_group=3
mpirun -np 12 -ppn 12 python -u ds_subgroup_allreduce.py --sub_group=4
mpirun -np 12 -ppn 12 python -u ds_subgroup_allreduce.py --sub_group=6
```

## deep speed scale-out tests
The ds_p2p_crossnodes.py test case should be run on 3 nodes 
```bash
mpirun -host x1002c4s1b0n0,x1002c4s2b0n0,x1002c4s3b0n0 -np 36 -ppn 12 python -u ds_p2p_crossnodes.py --dist_url 10.0.1.141 --world_size 36
```
-host is the name for this 3 nodes
--dist_url is the IP on your node, you can use (hostname -I) to get.