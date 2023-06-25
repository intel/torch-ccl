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
python testccl_gpu.py
```

Note this unit test is a stress test with a long time to start. You may need to wait ~5min to get the log "starting to initialize tensors ...".

