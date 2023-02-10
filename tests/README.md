# Intel® oneCCL Bindings for PyTorch* unit tests

This tests provides validation of the functionality and performance for collective communication primitives in Intel® oneCCL Bindings for PyTorch*.

## functionality validation of collective communication primitives
To start the test_c10d_ccl.py test, run: 

```bash
python test_c10d_ccl.py
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

