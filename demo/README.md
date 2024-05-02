# Simple Demo for Intel® oneCCL Bindings for PyTorch*

This simple demo show case the functionality for collective communication primitives in Intel® oneCCL Bindings for PyTorch*.

## Single Node Run
To run the simple demo on a single node with 2 instances, run: 

```bash
mpirun -n 2 -l python demo.py

```
The demo could be also run on XPU with " --device xpu " argument. 

```bash
mpirun -n 2 -l python demo.py --device xpu
```

## Multiple Nodes Run
To run the simple demo on multiple nodes, please follow below instructions:

### Ethernet
1. Identify the network interface name for collective communication. ex: eth0
2. Identify the IPs of all nodes. ex: 10.0.0.1,10.0.0.2
3. Identify the master node IP. ex: 10.0.0.1
4. Set the value of np for the total number of instances. ex: 2
5. Set the value of ppn for the number of instance per node. ex: 1

Here is a run command example for cpu according to above steps:

```bash
FI_TCP_IFACE=eth0 I_MPI_OFI_PROVIDER=tcp I_MPI_HYDRA_IFACE=eth0  I_MPI_DEBUG=121 mpirun -host 10.0.0.1,10.0.0.2 -np 2 -ppn 1  --map-by node  python demo.py --device cpu --dist_url 10.0.0.1  --dist_port 29500
```
The demo could be also run on XPU by changing " --device cpu " to " --device xpu " argument. 

