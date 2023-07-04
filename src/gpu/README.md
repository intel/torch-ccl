Dependencies:
  1. MPI
  2. Level-Zero
  3. SYCL enabled compiler

Build:
  make

Run:
  ```mpirun -np <N> allreduce -c 1024 -t <float>```
