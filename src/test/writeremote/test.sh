#!/bin/bash
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
mpirun --prepend-rank -n 2 -ppn 2 ./simple_test $1
#mpirun --prepend-rank -n 8 -ppn 8 ./simple_test
#mpirun --prepend-rank -n 2 -ppn 2  -outfile-pattern log/out.%r.log -errfile-pattern log/err.%r.log ze_tracer -c ./simple_test good
#mpirun --prepend-rank -n 8 -ppn 8  -outfile-pattern log/out.%r.log -errfile-pattern log/err.%r.log ze_tracer -c ./simple_test