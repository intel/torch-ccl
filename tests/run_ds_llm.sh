#/bin/bash
# run the ds_subgroup_allreduce.py
# for OAM (sub_group=2/4)
mpirun -np 8 -ppn 8 python -u ds_subgroup_allreduce.py --sub_group=2
mpirun -np 8 -ppn 8 python -u ds_subgroup_allreduce.py --sub_group=4
# for Aurora System(TP=2/3/4/6)
mpirun -np 12 -ppn 12 python -u ds_subgroup_allreduce.py --sub_group=2
mpirun -np 12 -ppn 12 python -u ds_subgroup_allreduce.py --sub_group=3
mpirun -np 12 -ppn 12 python -u ds_subgroup_allreduce.py --sub_group=4
mpirun -np 12 -ppn 12 python -u ds_subgroup_allreduce.py --sub_group=6

# should run the ds_p2p_crossnodes.py on 3 nodes 
# -host is the name for this 3 nodes
# --dist_url is the IP on your node, you can use (hostname -I) to get.
mpirun -host x1002c4s1b0n0,x1002c4s2b0n0,x1002c4s3b0n0 -np 36 -ppn 12 python -u ds_p2p_crossnodes.py --dist_url 10.0.1.141 --world_size 36
