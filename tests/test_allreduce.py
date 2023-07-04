import torch
import intel_extension_for_pytorch
import oneccl_bindings_for_pytorch
import torch.distributed as dist
import os
import time

tokens = 16
rounds = 70 * 2 * tokens

count = 14336

total = 1024 * 1024 * 72
repeat = 4

# profiling = False
# profiling = True

datatype = torch.float16
# datatype = torch.float32

os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'

dist.init_process_group("ccl")
rank = dist.get_rank()
size = dist.get_world_size()

device = "xpu:{}".format(rank)
# allreduce data
data = (torch.ones(count, dtype=datatype) * 0.1).to(device)

a = (torch.zeros((int(total / count), count), dtype=datatype)).to(device)

# warm up
for i in range(5):
    a[0] += (data * 0.1)
    for j in range(repeat):
        a += 0.01
    dist.all_reduce(data)
    data /= size
    sync = data.cpu()

#start_events = []
#end_events = []

dist.barrier()
start = time.time()
for i in range(rounds):
#    start_event = None
#    end_event = None
#    if profiling:
#        start_event = torch.xpu.Event(enable_timing=True)
#        end_event = torch.xpu.Event(enable_timing=True)
    a[0] += (data * 0.1)
    for j in range(repeat):
        a += 0.01
    #print("XPU: {} {}".format(i, a[0][0]))
#    if profiling:
#        start_event.record()
    dist.all_reduce(data)
#    if profiling:
#        end_event.record()
    data /= size
    sync = data.cpu()
#    if profiling:
#        start_events.append(start_event)
#        end_events.append(end_event)

# print(data[0])
data = data.cpu()
# torch.xpu.synchronize('xpu:{}'.format(rank))
span = time.time() - start
print('{} rounds on reducing {} elements. Time used {}'.format(rounds, count, span))

tmp_a = torch.zeros(1, dtype=datatype)
tmp_data = torch.ones(1, dtype=datatype) * 0.1
for i in range(5):
    tmp_a += (tmp_data * 0.1)
    for j in range(repeat):
        tmp_a += 0.01
    tmp_data *= size
    tmp_data /= size

for i in range(rounds):
    tmp_a += (tmp_data * 0.1)
    for j in range(repeat):
        tmp_a += 0.01
    #print("CPU: {} {}".format(i, tmp_a[0]))
    tmp_data *= size
    tmp_data /= size

a = a.cpu()

error = False
for i in range(count):
    if tmp_a[0] != a[0][i]:
        if not error:
            print("Error on {}: {} vs {}".format(i, tmp_a[0], a[0][i]))
            error = True
    else:
        if error:
            print("No error on {}".format(i))
            error = False

#if profiling:
#    for i in range(len(start_events)):
#        allreduce_time = start_events[i].elapsed_time(end_events[i])
#        print('Round %d allreduce time %.3fms' % (i, allreduce_time))
#        if i != len(start_events) - 1:
#            compute_time = end_events[i].elapsed_time(start_events[i + 1])
#            print('Round %d compute time %.3fms' % (i + 1, compute_time))



