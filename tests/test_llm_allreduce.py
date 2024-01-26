import torch
import intel_extension_for_pytorch
import oneccl_bindings_for_pytorch
import torch.distributed as dist
import os

os.environ['RANK'] = str(os.environ.get('PMI_RANK', 0))
os.environ['WORLD_SIZE'] = str(os.environ.get('PMI_SIZE', 1))
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'
dist.init_process_group("ccl")
rank = dist.get_rank()
size = dist.get_world_size()

device = "xpu:{}".format(rank)
llm_shapes = [
    # GPT-J 6B
    (1, 32, 4096), (1, 1024, 4096), (1, 1, 4096), (1, 4, 4096),
    # Llama 7B
    (1, 32, 4096), (1, 1024, 4096), (1, 1, 4096), (1, 4, 4096),
    # Llama 13B
    (1, 32, 5120), (1, 1024, 5120), (1, 4, 5120), (1, 1, 5120),
    # Llama2 7B
    (1, 32, 4096), (1, 1024, 4096), (1, 1, 4096), (1, 4, 4096),
    # Llama2 13B
    (1, 32, 5120), (1, 1024, 5120), (1, 4, 5120), (1, 1, 5120),
    # Llama2 70B
    (1, 32, 8192), (1, 1024, 8192), (1, 1, 8192), (1, 4, 8192),
    # OPT 6.7B
    (1, 32, 4096), (1, 1024, 4096), (1, 1, 4096), (1, 4, 4096),
    # OPT 30B
    (1, 32, 7168), (1, 1, 7168), (1, 1024, 7168), (1, 4, 7168),
    # Bloom 7B
    (1, 33, 4096), (1, 1, 4096), (1, 4, 4096), (1, 1028, 4096),
    # Bloom 176B
    (1, 4, 14336), (1, 1028, 14336), (1, 33, 14336), (1, 1, 14336)
]

os.environ['TORCH_LLM_ALLREDUCE_DEBUG'] = '1'
for shape in llm_shapes:
    data = torch.rand(shape, dtype=torch.float16).to(device)
    # Expected value is identical to input for average allreduce.
    expect_result = data
    # Allreduce is an inplace op, data will represent output.
    dist.all_reduce(data)
    assert torch.allclose(data, expect_result)
