import os
import torch
import torch.distributed as dist

def setup_distributed():
    required = ["RANK", "WORLD_SIZE", "LOCAL_RANK"]
    for k in required:
        assert k in os.environ, f"Missing env var {k}"
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}")

def cleanup_distributed():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


