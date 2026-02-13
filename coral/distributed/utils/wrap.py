import torch
import torch.nn as nn
from functools import partial

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, BackwardPrefetch
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp import ShardingStrategy

from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
)

from coral.distributed.utils.activation_ckpt import apply_fsdp_checkpointing
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy


mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16, 
    reduce_dtype=torch.float32,
    buffer_dtype=torch.bfloat16,
)

def setup_fsdp_and_checkpointing(model, fsdp=True):
    if fsdp:
        auto_wrap = partial(
            size_based_auto_wrap_policy,
            min_num_params=10_000_000
        )

        fsdp_model = FSDP(
            model,
            auto_wrap_policy=auto_wrap,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            mixed_precision=mp_policy,
            use_orig_params=True,
            forward_prefetch=True,                            
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=torch.cuda.current_device(),
        )
        apply_fsdp_checkpointing(fsdp_model)
        return fsdp_model
    else:
        local_rank = torch.cuda.current_device()
        model = model.to(local_rank)
        ddp_model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,  
        )
        apply_fsdp_checkpointing(ddp_model)
        return ddp_model