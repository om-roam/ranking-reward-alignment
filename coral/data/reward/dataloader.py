import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from coral.data.reward.dataset import ListwiseRankingDataset
from coral.data.reward.collator import listwise_collate_fn

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

def data_loader(
    data_path: str,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
    pretokenize: bool = False,
    tokenizer=None,
) -> DataLoader:
    # Initialize dataset
    dataset = ListwiseRankingDataset(data_path)
    if pretokenize:
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided if pretokenize=True")
        print("--> Pre-tokenizing dataset on CPU...")
        dataset.pretokenize(tokenizer) 

    if dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(
            dataset,
            shuffle=True,
            drop_last=True 
        )
        shuffle = False 
    else:
        sampler = None
        shuffle = True

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        collate_fn=listwise_collate_fn,
        num_workers=max(2, num_workers),       
        pin_memory=pin_memory,                
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2,                    
        drop_last=True                       
    )

    return loader


# def data_loader(
#     data_path: str,
#     batch_size: int = 8,
#     num_workers: int = 0,
#     pin_memory: bool = False,
# ) -> DataLoader:
#     """
#     Returns a DataLoader for FSDP or single-node training.
    
#     Args:
#         data_path: Path to the dataset.
#         batch_size: Batch size per GPU.
#         num_workers: Number of workers for data loading.
#         pin_memory: Whether to pin memory for faster GPU transfer.
    
#     Returns:
#         DataLoader object ready for distributed training.
#     """
#     dataset = ListwiseRankingDataset(data_path)

#     if dist.is_available() and dist.is_initialized():
#         # Distributed sampler for multi-node/multi-GPU FSDP
#         sampler = DistributedSampler(
#             dataset,
#             shuffle=True,   # shuffle within each node
#             drop_last=True, # important for FSDP to avoid uneven batches
#         )
#         shuffle = False  # DataLoader should NOT shuffle when using DistributedSampler
#     else:
#         sampler = None
#         shuffle = True

#     # sampler.set_epoch(0)
#     loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         sampler=sampler,
#         shuffle=shuffle,
#         collate_fn=listwise_collate_fn,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#         persistent_workers=False,  # keeps workers alive for multiple epochs
#     )

#     return loader
