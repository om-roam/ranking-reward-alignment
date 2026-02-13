from transformers import AutoTokenizer

import torch.distributed as dist
from transformers import AutoTokenizer


def _is_dist_initialized():
    return dist.is_available() and dist.is_initialized()


# class RewardTokenizer:
#     def __init__(
#         self,
#         model_name: str,
#         max_length: int = 512,
#         cache_dir: str | None = None,
#     ):
#         is_dist = _is_dist_initialized()
#         rank = dist.get_rank() if is_dist else 0

#         # ----------------------------
#         # Rank 0: download / resolve
#         # ----------------------------
#         if rank == 0:
#             self.tokenizer = AutoTokenizer.from_pretrained(
#                 model_name,
#                 use_fast=True,
#                 trust_remote_code=True,
#                 cache_dir=cache_dir,
#                 local_files_only=False,
#             )

#         if is_dist:
#             dist.barrier()

#         # ----------------------------
#         # Other ranks: local-only load
#         # ----------------------------
#         if rank != 0:
#             self.tokenizer = AutoTokenizer.from_pretrained(
#                 model_name,
#                 use_fast=True,
#                 trust_remote_code=True,
#                 cache_dir=cache_dir,
#                 local_files_only=True,  # 🔒 critical
#             )

#         # ----------------------------
#         # Post-init normalization
#         # ----------------------------
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token

#         self.max_length = max_length

#     def encode(self, query: str, doc: str):
#         return self.tokenizer(
#             query,
#             doc,
#             padding=True,
#             truncation=True,
#             max_length=512,
#             return_tensors="pt",
#         )

#     def batch_encode(self, queries: list[str], docs: list[str]):
#         return self.tokenizer(
#             queries,
#             docs,
#             padding=True,
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt",
#         )



class RewardTokenizer:
    def __init__(
        self,
        model_name: str,
        max_length: int = 2048,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = max_length

    def encode(
        self,
        query: str,
        doc: str,
    ):
        return self.tokenizer(
            query, 
            doc, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt")


    def batch_encode(
        self,
        queries: list[str],
        docs: list[str],
    ):

        return self.tokenizer(
            queries, 
            docs, 
            padding=True, 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt")