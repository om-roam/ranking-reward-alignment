# CORAL Reward Model Framework (Ranking & Retrieval Alignment)

A modular PyTorch framework for training **reward models** for search/recommendation and using them to **align retrieval and ranking** with business objectives (e.g., relevance, revenue, and safety/constraints). Designed for production-scale experimentation with clean separation of data, losses, and distributed training.

## What this repo provides
- **Reward model training** for query–item scoring using preference/relative signals
- **Multi-objective calibration** utilities (e.g., gap/scale constraints, score distribution control)
- **Relative policy-style optimization** hooks (GRPO-style relative normalization) for ranking/retrieval alignment
- **Distributed training support** (DDP/FSDP), mixed precision, checkpointing
- **PEFT support** (LoRA/DoRA) for efficient finetuning of transformer backbones

## Core idea
Traditional supervised ranking optimizes proxy labels (clicks, purchases) and can drift from true business objectives. This framework trains a reward model that produces **calibrated, comparable scores** across queries and time, and supports **relative optimization** to reduce reward hacking and stabilize training.

## Contact
Omkar Patil @ informomp@gmail.com
