from dataclasses import dataclass
from typing import Optional, Dict, List
import torch

# ----------------------------
# Model
# ----------------------------
@dataclass
class ModelConfig:
    name: str
    revision: Optional[str]
    torch_dtype: torch.dtype
    trust_remote_code: bool


# ----------------------------
# Run
# ----------------------------
@dataclass
class RunConfig:
    save_path: str
    log_path: str
    start_from_step: int
    log_every: int
    save_every: int
    seed: int


# ----------------------------
# Data
# ----------------------------
@dataclass
class DataConfig:
    num_classes: int
    max_length: int
    shuffle_docs_per_query: bool
    train_path: str
    eval_path: str


# ----------------------------
# Training
# ----------------------------
@dataclass
class TrainingConfig:
    device: str
    batch_size: int
    accum_steps: int
    block_size: int
    focal_warmup_steps: int
    num_layers: int
    peft: bool


# ----------------------------
# Optimizer
# ----------------------------
@dataclass
class OptimizerParamGroupConfig:
    lr: float
    name_prefix: Optional[str]


@dataclass
class OptimizerConfig:
    name: str
    weight_decay: float
    param_groups: Dict[str, OptimizerParamGroupConfig]
    betas: List[float]
    eps: float
    max_grad_norm: float


# ----------------------------
# Scheduler
# ----------------------------
@dataclass
class SchedulerConfig:
    name: str
    warmup_ratio: float


# ----------------------------
# Loss
# ----------------------------
@dataclass
class LossConfig:
    name: str
    gamma_init: float
    gamma: float
    mono_coef: float
    gap_coef: float
    scale_coef: float
    min_logit_scale: float
    use_ndcg: bool
    lambda_weight: float


# ----------------------------
# Mixed Precision
# ----------------------------
@dataclass
class MixedPrecisionConfig:
    enabled: bool
    dtype: torch.dtype

# ----------------------------
# Distributed
# ----------------------------
@dataclass
class DistributedConfig:
    backend: str  # e.g., "fsdp", "nccl", etc.

# ----------------------------
# Root config
# ----------------------------
@dataclass
class RewardModelConfig:
    model: ModelConfig
    run: RunConfig
    data: DataConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    loss: LossConfig
    mixed_precision: MixedPrecisionConfig
    distributed: DistributedConfig



