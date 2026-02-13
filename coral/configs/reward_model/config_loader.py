import yaml
import torch
from pathlib import Path
from dataclasses import replace

from configs.reward_model.config import (
    RewardModelConfig,
    ModelConfig,
    RunConfig,
    DataConfig,
    TrainingConfig,
    OptimizerConfig,
    OptimizerParamGroupConfig,
    SchedulerConfig,
    LossConfig,
    MixedPrecisionConfig,
    DistributedConfig
)


def load_reward_model_config(path: str | Path) -> RewardModelConfig:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    return RewardModelConfig(
        model=ModelConfig(
            name=cfg["model"]["name"],
            revision=cfg["model"]["revision"],
            torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
            trust_remote_code=cfg["model"]["trust_remote_code"],
        ),
        run=RunConfig(**cfg["run"]),
        data=DataConfig(**cfg["data"]),
        training=TrainingConfig(**cfg["training"]),
        optimizer=OptimizerConfig(
            name=cfg["optimizer"]["name"],
            weight_decay=cfg["optimizer"]["weight_decay"],
            param_groups={
                k: OptimizerParamGroupConfig(**v)
                for k, v in cfg["optimizer"]["param_groups"].items()
            },
            betas=cfg["optimizer"]["betas"],
            eps=cfg["optimizer"]["eps"],
            max_grad_norm=cfg["optimizer"]["max_grad_norm"],
        ),
        scheduler=SchedulerConfig(**cfg["scheduler"]),
        loss=LossConfig(**cfg["loss"]),
        mixed_precision=MixedPrecisionConfig(
            enabled=cfg["mixed_precision"]["enabled"],
            dtype=getattr(torch, cfg["mixed_precision"]["dtype"]),
        ),
        distributed=DistributedConfig(**cfg["distributed"]),  # <-- added this
    )


def update_config(base_config, tunable_config):
    for section_name, section_values in tunable_config.items():
        section_obj = getattr(base_config, section_name)
        if isinstance(section_values, dict):
            if section_name == "optimizer" and "param_groups" in section_values:
                pg_updates = section_values.pop("param_groups")
                for k, v in pg_updates.items():
                    pg_obj = section_obj.param_groups[k]
                    section_obj.param_groups[k] = replace(pg_obj, **v)
            section_obj = replace(section_obj, **section_values)
        setattr(base_config, section_name, section_obj)
