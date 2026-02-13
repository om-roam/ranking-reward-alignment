# training/reward_model/training_setup.py

import argparse
import yaml
import torch
from pathlib import Path

from models.reward.modeling import CoralModel
from models.reward.tokenizer import RewardTokenizer
from data.reward.dataloader import data_loader
from coral.utils.reward.optimization import build_optimizer, build_scheduler
from coral.training.reward.trainer import RewardModelTrainer
from training.reward.freeze import freeze_layers_encoder
from coral.models.peft.dora import apply_dora_last_n_layers, set_trainable_adapters_only

def training_init(cfg, device=None):
    model = CoralModel(
        cfg.model.name,
        torch_dtype=cfg.model.torch_dtype,
        trust_remote_code=cfg.model.trust_remote_code,
    ).to(device)

    if cfg.training.peft:
        apply_dora_last_n_layers(model.encoder, cfg.training.num_layers)
        set_trainable_adapters_only(model)
    elif cfg.training.num_layers > 0:
        freeze_layers_encoder(model, n_layers=cfg.training.num_layers)
    tokenizer = RewardTokenizer(
        cfg.model.name,
        max_length=cfg.data.max_length,
    )

    loader = data_loader(
        cfg.data.train_path,
        batch_size=cfg.training.batch_size,
    )

    optimizer = build_optimizer(model, cfg)
    num_update_steps = len(loader) // cfg.training.accum_steps
    
    scheduler = build_scheduler(
        optimizer=optimizer,
        cfg=cfg,
        num_training_steps=num_update_steps,
    )

    return model, tokenizer, loader, optimizer, scheduler

def build_trainer(*, model, tokenizer, loader, optimizer, scheduler, cfg, device):
    trainer = RewardModelTrainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    tokenizer=tokenizer,
    loader=loader,
    device=device,

    num_classes=cfg.data.num_classes,

    accum_steps=cfg.training.accum_steps,
    block_size=cfg.training.block_size,

    gamma_init=cfg.loss.gamma_init,
    gamma=cfg.loss.gamma,
    mono_coef=cfg.loss.mono_coef,
    gap_coef=cfg.loss.gap_coef,
    scale_coef=cfg.loss.scale_coef,
    min_logit_scale=cfg.loss.min_logit_scale,
    focal_warmup_steps=cfg.training.focal_warmup_steps,
    lambda_weight=cfg.loss.lambda_weight,
    use_ndcg=cfg.loss.use_ndcg,

    save_path=cfg.run.save_path,
    log_every=cfg.run.log_every,
    save_every=cfg.run.save_every,
    st_from=cfg.run.start_from_step,
    )
    return trainer