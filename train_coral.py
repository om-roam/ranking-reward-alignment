import os
import socket
import torch
import torch.distributed as dist

from coral.configs.reward_model.config_loader import (
    load_reward_model_config,
)
from training.reward.setup import (
    training_init,
    build_trainer,
)
from coral.model_checkpointing.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from coral.distributed.utils.wrap import setup_fsdp_and_checkpointing
from coral.distributed.utils.helpers import setup_distributed, cleanup_distributed
from coral.models.reward.utils import torch_model_setup

def train_coral_fsdp():
    device = setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(
        f"[START] host={socket.gethostname()} "
        f"rank={rank}/{world_size}",
        flush=True,
    )

    cfg = load_reward_model_config("/code/coral/configs/reward_model.yaml")
    model, tokenizer, loader, optimizer, scheduler = training_init(
        cfg,
        device=device,
    )
    model = setup_fsdp_and_checkpointing(model)
    torch_model_setup(model)
    
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        cfg=cfg,
        device=device
    )

    load_checkpoint(trainer, cfg)
    start_step = trainer.global_step
    for step, batch in enumerate(loader):
        if step < start_step:
            continue

        metrics = trainer.step(batch)

        if rank == 0:
            print(
                f"[step {step}] loss={metrics['loss']:.4f}",
                flush=True,
            )

        if step > 0 and step % cfg.run.save_every == 0:
            save_checkpoint(trainer, step, cfg)

    cleanup_distributed()
    print(f"[EXIT] rank={rank}", flush=True)


if __name__ == "__main__":
    train_coral_fsdp()
