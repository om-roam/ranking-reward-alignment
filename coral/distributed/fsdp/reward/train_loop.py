import os
import torch
import torch.distributed as dist
import mlflow

from dataclasses import asdict
from coral.configs.reward_model.config_loader import load_reward_model_config, update_config
from training.reward.setup import training_init, build_trainer
from coral.model_checkpointing.checkpoint import load_checkpoint, save_checkpoint
from coral.distributed.utils.wrap import wrap_model


def setup_distributed():
    assert "RANK" in os.environ
    assert "WORLD_SIZE" in os.environ
    assert "LOCAL_RANK" in os.environ

    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # Safe + fast defaults
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    return torch.cuda.current_device()


def train_coral_fsdp(config):
    device = setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"FSDP world_size={world_size}", flush=True)

    config_path = "/code/coral/configs/reward_model.yaml"
    cfg = load_reward_model_config(config_path)
    update_config(cfg, config) 

    model, tokenizer, loader, optimizer, scheduler = training_init(cfg, device=device)
    model = wrap_model(model, cfg)

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        loader=loader,
        optimizer=optimizer,
        scheduler=scheduler
    )

    best_exact = 0.0

    if rank == 0:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "file:/logs/mlruns"))
        mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", "coral-fsdp"))

        mlflow.start_run(
            run_name=f"fsdp_ws{world_size}",
        )

        mlflow.log_params(asdict(config))
        mlflow.log_param("world_size", world_size)
        mlflow.log_param("backend", "fsdp")

    load_checkpoint(trainer, config)
    start_step = trainer.global_step

    for step, batch in enumerate(loader):
        if step < start_step:
            continue

        metrics = trainer.train_step(batch)

        if rank == 0:
            mlflow.log_metrics(
                {
                    "loss": metrics.loss,
                    "pm1_acc": metrics.pm1_acc,
                    "exact_acc": metrics.exact_acc,
                    "gamma": metrics.gamma_eff,
                },
                step=step,
            )
            best_exact = max(best_exact, metrics.exact_acc)

        if step % config.run.save_every == 0 and step > 0:
            save_checkpoint(trainer, step, cfg)

    if rank == 0:
        mlflow.log_metric("best_exact_acc", best_exact)
        mlflow.end_run()

    dist.barrier()
    dist.destroy_process_group()
