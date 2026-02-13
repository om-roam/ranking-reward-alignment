import socket
from dataclasses import asdict

import mlflow
import torch.distributed as dist
from ray import train as ray_train

from coral.configs.reward_model.config_loader import (
    load_reward_model_config,
    update_config,
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
from coral.distributed.utils.helpers import (
    setup_distributed,
    cleanup_distributed,
)
from coral.models.reward.utils import torch_model_setup


def train_coral_fsdp_core(config, report_fn=None):
    device = setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(
        f"[START] host={socket.gethostname()} "
        f"rank={rank}/{world_size}",
        flush=True,
    )

    cfg = load_reward_model_config("/code/coral/configs/reward_model.yaml")
    update_config(cfg, config)

    if rank == 0:
        mlflow.set_experiment("coral_reward_model")

        run_name = None
        try:
            run_name = ray_train.get_context().get_trial_name()
        except Exception:
            pass

        mlflow.start_run(run_name=run_name)
        mlflow.log_params(asdict(cfg))
        mlflow.log_param(
            "effective_batch_size",
            cfg.train.batch_size * cfg.train.accum_steps * world_size,
        )
        mlflow.log_param(
            "trainable_params",
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )

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
        device=device,
    )

    load_checkpoint(trainer, cfg)
    start_step = trainer.global_step

    for step, batch in enumerate(loader):
        if step < start_step:
            continue

        metrics = trainer.step(batch)

        if rank == 0 and metrics["stepped"]:
            global_step = trainer.global_step

            print(
                f"[step {global_step}] "
                f"loss={metrics['loss']:.4f}",
                flush=True,
            )

            mlflow.log_metrics(
                {
                    "loss": metrics["loss"],
                    "exact_acc": metrics.get("exact_acc", 0.0),
                    "gamma": metrics.get("gamma", 0.0),
                },
                step=global_step,
            )
            if report_fn:
                report_fn(
                    loss=metrics["loss"],
                    exact_acc=metrics.get("exact_acc", 0.0),
                    global_step=global_step,
                )

        if (
            step > 0
            and step % cfg.run.save_every == 0
            and rank == 0
        ):
            save_checkpoint(trainer, step, cfg)

    if rank == 0:
        mlflow.end_run()

    cleanup_distributed()
    print(f"[EXIT] rank={rank}", flush=True)
