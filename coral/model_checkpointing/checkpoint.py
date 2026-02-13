import os
import re
import torch
import torch.distributed as dist

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    ShardedStateDictConfig,
)

_STEP_RE = re.compile(r"step_(\d+)")


def find_latest_step(save_path: str) -> int | None:
    if not os.path.exists(save_path):
        return None

    steps = []
    for name in os.listdir(save_path):
        m = _STEP_RE.fullmatch(name)
        if m:
            steps.append(int(m.group(1)))

    return max(steps) if steps else None


def save_checkpoint(trainer, step: int, config):
    rank = dist.get_rank()

    path = os.path.join(config.run.save_path, f"step_{step}")
    os.makedirs(path, exist_ok=True)

    if config.distributed.backend == "ddp":
        if rank == 0:
            torch.save(
                {
                    "model": trainer.model.module.state_dict(),
                    "optimizer": trainer.optimizer.state_dict(),
                    "step": step,
                },
                os.path.join(path, "ddp.pt"),
            )


    elif config.distributed.backend == "fsdp":
        with FSDP.state_dict_type(
            trainer.model,
            StateDictType.SHARDED_STATE_DICT,
            ShardedStateDictConfig(offload_to_cpu=True),
        ):
            model_state = trainer.model.state_dict()

        torch.save(
            model_state,
            os.path.join(path, f"model_rank{rank}.pt"),
        )

        optim_state = FSDP.optim_state_dict(
            trainer.model,
            trainer.optimizer,
        )

        torch.save(
            optim_state,
            os.path.join(path, f"optim_rank{rank}.pt"),
        )

    else:
        raise ValueError(f"Unknown backend: {config.distributed.backend}")


def load_checkpoint(trainer, config):
    rank = dist.get_rank()

    step = config.run.start_from_step

    if step == 0:
        if rank == 0:
            latest = find_latest_step(config.run.save_path)
            if latest is None:
                print("No checkpoint found, starting fresh")
                step = 0
            else:
                step = latest

        step_list = [step]
        dist.broadcast_object_list(step_list, src=0)
        step = step_list[0]

        if step == 0:
            return

    path = os.path.join(config.run.save_path, f"step_{step}")

    if rank == 0:
        print(f"Resuming from checkpoint: {path}")

    if config.distributed.backend == "ddp":
        if rank == 0:
            ckpt = torch.load(
                os.path.join(path, "ddp.pt"),
                map_location="cpu",
            )
        else:
            ckpt = None

        ckpt_list = [ckpt]
        dist.broadcast_object_list(ckpt_list, src=0)
        ckpt = ckpt_list[0]

        trainer.model.module.load_state_dict(ckpt["model"])
        trainer.optimizer.load_state_dict(ckpt["optimizer"])
        trainer.global_step = ckpt.get("step", step)

    elif config.distributed.backend == "fsdp":
        dist.barrier()
        model_state = torch.load(
            os.path.join(path, f"model_rank{rank}.pt"),
            map_location="cpu",
        )

        with FSDP.state_dict_type(
            trainer.model,
            StateDictType.SHARDED_STATE_DICT,
            ShardedStateDictConfig(offload_to_cpu=True),
        ):
            trainer.model.load_state_dict(model_state)

        optim_state = torch.load(
            os.path.join(path, f"optim_rank{rank}.pt"),
            map_location="cpu",
        )

        optim_state = FSDP.optim_state_dict_to_load(
            trainer.model,
            trainer.optimizer,
            optim_state,
        )

        trainer.optimizer.load_state_dict(optim_state)

        trainer.global_step = step
        dist.barrier()

    else:
        raise ValueError(f"Unknown backend: {config.distributed.backend}")
