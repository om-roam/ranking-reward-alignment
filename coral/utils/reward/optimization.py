# utils/optimization.py
import torch
from transformers import get_cosine_schedule_with_warmup

def has_attr_path(obj, attr_path: str) -> bool:
    curr = obj
    for attr in attr_path.split("."):
        if not hasattr(curr, attr):
            return False
        curr = getattr(curr, attr)
    return True

def resolve_name_prefix(model, semantic_prefix: str | None) -> str | None:
    if semantic_prefix is None:
        return None

    if semantic_prefix == "encoder_layers":
        if has_attr_path(model, "encoder.backbone.encoder.layer"):
            return "encoder.backbone.encoder.layer"
        if has_attr_path(model, "encoder.backbone.layers"):
            return "encoder.backbone.layers"
        raise ValueError("Model does not expose transformer layers")

    if semantic_prefix == "coral_head":
        return "coral_head"

    raise ValueError(f"Unsupported name_prefix: {semantic_prefix}")


def build_optimizer(model, cfg):
    if cfg.optimizer.name.lower() != "adamw":
        raise ValueError(f"Unsupported optimizer {cfg.optimizer.name}")

    named_params = list(model.named_parameters())
    assigned_params = set()
    param_groups = []

    resolved_groups = []
    for group_name, group_cfg in cfg.optimizer.param_groups.items():
        prefix = resolve_name_prefix(model, group_cfg.name_prefix)
        resolved_groups.append((group_name, group_cfg, prefix))

    resolved_groups.sort(key=lambda x: x[2] is None)

    for group_name, group_cfg, prefix in resolved_groups:
        group_params = []

        for name, p in named_params:
            if not p.requires_grad or p in assigned_params:
                continue

            if prefix is None or name.startswith(prefix):
                group_params.append(p)
                assigned_params.add(p)

        if not group_params:
            raise ValueError(
                f"No parameters matched for optimizer group '{group_name}' "
                f"(prefix={prefix})"
            )

        param_groups.append(
            {
                "params": group_params,
                "lr": group_cfg.lr,
            }
        )

        print(
            f"[optimizer] group '{group_name}': "
            f"{len(group_params)} params | lr={group_cfg.lr}"
        )

    return torch.optim.AdamW(
        param_groups,
        weight_decay=cfg.optimizer.weight_decay,
        betas=tuple(cfg.optimizer.betas),
        eps=cfg.optimizer.eps,
    )


def build_scheduler(*, optimizer, cfg, num_training_steps):
    if cfg.scheduler.name == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(cfg.scheduler.warmup_ratio * num_training_steps),
            num_training_steps=num_training_steps,
        )
    else:
        raise ValueError(f"Unknown scheduler: {cfg.scheduler.name}")
