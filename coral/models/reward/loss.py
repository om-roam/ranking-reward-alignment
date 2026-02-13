import torch
import torch.nn.functional as F

def focal_coral_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    gamma: float,
    mono_coef: float,
    gap_coef: float,
    scale_coef: float,
    min_logit_scale: float,
):
    """
    Focal CORAL loss with monotonicity and regularization.

    Args:
        logits:          (B, K-1)
        labels:          (B, K-1) binary CORAL targets
        candidate_mask:  (B,) or (B, 1) or (B, K-1)
        gamma:           focal gamma
        mono_coef:       monotonicity coefficient
        gap_coef:        threshold gap coefficient
        scale_coef:      logit scale coefficient
        min_logit_scale: minimum allowed |logit|
        accum_steps:     gradient accumulation steps
        num_blocks:      microbatch blocks

    Returns:
        loss: scalar tensor
        metrics: dict[str, tensor] (detached)
    """

    probs = torch.sigmoid(logits)

    pt = probs * labels + (1.0 - probs) * (1.0 - labels)
    focal_weight = (1.0 - pt).pow(gamma)

    bce = F.binary_cross_entropy_with_logits(
        logits,
        labels.float(),
        reduction="none",
    )

    focal_loss = focal_weight * bce

    if candidate_mask.dim() == 1:
        candidate_mask = candidate_mask.unsqueeze(-1)

    focal_loss = focal_loss * candidate_mask.float()
    coral_loss = focal_loss.mean()

    mono_loss = torch.relu(
        logits[:, 1:] - logits[:, :-1]
    ).mean()

    gap_loss = torch.relu(
        0.5 - (logits[:, 1:] - logits[:, :-1])
    ).mean()

    scale_loss = torch.relu(
        min_logit_scale - logits.abs()
    ).mean()

    total_loss = (
        coral_loss
        + mono_coef * mono_loss
        + gap_coef * gap_loss
        + scale_coef * scale_loss
    )

    metrics = {
        "loss/coral": coral_loss.detach(),
        "loss/mono": mono_loss.detach(),
        "loss/gap": gap_loss.detach(),
        "loss/scale": scale_loss.detach(),
        "loss/total": total_loss.detach(),
    }

    return total_loss, metrics
