from contextlib import contextmanager, nullcontext

import torch
import torch.nn.functional as F
import torch.distributed as dist

from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

from coral.models.reward.loss import focal_coral_loss
from coral.models.reward.loss_lambda_ndcg import (
    compute_lambda_ndcg_loss_from_logits,
)
from coral.training.reward.inputs import get_inputs
from coral.training.reward.permutation import permute_per_query

class RewardModelTrainer:
    def __init__(
        self,
        *,
        model,
        optimizer,
        scheduler,
        tokenizer,
        loader,
        device,
        save_path,
        num_classes,
        accum_steps=2,
        block_size=16,
        gamma_init=1.05,
        gamma=2.0,
        mono_coef=0.1,
        gap_coef=0.05,
        scale_coef=0.01,
        min_logit_scale=0.2,
        focal_warmup_steps=300,
        lambda_weight=0.1,
        use_ndcg=False,
        st_from=0,
        save_every=0,
        log_every=2,
        use_amp=False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_amp = use_amp

        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        self.tokenizer = tokenizer
        self.loader = loader
        self.device = device
        self.save_path = save_path
        self.num_classes = num_classes

        self.accum_steps = accum_steps
        self.block_size = block_size

        self.gamma_init = gamma_init
        self.gamma = gamma
        self.mono_coef = mono_coef
        self.gap_coef = gap_coef
        self.scale_coef = scale_coef
        self.min_logit_scale = min_logit_scale
        self.focal_warmup_steps = focal_warmup_steps
        self.lambda_weight = lambda_weight
        self.use_ndcg = use_ndcg

        self.global_step = 0
        self.grad_counter = 0
        self.st_from = st_from
        self.save_every = save_every
        self.log_every = log_every

        self.optimizer.zero_grad(set_to_none=True)

        # torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

    # Disable synchronisation for grad accumulation
    @contextmanager
    def _maybe_no_sync(self):
        if (
            self.accum_steps > 1
            and hasattr(self.model, "no_sync")
            and self.grad_counter < self.accum_steps - 1
        ):
            with self.model.no_sync():
                yield
        else:
            yield

    def optimizer_step(self):
        if self.use_amp:
            self.scaler.unscale_(self.optimizer)

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

    @torch.no_grad()
    def log_metrics(self, inputs, coral_labels, avg_loss, gamma_eff):
        logits = self.model(inputs)

        pred = (torch.sigmoid(logits) > 0.5).sum(dim=1)
        true = coral_labels.sum(dim=1)

        acc_pm1 = (torch.abs(pred - true) <= 1).float().mean()
        acc_exact = (pred == true).float().mean()

        print(
            f"step {self.global_step} | "
            f"loss {avg_loss:.4f} | "
            f"±1 acc {acc_pm1:.3f} | "
            f"exact acc {acc_exact:.3f} | "
            f"gamma {gamma_eff:.2f}",
            flush=True,
        )

    def step(self, batch):
        self.model.train()

        inputs, coral_labels, mask = get_inputs(
            batch,
            tokenizer=self.tokenizer,
            device=self.device,
            num_classes=self.num_classes,
        )

        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
        coral_labels = coral_labels.to(self.device, non_blocking=True)
        mask = mask.to(self.device, non_blocking=True)

        B, N = mask.shape
        _, C = coral_labels.shape

        inputs, coral_labels, mask_q, mask_flat = \
            permute_per_query(inputs, coral_labels, mask, device=self.device)

        gamma_eff = (
            self.gamma_init
            if self.global_step < self.focal_warmup_steps
            else self.gamma
        )

        with self._maybe_no_sync():
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                logits = self.model(inputs)

                coral_loss, _ = focal_coral_loss(
                    logits=logits,
                    labels=coral_labels,
                    candidate_mask=mask_flat,
                    gamma=gamma_eff,
                    mono_coef=self.mono_coef,
                    gap_coef=self.gap_coef,
                    scale_coef=self.scale_coef,
                    min_logit_scale=self.min_logit_scale,
                )

                loss = coral_loss

                if self.use_ndcg:
                    loss = loss + self.lambda_weight * compute_lambda_ndcg_loss_from_logits(
                        logits=logits.view(B, N, C),
                        labels=coral_labels.view(B, N, C),
                        mask=mask_q,
                    )

                loss = loss / self.accum_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

        self.grad_counter += 1
        stepped = False

        if self.grad_counter == self.accum_steps:
            self.optimizer_step()
            self.grad_counter = 0
            self.global_step += 1
            stepped = True

            avg_loss = loss.detach().item() * self.accum_steps

            self.log_metrics(
                inputs=inputs,
                coral_labels=coral_labels,
                avg_loss=avg_loss,
                gamma_eff=gamma_eff,
            )

        return {
            "loss": loss.detach().item() * self.accum_steps,
            "gamma": gamma_eff,
            "stepped": stepped,
            "global_step": self.global_step,
        }
    
    @torch.no_grad()
    def forward_inference(self, batch):
        self.model.eval()
        inputs, _, mask = get_inputs(
            batch,
            tokenizer=self.tokenizer,
            device=self.device,
            num_classes=self.num_classes,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        mask = mask.to(self.device)
        B, N = mask.shape

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = self.model(inputs)
        logits = logits.view(B, N, -1)
        labels = (torch.sigmoid(logits) > 0.5).sum(dim=1)
        
        return {
            "logits": logits,
            "mask": mask,
            "labels": labels
        }
