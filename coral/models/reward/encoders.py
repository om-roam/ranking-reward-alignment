import torch
import torch.nn as nn
from transformers import AutoModel

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        pooling: str = "last",
        torch_dtype: torch.dtype | None = torch.float32,
    ):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            return_dict=True
        )

        self.pooling = pooling
        self.hidden_size = self.backbone.config.hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            pooled: (batch, hidden_size)
        """

        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden = outputs.last_hidden_state

        if self.pooling == "last":
            pooled = hidden[:, -1, :]

        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return pooled