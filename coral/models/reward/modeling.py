import torch
import torch.nn as nn

from coral.models.reward.heads import QwenStyleCORALHead
from coral.models.reward.encoders import TransformerEncoder

class CoralModel(nn.Module):
    def __init__(
        self, 
        model_n, 
        torch_dtype=torch.float32, 
        trust_remote_code=True,
        k=3
    ):
        super().__init__()
        self.encoder = TransformerEncoder(model_n, torch_dtype=torch_dtype)
        self.coral_head = QwenStyleCORALHead(
            hidden_dim=self.encoder.hidden_size,
            k=k
        )
        self.log_T = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        out = self.encoder(**x)
        cls = out  
        logits = self.coral_head(cls)    
        T = torch.exp(self.log_T)
        logits = logits / T
        return logits

    def score_from_logits(self, logits):
        return torch.sigmoid(logits).sum(dim=-1)