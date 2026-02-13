import torch
import torch.nn as nn

from typing import Dict
from coral.models.reward.utils import get_transformer_layers


class DoRALinear(nn.Module):
    """
    DDP-safe DoRA wrapper
    """
    def __init__(self, orig_linear: nn.Linear, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        assert isinstance(orig_linear, nn.Linear)
        self._device = orig_linear.weight.device 
        self._dtype = orig_linear.weight.dtype
        self.in_features = orig_linear.in_features
        self.out_features = orig_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(1, r)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.weight = orig_linear.weight  
        self.bias = orig_linear.bias      

        with torch.no_grad():
            init_mag = torch.norm(self.weight.data, dim=1, keepdim=True)  
            init_mag = init_mag.clamp(min=1e-6)
        self.magnitude = nn.Parameter(init_mag.to(device=self._device, dtype=self._dtype))  

        if r > 0:
            self.A = nn.Parameter(torch.randn(r, self.in_features, device=self._device, dtype=self._dtype) * 0.01)  # small init
            self.B = nn.Parameter(torch.zeros(self.out_features, r, device=self._device, dtype=self._dtype))       # zero init
        else:
            self.A = None
            self.B = None

    def compute_direction(self) -> torch.Tensor:
        w = self.weight
        row_norm = torch.norm(w, dim=1, keepdim=True).clamp(min=1e-12)
        direction = w / row_norm
        return direction

    def forward(self, x: torch.Tensor):
        if self.r is None or self.r == 0:
            return torch.nn.functional.linear(x, self.weight, self.bias)

        delta_d = (self.B @ self.A) * self.scaling  
        direction = self.compute_direction()  
        d_new = direction + delta_d
        d_row_norm = torch.norm(d_new, dim=1, keepdim=True).clamp(min=1e-12)
        d_unit = d_new / d_row_norm  
        W_eff = self.magnitude * d_unit  
        return torch.nn.functional.linear(x, W_eff, self.bias)

    def adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        sd = {}
        if self.A is not None:
            sd["A"] = self.A.detach().cpu()
            sd["B"] = self.B.detach().cpu()
        sd["magnitude"] = self.magnitude.detach().cpu()
        return sd

    def load_adapter_state_dict(self, sd: Dict[str, torch.Tensor], map_location=None):
        if "A" in sd and self.A is not None:
            self.A.data.copy_(sd["A"].to(self.A.device) if map_location is None else sd["A"].to(map_location))
            self.B.data.copy_(sd["B"].to(self.B.device) if map_location is None else sd["B"].to(map_location))
        if "magnitude" in sd:
            self.magnitude.data.copy_(sd["magnitude"].to(self.magnitude.device) if map_location is None else sd["magnitude"].to(map_location))

    def merge_to_linear(self) -> nn.Linear:
        if self.r is None or self.r == 0:
            new_lin = nn.Linear(self.in_features, self.out_features, bias=(self.bias is not None))
            new_lin.weight.data.copy_(self.weight.data)
            if self.bias is not None:
                new_lin.bias.data.copy_(self.bias.data)
            return new_lin

        delta_d = (self.B @ self.A) * self.scaling
        direction = self.compute_direction()
        d_new = direction + delta_d
        d_row_norm = torch.norm(d_new, dim=1, keepdim=True).clamp(min=1e-12)
        d_unit = d_new / d_row_norm
        W_merged = (self.magnitude * d_unit).to(self.weight.dtype)

        new_lin = nn.Linear(self.in_features, self.out_features, bias=(self.bias is not None))
        new_lin.weight.data.copy_(W_merged)
        if self.bias is not None:
            new_lin.bias.data.copy_(self.bias.data)
        return new_lin
    

def apply_dora_to_transformer(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
    target_module_names=None,
):
    if target_module_names is None:
        target_module_names = (
            # RoBERTa / BERT 
            "query",
            "key",
            "value",
            "dense",

            # Qwen
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )

    for name, child in list(model.named_children()):
        apply_dora_to_transformer(
            child,
            r=r,
            alpha=alpha,
            dropout=dropout,
            target_module_names=target_module_names,
        )
        if isinstance(child, nn.Linear):
            lname = name.lower()
            if any(t in lname for t in target_module_names):
                dora = DoRALinear(child, r=r, alpha=alpha, dropout=dropout)
                setattr(model, name, dora)

def set_trainable_adapters_only(model: nn.Module):
    for n, p in model.named_parameters():
        if ".A" in n or ".B" in n or ".magnitude" in n or n.startswith('coral_head'):
            p.requires_grad = True
        else:
            p.requires_grad = False

def apply_dora_last_n_layers(
    model: nn.Module,
    n_layers: int,
    *,
    r: int = 96,
    alpha: int = 96,
    dropout: float = 0.01,
):
    """
    Apply DoRA ONLY to the last n transformer layers.
    CORAL head is left untouched.
    """
    layers = get_transformer_layers(model)
    total = len(layers)
    start = max(0, total - n_layers)
    sample = layers[0]

    if hasattr(sample, "self_attn") and hasattr(sample.self_attn, "q_proj"):
        # Qwen-style
        target_names = (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )
        arch = "qwen"
    else:
        # RoBERTa / BERT-style
        target_names = (
            "query",
            "key",
            "value",
            "dense",
        )
        arch = "roberta"

    print(f"[DoRA] arch={arch} | layers {start}..{total - 1}")


    for layer_idx in range(start, total):
        layer = layers[layer_idx]

        for name, module in list(layer.named_modules()):
            if not isinstance(module, nn.Linear):
                continue

            lname = name.lower()
            if not any(t in lname for t in target_names):
                continue

            parent = layer
            *path, child_name = name.split(".")
            for p in path:
                parent = getattr(parent, p)

            if "coral" in child_name.lower():
                continue

            setattr(
                parent,
                child_name,
                DoRALinear(
                    module,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                ),
            )

def merge_all_dora_to_linear(model: nn.Module):
    """
    Recursively replace all DoRALinear modules with merged nn.Linear modules.
    """
    for name, child in list(model.named_children()):
        merge_all_dora_to_linear(child)
        if isinstance(child, DoRALinear):
            merged = child.merge_to_linear()
            merged = merged.to(
                device=child.weight.device,
                dtype=child.weight.dtype,
            )
            setattr(model, name, merged)