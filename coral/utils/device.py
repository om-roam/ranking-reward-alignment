import torch

def move_to_device(batch, device='cpu'):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)
    return batch