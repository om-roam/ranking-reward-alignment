import torch

def coral_targets(y, num_classes=4):
    y = y.view(-1, 1)
    thresholds = torch.arange(0, num_classes-1).view(1, -1).to(y.device)
    return (y > thresholds).float()