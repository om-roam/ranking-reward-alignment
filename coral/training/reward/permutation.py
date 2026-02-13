import torch

def permute_per_query(inputs, labels, mask, device=None):
    B, N = mask.shape

    perm = torch.stack(
        [torch.randperm(N, device=device) for _ in range(B)]
    )

    def permute_bn(x):
        x = x.view(B, N, *x.shape[1:])
        x = x.gather(
            1,
            perm.unsqueeze(-1).expand(-1, -1, *x.shape[2:])
        )
        return x.view(B * N, *x.shape[2:])

    inputs = {k: permute_bn(v) for k, v in inputs.items()}
    labels = permute_bn(labels)

    mask_q = mask.gather(1, perm)
    mask_flat = mask_q.view(-1, 1)

    return inputs, labels, mask_q, mask_flat