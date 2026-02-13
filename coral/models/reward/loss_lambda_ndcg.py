import torch

def lambda_ndcg_batch(
    scores: torch.Tensor,   
    rels: torch.Tensor,     
    sigmoid_scale: float = 1.0,
    eps: float = 1e-12,
):
    B, N = scores.shape
    device = scores.device

    s = scores.detach() 
    gains = (2.0 ** rels) - 1.0  

    sorted_s, order = torch.sort(s, dim=1, descending=True)
    ranks = torch.arange(1, N + 1, device=device).float()
    discounts = 1.0 / torch.log2(1.0 + ranks)  
    discounts = discounts.unsqueeze(0) 

    rels_sorted = torch.gather(gains, 1, order)        
    dcg = (rels_sorted * discounts).sum(dim=1)        

    ideal_order = torch.argsort(rels, dim=1, descending=True)
    ideal_sorted = torch.gather(gains, 1, ideal_order)
    idcg = (ideal_sorted * discounts).sum(dim=1).clamp_min(eps)

    ndcg = (dcg / idcg).detach()

    g_i = gains.unsqueeze(2)    
    g_j = gains.unsqueeze(1)     
    rel_i = rels.unsqueeze(2)
    rel_j = rels.unsqueeze(1)
    pos_mask = (rel_i > rel_j).float()  

    s_i = s.unsqueeze(2)
    s_j = s.unsqueeze(1)
    rho = torch.sigmoid(sigmoid_scale * (s_j - s_i)) 

    ranks_full = torch.arange(1, N+1, device=device).float()
    disc_full = 1.0 / torch.log2(1.0 + ranks_full)
    d_i = disc_full.view(1, N, 1)
    d_j = disc_full.view(1, 1, N)
    delta_d = d_j - d_i

    delta_ndcg = torch.abs((g_i - g_j) * delta_d) / idcg.view(B, 1, 1)

    lambda_mat = rho * delta_ndcg * pos_mask  
    lambda_i = lambda_mat.sum(2) - lambda_mat.sum(1)  

    return lambda_i, ndcg


def compute_lambda_ndcg_loss_from_logits(
    *,
    logits: torch.Tensor,   # [B, N, C]
    labels: torch.Tensor,   # [B, N, C]
    mask: torch.Tensor,     # [B, N] (True = valid doc)
):
    """
    Batch-compatible LambdaNDCG surrogate loss.
    - No query mixing
    - Mask handled correctly
    - Works for B >= 1
    """
    B, N, C = logits.shape
    device = logits.device

    # ---- Expected ordinal score (CORAL expectation) ----
    scores = torch.sigmoid(logits).sum(dim=2)   # [B, N]
    rels = labels.sum(dim=2)                    # [B, N]

    # ---- Apply mask BEFORE LambdaNDCG ----
    mask = mask.bool()
    scores = scores.masked_fill(~mask, -1e9)
    rels = rels * mask.float()

    # ---- LambdaRank gradients ----
    lambda_grads, _ = lambda_ndcg_batch(
        scores=scores,
        rels=rels,
    )  # [B, N]

    # ---- Surrogate loss (LambdaRank trick) ----
    # Important: use ORIGINAL scores (not detached)
    loss_per_query = (scores * lambda_grads).sum(dim=1)  # [B]

    return loss_per_query.mean()
