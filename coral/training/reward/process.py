import torch

def batch_process(batch, label_pad=-1):
    B = len(batch["queries"])
    N_max = max(len(docs) for docs in batch["docs"])

    candidate_mask = torch.zeros(B, N_max, dtype=torch.bool)
    labels = torch.full((B, N_max), label_pad, dtype=torch.long)

    queries_batched = []
    docs_batched = []

    for i in range(B):
        docs_i = batch["docs"][i]
        labels_i = batch["labels"][i]
        n_i = len(docs_i)

        queries_batched.append(
            [batch["queries"][i]] * n_i + ["<pad>"] * (N_max - n_i)
        )
        docs_i_unwrapped = [
            d[0] if isinstance(d, list) else d for d in docs_i
        ]
        docs_batched.append(
            docs_i_unwrapped + ["<pad>"] * (N_max - n_i)
        )

        candidate_mask[i, :n_i] = True
        labels[i, :n_i] = torch.tensor(labels_i, dtype=torch.long)

    queries_flat = [q for q_row in queries_batched for q in q_row]
    docs_flat = [d for d_row in docs_batched for d in d_row]

    return queries_flat, docs_flat, labels, candidate_mask