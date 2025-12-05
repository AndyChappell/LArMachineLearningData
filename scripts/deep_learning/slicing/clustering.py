import torch
import torch.nn.functional as F


def oc_cluster(beta, embed, beta_threshold=0.1, top_k_fallback=1):
    """
    Perform Object Condensation clustering.
    Inputs:
        beta:  (N,) or (1, N, 1) tensor of predicted beta values
        embed: (N, D) or (1, N, D) tensor of embeddings
    Output:
        cluster_ids: (N,) tensor assigning each hit to a cluster ID
                     Background is not handled here (labels must determine that)
    """

    if beta.dim() == 3:
        beta = beta[0, :, 0]   # (N,)
        embed = embed[0]       # (N, D)

    N = beta.shape[0]
    beta = beta.sigmoid()     # convert logits → β ∈ [0,1]

    # 1. Identify condensation points (CP candidates)
    cp_mask = beta > beta_threshold
    cp_indices = torch.where(cp_mask)[0]

    # Fallback: if no CP found → take top-k β values
    if len(cp_indices) == 0:
        topk = min(top_k_fallback, N)
        cp_indices = torch.topk(beta, topk).indices

    cp_embeds = embed[cp_indices]          # (#CP, D)

    # 2. Compute distance of every hit to every CP
    dist = torch.cdist(embed.unsqueeze(0), cp_embeds.unsqueeze(0)).squeeze(0)  # (N, #CP)

    # 3. Assign each hit to nearest CP
    cluster_ids = torch.argmin(dist, dim=1)   # (N,)

    return cluster_ids, cp_indices


import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_cluster_assignments(beta, embed, slice_labels, mask, beta_threshold=0.5):
    """
    Convert OC outputs into cluster assignments.

    Args:
        beta:  (B, N, 1) predicted condensation score
        embed: (B, N, D) embedding vectors
        slice_labels: (B, N) ground-truth instance labels; used only to mask out background hits
        mask:  (B, N) boolean mask for valid hits
        beta_threshold: threshold above which hits become condensation points

    Returns:
        cluster_ids: (B, N) long tensor of cluster assignments, -1 for background/padded
    """
    B, N, D = embed.shape
    device = embed.device

    cluster_assignments = torch.full((B, N), -1, dtype=torch.long, device=device)

    for b in range(B):

        valid = mask[b]                                      # (N)
        beta_b = beta[b, valid, 0]                          # (Nv)
        embed_b = embed[b, valid]                           # (Nv, D)

        # background hits must get cluster -1
        slice_b = slice_labels[b, valid]                    # (Nv)
        bg_mask = slice_b == -1

        # Identify condensation points
        condensation_mask = beta_b > beta_threshold         # (Nv)
        condensation_idxs = torch.where(condensation_mask)[0]

        if len(condensation_idxs) == 0:
            # fallback: pick top-1 beta
            condensation_idxs = torch.tensor(
                [torch.argmax(beta_b)], device=device
            )

        # Compute distance of each hit to each condensation point
        cp_embeds = embed_b[condensation_idxs]              # (Nc, D)

        # (Nv, D) -> (Nv, 1, D)
        hits_expand = embed_b.unsqueeze(1)                  # (Nv,1,D)
        # (Nc, D) -> (1, Nc, D)
        cp_expand = cp_embeds.unsqueeze(0)                  # (1,Nc,D)

        # L2 distance
        dists = torch.norm(hits_expand - cp_expand, dim=-1) # (Nv, Nc)

        # Take nearest condensation point
        assigned = torch.argmin(dists, dim=1)               # (Nv)

        # Background hits assigned to -1
        assigned[bg_mask] = -1

        # Remap condensation point indices to unique cluster IDs
        # condensation_idxs might be [4,19,33] → cluster ids become [0,1,2]
        # But assigned[] currently refers to index in condensation list, so OK.

        # Fill results back into padded structure
        cluster_assignments[b][valid] = assigned

    return cluster_assignments
