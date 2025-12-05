import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectCondensationLoss(nn.Module):
    def __init__(self, attraction_weight=1.0, repulsion_weight=1.0, beta_positive_weight=1.0, beta_negative_weight=0.5):
        super().__init__()
        self.attraction_weight = attraction_weight
        self.repulsion_weight = repulsion_weight
        self.beta_positive_weight = beta_positive_weight
        self.beta_negative_weight = beta_negative_weight

    def forward(self, pred, slice_id, is_cp):
        """
        pred: {"beta": (B, N, 1), "embed": (B, N, D)}
        slice_id: (B, N) long, -1 = background
        is_cp: (B, N) bool (1 = condensation point)

        Returns: scalar loss
        """
        beta = pred["beta"].squeeze(-1)      # (B, N)
        embed = pred["embed"]                # (B, N, D)
        B, N, D = embed.shape

        total_loss = 0.0
        count = 0

        for b in range(B):
            sid = slice_id[b]               # (N)
            cp_mask = is_cp[b].bool()       # (N)
            beta_b = beta[b]                # (N,)
            emb_b = embed[b]                # (N, D)

            valid = sid >= 0                # ignore background hits
            if valid.sum() == 0:
                continue

            # ---------------------------------------
            # 1. Identify condensation points
            # ---------------------------------------
            cp_idx = torch.where(cp_mask & valid)[0]
            if len(cp_idx) == 0:
                continue

            # condensation point embeddings & beta
            cp_emb = emb_b[cp_idx]          # (M, D)
            cp_beta = beta_b[cp_idx]        # (M)
            cp_sid  = sid[cp_idx]           # (M)

            # ---------------------------------------
            # 2. β Loss
            #    - positive CPs: beta  1
            #    - all other hits (including background): beta 0
            # ---------------------------------------
            pos_beta_loss = F.binary_cross_entropy_with_logits(cp_beta, torch.ones_like(cp_beta), reduction="mean")

            # non-CP = everything except is_cp
            non_cp_mask = (~cp_mask)
            neg_beta_loss = F.binary_cross_entropy_with_logits(beta_b[non_cp_mask], torch.zeros_like(beta_b[non_cp_mask]), reduction="mean") if non_cp_mask.sum() > 0 else 0.0

            beta_loss = (self.beta_positive_weight * pos_beta_loss + self.beta_negative_weight * neg_beta_loss)

            # ---------------------------------------
            # 3. Attraction loss
            #    - for each instance, hits are pulled toward its CP embedding
            # ---------------------------------------
            attraction_loss = 0.0

            unique_ids = sid[valid].unique()
            for inst in unique_ids:
                inst_mask = (sid == inst)

                inst_hits = emb_b[inst_mask]               # (K, D)
                inst_cp = emb_b[inst_mask & cp_mask]       # (1, D)

                if inst_cp.numel() == 0:
                    continue

                inst_cp = inst_cp[0]                       # (D,)

                # squared L2 distance
                d2 = (inst_hits - inst_cp).pow(2).sum(dim=1)
                attraction_loss += d2.mean()

            attraction_loss = attraction_loss * self.attraction_weight

            # ---------------------------------------
            # 4. Repulsion loss
            #    - condensation points of different slices must repel
            # ---------------------------------------
            repulsion_loss = 0.0
            M = len(cp_idx)
            if M > 1:
                # pairwise distances between CP embeddings
                diff = cp_emb.unsqueeze(1) - cp_emb.unsqueeze(0)
                d2 = (diff ** 2).sum(dim=2)                 # (M, M)
                repulsion_loss = torch.exp(-d2).mean()      # repel by minimizing exp(-d^2)

                repulsion_loss = repulsion_loss * self.repulsion_weight

            # ---------------------------------------
            # Accumulate per-batch-element
            # ---------------------------------------
            loss_b = beta_loss + attraction_loss + repulsion_loss
            total_loss += loss_b
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=embed.device)

        return total_loss / count
