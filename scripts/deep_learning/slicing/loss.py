import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectCondensationLoss(nn.Module):
    def __init__(
        self,
        attraction_weight=1.0,
        repulsion_weight=1.0,
        beta_positive_weight=10.0,
        beta_negative_weight_signal=3.0,
        margin_weight=10.0,
        threshold=0.5,
        margin=0.3
    ):
        super().__init__()
        self.attraction_weight = attraction_weight
        self.repulsion_weight = repulsion_weight
        self.beta_positive_weight = beta_positive_weight
        self.beta_negative_weight_signal = beta_negative_weight_signal
        self.margin_weight = margin_weight
        self.threshold = threshold
        self.margin = margin

    def forward(self, pred, slice_id, is_cp):
        beta  = pred["beta"].squeeze(-1)      # (B,N)
        embed = pred["embed"]                # (B,N,D)
        B,N,D = embed.shape

        total_loss = 0.
        beta_log = attr_log = repl_log = 0.
        count = 0

        for b in range(B):
            sid = slice_id[b]             # (N,)
            cp_mask = (is_cp[b] == 1)     # CP=1, nonCP=0
            noncp_mask = ~cp_mask         # all are real slices now

            beta_b = beta[b]
            emb_b  = embed[b]

            pos_count = cp_mask.sum()
            neg_count = noncp_mask.sum()
            if pos_count < 1 or neg_count < 1: 
                continue

            # --------------------------
            # β classification loss (balanced BCE)
            # --------------------------
            labels = cp_mask.float()

            pos_weight = (neg_count / (pos_count + 1e-6)).detach()

            beta_ce = F.binary_cross_entropy_with_logits(
                beta_b, labels,
                pos_weight=torch.tensor(pos_weight, device=beta_b.device)
            )

            beta_loss = beta_ce
            beta_log += beta_loss.detach()

            # --------------------------
            # Attraction
            # --------------------------
            attraction_loss = 0.
            unique_ids = sid.unique()

            for inst in unique_ids:
                inst_mask = sid == inst
                inst_hits = emb_b[inst_mask]
                inst_cp   = emb_b[inst_mask & cp_mask]

                if inst_cp.numel() == 0:
                    continue

                inst_cp = inst_cp[0]
                d2 = (inst_hits - inst_cp).pow(2).sum(dim=1)
                attraction_loss += d2.mean()

            attraction_loss *= self.attraction_weight
            attr_log += attraction_loss.detach()

            # --------------------------
            # Repulsion between CP embeddings
            # --------------------------
            repulsion_loss = 0.
            cp_idx = torch.where(cp_mask)[0]

            if len(cp_idx) > 1:
                cp_emb = emb_b[cp_idx]
                diff = cp_emb.unsqueeze(1) - cp_emb.unsqueeze(0)
                d2 = (diff**2).sum(dim=2)
                repulsion_loss = torch.exp(-d2).mean() * self.repulsion_weight

            repl_log += repulsion_loss.detach()

            total_loss += (beta_loss + attraction_loss + repulsion_loss)
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=embed.device), {}

        final_loss = total_loss / count

        extras = {
            "beta_loss": beta_log / count,
            "attr_loss": attr_log / count,
            "repl_loss": repl_log / count
        }

        return final_loss, extras
