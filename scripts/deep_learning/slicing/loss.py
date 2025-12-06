import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectCondensationLoss(nn.Module):
    def __init__(
            self,
            attraction_weight=1.0,
            repulsion_weight=1.0,
            beta_positive_weight=10.0,#1.0,
            beta_negative_weight_signal=3.0,#0.5,
            beta_negative_weight_background=6.0,#0.5,
            margin_weight=10.0,#1.0,
            threshold=0.5,
            margin=0.3#0.1
        ):
        super().__init__()
        self.attraction_weight = attraction_weight
        self.repulsion_weight = repulsion_weight
        self.beta_positive_weight = beta_positive_weight
        self.beta_negative_weight_signal = beta_negative_weight_signal
        self.beta_negative_weight_background = beta_negative_weight_background
        self.margin_weight = margin_weight
        self.threshold = threshold
        self.margin = margin

    def forward(self, pred, slice_id, is_cp):
        beta = pred["beta"].squeeze(-1)      # (B, N)
        embed = pred["embed"]                # (B, N, D)
        B, N, D = embed.shape

        total_loss = 0.0
        beta_log = 0.0
        attr_log = 0.0
        repl_log = 0.0
        count = 0

        for b in range(B):
            sid = slice_id[b]     # (N,)
            cp_mask    = (is_cp[b] == 1)          # condensation points
            noncp_mask = (sid >= 0) & (~cp_mask)  # foreground non-CP
            bg_mask    = (sid < 0)                # background hits

            beta_b = beta[b]      # logits (N,)
            emb_b  = embed[b]     # embeddings (N,D)

            # --- Skip if no usable hits ---
            pos_count = cp_mask.sum()
            neg_count = noncp_mask.sum() + bg_mask.sum()
            if pos_count < 1 or neg_count < 1:
                continue

            # ===============================
            # 1) β LOSS (This is the key update)
            # ===============================
            labels = cp_mask.float()   # 1 for CP, 0 otherwise

            # dynamic weighting
            pos_weight    = (neg_count / (pos_count + 1e-6)).detach()
            noncp_weight  = 1.0
            bg_weight     = 2.0        # <- pushes background down harder

            # create per-hit weight vector
            weights = torch.zeros_like(labels)
            weights[cp_mask]    = pos_weight
            weights[noncp_mask] = noncp_weight
            weights[bg_mask]    = bg_weight

            beta_ce = F.binary_cross_entropy_with_logits(
                beta_b,
                labels,
                weight=weights,
                reduction='mean'
            )
            beta_loss = beta_ce
            beta_log += beta_loss.detach()

            # ===============================
            # 2) Attraction loss (unchanged)
            # ===============================
            attraction_loss = 0.0
            valid = sid >= 0
            unique_ids = sid[valid].unique()

            for inst in unique_ids:
                inst_mask = sid == inst
                inst_hits = emb_b[inst_mask]
                inst_cp   = emb_b[inst_mask & cp_mask]

                if inst_cp.numel() == 0:
                    continue

                inst_cp = inst_cp[0]
                d2 = (inst_hits - inst_cp).pow(2).sum(dim=1)
                attraction_loss += d2.mean()

            attraction_loss = self.attraction_weight * attraction_loss
            attr_log += attraction_loss.detach()

            # ===============================
            # 3) Repulsion loss (unchanged)
            # ===============================
            repulsion_loss = 0.0
            cp_idx = torch.where(cp_mask)[0]
            if len(cp_idx) > 1:
                cp_emb = emb_b[cp_idx]
                diff = cp_emb.unsqueeze(1) - cp_emb.unsqueeze(0)
                d2   = (diff**2).sum(dim=2)
                repulsion_loss = torch.exp(-d2).mean() * self.repulsion_weight

            repl_log += repulsion_loss.detach()

            # ===============================
            # Accumulate total
            # ===============================
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


def focal_bce(logits, targets, alpha=0.75, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = p * targets + (1 - p) * (1 - targets)
    return (alpha * (1 - pt)**gamma * ce).mean()
