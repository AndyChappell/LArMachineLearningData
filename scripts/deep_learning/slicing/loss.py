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
        count = 0

        for b in range(B):
            sid = slice_id[b]
            cp_mask = is_cp[b].bool()
            beta_b = beta[b]
            emb_b = embed[b]

            valid = sid >= 0
            if valid.sum() == 0:
                continue

            cp_idx = torch.where(cp_mask & valid)[0]
            if len(cp_idx) == 0:
                continue

            cp_emb = emb_b[cp_idx]
            cp_beta = beta_b[cp_idx]
            cp_sid  = sid[cp_idx]

            # -------------------------------------------------------
            # (1) β BCE losses (same as before)
            # -------------------------------------------------------

            #pos_bce = F.binary_cross_entropy_with_logits(
            #    cp_beta,
            #    torch.ones_like(cp_beta),
            #    reduction="mean"
            #)

            #pos_bce = focal_bce(cp_beta, torch.ones_like(cp_beta), alpha=0.75, gamma=2.0)
            pos_bce_accum = 0.0
            total_w = 0.0
            
            unique_ids = sid[valid].unique()
            for inst in unique_ids:
                inst_mask = (sid == inst)
                inst_cp_mask = inst_mask & cp_mask
            
                if inst_cp_mask.sum() == 0:
                    continue
            
                inst_size = inst_mask.sum().float()         # number of hits in slice
                inst_cp_beta = beta_b[inst_cp_mask]
                inst_loss = focal_bce(inst_cp_beta, torch.ones_like(inst_cp_beta), alpha=0.75, gamma=2.0)
            
                pos_bce_accum += inst_size * inst_loss       # CP represents full instance
                total_w += inst_size
            
            pos_bce = pos_bce_accum / total_w            

            non_cp_mask = (~cp_mask)
            if non_cp_mask.sum() > 0:
                neg_bce = F.binary_cross_entropy_with_logits(
                    beta_b[non_cp_mask],
                    torch.zeros_like(beta_b[non_cp_mask]),
                    reduction="mean"
                )
            else:
                neg_bce = 0.0

            # -------------------------------------------------------
            # (2) β margin-based hinge losses (new)
            # -------------------------------------------------------
            # convert logits to probabilities for threshold comparisons
            beta_prob = torch.sigmoid(beta_b)

            # positive: β > threshold + margin
            pos_margin_loss = F.relu(
                (self.threshold + self.margin) - beta_prob[cp_mask]
            ).mean()

            # negative: β < threshold - margin
            if non_cp_mask.sum() > 0:
                neg_margin_loss = F.relu(
                    beta_prob[non_cp_mask] - (self.threshold - self.margin)
                ).mean()
            else:
                neg_margin_loss = 0.0

            # Background suppression
            bg_mask = (sid == -1)

            if bg_mask.sum() > 0:
                bg_bce = F.binary_cross_entropy_with_logits(
                    beta_b[bg_mask],
                    torch.zeros_like(beta_b[bg_mask]),
                    reduction="mean"
                )
            else:
                bg_bce = 0.0

            beta_loss = (
                self.beta_positive_weight * pos_bce +
                self.beta_negative_weight_signal * neg_bce +
                self.beta_negative_weight_background * bg_bce +
                self.margin_weight * (pos_margin_loss + neg_margin_loss)
            )

            # -------------------------------------------------------
            # (3) Attraction loss (unchanged)
            # -------------------------------------------------------
            attraction_loss = 0.0
            unique_ids = sid[valid].unique()

            for inst in unique_ids:
                inst_mask = (sid == inst)
                inst_hits = emb_b[inst_mask]
                inst_cp = emb_b[inst_mask & cp_mask]

                if inst_cp.numel() == 0:
                    continue

                inst_cp = inst_cp[0]
                d2 = (inst_hits - inst_cp).pow(2).sum(dim=1)
                attraction_loss += d2.mean()

            attraction_loss = self.attraction_weight * attraction_loss

            # -------------------------------------------------------
            # (4) Repulsion loss (unchanged)
            # -------------------------------------------------------
            repulsion_loss = 0.0
            M = len(cp_idx)
            if M > 1:
                diff = cp_emb.unsqueeze(1) - cp_emb.unsqueeze(0)
                d2 = (diff ** 2).sum(dim=2)
                repulsion_loss = torch.exp(-d2).mean()
                repulsion_loss *= self.repulsion_weight

            # -------------------------------------------------------
            # accumulate
            # -------------------------------------------------------
            loss_b = beta_loss + attraction_loss + repulsion_loss
            total_loss += loss_b
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=embed.device)

        final_loss = total_loss / count

        # diagnostics (CPU-safe scalar values)
        extras = {
            "pos_bce": pos_bce.detach(),
            "neg_bce": neg_bce.detach() if isinstance(neg_bce, torch.Tensor) else torch.tensor(neg_bce),
            "pos_margin": pos_margin_loss.detach(),
            "neg_margin": neg_margin_loss.detach()
        }

        return final_loss, extras


def focal_bce(logits, targets, alpha=0.75, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    pt = p * targets + (1 - p) * (1 - targets)
    return (alpha * (1 - pt)**gamma * ce).mean()
