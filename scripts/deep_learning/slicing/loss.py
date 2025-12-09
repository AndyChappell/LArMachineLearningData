import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectCondensationLoss(nn.Module):
    def __init__(
        self,
        attraction_weight=1.0,
        repulsion_weight=1.0,
        margin_weight=5.0,
        threshold=0.5,
        margin=0.2
    ):
        super().__init__()
        self.attraction_weight = attraction_weight
        self.repulsion_weight  = repulsion_weight
        self.margin_weight     = margin_weight
        self.threshold         = threshold
        self.margin            = margin

    def forward(self, pred, slice_id, is_cp):
        beta  = pred["beta"].squeeze(-1)  # (B,N)
        embed = pred["embed"]             # (B,N,D)
        B, N, D = embed.shape

        # Global finite check: if something is already broken, punt
        if not torch.isfinite(beta).all() or not torch.isfinite(embed).all():
            beta  = torch.nan_to_num(beta,  0.0, 0.0, 0.0)
            embed = torch.nan_to_num(embed, 0.0, 0.0, 0.0)

        total_loss = 0.0
        beta_log = attr_log = repl_log = 0.0
        count = 0

        for b in range(B):
            sid    = slice_id[b]             # (N,)
            cp_mask    = (is_cp[b] == 1)
            noncp_mask = ~cp_mask

            beta_b = beta[b]
            emb_b  = embed[b]

            # Per-event finite check
            if (not torch.isfinite(beta_b).all()) or (not torch.isfinite(emb_b).all()):
                continue

            pos_count = cp_mask.sum()
            neg_count = noncp_mask.sum()
            if pos_count < 1 or neg_count < 1:
                continue

            # --------------------------------
            # 1) β loss: BCE + margin
            # --------------------------------
            labels = cp_mask.float()

            # clamp logits before BCE for stability
            beta_b_clamped = beta_b.clamp(min=-20.0, max=20.0)

            pos_weight = (neg_count / (pos_count + 1e-6)).detach()
            bce_loss = F.binary_cross_entropy_with_logits(
                beta_b_clamped,
                labels,
                pos_weight=torch.tensor(pos_weight, device=beta_b.device)
            )

            beta_prob = torch.sigmoid(beta_b_clamped)

            thr  = self.threshold
            marg = self.margin

            if cp_mask.any():
                pos_m = F.relu((thr + marg) - beta_prob[cp_mask]).mean()
            else:
                pos_m = torch.tensor(0.0, device=beta_b.device)

            if noncp_mask.any():
                neg_m = F.relu(beta_prob[noncp_mask] - (thr - marg)).mean()
            else:
                neg_m = torch.tensor(0.0, device=beta_b.device)

            margin_loss = pos_m + neg_m

            beta_loss = bce_loss + self.margin_weight * margin_loss
            beta_log += beta_loss.detach()

            # --------------------------------
            # 2) Attraction loss
            # --------------------------------
            attraction_loss = 0.0
            unique_ids = sid.unique()

            for inst in unique_ids:
                inst_mask = (sid == inst)
                inst_hits = emb_b[inst_mask]
                inst_cp   = emb_b[inst_mask & cp_mask]

                if inst_cp.numel() == 0:
                    continue

                inst_cp = inst_cp[0]
                d2 = (inst_hits - inst_cp).pow(2).sum(dim=1)  # (K,)
                d2 = d2.clamp(max=50.0)                       # clamp distances
                attraction_loss += d2.mean()

            attraction_loss *= self.attraction_weight
            attr_log += attraction_loss.detach()

            # --------------------------------
            # 3) Repulsion loss
            # --------------------------------
            repulsion_loss = 0.0
            cp_idx = torch.where(cp_mask)[0]
            if len(cp_idx) > 1:
                cp_emb = emb_b[cp_idx]
                diff   = cp_emb.unsqueeze(1) - cp_emb.unsqueeze(0)
                d2     = (diff ** 2).sum(dim=2)
                d2     = d2.clamp(max=50.0)
                repulsion_loss = torch.exp(-d2).mean() * self.repulsion_weight

            repl_log += repulsion_loss.detach()

            total_loss += (beta_loss + attraction_loss + repulsion_loss)
            count += 1

        if count == 0:
            zero = torch.tensor(0.0, device=beta.device)
            return zero, {"beta_loss": zero, "attr_loss": zero, "repl_loss": zero}

        final_loss = total_loss / count
        extras = {
            "beta_loss": beta_log / count,
            "attr_loss": attr_log / count,
            "repl_loss": repl_log / count,
        }
        return final_loss, extras
