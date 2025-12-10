import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectCondensationLoss(nn.Module):
    """
    Object condensation loss with:
      - slice-wise softmax CP selection (one winner per slice),
      - non-CP suppression within each slice,
      - attraction to CP in embedding space,
      - repulsion only across slices between CP embeddings.
    """

    def __init__(
        self,
        attraction_weight=1.0,
        repulsion_weight=1.5,
        temperature_tau=0.7,
        noncp_supp_weight=0.5,
        clamp_value=20.0,
    ):
        super().__init__()
        self.attraction_weight = attraction_weight
        self.repulsion_weight = repulsion_weight
        self.tau = temperature_tau
        self.noncp_supp_weight = noncp_supp_weight
        self.clamp = clamp_value

    def forward(self, pred, slice_id, is_cp):
        """
        pred: dict with
          - pred["beta"]: (B, N, 1)
          - pred["embed"]: (B, N, D)
        slice_id: (B, N) long
        is_cp:    (B, N) long or bool, 1 == condensation point
        """
        beta  = pred["beta"].squeeze(-1)        # (B, N)
        embed = pred["embed"]                   # (B, N, D)
        B, N, D = embed.shape

        # Global NaN/Inf guard
        beta  = torch.nan_to_num(beta,  0.0, 0.0, 0.0)
        embed = torch.nan_to_num(embed, 0.0, 0.0, 0.0)

        total_loss = 0.0
        beta_log = 0.0
        attr_log = 0.0
        repl_log = 0.0
        valid_batches = 0

        for b in range(B):
            sid = slice_id[b]           # (N,)
            cp_mask = (is_cp[b] == 1)   # (N,)
            beta_b = beta[b].clamp(-self.clamp, self.clamp)  # (N,)
            emb_b  = embed[b]          # (N, D)

            # Per-event sanity
            if cp_mask.sum() == 0:
                continue
            if not torch.isfinite(beta_b).all() or not torch.isfinite(emb_b).all():
                continue

            inst_ids = sid.unique()

            # ======================================
            # 1) Slice-softmax β loss + non-CP suppression
            # ======================================
            beta_loss = 0.0
            slice_count = 0

            for inst in inst_ids:
                inst_mask = (sid == inst)          # hits in this slice
                cp_inst = (cp_mask & inst_mask)    # CP in this slice

                # Require exactly one CP in this slice
                if cp_inst.sum() != 1:
                    continue

                # Slice logits & softmax
                logits_slice = beta_b[inst_mask] / self.tau      # (K,)
                p = F.softmax(logits_slice, dim=0)               # (K,), sum=1

                # Build one-hot target inside this slice
                idx_slice = torch.where(inst_mask)[0]            # global indices of hits in this slice
                cp_global_idx = torch.where(cp_inst)[0][0]       # scalar global index of CP
                cp_local_idx = (idx_slice == cp_global_idx).nonzero(as_tuple=True)[0]  # local index
                target = torch.zeros_like(p)
                target[cp_local_idx] = 1.0

                # Softmax cross-entropy: encourage CP hit prob→1, others→0
                slice_ce = (-target * torch.log(p + 1e-9)).sum()

                # Non-CP suppression within slice: penalize mass on non-CP hits
                noncp_slice_mask = (target == 0)
                if noncp_slice_mask.any():
                    noncp_probs = p[noncp_slice_mask]
                    supp_loss = noncp_probs.mean()
                else:
                    supp_loss = torch.tensor(0.0, device=beta_b.device)

                slice_loss = slice_ce + self.noncp_supp_weight * supp_loss
                beta_loss += slice_loss
                slice_count += 1

            if slice_count == 0:
                # No valid slices in this event
                continue

            beta_loss = beta_loss / slice_count
            beta_log += beta_loss.detach()

            # ======================================
            # 2) Attraction loss (unchanged)
            # pull hits toward CP embedding of their slice
            # ======================================
            attraction = 0.0
            att_count = 0

            for inst in inst_ids:
                inst_mask = (sid == inst)
                cp_inst = (cp_mask & inst_mask)
                if cp_inst.sum() == 0:
                    continue

                cp_vec = emb_b[cp_inst][0]                     # (D,)
                inst_hits = emb_b[inst_mask]                   # (K, D)
                d2 = (inst_hits - cp_vec).pow(2).sum(dim=-1)   # (K,)
                d2 = d2.clamp(max=50.0)
                attraction += d2.mean()
                att_count += 1

            if att_count > 0:
                attraction = self.attraction_weight * (attraction / att_count)
            else:
                attraction = torch.tensor(0.0, device=beta_b.device)

            attr_log += attraction.detach()

            # ======================================
            # 3) Repulsion loss
            # repel CP embeddings across different slices only
            # ======================================
            repulsion = 0.0
            cp_idx = torch.where(cp_mask)[0]      # (M,)

            if len(cp_idx) > 1:
                cp_emb = emb_b[cp_idx]            # (M, D)
                cp_sid = sid[cp_idx]              # (M,)

                # Pairwise squared distances between CP embeddings
                diff = cp_emb.unsqueeze(1) - cp_emb.unsqueeze(0)   # (M, M, D)
                d2 = (diff ** 2).sum(dim=-1)                       # (M, M)

                # Only keep CP–CP pairs from different slices
                cross_slice_mask = (cp_sid.unsqueeze(1) != cp_sid.unsqueeze(0))  # (M, M)
                if cross_slice_mask.any():
                    d2 = d2[cross_slice_mask]
                    d2 = d2.clamp(max=50.0)
                    repulsion = torch.exp(-d2).mean() * self.repulsion_weight
                else:
                    repulsion = torch.tensor(0.0, device=beta_b.device)

            repl_log += repulsion.detach()

            # ======================================
            # Accumulate per-event
            # ======================================
            loss_b = beta_loss + attraction + repulsion
            total_loss += loss_b
            valid_batches += 1

        if valid_batches == 0:
            zero = torch.tensor(0.0, device=beta.device)
            return zero, {
                "beta_loss": zero,
                "attr_loss": zero,
                "repl_loss": zero,
            }

        final_loss = total_loss / valid_batches
        extras = {
            "beta_loss": beta_log / valid_batches,
            "attr_loss": attr_log / valid_batches,
            "repl_loss": repl_log / valid_batches,
        }
        return final_loss, extras
