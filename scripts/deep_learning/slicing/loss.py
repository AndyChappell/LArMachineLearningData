import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectCondensationLoss(nn.Module):
    def __init__(self, attraction_weight=1.0, repulsion_weight=1.0, bg_weight=2.0):
        super().__init__()
        self.attraction_weight = attraction_weight
        self.repulsion_weight  = repulsion_weight
        self.bg_weight = bg_weight    # can tune later (start ~2)

    def forward(self, pred, slice_id, is_cp):
        beta  = pred["beta"].squeeze(-1)   # (B,N)
        embed = pred["embed"]             # (B,N,D)

        total_loss = beta_log = attr_log = repl_log = 0.0
        count = 0

        B,N,_ = embed.shape

        for b in range(B):
            sid  = slice_id[b]            # (N,)
            cp_mask = (is_cp[b] == 1)     # CP hits
            bg_mask = (sid < 0)           # background
            fg_mask = (sid >= 0)

            beta_b = beta[b]
            emb_b  = embed[b]

            # ---- must define unique ids early ----
            unique_ids = sid[fg_mask].unique()

            # ============================
            # Slice-softmax CP classification
            # ============================
            beta_loss = 0.0
            slice_count = 0

            for inst in unique_ids:
                inst_mask = (sid == inst)
                cp_inst   = inst_mask & cp_mask
                if cp_inst.sum() != 1: 
                    continue

                inst_idx = torch.where(inst_mask)[0]
                cp_idx   = torch.where(cp_inst)[0][0]

                logits = beta_b[inst_idx].unsqueeze(0)  # (1,K)
                target = (inst_idx == cp_idx).nonzero(as_tuple=True)[0]  # (1,)

                beta_loss += F.cross_entropy(logits, target)
                slice_count += 1

            if slice_count == 0:
                continue

            beta_loss /= slice_count

            # ============================
            # Background suppression
            # ============================
            if bg_mask.any():
                bg_beta = torch.sigmoid(beta_b[bg_mask])
                beta_loss += self.bg_weight * bg_beta.mean()

            # ============================
            # Attraction loss
            # ============================
            attraction_loss = 0.0
            for inst in unique_ids:
                inst_mask = (sid == inst)
                inst_hits = emb_b[inst_mask]
                inst_cp   = emb_b[inst_mask & cp_mask]
                if inst_cp.numel()==0: continue
                inst_cp = inst_cp[0]
                attraction_loss += ((inst_hits-inst_cp)**2).sum(dim=1).mean()

            attraction_loss *= self.attraction_weight

            # ============================
            # Repulsion loss
            # ============================
            cp_idx = torch.where(cp_mask)[0]
            repulsion_loss = 0.0
            if len(cp_idx)>1:
                cp_emb = emb_b[cp_idx]
                diff = cp_emb.unsqueeze(1)-cp_emb.unsqueeze(0)
                dist2= (diff**2).sum(dim=2)
                repulsion_loss = torch.exp(-dist2).mean()*self.repulsion_weight

            # ============================
            # Accumulate
            # ============================
            loss_b = beta_loss+attraction_loss+repulsion_loss
            total_loss += loss_b
            beta_log   += beta_loss.detach()
            attr_log   += attraction_loss.detach()
            repl_log   += repulsion_loss.detach()
            count+=1

        if count==0:
            return torch.tensor(0.,device=beta.device), {}

        return total_loss/count, {
            "beta_loss": beta_log/count,
            "attr_loss": attr_log/count,
            "repl_loss": repl_log/count
        }

