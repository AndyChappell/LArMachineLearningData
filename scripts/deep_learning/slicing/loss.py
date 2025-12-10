import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class ObjectCondensationLoss(nn.Module):
    """
    Stage 1 upgrade:
    ----------------
    Replace BCE with slice-softmax competition to enforce single CP candidate per slice.
    Attraction + repulsion remain unchanged.
    """

    def __init__(
        self,
        attraction_weight = 1.0,
        repulsion_weight  = 1.5,
        temperature_tau   = 0.7,     # controls sharpness of softmax competition
        clamp_value       = 20.0     # safety for logits
    ):
        super().__init__()
        self.attraction_weight = attraction_weight
        self.repulsion_weight  = repulsion_weight
        self.tau               = temperature_tau
        self.clamp             = clamp_value

    def forward(self, pred, slice_id, is_cp):
        beta  = pred["beta"].squeeze(-1)       # (B,N)
        embed = pred["embed"]                  # (B,N,D)
        B,N,D = embed.shape

        # global NaN guard
        beta  = torch.nan_to_num(beta, 0.0, 0.0, 0.0)
        embed = torch.nan_to_num(embed,0.0,0.0,0.0)

        total = 0.0
        beta_log = attr_log = rep_log = 0.0
        valid_batches = 0

        for b in range(B):
            sid = slice_id[b]           # (N,)
            cp_mask = (is_cp[b] == 1)   # (N,)
            beta_b = beta[b].clamp(-self.clamp, self.clamp)  # clamp logits for stability
            emb_b  = embed[b]

            # safety
            if cp_mask.sum()==0:
                continue

            # ===========================
            # 1) Slice-softmax β loss
            # ===========================
            inst_ids = sid.unique()
            beta_loss = 0.0
            slice_count = 0

            for inst in inst_ids:
                inst_mask = (sid == inst)

                # must have exactly 1 CP for a well-defined target
                if (cp_mask & inst_mask).sum() != 1:
                    continue

                logits = beta_b[inst_mask] / self.tau           # temperature scaling
                p = F.softmax(logits, dim=0)                     # slice-wise competition

                target = torch.zeros_like(p)
                cp_local = (torch.where(inst_mask)[0] == torch.where(cp_mask & inst_mask)[0][0]).nonzero(as_tuple=True)[0]
                target[cp_local] = 1.0

                slice_ce = (-target * torch.log(p + 1e-9)).sum() # force single winner
                beta_loss += slice_ce
                slice_count += 1

            if slice_count>0:
                beta_loss /= slice_count
            else:
                continue

            # ===========================
            # 2) Attraction loss
            # pull hits toward CP of the same slice
            # vectorized except for per-slice loop
            # ===========================
            attraction = 0.0
            att_count  = 0

            for inst in inst_ids:
                inst_mask = (sid == inst)
                cp_inst   = (cp_mask & inst_mask)

                if cp_inst.sum()==0: 
                    continue

                cp_vec = emb_b[cp_inst][0]                       # (D,)
                d2 = (emb_b[inst_mask] - cp_vec).pow(2).sum(-1) # (K,)
                attraction += torch.clamp(d2, max=50).mean()
                att_count += 1

            if att_count>0:
                attraction = self.attraction_weight*(attraction/att_count)
            else:
                attraction = torch.tensor(0.0, device=beta.device)

            # ===========================
            # 3) Repulsion loss (unchanged)
            # CP-to-CP push only
            # ===========================
            repulsion = 0.0
            cp_idx = torch.where(cp_mask)[0]

            if len(cp_idx)>1:
                cp_emb = emb_b[cp_idx]                           # (M,D)
                diff   = (cp_emb[:,None,:]-cp_emb[None,:,:]).pow(2).sum(-1) # (M,M)
                repulsion = torch.exp(-torch.clamp(diff,max=50)).mean()*self.repulsion_weight

            # ===========================
            # accumulate
            # ===========================
            loss_b = beta_loss + attraction + repulsion

            total    += loss_b
            beta_log += beta_loss.detach()
            attr_log += attraction.detach()
            rep_log  += repulsion.detach()
            valid_batches += 1

        if valid_batches==0:
            zero = torch.tensor(0.0,device=beta.device)
            return zero, {"beta_loss":zero,"attr_loss":zero,"repl_loss":zero}

        final = total/valid_batches
        extras = {
            "beta_loss": beta_log/valid_batches,
            "attr_loss": attr_log/valid_batches,
            "repl_loss": rep_log/valid_batches
        }
        return final, extras
