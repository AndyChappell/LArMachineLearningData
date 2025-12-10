import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectCondensationLoss(nn.Module):
    def __init__(
        self,
        attraction_weight=1.0,
        repulsion_weight=1.5,
        margin_weight=5.0,
        threshold=0.5,
        margin=0.2,
        noncp_supp_weight=0.5,
        repulsion_across_slices_only=True
    ):
        super().__init__()
        self.attraction_weight = attraction_weight
        self.repulsion_weight  = repulsion_weight
        self.margin_weight     = margin_weight
        self.threshold         = threshold
        self.margin            = margin
        self.noncp_supp_weight = noncp_supp_weight
        self.repulsion_across_slices_only = repulsion_across_slices_only

    def forward(self, pred, slice_id, is_cp):
        beta  = pred["beta"].squeeze(-1)   # (B,N)
        embed = pred["embed"]              # (B,N,D)
        B,N,D = embed.shape

        # Sanitise NaNs — prevents cusolver explosions
        beta  = torch.nan_to_num(beta)
        embed = torch.nan_to_num(embed)

        total_loss = 0; beta_acc=0; attr_acc=0; rep_acc=0; valid_batches=0

        for b in range(B):
            sid   = slice_id[b]       # (N)
            cp    = (is_cp[b]==1)     # condensation points
            noncp = ~cp               # everything else is non-CP
            beta_b = beta[b]
            emb_b  = embed[b]

            if cp.sum()==0: continue

            #================#
            #   BETA LOSS    #
            #================#
            labels = cp.float()
            pos_wt = (noncp.sum() / (cp.sum()+1e-6)).detach()
            beta_logits = beta_b.clamp(-20,20)

            bce = F.binary_cross_entropy_with_logits(
                beta_logits, labels, pos_weight=torch.tensor(pos_wt,device=beta_b.device)
            )
            prob = torch.sigmoid(beta_logits)

            # margin terms
            pos_margin = F.relu((self.threshold+self.margin)-prob[cp]).mean()
            neg_margin = F.relu(prob[noncp]-(self.threshold-self.margin)).mean()

            # non-CP suppression
            ncp_supp = (prob[noncp]**2).mean()

            beta_loss = bce + self.margin_weight*(pos_margin+neg_margin) + self.noncp_supp_weight*ncp_supp
            beta_acc += beta_loss.detach()

            #================#
            # ATTRACTION     #
            #================#
            # Compute cp_embeddings per instance (vectorized)
            inst_ids = sid.unique()
            cp_emb = []
            att_sum = 0; att_count = 0

            for inst in inst_ids:
                inst_mask = sid==inst
                if (cp & inst_mask).sum()==0: continue
                cp_vec = emb_b[cp & inst_mask][0]            # single CP vector
                d2 = (emb_b[inst_mask]-cp_vec).pow(2).sum(-1)
                att_sum += d2.clamp(max=50).mean()
                att_count+=1

            if att_count>0:
                attraction = self.attraction_weight*(att_sum/att_count)
            else:
                attraction = torch.tensor(0.,device=beta.device)

            attr_acc+= attraction.detach()

            #================#
            # REPULSION      #
            #================#
            repulsion=0
            cp_idx = torch.where(cp)[0]

            if len(cp_idx)>1:
                cp_embs = emb_b[cp_idx]         # (M,D)
                sid_cp  = sid[cp_idx]           # (M)

                diff = (cp_embs.unsqueeze(1)-cp_embs.unsqueeze(0)).pow(2).sum(-1)  # (M,M)

                if self.repulsion_across_slices_only:
                    mask = (sid_cp.unsqueeze(1)!=sid_cp.unsqueeze(0))
                    if mask.any():
                        repulsion = torch.exp(-diff[mask].clamp(max=50)).mean()*self.repulsion_weight
                else:
                    repulsion = torch.exp(-diff.clamp(max=50)).mean()*self.repulsion_weight

            rep_acc+=repulsion.detach()

            total_loss += (beta_loss+attraction+repulsion)
            valid_batches+=1

        if valid_batches==0:
            zero=torch.tensor(0.,device=beta.device)
            return zero, {"beta_loss":zero,"attr_loss":zero,"repl_loss":zero}

        final = total_loss/valid_batches
        extras={
            "beta_loss":beta_acc/valid_batches,
            "attr_loss":attr_acc/valid_batches,
            "repl_loss":rep_acc/valid_batches
        }
        return final, extras
