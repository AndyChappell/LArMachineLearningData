import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================
# 1. KNN geometric context features
# ==========================================================
def knn_geometry_features(coords, k=8):
    """
    coords: (B, N, 2) = (x,z)
    returns:
        mean_dist   (B,N,1)
        density     (B,N,1)
    """
    # Compute pairwise distances (B,N,N)
    dist = torch.cdist(coords, coords)

    # Get k nearest neighbors (exclude self => topk takes 0==self but fine)
    knn = dist.topk(k, largest=False).values        # (B,N,k)

    mean_dist = knn.mean(dim=-1, keepdim=True)      # (B,N,1)
    density   = 1.0 / (mean_dist + 1e-6)            # (B,N,1)

    return mean_dist, density



# ==========================================================
# Flash attention blocks (same as before)
# ==========================================================
class FlashMHA(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv_proj(x).view(B,N,3,self.num_heads,self.head_dim).permute(2,0,3,1,4)
        q,k,v = qkv[0], qkv[1], qkv[2]

        if x.device.type=="cuda":
            out = F.scaled_dot_product_attention(
                q,k,v,
                dropout_p=self.dropout if self.training else 0.0
            )
        else:
            attn = torch.softmax((q @ k.transpose(-2,-1))/self.head_dim**0.5, dim=-1)
            out = attn @ v

        return self.o_proj(out.permute(0,2,1,3).contiguous().view(B,N,C))



class FlashTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = FlashMHA(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
    def forward(self,x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ==========================================================
# Object Condensation Model w/ KNN geometry
# ==========================================================
class ObjectCondensationNet(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, ff_dim,
                 dropout=0.1, knn_k=8):
        """
        input_dim = original hit feature count (x,z,v,...)
        extra 2 features added: mean_knn_dist, density
        new_input_dim = input_dim + 2
        """
        super().__init__()

        self.knn_k = knn_k
        total_in = input_dim + 2                         # + mean_dist + density

        self.input_proj = nn.Linear(total_in, embed_dim)

        self.layers = nn.ModuleList([
            FlashTransformerBlock(embed_dim,num_heads,ff_dim,dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.beta_head  = nn.Linear(embed_dim, 1)
        self.embed_head = nn.Linear(embed_dim, embed_dim)


    def forward(self, hits):
        """
        hits: (B,N,F) — must contain x=hits[...,0], z=hits[...,1]
        """

        coords = hits[...,:2]                        # (B,N,2)

        mean_dist, density = knn_geometry_features(coords, k=self.knn_k)

        # augment features
        hits_aug = torch.cat([hits, mean_dist, density], dim=-1)

        x = self.input_proj(hits_aug)

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        return {"beta": self.beta_head(x), "embed": self.embed_head(x)}
