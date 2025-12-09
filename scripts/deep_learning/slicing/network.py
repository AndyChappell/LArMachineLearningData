import torch, torch.nn as nn, torch.nn.functional as F

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
        """
        Mask-free forward. Do NOT pass a key_padding_mask here if you want FlashAttention fast path.
        x: (B, N, C)
        """
        B, N, C = x.shape
        qkv = self.qkv_proj(x)                                    # (B, N, 3*C)
        qkv = qkv.view(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)                          # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]                          # each (B, heads, N, head_dim)

        if not torch.jit.is_scripting() and x.device.type == "cuda":
            # Fast path: no attn mask / key_padding_mask passed
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False
            )  # (B*heads, N, head_dim) or fused output shaped to (B, heads, N, head_dim)
        else:
            # CPU fallback: regular attention - needed for LibTorch
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn_probs = torch.softmax(attn_scores, dim=-1)
            out = torch.matmul(attn_probs, v)

        out = out.view(B, self.num_heads, N, self.head_dim).permute(0, 2, 1, 3).reshape(B, N, C)
        return self.o_proj(out)


class FlashTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = FlashMHA(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )

    def forward(self, x):
        # No key_padding_mask forwarded here — attention runs on full sequence (including pads)
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F


class ObjectCondensationNet(nn.Module):
    def __init__(
        self,
        input_dim,      # number of raw hit features, e.g. [x,z,view,...]
        embed_dim,
        num_heads,
        num_layers,
        ff_dim,
        num_classes=None,   # kept for compatibility
        dropout=0.1,
        knn_k=16            # neighbors for local geometry
    ):
        super().__init__()

        self.knn_k = knn_k

        # We now add ONLY 2 geometric features: mean_knn_dist, density
        geom_dim = 2
        total_in = input_dim + geom_dim

        self.input_proj = nn.Linear(total_in, embed_dim)

        self.layers = nn.ModuleList([
            FlashTransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # OC heads
        self.beta_head  = nn.Linear(embed_dim, 1)
        self.embed_head = nn.Linear(embed_dim, embed_dim)

        # Stabilising init for β
        nn.init.constant_(self.beta_head.bias, -5.0)
        nn.init.constant_(self.beta_head.weight, 0.0)


    def _local_geometry(self, hits, sample_size=4096):
        """
        Memory-safe approximate kNN geometry.
    
        hits: (B,N,F) with at least x,z as the first 2 features.
        Returns:
            mean_knn_dist: (B,N,1)
            density:       (B,N,1)
        """
        B, N, F = hits.shape
        device = hits.device
        dtype  = hits.dtype
    
        if N < 2:
            zero = torch.zeros(B, N, 1, device=device, dtype=dtype)
            return zero, zero
    
        coords   = hits[..., :2]          # (B,N,2)
        has_view = F >= 3
        views    = hits[..., 2:3] if has_view else None
    
        # -----------------------------
        # 1) Choose reference subset
        # -----------------------------
        if N > sample_size:
            idx = torch.randperm(N, device=device)[:sample_size]
            ref_coords = coords[:, idx, :]          # (B,S,2)
            ref_views  = views[:, idx, :] if has_view else None
        else:
            ref_coords = coords                     # (B,N,2)
            ref_views  = views
    
        # -----------------------------
        # 2) Distances only to refs
        # -----------------------------
        dist = torch.cdist(coords, ref_coords)      # (B,N,S)
    
        # Limit insane distances to something reasonable
        dist = torch.nan_to_num(dist, nan=0.0, posinf=1e4, neginf=0.0)
        dist = dist.clamp(min=0.0, max=1e4)
    
        if has_view:
            view_equal = (views == ref_views.transpose(-2, -1))  # (B,N,S)
            # Very large penalty for cross-view, but finite
            dist = dist + (~view_equal) * 1e4
    
        # -----------------------------
        # 3) kNN distances
        # -----------------------------
        k = min(self.knn_k, dist.size(-1))
        knn_dists = dist.topk(k, largest=False).values    # (B,N,k)
    
        mean_knn_dist = knn_dists.mean(dim=-1, keepdim=True)   # (B,N,1)
        # Clamp to avoid division extremes
        mean_knn_dist = mean_knn_dist.clamp(min=1e-3, max=1e3)
    
        density = 1.0 / mean_knn_dist
        density = density.clamp(min=0.0, max=1e3)
    
        mean_knn_dist = torch.nan_to_num(mean_knn_dist, nan=0.0, posinf=1e3, neginf=0.0)
        density       = torch.nan_to_num(density,       nan=0.0, posinf=1e3, neginf=0.0)
    
        return mean_knn_dist.to(dtype), density.to(dtype)


    # -------------------------------
    # Forward
    # -------------------------------
    def forward(self, hits):
        """
        hits: (B, N, F_raw)
          with at least x,z in the first 2 channels.
        """
        mean_knn_dist, density = self._local_geometry(hits)
        geom_feats = torch.cat([mean_knn_dist, density], dim=-1)
        hits_aug   = torch.cat([hits, geom_feats], dim=-1)

        x = self.input_proj(hits_aug)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        beta  = self.beta_head(x)    # (B,N,1)
        embed = self.embed_head(x)   # (B,N,D)

        return {"beta": beta, "embed": embed}




def full_local_geometry(self, hits):
    """
    hits: (B, N, F) with at least x,z as first 2 features.
          If view is present as 3rd feature, we ignore cross-view neighbours.

    Returns:
        mean_knn_dist: (B,N,1)
        density:       (B,N,1)
    """
    B, N, F = hits.shape
    device = hits.device
    dtype  = hits.dtype

    if N < 2:
        zero = torch.zeros(B, N, 1, device=device, dtype=dtype)
        return zero, zero

    coords = hits[..., :2]     # (B,N,2)  -> [x,z]
    has_view = F >= 3

    if has_view:
        views = hits[..., 2:3]           # (B,N,1)
    else:
        views = None

    # pairwise distances (B,N,N)
    dist = torch.cdist(coords, coords)

    # optionally restrict neighbors to same view
    if has_view:
        view_equal = (views == views.transpose(-2, -1))  # (B,N,N)
        # large penalty for cross-view so they never appear in top-k
        dist = dist + (~view_equal) * 1e6

    # choose k neighbours (may include self; that's fine)
    k = min(self.knn_k, N)
    knn = dist.topk(k, largest=False)
    knn_dists = knn.values    # (B,N,k)

    mean_knn_dist = knn_dists.mean(dim=-1, keepdim=True)   # (B,N,1)
    density       = 1.0 / (mean_knn_dist + 1e-6)          # (B,N,1)

    return mean_knn_dist.to(dtype), density.to(dtype)