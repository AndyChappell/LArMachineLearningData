import torch
import torch.nn as nn
import torch.nn.functional as F


class PowerMHA(nn.Module):
    """
    Multi-Head Linear Power Attention

    Replaces softmax(QK^T)V with:
        φ(Q) (φ(K)^T V) / (φ(Q) (φ(K)^T 1))

    where φ(x) = relu(x)^p

    Complexity: O(N) in sequence length
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0, power=2.0, eps=1e-6):
        """
        Constructor
        """
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = dropout
        self.power = power
        self.eps = eps

    def feature_map(self, x):
        """
        Apply relu and the exponentation as the feature map
        """
        return torch.relu(x) ** self.power

    def forward(self, x):
        """
        Run inference

        x: (B, N, C)
        """
        B, N, C = x.shape

        # ---- Project to Q, K, V ----
        qkv = self.qkv_proj(x)  # (B, N, 3C)
        qkv = qkv.view(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)

        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, D)

        # ---- Apply feature map ----
        q = self.feature_map(q)  # (B, H, N, D)
        k = self.feature_map(k)  # (B, H, N, D)

        # ---- Linear attention computation ----

        # Step 1: Aggregate K^T V
        # (B, H, D, D)
        kv = torch.einsum("bhnd,bhne->bhde", k, v)

        # Step 2: Compute normalization denominator
        # sum over sequence dimension of K
        k_sum = k.sum(dim=2)  # (B, H, D)

        # (B, H, N)
        denom = torch.einsum("bhnd,bhd->bhn", q, k_sum)
        denom = denom + self.eps  # numerical stability

        # Step 3: Compute output
        # (B, H, N, D)
        out = torch.einsum("bhnd,bhde->bhne", q, kv)

        # Normalize
        out = out / denom.unsqueeze(-1)

        # ---- Dropout on output ----
        if self.training and self.dropout > 0:
            out = F.dropout(out, p=self.dropout)

        # ---- Merge heads ----
        out = (
            out.permute(0, 2, 1, 3)
               .contiguous()
               .view(B, N, C)
        )

        return self.o_proj(out)


class TransformerBlock(nn.Module):
    """
    Transformer block with power attention
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        """
        Constructor
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = PowerMHA(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )

    def forward(self, x):
        """
        Run inference
        """
        # No key_padding_mask forwarded here — attention runs on full sequence (including pads)
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
