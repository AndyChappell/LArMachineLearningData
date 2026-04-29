import torch
import torch.nn as nn
import torch.nn.functional as F


class PowerMHA(nn.Module):
    """
    Multi-Head Linear Power Attention

    Replaces softmax(QK^T)V with:
        φ(Q) (φ(K)^T V) / (φ(Q) (φ(K)^T 1))

    where φ(x) = relu(x)^p

    Complexity: O(N) in sequence length.

    Based on the linear attention mechanism of:
    Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention (Katharopoulos et al., 2020)
     	arXiv:2006.16236
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

    def forward(self, x, mask=None):
        """
        Run inference

        x: (B, N, C)
        mask: (B, N) boolean (True = valid)
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
        
        # ---- Scale QK^T / sqrt(d) ----
        scale = self.head_dim ** -0.5
        q = q * scale
        k = k * scale

        if mask is not None:
            mask_ = mask.unsqueeze(1).unsqueeze(-1).to(q.dtype)  # (B, 1, N, 1) (also ensure float)
    
            k = k * mask_
            v = v * mask_

        # ---- Linear attention computation ----

        # Step 1: Aggregate K^T V
        # (B, H, D, D)
        kv = torch.einsum("bhnd,bhne->bhde", k, v)

        # Step 2: Compute normalization denominator
        # sum over sequence dimension of K
        k_sum = k.sum(dim=2)  # (B, H, D)

        # (B, H, N)
        denom = torch.einsum("bhnd,bhd->bhn", q, k_sum)
        denom = denom.clamp(min=self.eps) # numerical stability

        # Step 3: Compute output
        # (B, H, N, D)
        out = torch.einsum("bhnd,bhde->bhne", q, kv)

        # Normalize
        out = out / denom.unsqueeze(-1)

        # ---- Dropout ----
        out = F.dropout(out, p=self.dropout, training=self.training)

        # Zero out padded outputs explicitly (clean)
        if mask is not None:
            out = out * mask_

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

    def forward(self, x, mask=None):
        """
        Run inference
        """
        if mask is not None:
            mask_ = mask.unsqueeze(-1).to(x.dtype)

            # Attention sub-layer
            x = x * mask_
            x = x + self.attn(self.norm1(x), mask=mask)
            x = x * mask_

            # FFN sub-layer — zero padded tokens before norm2 so LayerNorm
            # does not see zero vectors and produce spurious non-zero outputs.
            normed = self.norm2(x) * mask_
            x = x + self.ffn(normed)
            x = x * mask_

        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.ffn(self.norm2(x))

        return x


class LArTPCTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, ff_dim, num_layers, num_classes, dropout=0.1):
        """
        Constructor
        """
        super().__init__()

        # ---- Input embedding ----
        self.input_proj = nn.Linear(input_dim, embed_dim)

        # ---- Transformer stack ----
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # ---- Output head ----
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, mask=None):
        """
        Run inference
        """

        # ---- Input projection ----
        x = self.input_proj(x)

        # ---- Transformer ----
        for layer in self.layers:
            x = layer(x, mask=mask)

        x = self.norm(x)

        # ---- Classification ----
        logits = self.classifier(x)  # (B, N, num_classes)

        return logits