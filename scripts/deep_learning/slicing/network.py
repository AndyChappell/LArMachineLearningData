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


class ObjectCondensationNet(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, ff_dim, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, embed_dim)

        self.layers = nn.ModuleList([
            FlashTransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Heads
        self.beta_head = nn.Linear(embed_dim, 1)
        self.embed_head = nn.Linear(embed_dim, embed_dim)

        # -------------------------------
        # Important stabilizing init
        # Makes model start with low β everywhere instead of 0.5 confusion
        # -------------------------------
        nn.init.constant_(self.beta_head.bias, -5.0)
        nn.init.constant_(self.beta_head.weight, 0.0)
        

    def forward(self, hits):
        """
        hits: (B, N, F)
        NOTE: mask is intentionally NOT passed through. Keep mask only for loss/metrics.
        """
        x = self.input_proj(hits)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        beta = self.beta_head(x)             # (B, N, 1)
        embed = self.embed_head(x)           # (B, N, D)

        return {"beta": beta, "embed": embed}