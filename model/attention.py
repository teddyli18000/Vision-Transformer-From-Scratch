import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=128, num_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Explicit Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        assert C == self.d_model, f"Input feature dim {C} does not match d_model {self.d_model}"

        # Project and reshape to [B, num_heads, N, d_k]
        q = self.q_proj(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention
        # scores: [B, num_heads, N, N]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        # out: [B, num_heads, N, d_k]
        out = torch.matmul(attn, v)

        # Reshape back to [B, N, d_model]
        out = out.transpose(1, 2).contiguous().view(B, N, C)

        out = self.out_proj(out)
        out = self.resid_drop(out)

        return out