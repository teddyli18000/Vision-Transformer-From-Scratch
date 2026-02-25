import torch
import torch.nn as nn
from .patch_embedding import PatchEmbedding
from .encoder import TransformerEncoder


class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 d_model=128, depth=4, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()

        self.patch_embed = PatchEmbedding(in_channels, patch_size, d_model, img_size)
        num_patches = self.patch_embed.num_patches

        # Learnable CLS token and Positional Embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer Encoder
        self.encoder = TransformerEncoder(depth, d_model, num_heads, mlp_ratio, dropout)

        # Classification Head
        self.head = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        # Truncated normal initialization is standard for ViTs
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]

        # 1. Patch Embedding
        x = self.patch_embed(x)  # [B, num_patches, d_model]

        # 2. Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches + 1, d_model]

        assert x.shape == (B, self.patch_embed.num_patches + 1, self.patch_embed.proj.out_channels)

        # 3. Add Positional Embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # 4. Transformer Encoder
        x = self.encoder(x)  # [B, num_patches + 1, d_model]

        # 5. Classification Head (using CLS token output)
        cls_out = x[:, 0]  # [B, d_model]
        out = self.head(cls_out)  # [B, num_classes]

        return out