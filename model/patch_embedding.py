import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, d_model=128, img_size=32):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Using Conv2d with stride=patch_size is the standard efficient way
        # to extract non-overlapping patches and project them to d_model.
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape

        # [B, C, H, W] -> [B, d_model, H/patch_size, W/patch_size]
        x = self.proj(x)

        # Flatten spatial dimensions: [B, d_model, num_patches]
        x = x.flatten(2)

        # Transpose to sequence format: [B, num_patches, d_model]
        x = x.transpose(1, 2)

        # Stability check
        assert x.shape == (B, self.num_patches, self.proj.out_channels), "Patch embedding shape mismatch"

        return x