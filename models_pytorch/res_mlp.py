import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange
from .utils import pair, check_sizes

'''
https://github.com/rishikksh20/ResMLP-pytorch/blob/master/resmlp.py
'''

class Aff(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones([1, 1, dim]))
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]))

    def forward(self, x):
        x = x * self.alpha + self.beta
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MLPblock(nn.Module):
    def __init__(self, num_patch, dim, mlp_dim, dropout = 0., depth=18):
        super().__init__()

        if depth <= 18:
            init_values = 0.1
        elif depth > 18 and depth <= 24:
            init_values = 1e-5
        else:
            init_values = 1e-6

        self.pre_affine = Aff(dim)
        self.token_mix = nn.Conv1d(num_patch, num_patch, kernel_size=1)
        self.ff = FeedForward(dim, mlp_dim, dropout)
        self.post_affine = Aff(dim)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = self.pre_affine(x)
        x = x + self.gamma_1 * self.token_mix(x)
        x = self.post_affine(x)
        x = x + self.gamma_2 * self.ff(x)
        return x

class ResMLP(nn.Module):
    def __init__(self, num_patch, d_model, depth, expansion_factor):
        super().__init__()
        self.model = nn.Sequential(
            *[MLPblock(num_patch, d_model, d_model * expansion_factor, depth=depth) for _ in range(depth)]
        )

    def forward(self, x):
        return self.model(x)

class ResMLPForImageClassification(ResMLP):
    def __init__(
        self, 
        in_channels = 3, 
        d_model = 384, 
        num_classes = 1000, 
        patch_size = 16, 
        image_size = 224, 
        depth = 12, 
        expansion_factor = 4):
        num_patches = check_sizes(image_size, patch_size)
        super().__init__(num_patches, d_model, expansion_factor, depth)

        self.patcher = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size),
        )

        self.affine = Aff(d_model)
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        patches = self.patcher(x)
        batch_size, num_channels, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 1)
        patches = patches.view(batch_size, -1, num_channels)
        embedding = self.model(patches)
        embedding = embedding.mean(dim=1)
        out = self.mlp_head(embedding)
        return out
