import torch
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
from .utils import pair, check_sizes

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class ParallelSum(nn.Module):
    def __init__(self, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        return sum(map(lambda fn: fn(x), self.fns))

class ParallelWeightedSum(nn.Module):
    def __init__(self, sa, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)
        self.split_attention = sa

    def forward(self, x):
        x1 = self.fns[0](x)
        x2 = self.fns[1](x)
        x3 = self.fns[2](x)
        x_all = torch.stack([x1, x2, x3], 1)
        return self.split_attention(x_all)

class SplitAttention(nn.Module):
    def __init__(self, channel = 512, k = 3):
        super().__init__()
        self.channel = channel
        self.k = k
        self.mlp1 = nn.Linear(channel, channel, bias = False)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(channel, channel * k, bias = False)
        self.softmax = nn.Softmax(1)
    
    def forward(self, x_all):
        b, k, h, w, c = x_all.shape
        x_all = x_all.reshape(b, k, -1, c)              # bs,k,n,c
        a = torch.sum(torch.sum(x_all, 1), 1)           # bs,c
        hat_a = self.mlp2(self.gelu(self.mlp1(a)))      # bs,kc
        hat_a = hat_a.reshape(b, self.k, c)             # bs,k,c
        bar_a = self.softmax(hat_a)                     # bs,k,c
        attention = bar_a.unsqueeze(-2)                 # # bs,k,1,c
        out = attention * x_all                         # # bs,k,n,c
        out = torch.sum(out, 1).reshape(b, h, w, c)
        return out

class WeightedPermutator(nn.Module):
    def __init__(self, height, width, d_model, depth, segments, expansion_factor = 4, dropout = 0.):
        super().__init__()

        self.model = nn.Sequential(
            *[nn.Sequential(
                PreNormResidual(d_model, nn.Sequential(
                    ParallelWeightedSum(
                        SplitAttention(d_model, k = 3),
                        nn.Sequential(
                            Rearrange('b h w (c s) -> b w c (h s)', s = segments),
                            nn.Linear(height * segments, height * segments),
                            Rearrange('b w c (h s) -> b h w (c s)', s = segments),
                        ),
                        nn.Sequential(
                            Rearrange('b h w (c s) -> b h c (w s)', s = segments),
                            nn.Linear(width * segments, width * segments),
                            Rearrange('b h c (w s) -> b h w (c s)', s = segments),
                        ),
                        nn.Linear(d_model, d_model)
                    ),
                    nn.Linear(d_model, d_model)
                )),
                PreNormResidual(d_model, nn.Sequential(
                    nn.Linear(d_model, d_model * expansion_factor),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * expansion_factor, d_model),
                    nn.Dropout(dropout)
                ))
            ) for _ in range(depth)]
        )

    def forward(self, x):
        return self.model(x)

class Permutator(nn.Module):
    def __init__(self, height, width, d_model, depth, segments, expansion_factor = 4, dropout = 0.):
        super().__init__()

        self.model = nn.Sequential(
            *[nn.Sequential(
                PreNormResidual(d_model, nn.Sequential(
                    ParallelSum(
                        nn.Sequential(
                            Rearrange('b h w (c s) -> b w c (h s)', s = segments),
                            nn.Linear(height * segments, height * segments),
                            Rearrange('b w c (h s) -> b h w (c s)', s = segments),
                        ),
                        nn.Sequential(
                            Rearrange('b h w (c s) -> b h c (w s)', s = segments),
                            nn.Linear(width * segments, width * segments),
                            Rearrange('b h c (w s) -> b h w (c s)', s = segments),
                        ),
                        nn.Linear(d_model, d_model)
                    ),
                    nn.Linear(d_model, d_model)
                )),
                PreNormResidual(d_model, nn.Sequential(
                    nn.Linear(d_model, d_model * expansion_factor),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * expansion_factor, d_model),
                    nn.Dropout(dropout)
                ))
            ) for _ in range(depth)]
        )

    def forward(self, x):
        return self.model(x)

class ViP(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        d_model=256,
        depth=30,
        segments = 14,
        expansion_factor = 4,
        weighted = True
    ):
        image_size = pair(image_size)
        patch_size = pair(patch_size)
        assert (image_size[0] % patch_size[0]) == 0, 'image must be divisible by patch size'
        assert (image_size[1] % patch_size[1]) == 0, 'image must be divisible by patch size'
        assert (d_model % segments) == 0, 'dimension must be divisible by the number of segments'
        height = image_size[0] // patch_size[0]
        width = image_size[1] // patch_size[1]
        super().__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        )

        if weighted:
            self.blocks = WeightedPermutator(height, width, d_model, depth, segments, expansion_factor, dropout = 0.)
        else:
            self.blocks = Permutator(height, width, d_model, depth, segments, expansion_factor, dropout = 0.)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model),
            Reduce('b h w c -> b c', 'mean'),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        patches = self.patcher(x)
        patches = patches.permute(0, 2, 3, 1)
        embedding = self.blocks(patches)
        out = self.mlp_head(embedding)
        return out
