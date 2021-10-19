import torch
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce
from .utils import pair


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn, norm = nn.LayerNorm):
        super().__init__()
        self.fn = fn
        self.norm = norm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, H, W, C = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H // 2, W // 2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class sMLPBlock(nn.Module):
    def __init__(self, h = 224, w = 224, d_model = 3):
        super().__init__()
        self.proj_h = nn.Linear(h, h)
        self.proj_w = nn.Linear(w, w)
        self.fuse = nn.Conv2d(3 * d_model, d_model, kernel_size = 1)
    
    def forward(self,x):
        x_h = self.proj_h(x.permute(0,1,3,2)).permute(0,1,3,2)
        x_w = self.proj_w(x)
        x_id = x
        x_fuse = torch.cat([x_h, x_w, x_id], dim=1)
        out = self.fuse(x_fuse)
        return out

class sMLPStage(nn.Module):
    def __init__(self, height, width, d_model, depth, expansion_factor = 2, dropout = 0., pooling = False):
        super().__init__()

        self.pooling = pooling
        self.patch_merge = nn.Sequential(
            Rearrange('b c h w -> b h w c'),
            PatchMerging((height, width), d_model),
            Rearrange('b h w c -> b c h w'),
        )

        self.model = nn.Sequential(
            *[nn.Sequential(
                PreNormResidual(d_model, nn.Sequential(
                    nn.Conv2d(d_model, d_model, kernel_size = 3, padding = 1, groups = d_model), 
                ), norm = nn.BatchNorm2d),
                PreNormResidual(d_model, nn.Sequential(
                    sMLPBlock(
                        height, width, d_model
                    )
                ), norm = nn.BatchNorm2d),
                Rearrange('b c h w -> b h w c'),
                PreNormResidual(d_model, nn.Sequential(
                    nn.Linear(d_model, d_model * expansion_factor),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * expansion_factor, d_model),
                    nn.Dropout(dropout),
                ), norm = nn.LayerNorm),
                Rearrange('b h w c -> b c h w'),
            ) for _ in range(depth)]
        )

    def forward(self, x):
        x = self.model(x)
        if self.pooling:
            x = self.patch_merge(x)
        return x

class SparseMLP(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=4,
        in_channels=3,
        num_classes=1000,
        d_model=96,
        depth=[2,10,24,2],
        expansion_factor = 2,
        patcher_norm = False,
    ):
        image_size = pair(image_size)
        patch_size = pair(patch_size)
        assert (image_size[0] % patch_size[0]) == 0, 'image must be divisible by patch size'
        assert (image_size[1] % patch_size[1]) == 0, 'image must be divisible by patch size'
        height = image_size[0] // patch_size[0]
        width = image_size[1] // patch_size[1]
        super().__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size),

            nn.Identity() if (not patcher_norm) else nn.Sequential(
                Rearrange('b c h w -> b h w c'),
                nn.LayerNorm(d_model),
                Rearrange('b h w c -> b c h w'),
            )
        )

        self.layers = nn.ModuleList()
        for i_layer in range(len(depth)):
            i_depth = depth[i_layer]
            i_stage = sMLPStage(height // (2**i_layer), width // (2**i_layer), d_model, i_depth, expansion_factor = expansion_factor, pooling = ((i_layer + 1) < len(depth)))
            self.layers.append(i_stage)

            if (i_layer + 1) < len(depth):
                d_model = d_model * 2

        self.mlp_head = nn.Sequential(
            Rearrange('b c h w -> b h w c'),
            nn.LayerNorm(d_model),
            Reduce('b h w c -> b c', 'mean'),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        i = 0
        embedding = self.patcher(x)
        for layer in self.layers:
            i += 1
            embedding = layer(embedding)
        out = self.mlp_head(embedding)
        return out
