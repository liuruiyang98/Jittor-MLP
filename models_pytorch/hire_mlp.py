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

class PatchEmbedding(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride, padding, norm_layer=False):
        super().__init__()
        self.reduction = nn.Sequential(
                            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding = padding),

                            nn.Identity() if (not norm_layer) else nn.Sequential(
                                Rearrange('b c h w -> b h w c'),
                                nn.LayerNorm(dim_out),
                                Rearrange('b h w c -> b c h w'),
                            )
                        )

    def forward(self, x):
        return self.reduction(x)

class FeedForward(nn.Module):
    def __init__(self, dim_in, hidden_dim, dim_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, hidden_dim, kernel_size = 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim_out, kernel_size = 1),
        )
    def forward(self, x):
        return self.net(x)

class CrossRegion(nn.Module):
    def __init__(self, step = 1, dim = 1):
        super().__init__()
        self.step = step
        self.dim = dim

    def forward(self, x):
        return torch.roll(x, self.step, self.dim)

class InnerRegionW(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = w
        self.region = nn.Sequential(
            Rearrange('b c h (w group) -> b (c w) h group', w = self.w)
        )

    def forward(self, x):
        return self.region(x)

class InnerRegionH(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.region = nn.Sequential(
            Rearrange('b c (h group) w -> b (c h) group w', h = self.h)
        )

    def forward(self, x):
        return self.region(x)

class InnerRegionRestoreW(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = w
        self.region = nn.Sequential(
            Rearrange('b (c w) h group -> b c h (w group)', w = self.w)
        )

    def forward(self, x):
        return self.region(x)

class InnerRegionRestoreH(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.region = nn.Sequential(
            Rearrange('b (c h) group w -> b c (h group) w', h = self.h)
        )

    def forward(self, x):
        return self.region(x)

class HireMLPBlock(nn.Module):
    def __init__(self, h, w, d_model, cross_region_step = 1, cross_region_id = 0, cross_region_interval = 2):
        super().__init__()

        # cross region every cross_region_interval HireMLPBlock
        self.cross_region = (cross_region_id % cross_region_interval == 0)

        if self.cross_region:
            self.cross_regionW = CrossRegion(step = cross_region_step, dim = 3)
            self.cross_regionH = CrossRegion(step = cross_region_step, dim = 2)
            self.cross_region_restoreW = CrossRegion(step = -cross_region_step, dim = 3)
            self.cross_region_restoreH = CrossRegion(step = -cross_region_step, dim = 2)
        else:
            self.cross_regionW = nn.Identity()
            self.cross_regionH = nn.Identity()
            self.cross_region_restoreW = nn.Identity()
            self.cross_region_restoreH = nn.Identity()

        self.inner_regionW = InnerRegionW(w)
        self.inner_regionH = InnerRegionH(h)
        self.inner_region_restoreW = InnerRegionRestoreW(w)
        self.inner_region_restoreH = InnerRegionRestoreH(h)


        self.proj_h = FeedForward(h * d_model, d_model // 2, h * d_model)
        self.proj_w = FeedForward(w * d_model, d_model // 2, w * d_model)
        self.proj_c = nn.Conv2d(d_model, d_model, kernel_size = 1)
    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x_h = self.inner_regionH(self.cross_regionH(x))
        x_w = self.inner_regionW(self.cross_regionW(x))
        
        x_h = self.proj_h(x_h)
        x_w = self.proj_w(x_w)
        x_c = self.proj_c(x)

        x_h = self.cross_region_restoreH(self.inner_region_restoreH(x_h))
        x_w = self.cross_region_restoreW(self.inner_region_restoreW(x_w))

        out = x_c + x_h + x_w
        out = out.permute(0, 2, 3, 1)
        return out

class HireMLPStage(nn.Module):
    def __init__(self, h, w, d_model_in, d_model_out, depth, cross_region_step, cross_region_interval, expansion_factor = 2, dropout = 0., pooling = False):
        super().__init__()

        self.pooling = pooling
        self.patch_merge = nn.Sequential(
            Rearrange('b h w c -> b c h w'),
            PatchEmbedding(d_model_in, d_model_out, kernel_size = 3, stride = 2, padding=1, norm_layer=False),
            Rearrange('b c h w -> b h w c'),
        )

        self.model = nn.Sequential(
            *[nn.Sequential(
                PreNormResidual(d_model_in, nn.Sequential(
                    HireMLPBlock(
                        h, w, d_model_in, cross_region_step = cross_region_step, cross_region_id = i_depth + 1, cross_region_interval = cross_region_interval
                    )
                ), norm = nn.LayerNorm),
                PreNormResidual(d_model_in, nn.Sequential(
                    nn.Linear(d_model_in, d_model_in * expansion_factor),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model_in * expansion_factor, d_model_in),
                    nn.Dropout(dropout),
                ), norm = nn.LayerNorm),
            ) for i_depth in range(depth)]
        )

    def forward(self, x):
        x = self.model(x)
        if self.pooling:
            x = self.patch_merge(x)
        return x

class HireMLP(nn.Module):
    def __init__(
        self,
        patch_size=4,
        in_channels=3,
        num_classes=1000,
        d_model=[64, 128, 320, 512],
        h = [4,2,2,1],
        w = [4,2,2,1],
        cross_region_step = [2,2,1,1],
        cross_region_interval = 2,
        depth=[4,6,24,3],
        expansion_factor = 2,
        patcher_norm = False,
    ):
        patch_size = pair(patch_size)
        super().__init__()
        self.patcher = PatchEmbedding(dim_in = in_channels, dim_out = d_model[0], kernel_size = 7, stride = patch_size, padding = 3, norm_layer=patcher_norm)
        

        self.layers = nn.ModuleList()
        for i_layer in range(len(depth)):
            i_depth = depth[i_layer]
            i_stage = HireMLPStage(h[i_layer], w[i_layer], d_model[i_layer], d_model_out = d_model[i_layer + 1] if (i_layer + 1 < len(depth)) else d_model[-1],
                depth = i_depth, cross_region_step = cross_region_step[i_layer], cross_region_interval = cross_region_interval,
                expansion_factor = expansion_factor, pooling = ((i_layer + 1) < len(depth)))
            self.layers.append(i_stage)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(d_model[-1]),
            Reduce('b h w c -> b c', 'mean'),
            nn.Linear(d_model[-1], num_classes)
        )

    def forward(self, x):
        embedding = self.patcher(x)
        embedding = embedding.permute(0, 2, 3, 1)
        for layer in self.layers:
            embedding = layer(embedding)
        out = self.mlp_head(embedding)
        return out
