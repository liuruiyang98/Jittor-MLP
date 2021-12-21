import torch
from torch import nn
from einops.layers.torch import Reduce
from .utils import pair

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

class Spatial_Shift(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        b,w,h,c = x.size()
        x[:,1:,:,:c//4] = x[:,:w-1,:,:c//4]
        x[:,:w-1,:,c//4:c//2] = x[:,1:,:,c//4:c//2]
        x[:,:,1:,c//2:c*3//4] = x[:,:,:h-1,c//2:c*3//4]
        x[:,:,:h-1,3*c//4:] = x[:,:,1:,3*c//4:]
        return x

class S2Block(nn.Module):
    def __init__(self, d_model, depth, expansion_factor = 4, dropout = 0.):
        super().__init__()

        self.model = nn.Sequential(
            *[nn.Sequential(
                PreNormResidual(d_model, nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    Spatial_Shift(),
                    nn.Linear(d_model, d_model),
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
        x = x.permute(0, 2, 3, 1)
        x = self.model(x)
        x = x.permute(0, 3, 1, 2)
        return x

class S2MLPv1(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=[7, 2],
        in_channels=3,
        num_classes=1000,
        d_model=[192, 384],
        depth=[4, 14],
        expansion_factor = [3, 3],
    ):
        image_size = pair(image_size)
        oldps = [1, 1]
        for ps in patch_size:
            ps = pair(ps)
            assert (image_size[0] % (ps[0] * oldps[0])) == 0, 'image must be divisible by patch size'
            assert (image_size[1] % (ps[1] * oldps[1])) == 0, 'image must be divisible by patch size'
            oldps[0] = oldps[0] * ps[0]
            oldps[1] = oldps[1] * ps[1]
        assert (len(patch_size) == len(depth) == len(d_model) == len(expansion_factor)), 'patch_size/depth/d_model/expansion_factor must be a list'
        super().__init__()

        self.stage = len(patch_size)
        self.stages = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(in_channels if i == 0 else d_model[i - 1], d_model[i], kernel_size=patch_size[i], stride=patch_size[i]),
                S2Block(d_model[i], depth[i], expansion_factor[i], dropout = 0.)
            ) for i in range(self.stage)]
        )

        self.mlp_head = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(d_model[-1], num_classes)
        )

    def forward(self, x):
        embedding = self.stages(x)
        out = self.mlp_head(embedding)
        return out

def S2MLPv1_deep(num_classes: int = 1000, **kwargs):
    model = S2MLPv1(image_size=224,
                    patch_size=[16],
                    d_model=[384],
                    depth=[36],
                    num_classes=num_classes,
                    expansion_factor=[4],
                    **kwargs)
    return model

def S2MLPv1_wide(num_classes: int = 1000, **kwargs):
    model = S2MLPv1(image_size=224,
                    patch_size=[16],
                    d_model=[768],
                    depth=[12],
                    num_classes=num_classes,
                    expansion_factor=[4],
                    **kwargs)
    return model

# model = S2MLPv1_wide()
# images = torch.randn(8, 3, 224, 224)
# with torch.no_grad():
#     output = model(images)
# print(output.shape) # （8， 1000）

# total_params = sum(p.numel() for p in model.parameters())
# print(f'{total_params:,} total parameters.')
# total_trainable_params = sum(
#     p.numel() for p in model.parameters() if p.requires_grad)
# print(f'{total_trainable_params:,} training parameters.')