import jittor as jt
from jittor import nn 
from jittor import Module
from jittor import init
from .utils import pair, check_sizes

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def execute(self, x):
        return self.fn(self.norm(x)) + x

class ParallelSum(nn.Module):
    def __init__(self, fn1, fn2, fn3):
        super().__init__()
        self.fn1 = fn1
        self.fn2 = fn2
        self.fn3 = fn3

    def execute(self, x):
        x1 = self.fn1(x)
        x2 = self.fn2(x)
        x3 = self.fn3(x)
        return x1 + x2 + x3

class ParallelWeightedSum(nn.Module):
    def __init__(self, sa, fn1, fn2, fn3):
        super().__init__()
        self.fn1 = fn1
        self.fn2 = fn2
        self.fn3 = fn3
        self.split_attention = sa

    def execute(self, x):
        x1 = self.fn1(x)
        x2 = self.fn2(x)
        x3 = self.fn3(x)
        x_all = jt.stack([x1, x2, x3], 1)
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
    
    def execute(self, x_all):
        b, k, h, w, c = x_all.shape
        x_all = x_all.reshape(b, k, -1, c)              # bs,k,n,c
        a = jt.sum(jt.sum(x_all, 1), 1)           # bs,c
        hat_a = self.mlp2(self.gelu(self.mlp1(a)))      # bs,kc
        hat_a = hat_a.reshape(b, self.k, c)             # bs,k,c
        bar_a = self.softmax(hat_a)                     # bs,k,c
        attention = bar_a.unsqueeze(-2)                 # # bs,k,1,c
        out = attention * x_all                         # # bs,k,n,c
        out = jt.sum(out, 1).reshape(b, h, w, c)
        return out

class Rearrange1(nn.Module):
    '''
    'b h w (c s) -> b w c (h s)'
    '''
    def __init__(self, segments):
        super().__init__()
        self.segments = segments
    def execute(self, x):
        b, h, w, cs = x.shape
        c = cs // self.segments
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(b, w, c, -1)
        return x

class Rearrange2(nn.Module):
    '''
    'b w c (h s) -> b h w (c s)'
    '''
    def __init__(self, segments):
        super().__init__()
        self.segments = segments
    def execute(self, x):
        b, w, c, hs = x.shape
        h = hs // self.segments
        x = x.reshape(b, w, h, -1)
        x = x.permute(0, 2, 1, 3)
        return x

class Rearrange3(nn.Module):
    '''
    'b h w (c s) -> b h c (w s)'
    '''
    def __init__(self, segments):
        super().__init__()
        self.segments = segments
    def execute(self, x):
        b, h, w, cs = x.shape
        c = cs // self.segments
        x = x.reshape(b, h, c, -1)
        return x

class Rearrange4(nn.Module):
    '''
    'b h c (w s) -> b h w (c s)'
    '''
    def __init__(self, segments):
        super().__init__()
        self.segments = segments
    def execute(self, x):
        b, h, c, ws = x.shape
        w = ws // self.segments
        x = x.view(b, h, w, -1)
        return x


class WeightedPermutator(nn.Module):
    def __init__(self, height, width, d_model, depth, segments, expansion_factor = 4, dropout = 0.):
        super().__init__()

        self.model = nn.Sequential(
            *[nn.Sequential(
                PreNormResidual(d_model, nn.Sequential(
                    ParallelWeightedSum(
                        SplitAttention(d_model, k = 3),
                        nn.Sequential(
                            Rearrange1(segments),
                            nn.Linear(height * segments, height * segments),
                            Rearrange2(segments),
                        ),
                        nn.Sequential(
                            Rearrange3(segments),
                            nn.Linear(width * segments, width * segments),
                            Rearrange4(segments),
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

    def execute(self, x):
        return self.model(x)

class Permutator(nn.Module):
    def __init__(self, height, width, d_model, depth, segments, expansion_factor = 4, dropout = 0.):
        super().__init__()

        self.model = nn.Sequential(
            *[nn.Sequential(
                PreNormResidual(d_model, nn.Sequential(
                    ParallelSum(
                        nn.Sequential(
                            Rearrange1(segments),
                            nn.Linear(height * segments, height * segments),
                            Rearrange2(segments),
                        ),
                        nn.Sequential(
                            Rearrange3(segments),
                            nn.Linear(width * segments, width * segments),
                            Rearrange4(segments),
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

    def execute(self, x):
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
            self.blocks = WeightedPermutator(height, width, d_model, depth, segments, expansion_factor = 4, dropout = 0.)
        else:
            self.blocks = Permutator(height, width, d_model, depth, segments, expansion_factor = 4, dropout = 0.)

        self.active = nn.LayerNorm(d_model)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        self.mlp_head = nn.Sequential(
            nn.Linear(d_model, num_classes)
        )

    def execute(self, x):
        patches = self.patcher(x)
        patches = patches.permute(0, 2, 3, 1)
        embedding = self.blocks(patches)
        embedding = self.active(embedding)
        embedding = embedding.permute(0, 3, 1, 2)
        embedding = self.avgpool(embedding)
        embedding = jt.flatten(embedding, 1)
        out = self.mlp_head(embedding)
        return out

