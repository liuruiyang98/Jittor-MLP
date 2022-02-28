import jittor as jt
from jittor import nn 
from jittor import Module
from jittor import init
from .utils import pair

from .einops_my.layers.jittor import Rearrange, Reduce


############################################## Drop Path ##############################################
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + jt.rand(shape, dtype=x.dtype)
    random_tensor.floor()  # binarize
    output = jt.divide(x, keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def execute(self, x):
        return drop_path(x, self.drop_prob, self.is_training)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def execute(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

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
    def execute(self, x):
        return self.net(x)

class DynaMixerOp_w(nn.Module):
    def __init__(self, w, dim, hidden_dim, segment):
        super().__init__()
        self.segment = segment
        self.reshape = Rearrange('b h w (s d) -> b h s w d', s = segment)

        self.Wd = nn.ModuleList([nn.Linear(dim, hidden_dim) for i in range(segment)])
        self.attend = nn.Sequential(
            Rearrange('b h w (s d) -> b h s (w d)', s = segment),
            nn.Linear(int(hidden_dim * w), w * w),
            Rearrange('b h s (w1 w2) -> b h s w1 w2', w1 = w),
            nn.Softmax(dim = -1),
        )
        self.recover = Rearrange('b h s w d -> b h w (s d)', s = segment)
        self.proc = nn.Linear(dim, dim)

    def execute(self, x):
        # b h w d = X.shape
        input = x

        x_ = []
        for i in range(self.segment):
            x_.append(self.Wd[i](x))
        x_ = jt.concat(x_, -1)
        attn = self.attend(x_)

        input = self.reshape(input)
        x = jt.matmul(attn, input)
        x = self.recover(x)
        return self.proc(x)

class DynaMixerOp_h(nn.Module):
    def __init__(self, h, dim, hidden_dim, segment):
        super().__init__()
        self.segment = segment
        self.reshape = Rearrange('b h w (s d) -> b w s h d', s = segment)

        self.Wd = nn.ModuleList([nn.Linear(dim, hidden_dim) for i in range(segment)])
        self.attend = nn.Sequential(
            Rearrange('b h w (s d) -> b w s (h d)', s = segment),
            nn.Linear(int(hidden_dim * h), h * h),
            Rearrange('b w s (h1 h2) -> b w s h1 h2', h1 = h),
            nn.Softmax(dim = -1),
        )
        self.recover = Rearrange('b w s h d -> b h w (s d)', s = segment)
        self.proc = nn.Linear(dim, dim)

    def execute(self, x):
        # b h w d = X.shape
        input = x

        x_ = []
        for i in range(self.segment):
            x_.append(self.Wd[i](x))
        x_ = jt.concat(x_, -1)
        attn = self.attend(x_)

        input = self.reshape(input)
        x = jt.matmul(attn, input)
        x = self.recover(x)
        return self.proc(x)


class DynaBlock(nn.Module):
    def __init__(self, h, w, dim, hidden_dim_DMO = 2, segment = 8):
        super().__init__()
        self.proj_c = nn.Linear(dim, dim)
        self.proj_o = nn.Linear(dim, dim)

        self.DynaMixerOp_w = DynaMixerOp_w(w, dim, hidden_dim_DMO, segment)
        self.DynaMixerOp_h = DynaMixerOp_h(h, dim, hidden_dim_DMO, segment)

    def execute(self, x):
        Y_c = self.proj_c(x)
        Y_h = self.DynaMixerOp_h(x)
        Y_w = self.DynaMixerOp_w(x)
        Y_out = Y_h + Y_w + Y_c
        Y_out = self.proj_o(Y_out)
        return Y_out

class DynaMLPBlock(nn.Module):
    def __init__(self, depth, h, w, dim, hidden_dim_DMO, segment, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()

        self.reshape = Rearrange('b c h w -> b h w c')
        self.recover = Rearrange('b h w c -> b c h w')
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, DynaBlock(h, w, dim, hidden_dim_DMO, segment)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = 0.)),
            ]))
    def execute(self, x):
        x = self.reshape(x)
        for attn, ff in self.layers:
            x = self.drop_path(attn(x)) + x
            x = self.drop_path(ff(x)) + x
        x = self.recover(x)
        return x


dynamlp_settings = {
    'T': [[7, 2], [192, 384], [4, 14], [8, 16], 3, 0.1, 2],       # [layers]
    'M': [[7, 2], [256, 512], [7, 17], [8, 16], 3, 0.1, 2],
    'L': [[7, 2], [256, 512], [9, 27], [8, 16], 3, 0.3, 8],
}

class DynaMixer(nn.Module):
    def __init__(self, model_name: str = 'M', image_size = 224, in_channels: int = 3, num_classes: int = 1000):
        super().__init__()
        assert model_name in dynamlp_settings.keys(), f"DynaMLP model name should be in {list(dynamlp_settings.keys())}"
        patch_size, embed_dims, depths, segment, mlp_ratio, dropout, hidden_dim_DMO = dynamlp_settings[model_name]

        image_height, image_width = pair(image_size)
        h = []
        w = []
        oldps = [1, 1]
        for ps in patch_size:
            ps = pair(ps)
            try:
                h.append(int(h[-1] / ps[0]))
                w.append(int(w[-1] / ps[1]))
            except:
                h.append((int)(image_height / ps[0]))
                w.append((int)(image_width / ps[1]))
            assert (image_height % (ps[0] * oldps[0])) == 0, 'image must be divisible by patch size'
            assert (image_width % (ps[1] * oldps[1])) == 0, 'image must be divisible by patch size'
            oldps[0] = oldps[0] * ps[0]
            oldps[1] = oldps[1] * ps[1]


        self.stage = len(patch_size)
        self.stages = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(in_channels if i == 0 else embed_dims[i - 1], embed_dims[i], kernel_size=patch_size[i], stride=patch_size[i]),
                DynaMLPBlock(depth = depths[i], h = h[i], w = w[i], dim = embed_dims[i], hidden_dim_DMO = hidden_dim_DMO, segment = segment[i], 
                    mlp_dim = embed_dims[i] * mlp_ratio, dropout = dropout)
            ) for i in range(self.stage)]
        )
        
        self.mlp_head = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(embed_dims[-1], num_classes)
        )

    def execute(self, x):
        embedding = self.stages(x)
        out = self.mlp_head(embedding)
        return out

