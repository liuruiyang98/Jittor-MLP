import jittor as jt
from jittor import nn 
from jittor import Module
from jittor import init
from .einops_my.layers.jittor import Rearrange


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

'''
Use BN instead of LN, keep consistent with WaveMLP
'''

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None) -> None:
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, out_dim, 1)

    def execute(self, x):
        return self.fc2(self.act(self.fc1(x)))


class MorphFC(nn.Module):
    def __init__(self, L, C):
        super().__init__()
        
        assert (C % L == 0)

        self.L = L
        self.C = C
        self.D = (int)(self.C / self.L)
        
        self.reshape_h = Rearrange('b (D group_C) (L group_H) w -> b (D L) (group_C group_H) w', D = self.D, L = self.L)
        self.recover_h = Rearrange('b (D L) (group_C group_H) w -> b (D group_C) (L group_H) w', D = self.D, group_C = self.L)

        self.reshape_w = Rearrange('b (D group_C) h (L group_W) -> b (D L) h (group_C group_W)', D = self.D, L = self.L)
        self.recover_w = Rearrange('b (D L) h (group_C group_W) -> b (D group_C) h (L group_W)', D = self.D, group_C = self.L)

        self.fc_h = nn.Conv2d(C, C, 1)      # L * D = C
        self.fc_w = nn.Conv2d(C, C, 1)      # L * D = C
        self.fc_c = nn.Conv2d(C, C, 1)

        # self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d(output_size=1)
        # self.proj = nn.Conv2d(C, C, 1)

        # self.reweight = MLP(C, C//4, C*3)   # instead of Add!

    def execute(self, x):
        B, C, H, W = x.shape

        need_padding_h = H % self.L > 0
        need_padding_w = W % self.L > 0
        P_l, P_r, P_t, P_b = (self.L - W % self.L) // 2, (self.L - W % self.L) - (self.L - W % self.L) // 2, (self.L - H % self.L) // 2, (self.L - H % self.L) - (self.L - H % self.L) // 2

        x_h = nn.pad(x, [0, 0, P_t, P_b, 0, 0], "constant", 0) if need_padding_h else x
        x_w = nn.pad(x, [P_l, P_r, 0, 0, 0, 0], "constant", 0) if need_padding_w else x
        
        x_h = self.fc_h(x_h)
        x_w = self.fc_w(x_w)
        x_c = self.fc_c(x)

        x_h = x_h[:, :, P_t:-P_b, :] if need_padding_h else x_h
        x_w = x_w[:, :, :, P_l:-P_r] if need_padding_w else x_w

        x = x_h + x_w + x_c

        # a = self.adaptive_avg_pool2d(x_h + x_w + x_c)
        # a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        # x = x_h * a[0] + x_w * a[1] + x_c * a[2]

        # x = self.proj(x)
        return x

    
class Block(nn.Module):
    def __init__(self, dim, L, mlp_ratio=4, dpr=0.):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = MorphFC(C = dim, L = L)
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = MLP(dim, int(dim*mlp_ratio))

    def execute(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) 
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbedOverlap(nn.Module):
    """Image to Patch Embedding with overlapping
    """
    def __init__(self, patch_size=16, stride=16, padding=0, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, patch_size, stride, padding)
        self.norm = nn.BatchNorm2d(embed_dim)

    def execute(self, x):
        return self.norm(self.proj(x))


class Downsample(nn.Module):
    """Downsample transition stage"""
    def __init__(self, c1, c2):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 3, 2, 1)
        self.norm = nn.BatchNorm2d(c2)

    def execute(self, x):
        return self.norm(self.proj(x))


morphmlp_settings = {
    'T': [[3, 4, 7, 3], [4, 4, 4, 4], [84, 168, 336, 588], [14, 28, 28, 49], [0.1, 0.1, 0.1, 0.1]],       # [layers]
    'S': [[3, 4, 9, 3], [4, 4, 4, 4], [112, 224, 392, 784], [14, 28, 28, 49], [0.1, 0.1, 0.1, 0.1]],
    'B': [[4, 6, 15 ,4], [4, 4, 4, 4], [112, 224, 392, 784], [14, 28, 28, 49], [0.3, 0.3, 0.3, 0.3]],
    'L': [[4, 8, 18, 6], [4, 4, 4, 4], [112, 224, 392, 784], [14, 28, 28, 49], [0.4, 0.4, 0.4, 0.4]]
}


class MorphMLP(nn.Module):     
    def __init__(self, model_name: str = 'T', pretrained: str = None, num_classes: int = 1000, *args, **kwargs) -> None:
        super().__init__()
        assert model_name in morphmlp_settings.keys(), f"WaveMLP model name should be in {list(morphmlp_settings.keys())}"
        layers, mlp_ratios, embed_dims, chunk_len, stoch_drop = morphmlp_settings[model_name]
    
        self.patch_embed = PatchEmbedOverlap(7, 4, 2, embed_dims[0])

        network = []

        for i in range(len(layers)):
            stage = nn.Sequential(*[
                Block(embed_dims[i], chunk_len[i], mlp_ratios[i], stoch_drop[i])
            for _ in range(layers[i])])
            
            network.append(stage)
            if i >= len(layers) - 1: break
            network.append(Downsample(embed_dims[i], embed_dims[i+1]))

        self.network = nn.ModuleList(network)
        self.norm = nn.BatchNorm2d(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes)

        self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d(output_size=1)

        # use as a backbone
        # self.out_indices = [0, 2, 4, 6]
        # for i, layer in enumerate(self.out_indices):
        #     self.add_module(f"norm{layer}", nn.BatchNorm2d(embed_dims[i]))

        self._init_weights(pretrained)

    def _init_weights(self, pretrained: str = None) -> None:
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if n.startswith('head'):
                    nn.init.zero_(m.weight)
                    nn.init.zero_(m.bias)
                else:
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zero_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.one_(m.weight)
                nn.init.zero_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zero_(m.bias)
                
    def return_features(self, x):
        x = self.patch_embed(x)
        outs = []

        for i, blk in enumerate(self.network):
            x = blk(x)
            if i in self.out_indices:
                out = getattr(self, f"norm{i}")(x)
                outs.append(out)
        return outs
        
    def execute(self, x):
        x = self.patch_embed(x)          

        for blk in self.network:
            x = blk(x)

        x = self.norm(x)
        x = self.head(self.adaptive_avg_pool2d(x).flatten(1))
        return x
