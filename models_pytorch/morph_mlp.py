import torch
from torch import nn, Tensor
from torch.nn import functional as F
from timm.models.layers import DropPath
from einops.layers.torch import Rearrange

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

    def forward(self, x: Tensor) -> Tensor:
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
        # self.proj = nn.Conv2d(C, C, 1)

        # self.reweight = MLP(C, C//4, C*3)   # instead of Add!

    def forward(self, x):
        B, C, H, W = x.shape

        need_padding_h = H % self.L > 0
        need_padding_w = W % self.L > 0
        P_l, P_r, P_t, P_b = (self.L - W % self.L) // 2, (self.L - W % self.L) - (self.L - W % self.L) // 2, (self.L - H % self.L) // 2, (self.L - H % self.L) - (self.L - H % self.L) // 2

        x_h = F.pad(x, [0, 0, P_t, P_b, 0, 0], "constant", 0) if need_padding_h else x
        x_w = F.pad(x, [P_l, P_r, 0, 0, 0, 0], "constant", 0) if need_padding_w else x
        
        x_h = self.fc_h(x_h)
        x_w = self.fc_w(x_w)
        x_c = self.fc_c(x)

        x_h = x_h[:, :, P_t:-P_b, :].contiguous() if need_padding_h else x_h
        x_w = x_w[:, :, :, P_l:-P_r].contiguous() if need_padding_w else x_w

        x = x_h + x_w + x_c

        # a = F.adaptive_avg_pool2d(x_h + x_w + x_c, output_size=1)
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

    def forward(self, x: Tensor) -> Tensor:
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

    def forward(self, x: torch.Tensor) -> Tensor:
        return self.norm(self.proj(x))


class Downsample(nn.Module):
    """Downsample transition stage"""
    def __init__(self, c1, c2):
        super().__init__()
        self.proj = nn.Conv2d(c1, c2, 3, 2, 1)
        self.norm = nn.BatchNorm2d(c2)

    def forward(self, x: Tensor) -> Tensor:
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

        # use as a backbone
        # self.out_indices = [0, 2, 4, 6]
        # for i, layer in enumerate(self.out_indices):
        #     self.add_module(f"norm{layer}", nn.BatchNorm2d(embed_dims[i]))

        self._init_weights(pretrained)

    def _init_weights(self, pretrained: str = None) -> None:
        if pretrained:
            self.load_state_dict(torch.load(pretrained, map_location='cpu')['model'])
        else:
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    if n.startswith('head'):
                        nn.init.zeros_(m.weight)
                        nn.init.zeros_(m.bias)
                    else:
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                
    def return_features(self, x):
        x = self.patch_embed(x)
        outs = []

        for i, blk in enumerate(self.network):
            x = blk(x)
            if i in self.out_indices:
                out = getattr(self, f"norm{i}")(x)
                outs.append(out)
        return outs
        
    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)          

        for blk in self.network:
            x = blk(x)

        x = self.norm(x)
        x = self.head(F.adaptive_avg_pool2d(x, output_size=1).flatten(1))
        return x

if __name__ == '__main__':
    model = MorphMLP('B')
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)