import torch.nn as nn
from einops.layers.torch import Rearrange
from .utils import pair, check_sizes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class ConvMixer(nn.Module):
    def __init__(self, dim, depth, kernel_size=9, patch_size=7, n_classes=1000):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        self.blocks = nn.Sequential(
            *[nn.Sequential(
                    Residual(nn.Sequential(
                        nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                        nn.GELU(),
                        nn.BatchNorm2d(dim)
                    )),
                    nn.Conv2d(dim, dim, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
            ) for i in range(depth)],
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(dim, n_classes)
        )
    
    def forward(self, x):
        embedding = self.embedding(x)
        embedding = self.blocks(embedding)
        out = self.classifier(embedding)
        return out


class ConvMixerV2(nn.Module):
    def __init__(self, dim, depth, kernel_size=9, patch_size=7, image_size=224, n_classes=1000):
        super().__init__()
        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        num_patches = check_sizes(image_size, patch_size)
        assert (dim % (self.patch_size[0] * self.patch_size[1])) == 0, 'dim must be divisible by patch size * patch size'

        self.embedding = nn.Sequential(
            nn.Conv2d(3, dim // (self.patch_size[0] * self.patch_size[1]), kernel_size=7, stride=1, padding=3),
            nn.GELU(),
            nn.BatchNorm2d(dim // (self.patch_size[0] * self.patch_size[1])),
            Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = self.patch_size[0], p2 = self.patch_size[1]),
        )
        self.blocks = nn.Sequential(
            *[nn.Sequential(
                    Residual(nn.Sequential(
                        nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                        nn.GELU(),
                        nn.BatchNorm2d(dim)
                    )),
                    nn.Conv2d(dim, dim, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
            ) for i in range(depth)],
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(dim, n_classes)
        )
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.blocks(x)
        x = self.classifier(x)
        return x