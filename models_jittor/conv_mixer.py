import jittor as jt
from jittor import nn 
from jittor import Module
from jittor import init
from .utils import pair, check_sizes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def execute(self, x):
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
                        nn.Conv2d(dim, dim, kernel_size, groups=dim, padding=(int)((kernel_size - 1) // 2)),
                        nn.GELU(),
                        nn.BatchNorm2d(dim)
                    )),
                    nn.Conv2d(dim, dim, kernel_size=1),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
            ) for i in range(depth)],
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        self.classifier = nn.Sequential(
            nn.Linear(dim, n_classes)
        )
    
    def execute(self, x):
        embedding = self.embedding(x)
        embedding = self.blocks(embedding)
        embedding = self.avgpool(embedding)
        embedding = jt.flatten(embedding, 1)
        out = self.classifier(embedding)
        return out