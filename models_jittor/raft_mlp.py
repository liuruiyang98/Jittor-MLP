import math
from functools import reduce
from typing import List, Dict
from abc import ABC

from .einops_my.layers.jittor import Rearrange, Reduce
import jittor as jt
from jittor import nn 
from jittor import Module
from jittor import init


PATCH_SIZE = "patch_size"
RAFT_SIZE = "raft_size"
LOGGER_NAME = "RaftMLP"
DIM = "dim"
DEPTH = "depth"
SER_PM = "ser_pm"
SEP_LN_CODIM_TM = "sep_ln_codim_tm"
SEP_LN_CH_TM = "sep_ln_ch_tm"
ORIGINAL_TM = "original_tm"

TOKEN_MIXING_TYPES = [
    SER_PM,
    SEP_LN_CODIM_TM,
    SEP_LN_CH_TM,
    ORIGINAL_TM,
]


####################################### Modules #######################################
class DropPath(nn.Module):
    def __init__(self, drop_path_rate=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_path_rate

    def execute(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random = keep_prob + jt.rand(shape, dtype=x.dtype)
        random.floor_()
        return x.div(keep_prob) * random


class Block(nn.Module):
    def __init__(
        self, dim, expansion_factor=4, dropout=0.0, drop_path_rate=0.0
    ):
        super().__init__()
        self.norm = nn.Identity()
        self.drop = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        self.fn = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(dropout),
        )

    def execute(self, x):
        return self.drop(self.fn(self.norm(x))) + x


class ChannelBlock(Block):
    def __init__(
        self, dim, expansion_factor=4, dropout=0.0, drop_path_rate=0.0
    ):
        super().__init__(dim, expansion_factor, dropout, drop_path_rate)
        self.norm = nn.LayerNorm(dim)


class TokenBlock(Block):
    def __init__(
        self,
        dim,
        channels,
        expansion_factor=4,
        dropout=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__(dim, expansion_factor, dropout, drop_path_rate)
        self.norm = nn.Sequential(
            *[
                Rearrange("b c o -> b o c"),
                nn.LayerNorm(channels),
                Rearrange("b o c -> b c o"),
            ]
        )


class SpatiallySeparatedTokenBlock(Block):
    def __init__(
        self,
        dim,
        channels,
        expansion_factor=4,
        dropout=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__(dim, expansion_factor, dropout, drop_path_rate)
        self.norm = nn.Sequential(
            *[
                Rearrange("b (c o1) o2 -> b (o1 o2) c", c=channels, o2=dim),
                nn.LayerNorm(channels),
                Rearrange("b (o1 o2) c -> b (c o1) o2", c=channels, o2=dim),
            ]
        )


class PermutedBlock(Block):
    def __init__(
        self,
        spatial_dim,
        channels,
        raft_size,
        expansion_factor=4,
        dropout=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__(
            spatial_dim * raft_size,
            expansion_factor,
            dropout,
            drop_path_rate,
        )
        self.norm = nn.Sequential(
            *[
                Rearrange(
                    "b (c1 o1) (c2 o2) -> b (o1 o2) (c1 c2)",
                    c1=channels // raft_size,
                    c2=raft_size,
                    o2=spatial_dim,
                ),
                nn.LayerNorm(channels),
                Rearrange(
                    "b (o1 o2) (c1 c2) -> b (c1 o1) (c2 o2)",
                    c1=channels // raft_size,
                    c2=raft_size,
                    o2=spatial_dim,
                ),
            ]
        )


class Level(nn.Module, ABC):
    def __init__(self, image_size=224, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.fn = nn.Identity()
        self._bh = self._bw = image_size // patch_size
        self._h = self._w = math.ceil(image_size / patch_size)

    def execute(self, input):
        if not (self._bh == self._h and self._bw == self._w):
            input = nn.interpolate(
                input,
                (self._h * self.patch_size, self._w * self.patch_size),
                mode="bilinear",
                align_corners=False,
            )
        return self.fn(input)


class SeparatedLNCodimLevel(Level):
    def __init__(
        self,
        in_channels,
        out_channels,
        depth=4,
        image_size=224,
        patch_size=4,
        token_expansion_factor=2,
        channel_expansion_factor=4,
        dropout=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__(image_size, patch_size)
        self.fn = nn.Sequential(
            *[
                Rearrange(
                    "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                    p1=patch_size,
                    p2=patch_size,
                ),
                nn.Linear((patch_size ** 2) * in_channels, out_channels)
                if patch_size != 1
                or (patch_size == 1 and in_channels == out_channels)
                else nn.Identity(),
                *[
                    nn.Sequential(
                        *[
                            # vertical mixer
                            Rearrange("b (h w) c -> b (c w) h", h=self._h),
                            TokenBlock(
                                self._h,
                                out_channels * self._w,
                                token_expansion_factor,
                                dropout,
                                drop_path_rate,
                            ),
                            # horizontal mixer
                            Rearrange(
                                "b (c w) h -> b (c h) w", h=self._h, w=self._w
                            ),
                            TokenBlock(
                                self._w,
                                out_channels * self._h,
                                token_expansion_factor,
                                dropout,
                                drop_path_rate,
                            ),
                            # channel mixer
                            Rearrange(
                                "b (c h) w -> b (h w) c", h=self._h, w=self._w
                            ),
                            ChannelBlock(
                                out_channels,
                                channel_expansion_factor,
                                dropout,
                                drop_path_rate,
                            ),
                        ]
                    )
                    for _ in range(depth)
                ],
                Rearrange("b (h w) c -> b c h w", h=self._h, w=self._w),
            ]
        )


class SeparatedLNChannelLevel(Level):
    def __init__(
        self,
        in_channels,
        out_channels,
        depth=4,
        image_size=224,
        patch_size=4,
        token_expansion_factor=2,
        channel_expansion_factor=4,
        dropout=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__(image_size, patch_size)
        self.fn = nn.Sequential(
            *[
                Rearrange(
                    "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                    p1=patch_size,
                    p2=patch_size,
                ),
                nn.Linear((patch_size ** 2) * in_channels, out_channels)
                if patch_size != 1
                or (patch_size == 1 and in_channels == out_channels)
                else nn.Identity(),
                *[
                    nn.Sequential(
                        *[
                            # vertical mixer
                            Rearrange("b (h w) c -> b (c w) h", h=self._h),
                            SpatiallySeparatedTokenBlock(
                                self._h,
                                out_channels,
                                token_expansion_factor,
                                dropout,
                                drop_path_rate,
                            ),
                            # horizontal mixer
                            Rearrange(
                                "b (c w) h -> b (c h) w", h=self._h, w=self._w
                            ),
                            SpatiallySeparatedTokenBlock(
                                self._w,
                                out_channels,
                                token_expansion_factor,
                                dropout,
                                drop_path_rate,
                            ),
                            # channel mixer
                            Rearrange(
                                "b (c h) w -> b (h w) c", h=self._h, w=self._w
                            ),
                            ChannelBlock(
                                out_channels,
                                channel_expansion_factor,
                                dropout,
                                drop_path_rate,
                            ),
                        ]
                    )
                    for _ in range(depth)
                ],
                Rearrange("b (h w) c -> b c h w", h=self._h, w=self._w),
            ]
        )


class SerialPermutedLevel(Level):
    def __init__(
        self,
        in_channels,
        out_channels,
        depth=4,
        image_size=224,
        patch_size=4,
        token_expansion_factor=2,
        channel_expansion_factor=4,
        dropout=0.0,
        drop_path_rate=0.0,
        raft_size=4,
    ):
        super().__init__(image_size, patch_size)

        assert out_channels % raft_size == 0
        self.fn = nn.Sequential(
            *[
                Rearrange(
                    "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                    p1=patch_size,
                    p2=patch_size,
                ),
                nn.Linear((patch_size ** 2) * in_channels, out_channels)
                if patch_size != 1
                or (patch_size == 1 and in_channels == out_channels)
                else nn.Identity(),
                *[
                    nn.Sequential(
                        *[
                            # vertical-channel mixer
                            Rearrange(
                                "b (h w) (chw co) -> b (co w) (chw h)",
                                h=self._h,
                                w=self._w,
                                chw=raft_size,
                            ),
                            PermutedBlock(
                                self._h,
                                out_channels,
                                raft_size,
                                token_expansion_factor,
                                dropout,
                                drop_path_rate,
                            ),
                            # horizontal-channel mixer
                            Rearrange(
                                "b (co w) (chw h) -> b (co h) (chw w)",
                                h=self._h,
                                w=self._w,
                                chw=raft_size,
                            ),
                            PermutedBlock(
                                self._w,
                                out_channels,
                                raft_size,
                                token_expansion_factor,
                                dropout,
                                drop_path_rate,
                            ),
                            # channel mixer
                            Rearrange(
                                "b (co h) (chw w) -> b (h w) (chw co)",
                                h=self._h,
                                w=self._w,
                                chw=raft_size,
                            ),
                            ChannelBlock(
                                out_channels,
                                channel_expansion_factor,
                                dropout,
                                drop_path_rate,
                            ),
                        ]
                    )
                    for _ in range(depth)
                ],
                Rearrange("b (h w) c -> b c h w", h=self._h, w=self._w),
            ]
        )


class OriginalLevel(Level):
    def __init__(
        self,
        in_channels,
        out_channels,
        depth=4,
        image_size=224,
        patch_size=4,
        token_expansion_factor=2,
        channel_expansion_factor=4,
        dropout=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__(image_size, patch_size)
        self.fn = nn.Sequential(
            *[
                Rearrange(
                    "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                    p1=patch_size,
                    p2=patch_size,
                ),
                nn.Linear((patch_size ** 2) * in_channels, out_channels),
                *[
                    nn.Sequential(
                        *[
                            # token mixer
                            Rearrange(
                                "b (h w) c -> b c (h w)", h=self._h, w=self._w
                            ),
                            TokenBlock(
                                self._h * self._w,
                                out_channels,
                                token_expansion_factor,
                                dropout,
                                drop_path_rate,
                            ),
                            # channel mixer
                            Rearrange(
                                "b c (h w) -> b (h w) c", h=self._h, w=self._w
                            ),
                            ChannelBlock(
                                out_channels,
                                channel_expansion_factor,
                                dropout,
                                drop_path_rate,
                            ),
                        ]
                    )
                    for _ in range(depth)
                ],
                Rearrange("b (h w) c -> b c h w", h=self._h, w=self._w),
            ]
        )



####################################### RaftMLP #######################################
class RaftMLP(nn.Module):
    def __init__(
        self,
        layers: List[Dict],
        in_channels: int = 3,
        image_size: int = 224,
        num_classes: int = 1000,
        token_expansion_factor: int = 2,
        channel_expansion_factor: int = 4,
        dropout: float = 0.0,
        token_mixing_type: str = SER_PM,
        shortcut: bool = True,
        gap: bool = True,
        drop_path_rate: float = 0.0,
    ):
        assert token_mixing_type in TOKEN_MIXING_TYPES
        for i, layer in enumerate(layers):
            assert DEPTH in layer
            assert DIM in layer
            assert PATCH_SIZE in layer
            assert token_mixing_type != SER_PM or RAFT_SIZE in layer
            assert 0 < layer.get(DIM)
        super().__init__()
        self.layers = layers
        self.shortcut = shortcut
        self.gap = gap
        if token_mixing_type == ORIGINAL_TM:
            level = OriginalLevel
        elif token_mixing_type == SEP_LN_CODIM_TM:
            level = SeparatedLNCodimLevel
        elif token_mixing_type == SEP_LN_CH_TM:
            level = SeparatedLNChannelLevel
        else:
            level = SerialPermutedLevel
        levels = []
        heads = []
        for i, layer in enumerate(self.layers):
            params = {
                "in_channels": in_channels
                if i == 0
                else self.layers[i - 1].get(DIM),
                "out_channels": layer.get(DIM),
                "depth": layer.get(DEPTH),
                "image_size": image_size,
                "patch_size": layer.get(PATCH_SIZE),
                "token_expansion_factor": token_expansion_factor,
                "channel_expansion_factor": channel_expansion_factor,
                "dropout": dropout,
                "drop_path_rate": drop_path_rate,
            }
            if token_mixing_type == SER_PM:
                params["raft_size"] = layer.get(RAFT_SIZE)
            levels.append(level(**params))
            heads_seq = []
            if self.shortcut or len(self.layers) == i + 1:
                heads_seq.append(Rearrange("b c h w -> b h w c"))
                heads_seq.append(nn.LayerNorm(layer.get(DIM)))
                heads_seq.append(Rearrange("b h w c -> b c h w"))
                if gap or len(self.layers) != i + 1:
                    heads_seq.append(Reduce("b c h w -> b c", "mean"))
                if len(self.layers) != i + 1:
                    heads_seq.append(
                        nn.Linear(layer.get(DIM), self.layers[-1].get(DIM) * 2)
                    )
                heads.append(nn.Sequential(*heads_seq))
            image_size = math.ceil(image_size / layer.get(PATCH_SIZE))
        self.levels = nn.ModuleList(levels)
        self.heads = nn.ModuleList(heads)
        self.classifier = nn.Linear(
            self.layers[-1].get(DIM)
            if gap
            else self.layers[-1].get(DIM) * (image_size ** 2),
            num_classes,
        )
        # if not gap:
        #     self.flatten = nn.Flatten()

    def execute(self, input):
        output = []
        for i, layer in enumerate(self.layers):
            input = self.levels[i](input)
            if self.shortcut:
                output.append(self.heads[i](input))
        if not self.shortcut:
            output = self.heads[0](input)
        else:
            output = (
                reduce(
                    lambda a, b: b[:, : self.layers[-1].get(DIM)] * a
                    + b[:, self.layers[-1].get(DIM) :],
                    output[::-1],
                )
                if self.gap
                else reduce(
                    lambda a, b: b[:, : self.layers[-1].get(DIM)].view(
                        -1, self.layers[-1].get(DIM), 1, 1
                    )
                    * a
                    + b[:, self.layers[-1].get(DIM) :].view(
                        -1, self.layers[-1].get(DIM), 1, 1
                    ),
                    output[::-1],
                )
            )
        # if not self.gap:
        #     output = self.flatten(output)
        if not self.gap:
            output = output.flatten()
        return self.classifier(output)