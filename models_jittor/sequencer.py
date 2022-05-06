import jittor as jt
from jittor import nn 
from .einops_my.layers.jittor import Reduce

sequencer_settings = {
    'S': [[4, 3,  8, 3], [192, 384, 384, 384], [48, 96, 96, 96], 3],
    'M': [[4, 3, 14, 3], [192, 384, 384, 384], [48, 96, 96, 96], 3],
    'L': [[8, 8, 16, 4], [192, 384, 384, 384], [48, 96, 96, 96], 3]
}

class PatchEmbedOverlap(nn.Module):
    """Image to Patch Embedding with overlapping
    """
    def __init__(self, patch_size=16, stride=16, padding=0, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, patch_size, stride, padding)
        self.norm = nn.BatchNorm2d(embed_dim)

    def execute(self, x):
        return self.norm(self.proj(x))

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def execute(self, x):
        return self.fn(self.norm(x)) + x

class BiLSTM2D(nn.Module):
    def __init__(self, d_model, hidden_d_model):
        super().__init__()
        self.rnn_v = nn.LSTM(d_model, hidden_d_model, num_layers = 1, batch_first = True, bias = True, bidirectional = True)
        self.rnn_h = nn.LSTM(d_model, hidden_d_model, num_layers = 1, batch_first = True, bias = True, bidirectional = True)
        self.fc = nn.Linear(4 * hidden_d_model, d_model)
    
    def execute(self, x):
        B, H, W, C = x.shape
        v, _ = self.rnn_v(x.permute(0, 2, 1, 3).reshape(-1, H, C))
        v = v.reshape(B, W, H, -1).permute(0, 2, 1, 3)
        h, _ = self.rnn_h(x.reshape(-1, W, C))
        h = h.reshape(B, H, W, -1)
        x = jt.concat([v, h], dim = -1)
        x = self.fc(x)
        return x


class Sequencer2DBlock(nn.Module):
    def __init__(self, d_model, depth, hidden_d_model, expansion_factor = 3, dropout = 0.):
        super().__init__()

        self.model = nn.Sequential(
            *[nn.Sequential(
                PreNormResidual(d_model, nn.Sequential(
                    BiLSTM2D(d_model, hidden_d_model)
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
        x = x.permute(0, 2, 3, 1)
        x = self.model(x)
        x = x.permute(0, 3, 1, 2)
        return x

class Sequencer2D(nn.Module):
    def __init__(self, model_name: str = 'T', pretrained: str = None, num_classes: int = 1000, in_channels = 3, *args, **kwargs) -> None:
        super().__init__()
        assert model_name in sequencer_settings.keys(), f"Sequencer model name should be in {list(sequencer_settings.keys())}"
        depth, embed_dims, hidden_dims, expansion_factor = sequencer_settings[model_name]

        self.patch_size = [7, 2, 1, 1]

        self.stage = len(depth)
        self.stages = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(in_channels if i == 0 else embed_dims[i - 1], embed_dims[i], kernel_size=self.patch_size[i], stride=self.patch_size[i]),
                Sequencer2DBlock(embed_dims[i], depth[i], hidden_dims[i], expansion_factor, dropout = 0.)
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

if __name__ == '__main__':
    model = Sequencer2D('L')
    images = jt.randn(8, 3, 224, 224)

    with jt.no_grad():
        output = model(images)
    print(output.shape)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')