from torch import nn
from einops.layers.torch import Rearrange

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0.):
    return nn.Sequential(
        nn.Linear(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )

def MLPMixer(*, image_size, patch_size, dim, depth, num_classes, expansion_factor = 4, dropout = 0.):
    assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_size // patch_size) ** 2

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * 3, dim, 1),
        *[nn.Sequential(
            PreNormResidual(dim, nn.Sequential(
                Rearrange('b n c -> b c n'),
                FeedForward(num_patches, expansion_factor, dropout),
                Rearrange('b c n -> b n c'),
            )),
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Rearrange('b n c -> b c n'),
        nn.AdaptiveAvgPool1d(1),
        Rearrange('b c () -> b c'),
        nn.Linear(dim, num_classes)
    )