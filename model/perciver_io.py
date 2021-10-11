import torch
import torch.nn as nn

class PerceiverEncoder(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mha_norm = nn.LayerNorm(dim)
        self.ff = nn.Linear(dim, dim)
        self.ff_norm = nn.LayerNorm(dim)
    
    def forward(self, x, latent):
        x1 = self.mha(latent, x, x)
        x = self.mha_norm(x1 + x)
        x1 = self.ff(x)
        x = self.ff_norm(x1 + x)
        return x

class PerceiverProcessor(nn.Module):
    def __init__(self, dim, heads, depth):
        super().__init__()
        self.layers = []
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList([
                    nn.MultiheadAttention(dim, heads, batch_first=True),
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim),
                    nn.LayerNorm(dim)
                ])
            )
    
    def forward(self, x):
        for mha, mha_norm, ff, ff_norm in self.layers:
            x1 = mha(x, x, x)
            x = mha_norm(x + x1)
            x1 = ff(x)
            x = ff_norm(x + x1)
        return x

class PerceiverDecoder(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mha_norm = nn.LayerNorm(dim)
        self.ff = nn.Linear(dim, dim)
        self.ff_norm = nn.LayerNorm(dim)
    
    def forward(self, x, output):
        x1 = self.mha(output, x, x)
        x = self.mha_norm(x1 + x)
        x1 = self.ff(x)
        x = self.ff_norm(x + x1)
        return x

class PerceiverIO(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
