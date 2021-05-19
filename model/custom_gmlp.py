from random import randrange
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops.layers.torch import Rearrange, Reduce

# functions

def exists(val):
    return val is not None

def dropout_layers(layers, prob_survival):
    if prob_survival == 1:
        return layers

    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0., 1.) > prob_survival

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers

# helper classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class Attention(nn.Module):
    def __init__(self, dim_in, dim_out, dim_inner, causal = False):
        super().__init__()
        self.scale = dim_inner ** -0.5
        self.causal = causal

        self.to_qkv = nn.Linear(dim_in, dim_inner * 3, bias = False)
        self.to_out = nn.Linear(dim_inner, dim_out)

    def forward(self, x):
        device = x.device
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if self.causal:
            mask = torch.ones(sim.shape[-2:], device = device).triu(1).bool()
            sim.masked_fill_(mask[None, ...], -torch.finfo(q.dtype).max)

        attn = sim.softmax(dim = -1)
        out = einsum('b i j, b j d -> b i d', attn, v)
        return self.to_out(out)

class SpatialGatingUnit(nn.Module):
    def __init__(self, dim, dim_seq, attn_dim = None, causal = False):
        super().__init__()
        dim_out = dim // 2
        self.causal = causal

        self.norm = nn.LayerNorm(dim_out)
        self.proj = nn.Conv1d(dim_seq, dim_seq, 1)
        self.attn = Attention(dim, dim_out, attn_dim, causal) if exists(attn_dim) else None
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 1.)

    def forward(self, x):
        device = x.device

        res, gate = x.chunk(2, dim = -1)
        gate = self.norm(gate)

        weight, bias = self.proj.weight, self.proj.bias
        if self.causal:
            mask = torch.ones(weight.shape[:2], device = device).triu_(1).bool()
            weight = weight.masked_fill(mask[..., None], 0.)

        gate = F.conv1d(gate, weight, bias)

        if exists(self.attn):
            gate += self.attn(x)
        return gate * res

# main classes

class gMLP(nn.Module):
    def __init__(
        self,
        *,
        num_tokens = None,
        dim,
        depth,
        seq_len,
        ff_mult = 4,
        attn_dim = None,
        prob_survival = 1.,
        causal = False
    ):
        super().__init__()
        dim_ff = dim * ff_mult
        self.seq_len = seq_len
        self.prob_survival = prob_survival

        self.to_embed = nn.Embedding(num_tokens, dim) if exists(num_tokens) else nn.Identity()

        self.layers = nn.ModuleList([Residual(nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_ff),
            nn.GELU(),
            SpatialGatingUnit(dim_ff, seq_len, attn_dim, causal),
            nn.Linear(dim_ff // 2, dim)
        )) for i in range(depth)])

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        ) if exists(num_tokens) else nn.Identity()

    def forward(self, x):
        x = self.to_embed(x)
        layers = self.layers if not self.training else dropout_layers(self.layers, self.prob_survival)
        out = nn.Sequential(*layers)(x)
        return self.to_logits(out)

class gMLPVision(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        dim,
        depth,
        ff_mult = 4,
        channels = 3,
        attn_dim = None,
        prob_survival = 1.
    ):
        super().__init__()
        assert (image_size % patch_size) == 0, 'image size must be divisible by the patch size'
        dim_ff = dim * ff_mult
        num_patches = (image_size // patch_size) ** 2

        self.to_patch_embed = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (c p1 p2)', p1 = patch_size, p2 = patch_size),
            nn.Linear(channels * patch_size ** 2, dim)
        )

        self.prob_survival = prob_survival

        self.layers = nn.ModuleList([Residual(nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim_ff),
            nn.GELU(),
            SpatialGatingUnit(dim_ff, num_patches, attn_dim),
            nn.Linear(dim_ff // 2, dim)
        )) for i in range(depth)])

        self.to_latent = nn.Identity()

    def forward(self, x):
        x = self.to_patch_embed(x)
        layers = self.layers if not self.training else dropout_layers(self.layers, self.prob_survival)
        x = nn.Sequential(*layers)(x)
        return self.to_latent(x)