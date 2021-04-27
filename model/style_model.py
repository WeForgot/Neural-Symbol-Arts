from einops import repeat, rearrange
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
from vit_pytorch.vit import PreNorm, FeedForward, Attention

class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding = 0, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class StyleFormer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.InstanceNorm1d(dim),
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x, styles):
        for idx, (ln, attn, ff) in enumerate(self.layers):
            x = x + ln(styles[idx])
            x = attn(x)
            x = ff(x) + x
        return x


class StyleViT(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'cls', channels = 4, dim_head = 64, dropout = 0.0, emb_dropout = 0.0, num_latents = 1):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert pool in {'cls', 'mean'}, 'Pool type must be either cls (token) or mean'
        self.to_patch = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)
        self.patch_emb = nn.Linear(patch_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.style_convs = nn.ModuleList([
            nn.ModuleList([
                DepthWiseConv2d(dim_in=channels, dim_out=channels, kernel_size=5, padding='same', bias=False),
                nn.Linear(patch_dim, dim, bias=False),
                nn.Sequential(*([nn.Linear(dim, dim, bias=False) for _ in range(num_latents)]))
            ]) for _ in range(depth)
        ])
        self.transformer = StyleFormer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()
    
    def forward(self, img):
        styles = []
        x = img
        x = self.to_patch(img)
        x = self.patch_emb(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        z = img
        for conv, proj, style in self.style_convs:
            z = conv(z)
            y = self.to_patch(z)
            y = proj(y)
            y = style(y)
            y = torch.cat((torch.zeros_like(cls_tokens), y), dim=1)
            styles.append(y)
        x = self.transformer(x, styles)
        return self.to_latent(x)

if __name__ == '__main__':
    main()