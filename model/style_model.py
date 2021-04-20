from einops import repeat, rearrange
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
from vit_pytorch.vit import PreNorm, FeedForward, Attention


class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, eps = 1e-8, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        self.eps = eps
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x

class StyleFormer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

    def forward(self, x, styles):
        for idx, (ln, attn, ff) in enumerate(self.layers):
            x = x + styles[idx]
            x = ln(x)
            x = attn(x)
            x = ff(x) + x
        return x


class StyleViT(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0.0, emb_dropout = 0.0):
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
                nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding='same', groups=3, bias=False),
                nn.Linear(patch_dim, dim)
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
        for conv, style in self.style_convs:
            z = conv(z)
            y = self.to_patch(z)
            y = style(y)
            y = torch.cat((cls_tokens, y), dim=1)
            styles.append(y)
        x = self.transformer(x, styles)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        return self.to_latent(x)

if __name__ == '__main__':
    main()