import torch
import torch.nn as nn

from einops.layers.torch import Rearrange

from axial_positional_embedding import AxialPositionalEmbedding

class TorchEncoder(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, channels=3, emb_dropout = 0.1):
        super(TorchEncoder, self).__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.to_patch = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)
        self.patch_emb = nn.Linear(patch_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model = dim,
                nhead = heads,
                activation = 'gelu'
            ),
            num_layers = depth
        )
    
    def forward(self, x):
        x = self.to_patch(x)
        x = self.patch_emb(x)
        b, n, _ = x.shape
        x += self.pos_embedding[:,:n]
        x = self.dropout(x)
        x = self.transformer(x)
        return x

class TorchDecoder(nn.Module):
    def __init__(self, dim, depth, heads):
        super(TorchDecoder, self).__init__()
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model = dim,
                nhead = heads,
                activation = 'gelu'
            ),
            num_layers = depth
        )
    
    def forward(self, x, context, input_mask):
        x = x.permute(1,0,2)
        #input_mask = input_mask.permute(1,0)
        output = self.transformer(x, context, tgt_key_padding_mask=input_mask)
        return output.permute(1,0,2)
        
