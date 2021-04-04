import numpy as np
import torch
import torch.nn as nn
from x_transformers import ViTransformerWrapper, Encoder, Decoder, AutoregressiveWrapper
from x_transformers.x_transformers import AbsolutePositionalEmbedding, AttentionLayers, always
def top_p(logits, thres = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

# topk

def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

class TransformerWrapper(nn.Module):
    def __init__(self, *, max_seq_len, attn_layers, emb_dim, use_pos_emb = True, emb_dropout = 0.0):
        super().__init__()
        assert isinstance(attn_layers, AttentionLayers), 'Attention layers must either be an Encoder or Decoder'
        # MIGHT BE SOME ISSUES SINCE I MOVED EMB_DIM TO A VARIABLE
        dim = attn_layers.dim
        self.max_seq_len = 226
        self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len) if (use_pos_emb and not attn_layers.has_pos_emb) else always(0)
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)
        self.to_logits = nn.Linear(dim, emb_dim)
    
    def forward(self, x, mask=None, return_attn=False, **kwargs):
        x += self.pos_emb(x)
        x = self.emb_dropout(x)
        x = self.project_emb(x)
        x, intermediates = self.attn_layers(x, mask=mask, return_hiddens=True, **kwargs)
        x = self.norm(x)
        out = self.to_logits(x)
        return out

class BasicNSA(nn.Module):
    def __init__(self, row_len, dim = 128, patch_size=32, emb_dropout=0.):
        super(BasicNSA, self).__init__()
        sa_size = (576, 288)
        saml_sz = (226, row_len) # 8 embeddings + 3 color channels + alpha + ltx + lty + tbx + lby + rtx + rty + rbx + rby
        assert max(sa_size) % patch_size == 0, 'Patch size must be a multiple of the max image dimension'
        self.row_len = row_len
        self.encoder = ViTransformerWrapper(
            image_size = max(sa_size),
            patch_size = patch_size,
            attn_layers = Encoder(
                dim = 512,
                depth = 6,
                heads = 8
            )
        )

        self.decoder = TransformerWrapper(
            max_seq_len = 226,
            attn_layers = Decoder(
                dim = 512,
                depth = 6,
                heads = 8
            ),
            emb_dim=row_len
        )
    

    def forward(self, src, tgt, tgt_mask=None):
        enc = self.encoder(src, return_embeddings=True)
        out = self.decoder(tgt, context=enc, mask=tgt_mask.bool())
        return out
    
    @torch.no_grad()
    def generate(self, src, max_seq_len):
        src = torch.unsqueeze(src, 0)
        enc = self.encoder(src, return_embeddings=True)
        out = torch.zeros((1,226,self.row_len), device=src.device)
        mask = torch.zeros((1,226,), dtype=torch.bool, device=out.device)
        idx = 0
        while idx < 226:
            mask[:,idx] = 1
            x = self.decoder(out, context=src, mask=mask)[0,-1,:]
            print(x.shape)
            print(out.shape)
            out[0,idx,:] = x
            idx += 1
        return out
        