import json
import os
import pickle
import random

from dotenv import load_dotenv
import numpy as np
from skimage import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from vit_pytorch.vit import Transformer
from x_transformers import Decoder
from nystrom_attention import Nystromformer
from byol_pytorch import BYOL

from model.datasets import SADataset
from model.utils import get_parameter_count, Vocabulary

load_dotenv()

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)
    
    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim = -1)
        return x * F.gelu(gate)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out = None, mult = 4, glu = False, dropout = 0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.LayerNorm(),
            nn.Dropout(p=dropout),
            nn.Linear(inner_dim, dim_out)
        )
    
    def forward(self, x):
        return self.net(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, transformer, pool = 'cls', channels = 3):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = transformer

        self.pool = pool
        self.to_latent = nn.Identity()

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.transformer(x)

        return self.to_latent(x)

class BasicEncoder(nn.Module):
    def __init__(self, patch_size = 32, dim = 16, e_depth = 6, e_heads = 8):
        super(BasicEncoder, self).__init__()
        self.encoder = ViT(
            image_size = 576,
            patch_size = patch_size,
            dim = dim,
            transformer = Nystromformer(
                dim = dim,
                depth = e_depth,
                heads = e_heads,
            )
        )
    
    def forward(self, x):
        return self.encoder(x)
    

def pretrain_encoder(model, dataloader, device, epochs = 100, max_patience = 10, hidden_layer = 'encoder.to_latent'):
    learner = BYOL(net = model, image_size = 576, hidden_layer=hidden_layer, use_momentum = False).to(device)
    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)
    print('Beginning pretraining of encoder')
    best_encoder = None
    best_loss = None
    patience = 0
    for edx in range(epochs):
        batch_loss = 0
        for bdx, i_batch in enumerate(dataloader):
            feature = i_batch['feature'].to(device)
            batch_loss += learner(i_batch['feature'].to(device))
        scalar_loss = batch_loss.item()
        print('PRETRAINING: Epoch: {}, Loss: {}'.format(edx, scalar_loss))
        opt.zero_grad()
        batch_loss.backward()
        opt.step()
        if best_loss is None or scalar_loss < best_loss:
            best_loss = scalar_loss
            best_encoder = model.state_dict()
            patience = 0
        if patience > max_patience:
            print('Out of patience. Breaking')
            break
        patience += 1
    model.load_state_dict(best_encoder)
    return model


class AutoregressiveDecoder(nn.Module):
    def __init__(self, layer_count = 388, emb_dim = 8, dim = 16, d_depth = 12, d_heads = 8, emb_drop = 0.1):
        super(AutoregressiveDecoder, self).__init__()
        self.layer_count = layer_count
        self.emb_dim = emb_dim
        self.latent_dim = emb_dim + 12
        self.logit_dim = layer_count + 12
        self.max_seq_len = 225

        self.embedding_dim = nn.Embedding(layer_count, emb_dim)
        self.absolute_positional_embeddings = nn.Parameter(torch.rand((self.max_seq_len, self.latent_dim)))
        self.emb_dropout = nn.Dropout(p=emb_drop)
        self.decoder = Decoder(
            dim = self.latent_dim,
            depth = d_depth,
            heads = d_heads
        )

        self.to_classes = FeedForward(self.latent_dim, dim_out=self.layer_count, glu=True, dropout=0.1)
        self.to_colors = FeedForward(self.latent_dim, dim_out=4)
        self.to_positions = FeedForward(self.latent_dim, dim_out=8)

    
    def forward(self, src, mask=None, context=None, return_both_loss=False, return_predictions=False):
        features, labels = src[:,:-1,:], src[:,1:,:]
        feature_mask, label_mask = mask[:,:-1], mask[:,1:]
        label_emb, label_cols, label_posi = torch.split(labels, [1,4,8], dim=-1)

        feature_emb, feature_met = torch.split(features, [1, features.shape[-1] - 1], dim=-1)
        embs = self.embedding_dim(feature_emb.int()).squeeze(dim=2)
        embs = self.emb_dropout(embs)
        y = torch.cat([embs, feature_met], dim=-1)
        y += self.absolute_positional_embeddings[:y.shape[1],:]
        x = self.decoder(y, context=context, mask=feature_mask)

        pred_embs = self.to_classes(x)
        pred_cols = self.to_colors(x).sigmoid()
        pred_posi = self.to_positions(x).tanh()

        if return_predictions:
            return pred_embs, pred_cols, pred_posi
        if return_both_loss:
            return F.cross_entropy(pred_embs.transpose(1,2), label_emb.long().squeeze(-1)), F.l1_loss(pred_cols, label_cols), F.l1_loss(pred_posi, label_posi)
        return F.cross_entropy(pred_embs.transpose(1,2), label_emb.long().squeeze(-1)) + F.l1_loss(pred_cols, label_cols) + F.l1_loss(pred_posi, label_posi)
    
    @torch.no_grad()
    def generate(self, context, vocab, max_len):
        device = context.device
        out = [[vocab['<SOS>']] + [0] * 12]
        eos_token = vocab['<EOS>']
        mask = [True]
        
        while len(out) <= max_len:
            x = torch.unsqueeze(torch.from_numpy(np.asarray(out, dtype=np.float32)), dim=0).to(device)
            out_mask = torch.unsqueeze(torch.from_numpy(np.asarray(mask, dtype=np.bool)), dim=0).to(device)

            out_embs, out_locs = torch.split(x, [1, x.shape[-1]-1], dim=-1)
            out_embs = self.embedding_dim(out_embs.int()).squeeze(dim=2)

            x = torch.cat([out_embs, out_locs], dim=-1)
            x += self.absolute_positional_embeddings[:x.shape[1],:]
            x = self.decoder(x, context=context, mask=out_mask)
            out_embs, out_colors, out_positions = self.to_classes(x), self.to_colors(x), self.to_positions(x)
            out_embs, out_colors, out_positions = out_embs[:,-1:,:], out_colors[:,-1:,:].sigmoid(), out_positions[:,-1:,:].sigmoid()
            emb_idx = torch.topk(out_embs, 1)[1].item()
            out.append([emb_idx] + list(map(round, out_colors.squeeze().tolist())) + list(map(round, out_positions.squeeze().tolist())))
            mask.append(True)
            if emb_idx == eos_token:
                break
        if out[-1][0] != eos_token:
            out.append([eos_token] + [0] * 12)
        return np.asarray(out)
