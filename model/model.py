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
from vit_pytorch import ViT
from vit_pytorch.cvt import CvT
from vit_pytorch.mpp import MPP
from x_transformers import Decoder
from x_transformers.x_transformers import FeedForward
from byol_pytorch import BYOL
from routing_transformer import RoutingTransformer

from model.datasets import SADataset
from model.utils import get_parameter_count, Vocabulary

load_dotenv()

class BasicEncoder(nn.Module):
    def __init__(self, patch_size = 32, dim = 16, e_depth = 6, e_heads = 8, mlp_dim = 128, encoder_type = 'vit'):
        super(BasicEncoder, self).__init__()
        if encoder_type == 'cvt':
            print('Using CvT as encoder')
            self.encoder = CvT(
            num_classes = 1,
            )
            self.encoder.layers[-1] = nn.Identity()
            self.encoder.layers[-2] = nn.Identity()
        else:
            print('Using ViT as encoder')
            self.encoder = ViT(
                image_size = 576,
                patch_size = patch_size,
                dim = dim,
                depth = e_depth,
                heads = e_heads,
                mlp_dim = mlp_dim,
                num_classes = 1
            )
            self.encoder.mlp_head = nn.Identity()
        
    def forward(self, x):
        return self.encoder(x)
    

def pretrain_encoder(model, dataloader, device, epochs = 100, max_patience = 15, hidden_layer = -1):
    learner = BYOL(net = model, image_size = 576, hidden_layer=hidden_layer, use_momentum = False).to(device)
    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)
    print('Beginning pretraining of encoder')
    best_encoder = None
    best_loss = None
    patience = 0
    for edx in range(epochs):
        running_loss = 0
        for bdx, i_batch in enumerate(dataloader):
            opt.zero_grad()
            feature = i_batch['feature'].to(device)
            batch_loss = learner(i_batch['feature'].to(device))
            running_loss += batch_loss.item()
            batch_loss.backward()
            opt.step()
        print('PRETRAINING: Epoch: {}, Loss: {}'.format(edx, running_loss))
        if best_loss is None or running_loss < best_loss:
            best_loss = running_loss
            best_encoder = model.state_dict()
            patience = 0
        if patience > max_patience:
            print('Out of patience. Breaking')
            break
        patience += 1
    model.load_state_dict(best_encoder)
    return model


class AutoregressiveDecoder(nn.Module):
    def __init__(self, layer_count = 388, emb_dim = 8, d_dim = 16, d_depth = 12, d_heads = 8, emb_drop = 0.1, decoder_type = ''):
        super(AutoregressiveDecoder, self).__init__()
        self.layer_count = layer_count
        self.emb_dim = emb_dim
        self.d_dim = d_dim
        self.latent_dim = emb_dim + 12
        self.logit_dim = layer_count + 12
        self.max_seq_len = 225
        self.routing = False

        self.embedding_dim = nn.Embedding(layer_count, emb_dim)
        self.emb_dropout = nn.Dropout(p=emb_drop)
        self.projection = FeedForward(self.latent_dim, dim_out=d_dim, glu=True)
        self.latent = FeedForward(d_dim, dim_out=d_dim, glu=True)

        if decoder_type.lower() == 'routing':
            self.decoder = RoutingTransformer(
                dim = d_dim,
                depth = d_depth,
                max_seq_len = 256,
                heads = d_heads,
                ff_glu = True,
                use_scale_norm = True,
            )
            self.routing = True
        else:
            self.decoder = Decoder(
                dim = d_dim,
                depth = d_depth,
                heads = d_heads,
                ff_glu = True,
                rel_pos_bias=True,
                position_infused_attn=True,
                attn_talking_heads = True
            )

        self.to_classes = FeedForward(d_dim, dim_out=self.layer_count, glu=True, dropout=0.1)
        self.to_colors = FeedForward(d_dim, dim_out=4, glu=True, dropout=0.1)
        self.to_positions = FeedForward(d_dim, dim_out=8, glu=True, dropout=0.1)

    
    def forward(self, src, mask=None, context=None, return_both_loss=False, return_predictions=False, loss_func=F.l1_loss, use_activations=False):
        features, labels = src[:,:-1,:], src[:,1:,:]
        feature_mask, label_mask = mask[:,:-1], mask[:,1:]
        label_emb, label_cols, label_posi = torch.split(labels, [1,4,8], dim=-1)

        feature_emb, feature_met = torch.split(features, [1, features.shape[-1] - 1], dim=-1)
        embs = self.embedding_dim(feature_emb.int()).squeeze(dim=2)
        embs = self.emb_dropout(embs)
        y = torch.cat([embs, feature_met], dim=-1)
        y = self.projection(y)
        aux_loss = None
        if self.routing:
            x, aux_loss = self.decoder(y, context=context, mask=feature_mask)
        else:
            x = self.decoder(y, context=context, mask=feature_mask)
        
        x = self.latent(x)

        pred_embs = self.to_classes(x)
        pred_cols = self.to_colors(x).sigmoid() if use_activations else self.to_colors(x)
        pred_posi = self.to_positions(x).tanh() if use_activations else self.to_positions(x)

        if self.routing:
            if return_predictions:
                return pred_embs, pred_cols, pred_posi, aux_loss
            if return_both_loss:
                return F.cross_entropy(pred_embs.transpose(1,2), label_emb.long().squeeze(-1)), loss_func(pred_cols, label_cols), loss_func(pred_posi, label_posi), aux_loss
            return F.cross_entropy(pred_embs.transpose(1,2), label_emb.long().squeeze(-1)) + loss_func(pred_cols, label_cols) + loss_func(pred_posi, label_posi) + aux_loss
        else:
            if return_predictions:
                return pred_embs, pred_cols, pred_posi, None
            if return_both_loss:
                return F.cross_entropy(pred_embs.transpose(1,2), label_emb.long().squeeze(-1)), loss_func(pred_cols, label_cols), loss_func(pred_posi, label_posi), None
            return F.cross_entropy(pred_embs.transpose(1,2), label_emb.long().squeeze(-1)) + loss_func(pred_cols, label_cols) + loss_func(pred_posi, label_posi)

    @torch.no_grad()
    def generate(self, context, vocab, max_len, k=1, use_activations=False):
        device = context.device
        out = [[vocab['<SOS>']] + [0] * 12]
        eos_token = vocab['<EOS>']
        mask = [True]
        
        while len(out) <= max_len:
            x = torch.unsqueeze(torch.from_numpy(np.asarray(out, dtype=np.float32)), dim=0).to(device)
            out_mask = torch.unsqueeze(torch.from_numpy(np.asarray(mask, dtype=np.bool)), dim=0).to(device)
            feature_emb, feature_met = torch.split(x, [1, x.shape[-1] - 1], dim=-1)
            embs = self.embedding_dim(feature_emb.int()).squeeze(dim=2)
            embs = self.emb_dropout(embs)
            y = torch.cat([embs, feature_met], dim=-1)
            y = self.projection(y)
            if self.routing:
                x, _ = self.decoder(y, context=context, mask=out_mask)
            else:
                x = self.decoder(y, context=context, mask=out_mask)
            
            x = self.latent(x)

            out_embs = self.to_classes(x)
            out_colors = self.to_colors(x).sigmoid() if use_activations else self.to_colors(x)
            out_positions = self.to_positions(x).tanh() if use_activations else self.to_positions(x)
            out_embs, out_colors, out_positions = out_embs[:,-1:,:], out_colors[:,-1:,:], out_positions[:,-1:,:]
            emb_idx = torch.topk(out_embs, 1)[1].item() if k == 1 else random.choice(torch.topk(out_embs, 1)[1]).item()
            out.append([emb_idx] + list(map(float, out_colors.squeeze().tolist())) + list(map(float, out_positions.squeeze().tolist())))
            mask.append(True)
            if emb_idx == eos_token:
                break
        out = out[1:]
        return np.asarray(out)
