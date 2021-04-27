import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from vit_pytorch.efficient import ViT as EfficientViT
from vit_pytorch.t2t import T2TViT
from vit_pytorch.levit import LeViT
from vit_pytorch.cvt import CvT
from vit_pytorch.rvt import RvT
from nystrom_attention import Nystromformer
from routing_transformer import RoutingTransformer
from model.style_model import StyleViT
from model.custom_vit import ViT
from model.custom_local_vit import LocalViT

from x_transformers import ContinuousTransformerWrapper, Decoder
from x_transformers.x_transformers import FeedForward

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

def make_vit(image_size, patch_size, dim, depth, heads, mlp_dim, channels):
    enc = ViT(image_size = image_size, patch_size = patch_size, num_classes = 1, dim = dim, depth = depth, heads = heads, mlp_dim = mlp_dim, channels = channels)
    enc.mlp_head = nn.Identity()
    return enc

def make_cvt():
    enc = CvT(num_classes=1)
    enc.layers[-1] = nn.Identity()
    return enc

def make_nystrom(image_size, patch_size, dim, depth, heads, channels):
    raise NotImplementedError('There are issues with this right now. Come back later :3c')
    enc = EfficientViT(image_size = 576, patch_size = patch_size, num_classes = 1, dim = dim,
                       transformer = Nystromformer(
                           dim = dim,
                           depth = depth,
                           heads = heads,
                           num_landmarks = 128))
    enc.mlp_head = nn.Identity()
    return enc

def make_style(image_size, patch_size, dim, depth, heads, mlp_dim, num_latents, channels):
    enc = StyleViT(image_size = image_size, patch_size = patch_size, dim = dim, depth = depth, heads = heads, mlp_dim = mlp_dim, num_latents = num_latents, channels = channels)
    return enc

def make_local(image_size, patch_size, dim, depth, heads, mlp_dim, channels):
    return LocalViT(image_size, patch_size, dim, depth, heads, mlp_dim, channels)

def make_autoencoder(ae_path):
    enc = torch.load(ae_path)
    return enc

def make_decoder(dim, depth, heads, use_scalenorm, rel_pos_bias, rotary_pos_emb, attn_talking_heads):
    return ContinuousTransformerWrapper(max_seq_len = 256, attn_layers = Decoder(dim = dim, depth = depth, heads = heads, use_scalenorm = use_scalenorm, rel_pos_bias = rel_pos_bias, rotary_emb_dim = rotary_pos_emb, attn_talking_heads = attn_talking_heads), dim_in = dim, dim_out = dim)

def make_routing(dim, depth, heads):
    return RoutingTransformer(dim = dim, depth = depth, max_seq_len = 256, heads = heads, ff_glu = True, use_scale_norm = True, causal = True, receives_context=True)


def make_conv(dim):
    enc = torch.load('best_encoder_{}.pt'.format(dim))
    for param in enc.parameters():
        param.requires_grad = False
    return enc
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels=4, out_channels=8, kernel_size=7, bias=False),
        nn.LeakyReLU(),
        nn.Dropout2d(p=0.2),
        nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, bias=False),
        nn.LeakyReLU(),
        nn.Dropout2d(p=0.2),
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, bias=False),
        nn.LeakyReLU(),
        nn.Dropout2d(p=0.2),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False),
        nn.LeakyReLU(),
        nn.Dropout2d(p=0.2),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False),
        nn.LeakyReLU(),
        nn.Dropout2d(p=0.2),
        nn.AdaptiveAvgPool2d((100,dim))
    )
    '''

def make_mobilenet(dim):
    model = models.mobilenet_v3_small()
    # MobilenetV3-Small
    model.classifier = nn.Sequential(
        nn.Linear(in_features=576, out_features=dim, bias=True),
    )
    return model


possible_encoders = ['vit', 'cvt', 'nystrom', 'conv', 'style', 'mobilenet', 'local']
possible_decoders = ['decoder', 'routing', 'linear']
class EndToEndModel(nn.Module):
    def __init__(self, e_type, d_type, layer_count, image_size = 192, patch_size = 32, channels = 3,
                       dim = 32, emb_dim = 4, e_depth = 1, e_heads = 8, d_depth = 1, d_heads = 8, mlp_dim = 32,
                       num_latents = 2, use_scalenorm = True, rel_pos_bias = False, rotary_pos_emb = True, attn_talking_heads = True, emb_drop = 0.1, thicc_ff=False, pretrain_embeddings=None):
        super().__init__()
        assert e_type in possible_encoders, 'Please select an encoder from {}'.format(possible_encoders)
        assert d_type in possible_decoders, 'Please select a decoder from {}'.format(possible_decoders)
        if e_type == 'vit':
            self.encoder = make_vit(image_size, patch_size, dim, e_depth, e_heads, mlp_dim, channels)
        elif e_type == 'cvt':
            self.encoder = make_cvt()
        elif e_type == 'nystrom':
            self.encoder = make_nystrom(image_size, patch_size, dim, e_depth, e_heads, channels)
        elif e_type == 'style':
            self.encoder = make_style(image_size, patch_size, dim, e_depth, e_heads, mlp_dim, num_latents, channels)
        elif e_type == 'conv':
            self.encoder = make_conv(dim)
        elif e_type == 'mobilenet':
            self.encoder = make_mobilenet(dim)
        elif e_type == 'local':
            self.encoder = make_local(image_size, patch_size, dim, e_depth, e_heads, mlp_dim, channels)
        else:
            raise TypeError('{} not among types {}'.format(e_type, possible_encoders))
        self.routing = False # Because routing transformers have an additional auxilary loss
        if d_type == 'decoder':
            self.decoder = make_decoder(dim = dim, depth = d_depth, heads = d_heads, use_scalenorm = use_scalenorm, rel_pos_bias = rel_pos_bias, rotary_pos_emb = rotary_pos_emb, attn_talking_heads = attn_talking_heads)
        elif d_type == 'routing':
            self.routing = True
            self.decoder = make_routing(dim = dim, depth = d_depth, heads = d_heads)
        else:
            raise TypeError('{} not among types {}'.format(d_type, possible_decoders))
        
        # Decoder only parts
        if pretrain_embeddings is not None:
            print('Loading pretrained embeddings')
            self.embedding_dim = nn.Embedding.from_pretrained(pretrain_embeddings, freeze=True)
            layer_count = self.embedding_dim.num_embeddings
            emb_dim = self.embedding_dim.embedding_dim
        else:
            self.embedding_dim = nn.Embedding(layer_count, emb_dim)
        self.emb_dropout = nn.Dropout(p=emb_drop)
        self.projection = nn.Linear(emb_dim + 12, dim)

        self.norm = nn.LayerNorm(dim)

        self.to_classes = nn.Linear(dim, layer_count, bias=False) if not thicc_ff else nn.Sequential(
            FeedForward(dim=dim, dim_out=dim, glu=True),
            nn.LayerNorm(dim),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=dim, out_features=layer_count)
        )

        self.to_colors = nn.Linear(dim, 4, bias=False) if not thicc_ff else nn.Sequential(
            FeedForward(dim=dim, dim_out=dim, glu=True),
            nn.LayerNorm(dim),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=dim, out_features=4, bias=False)
        )
        self.to_positions = nn.Linear(dim, 8, bias=False) if not thicc_ff else nn.Sequential(
            FeedForward(dim=dim, dim_out=dim, glu=True),
            nn.LayerNorm(dim),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=dim, out_features=8, bias=False)
        )
    
    def freeze_embeddings(self, freeze=True):
        self.embedding_dim.weight.requires_grad = not freeze
    
    def freeze_encoder(self, freeze=True):
        for param in self.encoder[-1].parameters():
            param.requires_grad = freeze
        #for param in self.encoder.parameters():
        #    param.requires_grad = freeze
        
    
    def forward(self, x, src, mask = None, use_activations = False, return_predictions = False):
        context = self.encoder(x)
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
            x, aux_loss = self.decoder(y, context=context, input_mask=feature_mask)
        else:
            x = self.decoder(y, context=context, mask=feature_mask)
        x = F.hardswish(x)
        pred_embs = self.to_classes(x)
        pred_cols = self.to_colors(x).sigmoid() if use_activations else self.to_colors(x)
        pred_posi = self.to_positions(x).tanh() if use_activations else self.to_positions(x)
        if return_predictions:
            return pred_embs, pred_cols, pred_posi, aux_loss
        else:
            return F.cross_entropy(pred_embs.transpose(1,2), label_emb.long().squeeze(-1)), F.mse_loss(pred_cols, label_cols), F.mse_loss(pred_posi, label_posi), aux_loss
    
    @torch.no_grad()
    def generate(self, x, vocab, max_len=225, filter_logits_fn = top_k, p = 0.9, temperature = 1.0, use_activations=False):
        context = self.encoder(x)
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

            out_embs = self.to_classes(x)
            out_colors = self.to_colors(x).sigmoid() if use_activations else self.to_colors(x)
            out_positions = self.to_positions(x).tanh() if use_activations else self.to_positions(x)
            out_embs, out_colors, out_positions = out_embs[:,-1:,:], out_colors[:,-1:,:], out_positions[:,-1:,:]
            out_embs = out_embs.squeeze(0)
            filtered_logits = filter_logits_fn(out_embs, thres = p)
            probs = F.softmax(filtered_logits / temperature, dim = -1)
            sample = torch.multinomial(probs, 1)
            emb_idx = sample.item()
            out.append([emb_idx] + list(map(float, out_colors.squeeze().tolist())) + list(map(float, out_positions.squeeze().tolist())))
            mask.append(True)
            if emb_idx == eos_token:
                break
        out = out[1:]
        return np.asarray(out)