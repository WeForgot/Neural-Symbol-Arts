import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from einops.layers.torch import Rearrange

from vit_pytorch.t2t import T2TViT
from vit_pytorch.levit import LeViT
from model.custom_cvt import CvT
from vit_pytorch.rvt import RvT
from nystrom_attention import Nystromformer
from routing_transformer import RoutingTransformer
from routing_transformer import Autopadder as RoutingAutopadder
from linear_attention_transformer import LinearAttentionTransformer
from reformer_pytorch import Reformer
from reformer_pytorch import Autopadder as ReformerAutopadder
from model.style_model import StyleViT
from model.custom_vit import ViT
#from glom_pytorch import Glom
from model.custom_glom import Glom
from model.fc import FCModel, SimpleConv
from model.torchformer import TorchEncoder, TorchDecoder

from x_transformers import ContinuousTransformerWrapper, Decoder
from x_transformers.x_transformers import FeedForward
from axial_positional_embedding import AxialPositionalEmbedding

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
    enc = ViT(image_size = image_size, patch_size = patch_size, dim = dim, depth = depth, heads = heads, mlp_dim = mlp_dim, channels = channels)
    return enc

def make_cvt(dim):
    enc = CvT(out_dim=dim, s1_emb_dim=16, s2_emb_dim=32, s3_emb_dim=64)
    return enc

def make_efficient(image_size, patch_size, dim, depth, heads, mlp_dim, channels):
    enc = ViT(image_size = image_size, patch_size = patch_size, dim = dim, depth = depth, heads = heads, mlp_dim = mlp_dim, channels = channels,
                        transformer = LinearAttentionTransformer(
                            dim = dim,
                            depth = depth,
                            max_seq_len = 576,
                            heads = heads,
                            ff_glu = True,
                            
                        ))
    return enc

def make_style(image_size, patch_size, dim, depth, heads, mlp_dim, num_latents, channels):
    enc = StyleViT(image_size = image_size, patch_size = patch_size, dim = dim, depth = depth, heads = heads, mlp_dim = mlp_dim, num_latents = num_latents, channels = channels)
    return enc

def make_autoencoder(ae_path):
    enc = torch.load(ae_path)
    return enc

def make_glom(image_size, patch_size, dim, levels):
    return Glom(dim = dim, levels = levels, image_size = image_size, patch_size = patch_size, consensus_self=True)

def make_decoder(dim, depth, heads, use_scalenorm, rel_pos_bias, rotary_pos_emb, attn_talking_heads):
    return ContinuousTransformerWrapper(max_seq_len = 256, attn_layers = Decoder(dim = dim, depth = depth, heads = heads, use_scalenorm = use_scalenorm, rel_pos_bias = rel_pos_bias, rotary_emb_dim = rotary_pos_emb, attn_talking_heads = attn_talking_heads), dim_in = dim, dim_out = dim)

def make_routing(dim, depth, heads):
    return RoutingAutopadder(RoutingTransformer(dim = dim, depth = depth, max_seq_len = 256, heads = heads, ff_glu = True, use_scale_norm = True, causal = True, receives_context=True))

def make_reformer(dim, depth, heads):
    return ReformerAutoPadder(Reformer(dim = dim, depth = depth, heads = heads, causal = True, ff_glu = True, use_scale_norm = True))

def make_conv(dim, patch_size, channels):
    return SimpleConv(dim, blocks = 3, channels = channels)
    #return FCModel(dim, patch_size, channels=channels)

def make_torch_enc(image_size, patch_size, dim, depth, heads, channels):
    return TorchEncoder(image_size, patch_size, dim, depth, heads, channels)

def make_torch_dec(dim, depth, heads):
    return TorchDecoder(dim, depth, heads)

def make_mobilenet(dim):
    model = models.mobilenet_v3_small()
    # MobilenetV3-Small
    model.classifier = nn.Sequential(
        nn.Linear(in_features=576, out_features=dim, bias=True),
    )
    return model


possible_encoders = ['vit', 'cvt', 'efficient', 'conv', 'style', 'mobilenet', 'glom', 'torch']
possible_decoders = ['decoder', 'routing', 'linear', 'reformer', 'torch']
class EndToEndModel(nn.Module):
    def __init__(self, e_type, d_type, layer_count, image_size = 256, patch_size = 32, channels = 3,
                       dim = 32, emb_dim = 4, e_depth = 1, e_heads = 8, d_depth = 1, d_heads = 8, mlp_dim = 32,
                       num_latents = 2, use_scalenorm = True, rel_pos_bias = False, rotary_pos_emb = True, attn_talking_heads = True, emb_drop = 0.1, thicc_ff=False, pretrain_embeddings=None,
                       use_activations = False):
        super().__init__()
        assert e_type in possible_encoders, 'Please select an encoder from {}'.format(possible_encoders)
        assert d_type in possible_decoders, 'Please select a decoder from {}'.format(possible_decoders)
        # Decoder only parts
        if pretrain_embeddings is not None:
            print('Loading pretrained embeddings')
            self.embedding_dim = nn.Embedding.from_pretrained(pretrain_embeddings, freeze=True)
            layer_count = self.embedding_dim.num_embeddings
            emb_dim = self.embedding_dim.embedding_dim
        else:
            self.embedding_dim = nn.Embedding(layer_count, emb_dim)
        
        self.emb_dropout = nn.Dropout(p=0.1)
        self.enc_route = False
        self.dec_route = False
        self.glom = False # Because GLOM is a fucking shitfest
        if e_type == 'vit':
            self.encoder = make_vit(image_size, patch_size, dim, e_depth, e_heads, mlp_dim, channels)
        elif e_type == 'cvt':
            self.encoder = make_cvt(dim)
        elif e_type == 'efficient':
            self.enc_route = True
            self.encoder = make_efficient(image_size, patch_size, dim, e_depth, e_heads, mlp_dim, channels)
        elif e_type == 'style':
            self.encoder = make_style(image_size, patch_size, dim, e_depth, e_heads, mlp_dim, num_latents, channels)
        elif e_type == 'conv':
            self.encoder = make_conv(dim, patch_size, channels)
        elif e_type == 'mobilenet':
            self.encoder = make_mobilenet(dim)
        elif e_type == 'glom':
            self.glom = True
            self.encoder = nn.Sequential(
                make_glom(image_size, patch_size, dim, 6),
                Rearrange('b p l d -> b l p d'),
                nn.Conv2d(in_channels = 6, out_channels = 3, kernel_size=3, stride=1, bias=False, padding='same'),
                nn.Conv2d(in_channels = 3, out_channels = 1, kernel_size=3, stride=1, bias=False, padding='same'),
            )
            #self.encoder = make_glom(image_size, patch_size, dim, 6)
        elif e_type == 'torch':
            self.encoder = make_torch_enc(image_size, patch_size, dim, e_depth, e_heads, channels)
        else:
            raise TypeError('{} not among types {}'.format(e_type, possible_encoders))
        self.routing = False # Because routing transformers have an additional auxilary loss
        if d_type == 'decoder':
            self.decoder = make_decoder(dim = dim, depth = d_depth, heads = d_heads, use_scalenorm = use_scalenorm, rel_pos_bias = rel_pos_bias, rotary_pos_emb = rotary_pos_emb, attn_talking_heads = attn_talking_heads)
        elif d_type == 'routing':
            self.dec_route = True
            self.decoder = make_routing(dim = dim, depth = d_depth, heads = d_heads)
        elif d_type == 'reformer':
            self.decoder = make_reformer(dim = dim, depth = d_depth, heads = d_heads)
        elif d_type == 'torch':
            self.decoder = make_torch_dec(dim, d_depth, d_heads)
        else:
            raise TypeError('{} not among types {}'.format(d_type, possible_decoders))

        
        self.latent_pre_dim = emb_dim + 4 + 8
        self.latent_post_dim = layer_count + 4 + 8
        
        self.project_in = nn.Linear(in_features=self.latent_pre_dim, out_features=dim)
        self.pre_out_norm = nn.InstanceNorm1d(num_features=dim)
        #self.project_out = nn.Linear(in_features=dim, out_features=self.latent_post_dim)
        self.to_classes = nn.Linear(in_features=dim, out_features=layer_count, bias=False)
        self.to_colors = nn.Linear(in_features=dim, out_features=4, bias=False)
        self.to_positions = nn.Linear(in_features=dim, out_features=8, bias=False)

        #self.to_classes = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, layer_count, bias=False)) if not thicc_ff else nn.Sequential(
        #    FeedForward(dim=emb_dim, dim_out=emb_dim, glu=True),
        #    nn.LayerNorm(emb_dim),
        #    nn.Dropout(p=0.1),
        #    nn.Linear(in_features=emb_dim, out_features=layer_count)
        #)

        self.color_activation = nn.Sigmoid() if use_activations else nn.ReLU()
        self.position_activation = nn.Tanh() if use_activations else nn.Identity()
        
        class_weights = torch.ones((layer_count,), dtype=torch.float32)
        class_weights[0] = 0
        self.class_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.color_loss = nn.MSELoss()
        self.position_loss = nn.MSELoss()
    
    def freeze_embeddings(self, freeze=True):
        self.embedding_dim.weight.requires_grad = not freeze
    
    def freeze_encoder(self, freeze=True):
        for param in self.encoder[-1].parameters():
            param.requires_grad = freeze
        
    
    def forward(self, x, src, mask = None, use_activations = False, return_predictions = False):
        enc_aux_loss = None
        if self.glom:
            context = self.encoder(x).squeeze(1)
        else:
            context = self.encoder(x)
        features, labels = src[:,:-1,:], src[:,1:,:]
        feature_mask, label_mask = mask[:,:-1], mask[:,1:]
        label_emb, label_cols, label_posi = torch.split(labels, [1,4,8], dim=-1)
        feature_emb, feature_met = torch.split(features, [1, features.shape[-1] - 1], dim=-1)
        embs = self.embedding_dim(feature_emb.int()).squeeze(dim=2)
        embs = self.emb_dropout(embs)
        y = torch.cat([embs, feature_met], dim=-1)
        y = self.project_in(y)
        dec_aux_loss = None
        if self.dec_route:
            x, aux_loss = self.decoder(y, context=context, input_mask=feature_mask)
        else:
            if isinstance(self.decoder, ContinuousTransformerWrapper):
                x = self.decoder(y, context=context, mask=feature_mask)
            else:
                x = self.decoder(y, context=context, input_mask=feature_mask)
        pred_embs, pred_cols, pred_posi = self.to_classes(x), self.to_colors(x), self.to_positions(x)
        pred_cols, pred_posi = self.color_activation(pred_cols), self.position_activation(pred_posi)
        if return_predictions:
            return pred_embs, pred_cols, pred_posi
        else:
            return self.class_loss(pred_embs.transpose(1,2), label_emb.long().squeeze(-1)), self.color_loss(pred_cols, label_cols), self.position_loss(pred_posi, label_posi), dec_aux_loss
    
    @torch.no_grad()
    def generate(self, x, vocab, max_len=225, filter_logits_fn = top_k, p = 0.9, temperature = 1.0):
        if self.glom:
            context = self.encoder(x).squeeze(1)
        else:
            context = self.encoder(x)
        device = context.device
        out = [[vocab['<SOS>']] + [0] * 12]
        eos = vocab['<EOS>']
        input_mask = [True]
        while len(out) < max_len+1:
            x = torch.tensor(out).unsqueeze(0).to(device)
            mask = torch.tensor(input_mask, dtype=torch.bool).unsqueeze(0).to(device)
            emb, other = torch.split(x, [1,12], dim=-1)
            emb = self.embedding_dim(emb.int()).squeeze(2)
            x = torch.cat([emb, other.to(device)], dim=-1)
            x = self.project_in(x)
            if self.dec_route:
                x, _ = self.decoder(x, context=context, mask=mask)
            else:
                x = self.decoder(x, context=context, mask=mask)
            layer_out, color_out, position_out = self.to_classes(x)[:,-1,:], self.to_colors(x)[:,-1,:], self.to_positions(x)[:,-1,:]
            filtered_logits = filter_logits_fn(layer_out, thres=p)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)
            emb_idx = sample.item()
            if emb_idx == eos:
                break
            colors = list(map(float, color_out.squeeze().tolist()))
            positions = list(map(float, position_out.squeeze().tolist()))
            out.append([emb_idx] + colors + positions)
            input_mask.append(True)
        return np.asarray(out)
            


            
            