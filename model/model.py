import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from einops.layers.torch import Rearrange

from vit_pytorch.dino import Dino
from vit_pytorch.t2t import T2TViT
from vit_pytorch.levit import LeViT
from model.custom_cvt import CvT
from vit_pytorch.rvt import RvT
from nystrom_attention import Nystromformer
from routing_transformer import RoutingTransformer
from routing_transformer import Autopadder as RoutingAutopadder
from linear_attention_transformer import LinearAttentionTransformer
from linear_attention_transformer.autopadder import Autopadder as LinearAutopadder
from reformer_pytorch import Reformer
from reformer_pytorch import Autopadder as ReformerAutopadder
from model.style_model import StyleViT
from model.custom_vit import ViT
from model.custom_perceiver import Perceiver
from model.custom_gmlp import gMLPVision
#from glom_pytorch import Glom
from model.custom_glom import Glom
from model.fc import FCModel, SimpleConv
from model.torchformer import TorchEncoder, TorchDecoder
from model.mobilenetv3 import MyMobileNetV3

from x_transformers import ContinuousTransformerWrapper, Decoder, Encoder, ViTransformerWrapper
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
                            attend_axially = True
                        ))
    return enc

def make_style(image_size, patch_size, dim, depth, heads, mlp_dim, num_latents, channels):
    enc = StyleViT(image_size = image_size, patch_size = patch_size, dim = dim, depth = depth, heads = heads, mlp_dim = mlp_dim, num_latents = num_latents, channels = channels)
    return enc

def make_autoencoder(ae_path):
    enc = torch.load(ae_path)
    return enc

def make_glom(image_size, patch_size, dim, levels, iters):
    return Glom(dim = dim, levels = levels, image_size = image_size, patch_size = patch_size, consensus_self=True, iters=iters)

def make_perceiver(input_channels, dim, depth):
    return Perceiver(input_channels=input_channels, num_freq_bands=6, depth=depth, max_freq=10., latent_dim=dim, num_latents=128)

def make_gmlp(image_size, patch_size, dim, depth, channels):
    return gMLPVision(image_size=image_size, patch_size=patch_size, dim=dim, depth=depth, channels=channels)

def make_decoder(dim, depth, heads, use_scalenorm, rel_pos_bias, rotary_pos_emb):
    return ContinuousTransformerWrapper(max_seq_len = 256, attn_layers = Decoder(dim = dim, depth = depth, heads = heads, use_scalenorm = use_scalenorm, rel_pos_bias = rel_pos_bias, rotary_emb_dim = rotary_pos_emb), dim_in = dim, dim_out = dim)

def make_routing(dim, depth, heads):
    return RoutingAutopadder(RoutingTransformer(dim = dim, depth = depth, max_seq_len = 256, heads = heads, ff_glu = True, use_scale_norm = True, causal = True, receives_context=True, ff_dropout=0.2, attn_dropout=0.2, attn_layer_dropout=0.2))

def make_reformer(dim, depth, heads):
    return ReformerAutopadder(Reformer(dim = dim, depth = depth, heads = heads, causal = True, ff_glu = True, use_scale_norm = True))

def make_linear(dim, depth, heads):
    local_heads = int(heads/2)
    global_heads = heads - local_heads
    return LinearAutopadder(LinearAttentionTransformer(dim = dim, depth = depth, max_seq_len=256, heads=global_heads, ff_glu=True, causal=True, receives_context=True, n_local_attn_heads=local_heads, local_attn_window_size=32))

def make_conv(dim, patch_size, channels):
    return torch.load('vae.pt')
    #return SimpleConv(dim, channels = channels)

def make_torch_enc(image_size, patch_size, dim, depth, heads, channels):
    return TorchEncoder(image_size, patch_size, dim, depth, heads, channels)

def make_torch_dec(dim, depth, heads):
    return TorchDecoder(dim, depth, heads)

def make_mobilenet(dim):
    return MyMobileNetV3(dim)


possible_encoders = ['vit', 'cvt', 'efficient', 'conv', 'style', 'mobilenet', 'glom', 'torch', 'perceiver', 'encoder', 'gmlp']
possible_decoders = ['decoder', 'routing', 'linear', 'reformer', 'torch']
class EndToEndModel(nn.Module):
    def __init__(self, e_type, d_type, layer_count, image_size = 256, patch_size = 32, channels = 3,
                       dim = 32, emb_dim = 4, e_depth = 1, e_heads = 8, d_depth = 1, d_heads = 8, mlp_dim = 32,
                       num_latents = 2, use_scalenorm = True, rel_pos_bias = False, rotary_pos_emb = True, pretrain_embeddings=None,
                       use_activations = False, **kwarg):
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
            # In case I want to make an argument for this at some point
            levels = 6
            prop_mult = max(2, 2) 

            self.encoder = nn.Sequential(
                make_glom(image_size, patch_size, dim, levels, levels * prop_mult),
                Rearrange('b p l d -> b (p l) d'),
            )
            #self.encoder = make_glom(image_size, patch_size, dim, 3)
        elif e_type == 'torch':
            self.encoder = make_torch_enc(image_size, patch_size, dim, e_depth, e_heads, channels)
        elif e_type == 'perceiver':
            self.encoder = make_perceiver(channels, dim, e_depth)
        elif e_type == 'gmlp':
            self.encoder = make_gmlp(image_size, patch_size, dim, e_depth, channels)
        else:
            raise TypeError('{} not among types {}'.format(e_type, possible_encoders))
        self.routing = False # Because routing transformers have an additional auxilary loss
        if d_type == 'decoder':
            self.decoder = make_decoder(dim = dim, depth = d_depth, heads = d_heads, use_scalenorm = use_scalenorm, rel_pos_bias = rel_pos_bias, rotary_pos_emb = rotary_pos_emb)
        elif d_type == 'routing':
            self.dec_route = True
            self.decoder = make_routing(dim = dim, depth = d_depth, heads = d_heads)
        elif d_type == 'reformer':
            self.decoder = make_reformer(dim = dim, depth = d_depth, heads = d_heads)
        elif d_type == 'torch':
            self.decoder = make_torch_dec(dim, d_depth, d_heads)
        elif d_type == 'linear':
            self.decoder = make_linear(dim, d_depth, d_heads)
        else:
            raise TypeError('{} not among types {}'.format(d_type, possible_decoders))

        
        self.latent_pre_dim = emb_dim + 4 + 8
        self.latent_post_dim = layer_count + 4 + 8

        self.post_in_norm = nn.LayerNorm(normalized_shape=dim)
        self.pre_out_norm = nn.LayerNorm(normalized_shape=dim)
        
        
        self.project_in = nn.Linear(in_features=self.latent_pre_dim, out_features=dim, bias=False)
        self.to_classes = nn.Linear(in_features=dim, out_features=layer_count, bias=False)
        self.to_colors = nn.Linear(in_features=dim, out_features=4, bias=False)
        self.to_positions = nn.Linear(in_features=dim, out_features=8, bias=False)

        self.color_activation = nn.GELU() if use_activations else nn.Identity()
        self.position_activation = nn.GELU() if use_activations else nn.Identity()
        
        self.class_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.color_loss = nn.MSELoss()
        self.position_loss = nn.MSELoss()
    
    def freeze_embeddings(self, freeze=True):
        self.embedding_dim.weight.requires_grad = not freeze
    
    def freeze_encoder(self, freeze=True):
        for param in self.encoder[-1].parameters():
            param.requires_grad = freeze
        
    
    def forward(self, x, src, mask = None, use_activations = False, return_predictions = False):
        if isinstance(self.encoder, Perceiver):
            context = self.encoder(x.permute(0,2,3,1))
        else:
            context = self.encoder(x)
        features, labels = src[:,:-1,:], src[:,1:,:]
        feature_mask, _ = mask[:,:-1], mask[:,1:]
        label_emb, label_cols, label_posi = torch.split(labels, [1,4,8], dim=-1)
        feature_emb, feature_met = torch.split(features, [1, features.shape[-1] - 1], dim=-1)
        embs = self.embedding_dim(feature_emb.int()).squeeze(dim=2)
        embs = self.emb_dropout(embs)
        y = torch.cat([embs, feature_met], dim=-1)

        y = self.project_in(y)
        y = self.post_in_norm(y)
        dec_aux_loss = None
        if self.dec_route:
            x, dec_aux_loss = self.decoder(y, context=context, input_mask=feature_mask)
        else:
            if isinstance(self.decoder, ContinuousTransformerWrapper):
                x = self.decoder(y, context=context, mask=feature_mask)
            else:
                x = self.decoder(y, context=context, input_mask=feature_mask)
        
        x = self.pre_out_norm(x)
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
            x = self.pre_out_norm(x)
            layer_out, color_out, position_out = self.to_classes(x)[:,-1,:], self.to_colors(x)[:,-1,:], self.to_positions(x)[:,-1,:]
            color_out, position_out = self.color_activation(color_out), self.position_activation(position_out)
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

def pretrain_dino(model, train_dataset, valid_dataset, img_size, epochs=1000000, patience=50, hidden_layer='to_latent', device='cpu'):
    if not isinstance(model, (StyleViT, ViT, gMLPVision)):
        print('No support for pretraining encoder with {}'.format(type(model).__name__))
        return model

    import torch.optim as optim
    from torch.utils.data import DataLoader

    learner = Dino(model, image_size=img_size, hidden_layer=hidden_layer).to(device)
    opt = optim.AdamW(learner.parameters(), lr=3e-4)

    dataloader = DataLoader(train_dataset, batch_size=128, shuffle = True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32)

    best_loss = None
    best_model = None

    cur_pat = 0

    print('BEGINNING ENCODER PRETRAINING')
    for edx in range(epochs):
        running_loss = 0.0
        for bdx, i_batch in enumerate(dataloader):
            features = i_batch['feature'].to(device)
            loss = learner(features)
            running_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average()
            print('\tBatch #{}, Loss: {}'.format(bdx, loss.item()))

        running_loss = 0.0
        learner.eval()
        model.eval()
        with torch.no_grad():
            for bdx, i_batch in enumerate(valid_dataloader):
                features = i_batch['feature'].to(device)
                running_loss += learner(features).item()
        learner.train()
        model.train()
        
        if best_loss is None or running_loss < best_loss:
            best_loss = running_loss
            best_model = model.state_dict()
            cur_pat = 0
        else:
            cur_pat += 1
        print('Epoch #{}, Loss: {}, Patience: {}/{}'.format(edx, running_loss, cur_pat, patience))
        if cur_pat > patience:
            print('Out of patience')
            break
    print('Best model loss: {}\nFreezing encoder'.format(best_loss))
    model.load_state_dict(best_model)
    for p in model.parameters():
        p.requires_grad = False
    return model