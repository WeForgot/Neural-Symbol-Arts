# Imports
import json
import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from torch.utils.data import DataLoader
from adabelief_pytorch import AdaBelief

from tqdm import tqdm

#from byol_pytorch import BYOL
from model.mobilenetv3 import mobilenet_v3_small
from model.custom_gmlp import gMLPVision
from vit_pytorch import Dino
from model.custom_vit import ViT
from x_transformers import ContinuousTransformerWrapper, Decoder, ViTransformerWrapper, Encoder



from model.datasets import SADataset
from model.utils import Vocabulary, convert_numpy_to_saml, get_parameter_count, load_data, load_image, get_cosine_with_hard_restarts_schedule_with_warmup

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    logits = logits.squeeze()
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(normalized_shape=dim)
        self.fn = fn
    
    def forward(self, x):
        return self.fn(self.norm(x))

class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn
    
    def forward(self, x):
        return self.fn(x) + x

class Permute(nn.Module):
    def __init__(self, *args):
        super(Permute, self).__init__()
        self.permutation = [0, *args]
    
    def forward(self, x):
        return x.permute(*self.permutation)

# Model classes
class XEncoder(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads):
        super(XEncoder, self).__init__()
        self.encoder = gMLPVision(
            image_size = image_size,
            patch_size = patch_size,
            dim = dim,
            depth = depth,
            prob_survival = 0.9
        )
        self.to_latent = nn.Identity()
    
    def forward(self, x):
        x = self.encoder(x)
        return self.to_latent(x)

class XDecoder(nn.Module):
    def __init__(self, num_layers, max_seq_len, dim, depth, heads):
        super(XDecoder, self).__init__()
        self.max_seq_len = max_seq_len
        self.dim = dim

        self.pos_embs = nn.parameter.Parameter(torch.randn((max_seq_len, dim)), requires_grad=True)
        self.decoder = ContinuousTransformerWrapper(
            max_seq_len = 225,
            attn_layers = Decoder(
                dim = dim,
                depth = depth,
                heads = heads,
                cross_attend = True,
                cross_only = True
            ),
            dim_out = dim
        )
        self.post_norm = nn.LayerNorm(dim)
        self.post_drop = nn.Dropout(p=0.2)

        ffc = nn.ModuleList(
            [nn.Sequential(
                PreNorm(dim, Residual(nn.Linear(dim, dim))),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
            for _ in range(3)]
        )

        self.to_classes = nn.Sequential(
            *ffc,
            nn.Linear(dim, num_layers),
        )

        ffl = nn.ModuleList(
            [nn.Sequential(
                PreNorm(dim, Residual(nn.Linear(dim, dim))),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
            for _ in range(3)]
        )
        self.to_colors = nn.Sequential(
            *ffl,
            nn.Linear(dim, 4),
            nn.Sigmoid(),
        )

        ffp = nn.ModuleList(
                [nn.Sequential(
                    PreNorm(dim, Residual(nn.Linear(dim, dim))),
                    nn.LayerNorm(dim),
                    nn.GELU(),
                    nn.Dropout(0.1)
                )
            for _ in range(3)]
        )
        self.to_positions = nn.Sequential(
            *ffp,
            nn.Linear(dim, 8),
            nn.Sigmoid(),
        )
    
    def forward(self, context):
        b = context.shape[0]
        x = self.pos_embs.repeat(b, 1, 1)
        out = self.decoder(x,context=context, return_embeddings=True)
        #out = self.decoder(x, context = context)
        emb_guess, col_guess, pos_guess = self.to_classes(out), self.to_colors(out), self.to_positions(out)
        return emb_guess, col_guess, pos_guess

class NeuralTransformer(nn.Module):
    def __init__(self, image_size, patch_size, dim, e_depth, e_heads, vocab, d_depth, d_heads, max_seq_len=225):
        super(NeuralTransformer, self).__init__()
        self.encoder = XEncoder(image_size, patch_size, dim, e_depth, e_heads)
        self.decoder = XDecoder(len(vocab), max_seq_len, dim, d_depth, d_heads)
    
    def forward(self, src, labels, return_loss=False, emb_alpha=1.0, col_alpha=1.0, pos_alpha=1.0):
        enc = self.encoder(src)
        emb_guess, col_guess, pos_guess = self.decoder(enc)
        if return_loss:
            emb_target, col_target, pos_target = torch.split(labels, [1, 4, 8], dim=-1)
            return  F.cross_entropy(emb_guess.transpose(1,2), emb_target.squeeze(-1).long(), ignore_index=0) * emb_alpha + F.l1_loss(col_guess, col_target) * col_alpha + F.l1_loss(pos_guess, pos_target) * pos_alpha
        return torch.cat([emb_guess, col_guess, pos_guess], dim=-1)
    
    @torch.no_grad()
    def generate(self, src: torch.Tensor, temperature = 1.0, device=None) -> np.ndarray:
        if len(src.shape) == 3:
            src = src.unsqueeze(0).to(device)
        context = self.encoder(src)
        emb_out, col_out, pos_out = self.decoder(context)
        assert emb_out.shape[1] == col_out.shape[1] and emb_out.shape[1] == pos_out.shape[1], 'On generation, all dimensions must be equal'
        out = []
        for idx in range(emb_out.shape[1]):
            cur_emb, cur_col, cur_pos = emb_out[0, idx], col_out[0, idx], pos_out[0, idx]
            filtered_logits = top_k_top_p_filtering(cur_emb, top_k=1)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)
            emb_idx = sample.item()
            if emb_idx == 0:
                break
            out.append([emb_idx] + cur_col.tolist() + cur_pos.tolist())
        return np.asarray(out)

def piecewise_decay(epoch, steps, alphas):
    assert len(steps) == len(alphas), 'In piecewise_decay: Both steps and alphas need to have the same number of elements'
    steps, alphas = zip(*(sorted(zip(steps, alphas), reverse=True)))
    for idx in range(len(steps)):
        if epoch > steps[idx]:
            return alphas[idx]
    return 1

def linear_decay(epoch, start, end):
    return start + min(end, ((end-start)/float(epoch)))

def pretrain_encoder(model: nn.Module, image_size, train_data, valid_data, device, batch_size = 32, max_epochs = 100, max_patience = 5, batch_metrics=True, freeze_best=False):
    learner = Dino(
        model,
        image_size = image_size,
        hidden_layer= 'to_latent',
    ).to(device)    

    train_dataloder = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, drop_last=True)

    opt = optim.AdamW(learner.parameters(), lr=1e-2, weight_decay=0.04)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0 = len(train_dataloder) * 10, eta_min = 1e-6)
    
    best_loss = None
    best_model = None
    patience = 0

    for edx in range(max_epochs):
        model.train()
        learner.train()
        running_loss = 0.0
        for bdx, i_batch in enumerate(train_dataloder):
            img = i_batch['feature'].to(device)
            loss = learner(img)
            running_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average()
            if batch_metrics:
                print('\tBatch #{}, Loss: {}'.format(bdx, loss.item()))
            scheduler.step()
        
        print('TRAINING Epoch #{}, Total Loss: {}'.format(edx, running_loss))

        model.eval()
        learner.eval()
        running_loss = 0.0
        with torch.no_grad():
            for bdx, i_batch in enumerate(valid_dataloader):
                img = i_batch['feature'].to(device)
                loss = learner(img)
                running_loss += loss.item()
        
        if best_loss is None or running_loss < best_loss:
            best_loss = running_loss
            best_model = model.state_dict()
            patience = 0
        else:
            patience += 1
        
        print('VALIDATION Epoch #{}, Total Loss: {}, Patience: {}'.format(edx, running_loss, patience))

        if patience > max_patience:
            print('Out of patience')
            break

        if best_loss == 0.0:
            print("We aren't doing any better than this...")
            break
    
    model.load_state_dict(best_model)
    if freeze_best:
        for param in model.parameters():
            param.requires_grad = False
    return model


# The main function
def main():
    x_settings = {'image_size': 192, 'patch_size': 8, 'dim': 128, 'e_depth': 1, 'e_heads': 12, 'd_depth': 1, 'd_heads': 12, 'clamped_values': True}
    debug = False # Debugging your model on CPU is leagues easier
    if debug:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0') if torch.cuda.device_count() > 0 else torch.device('cpu')
    if os.path.exists('x_train.csv'):
        os.remove('x_train.csv')

    batch_size = 16
    valid_split = 0.2
    epochs = 1000000
    eval_every = 1
    patience = 0
    best_loss = None
    best_model = None

    vocab, data = load_data(clamp_values=x_settings['clamped_values'])
    random.shuffle(data)
    
    model = NeuralTransformer(
        image_size = x_settings['image_size'],
        patch_size = x_settings['patch_size'],
        dim=x_settings['dim'],
        e_depth=x_settings['e_depth'],
        e_heads=x_settings['e_heads'],
        vocab=vocab,
        d_depth=x_settings['d_depth'],
        d_heads=x_settings['d_heads'],
    ).to(device)
    with open('best_model.json', 'w') as f:
        json.dump(x_settings, f, indent=3)

    train_split, valid_split = data[int(len(data)*valid_split):], data[:int(len(data)*valid_split)]

    train_dataset, valid_dataset = SADataset(train_split, img_size=x_settings['image_size']), SADataset(valid_split, img_size=x_settings['image_size'])
    train_dataloader, valid_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(valid_dataset, batch_size=batch_size)

    #model.encoder = pretrain_encoder(model.encoder, x_settings['image_size'], train_dataset, valid_dataset, device, max_patience=20)

    print('Total model parameters:\n\tTrainable: {}\n\tUntrainable: {}'.format(*(get_parameter_count(model))))

    # With AdamW
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay = 1e-4)
    #optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay = 1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=len(train_dataloader)*10, eta_min = 1e-6)

    scaler = GradScaler()

    for edx in range(epochs):
        running_loss = 0
        model.train()
        for bdx, i_batch in enumerate(train_dataloader):
            img, saml = i_batch['feature'].to(device), i_batch['label'].to(device)
            loss = 0
            optimizer.zero_grad()

            with autocast():
                loss = model(img, saml, return_loss=True)

            running_loss += loss.item()
            scaler.scale(loss).backward()

            #scaler.unscale_(optimizer)
            
            #torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), 0.5)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            print('\tBatch #{}, Loss: {}'.format(bdx, loss.item()))

        print('Training Epoch #{}, Loss: {}, LR: {:.4e}'.format(edx, running_loss/len(train_dataloader), scheduler.get_last_lr()[0]))
        #print('Training Epoch #{}, Loss: {}'.format(edx, running_loss/len(train_dataloader)))
        train_loss = running_loss
        model.eval()
        running_loss = 0
        

        with torch.no_grad():
            for bdx, i_batch in enumerate(tqdm(valid_dataloader, desc='Validation', leave=False)):
                img, saml = i_batch['feature'].to(device), i_batch['label'].to(device)
                running_loss += model(img, saml, return_loss=True).item()
                    
                
        
        
        with open('x_train.csv', 'a') as f:
            f.write('{},{},{},{},{}\n'.format(edx, train_loss, running_loss, train_loss/len(train_dataloader), running_loss/len(valid_dataloader)))
        
        if edx % eval_every == 0:
            feature = load_image('PleaseWork.png', image_size=x_settings['image_size'])
            saml = model.generate(feature, device=device)
            convert_numpy_to_saml(saml, vocab, name='xtransform', values_clamped=x_settings['clamped_values'])

            feature = train_dataset[0]['feature']
            saml = model.generate(feature, device=device)
            convert_numpy_to_saml(saml, vocab, name='xtraintransform', values_clamped=x_settings['clamped_values'])

        if best_loss is None or running_loss < best_loss:
            best_loss = running_loss
            best_model = model.state_dict()
            patience = 0
        else:
            patience += 1
        
        print('Validation Epoch #{}, Loss: {}, Patience: {}/50'.format(edx, running_loss/len(valid_dataloader), patience))

        if patience > 50:
            print('Out of patience')
            break

    model.load_state_dict(best_model)
    feature = load_image('PleaseWork.png', image_size=x_settings['image_size'])
    saml = model.generate(feature, device=device)
    convert_numpy_to_saml(saml, vocab, name='xtransform', values_clamped=x_settings['clamped_values'])
    torch.save(best_model, 'best_model.pt')


if __name__ == '__main__':
    main()
