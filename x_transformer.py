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

from torchvision.utils import save_image

#from byol_pytorch import BYOL
from model.resnet import resnet18, resnet50
from model.custom_gmlp import gMLPVision
from vit_pytorch import Dino
from model.custom_vit import ViT
from x_transformers import ContinuousTransformerWrapper, Decoder, ViTransformerWrapper, Encoder
from model.remixer import RemixerEncoder, RemixerDecoder


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

class RemixerLayer(nn.Module):
    def __init__(self, seq_len, dim, heads):
        super().__init__()
        self.memory = nn.parameter.Parameter(torch.zeros(seq_len, dim))
        self.alpha = nn.parameter.Parameter(torch.zeros(1))

        self.mha = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.lin = nn.Linear(dim, dim*2)
        self.glu = nn.GLU()
    
    def forward(self, x):
        x1, _ = self.mha(x, x, x)
        x = self.norm(x + x1)
        x = self.lin(x)
        x = self.glu(x)
        gr = self.memory.softmax(dim=-1)
        gr = x * gr
        dt = x * gr
        sb = x - gr
        x = self.alpha * dt + (1 - self.alpha) * sb
        return x
    
    def _init_weights(self):
        nn.init.xavier_normal_(self.memory)
        nn.init.uniform_(self.alpha, 0, 1)


# Model classes
class XEncoder(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, pool_dim=256):
        super(XEncoder, self).__init__()

        # Backbone
        self.backbone = resnet50(dim)
        self.reduction = nn.AdaptiveAvgPool2d((pool_dim, pool_dim))
        self.projection = nn.Linear(pool_dim, dim)


        # Encoder
        #encoder_layer = nn.TransformerEncoderLayer(dim, heads, batch_first=True)
        #self.encoder = nn.TransformerEncoder(encoder_layer, depth)
        #stacks = []
        #for x in range(depth):
        #    stacks.append(RemixerLayer(pool_dim, dim, heads))
        #self.encoder = nn.Sequential(*stacks)
        self.encoder = RemixerEncoder(pool_dim, dim, heads, depth)
        self.pos_emb = nn.Embedding(pool_dim, dim)
        self.to_latent = nn.Identity()
    
    def forward(self, x):
        b = x.shape[0]
        x = self.backbone(x)
        x = self.reduction(x)
        x = self.projection(x)
        x = x + self.pos_emb.weight.repeat(b, 1, 1)
        x = self.encoder(x)



        
        return self.to_latent(x)
        '''
        x = self.encoder(x)
        return self.to_latent(x)
        '''


class XDecoder(nn.Module):
    def __init__(self, num_layers, max_seq_len, dim, depth, heads, ff_mult=1):
        super(XDecoder, self).__init__()
        self.max_seq_len = max_seq_len
        self.dim = dim

        self.pos_embs = nn.Embedding(max_seq_len, dim)
        #decoder_layer = nn.TransformerDecoderLayer(dim, heads, batch_first=True)
        #self.decoder = nn.TransformerDecoder(decoder_layer, depth) 
        self.decoder = RemixerDecoder(max_seq_len, dim, heads, depth)

        self.to_classes = nn.Linear(dim, num_layers)

        self.to_colors = nn.Sequential(
            *(
                [
                    nn.Sequential(
                        nn.Linear(dim, dim),
                        nn.LayerNorm(dim),
                    )
                    for _ in range(3)
                ]
            ),
            nn.Linear(dim, 4)
        )

        self.to_positions = nn.Sequential(
            *(
                [
                    nn.Sequential(
                        nn.Linear(dim, dim),
                        nn.LayerNorm(dim),
                    )
                    for _ in range(3)
                ]
            ),
            nn.Linear(dim, 8)
        )
    
    def forward(self, context, mask=None):
        b = context.shape[0]
        if mask is None:
            mask = torch.zeros((b, context.shape[0])).bool().cuda()
        x = self.pos_embs.weight.repeat(b, 1, 1)
        out = self.decoder(x, context)
        emb_guess, col_guess, pos_guess = self.to_classes(out), self.to_colors(out).sigmoid(), self.to_positions(out).tanh()
        return emb_guess, col_guess, pos_guess

def masked_mse(guess, target, mask):
    out = (guess[~mask]-target[~mask])**2
    loss = out.mean()
    return loss

def masked_l1(guess, target, mask):
    out = torch.abs((guess[~mask]-target[~mask]))
    loss = out.mean()
    return loss

def masked_smooth_l1(guess, target, mask, beta=1.0):
    n = torch.abs(guess[~mask]-target[~mask])
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    return loss

def focal_loss(guess, target, gamma=2):
    ce_loss = F.cross_entropy(guess, target) 
    pt = torch.exp(-ce_loss)
    focal_loss = ((1 - pt) ** gamma * ce_loss).mean()
    return focal_loss

class NeuralTransformer(nn.Module):
    def __init__(self, image_size, patch_size, dim, e_depth, e_heads, vocab, d_depth, d_heads, max_seq_len=225):
        super(NeuralTransformer, self).__init__()
        self.encoder = XEncoder(image_size, patch_size, dim, e_depth, e_heads)
        self.decoder = XDecoder(len(vocab), max_seq_len, dim, d_depth, d_heads)
    
    def forward(self, src, labels, mask, return_loss=False, emb_alpha=1.0, col_alpha=1.0, pos_alpha=1.0):
        enc = self.encoder(src)
        emb_guess, col_guess, pos_guess = self.decoder(enc, mask)
        if return_loss:
            emb_target, col_target, pos_target = torch.split(labels, [1, 4, 8], dim=-1)
            emb_loss = focal_loss(emb_guess.transpose(1,2), emb_target.squeeze(-1).long()) * emb_alpha
            col_loss = masked_smooth_l1(col_guess, col_target, mask) * col_alpha
            pos_loss = masked_smooth_l1(pos_guess, pos_target, mask) * pos_alpha
            return emb_loss + col_loss + pos_loss
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
            out.append([emb_idx] + cur_col.tolist() + cur_pos.tolist())
        return np.asarray(out)


# The main function
def main():
    #x_settings = {'image_size': 224, 'patch_size': 8, 'dim': 768, 'e_depth': 4, 'e_heads': 12, 'd_depth': 7, 'd_heads': 12, 'clamped_values': True} # GPT-2 style
    x_settings = {'image_size': 576, 'patch_size': 8, 'dim': 36, 'e_depth': 2, 'e_heads': 6, 'd_depth': 2, 'd_heads': 6, 'clamped_values': True}
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

    print('Total model parameters:\n\tTrainable: {}\n\tUntrainable: {}'.format(*(get_parameter_count(model))))
    print('Total data points:\n\tTraining: {}\n\tValidation: {}'.format(len(train_split), len(valid_split)))

    #optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay = 1e-4)
    optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=len(train_dataloader)*10, eta_min = 1e-6)

    scaler = GradScaler()

    for edx in range(epochs):
        running_loss = 0
        model.train()
        for bdx, i_batch in enumerate(train_dataloader):
            img, saml, mask = i_batch['feature'].to(device), i_batch['label'].to(device), i_batch['mask'].to(device)
            loss = 0
            optimizer.zero_grad()

            with autocast():
                loss = model(img, saml, mask, return_loss=True)

            running_loss += loss.item()
            scaler.scale(loss).backward()

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
                img, saml, mask = i_batch['feature'].to(device), i_batch['label'].to(device), i_batch['mask'].to(device)
                with autocast():
                    running_loss += model(img, saml, mask, return_loss=True).item()
                    
                
        
        
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
        
        print('Validation Epoch #{}, Loss: {}, Patience: {}/20'.format(edx, running_loss/len(valid_dataloader), patience))

        if patience > 20:
            print('Out of patience')
            break

    model.load_state_dict(best_model)
    feature = load_image('PleaseWork.png', image_size=x_settings['image_size'])
    saml = model.generate(feature, device=device)
    convert_numpy_to_saml(saml, vocab, name='xtransform', values_clamped=x_settings['clamped_values'])
    torch.save(best_model, 'best_model.pt')


if __name__ == '__main__':
    main()
