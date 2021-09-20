# Imports
import json
import os
import random

import numpy as np

import torch
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm
from model.resnet import resnet18

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

# Model classes
class XEncoder(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads):
        super(XEncoder, self).__init__()
        self.encoder = resnet18(dim)

        self.to_latent = nn.Identity()
    
    def forward(self, x):
        x = self.encoder(x)
        return self.to_latent(x)

class XDecoder(nn.Module):
    def __init__(self, num_layers, emb_dim, max_seq_len, dim, depth, heads, ff_mult=2, final_depth=2):
        super(XDecoder, self).__init__()
        self.max_seq_len = max_seq_len

        decoder_layer = nn.TransformerDecoderLayer(d_model = dim, nhead = heads, batch_first = True)
        self.decoder = nn.TransformerDecoder(decoder_layer, depth)


        self.embedding = nn.Embedding(num_embeddings=num_layers, embedding_dim=emb_dim)
        self.projection = nn.Linear(emb_dim+12, dim)
        self.proj_norm = nn.LayerNorm(dim)

        ffc = nn.ModuleList(
            [nn.Sequential(
                PreNorm(dim*ff_mult, Residual(nn.Linear(dim*ff_mult, dim*ff_mult, bias=False))),
                nn.GELU(),
                nn.Dropout(0.1)
            )
            for _ in range(final_depth)]
        )

        self.to_classes = nn.Sequential(
            nn.Linear(dim, dim*ff_mult, bias=False),
            *ffc,
            nn.Linear(dim*ff_mult, num_layers, bias=False),
        )

        ffl = nn.ModuleList(
            [nn.Sequential(
                PreNorm(dim*ff_mult, Residual(nn.Linear(dim*ff_mult, dim*ff_mult, bias=False))),
                nn.GELU(),
                nn.Dropout(0.1)
            )
            for _ in range(final_depth)]
        )
        self.to_colors = nn.Sequential(
            nn.Linear(dim, dim*ff_mult, bias=False),
            *ffl,
            nn.Linear(dim*ff_mult, 4, bias=False),
            nn.Sigmoid(),
        )

        ffp = nn.ModuleList(
                [nn.Sequential(
                PreNorm(dim*ff_mult, Residual(nn.Linear(dim*ff_mult, dim*ff_mult, bias=False))),
                nn.GELU(),
                nn.Dropout(0.1)
            )
            for _ in range(final_depth)]
        )
        self.to_positions = nn.Sequential(
            nn.Linear(dim, dim*ff_mult, bias=False),
            *ffp,
            nn.Linear(dim*ff_mult, 8, bias=False),
            nn.Tanh(),
        )

    def embed_saml(self, saml):
        x, y = torch.split(saml, [1, 12], dim=-1)
        x = self.embedding(x.long()).squeeze(2)
        return torch.cat([x, y], dim=-1)
    
    def forward(self, saml, context=None, mask=None):
        x = self.embed_saml(saml)
        x = self.proj_norm(self.projection(x))
        out = self.decoder(x, context, tgt_key_padding_mask=~mask)
        emb_guess, col_guess, pos_guess = self.to_classes(out), self.to_colors(out), self.to_positions(out)
        return emb_guess, col_guess, pos_guess

class NeuralTransformer(nn.Module):
    def __init__(self, image_size, patch_size, dim, e_depth, e_heads, vocab, emb_dim, d_depth, d_heads, max_seq_len=227):
        super(NeuralTransformer, self).__init__()
        self.encoder = XEncoder(image_size, patch_size, dim, e_depth, e_heads)
        self.decoder = XDecoder(len(vocab), emb_dim, max_seq_len, dim, d_depth, d_heads)
        self.emb_loss = nn.CrossEntropyLoss()
        self.col_loss = nn.SmoothL1Loss()
        self.pos_loss = nn.SmoothL1Loss()
    
    def forward(self, src, tgt, mask=None, return_loss=False, emb_alpha=1.0, col_alpha=1.0, pos_alpha=1.0):
        enc = self.encoder(src)
        xi = tgt[:, :-1]
        xo = tgt[:, 1:]
        mask = torch.ones(tgt.shape[:2]).bool().cuda()
        if mask is not None and mask.shape[1] == tgt.shape[1]:
            mask = mask[:, :-1]
        emb_guess, col_guess, pos_guess = self.decoder(xi, context=enc, mask=mask)
        if return_loss:
            emb_target, col_target, pos_target = torch.split(xo, [1, 4, 8], dim=-1)
            emb_loss = self.emb_loss(emb_guess.transpose(1,2), emb_target.squeeze(-1).long()) * emb_alpha
            col_loss = self.col_loss(col_guess, col_target) * col_alpha
            pos_loss = self.pos_loss(pos_guess, pos_target) * pos_alpha
            return emb_loss + col_loss + pos_loss
        return torch.cat([emb_guess, col_guess, pos_guess], dim=-1)
    
    @torch.no_grad()
    def generate(self, src: torch.Tensor, vocab: Vocabulary, max_len=226, temperature = 1.0, k=1, device=None) -> np.ndarray:
        if len(src.shape) == 3:
            src = src.unsqueeze(0).to(device)
        context = self.encoder(src)
        out = [[vocab['<SOS>']] + [0] * 12]
        eos_token = vocab['<EOS>']
        while len(out) < max_len: # + 3 for <SOS>, <EOS> and to loop the right amount of times
            temp = list(out)
            x = torch.tensor(temp).unsqueeze(0).to(device)
            mask = torch.ones(x.shape[:2]).bool().cuda()
            emb_out, col_out, pos_out = self.decoder(x, context=context, mask=mask)
            emb_out, col_out, pos_out = emb_out[:, -1, :], col_out[:, -1, :], pos_out[:, -1, :]
            filtered_logits = top_k_top_p_filtering(emb_out, top_k=k)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)
            emb_idx = sample.item()
            if emb_idx == eos_token:
                break
            colors = list(map(float, col_out.squeeze().tolist()))
            positions = list(map(float, pos_out.squeeze().tolist()))
            out.append([emb_idx] + colors + positions)
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

# The main function
def main():
    x_settings = {'image_size': 224, 'patch_size': 8, 'dim': 384, 'e_depth': 1, 'e_heads': 8, 'emb_dim': 4, 'd_depth': 4, 'd_heads': 12, 'clamped_values': True}
    debug = False # Debugging your model on CPU is leagues easier
    if debug:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0') if torch.cuda.device_count() > 0 else torch.device('cpu')
    if os.path.exists('x_train.csv'):
        os.remove('x_train.csv')
    vocab, data = load_data(clamp_values=x_settings['clamped_values'])
    print('Number of data points available: {}'.format(len(data)))
    random.shuffle(data)
    
    model = NeuralTransformer(
        image_size=x_settings['image_size'],
        patch_size=x_settings['patch_size'],
        dim=x_settings['dim'],
        e_depth=x_settings['e_depth'],
        e_heads=x_settings['e_heads'],
        vocab=vocab,
        emb_dim=x_settings['emb_dim'],
        d_depth=x_settings['d_depth'],
        d_heads=x_settings['d_heads']
    ).to(device)
    with open('best_model.json', 'w') as f:
        json.dump(x_settings, f, indent=3)

    
    valid_split = 0.1
    train_split, valid_split = data[int(len(data)*valid_split):], data[:int(len(data)*valid_split)]

    train_dataset, valid_dataset = SADataset(train_split, img_size=x_settings['image_size']), SADataset(valid_split, img_size=x_settings['image_size'])

    print('Total model parameters:\n\tTrainable: {}\n\tUntrainable: {}'.format(*(get_parameter_count(model))))

    # With AdamW
    #optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay = 1e-6)
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum = 0.9, weight_decay = 1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, eta_min = 1e-6)

    scaler = GradScaler()

    epochs = 1000000
    max_seq_len = 227
    accumulate = 64
    eval_every = 1
    idxs = list(range(len(train_dataset)))
    patience = 0
    best_loss = None
    best_model = None

    for edx in range(epochs):
        running_loss = 0
        batch_loss = 0
        model.train()
        loss = 0
        bdx = 0
        random.shuffle(idxs)
        #for ddx, i_batch in enumerate(train_dataset):
        for ddx, i_batch in enumerate(idxs):
            #img, saml = i_batch['feature'].to(device).unsqueeze(0), i_batch['label'].to(device).unsqueeze(0)
            img, saml = train_dataset[i_batch]['feature'].to(device).unsqueeze(0), train_dataset[i_batch]['label'].to(device).unsqueeze(0)
            idx = random.choice(list(range(2, len(saml[0]))))
            x = saml[:,:idx]
            with autocast():
                loss = model(img, x, return_loss=True, emb_alpha=(1/60.)) / accumulate
            scaler.scale(loss).backward()
            batch_loss += loss.item()


            if (ddx + 1) % accumulate == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                running_loss += batch_loss
                print('\tBatch #{}, Loss: {}'.format(bdx, batch_loss))
                batch_loss = 0
                bdx += 1
                

            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            
            
            
        if loss != 0:
            optimizer.zero_grad()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

        #print('Training Epoch #{}, Loss: {}, LR: {:.4e}'.format(edx, running_loss, scheduler.get_last_lr()[0]))
        print('Training Epoch #{}, Loss: {}'.format(edx, running_loss))
        train_loss = running_loss
        model.eval()
        running_loss = 0
        

        with torch.no_grad():
            for bdx, i_batch in enumerate(tqdm(valid_dataset, desc='Validation', leave=False)):
                img, saml = i_batch['feature'].to(device).unsqueeze(0), i_batch['label'].to(device).unsqueeze(0)
                for idx in range(2, max_seq_len):
                    with autocast():
                        running_loss += model(img, saml[:,:idx], return_loss=True, emb_alpha=(1/60.)).item()
            
                
        
        
        with open('x_train.csv', 'a') as f:
            f.write('{},{},{},{},{}\n'.format(edx, train_loss, running_loss, train_loss/len(train_dataset), running_loss/len(valid_dataset)))
        
        if edx % eval_every == 0:
            feature = load_image('PleaseWork.png', image_size=x_settings['image_size'])
            saml = model.generate(feature, vocab, device=device)
            convert_numpy_to_saml(saml, vocab, name='xtransform', values_clamped=x_settings['clamped_values'])

            feature = train_dataset[0]['feature']
            saml = model.generate(feature, vocab, device=device)
            convert_numpy_to_saml(saml, vocab, name='xtraintransform', values_clamped=x_settings['clamped_values'])

        if best_loss is None or running_loss < best_loss:
            best_loss = running_loss
            best_model = model.state_dict()
            patience = 0
        else:
            patience += 1
        
        print('Validation Epoch #{}, Loss: {}, Patience: {}/20'.format(edx, running_loss, patience))

        if patience > 20:
            print('Out of patience')
            break

    model.load_state_dict(best_model)
    feature = load_image('PleaseWork.png', image_size=x_settings['image_size'])
    saml = model.generate(feature, vocab, device=device)
    convert_numpy_to_saml(saml, vocab, name='xtransform', values_clamped=x_settings['clamped_values'])
    torch.save(best_model, 'best_model.pt')


if __name__ == '__main__':
    main()
