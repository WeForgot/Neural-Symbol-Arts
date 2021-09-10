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
from torch.utils.data import DataLoader, TensorDataset, random_split
from adabelief_pytorch import AdaBelief

from tqdm import tqdm
from model.custom_gmlp import gMLPVision
from model.mobilenetv3 import mobilenet_v3_small
#from model.resnet import resnet18
from model.custom_vit import ViT
from vit_pytorch.mpp import MPP
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
        #self.encoder = mobilenet_v3_small(dim)
        #self.encoder = resnet18(dim)
        '''
        self.encoder = gMLPVision(
            image_size = image_size,
            patch_size = patch_size,
            dim = dim,
            depth = depth,
            channels = 3,
            prob_survival = 0.9
        )
        '''
        self.encoder = ViT(
            image_size = image_size,
            patch_size = patch_size,
            dim = dim,
            depth = depth,
            heads = heads,
        )
        
        self.to_latent = nn.Identity()
    
    def forward(self, x):
        x = self.encoder(x)
        return self.to_latent(x)
    
    def pretrain(self, train_data, valid_data, patch_size, dim, mask_prob=0.15, random_patch_prob=0.30, replace_prob=0.50, device='cuda'):
        optimizer = optim.AdamW(self.encoder.parameters(), lr = 1e-4, weight_decay = 1e-4)
        train_data_t, valid_data_t = torch.cat([data['feature'].unsqueeze(0) for data in train_data]), torch.cat([data['feature'].unsqueeze(0) for data in valid_data])
        train_data_t, valid_data_t = TensorDataset(train_data_t), TensorDataset(valid_data_t)

        train_dataloader, valid_dataloader = DataLoader(train_data_t, batch_size = 32, shuffle = True), DataLoader(valid_data_t, batch_size = 32)

        mpp = MPP(
            transformer = self.encoder,
            patch_size = patch_size,
            dim = dim,
            mask_prob = mask_prob,
            random_patch_prob = random_patch_prob,
            replace_prob = replace_prob
        ).to(device)

        best_loss = None
        best_model = None

        patience = 0
        max_patience = 10

        for edx in range(100):
            self.encoder.train()
            running_loss = 0.0
            for bdx, i_batch in enumerate(train_dataloader):
                loss = mpp(i_batch[0].to(device))
                running_loss += loss.item()
                print('\tBatch #{}, Loss: {}'.format(bdx, loss.item()))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('Training epoch #{}, Running loss: {}'.format(edx, running_loss))
            
            self.encoder.eval()
            running_loss = 0.0
            for bdx, i_batch in enumerate(valid_dataloader):
                with torch.no_grad():
                    running_loss += mpp(i_batch[0].to(device)).item()
            if best_loss is None or running_loss < best_loss:
                best_loss = running_loss
                best_model = self.encoder.state_dict()
                patience = 0
            else:
                patience += 1
            print('Validation epoch #{}, Running loss: {}, Patience: {}/{}'.format(edx, running_loss, patience, max_patience))

            if patience >= max_patience:
                print('Out of patience')
                break
        
        self.encoder.load_state_dict(best_model)
        for param in self.encoder.parameters():
            param.requires_grad = False


class XDecoder(nn.Module):
    def __init__(self, num_layers, emb_dim, max_seq_len, dim, depth, heads, ff_mult=2, final_depth=2):
        super(XDecoder, self).__init__()
        self.max_seq_len = max_seq_len

        self.decoder = ContinuousTransformerWrapper(
            max_seq_len = 227,
            attn_layers = Decoder(
                dim = dim,
                depth = depth,
                heads = heads,
                cross_attend = True,
                alibi_pos_emb = True,
            ),
            dim_in = emb_dim + 12,
            dim_out = dim,
        )


        self.embedding = nn.Embedding(num_embeddings=num_layers, embedding_dim=emb_dim)
        self.pos_emb = nn.parameter.Parameter(torch.zeros((227, dim)), requires_grad = True)
        self.post_proj = nn.Linear(emb_dim+12, dim)
        self.post_norm = nn.LayerNorm(dim)
        self.post_drop = nn.Dropout(p=0.2)

        ffc = nn.ModuleList(
            [nn.Sequential(
                PreNorm(dim*ff_mult, Residual(nn.Linear(dim*ff_mult, dim*ff_mult))),
                nn.GELU(),
                nn.Dropout(0.1)
            )
            for _ in range(final_depth)]
        )

        self.to_classes = nn.Sequential(
            nn.Linear(dim, dim*ff_mult),
            *ffc,
            nn.Linear(dim*ff_mult, num_layers),
        )

        ffl = nn.ModuleList(
            [nn.Sequential(
                PreNorm(dim*ff_mult, Residual(nn.Linear(dim*ff_mult, dim*ff_mult))),
                nn.GELU(),
                nn.Dropout(0.1)
            )
            for _ in range(final_depth)]
        )
        self.to_colors = nn.Sequential(
            nn.Linear(dim, dim*ff_mult),
            *ffl,
            nn.Linear(dim*ff_mult, 4),
            nn.Sigmoid(),
        )

        ffp = nn.ModuleList(
                [nn.Sequential(
                PreNorm(dim*ff_mult, Residual(nn.Linear(dim*ff_mult, dim*ff_mult))),
                nn.GELU(),
                nn.Dropout(0.1)
            )
            for _ in range(final_depth)]
        )
        self.to_positions = nn.Sequential(
            nn.Linear(dim, dim*ff_mult),
            *ffp,
            nn.Linear(dim*ff_mult, 8),
            nn.Tanh(),
        )

    def embed_saml(self, saml):
        x, y = torch.split(saml, [1, 12], dim=-1)
        x = self.embedding(x.long()).squeeze(2)
        return torch.cat([x, y], dim=-1)
    
    def forward(self, saml, context=None, mask=None):
        x = self.embed_saml(saml)
        out = self.decoder(x, context=context, mask=mask)
        emb_guess, col_guess, pos_guess = self.to_classes(out), self.to_colors(out), self.to_positions(out)
        return emb_guess, col_guess, pos_guess

class NeuralTransformer(nn.Module):
    def __init__(self, image_size, patch_size, dim, e_depth, e_heads, vocab, emb_dim, d_depth, d_heads, max_seq_len=227):
        super(NeuralTransformer, self).__init__()
        self.encoder = XEncoder(image_size, patch_size, dim, e_depth, e_heads)
        self.decoder = XDecoder(len(vocab), emb_dim, max_seq_len, dim, d_depth, d_heads)
    
    def forward(self, src, tgt, mask=None, return_loss=False, emb_alpha=1.0, col_alpha=1.0, pos_alpha=1.0):
        enc = self.encoder(src)
        mask = torch.ones(tgt.shape[:2]).bool().cuda()
        mask[0][-1] = False
        emb_guess, col_guess, pos_guess = self.decoder(tgt, context=enc, mask=mask)
        emb_guess, col_guess, pos_guess = emb_guess[:, -1:, :], col_guess[:, -1:, :], pos_guess[:, -1:, :]
        if return_loss:
            emb_target, col_target, pos_target = torch.split(tgt, [1, 4, 8], dim=-1)
            emb_target, col_target, pos_target = emb_target[:, -1:, :], col_target[:, -1:, :], pos_target[:, -1:, :]
            emb_loss = F.cross_entropy(emb_guess.transpose(1,2), emb_target.squeeze(-1).long()) * emb_alpha
            col_loss = F.smooth_l1_loss(col_guess, col_target) * col_alpha
            pos_loss = F.smooth_l1_loss(pos_guess, pos_target) * pos_alpha
            return emb_loss + col_loss + pos_loss
        return torch.cat([emb_guess, col_guess, pos_guess], dim=-1)
    
    @torch.no_grad()
    def generate(self, src: torch.Tensor, vocab: Vocabulary, max_len=226, temperature = 1.0, device=None) -> np.ndarray:
        if len(src.shape) == 3:
            src = src.unsqueeze(0).to(device)
        context = self.encoder(src)
        out = [[vocab['<SOS>']] + [0] * 12]
        eos_token = vocab['<EOS>']
        while len(out) < max_len: # + 3 for <SOS>, <EOS> and to loop the right amount of times
            temp = list(out)
            temp.append([0.0] * 13)
            x = torch.tensor(temp).unsqueeze(0).to(device)
            mask = mask = torch.ones(x.shape[:2]).bool().cuda()
            mask[0][-1] = False
            emb_out, col_out, pos_out = self.decoder(x, context=context, mask=mask)
            emb_out, col_out, pos_out = emb_out[:, -1:, :], col_out[:, -1:, :], pos_out[:, -1:, :]
            filtered_logits = top_k_top_p_filtering(emb_out, top_k=1)
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
    x_settings = {'image_size': 224, 'patch_size': 8, 'dim': 128, 'e_depth': 3, 'e_heads': 6, 'emb_dim': 8, 'd_depth': 6, 'd_heads': 16, 'clamped_values': True}
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

    
    valid_split = 0.2
    train_split, valid_split = data[int(len(data)*valid_split):], data[:int(len(data)*valid_split)]

    train_dataset, valid_dataset = SADataset(train_split, img_size=x_settings['image_size']), SADataset(valid_split, img_size=x_settings['image_size'])
    #train_dataloader, valid_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(valid_dataset, batch_size=batch_size)

    #model.encoder.pretrain(train_dataset, valid_dataset, x_settings['patch_size'], x_settings['dim'], device=device)

    print('Total model parameters:\n\tTrainable: {}\n\tUntrainable: {}'.format(*(get_parameter_count(model))))

    # With AdamW
    #optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay = 1e-4)
    optimizer = optim.SGD(model.parameters(), lr=1e-2, weight_decay = 1e-4)
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
                loss = model(img, x, return_loss=True) / accumulate
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
                        running_loss += model(img, saml[:,:idx], return_loss=True).item()
            
                
        
        
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
