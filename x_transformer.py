# Imports
import random

import numpy as np

import skimage.io as io

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

from x_transformers import ContinuousTransformerWrapper, ViTransformerWrapper, Encoder, Decoder

from model.datasets import SADataset
from model.utils import Vocabulary, convert_numpy_to_saml, get_parameter_count, load_data, load_image

# Sampler for generation
def top_p(logits, thres = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

# Model classes
class XEncoder(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads):
        super(XEncoder, self).__init__()
        self.encoder = ViTransformerWrapper(
            image_size=image_size,
            patch_size=patch_size,
            attn_layers=Encoder(
                dim=dim,
                depth=depth,
                heads=heads,
                use_scalenorm=True,
                rotary_pos_emb=True,
            )
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return x

class XDecoder(nn.Module):
    def __init__(self, num_layers, emb_dim, max_seq_len, dim, depth, heads):
        super(XDecoder, self).__init__()
        self.max_seq_len = max_seq_len
        dim_in = emb_dim + 12
        dim_out = dim_in
        self.decoder = ContinuousTransformerWrapper(
            dim_in=dim_in,
            dim_out=dim_out,
            max_seq_len=max_seq_len,
            attn_layers=Decoder(
                dim=dim,
                depth=depth,
                heads=heads,
                rel_pos_emb=True,
                rel_pos_num_buckets=8,
                rel_pos_max_distance=32,
                rotary_pos_emb=True,
                use_scalenorm=True,
            ),
        )

        self.embedding = nn.Embedding(num_embeddings=num_layers, embedding_dim=emb_dim)
        self.to_classes = nn.Linear(emb_dim, num_layers)
    
    def embed_saml(self, saml):
        x, y = torch.split(saml, [1, 12], dim=-1)
        x = self.embedding(x.long()).squeeze(2)
        return torch.cat([x, y], dim=-1)
    
    def forward(self, saml, mask=None, context=None):
        x = self.embed_saml(saml)
        out = self.decoder(x, mask=mask, context=context)
        emb_guess, col_guess, pos_guess = torch.split(out, [self.embedding.embedding_dim, 4,8], dim=-1)
        col_guess = torch.sigmoid(col_guess)
        pos_guess = torch.tanh(pos_guess)
        emb_guess = self.to_classes(emb_guess)
        return emb_guess, col_guess, pos_guess

class NeuralTransformer(nn.Module):
    def __init__(self, image_size, patch_size, dim, e_depth, e_heads, vocab, emb_dim, d_depth, d_heads, max_seq_len=227):
        super(NeuralTransformer, self).__init__()
        self.encoder = XEncoder(image_size, patch_size, dim, e_depth, e_heads)
        self.decoder = XDecoder(len(vocab), emb_dim, max_seq_len, dim, d_depth, d_heads)
    
    def forward(self, src, tgt, tgt_mask=None, return_loss=False):
        enc = self.encoder(src)
        features, labels = tgt[:,:-1,:], tgt[:,1:,:]
        fmask, _ = None, None
        if tgt_mask is not None:
            fmask, _ = tgt_mask[:,:-1], tgt_mask[:,1:]
        emb_guess, col_guess, pos_guess = self.decoder(features, mask=fmask, context=enc)
        if return_loss:
            emb_target, col_target, pos_target = torch.split(labels, [1, 4, 8], dim=-1)
            return F.cross_entropy(emb_guess.transpose(1,2), emb_target.squeeze(-1).long()) + F.mse_loss(col_guess, col_target) + F.mse_loss(pos_guess, pos_target)
        return torch.cat([emb_guess, col_guess, pos_guess], dim=-1)
    
    @torch.no_grad()
    def generate(self, src: torch.Tensor, vocab: Vocabulary, max_len=225, temperature = 1.0, device=None) -> np.ndarray:
        if len(src.shape) == 3:
            src = src.unsqueeze(0).to(device)
        context = self.encoder(src)
        out = [[vocab['<SOS>']] + [0] * 12]
        eos_token = vocab['<EOS>']
        input_mask = [True]
        while len(out) < max_len+3: # + 3 for <SOS>, <EOS> and to loop the right amount of times
            x = torch.tensor(out).unsqueeze(0).to(device)
            mask = torch.tensor(input_mask, dtype=torch.bool).unsqueeze(0).to(device)
            emb_out, col_out, pos_out = self.decoder(x, mask, context)
            emb_out, col_out, pos_out = emb_out[0][-1:], col_out[0][-1:], pos_out[0][-1:]
            filtered_logits = top_p(emb_out)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)
            emb_idx = sample.item()
            if emb_idx == eos_token:
                break
            colors = list(map(float, col_out.squeeze().tolist()))
            positions = list(map(float, pos_out.squeeze().tolist()))
            out.append([emb_idx] + colors + positions)
            input_mask.append(True)
        return np.asarray(out)

# The main function
def main():
    debug = False # Debugging your model on CPU is leagues easier
    if debug:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0') if torch.cuda.device_count() > 0 else torch.device('cpu')
    
    vocab, data = load_data(clamp_values=True)
    random.shuffle(data)
    model = NeuralTransformer(
        image_size=192,
        patch_size=64,
        dim=64,
        e_depth=2,
        e_heads=8,
        vocab=vocab,
        emb_dim=16,
        d_depth=4,
        d_heads=16
    ).to(device)
    print('Total model parameters:\n\tTrainable: {}\n\tUntrainable: {}'.format(*(get_parameter_count(model))))
    valid_split = 0.2
    train_split, valid_split = data[int(len(data)*valid_split):], data[:int(len(data)*valid_split)]
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    train_dataset, valid_dataset = SADataset(train_split, img_size=192), SADataset(valid_split, img_size=192)
    train_dataloader, valid_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True), DataLoader(valid_dataset, batch_size=4)

    epochs = 100000
    max_seq_len = 227
    accumulate = 8
    eval_every = 1
    idxs = []
    patience = 0
    best_loss = None
    best_model = None

    for edx in range(epochs):
        running_loss = 0
        model.train()
        for bdx, i_batch in enumerate(train_dataloader):
            img, saml, mask = i_batch['feature'].to(device), i_batch['label'].to(device), i_batch['mask'].to(device)
            loss = 0
            optimizer.zero_grad()

            for _ in range(accumulate):
                if len(idxs) == 0:
                    idxs = list(range(2,max_seq_len))
                    random.shuffle(idxs)
                
                idx = idxs.pop(0)
                x, xm = saml[:,:idx], mask[:,:idx]
                loss += model(img, x, xm, return_loss=True)
            
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            print('\tBatch #{}, Loss: {}'.format(bdx, loss.item()))
        print('Training Epoch #{}, Loss: {}'.format(edx, running_loss))
        model.eval()
        running_loss = 0

        for bdx, i_batch in enumerate(tqdm(valid_dataloader, desc='Validation', leave=False)):
            img, saml, mask = i_batch['feature'].to(device), i_batch['label'].to(device), i_batch['mask'].to(device)
            for idx in range(2, len(saml[0])):
                running_loss += model(img, saml[:idx], mask[:idx], return_loss=True).item()
        
        print('Validation Epoch #{}, Loss: {}'.format(edx, running_loss))

        if edx % eval_every == 0:
            feature = load_image('PleaseWork.png', image_size=192)
            saml = model.generate(feature, vocab, device=device)
            convert_numpy_to_saml(saml, vocab, name='xtransform', values_clamped=True)

        if best_loss is None or running_loss < best_loss:
            best_loss = running_loss
            best_model = model.state_dict()
            patience = 0
        else:
            patience += 1

        if patience > 50:
            print('Out of patience')
            break

    model.load_state_dict(best_model)
    torch.save(model, 'best_model.pt')


if __name__ == '__main__':
    main()