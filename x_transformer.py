# Short and sweet one file version of what I am trying to do. Under 200 lines and includes the model right in here. Only things external are data related

import glob
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

from x_transformers import ContinuousTransformerWrapper, ViTransformerWrapper, Encoder, Decoder

from model.datasets import SADataset
from model.utils import load_data

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
    
    def forward(self, saml, mask=None, context=None, return_loss=False):
        if mask is not None and mask.shape[1] == saml.shape[1]:
            mask = mask[:, :-1]
        x, y = saml[:, :-1], saml[:, 1:]
        x = self.embed_saml(x)
        out = self.decoder(x, mask=mask, context=context)
        emb_guess, val_guess = torch.split(out, [self.embedding.embedding_dim, 12], dim=-1)
        emb_guess = self.to_classes(emb_guess).squeeze(-1).transpose(1,2)
        if return_loss:
            emb_target, val_target = torch.split(y, [1, 12], dim=-1)
            return F.cross_entropy(emb_guess, emb_target.squeeze(-1).long()) + F.mse_loss(val_guess, val_target)
        out = torch.cat([emb_guess, val_guess], dim=-1)
        return out

class NeuralTransformer(nn.Module):
    def __init__(self, image_size, patch_size, dim, e_depth, e_heads, vocab, emb_dim, d_depth, d_heads, max_seq_len=227):
        super(NeuralTransformer, self).__init__()
        self.encoder = XEncoder(image_size, patch_size, dim, e_depth, e_heads)
        self.decoder = XDecoder(len(vocab), emb_dim, max_seq_len, dim, d_depth, d_heads)
    
    def forward(self, src, tgt, tgt_mask=None, return_loss=False):
        enc = self.encoder(src)
        out = self.decoder(tgt, mask=tgt_mask, context=enc, return_loss=return_loss)
        return out
    
    def generate(self, src):
        pass

if __name__ == '__main__':
    debug = False # Debugging your model on CPU is leagues easier
    if debug:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0') if torch.cuda.device_count() > 0 else torch.device('cpu')
    
    vocab, data = load_data(clamp_values=True)
    random.shuffle(data)
    model = NeuralTransformer(
        image_size=192,
        patch_size=32,
        dim=32,
        e_depth=1,
        e_heads=8,
        vocab=vocab,
        emb_dim=8,
        d_depth=1,
        d_heads=1
    ).to(device)
    valid_split = 0.2
    train_split, valid_split = data[int(len(data)*valid_split):], data[:int(len(data)*valid_split)]
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    train_dataset, valid_dataset = SADataset(train_split, img_size=192), SADataset(valid_split, img_size=192)
    train_dataloader, valid_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True), DataLoader(valid_dataset, batch_size=4)

    epochs = 10
    max_seq_len = 227
    accumulate = 8
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
        
        if best_loss is None or running_loss < best_loss:
            best_loss = running_loss
            best_model = model.state_dict()
            patience = 0
        else:
            patience += 1


        if patience > 10:
            print('Out of patience')
            break
    
    model.load_state_dict(best_model)
    torch.save(model, 'best_model.pt')