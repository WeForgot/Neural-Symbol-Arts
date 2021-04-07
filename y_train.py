import os
import pickle
import random

from dotenv import load_dotenv
import numpy as np
from skimage import io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from vit_pytorch.vit import Transformer
from x_transformers import Decoder, AutoregressiveWrapper
from nystrom_attention import Nystromformer

from models.datasets import SADataset
from models.utils import get_parameter_count, Vocabulary

load_dotenv()

def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, transformer, pool = 'cls', channels = 3):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = transformer

        self.pool = pool
        self.to_latent = nn.Identity()

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.transformer(x)

        return self.to_latent(x)

class BasicEncoder(nn.Module):
    def __init__(self, patch_size = 32, dim = 16, e_depth = 6, e_heads = 8):
        super(BasicEncoder, self).__init__()
        self.encoder = ViT(
            image_size = 576,
            patch_size = patch_size,
            dim = dim,
            transformer = Nystromformer(
                dim = dim,
                depth = e_depth,
                heads = e_heads,
            )
        )
    
    def forward(self, x):
        return self.encoder(x)

class BasicDecoder(nn.Module):
    def __init__(self, layer_count = 388, emb_dim = 8, dim = 16, d_depth = 12, d_heads = 8):
        super(BasicDecoder, self).__init__()
        self.layer_count = layer_count
        self.emb_dim = emb_dim
        self.latent_dim = emb_dim + 12
        self.logit_dim = layer_count + 12
        self.max_seq_len = 225

        self.embedding_dim = nn.Embedding(layer_count, emb_dim)
        self.decoder = Decoder(
            dim = self.latent_dim,
            depth = d_depth,
            heads = d_heads
        )
        self.to_logits = nn.Linear(self.latent_dim, self.logit_dim)
    
    def forward(self, src, mask=None, context=None):
        emb_idx, metrics = torch.split(src, [1, src.shape[-1] - 1], dim=-1)
        embs = self.embedding_dim(emb_idx.int()).squeeze(dim=2)
        y = torch.cat([embs, metrics], dim=-1)
        x = self.decoder(y, context=context, mask=mask)
        x = self.to_logits(x)
        x = torch.split(x, [self.layer_count, 12], dim=-1)
        return x

def CrossL1Loss(pred_embs: torch.Tensor, pred_locs: torch.Tensor, true: torch.Tensor, emb_dim: int):
    cl = nn.CrossEntropyLoss()
    ll = nn.L1Loss()
    true_emb_idx, true_l1_logits = torch.split(true, [1, true.shape[-1] - 1], dim=-1)
    true_emb_idx = torch.squeeze(true_emb_idx, dim=-1)
    
    cross_loss = nn.functional.cross_entropy(pred_embs.transpose(1,2), true_emb_idx.long())
    l1_loss = nn.functional.l1_loss(pred_locs, true_l1_logits)

    return cross_loss, l1_loss

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA available')
    else:
        device = torch.device('cpu')
        print('CUDA not available')
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
    encoder = BasicEncoder(
        dim= int(os.getenv('DIM', 128)),
        patch_size= int(os.getenv('PATCH_SIZE', 32)),
        e_depth = int(os.getenv('E_DEPTH', 6)),
        e_heads = int(os.getenv('E_HEADS', 8)),
    ).to(device)
    decoder = BasicDecoder(
        layer_count = len(vocab),
        emb_dim = int(os.getenv('EMB_DIM', 8)),
        dim= int(os.getenv('DIM', 128)),
        d_depth = int(os.getenv('D_DEPTH', 6)),
        d_heads = int(os.getenv('D_HEADS', 8)),
    ).to(device)
    print(encoder)
    trainable, untrainable = get_parameter_count(encoder)
    print('Total encoder paramters\n\tTrainable:\t{}\n\tUntrainable:\t{}'.format(trainable, untrainable))
    print(decoder)
    trainable, untrainable = get_parameter_count(decoder)
    print('Total decoder paramters\n\tTrainable:\t{}\n\tUntrainable:\t{}'.format(trainable, untrainable))
    
    
    dataset = SADataset(data)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    encoder_opt = optim.Adam(encoder.parameters(), lr=3e-4)
    decoder_opt = optim.Adam(decoder.parameters(), lr=3e-4)
    emb_loss = nn.CrossEntropyLoss()
    loc_loss = nn.L1Loss()
    target_length = 225
    teacher_forcing_ratio = 0.5
    SOS_token = vocab['<SOS>']
    EOS_token = vocab['<EOS>']
    max_epochs = int(os.getenv('EPOCHS', 100))
    best_loss = None
    best_encoder = None
    best_decoder = None
    batch_metrics = True
    max_patience = int(os.getenv('MAX_PATIENCE', 5))
    cur_patience = 0
    criteria = 'sum'
    if criteria == 'sum':
        loss_func = lambda x: sum(x)
    elif criteria == 'mean':
        loss_func = lambda x: sum(x) / len(x)
    else:
        loss_func = lambda x: max(x)
    with open('train_metrics.csv', 'w') as f:
        for edx in range(max_epochs):
            losses = []
            for idx, i_batch in enumerate(dataloader):
                encoder_opt.zero_grad()
                decoder_opt.zero_grad()
                batch_emb_loss = 0
                batch_loc_loss = 0
                feature, label, mask = i_batch['feature'].to(device), i_batch['label'].to(device), i_batch['mask'].to(device)
                emb_label, loc_label = torch.split(label, [1, label.shape[-1] - 1], dim=-1)
                emb_label = torch.squeeze(emb_label, dim=-1)
                enc_out = encoder(feature)
                src = torch.tensor([[SOS_token]])
                use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
                if use_teacher_forcing:
                    for di in range(target_length):
                        tgt = label[:,:di+1,:]
                        msk = mask[:,:di+1]
                        emb_out, loc_out = decoder(tgt, context=enc_out, mask=msk)
                        emb_out, loc_out = emb_out[:,-1:,:], loc_out[:,-1:,:]
                        short_label, short_loc = emb_label[:,di:di+1], loc_label[:,di:di+1,:]
                        batch_emb_loss += emb_loss(emb_out.transpose(1,2), short_label.long())
                        batch_loc_loss += loc_loss(loc_out, short_loc)
                        if short_label[0].item() == EOS_token:
                            break
                else:
                    tgt = label[:,:1,:]
                    for di in range(target_length):
                        msk = mask[:,:di+1]
                        emb_out, loc_out = decoder(tgt, context=enc_out, mask=msk)
                        emb_out, loc_out = emb_out[:,-1:,:], loc_out[:,-1:,:]
                        short_label, short_loc = emb_label[:,di:di+1], loc_label[:,di:di+1,:]
                        batch_emb_loss += emb_loss(emb_out.transpose(1,2), short_label.long())
                        batch_loc_loss += loc_loss(loc_out, short_loc)
                        top_v, top_i = emb_out.topk(1)
                        if short_label[0].item() == EOS_token:
                            break
                        new_layer = torch.cat([top_i, loc_out], dim=-1)[:,-1,:]
                        tgt = torch.cat([tgt,torch.unsqueeze(new_layer, dim=0)], dim=1)
                
                batch_emb_loss.backward(retain_graph=True)
                batch_loc_loss.backward()
                encoder_opt.step()
                decoder_opt.step()
                total_loss = batch_emb_loss.item()+batch_loc_loss.item()
                losses.append(total_loss)
                if batch_metrics:
                    print('\tBatch #{}, Teacher forcing: {}, Loss: {}'.format(idx, use_teacher_forcing, total_loss))
            loss_val = loss_func(losses)
            f.write('{},{}\n'.format(edx, loss_val))
            f.flush()
            print('Epoch #{}, Loss: {}'.format(edx, loss_val))
            if best_loss is None or loss_val < best_loss:
                best_loss = loss_val
                best_encoder = encoder.state_dict()
                best_decoder = decoder.state_dict()
                cur_patience = 0
            else:
                cur_patience += 1
            if cur_patience > max_patience:
                print('Out of patience. Breaking')
                break
    
    encoder = encoder.to('cpu')
    decoder = decoder.to('cpu')
    please_work = torch.from_numpy(io.imread('PleaseWork.png')[:,:,:3].transpose((2, 0, 1)).astype(np.float32)).to('cpu')
    enc_out = encoder(torch.unsqueeze(please_work, dim=0))
    out = torch.tensor([[[SOS_token] + [0] * 12]]).to('cpu')
    mask = torch.ones((1,1), device='cpu').bool()
    for di in range(target_length):
        msk = mask[:,:di+1]
        emb_out, loc_out = decoder(out, context=enc_out, mask=msk)
        emb_out, loc_out = emb_out[:,-1:,:], loc_out[:,-1:,:]
        short_label, short_loc = emb_label[:,di:di+1], loc_label[:,di:di+1,:]
        top_v, top_i = emb_out.topk(1)
        if top_i.squeeze(dim=-1)[0][-1] == EOS_token:
            break
        new_layer = torch.cat([top_i, loc_out], dim=-1)[:,-1,:]
        out = torch.cat([out,torch.unsqueeze(new_layer, dim=0)], dim=1)
    np_out = out.squeeze(dim=0).detach().numpy()
    np.save('testing.npy', np_out)

    #please_work = torch.from_numpy(io.imread('PleaseWork.png')[:,:,:3].transpose((2, 0, 1)).astype(np.float32)).to(device)
    #out = model.generate(please_work, 226, vocab['<EOS>'])
    #out = out.to('cpu')[0].numpy().astype(np.int16)
    #for line in out:
    #    print(line)
    

if __name__ == '__main__':
    main()