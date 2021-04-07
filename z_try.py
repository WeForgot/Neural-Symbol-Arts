import os
import pickle

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

class BasicNSA(nn.Module):
    def __init__(self, layer_count, emb_dim, patch_size = 32, dim = 16, e_depth = 6, e_heads = 8, d_depth = 12, d_heads=8):
        super(BasicNSA, self).__init__()
        self.layer_count = layer_count
        self.emb_dim = emb_dim
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


        self.embedding_dim = nn.Embedding(layer_count, emb_dim)

        self.decoder = Decoder(
            dim = emb_dim+12,
            depth = d_depth,
            heads = d_heads
        )
        self.to_logits = nn.Linear(emb_dim+12, layer_count+12)
    
    def forward(self, src, tgt=None, mask=None):
        emb_idx, metrics = torch.split(tgt, [1, tgt.shape[-1] - 1], dim=-1)
        embs = self.embedding_dim(emb_idx.int()).squeeze(dim=2)
        y = torch.cat([embs, metrics], dim=-1)
        x = self.encoder(src)
        x = self.decoder(y, context=x, mask=mask)
        x = self.to_logits(x)
        x = torch.split(x, [self.layer_count, 12], dim=-1)
        return x
    
    @torch.no_grad()
    def generate(self, src, max_seq_len, eos_token):
        self.eval()
        src = torch.unsqueeze(src, 0)
        enc = self.encoder(src)
        out = torch.zeros((1,max_seq_len,13), device=src.device)
        mask = torch.zeros((1,max_seq_len,), dtype=torch.bool, device=out.device)
        idx = 1
        mask[0][0] = True
        while idx < max_seq_len:
            mask[0][idx] = True
            pred_emb_logits, pred_l1_logits = self(src, tgt=out, mask=mask)
            pred_emb_logits, pred_l1_logits = pred_emb_logits[:,-1,:], pred_l1_logits[:,-1,:]
            print(pred_l1_logits)
            filtered_logits = top_k(pred_emb_logits, thres=0.9)
            probs = nn.functional.softmax(filtered_logits / 1.0, dim=-1)
            sample = torch.multinomial(probs, 1)
            out[0,idx,0] = sample[0][0]
            sample = torch.cat([sample, pred_l1_logits], dim=-1)
            out[0,idx,:] = sample
            if out[0,idx,0] == eos_token:
                break
            idx += 1
        return out

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
    layer_count = 388
    emb_dim = 8
    model = BasicNSA(layer_count, 
        emb_dim = int(os.getenv('EMB_DIM', 8)),
        dim= int(os.getenv('DIM', 128)),
        patch_size= int(os.getenv('PATCH_SIZE', 32)),
        e_depth = int(os.getenv('E_DEPTH', 6)),
        e_heads = int(os.getenv('E_HEADS', 8)),
        d_depth = int(os.getenv('D_DEPTH', 6)),
        d_heads = int(os.getenv('D_HEADS', 8)),
    ).to(device)
    print(model)
    trainable, untrainable = get_parameter_count(model)
    print('Total paramters\n\tTrainable:\t{}\n\tUntrainable:\t{}'.format(trainable, untrainable))
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    dataset = SADataset(data)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    opt = optim.AdamW(model.parameters(), lr=3e-4)
    max_epochs = int(os.getenv('EPOCHS', 100))
    best_loss = None
    best_model = None
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
                feature, label, mask = i_batch['feature'], i_batch['label'], i_batch['mask']
                preds = model(feature.to(device), label.to(device), mask.to(device))
                cross_loss, loc_loss = CrossL1Loss(*preds, label.to(device), emb_dim=layer_count)
                loss = cross_loss + loc_loss
                if np.isnan(loss.item()):
                    print('\tNaN loss. Breaking')
                    return
                losses.append(loss.item())
                opt.zero_grad()
                cross_loss.backward(retain_graph=True)
                loc_loss.backward()
                opt.step()
                if batch_metrics:
                    print('\tBatch #{}, Loss: {}'.format(idx, loss.item()))
            loss_val = loss_func(losses)
            f.write('{},{}\n'.format(edx, loss_val))
            f.flush()
            print('Epoch #{}, Loss: {}'.format(edx, loss_val))
            if best_loss is None or loss_val < best_loss:
                best_loss = loss_val
                best_model = model.state_dict()
                cur_patience = 0
            else:
                cur_patience += 1
            if cur_patience > max_patience:
                print('Out of patience. Breaking')
                break

    please_work = torch.from_numpy(io.imread('PleaseWork.png')[:,:,:3].transpose((2, 0, 1)).astype(np.float32)).to(device)
    out = model.generate(please_work, 226, vocab['<EOS>'])
    out = out.to('cpu')[0].numpy().astype(np.int16)
    #for line in out:
    #    print(line)
    

if __name__ == '__main__':
    main()