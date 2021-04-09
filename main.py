import json
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
from x_transformers import Decoder
from nystrom_attention import Nystromformer

from model.model import BasicEncoder, AutoregressiveDecoder
from model.datasets import SADataset
from model.utils import get_parameter_count, Vocabulary

load_dotenv()

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA available', flush=True)
    else:
        device = torch.device('cpu')
        print('CUDA not available', flush=True)
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
    decoder = AutoregressiveDecoder(
        layer_count = len(vocab),
        emb_dim = int(os.getenv('EMB_DIM', 8)),
        dim= int(os.getenv('DIM', 128)),
        d_depth = int(os.getenv('D_DEPTH', 6)),
        d_heads = int(os.getenv('D_HEADS', 8)),
    ).to(device)
    with open('meta.json', 'w') as f:
        info = {
            'dim': int(os.getenv('DIM', 128)),
            'patch_size': int(os.getenv('PATCH_SIZE', 32)),
            'e_depth': int(os.getenv('E_DEPTH', 6)),
            'e_heads': int(os.getenv('E_HEADS', 8)),
            'layer_count': len(vocab),
            'emb_dim': int(os.getenv('EMB_DIM', 8)),
            'd_depth': int(os.getenv('D_DEPTH', 6)),
            'd_heads': int(os.getenv('D_HEADS', 8)),
            'emb_drop': float(os.getenv('EMB_DROP', 0.0))
        }
        json.dump(info, f)
    print(encoder, flush=True)
    trainable, untrainable = get_parameter_count(encoder)
    print('Total encoder paramters\n\tTrainable:\t{}\n\tUntrainable:\t{}'.format(trainable, untrainable), flush=True)
    print(decoder, flush=True)
    trainable, untrainable = get_parameter_count(decoder)
    print('Total decoder paramters\n\tTrainable:\t{}\n\tUntrainable:\t{}'.format(trainable, untrainable), flush=True)
    
    
    dataset = SADataset(data)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    #encoder_opt = optim.Adam(encoder.parameters(), lr=3e-4)
    #decoder_opt = optim.Adam(decoder.parameters(), lr=3e-4)
    encoder_opt = optim.SGD(encoder.parameters(), lr=1e-3)
    decoder_opt = optim.SGD(decoder.parameters(), lr=1e-3)
    target_length = int(os.getenv('TARGET_LENGTH', 225))
    eval_every = int(os.getenv('EVAL_EVERY', 20))
    SOS_token = vocab['<SOS>']
    EOS_token = vocab['<EOS>']
    max_epochs = int(os.getenv('EPOCHS', 100))
    best_loss = None
    best_encoder = None
    best_decoder = None
    batch_metrics = True
    use_blended_loss = False
    use_switch_loss = False
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
            for bdx, i_batch in enumerate(dataloader):
                feature, label, mask = i_batch['feature'].to(device), i_batch['label'].to(device), i_batch['mask'].to(device)
                batch_emb_loss = 0
                batch_met_loss = 0
                enc = encoder(feature)
                encoder_opt.zero_grad()
                decoder_opt.zero_grad()
                for idx in range(2, len(label[0])):
                    emb_loss, met_loss = decoder(label[:,:idx],mask[:,:idx], return_both_loss=True)
                    batch_emb_loss += emb_loss
                    batch_met_loss += met_loss
                if use_blended_loss:
                    alpha = batch_emb_loss.item() / (batch_emb_loss.item() + batch_met_loss.item())
                    total_loss = alpha * batch_emb_loss + (1 - alpha) * batch_met_loss
                    total_loss.backward()
                elif use_switch_loss:
                    if random.random() > 0.5:
                        batch_emb_loss.backward()
                    else:
                        batch_met_loss.backward()
                else:
                    total_loss = batch_emb_loss + batch_met_loss
                    total_loss.backward()
                encoder_opt.step()
                decoder_opt.step()
                print('Batch #{}, Embedding Loss: {}, Metric Loss: {}'.format(bdx, batch_emb_loss, batch_met_loss), flush=True)
                losses.append(batch_emb_loss.item()+batch_met_loss.item())
            loss_val = loss_func(losses)
            f.write('{},{}\n'.format(edx, loss_val))
            f.flush()
            print('Epoch #{}, Loss: {}'.format(edx, loss_val), flush=True)
            if best_loss is None or loss_val < best_loss:
                best_loss = loss_val
                best_encoder = encoder.state_dict()
                best_decoder = decoder.state_dict()
                cur_patience = 0
            else:
                cur_patience += 1
            
            if edx % eval_every == 0:
                torch.save(best_encoder, 'encoder.pt')
                torch.save(best_decoder, 'decoder.pt')
                encoder.eval()
                decoder.eval()
                feature = io.imread('PleaseWork.png')[:,:,:3].astype(np.float32)
                feature = torch.from_numpy(feature.transpose((2, 0, 1))).to(device)
                enc = encoder(feature.unsqueeze(0))
                generated = np.asarray(decoder.generate(enc, vocab, 225))
                np.save('test.npy', generated)
                encoder.train()
                decoder.train()
            
            if cur_patience > max_patience:
                print('Out of patience. Breaking', flush=True)
                break
    feature = io.imread('PleaseWork.png')[:,:,:3].astype(np.float32)
    feature = torch.from_numpy(feature.transpose((2, 0, 1))).to(device)
    enc = encoder(feature.unsqueeze(0))
    generated = np.asarray(decoder.generate(enc, vocab, 225))
    np.save('test.npy', generated)
    print(generated, flush=True)
    print(generated.shape, flush=True)

if __name__ == '__main__':
    main()