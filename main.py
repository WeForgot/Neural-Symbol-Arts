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
            'd_heads': int(os.getenv('D_HEADS', 8))
        }
        json.dump(info, f)
    print(encoder)
    trainable, untrainable = get_parameter_count(encoder)
    print('Total encoder paramters\n\tTrainable:\t{}\n\tUntrainable:\t{}'.format(trainable, untrainable))
    print(decoder)
    trainable, untrainable = get_parameter_count(decoder)
    print('Total decoder paramters\n\tTrainable:\t{}\n\tUntrainable:\t{}'.format(trainable, untrainable))
    
    
    dataset = SADataset(data)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    #encoder_opt = optim.Adam(encoder.parameters(), lr=3e-4)
    #decoder_opt = optim.Adam(decoder.parameters(), lr=3e-4)
    encoder_opt = optim.SGD(encoder.parameters(), lr=1e-3, momentum=0.9)
    decoder_opt = optim.SGD(decoder.parameters(), lr=1e-3, momentum=0.9)
    target_length = int(os.getenv('TARGET_LENGTH', 225))
    teacher_forcing_ratio = float(os.getenv('TEACHER_RATIO', 0.9))
    teacher_forcing_decay = float(os.getenv('TEACHER_DECAY', 0.99))
    teacher_forcing_min = float(os.getenv('TEACHER_MIN', 0.3))
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
                batch_emb_loss.backward(retain_graph=True)
                batch_met_loss.backward(retain_graph=False)
                encoder_opt.step()
                decoder_opt.step()
                print('Batch #{}, Embedding Loss: {}, Metric Loss: {}'.format(bdx, batch_emb_loss, batch_met_loss))
                losses.append(batch_emb_loss.item()+batch_met_loss.item())
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
            
            if edx % 20 == 0:
                torch.save(best_encoder, 'encoder.pt')
                torch.save(best_decoder, 'decoder.pt')
            
            if cur_patience > max_patience:
                print('Out of patience. Breaking')
                break
    feature = io.imread('PleaseWork.png')[:,:,:3].astype(np.float32)
    feature = torch.from_numpy(feature.transpose((2, 0, 1))).to(device)
    enc = encoder(feature.unsqueeze(0))
    generated = np.asarray(decoder.generate(enc, vocab, 225))
    print(generated)
    print(generated.shape)

if __name__ == '__main__':
    main()