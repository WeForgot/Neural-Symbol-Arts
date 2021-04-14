import json
from statistics import mean
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

from model.model import BasicEncoder, AutoregressiveDecoder, pretrain_encoder
from model.datasets import SADataset
from model.utils import get_parameter_count, Vocabulary

load_dotenv()

def freeze_model(model, freeze=True):
   for param in model.parameters():
      param.requires_grad = not freeze

def make_encoder(force_new = False):
    if os.path.exists('encoder_meta.json') and os.path.exists('encoder.pt') and not force_new:
        with open('encoder_meta.json', 'r') as f:
            metadata = json.load(f)
        encoder = BasicEncoder(
            dim = metadata['dim'],
            patch_size = metadata['patch_size'],
            e_depth = metadata['e_depth'],
            e_heads = metadata['e_heads']
        )
        encoder.load_state_dict(torch.load('encoder.pt'))
    else:
        with open('encoder_meta.json', 'w') as f:
            metadata = {
                'dim': int(os.getenv('DIM', 128)),
                'patch_size': int(os.getenv('PATCH_SIZE', 32)),
                'e_depth': int(os.getenv('E_DEPTH', 6)),
                'e_heads': int(os.getenv('E_HEADS', 8))
            }
            json.dump(metadata, f)

        encoder = BasicEncoder(
            dim = metadata['dim'],
            patch_size = metadata['patch_size'],
            e_depth = metadata['e_depth'],
            e_heads = metadata['e_heads']
        )
    return encoder

def make_decoder(vocab, force_new = False):
    if os.path.exists('decoder_meta.json') and os.path.exists('decoder.pt') and not force_new:
        with open('decoder_meta.json', 'r') as f:
            metadata = json.load(f)
        decoder = AutoregressiveDecoder(
            layer_count = metadata['layer_count'],
            emb_dim = metadata['emb_dim'],
            d_depth = metadata['d_depth'],
            d_heads = metadata['d_heads'],
            emb_drop = metadata['emb_drop']
        )
        decoder.load_state_dict(torch.load('decoder.pt'))
    else:
        with open('decoder_meta.json', 'w') as f:
            metadata = {
                'layer_count': len(vocab),
                'emb_dim': int(os.getenv('EMB_DIM', 8)),
                'd_depth': int(os.getenv('D_DEPTH', 6)),
                'd_heads': int(os.getenv('D_HEADS', 8)),
                'emb_drop': float(os.getenv('EMB_DROP', 0.0))
            }
            json.dump(metadata, f)

        decoder = AutoregressiveDecoder(
            layer_count = metadata['layer_count'],
            emb_dim = metadata['emb_dim'],
            d_depth = metadata['d_depth'],
            d_heads = metadata['d_heads'],
            emb_drop = metadata['emb_drop']
        )
    return decoder

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
    encoder = make_encoder(force_new=True).to(device)
    decoder = make_decoder(vocab, force_new=True).to(device)
    print(encoder, flush=True)
    trainable, untrainable = get_parameter_count(encoder)
    print('Total encoder paramters\n\tTrainable:\t{}\n\tUntrainable:\t{}'.format(trainable, untrainable), flush=True)
    print(decoder, flush=True)
    trainable, untrainable = get_parameter_count(decoder)
    print('Total decoder paramters\n\tTrainable:\t{}\n\tUntrainable:\t{}'.format(trainable, untrainable), flush=True)
    
    dataset = SADataset(data)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)
    should_pretrain = True if os.getenv('PRETRAIN_ENCODER', 'False').lower() == 'true' else False

    if should_pretrain:
        encoder = pretrain_encoder(encoder, dataloader, device)

    optimizer = os.getenv('OPTIMIZER', 'sgd')
    if optimizer.lower() == 'adam':
        encoder_opt = optim.Adam(encoder.parameters(), lr=1e-3)
        decoder_opt = optim.Adam(decoder.parameters(), lr=1e-3)
    elif optimizer.lower() == 'adamw':
        encoder_opt = optim.AdamW(encoder.parameters(), lr=3e-4)
        decoder_opt = optim.AdamW(decoder.parameters(), lr=3e-4)
    elif optimizer.lower() == 'asgd':
        encoder_opt = optim.ASGD(encoder.parameters(), lr=1e-3)
        decoder_opt = optim.ASGD(decoder.parameters(), lr=1e-3)
    elif optimizer.lower() == 'rmsprop':
        encoder_opt = optim.RMSprop(encoder.parameters(), lr=1e-3)
        decoder_opt = optim.RMSprop(decoder.parameters(), lr=1e-3)
    elif optimizer.lower() == 'adabelief':
        from adabelief_pytorch.AdaBelief import AdaBelief
        encoder_opt = AdaBelief(encoder.parameters(), lr=5e-4, weight_decay=1e-4, print_change_log=False)
        decoder_opt = AdaBelief(decoder.parameters(), lr=5e-4, weight_decay=1e-4, print_change_log=False)
    else:
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
    use_scaled_loss = True
    use_min_loss = False
    alpha = 0.99
    alpha_decay = 0.001
    encoder_warmup = 20
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
        if should_pretrain and should_pretrainencoder_warmup > 0:
           print('Freezing encoder for {} generations'.format(encoder_warmup))
           freeze_model(encoder, freeze = True)
        for edx in range(max_epochs):
            losses = []
            if should_pretrain and edx == encoder_warmup:
               print('Unfreezing encoder')
               freeze_model(encoder, freeze = False)
            for bdx, i_batch in enumerate(dataloader):
                feature, label, mask = i_batch['feature'].to(device), i_batch['label'].to(device), i_batch['mask'].to(device)

                encoder_opt.zero_grad()
                decoder_opt.zero_grad()

                enc = encoder(feature)
                emb_loss, color_loss, pos_loss = decoder(label,mask, context=enc, return_both_loss=True)

                scalar_emb_loss = emb_loss.item()
                scalar_color_loss = color_loss.item()
                scalar_position_loss = pos_loss.item()
                scaled_loss = min(scalar_emb_loss, scalar_color_loss, scalar_position_loss) if use_min_loss else mean([scalar_emb_loss, scalar_color_loss, scalar_position_loss])

                if use_scaled_loss:
                    emb_loss = scaled_loss * (emb_loss / scalar_emb_loss)
                    color_loss = scaled_loss * (color_loss / scalar_color_loss)
                    pos_loss = scaled_loss * (pos_loss / scalar_position_loss)
                total_loss = emb_loss + color_loss + pos_loss
                total_loss.backward()

                encoder_opt.step()
                decoder_opt.step()

                print('Batch #{}, Embedding Loss: {}, Color Loss: {}, Position Loss: {}, Balanced Loss: {}'.format(bdx, scalar_emb_loss, scalar_color_loss, scalar_position_loss, scaled_loss), flush=True)
                losses.append(total_loss.item())
                dataloader.dataset.new_rand()
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
                feature = io.imread('PleaseWork.png')[:,:,:3].astype(np.float32) / 255.
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
