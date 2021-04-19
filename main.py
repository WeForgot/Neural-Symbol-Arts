import json
from math import isnan
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from vit_pytorch.vit import ViT
from vit_pytorch.efficient import ViT as EfficientViT
from vit_pytorch.t2t import T2TViT
from vit_pytorch.levit import LeViT
from vit_pytorch.cvt import CvT
from vit_pytorch.rvt import RvT
from nystrom_attention import Nystromformer
from routing_transformer import RoutingTransformer

from model.model import AutoregressiveDecoder, pretrain_encoder
from model.datasets import SADataset
from model.utils import get_parameter_count, Vocabulary, convert_numpy_to_saml

load_dotenv()

def env_bool(variable):
    return True if os.getenv(variable, 'true').lower() == 'true' else False

def freeze_model(model, freeze=True):
   for param in model.parameters():
      param.requires_grad = not freeze

def make_encoder(force_new = False):
    if os.path.exists('encoder_meta.json') and not force_new:
        with open('encoder_meta.json', 'r') as f:
            metadata = json.load(f)
    else:
        with open('encoder_meta.json', 'w') as f:
            metadata = {
                'dim': int(os.getenv('DIM', 128)),
                'patch_size': int(os.getenv('PATCH_SIZE', 32)),
                'e_depth': int(os.getenv('E_DEPTH', 6)),
                'e_heads': int(os.getenv('E_HEADS', 8)),
                'encoder_type': os.getenv('E_TYPE', 'vit'),
                'mlp_dim': int(os.getenv('MLP_DIM', 128))
            }
            json.dump(metadata, f)
    encoder_type = metadata['encoder_type']
    if encoder_type == 'vit':
        encoder = ViT(
            image_size = 576,
            patch_size = metadata['patch_size'],
            num_classes = 1,
            dim = metadata['dim'],
            depth = metadata['e_depth'],
            heads = metadata['e_heads'],
            mlp_dim = metadata['mlp_dim']
        )
        encoder.mlp_head = nn.Identity()
    elif encoder_type == 't2t':
        encoder = T2TViT(
            image_size = 576,
            num_classes = 1,
            dim = metadata['dim'],
            depth = metadata['e_depth'],
            heads = metadata['e_heads'],
            mlp_dim = metadata['mlp_dim']
        )
        encoder.mlp_head = nn.Identity()
    elif encoder_type == 'levit':
        encoder = LeViT(
            image_size = 576,
            num_classes = 1,
            dim = metadata['dim'],
            depth = metadata['e_depth'],
            heads = metadata['e_heads'],
            mlp_mult = 2 # It is used as a default in the transformer but is a required arg in LeViT I guess?
        )
        encoder.mlp_head = nn.Identity()
    elif encoder_type == 'cvt':
        encoder = CvT(
            num_classes = 1
        )
        encoder.layers[-1] = nn.Identity()
    elif encoder_type == 'nystrom':
        encoder = EfficientViT(
            image_size = 576,
            patch_size = metadata['patch_size'],
            num_classes = 1,
            dim = metadata['dim'],
            transformer = Nystromformer(
                dim = metadata['dim'],
                depth = metadata['e_depth'],
                heads = metadata['e_heads']
            )
        )
        encoder.mlp_head = nn.Identity()
    elif encoder_type == 'rotary':
        encoder = RvT(
            image_size = 576,
            patch_size = metadata['patch_size'],
            num_classes = 1,
            dim = metadata['dim'],
            depth = metadata['e_depth'],
            heads = metadata['e_heads'],
            mlp_dim = metadata['mlp_dim']
        )
        encoder.mlp_head = nn.Identity()
    elif encoder_type == 'routing':
        encoder = EfficientViT(
            image_size = 576,
            patch_size = metadata['patch_size'],
            num_classes = 1,
            dim = metadata['dim'],
            transformer = RoutingTransformer(
                dim = metadata['dim'],
                depth = metadata['e_depth'],
                max_seq_len = 256,
                heads = metadata['e_heads']
            )
        )
    else:
        raise ValueError('Please choose an appropriate encoder type from [vit, t2t, levit, cvt]')
    return encoder

def make_decoder(vocab, force_new = False):
    if os.path.exists('decoder_meta.json') and os.path.exists('decoder.pt') and not force_new:
        with open('decoder_meta.json', 'r') as f:
            metadata = json.load(f)
        decoder = AutoregressiveDecoder(
            d_dim = metadata['dim'],
            layer_count = metadata['layer_count'],
            emb_dim = metadata['emb_dim'],
            d_depth = metadata['d_depth'],
            d_heads = metadata['d_heads'],
            emb_drop = metadata['emb_drop'],
            decoder_type = metadata['decoder_type']
        )
        decoder.load_state_dict(torch.load('decoder.pt'))
    else:
        with open('decoder_meta.json', 'w') as f:
            metadata = {
                'layer_count': len(vocab),
                'dim': int(os.getenv('DIM', 32)),
                'emb_dim': int(os.getenv('EMB_DIM', 8)),
                'd_depth': int(os.getenv('D_DEPTH', 6)),
                'd_heads': int(os.getenv('D_HEADS', 8)),
                'emb_drop': float(os.getenv('EMB_DROP', 0.0)),
                'decoder_type': os.getenv('D_TYPE', '')
            }
            json.dump(metadata, f)

        decoder = AutoregressiveDecoder(
            layer_count = metadata['layer_count'],
            emb_dim = metadata['emb_dim'],
            d_depth = metadata['d_depth'],
            d_heads = metadata['d_heads'],
            d_dim = metadata['dim'],
            emb_drop = metadata['emb_drop'],
            decoder_type = metadata['decoder_type']
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
    batch_size = int(os.getenv('BATCH_SIZE', 2))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
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
        encoder_opt = AdaBelief(encoder.parameters(), lr=1e-3, weight_decay=1e-4, print_change_log=False)
        decoder_opt = AdaBelief(decoder.parameters(), lr=1e-3, weight_decay=1e-4, print_change_log=False)
    else:
        encoder_opt = optim.SGD(encoder.parameters(), lr=1e-2, momentum=0.1)
        decoder_opt = optim.SGD(decoder.parameters(), lr=1e-2, momentum=0.1)
    
    
    target_length = int(os.getenv('TARGET_LENGTH', 225))
    eval_every = int(os.getenv('EVAL_EVERY', 20))
    SOS_token = vocab['<SOS>']
    EOS_token = vocab['<EOS>']
    max_epochs = int(os.getenv('EPOCHS', 100))
    best_loss = None
    best_encoder = None
    best_decoder = None
    batch_metrics = True if os.getenv('BATCH_METRICS', 'true').lower() == 'true' else False
    use_scaled_loss = False
    use_min_loss = False
    use_activations = env_bool('USE_ACTIVATIONS')
    enable_pretraining = True
    data_clamped = env_bool('CLAMP_DATA')
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
    if enable_pretraining:
        encoder = pretrain_encoder(encoder, dataloader, device, max_patience=10)
    with open('train_metrics.csv', 'w') as f:
        for edx in range(max_epochs):
            total_losses = []
            emb_losses = []
            color_losses = []
            position_losses = []
            for bdx, i_batch in enumerate(dataloader):
                feature, label, mask = i_batch['feature'].to(device), i_batch['label'].to(device), i_batch['mask'].to(device)

                encoder_opt.zero_grad()
                decoder_opt.zero_grad()

                enc = encoder(feature)
                emb_loss, color_loss, pos_loss, aux_loss = decoder(label,mask, context=enc, return_both_loss=True, loss_func=nn.functional.mse_loss, use_activations=use_activations)

                scalar_emb_loss = emb_loss.item()
                scalar_color_loss = color_loss.item()
                scalar_position_loss = pos_loss.item()
                scaled_loss = min(scalar_emb_loss, scalar_color_loss, scalar_position_loss) if use_min_loss else mean([scalar_emb_loss, scalar_color_loss, scalar_position_loss])

                if use_scaled_loss:
                    emb_loss = scaled_loss * (emb_loss / scalar_emb_loss)
                    color_loss = scaled_loss * (color_loss / scalar_color_loss)
                    pos_loss = scaled_loss * (pos_loss / scalar_position_loss)
                total_loss = emb_loss + color_loss + pos_loss + (aux_loss if aux_loss is not None else 0)
                total_losses.append(total_loss.item())
                emb_losses.append(emb_loss.item())
                color_losses.append(color_loss.item())
                position_losses.append(pos_loss.item())
                if isnan(total_losses[-1]):
                    print('Batch loss is NaN. Breaking')
                    break
                total_loss.backward()

                encoder_opt.step()
                decoder_opt.step()

                if batch_metrics:
                    print('Batch #{}, Embedding Loss: {}, Color Loss: {}, Position Loss: {}, Balanced Loss: {}'.format(bdx, scalar_emb_loss, scalar_color_loss, scalar_position_loss, scaled_loss), flush=True)
                dataloader.dataset.new_rand()

            total_val = loss_func(total_losses)
            emb_val = loss_func(emb_losses)
            color_val = loss_func(color_losses)
            position_val = loss_func(position_losses)
            if isnan(total_val):
                print('Loss is NaN. Returning', flush = True)
                return
            f.write('{},{},{},{},{}\n'.format(edx, total_val, emb_val, color_val, position_val))
            f.flush()
            print('Epoch #{}, Total Loss: {}, Embedding Loss: {}, Color Loss: {}, Position Loss: {}'.format(edx, total_val, emb_val, color_val, position_val), flush=True)
            if best_loss is None or total_val < best_loss:
                best_loss = total_val
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
                feature = io.imread(os.path.join('.','data','BetterSymbolArts','processed','ジェネＣ＠プラウ.png'))[:,:,:3].astype(np.float32) / 255.
                #feature = io.imread('EasyTest.png')[:,:,:3].astype(np.float32) / 255.
                feature = torch.from_numpy(feature.transpose((2, 0, 1))).to(device)
                enc = encoder(feature.unsqueeze(0))
                generated = np.asarray(decoder.generate(enc, vocab, 225, use_activations=use_activations))
                np.save('test.npy', generated)
                convert_numpy_to_saml('test.npy', vocab, values_clamped=data_clamped)
                encoder.train()
                decoder.train()
            
            if cur_patience > max_patience:
                print('Out of patience. Breaking', flush=True)
                break
            
    encoder.load_state_dict(best_encoder)
    decoder.load_state_dict(best_decoder)
    feature = io.imread(os.path.join('.','data','BetterSymbolArts','processed','ジェネＣ＠プラウ.png'))[:,:,:3].astype(np.float32) / 255.
    #feature = io.imread('EasyTest.png')[:,:,:3].astype(np.float32) / 255.
    feature = torch.from_numpy(feature.transpose((2, 0, 1))).to(device)
    enc = encoder(feature.unsqueeze(0))
    generated = np.asarray(decoder.generate(enc, vocab, 225, use_activations=use_activations))
    np.save('test.npy', generated)
    convert_numpy_to_saml('test.npy', vocab, values_clamped=data_clamped)
    print(generated, flush=True)
    print(generated.shape, flush=True)

if __name__ == '__main__':
    main()
