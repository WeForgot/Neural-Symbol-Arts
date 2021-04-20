from argparse import ArgumentParser
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
from model.style_model import StyleViT

from model.model import AutoregressiveDecoder, pretrain_encoder
from model.datasets import SADataset
from model.utils import get_parameter_count, Vocabulary, convert_numpy_to_saml, str2bool

load_dotenv()

def env_bool(variable):
    return True if os.getenv(variable, 'true').lower() == 'true' else False

def freeze_model(model, freeze=True):
   for param in model.parameters():
      param.requires_grad = not freeze

def make_encoder(args):
    with open('encoder_meta.json', 'w') as f:
        metadata = {
            'dim': args.dim,
            'patch_size': args.patch_size,
            'e_depth': args.e_depth,
            'e_heads': args.e_heads,
            'encoder_type': args.e_type,
            'mlp_dim': args.mlp_dim,
            'style_latents': args.style_latents
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
    elif encoder_type == 'style':
        encoder = StyleViT(
            image_size = 576,
            patch_size = metadata['patch_size'],
            dim = metadata['dim'],
            depth = metadata['e_depth'],
            heads = metadata['e_heads'],
            mlp_dim = metadata['mlp_dim'],
            num_latents = metadata['style_latents']
        )
    else:
        raise ValueError('Please choose an appropriate encoder type from [vit, t2t, levit, cvt]')
    return encoder

def make_decoder(vocab, args):
    with open('decoder_meta.json', 'w') as f:
        metadata = {
            'layer_count': len(vocab),
            'dim': args.dim,
            'emb_dim': args.emb_dim,
            'd_depth': args.d_depth,
            'd_heads': args.d_heads,
            'emb_drop': args.emb_drop,
            'decoder_type': args.d_type
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

def main(args):
    max_epochs = args.epochs
    max_patience = args.patience
    optimizer = args.optimizer
    batch_size = args.batch_size
    batch_metrics = args.batch_metrics
    use_activations = args.activations
    use_scaled_loss = args.scaled_loss
    valid_split = args.valid_split
    eval_every = args.eval_every

    layer_alpha = args.layer_alpha
    color_alpha = args.color_alpha
    position_alpha = args.position_alpha

    target_length = 225
    data_clamped = True

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
    encoder = make_encoder(args).to(device)
    decoder = make_decoder(vocab, args).to(device)
    print(encoder, flush=True)
    trainable, untrainable = get_parameter_count(encoder)
    print('Total encoder paramters\n\tTrainable:\t{}\n\tUntrainable:\t{}'.format(trainable, untrainable), flush=True)
    print(decoder, flush=True)
    trainable, untrainable = get_parameter_count(decoder)
    print('Total decoder paramters\n\tTrainable:\t{}\n\tUntrainable:\t{}'.format(trainable, untrainable), flush=True)
    
    dataset = SADataset(data)
    valid_size = int(len(dataset) * valid_split)
    train_size = len(dataset) - valid_size
    train_set, valid_set = torch.utils.data.random_split(SADataset(data), [train_size, valid_size])
    train_loader, valid_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True), DataLoader(valid_set)
    
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
    
    SOS_token = vocab['<SOS>']
    EOS_token = vocab['<EOS>']
    best_loss = None
    best_encoder = None
    best_decoder = None
    cur_patience = 0

    with open('train_metrics.csv', 'w') as f, open('valid_metrics.csv', 'w') as v:
        for edx in range(max_epochs):
            total_losses = 0
            emb_losses = 0
            color_losses = 0
            position_losses = 0
            encoder.train()
            decoder.train()
            for bdx, i_batch in enumerate(train_loader):
                feature, label, mask = i_batch['feature'].to(device), i_batch['label'].to(device), i_batch['mask'].to(device)

                encoder_opt.zero_grad()
                decoder_opt.zero_grad()

                enc = encoder(feature)
                emb_loss, color_loss, pos_loss, aux_loss = decoder(label,mask, context=enc, return_both_loss=True, loss_func=nn.functional.mse_loss, use_activations=use_activations)

                scalar_emb_loss = emb_loss.item()
                scalar_color_loss = color_loss.item()
                scalar_position_loss = pos_loss.item()
                scaled_loss = min([scalar_emb_loss, scalar_color_loss, scalar_position_loss])

                if use_scaled_loss:
                    emb_loss = scaled_loss * (emb_loss / scalar_emb_loss)
                    color_loss = scaled_loss * (color_loss / scalar_color_loss)
                    pos_loss = scaled_loss * (pos_loss / scalar_position_loss)
                total_loss = layer_alpha * emb_loss + color_alpha*  color_loss + position_alpha * pos_loss + (aux_loss if aux_loss is not None else 0)
                total_losses += total_loss.item()
                emb_losses += emb_loss.item() * layer_alpha
                color_losses += color_loss.item() * color_alpha
                position_losses += pos_loss.item() * position_alpha
                if isnan(total_losses):
                    print('Batch loss is NaN. Breaking')
                    break
                total_loss.backward()

                encoder_opt.step()
                decoder_opt.step()

                if batch_metrics:
                    print('Batch #{}, Embedding Loss: {}, Color Loss: {}, Position Loss: {}, Balanced Loss: {}'.format(bdx, scalar_emb_loss, scalar_color_loss, scalar_position_loss, scaled_loss), flush=True)
                train_loader.dataset.dataset.new_rand()
            if isnan(total_losses):
                print('Loss is NaN. Returning', flush = True)
                return
            f.write('{},{},{},{},{}\n'.format(edx, total_losses, emb_losses, color_losses, position_losses))
            f.flush()
            print('TRAINING Epoch #{}, Total Loss: {}, Embedding Loss: {}, Color Loss: {}, Position Loss: {}'.format(edx, total_losses, emb_losses, color_losses, position_losses), flush=True)
            encoder.eval()
            decoder.eval()

            valid_emb_loss = 0
            valid_color_loss = 0
            valid_position_loss = 0
            for bdx, i_batch in enumerate(valid_loader):
                feature, label, mask = i_batch['feature'].to(device), i_batch['label'].to(device), i_batch['mask'].to(device)
                enc = encoder(feature)
                emb_loss, color_loss, pos_loss, aux_loss = decoder(label,mask, context=enc, return_both_loss=True, loss_func=nn.functional.mse_loss, use_activations=use_activations)
                valid_emb_loss += emb_loss.item()
                valid_color_loss += color_loss.item()
                valid_position_loss += pos_loss.item()

            total_loss = valid_emb_loss + valid_color_loss + valid_position_loss
            print('VALIDATION Epoch #{}, Total Loss: {}, Embedding Loss: {}, Color Loss: {}, Position Loss: {}'.format(edx, total_loss, valid_emb_loss, valid_color_loss, valid_position_loss), flush=True)
            v.write('{},{},{},{},{}\n'.format(edx, total_loss, valid_emb_loss, valid_color_loss, valid_position_loss))
            print('------------------------------------------------------------------------------------------------')

            if best_loss is None or total_loss < best_loss:
                best_loss = total_loss
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
                generated = np.asarray(decoder.generate(enc, vocab, 225, use_activations=use_activations))
                dest_name = 'test_{}'.format(edx)
                np.save('test.npy', generated)
                convert_numpy_to_saml('test.npy', vocab, dest_path=dest_name+'.saml', name=dest_name, values_clamped=data_clamped)
                encoder.train()
                decoder.train()
            
            if cur_patience > max_patience:
                print('Out of patience. Breaking', flush=True)
                break
            
    encoder.load_state_dict(best_encoder)
    decoder.load_state_dict(best_decoder)
    feature = io.imread('PleaseWork.png')[:,:,:3].astype(np.float32) / 255.
    feature = torch.from_numpy(feature.transpose((2, 0, 1))).to(device)
    enc = encoder(feature.unsqueeze(0))
    generated = np.asarray(decoder.generate(enc, vocab, 225, use_activations=use_activations))
    np.save('test.npy', generated)
    convert_numpy_to_saml('test.npy', vocab, values_clamped=data_clamped)
    print(generated, flush=True)
    print(generated.shape, flush=True)

parser = ArgumentParser()
parser.add_argument('--epochs', default=100, type=int, help='Maximum number of epochs to train')
parser.add_argument('--patience', default=20, type=int, help='Maximum patience while training (set to equal --epochs if you want no patience)')
parser.add_argument('--optimizer', default='adam', type=str, help='Which optimizer to use, defaults to Adam')
parser.add_argument('--batch_size', default=4, type=int, help='What batch size to use')
parser.add_argument('--batch_metrics', default=False, type=str2bool, help='Whether or not to print metrics per batch')
parser.add_argument('--activations', default=False, type=str2bool, help='Whether to use sigmoid and tanh activations for color and position respectively. Otherwise defaults to linear')
parser.add_argument('--valid_split', default=0.1, type=float, help='What percent of the dataset should be used for validation')
parser.add_argument('--scaled_loss', default=False, type=str2bool, help='Whether to scale loss by the L1 norm')
parser.add_argument('--eval_every', default=10, type=int, help='How often (in epochs) to evaluate the model on a test image')
parser.add_argument('--dim', default=32, type=int, help='What the inner dimension of the transformer should be')
parser.add_argument('--mlp_dim', default=32, type=int, help='What the feed forward MLP dimension should be')
parser.add_argument('--emb_dim', default=4, type=int, help='Size of the embedding dimension for layers')
parser.add_argument('--emb_drop', default=0.1, type=float, help='How likely dropout occurs on layer embeddings')
parser.add_argument('--e_depth', default=1, type=int, help='How many layers should be in the encoder')
parser.add_argument('--e_heads', default=8, type=int, help='How many heads should be in the encoder')
parser.add_argument('--d_depth', default=1, type=int, help='How many layers should be in the decoder')
parser.add_argument('--d_heads', default=8, type=int, help='How many heads should be in the decoder')
parser.add_argument('--patch_size', default=32, type=int, help='How large each patch should be in the encoder. Does not apply to the CvT encoder')
parser.add_argument('--style_latents', default=1, type=int, help='Number of latent layers to use for StyleViT')
parser.add_argument('--e_type', default='vanilla', type=str, help='Which encoder to use. Valid values are: vanilla, nystrom, cvt, style')
parser.add_argument('--d_type', default='vanilla', type=str, help='Which decoder to use. Valid values are: vanilla, routing')

parser.add_argument('--layer_alpha', default=1.0, type=float, help='The scaling factor for the layer prediction loss')
parser.add_argument('--color_alpha', default=1.0, type=float, help='The scaling factor for the color prediction loss')
parser.add_argument('--position_alpha', default=1.0, type=float, help='The scaling factor for the position prediction loss')

args = parser.parse_args()

if __name__ == '__main__':
    main(args)
