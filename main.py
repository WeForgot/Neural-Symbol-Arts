from argparse import ArgumentParser
from contextlib import nullcontext
import json
from math import isnan
from statistics import mean
from glob import glob
import os
from pathlib import Path
import pickle
import random
import re
import time

from adabelief_pytorch import AdaBelief
from ranger_adabelief import RangerAdaBelief
from dotenv import load_dotenv
import numpy as np
from skimage import io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd

from model.model import EndToEndModel
from model.datasets import SADataset
from model.utils import get_parameter_count, Vocabulary, convert_numpy_to_saml, str2bool, load_data, linear_decay, piecewise_decay

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
    thicc_ff = args.thicc_ff
    name = args.name
    accumulate_gradient = args.accumulate_gradient
    unfreeze_embs_at = args.emb_cold_start
    fast_train = args.fast_train

    target_length = 225
    data_clamped = use_activations # CHANGE THIS IF IT DOESN'T WORK TODO PLEASE PLEASE
    reverse_data = False


    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA available')
    else:
        device = torch.device('cpu')
        print('CUDA not available')
    vocab, data = load_data(should_reverse=reverse_data, clamp_values=use_activations)
    random.seed(args.seed)
    random.shuffle(data)
    if name != '' and os.path.exists('{}.json'.format(name)) and os.path.exists('{}.pt'.format(name)) and args.load_checkpoint:
        print('Loading existing checkpoint')
        with open('{}.json'.format(name), 'r') as f:
            metadata = json.load(f)
        model = EndToEndModel(metadata['e_type'], metadata['d_type'], metadata['vocab_len'],
                            image_size = metadata['image_size'],
                            patch_size = metadata['patch_size'],
                            dim = metadata['dim'],
                            emb_dim = metadata['emb_dim'],
                            e_depth = metadata['e_depth'],
                            e_heads = metadata['e_heads'],
                            d_depth = metadata['d_depth'],
                            d_heads = metadata['d_heads'],
                            mlp_dim = metadata['mlp_dim'],
                            num_latents = metadata['num_latents'],
                            emb_drop = metadata['emb_drop'],
                            thicc_ff=metadata['thicc_ff'],
                            use_activations=metadata['use_activations']).to(device)
        model.load_state_dict(torch.load('{}.pt'.format(name)))
        epoch = metadata['epoch']
        train_loss_array = metadata['train']
        valid_loss_array = metadata['valid']
        
    else:
        print('Creating new checkpoint')
        if name == '':
            name = 'model'
        embs = None
        if args.load_embeddings != '':
            embs = torch.load(args.load_embeddings)
        model = EndToEndModel(args.e_type, args.d_type, len(vocab),
                            image_size = 576,
                            pretrain_embeddings=embs,
                            patch_size = args.patch_size,
                            dim = args.dim,
                            emb_dim = args.emb_dim,
                            e_depth = args.e_depth,
                            e_heads = args.e_heads,
                            d_depth = args.d_depth,
                            d_heads = args.d_heads,
                            mlp_dim = args.mlp_dim,
                            num_latents = args.style_latents,
                            emb_drop = args.emb_drop,
                            thicc_ff=thicc_ff,
                            use_activations=args.activations).to(device)
        torch.save(model.state_dict(), '{}.pt'.format(name))
        metadata = {
            'epoch': 0,
            'e_type': args.e_type,
            'd_type': args.d_type,
            'vocab_len': len(vocab),
            'image_size': 576,
            'patch_size': args.patch_size,
            'dim': args.dim,
            'emb_dim': args.emb_dim if args.load_embeddings == '' else embs.shape[1],
            'e_depth': args.e_depth,
            'e_heads': args.e_heads,
            'd_depth': args.d_depth,
            'd_heads': args.d_heads,
            'mlp_dim': args.mlp_dim,
            'num_latents': args.style_latents,
            'emb_drop': args.emb_drop,
            'thicc_ff': args.thicc_ff,
            'use_activations': args.activations
        }
        epoch = 0
        train_loss_array = []
        valid_loss_array = []
        with open('{}.json'.format(name), 'w') as f:
            json.dump(metadata, f)

    print(model)
    trainable, untrainable = get_parameter_count(model)
    print('Total encoder paramters\n\tTrainable:\t{}\n\tUntrainable:\t{}'.format(trainable, untrainable))
    
    dataset = SADataset(data)
    valid_size = int(len(dataset) * valid_split)
    train_size = len(dataset) - valid_size
    train_set, valid_set = torch.utils.data.random_split(SADataset(data), [train_size, valid_size])
    train_loader, valid_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True), DataLoader(valid_set, batch_size=batch_size, drop_last=True)
    resize = transforms.Resize((192,192))

    optimizer = args.optimizer
    model_scd = None
    if optimizer == 'adam':
        model_opt = optim.Adam(model.parameters(), lr=1e-3)
        model_scd = optim.lr_scheduler.CosineAnnealingWarmRestarts(model_opt, T_0=1000, T_mult=2)
    elif optimizer == 'adamw':
        model_opt = optim.AdamW(model.parameters(), lr=1e-3)
    elif optimizer == 'asgd':
        model_opt = optim.ASGD(model.parameters(), lr=1e-3)
    elif optimizer == 'adabelief':
        model_opt = AdaBelief(model.parameters(), lr=1e-3, betas=(0.9,0.999), eps=1e-8, weight_decay=1e-2, weight_decouple=True, rectify=False, print_change_log=False)
        model_scd = optim.lr_scheduler.MultiStepLR(model_opt, milestones=[50, 75, 90], gamma=1e-1)
    elif optimizer == 'ranger':
        model_opt = RangerAdaBelief(model.parameters(), lr=5e-4, betas=(.9,.999), eps=1e-4, weight_decay=1e-4, weight_decouple=True)
    else:
        model_opt = optim.SGD(model.parameters(), lr=1e-1, momentum=0.5)
        model_scd = optim.lr_scheduler.CyclicLR(model_opt, base_lr=1e-5, max_lr=1e-1, step_size_up=len(train_loader)*10, step_size_down=len(train_loader)*10)
    if os.path.exists('{}_optim.pt'.format(name)) and args.load_checkpoint:
        model_opt.load_state_dict(torch.load('{}_optim.pt'.format(name)))
    else:
        torch.save(model_opt.state_dict(), '{}_optim.pt'.format(name))
    SOS_token = vocab['<SOS>']
    EOS_token = vocab['<EOS>']
    best_loss = None
    best_model = None
    cur_patience = 0
    print('Training start. Starting at epoch {}'.format(epoch))
    for edx in range(epoch, max_epochs):
        # Unfreeze embeddings if it is time for that
        if edx == unfreeze_embs_at:
            model.freeze_embeddings(False)
            # I am lazy and don't feel like adding another parameter for this so we unfreeze convolutional encoder when we unfreeze the embeddings
            if args.e_type == 'conv':
                model.freeze_encoder(False)
        # Prep model for training
        model.train()

        total_losses = 0
        blended_losses = 0
        layer_losses = 0
        color_losses = 0
        position_losses = 0
        startTime = time.time()
        train_divide_by = 0
        for bdx, i_batch in enumerate(train_loader):
            feature, label, mask = i_batch['feature'].to(device), i_batch['label'].to(device), i_batch['mask'].to(device)

            total_loss = 0
            batch_layer, batch_color, batch_position = 0, 0, 0
            cur_grad = 0

            current_time_scaled = edx + (bdx/len(train_loader))
            layer_alpha = linear_decay(0.5, 2, 100, current_time_scaled)
            color_alpha = linear_decay(2, 1, 100, current_time_scaled)
            position_alpha = linear_decay(1.5, 1, 100, current_time_scaled)

            # Probably not required but because we are using accumulated gradients it can lead to less overfitting on a short timescale
            ldxs = list(range(2, label.shape[1]))
            random.shuffle(ldxs)
            # Train the model in a generative manner
            for ldx in ldxs:
                pad_label, pad_mask = torch.zeros_like(label), torch.zeros_like(mask).bool()
                pad_label[:,:ldx,:], pad_mask[:,:ldx] = label[:,:ldx,:], mask[:,:ldx]

                layer_loss, color_loss, position_loss, aux_loss = model(feature, pad_label, mask=pad_mask)

                total_loss += layer_loss * layer_alpha + \
                              color_loss * color_alpha + \
                              position_loss * position_alpha + \
                              (aux_loss if aux_loss is not None else 0)
                total_losses += layer_loss.item() + color_loss.item() + position_loss.item()
                blended_losses += total_loss.item()
                layer_losses += layer_loss.item()
                color_losses += color_loss.item()
                position_losses += position_loss.item()
                batch_layer += layer_loss.item()
                batch_color += color_loss.item()
                batch_position += position_loss.item()
                train_divide_by += len(i_batch)
                if isnan(total_loss.item()):
                    print('Batch loss is NaN. Breaking')
                    return
                # Accumulate gradients every
                cur_grad = (cur_grad + 1) % accumulate_gradient
                if cur_grad == 0:
                    total_loss.backward()
                    model_opt.step()
                    model_opt.zero_grad()
                    total_loss = 0
                    if fast_train:
                        break
            
            if cur_grad != 0:
                total_loss.backward()
                model_opt.step()
                model_opt.zero_grad()
            
            if model_scd is not None:
                model_scd.step()
            if batch_metrics:
                print('\tBatch #{}, Total Loss: {}, Layer Loss: {}, Color Loss: {}, Position Loss: {}'.format(bdx, batch_layer+batch_color+batch_position, batch_layer, batch_color, batch_position))
        if isnan(total_losses):
            print('Loss is NaN. Returning')
            return
        train_loss_array.append([edx, total_losses, total_losses/train_divide_by, layer_losses, layer_losses/train_divide_by, color_losses, color_losses/train_divide_by, position_losses, position_losses/train_divide_by])
        pd.DataFrame(np.asarray(train_loss_array)).to_csv('{}_train.csv'.format(name), header=['Epoch','Train Total','Train Total Average','Train Layer Total','Train Layer Average','Train Color Total', 'Train Color Average', 'Train Position Total', 'Train Position Average'], index=False)
        print('TRAINING Epoch #{}\n\tTime spent: {}\n\tTotal Loss: {}\n\tEmbedding Loss: {}\n\tColor Loss: {}\n\tPosition Loss: {}'.format(edx, time.time()-startTime, total_losses, layer_losses, color_losses, position_losses))

        # Prep model for validation
        model.eval()

        valid_emb_loss = 0
        valid_color_loss = 0
        valid_position_loss = 0
        valid_divide_by = 0
        for bdx, i_batch in enumerate(valid_loader):
            feature, label, mask = i_batch['feature'].to(device), i_batch['label'].to(device), i_batch['mask'].to(device)
            for ldx in range(2, label.shape[1]):
                valid_divide_by += len(i_batch)
                emb_loss, color_loss, pos_loss, aux_loss = model(feature, label[:,:ldx,:], mask=mask[:,:ldx])
                valid_emb_loss += emb_loss.item()
                valid_color_loss += color_loss.item()
                valid_position_loss += pos_loss.item()

        total_loss = valid_emb_loss + valid_color_loss + valid_position_loss
        print('VALIDATION Epoch #{}, Total Loss: {}, Embedding Loss: {}, Color Loss: {}, Position Loss: {}'.format(edx, total_loss, valid_emb_loss, valid_color_loss, valid_position_loss))
        valid_loss_array.append([edx, total_loss, total_loss/valid_divide_by, valid_emb_loss, valid_emb_loss/valid_divide_by, valid_color_loss, valid_color_loss/valid_divide_by, valid_position_loss, valid_position_loss/valid_divide_by])
        pd.DataFrame(np.asarray(valid_loss_array)).to_csv('{}_valid.csv'.format(name), header=['Epoch','Valid Total','Valid Total Average','Valid Layer Total','Valid Layer Average','Valid Color Total', 'Valid Color Average', 'Valid Position Total', 'Valid Position Average'], index=False)
        print('------------------------------------------------------------------------------------------------')


        # Saving after epoch
        torch.save(model.state_dict(), '{}.pt'.format(name))
        torch.save(model_opt.state_dict(), '{}_optim.pt'.format(name))
        metadata['epoch'] += 1
        metadata['train'] = train_loss_array
        metadata['valid'] = valid_loss_array
        with open('{}.json'.format(name), 'w') as f:
            json.dump(metadata, f)
        
        # Checking if it is a new PB
        if best_loss is None or total_loss < best_loss:
            best_loss = total_loss
            best_model = model.state_dict()
            cur_patience = 0
        else:
            cur_patience += 1
        
        # Evaluate model on test file if it is time
        if eval_every > 0 and edx % eval_every == 0:
            model.eval()
            feature = io.imread('PleaseWorkHard.png')[:,:,:3].astype(np.float32) / 255.
            #feature = io.imread('PleaseWork.png').astype(np.float32) / 255.
            feature = torch.from_numpy(feature.transpose((2, 0, 1))).to(device)
            feature = resize(feature)
            generated = np.asarray(model.generate(feature.unsqueeze(0), vocab, 225))
            dest_name = '{}_{}'.format(name, edx)
            np.save('test.npy', generated)
            convert_numpy_to_saml('test.npy', vocab, dest_path=dest_name+'.saml', name=dest_name, values_clamped=data_clamped)
        
        # Break if the progress has gone stale
        if cur_patience > max_patience:
            print('Out of patience. Breaking')
            break
    
    # Save over current model with best model. Could do optimizer too but that just doesn't make sense to me?
    torch.save(best_model, '{}.pt'.format(name))

parser = ArgumentParser()
parser.add_argument('--epochs', default=100, type=int, help='Maximum number of epochs to train')
parser.add_argument('--patience', default=100, type=int, help='Maximum patience while training (set to equal --epochs if you want no patience)')
parser.add_argument('--optimizer', default='adam', type=str, help='Which optimizer to use, defaults to Adam')
parser.add_argument('--batch_size', default=4, type=int, help='What batch size to use')
parser.add_argument('--batch_metrics', default=False, type=str2bool, help='Whether or not to print metrics per batch')
parser.add_argument('--seed', default=420, type=int, help='The random seed to use when initially shuffling the data')
parser.add_argument('--activations', default=False, type=str2bool, help='Whether to use sigmoid and tanh activations for color and position respectively. Otherwise defaults to linear')
parser.add_argument('--valid_split', default=0.3, type=float, help='What percent of the dataset should be used for validation')
parser.add_argument('--scaled_loss', default=False, type=str2bool, help='Whether to scale loss by the L1 norm')
parser.add_argument('--eval_every', default=0, type=int, help='How often (in epochs) to evaluate the model on a test image')
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
parser.add_argument('--e_type', default='vit', type=str, help='Which encoder to use. Valid values are: vanilla, nystrom, cvt, style')
parser.add_argument('--d_type', default='decoder', type=str, help='Which decoder to use. Valid values are: vanilla, routing')
parser.add_argument('--thicc_ff', default=False, type=str2bool, help='Whether to use the more robust feed forward sequence on final decoder outputs')
parser.add_argument('--name', default='', type=str, help='Name of checkpoint. If it already exists then we load it. You need to manually delete a checkpoint if you want to write over it')
parser.add_argument('--accumulate_gradient', default=8, type=int, help='How steps during one data point we should accumulate gradient before backpropgating and steping through')
parser.add_argument('--load_embeddings', default='', type=str, help='If you have an embeddings file you can load it using this')
parser.add_argument('--emb_cold_start', default=-1, type=int, help='Only applicable if you are using load_embeddings. Unfreezes embeddings at specified epoch. Defaults to never unfreezing')
parser.add_argument('--load_checkpoint', default=False, type=str2bool, help='Whether to load a pre-existing checkpoint')
parser.add_argument('--layer_alpha', default=1.0, type=float, help='The scaling factor for the layer prediction loss')
parser.add_argument('--color_alpha', default=1.0, type=float, help='The scaling factor for the color prediction loss')
parser.add_argument('--position_alpha', default=1.0, type=float, help='The scaling factor for the position prediction loss')

parser.add_argument('--fast_train', default=False, type=str2bool, help='Whether to only do "accumulate_gradient"s number of passes before going to next batch')

args = parser.parse_args()

if __name__ == '__main__':
    main(args)
