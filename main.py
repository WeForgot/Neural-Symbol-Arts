from argparse import ArgumentParser
import json
from math import isnan
from statistics import mean
from glob import glob
import os
from pathlib import Path
import pickle
import random
import re

from dotenv import load_dotenv
import numpy as np
from skimage import io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

from model.model import EndToEndModel
from model.datasets import SADataset
from model.utils import get_parameter_count, Vocabulary, convert_numpy_to_saml, str2bool

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

    layer_alpha = args.layer_alpha
    color_alpha = args.color_alpha
    position_alpha = args.position_alpha

    target_length = 225
    data_clamped = True

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
    if name != '' and os.path.exists('{}.json'.format(name)) and os.path.exists('{}.pt'.format(name)):
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
                            thicc_ff=metadata['thicc_ff']).to(device)
        model.load_state_dict(torch.load('{}.pt'.format(name)))
        epoch = metadata['epoch']
        
    else:
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
                            thicc_ff=thicc_ff).to(device)
        torch.save(model.state_dict(), '{}.pt'.format(name))
        metadata = {
            'epoch': 0,
            'e_type': args.e_type,
            'd_type': args.d_type,
            'vocab_len': len(vocab),
            'image_size': 576,
            'patch_size': args.patch_size,
            'dim': args.dim,
            'emb_dim': args.emb_dim,
            'e_depth': args.e_depth,
            'e_heads': args.e_heads,
            'd_depth': args.d_depth,
            'd_heads': args.d_heads,
            'mlp_dim': args.mlp_dim,
            'num_latents': args.style_latents,
            'emb_drop': args.emb_drop,
            'thicc_ff': args.thicc_ff
        }
        epoch = 0
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

    optimizer = args.optimizer
    model_scd = None
    if optimizer == 'adam':
        model_opt = optim.Adam(model.parameters(), lr=1e-3)
    elif optimizer == 'adamw':
        model_opt = optim.AdamW(model.parameters(), lr=1e-3)
    elif optimizer == 'asgd':
        model_opt = optim.ASGD(model.parameters(), lr=1e-3)
        #model_scd = optim.lr_scheduler.CyclicLR(model_opt, base_lr=1e-5, max_lr=1e-3, step_size_up=500, step_size_down=500, mode='triangular')
    else:
        model_opt = optim.SGD(model.parameters(), lr=1e-5)
        model_scd = optim.lr_scheduler.CyclicLR(model_opt, base_lr=1e-5, max_lr=1e-3, step_size_up=500, step_size_down=500, mode='triangular')
    if os.path.exists('{}_optim.pt'.format(name)):
        model_opt.load_state_dict(torch.load('{}_optim.pt'.format(name)))
    else:
        torch.save(model_opt.state_dict(), '{}_optim.pt'.format(name))
    
    SOS_token = vocab['<SOS>']
    EOS_token = vocab['<EOS>']
    best_loss = None
    best_model = None
    cur_patience = 0
    train_loss_array = []
    valid_loss_array = []
    print('Training start. Starting at epoch {}'.format(epoch))
    for edx in range(epoch, max_epochs):
        if edx == unfreeze_embs_at:
            model.freeze_embeddings(False)
        model.train()
        total_losses = 0
        blended_losses = 0
        layer_losses = 0
        color_losses = 0
        position_losses = 0
        for bdx, i_batch in enumerate(train_loader):
            feature, label, mask = i_batch['feature'].to(device), i_batch['label'].to(device), i_batch['mask'].to(device)

            model_opt.zero_grad()
            total_loss = 0
            batch_layer, batch_color, batch_position = 0, 0, 0
            cur_grad = 0
            for ldx in range(2, label.shape[1]):
                layer_loss, color_loss, position_loss, aux_loss = model(feature, label[:,:ldx,:], mask=mask[:,:ldx])

                total_loss += layer_alpha * layer_loss + color_alpha *  color_loss + position_alpha * position_loss + (aux_loss if aux_loss is not None else 0)
                total_losses += layer_loss.item() + color_loss.item() + position_loss.item()
                blended_losses += total_loss.item()
                layer_losses += layer_loss.item()
                color_losses += color_loss.item()
                position_losses += position_loss.item()
                batch_layer += layer_loss.item()
                batch_color += color_loss.item()
                batch_position += position_loss.item()
                if isnan(total_loss.item()):
                    print('Batch loss is NaN. Breaking')
                    return
                cur_grad = (cur_grad + 1) % accumulate_gradient
                if cur_grad == 0:
                    total_loss.backward()
                    model_opt.step()
                    if model_scd is not None:
                        model_scd.step()
                    total_loss = 0
            if cur_grad != 0:
                total_loss.backward()
                model_opt.step()
                if model_scd is not None:
                    model_scd.step()

            if batch_metrics:
                print('\tBatch #{}, Total Loss: {}, Layer Loss: {}, Color Loss: {}, Position Loss: {}'.format(bdx, batch_layer+batch_color+batch_position, batch_layer, batch_color, batch_position))
        if isnan(total_losses):
            print('Loss is NaN. Returning')
            return
        train_loss_array.append([edx, total_losses, layer_losses, color_losses, position_losses])
        pd.DataFrame(np.asarray(train_loss_array)).to_csv('train_loss.csv', header=['epoch','total','layer','color','position'], index=False)
        print('TRAINING Epoch #{}, Total Loss: {}, Embedding Loss: {}, Color Loss: {}, Position Loss: {}'.format(edx, total_losses, layer_losses, color_losses, position_losses))
        
        # Prep model for validation
        model.eval()

        valid_emb_loss = 0
        valid_color_loss = 0
        valid_position_loss = 0
        for bdx, i_batch in enumerate(valid_loader):
            feature, label, mask = i_batch['feature'].to(device), i_batch['label'].to(device), i_batch['mask'].to(device)
            for ldx in range(2, label.shape[1]):
                emb_loss, color_loss, pos_loss, aux_loss = model(feature, label[:,:ldx,:], mask=mask[:,:ldx])
                valid_emb_loss += emb_loss.item()
                valid_color_loss += color_loss.item()
                valid_position_loss += pos_loss.item()

        total_loss = valid_emb_loss + valid_color_loss + valid_position_loss
        print('VALIDATION Epoch #{}, Total Loss: {}, Embedding Loss: {}, Color Loss: {}, Position Loss: {}'.format(edx, total_loss, valid_emb_loss, valid_color_loss, valid_position_loss))
        valid_loss_array.append([edx, total_loss, valid_emb_loss, valid_color_loss, valid_position_loss])
        pd.DataFrame(np.asarray(valid_loss_array)).to_csv('valid_loss.csv', header=['epoch','total','layer','color','position'], index=False)
        print('------------------------------------------------------------------------------------------------')


        # Saving after epoch
        torch.save(model.state_dict(), '{}.pt'.format(name))
        torch.save(model_opt.state_dict(), '{}_optim.pt'.format(name))
        metadata['epoch'] += 1
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
            #feature = io.imread('PleaseWork.png')[:,:,:3].astype(np.float32) / 255.
            feature = io.imread('PleaseWork.png').astype(np.float32) / 255.
            feature = torch.from_numpy(feature.transpose((2, 0, 1))).to(device)
            generated = np.asarray(model.generate(feature.unsqueeze(0), vocab, 225, use_activations=use_activations))
            dest_name = 'test_{}'.format(edx)
            np.save('test.npy', generated)
            convert_numpy_to_saml('test.npy', vocab, dest_path=dest_name+'.saml', name=dest_name, values_clamped=data_clamped)
        
        # Break if the progress has gone stale
        if cur_patience > max_patience:
            print('Out of patience. Breaking')
            break
    model.load_state_dict(best_model)
    #feature = io.imread('PleaseWork.png')[:,:,:3].astype(np.float32) / 255.
    feature = io.imread('PleaseWork.png').astype(np.float32) / 255.
    feature = torch.from_numpy(feature.transpose((2, 0, 1))).to(device)
    generated = np.asarray(model.generate(feature.unsqueeze(0), vocab, 225, use_activations=use_activations))
    np.save('test.npy', generated)
    convert_numpy_to_saml('test.npy', vocab, values_clamped=data_clamped)
    print(generated)
    print(generated.shape)

parser = ArgumentParser()
parser.add_argument('--epochs', default=100, type=int, help='Maximum number of epochs to train')
parser.add_argument('--patience', default=100, type=int, help='Maximum patience while training (set to equal --epochs if you want no patience)')
parser.add_argument('--optimizer', default='adam', type=str, help='Which optimizer to use, defaults to Adam')
parser.add_argument('--batch_size', default=4, type=int, help='What batch size to use')
parser.add_argument('--batch_metrics', default=False, type=str2bool, help='Whether or not to print metrics per batch')
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

parser.add_argument('--layer_alpha', default=1.0, type=float, help='The scaling factor for the layer prediction loss')
parser.add_argument('--color_alpha', default=1.0, type=float, help='The scaling factor for the color prediction loss')
parser.add_argument('--position_alpha', default=1.0, type=float, help='The scaling factor for the position prediction loss')

args = parser.parse_args()

if __name__ == '__main__':
    main(args)
