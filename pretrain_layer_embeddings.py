import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from vit_pytorch.efficient import ViT
from nystrom_attention import Nystromformer

from models.datasets import LayersDataset
from models.utils import load_layers, get_parameter_count

def pretrain_embeddings():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA available')
    else:
        device = torch.device('cpu')
        print('CUDA not available')
    dataset = load_layers(os.path.join('.','data','Layers'))
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = ViT(
        image_size=576,
        patch_size=16,
        num_classes = len(dataset),
        dim = 16,
        transformer=Nystromformer(
            dim = 16,
            depth = 6,
            heads = 8,
            num_landmarks=16
        )
    ).to(device)
    '''
    model = CvT(
        num_classes=len(dataset)).to(device)
    '''
    print(model)
    trainable, untrainable = get_parameter_count(model)
    print('Total paramters\n\tTrainable:\t{}\n\tUntrainable:\t{}'.format(trainable, untrainable))
    opt = optim.AdamW(model.parameters(), lr=3e-4)
    lf = nn.CrossEntropyLoss()
    best_loss = None
    best_model = None
    batch_metrics = True
    max_patience = 50
    cur_patience = 0
    criteria = 'sum'
    if criteria == 'sum':
        loss_func = lambda x: sum(x)
    elif criteria == 'mean':
        loss_func = lambda x: sum(x) / len(x)
    else:
        loss_func = lambda x: max(x)
    print('Beginning training')
    with open('pretrain_metrics.csv', 'w') as f:
        for edx in range(1000):
            losses = []
            for idx, i_batch in enumerate(dataloader):
                image, label = i_batch['image'], i_batch['idx']
                preds = model(image.to(device))
                loss = lf(preds, label.to(device))
                if np.isnan(loss.item()):
                    print('\tNaN loss. Breaking')
                    return
                losses.append(loss.item())
                opt.zero_grad()
                loss.backward()
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
    print(dataset[0]['feature'].shape)
    print(dataset[0]['label'].shape)
    model.load_state_dict(best_model)
    model.eval()
    torch.save(model.state_dict(), 'better_trained.pt')
    model.mlp_head = nn.Identity()
    with open('weights.tsv', 'w') as f, open('names.tsv', 'w') as n:
        for sample in dataset:
            path, image = sample['path'], sample['image']
            weights = model(image.unsqueeze(0).to(device))[0].tolist()
            f.write('\t'.join(list(map(str, weights)))+'\n')
            n.write('{}\n'.format(os.path.split(path)[-1]))

if __name__ == '__main__':
    pretrain_embeddings()