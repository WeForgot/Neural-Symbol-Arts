import glob
import os
import pickle
import random

import numpy as np
from PIL import Image
from skimage.io import imread, imsave
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import models

from model.utils import Vocabulary, get_parameter_count

class ColorModify(nn.Module):
    def __init__(self):
        super().__init__()
    def __call__(self, sample):
        r,g,b = random.random(), random.random(), random.random()
        sample[0,:,:] *= r
        sample[1,:,:] *= g
        sample[2,:,:] *= b
        return sample
class LayersDataset(Dataset):
    def __init__(self, vocab, base_path):
        self.vocab = vocab
        sos_token, eos_token, pad_token = vocab['<SOS>'], vocab['<EOS>'], vocab['<PAD>']
        self.vocab.remove_item(sos_token)
        self.vocab.remove_item(eos_token)
        self.vocab.remove_item(pad_token)

        self.base_path = base_path
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop((288, 576)),
            transforms.RandomApply(nn.ModuleList([ColorModify()]), p=0.9),
            transforms.RandomApply(nn.ModuleList([transforms.RandomAffine(degrees=90, translate=(.2, .8), scale=(0.5, 2), shear=0.5, interpolation=transforms.InterpolationMode.BILINEAR)]), p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])
        self.no_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop((288, 576)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.use_transforms = True
    
    def __len__(self):
        return len(self.vocab)
    
    def __getitem__(self, idx):
        label = idx
        idx += 3 # We removed 3 tokens and +1 next line because images are not zero indexed
        feature = imread(os.path.join(self.base_path,'{}.png'.format(int(self.vocab[idx])+1)))[:,:,:3]
        if self.use_transforms:
            feature = self.transform(feature)
        else:
            feature = self.no_transform(feature)
        return feature, label




def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA available')
    else:
        device = torch.device('cpu')
        print('CUDA not available')
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    dataset = LayersDataset(vocab, os.path.join('.', 'data', 'Layers'))
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = models.mobilenet_v3_small()
    emb_size = 16
    # MobilenetV3-Small
    model.classifier = nn.Sequential(
        nn.Linear(in_features=576, out_features=emb_size, bias=True),
        nn.Hardswish(),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=emb_size, out_features=len(vocab), bias=True)
    )
    model = model.to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    trainable, untrainable = get_parameter_count(model)
    print('Trainable: {}, Untrainable: {}'.format(trainable, untrainable))
    epochs = 1000000000000000
    best_loss = None
    best_model = None
    patience = 0
    max_patience = 20
    for edx in range(epochs):
        running_loss = 0
        for bdx, batch in enumerate(dataloader):
            feature, label = batch
            feature, label = feature.to(device), label.to(device)
            optimizer.zero_grad()
            out = model(feature)
            loss = F.cross_entropy(out, label)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        if best_loss is None or running_loss < best_loss:
            best_loss = running_loss
            best_model = model.state_dict()
            patience = 0
        else:
            patience += 1
        if patience > max_patience:
            print('Out of patience. Breaking')
            break
        print('Epoch #{}, Loss: {}, Patience: {}/{}'.format(edx, running_loss, patience, max_patience+1))

    model.load_state_dict(best_model)

    embeddings = torch.zeros(len(vocab)+3, emb_size, dtype=torch.float32)
    nn.init.xavier_normal_(embeddings)

    model.classifier[-1] = nn.Identity().to(device)
    model.classifier[-2] = nn.Identity().to(device)
    model.classifier[-3] = nn.Identity().to(device)

    eval_dataloader = DataLoader(dataset)

    for fdx, (feature, label) in enumerate(eval_dataloader):
        feature = feature.to(device)
        with torch.no_grad():
            out = model(feature)
        embeddings[label+3] = out[0].cpu()
    torch.save(embeddings, 'layer_embeddings_{}.pt'.format(emb_size))



if __name__ == '__main__':
    main()