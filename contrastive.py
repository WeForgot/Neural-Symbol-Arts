import random

import numpy as np
from PIL import Image
from palettable.cartocolors.qualitative import *
from skimage import io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms.transforms import RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, Resize

from byol_pytorch import BYOL
from ContrastiveSaliency.model import EmbeddingConvNet, EmbeddingResnet
from model.utils import load_data, SADataset

palettes = [None, Bold_2.colors, Bold_3.colors, Bold_4.colors, Bold_5.colors, Bold_6.colors, Bold_7.colors, Bold_8.colors, Bold_9.colors, Bold_10.colors,
            Pastel_2.colors + Bold_9.colors, Pastel_2.colors + Bold_10.colors, Pastel_3.colors + Bold_10.colors, Pastel_4.colors + Bold_10.colors,
            Pastel_5.colors + Bold_10.colors, Pastel_6.colors + Bold_10.colors, Pastel_7.colors + Bold_10.colors, Pastel_8.colors + Bold_10.colors,
            Pastel_9.colors + Bold_10.colors, Pastel_10.colors + Bold_10.colors,]

class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA available')
    else:
        device = torch.device('cpu')
        print('CUDA not available')
    _, data = load_data()
    data = SADataset(data, img_size=192)
    dataloader = DataLoader(data, batch_size=8, shuffle=True, drop_last=True)
    resize = Resize((192, 192))
    hflip = RandomHorizontalFlip(1)
    vflip = RandomVerticalFlip(1)
    color = ColorJitter()
    classes = 15
    assert classes > 1 and classes <= len(palettes), 'For now we can only support up to 10 classes'
    colors = palettes[classes]

    #model = EmbeddingConvNet(classes=classes).to(device)
    model = EmbeddingResnet(classes=classes).to(device)
    print(model)
    learner = BYOL(
        model, 192, 'encoder.dep4'
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion1 = nn.TripletMarginLoss()
    criterion2 = nn.MSELoss()
    best_loss = None
    best_model = None
    patience = 0
    edx = 0
    while True:
        running_loss = 0.0
        for idx, i_batch in enumerate(dataloader):
            x = i_batch['feature'].to(device)
            loss = learner(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learner.update_moving_average()
            print('\tBatch {}, Loss: {}'.format(idx, loss.item()))
            running_loss += loss.item()
        
        if edx % 2 == 0:
            feature = io.imread('PleaseWorkHard.png')[:,:,:3].astype(np.float32) / 255.
            #feature = io.imread('PleaseWork.png')[:,:,:3].astype(np.float32) / 255.
            feature = torch.from_numpy(feature.transpose((2, 0, 1))).to(device)
            feature = resize(feature)
            test, _ = model(feature.unsqueeze(0))
            test = test[0]
            test = torch.argmax(test, 0)
            test = test.cpu().detach().numpy().astype(np.uint8)
            out = np.zeros((192,192,3), dtype=np.uint8)
            for x in range(len(test)):
                for y in range(len(test)):
                    out[x,y] = colors[test[x,y]]
            img = Image.fromarray(out, mode='RGB')
            img.save('contrastive.png')
        else:
            feature = io.imread('PleaseWork.png')[:,:,:3].astype(np.float32) / 255.
            feature = torch.from_numpy(feature.transpose((2, 0, 1))).to(device)
            feature = resize(feature)
            test, _ = model(feature.unsqueeze(0))
            test = test[0]
            test = torch.argmax(test, 0)
            test = test.cpu().detach().numpy().astype(np.uint8)
            out = np.zeros((192,192,3), dtype=np.uint8)
            for x in range(len(test)):
                for y in range(len(test)):
                    out[x,y] = colors[test[x,y]]
            img = Image.fromarray(out, mode='RGB')
            img.save('contrastive_easy.png')
        if best_loss is None or running_loss < best_loss:
            best_loss = running_loss
            best_model = model.state_dict()
            patience = 0
        else:
            patience += 1
        print('Epoch {}, Loss: {}, Patience: {}'.format(edx, running_loss, patience))
        if patience > 50:
            print('Out of patience')
            break
        edx += 1
    model.load_state_dict(best_model)
    #feature = io.imread('PleaseWorkHard.png')[:,:,:3].astype(np.float32) / 255.
    feature = io.imread('PleaseWork.png')[:,:,:3].astype(np.float32) / 255.
    feature = torch.from_numpy(feature.transpose((2, 0, 1))).to(device)
    feature = resize(feature)
    test, _ = model(feature.unsqueeze(0))
    test = test[0]
    test = torch.argmax(test, 0)
    test = test.cpu().detach().numpy().astype(np.uint8)
    out = np.zeros((192,192,3), dtype=np.uint8)
    for x in range(len(test)):
        for y in range(len(test)):
            out[x,y] = colors[test[x,y]]
    img = Image.fromarray(out, mode='RGB')
    img.save('contrastive_easy.png')

    '''
    while True:
        running_loss = 0.0
        for idx, i_batch in enumerate(dataloader):
            x = i_batch['feature'].to(device)
            x, y = x[:4], x[4:]
            xp = nn.Identity()(x)
            xx, xy = model(x)
            if random.random() > 0.5:
                xp = hflip(xp)
                xx = hflip(xx)
            if random.random() > 0.5:
                xp = vflip(xp)
                xx = vflip(xx)
            if random.random() > 0.5:
                xp = color(xp)
            xp = xp.to(device)
            xp, _ = model(xp)
            yx, yy = model(y)
            optimizer.zero_grad()
            loss = criterion1(xx, xp, yx) + criterion2(x, xy) + criterion2(y, yy)
            loss.backward()
            optimizer.step()
            print('\tBatch {}, Loss: {}'.format(idx, loss.item()))
            running_loss += loss.item()
        if edx % 2 == 0:
            feature = io.imread('PleaseWorkHard.png')[:,:,:3].astype(np.float32) / 255.
            #feature = io.imread('PleaseWork.png')[:,:,:3].astype(np.float32) / 255.
            feature = torch.from_numpy(feature.transpose((2, 0, 1))).to(device)
            feature = resize(feature)
            test, _ = model(feature.unsqueeze(0))
            test = test[0]
            test = torch.argmax(test, 0)
            test = test.cpu().detach().numpy().astype(np.uint8)
            out = np.zeros((192,192,3), dtype=np.uint8)
            for x in range(len(test)):
                for y in range(len(test)):
                    out[x,y] = colors[test[x,y]]
            img = Image.fromarray(out, mode='RGB')
            img.save('contrastive.png')
        else:
            feature = io.imread('PleaseWork.png')[:,:,:3].astype(np.float32) / 255.
            feature = torch.from_numpy(feature.transpose((2, 0, 1))).to(device)
            feature = resize(feature)
            test, _ = model(feature.unsqueeze(0))
            test = test[0]
            test = torch.argmax(test, 0)
            test = test.cpu().detach().numpy().astype(np.uint8)
            out = np.zeros((192,192,3), dtype=np.uint8)
            for x in range(len(test)):
                for y in range(len(test)):
                    out[x,y] = colors[test[x,y]]
            img = Image.fromarray(out, mode='RGB')
            img.save('contrastive_easy.png')
        
        if best_loss is None or running_loss < best_loss:
            best_loss = running_loss
            best_model = model.state_dict()
            patience = 0
        else:
            patience += 1
        print('Epoch {}, Loss: {}, Patience: {}'.format(edx, running_loss, patience))
        if patience > 50:
            print('Out of patience')
            break
        edx += 1
    model.load_state_dict(best_model)
    #feature = io.imread('PleaseWorkHard.png')[:,:,:3].astype(np.float32) / 255.
    feature = io.imread('PleaseWork.png')[:,:,:3].astype(np.float32) / 255.
    feature = torch.from_numpy(feature.transpose((2, 0, 1))).to(device)
    feature = resize(feature)
    test, _ = model(feature.unsqueeze(0))
    test = test[0]
    test = torch.argmax(test, 0)
    test = test.cpu().detach().numpy().astype(np.uint8)
    out = np.zeros((192,192,3), dtype=np.uint8)
    for x in range(len(test)):
        for y in range(len(test)):
            out[x,y] = colors[test[x,y]]
    img = Image.fromarray(out, mode='RGB')
    img.save('contrastive_easy.png')
    '''

            

if __name__ == '__main__':
    main()