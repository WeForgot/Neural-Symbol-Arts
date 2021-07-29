import random

from einops import rearrange

import numpy as np

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from torchvision.utils import save_image
from torchvision.transforms.functional import InterpolationMode, resize

from model.style_model import StyleViT
#from model.custom_vit import ViT, Attention
from vit_pytorch.vit import ViT, Attention
from vit_pytorch.recorder import Recorder
from vit_pytorch.dino import Dino

from model.datasets import SADataset
from model.utils import load_data


def find_modules(nn_module, type):
    return [module for module in nn_module.modules() if isinstance(module, type)]

class Recorder(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.vit = vit

        self.data = None
        self.recordings = []
        self.hooks = []
        self.hook_registered = False
        self.ejected = False

    def _hook(self, _, input, output):
        self.recordings.append(output.clone().detach())

    def _register_hook(self):
        modules = find_modules(self.vit.transformer, Attention)
        for module in modules:
            handle = module.attend.register_forward_hook(self._hook)
            self.hooks.append(handle)
        self.hook_registered = True

    def eject(self):
        self.ejected = True
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        return self.vit

    def clear(self):
        self.recordings.clear()

    def record(self, attn):
        recording = attn.clone().detach()
        self.recordings.append(recording)

    def forward(self, img):
        assert not self.ejected, 'recorder has been ejected, cannot be used anymore'
        self.clear()

        if not self.hook_registered:
            self._register_hook()

        pred = self.vit(img)
        attns = torch.stack(self.recordings, dim = 1)
        return pred, attns

def main():
    image_size = 224
    patch_size = 8
    dim = 384
    depth = 2
    heads = 6

    vocab, data = load_data()

    encoder = ViT(
        image_size = image_size,
        patch_size = patch_size,
        num_classes = len(vocab),
        dim = dim,
        depth = depth,
        heads = heads,
        mlp_dim = dim * 4,
        dropout = 0.1,
        emb_dropout = 0.1
    ).cuda()

    learner = Dino(
        encoder,
        image_size = image_size,
        hidden_layer= 'to_latent',
        num_classes_K = len(vocab)
    ).cuda()

    valid_split = 0.2
    train_split, valid_split = data[int(len(data)*valid_split):], data[:int(len(data)*valid_split)]

    batch_size = 16
    train_dataset, valid_dataset = SADataset(train_split, img_size=image_size), SADataset(valid_split, img_size=image_size)
    train_dataloader, valid_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(valid_dataset, batch_size=batch_size)

    opt = optim.AdamW(learner.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=len(train_dataloader) * 5, eta_min=1e-6)

    best_loss = None
    best_model = None
    patience = 0
    max_epochs = 100
    max_patience = 10

    for edx in range(max_epochs):
        encoder.train()
        learner.train()
        running_loss = 0.0
        for bdx, i_batch in enumerate(train_dataloader):
            img = i_batch['feature'].cuda()
            loss = learner(img)
            running_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average()
            print('\tBatch #{}, Loss: {}'.format(bdx, loss.item()))
            scheduler.step()
        
        print('TRAINING Epoch #{}, Total Loss: {}'.format(edx, running_loss))

        encoder.eval()
        learner.eval()
        running_loss = 0.0
        with torch.no_grad():
            for bdx, i_batch in enumerate(valid_dataloader):
                img = i_batch['feature'].cuda()
                loss = learner(img)
                running_loss += loss.item()
        
        if best_loss is None or running_loss < best_loss:
            best_loss = running_loss
            best_model = encoder.state_dict()
            patience = 0
        else:
            patience += 1
        
        print('VALIDATION Epoch #{}, Total Loss: {}, Patience: {}'.format(edx, running_loss, patience))

        if patience > max_patience:
            print('Out of patience')
            break

        if best_loss == 0.0:
            print("We aren't doing any better than this...")
            break
    encoder.load_state_dict(best_model)
    
    encoder = Recorder(encoder).cuda()
    img = random.choice(valid_dataset)['feature'].unsqueeze(0).cuda()
    save_image(img, 'original.png')

    _, attns = encoder(img)
    side_size = image_size // patch_size
    last_out = attns[0]
    for ldx in range(len(last_out)):
        for hdx in range(len(last_out[ldx])):
            head_img = last_out[ldx][hdx][0][1:]
            min_elem = torch.min(head_img)
            range_elem = torch.max(head_img) - min_elem
            head_img = (head_img - min_elem) / range_elem
            head_img = rearrange(head_img, '(s1 s2) -> s1 s2', s1=side_size, s2=side_size).unsqueeze(0)
            head_img = resize(head_img, (image_size, image_size))
            im = resize(torch.mul(img, head_img), (image_size, image_size), InterpolationMode.BICUBIC)
            save_image(im, 'attn_{}_{}.png'.format(ldx,hdx))


if __name__ == '__main__':
    main()