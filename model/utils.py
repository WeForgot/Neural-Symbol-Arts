import os
import glob
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.nn as nn
import torchvision
from webcolors import hex_to_rgb

from .datasets import SADataset, LayersDataset, RandomTransform, ToTensor

def get_parameter_count(model: nn.Module):
    t_model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    u_model_parameters = filter(lambda p: not p.requires_grad, model.parameters())
    t_params = sum([np.prod(p.size()) for p in t_model_parameters])
    u_params = sum([np.prod(p.size()) for p in u_model_parameters])
    return t_params, u_params

def load_layers(base_path):
    # TODO: Modify color by random hex value to ensure that latent is covered over all color values
    dataset = LayersDataset(base_path, transform=torchvision.transforms.Compose([
        RandomTransform(),
        ToTensor(),
    ]))
    return dataset

def load_data(base_path, weights):
    bases = [os.path.split(x)[-1][:-5] for x in glob.glob(os.path.join(base_path, '*.saml'))]
    dataset = SADataset(bases, weights)
    return dataset

def load_weights(weight_fp, names_fp):
    file_to_weight = {}
    with open(weight_fp, 'r') as wf, open(names_fp, 'r') as rf:
        for x, y in zip(rf, wf):
            file_to_weight[x.strip()] = list(map(float, y.strip().split('\t')))
    return file_to_weight


def load_saml(filepath, weights):
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        all_lines = [x for x in f.readlines()]
        _ = all_lines.pop(1) # This isn't valid XML so scrap it
    weightLen = len(list(weights.values())[0])
    rowLen = weightLen + 12 # N embeddings + 3 color channels + alpha + ltx + lty + tbx + lby + rtx + rty + rbx + rby
    temp = ''.join(all_lines)
    root = ET.fromstring(''.join(all_lines))
    saml = np.zeros((225, rowLen), dtype=np.float32)
    mask = np.zeros((225,), dtype=np.uint8)
    #{'name': 'Symbol 0', 'visible': 'true', 'type': '8', 'color': '#50342c', 'alpha': '1', 'ltx': '97', 'lty': '47', 'lbx': '97', 'lby': '25', 'rtx': '91', 'rty': '47', 'rbx': '91', 'rby': '25'}
    for ldx, layer in enumerate(root):
        processedLine = np.zeros((rowLen,), dtype=np.float32)
        value = int(layer.attrib['type'])
        formattedValue = '{}.png'.format(value+1)
        weight = weights[formattedValue]
        if formattedValue not in weights:
            print('{} not in weights'.format(formattedValue))
            break
        processedLine[:weightLen] = weight
        rgb = hex_to_rgb(layer.attrib['color'])
        processedLine[weightLen:weightLen+3] = list(rgb)
        processedLine[weightLen+3] = float(layer.attrib['alpha'])
        processedLine[weightLen+4:] = [
            layer.attrib['ltx'],
            layer.attrib['lty'],
            layer.attrib['lbx'],
            layer.attrib['lby'],
            layer.attrib['rtx'],
            layer.attrib['rty'],
            layer.attrib['rbx'],
            layer.attrib['rby']
        ]
        saml[ldx] = processedLine
        mask[ldx] = 1
    return saml, mask

# Remember that SAML layer type values need to add 1 to them to get the cooresponding layer name
class Vocabulary(object):
    def __init__(self):
        self.layer_to_idx = {'<SOS>': 0, '<EOS>': 1, '<PAD>': 2}
        self.idx_to_layer = {0: '<SOS>', 1: '<EOS>', 2: '<PAD>'}
    
    def load_layers(self, layer_path: str) -> None:
        imgs = glob.glob(os.path.join(layer_path, '[0-9]*.png'))
        for img in imgs:
            cur_len = len(self.layer_to_idx)
            idx = str(int(os.path.split(img)[-1].rstrip('.png')) - 1)
            self.layer_to_idx[idx] = cur_len
            self.idx_to_layer[cur_len] = idx
    
    def __len__(self):
        return len(self.layer_to_idx)
    
    def __repr__(self):
        to_return = ''
        for x in self.idx_to_layer:
            to_return += '{}: {}\n'.format(x, self.idx_to_layer[x])
        return to_return
    
    def __getitem__(self, idx):
        if isinstance(idx, str):
            return self.layer_to_idx[idx]
        elif isinstance(idx, int):
            return self.idx_to_layer[idx]
        else:
            raise ValueError('Vocabulary indices can only be strings or integers')
