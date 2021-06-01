import argparse
import os
import glob
import lxml.etree as ET
import xml.etree.ElementTree as ETO
from typing import Union, List, Tuple

import numpy as np
import skimage.io as io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as VF
from webcolors import hex_to_rgb, rgb_to_hex

from .datasets import SADataset, RandomTransform, ToTensor

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k >0: keep only top k tokens with highest probability (top-k filtering).
            top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def get_parameter_count(model: nn.Module):
    t_model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    u_model_parameters = filter(lambda p: not p.requires_grad, model.parameters())
    t_params = sum([np.prod(p.size()) for p in t_model_parameters])
    u_params = sum([np.prod(p.size()) for p in u_model_parameters])
    return t_params, u_params

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


def load_saml(filepath, weights) -> Tuple[np.ndarray, np.ndarray]:
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

def convert_numpy_to_saml(data, vocab, dest_path=None, name='Test', values_clamped=False) -> None:
    if dest_path is None:
        dest_path = name + '.saml'
    
    with open(dest_path, 'w') as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        saml_lines = []
        for line in data:
            if vocab[int(line[0])] == '<SOS>' or vocab[int(line[0])] == '<PAD>':
                continue
            elif vocab[int(line[0])] == '<EOS>':
                break
            else:
                saml_lines.append(line)
        saml_lines.reverse()
        xml_data = ET.Element('sa')
        #sa name="さっきゅん" visible="true" version="1" author="10716288" width="192" height="96" sound="3"
        xml_data.set('name', name)
        xml_data.set('visible', 'true')
        xml_data.set('version', '1')
        xml_data.set('author', '1337')
        xml_data.set('width', '192')
        xml_data.set('height', '96')
        xml_data.set('sound', '1')
        for ldx, line in enumerate(saml_lines):
            layer = ET.SubElement(xml_data, 'layer')
            layer.set('name', 'Symbol {}'.format(ldx))
            layer.set('visible', 'true')
            layer.set('type', '{}'.format(vocab[int(line[0])]))
            color_tup = [int(x * 255) for x in line[1:4]] if values_clamped else [int(x) for x in line[1:4]]
            color_tup = rgb_to_hex(color_tup)
            layer.set('color', str(color_tup))
            alpha_val = str(max(1, int((line[4]*255)+1))) if values_clamped else str(int(line[4]))
            layer.set('alpha', alpha_val)
            positions = list(map(lambda x: str(int(((x * 254.0) - 127.0))), line[5:])) if values_clamped else list(map(lambda x: str(int(x)), line[5:]))
            layer.set('ltx', positions[0])
            layer.set('lty', positions[1])
            layer.set('lbx', positions[2])
            layer.set('lby', positions[3])
            layer.set('rtx', positions[4])
            layer.set('rty', positions[5])
            layer.set('rbx', positions[6])
            layer.set('rby', positions[7])
        f.write(ET.tostring(xml_data, pretty_print=True).decode('utf8'))


# Remember that SAML layer type values need to add 1 to them to get the cooresponding layer name
class Vocabulary(object):
    def __init__(self):
        self.layer_to_idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2}
        self.idx_to_layer = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>'}
    
    def load_layers(self, layer_path: str) -> None:
        imgs = glob.glob(os.path.join(layer_path, '[0-9]*.png'))
        for img in imgs:
            cur_len = len(self.layer_to_idx)
            idx = str(int(os.path.split(img)[-1].rstrip('.png')) - 1)
            self.layer_to_idx[idx] = cur_len
            self.idx_to_layer[cur_len] = idx
    
    def remove_item(self, idx) -> None:
        if isinstance(idx, int):
            delstr = self.idx_to_layer[idx]
            del self.idx_to_layer[idx]
            del self.layer_to_idx[delstr]
        elif isinstance(idx, str):
            delint = self.layer_to_idx[idx]
            del self.idx_to_layer[delint]
            del self.layer_to_idx[idx]
        else:
            raise ValueError('Vocabulary indices can only be strings or integers')
    
    def __len__(self):
        return len(self.layer_to_idx)
    
    def __repr__(self):
        to_return = ''
        for x in self.idx_to_layer:
            to_return += '{}: {}\n'.format(x, self.idx_to_layer[x])
        return to_return
    
    def __getitem__(self, idx):
        if isinstance(idx, str):
            if idx not in self.layer_to_idx:
                print('Adding {}'.format(idx))
                self.layer_to_idx[idx] = len(self.layer_to_idx)
                self.idx_to_layer[self.layer_to_idx[idx]] = idx
            return self.layer_to_idx[idx]
        elif isinstance(idx, int):
            return self.idx_to_layer[idx]
        else:
            raise ValueError('Vocabulary indices can only be strings or integers')
    
    def __setitem__(self, key, value):
        if isinstance(key, str) and isinstance(value, int):
            self.layer_to_idx[key] = value
            self.idx_to_layer[value] = key
        elif isinstance(key, int) and isinstance(value, str):
            self.idx_to_layer[key] = value
            self.layer_to_idx[value] = key
        else:
            raise ValueError('Vocabulary indices can only be strings or integers')
    


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def clamp_array(arr):
    # Pos bound = +/-127
    # Color bound = 0-255
    # Layer type = Ignore
    assert len(arr) == 13, 'Layer array must be of length 13'
    #assert type(arr[0]) == np.float32, 'Array values must all be of type float'
    arr[1:5] = [(x - 1.0) / 255.0 for x in arr[1:5]]
    arr[5:] = [(x + 127.0)/254.0 for x in arr[5:]]
    return arr



def convert_saml(saml_path: str, vocab: Vocabulary, verbose: bool = False, max_length = 225, clamp_values: bool = False, reverse = False) -> np.ndarray:
    with open(saml_path, 'r', encoding='utf-8-sig') as f:
        all_lines = [x for x in f.readlines()]
        _ = all_lines.pop(1) # This isn't valid XML so scrap it
    root = ETO.fromstring(''.join(all_lines))
    sos_line = [vocab['<SOS>']] + [0] * 12
    eos_line = [vocab['<EOS>']] + [0] * 12
    pad_line = [vocab['<PAD>']] + [0] * 12
    saml_lines = []
    saml_mask = []
    if reverse:
        saml_lines.append(eos_line)
    else:
        saml_lines.append(sos_line)
    saml_mask.append(True)
    max_length += 2 # We are adding the SOS and EOS tokens
    for ldx, layer in enumerate(root):
        attribs = layer.attrib
        layer_type = attribs['type']
        color_tup = hex_to_rgb(attribs['color'])
        alpha = attribs['alpha']
        ltx, lty, lbx, lby = attribs['ltx'], attribs['lty'], attribs['lbx'], attribs['lby']
        rtx, rty, rbx, rby = attribs['rtx'], attribs['rty'], attribs['rbx'], attribs['rby']
        if attribs['visible'] == 'true':
            cur_line = list(map(float, [vocab[layer_type], *color_tup, alpha, ltx, lty, lbx, lby, rtx, rty, rbx, rby]))
            if clamp_values:
                saml_lines.append(np.asarray(clamp_array(cur_line), dtype=np.float32))
            else:
                saml_lines.append(np.asarray(cur_line, dtype=np.float32))
            saml_mask.append(True)
        if verbose:
            print('Layer #{}'.format(ldx+1))
            print('\tType: {}'.format(layer_type))
            print('\tColor: {}'.format(tuple(color_tup)))
            print('\tAlpha: {}'.format(alpha))
            print('\tLeft Coords: {},{}'.format((ltx, lty),(lbx, lby)))
            print('\tRight Coords: {},{}'.format((rtx, rty),(rbx, rby)))
    if reverse:
        saml_lines.append(sos_line)
        saml_lines.reverse()
    else:
        saml_lines.append(eos_line)
    saml_mask.append(True)
    while len(saml_lines) < max_length:
        saml_lines.append(pad_line)
        saml_mask.append(False)
    return np.asarray(saml_lines, dtype=np.float32), np.asarray(saml_mask, dtype=np.bool)

def load_data(dpath = None, should_reverse=False, clamp_values=False) -> Tuple[Vocabulary, list]:
    vocab = Vocabulary()
    print('Reversing SAMLs' if should_reverse else 'SAMLs in place')
    if dpath is None:
        all_samls = glob.glob(os.path.join('..','data','BetterSymbolArts','processed','*.saml'))
    else:
        all_samls = glob.glob(os.path.join(dpath, '*.saml'))
    data = []
    for x in all_samls:
        print('Working on {}'.format(x))
        img_path = x[:-5] + '.png'
        converted, mask = convert_saml(x, vocab, reverse = should_reverse, clamp_values=clamp_values)
        data.append({'feature': img_path, 'label': converted, 'mask': mask})
    return vocab, data


# Loss scaling functions

def linear_decay(min_val, max_val, max_t, t) -> float:
    return min_val + (max_val - min_val) / (max_t - t) if t < max_t else max_val

def piecewise_decay(time_val_pairings, t) -> float:
    vals_sorted = sorted(time_val_pairings, key=lambda x: x[0])
    if len(vals_sorted) == 1:
        return vals_sorted[0][1]
    for idx in range(len(vals_sorted)-1):
        if vals_sorted[idx][1] <= t and t <= vals_sorted[idx+1][1]:
            return vals_sorted[idx][1]
    return vals_sorted[-1][1]

def load_image(img_path, image_size=None):
    feature = io.imread(img_path)[:,:,:3].astype(np.float32) / 255.
    feature = torch.from_numpy(feature.transpose((2, 0, 1)).astype(np.float32))
    if image_size is not None:
        feature = VF.resize(feature, (image_size,image_size))
    return feature