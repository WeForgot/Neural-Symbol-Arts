import os
import glob
import lxml.etree as ET

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from webcolors import hex_to_rgb, rgb_to_hex

from .datasets import SADataset, LayersDataset, RandomTransform, ToTensor

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

def convert_numpy_to_saml(source_path, vocab, dest_path=None, values_clamped=False):
    if dest_path is None:
        dest_path = source_path[:-3] + 'saml'
    
    with open(dest_path, 'w') as f:
        f.write('<?xml version="1.0" encoding="utf-8"?>\n')
        data = np.load(source_path)
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
        xml_data.set('name', 'Test')
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
            alpha_val = str(max(1, int(line[4]*255))) if values_clamped else str(line[4])
            layer.set('alpha', alpha_val)
            positions = list(map(lambda x: str(int(x * 127)), line[5:])) if values_clamped else list(map(lambda x: str(int(x)), line[5:]))
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
