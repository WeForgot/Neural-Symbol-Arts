import glob
import os
import pickle
import xml.etree.ElementTree as ET

import numpy as np
from webcolors import hex_to_rgb

from model.utils import load_weights, load_saml, Vocabulary

def convert_saml(saml_path: str, vocab: Vocabulary, verbose: bool = False, max_length = 225) -> np.ndarray:
    with open(saml_path, 'r', encoding='utf-8-sig') as f:
        all_lines = [x for x in f.readlines()]
        _ = all_lines.pop(1) # This isn't valid XML so scrap it
    root = ET.fromstring(''.join(all_lines))
    sos_line = [vocab['<SOS>']] + [0] * 12
    eos_line = [vocab['<EOS>']] + [0] * 12
    pad_line = [vocab['<PAD>']] + [0] * 12
    saml_lines = []
    saml_mask = []
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
            saml_lines.append(list(map(int, [vocab[layer_type], *color_tup, alpha, ltx, lty, lbx, lby, rtx, rty, rbx, rby])))
            saml_mask.append(True)
        if verbose:
            print('Layer #{}'.format(ldx+1))
            print('\tType: {}'.format(layer_type))
            print('\tColor: {}'.format(tuple(color_tup)))
            print('\tAlpha: {}'.format(alpha))
            print('\tLeft Coords: {},{}'.format((ltx, lty),(lbx, lby)))
            print('\tRight Coords: {},{}'.format((rtx, rty),(rbx, rby)))
    saml_lines.append(eos_line)
    saml_mask.append(True)
    while len(saml_lines) < max_length:
        saml_lines.append(pad_line)
        saml_mask.append(False)
    return np.asarray(saml_lines, dtype=np.int16), np.asarray(saml_mask, dtype=np.bool)

# 386 layers + start token + pad token = 388 vocab size
def main():
    if not os.path.exists('vocab.pkl'):
        print('Creating new vocab')
        vocab = Vocabulary()
        vocab.load_layers(os.path.join('data','Layers'))
        with open('vocab.pkl', 'wb') as f:
            pickle.dump(vocab, f)
    else:
        print('Loading existing vocab')
        with open('vocab.pkl', 'rb') as f:
            vocab = pickle.load(f)
    all_samls = glob.glob(os.path.join('data','BetterSymbolArts','processed','*.saml'))
    all_data = []
    for x in all_samls:
        img_path = x[:-5] + '.png'
        converted, mask = convert_saml(x, vocab)
        all_data.append({'feature': img_path, 'label': converted, 'mask': mask})
    with open('data.pkl', 'wb') as f:
        pickle.dump(all_data, f)

if __name__ == '__main__':
    main()