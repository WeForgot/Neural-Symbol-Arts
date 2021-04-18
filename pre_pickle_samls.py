import glob
import os
import pickle
import xml.etree.ElementTree as ET

from dotenv import load_dotenv
import numpy as np
from webcolors import hex_to_rgb

from model.utils import load_weights, load_saml, Vocabulary

load_dotenv()

def clamp_array(arr):
    # Pos bound = +/-127
    # Color bound = 0-255
    # Layer type = Ignore
    assert len(arr) == 13, 'Layer array must be of length 13'
    #assert type(arr[0]) == np.float32, 'Array values must all be of type float'
    arr[1:5] = [(x - 1.0) / 255.0 for x in arr[1:5]]
    arr[5:] = [x/127.0 for x in arr[5:]]
    return arr


def convert_saml(saml_path: str, vocab: Vocabulary, verbose: bool = False, max_length = 225, clamp_values: bool = False, reverse = False) -> np.ndarray:
    with open(saml_path, 'r', encoding='utf-8-sig') as f:
        all_lines = [x for x in f.readlines()]
        _ = all_lines.pop(1) # This isn't valid XML so scrap it
    root = ET.fromstring(''.join(all_lines))
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

# 386 layers + start token + pad token = 388 vocab size
def main():
    should_reverse = True if os.getenv('REVERSE_DATA', 'false').lower() == 'true' else False
    print('Reversing SAMLs' if should_reverse else 'SAMLs in place')
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
        print('Working on {}'.format(x))
        img_path = x[:-5] + '.png'
        converted, mask = convert_saml(x, vocab, reverse = should_reverse)
        all_data.append({'feature': img_path, 'label': converted, 'mask': mask})
    with open('data.pkl', 'wb') as f:
        pickle.dump(all_data, f)

if __name__ == '__main__':
    main()