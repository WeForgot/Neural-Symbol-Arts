import glob
import os
import pickle

import numpy as np

from models.utils import load_weights, load_saml

def main():
    weights = load_weights('weights.tsv', 'names.tsv')
    base_path = os.path.join('data','BetterSymbolArts','processed')
    all_samls = glob.glob(os.path.join(base_path, '*.saml'))
    all_data = []
    for saml in all_samls:
        saml_name = os.path.split(saml)[-1][:-5]
        saml_np, mask_np = load_saml(saml, weights)
        all_data.append({'feature': saml_name+'.png', 'label': saml_np, 'mask': mask_np})
    with open('data.pkl', 'wb') as f:
        pickle.dump(all_data, f)


if __name__ == '__main__':
    main()