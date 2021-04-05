from collections import namedtuple

import numpy as np

class WeightPair(object):
    def __init__(self, weights = None, name = None):
        self.weight = weights
        self.name = name
    
    def __len__(self):
        return len(self.weight)

    def __repr__(self):
        return 'WeightPair\n\tWeight: {}\n\tName: {}'.format(self.weight, self.name)

def load_weights(weight_fp, names_fp):
    weights = []
    with open(weight_fp, 'r') as wf, open(names_fp, 'r') as rf:
        for x, y in zip(rf, wf):
            weights.append(WeightPair(weights=np.asarray(list(map(float, y.strip().split('\t')))), name=int(x.strip().rstrip('.png'))))
    return weights


def decode(inpt, embs, decoding_type='greedy'):
    assert isinstance(inpt, np.ndarray), 'Inputs must be an ndarray (of layer weights)'
    assert isinstance(embs, list), 'Embeddings must be in a list format'
    assert isinstance(embs[0], WeightPair), "Elements of embeddings must be WeightPair's"

    print(inpt.shape)

def write_saml(inpt):
    pass

if __name__ == '__main__':
    embs = load_weights('weights.tsv', 'names.tsv')
    print(len(embs[0].weight))
    inpt = np.load('testing.npy')
    decode(inpt, embs)