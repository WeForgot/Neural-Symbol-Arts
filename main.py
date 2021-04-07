import os
import pickle

from dotenv import load_dotenv
import numpy as np
from skimage import io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.model import BasicNSA
from models.datasets import SADataset
from models.utils import get_parameter_count

load_dotenv()

def load_image_infer(im_path):
    feature = io.imread(im_path)[:,:,:3]
    return torch.from_numpy(feature.transpose((2, 0, 1)).astype(np.float32))


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA available')
    else:
        device = torch.device('cpu')
        print('CUDA not available')
    if not os.path.exists('data.pkl'):
        print('No support (yet) for unpickled data. Please use pre_pickle_samls.py to process ahead of time')
        return
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
    base_path = os.path.join('.','data','BetterSymbolArts','processed')
    dataset = SADataset(base_path, data)
    row_length = len(dataset[0]['label'][0])
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = BasicNSA(
        row_len=row_length,
        dim= int(os.getenv('DIM', 128)),
        patch_size= int(os.getenv('PATCH_SIZE', 32)),
        e_depth = int(os.getenv('E_DEPTH', 6)),
        e_heads = int(os.getenv('E_HEADS', 8)),
        d_depth = int(os.getenv('D_DEPTH', 6)),
        d_heads = int(os.getenv('D_HEADS', 8)),
        emb_dropout= float(os.getenv('EMB_DROP', 0.1))
    ).to(device)
    print(model)
    trainable, untrainable = get_parameter_count(model)
    print('Total paramters\n\tTrainable:\t{}\n\tUntrainable:\t{}'.format(trainable, untrainable))
    opt = optim.AdamW(model.parameters(), lr=3e-4)
    lf = nn.L1Loss()
    max_epochs = int(os.getenv('EPOCHS', 100))
    best_loss = None
    best_model = None
    batch_metrics = True
    max_patience = int(os.getenv('MAX_PATIENCE', 5))
    cur_patience = 0
    criteria = 'sum'
    if criteria == 'sum':
        loss_func = lambda x: sum(x)
    elif criteria == 'mean':
        loss_func = lambda x: sum(x) / len(x)
    else:
        loss_func = lambda x: max(x)
    with open('train_metrics.csv', 'w') as f:
        for edx in range(max_epochs):
            losses = []
            for idx, i_batch in enumerate(dataloader):
                feature, label, mask = i_batch['feature'], i_batch['label'], i_batch['mask']
                preds = model(feature.to(device), label.to(device), mask.to(device))
                loss = lf(preds, label.to(device))
                if np.isnan(loss.item()):
                    print('\tNaN loss. Breaking')
                    return
                losses.append(loss.item())
                opt.zero_grad()
                loss.backward()
                opt.step()
                if batch_metrics:
                    print('\tBatch #{}, Loss: {}'.format(idx, loss.item()))
            loss_val = loss_func(losses)
            f.write('{},{}\n'.format(edx, loss_val))
            f.flush()
            print('Epoch #{}, Loss: {}'.format(edx, loss_val))
            if best_loss is None or loss_val < best_loss:
                best_loss = loss_val
                best_model = model.state_dict()
                cur_patience = 0
            else:
                cur_patience += 1
            if cur_patience > max_patience:
                print('Out of patience. Breaking')
                break
    out = model.generate(load_image_infer('PleaseWork.png').to(device), 10)
    out = out.to('cpu')
    np.save('testing.npy', out[0].numpy())
    print(out)
    model.load_state_dict(best_model)
    model.eval()
    torch.save(model.state_dict(), 'better_trained.pt')
    

if __name__ == '__main__':
    main()