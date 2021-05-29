import pickle

import numpy as np
import skimage.io as io
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from torchvision.utils import save_image
from model.datasets import SADataset
from model.utils import get_parameter_count, load_data
from model.vae import VanillaVAE, BetaVAE
from model.autoencoder import Autoencoder, ResidualAutoencoder

def main():
    dim = 64
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA available')
    else:
        device = torch.device('cpu')
        print('CUDA not available')
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    model = ResidualAutoencoder(192, 3, dim, layer_count=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    _, data = load_data()
    dataset = SADataset(data, img_size=192)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    trainable, untrainable = get_parameter_count(model)
    print(model)
    print('Parameter count:\n\tTrainable: {}\n\tUntrainable: {}'.format(trainable, untrainable))
    
    resize = Resize((192, 192))

    epochs = 10000
    patience = 0
    best_loss = None
    best_model = None
    
    for edx in range(epochs):
        running_loss = 0.0
        model.train()
        for bdx, i_batch in enumerate(dataloader):
            feature = i_batch['feature'].to(device)
            output = model(feature)
            optimizer.zero_grad()
            loss = model.loss_function(*output, M_N = 4 / len(dataset))['loss']
            loss.backward()
            optimizer.step()
            print('\tBatch {}, Loss: {}'.format(bdx, loss.item()))

            running_loss += loss.item()
        if best_loss is None or running_loss < best_loss:
            best_loss = running_loss
            best_model = model.state_dict()
            patience = 0
        else:
            patience += 1
        print('Epoch #{}, Loss: {}, Patience: {}'.format(edx, running_loss, patience))
        if patience > 50:
            print('Out of patience')
            break
        model.eval()
        if edx % 2 == 0:
            feature = io.imread('PleaseWorkHard.png')[:,:,:3].astype(np.float32) / 255.
            feature = torch.from_numpy(feature.transpose((2, 0, 1))).to(device)
            feature = resize(feature)
            output = model.generate(feature.unsqueeze(0))[0].permute(1,2,0)
            output = output.detach().cpu().numpy() * 255.
            io.imsave('recon_hard.png', output.astype(np.uint8))
        else:
            feature = io.imread('PleaseWork.png')[:,:,:3].astype(np.float32) / 255.
            feature = torch.from_numpy(feature.transpose((2, 0, 1))).to(device)
            feature = resize(feature)
            output = model.generate(feature.unsqueeze(0))[0].permute(1,2,0)
            output = output.detach().cpu().numpy() * 255.
            io.imsave('recon_easy.png', output.astype(np.uint8))

    model.load_state_dict(best_model)
    model.eval()
    torch.save(model, 'vae.pt')
    feature = io.imread('PleaseWork.png')[:,:,:3].astype(np.float32) / 255.
    feature = torch.from_numpy(feature.transpose((2, 0, 1))).to(device)
    feature = resize(feature)
    output = model.generate(feature.unsqueeze(0))[0].permute(1,2,0)
    output = output.detach().cpu().numpy() * 255.
    io.imsave('recon_easy.png', output.astype(np.uint8))
    feature = io.imread('PleaseWorkHard.png')[:,:,:3].astype(np.float32) / 255.
    feature = torch.from_numpy(feature.transpose((2, 0, 1))).to(device)
    feature = resize(feature)
    output = model.generate(feature.unsqueeze(0))[0].permute(1,2,0)
    output = output.detach().cpu().numpy() * 255.
    io.imsave('recon_hard.png', output.astype(np.uint8))

if __name__ == '__main__':
    main()