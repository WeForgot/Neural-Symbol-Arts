import pickle

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from model.autoencoder import Autoencoder
from model.datasets import SADataset
from model.utils import get_parameter_count

def main():
    dim = 32
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

    model = Autoencoder(dim = dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    trainable, untrainable = get_parameter_count(model)
    print('Total encoder paramters\n\tTrainable:\t{}\n\tUntrainable:\t{}'.format(trainable, untrainable))    
    valid_split = 0.1
    batch_size = 8
    epochs = 10000000
    max_patience = 20
    batch_metrics = False

    dataset = SADataset(data)
    valid_size = int(len(dataset) * valid_split)
    train_size = len(dataset) - valid_size
    train_set, valid_set = torch.utils.data.random_split(SADataset(data), [train_size, valid_size])
    train_loader, valid_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True), DataLoader(valid_set, batch_size=batch_size, drop_last=True)

    best_model = None
    best_loss = None
    cur_patience = 0

    for edx in range(epochs):
        running_loss = 0.0
        model.train()
        for bdx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            feature = batch['feature'].to(device)
            _, recon = model(feature)
            loss = F.mse_loss(recon, feature)
            scalar_loss = loss.item()
            loss.backward()
            optimizer.step()
            running_loss += scalar_loss
            if batch_metrics:
                print('\tBatch #{}, Loss: {}'.format(bdx, scalar_loss))
        print('TRAINING Epoch #{}, Total loss: {}'.format(edx, running_loss))
        
        model.eval()
        running_loss = 0.0
        for bdx, batch in enumerate(valid_loader):
            feature = batch['feature'].to(device)
            _, recon = model(feature)
            loss = F.mse_loss(recon, feature)
            running_loss += loss.item()
        
        if best_loss is None or running_loss < best_loss:
            best_model = model.state_dict()
            cur_patience = 0
            best_loss = running_loss
        else:
            cur_patience += 1
        print('VALIDATION Epoch #{}, Total loss: {}, Patience: {}/{}'.format(edx, running_loss, cur_patience, max_patience))
        if max_patience < cur_patience:
            print('Out of patience, breaking')
            break
    model.load_state_dict(best_model)
    torch.save(model.encoder, 'best_encoder_{}.pt'.format(dim))
    # Might change this if it is beneficial to add an auxilary reconstruction loss into the final end to end model
    #torch.save(model.decoder, 'best_decoder_{}.pt'.format(dim))
    #torch.save(model, 'best_autoencoder_{}.pt'.format(dim))

            


if __name__ == '__main__':
    main()