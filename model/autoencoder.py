import random

import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F

class DepthwiseSeperableConv(nn.Module):
    def __init__(self, nin, kernels_per_layer, nout):
        super(DepthwiseSeperableConv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=5, padding=2, groups=nin)
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# SE = Squeeze-Excitation
class ChannelSELayer(nn.Module):
    def __init__(self, num_channels, reduction_ratio=2):
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        batch_size, num_channels, h, w = x.size()
        squeeze_tensor = x.view(batch_size, num_channels, -1).mean(dim=2)
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(x, fc_out_2.view(a, b, 1, 1))
        return output_tensor

class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, squeeze_factor=1, bias=False):
        super(ResidualBlock, self).__init__()
        squeezed = int(channels / squeeze_factor)
        self.block = nn.Sequential(
            nn.BatchNorm2d(num_features=channels),
            nn.Hardswish(),
            nn.Conv2d(in_channels=channels, out_channels=squeezed, kernel_size=kernel_size, bias=bias, padding='same'),
            nn.BatchNorm2d(num_features=squeezed),
            nn.ReLU(),
            nn.Conv2d(in_channels=squeezed, out_channels=channels, kernel_size=kernel_size, bias=bias, padding='same'),
        )
    
    def forward(self, x):
        return x + self.block(x)

class Permute(nn.Module):
    def __init__(self, *args):
        super(Permute, self).__init__()
        self.permutation = [0, *args]
    
    def forward(self, x):
        return x.permute(*self.permutation)
    


class Autoencoder(nn.Module):
    def __init__(self, dim, channels=3, p=0.2):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=8, kernel_size=5, stride=2), # torch.Size([4, 8, 142, 286])
            nn.BatchNorm2d(num_features=8),
            nn.Dropout2d(p=p),
            nn.Hardswish(),
            DepthwiseSeperableConv(nin=8, kernels_per_layer=1, nout=16), # torch.Size([4, 16, 142, 286])
            nn.BatchNorm2d(num_features=16),
            nn.Dropout2d(p=p),
            nn.Hardswish(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2), # torch.Size([4, 16, 70, 142])
            nn.BatchNorm2d(num_features=16),
            nn.Dropout2d(p=p),
            nn.Hardswish(),
            DepthwiseSeperableConv(nin=16, kernels_per_layer=1, nout=32), # torch.Size([4, 32, 70, 142])
            nn.BatchNorm2d(num_features=32),
            nn.Dropout2d(p=p),
            nn.Hardswish(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2), # torch.Size([4, 32, 34, 70])
            nn.BatchNorm2d(num_features=32),
            nn.Dropout2d(p=p),
            nn.Hardswish(),
            nn.Flatten(start_dim=1, end_dim=2),
            nn.Linear(in_features=22, out_features=dim),
            nn.Tanh()
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=dim, out_features=22),
            nn.Unflatten(1, (32, 22)),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2),
            nn.BatchNorm2d(num_features=32),
            nn.Dropout2d(p=p),
            nn.Hardswish(),
            DepthwiseSeperableConv(nin=32, kernels_per_layer=3, nout=16), # torch.Size([4, 32, 70, 142])
            nn.BatchNorm2d(num_features=16),
            nn.Dropout2d(p=p),
            nn.Hardswish(),
            nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=4, stride=2),
            nn.BatchNorm2d(num_features=16),
            nn.Dropout2d(p=p),
            nn.Hardswish(),
            DepthwiseSeperableConv(nin=16, kernels_per_layer=3, nout=8), # torch.Size([4, 32, 70, 142])
            nn.BatchNorm2d(num_features=8),
            nn.Dropout2d(p=p),
            nn.Hardswish(),
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=6, stride=2),
            nn.BatchNorm2d(num_features=8),
            nn.Dropout2d(p=p),
            nn.Hardswish(),
            nn.ConvTranspose2d(in_channels=8, out_channels=channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    
    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return x, latent, recon
    
    def encode(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        return self.encoder(x)[0]
    
    def generate(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        return self.forward(x)[2]
    
    def loss_function(self, *args, **kwargs):
        return {'loss': F.mse_loss(args[0], args[2])}


class ResidualAutoencoder(nn.Module):
    def __init__(self, img_size, channels, dim, layer_count=3, kernel_size=3, p=0.2):
        super(ResidualAutoencoder, self).__init__()
        filters = 8
        padding = 0
        dilation = 1
        kernel_choices = [3, 5, 7]
        img_size = [img_size]
        layers = [DepthwiseSeperableConv(channels, 1, filters)]
        for _ in range(layer_count):
            kernel_size = random.choice(kernel_choices)
            layers.append(ResidualBlock(filters, kernel_size, 2))
            layers.append(nn.BatchNorm2d(filters))
            layers.append(nn.Hardswish())
            layers.append(nn.Dropout2d(p=p))
            layers.append(nn.Conv2d(in_channels=filters, out_channels=filters*2, kernel_size=kernel_size, stride=2, bias=False))
            filters *= 2
            img_size.append(int(((img_size[-1] + 2 * padding - dilation * (kernel_size - 1) - 1) / 2) + 1))
        layers.append(nn.Flatten(start_dim=2, end_dim=3))
        layers.append(nn.Linear(in_features=img_size[-1]**2, out_features=dim))
        self.encoder = nn.Sequential(*layers)

        layers = [nn.Linear(in_features=dim, out_features=img_size[-1]**2)]
        layers.append(nn.Unflatten(2, (img_size[-1], img_size[-1])))
        for _ in range(layer_count):
            img_temp = img_size.pop(-1)
            kernel_target = img_size[-1] - (img_temp - 1) * 2
            layers.append(ResidualBlock(filters, kernel_size, 2))
            layers.append(nn.BatchNorm2d(filters))
            layers.append(nn.Hardswish())
            layers.append(nn.ConvTranspose2d(in_channels=filters, out_channels=int(filters/2), kernel_size=kernel_target, stride=2, bias=False))
            filters = int(filters/2)
        layers.append(DepthwiseSeperableConv(filters, 1, channels))
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return x, latent, recon
    
    def encode(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        return self.encoder(x)[0]
    
    def decode(self, x):
        return x
    
    def generate(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        return self.forward(x)[2]
    
    def loss_function(self, *args, **kwargs):
        return {'loss': F.mse_loss(args[0], args[2])}