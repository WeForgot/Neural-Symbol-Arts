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
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1, stride=2), # torch.Size([4, 32, 34, 70])
            nn.BatchNorm2d(num_features=32),
            nn.Dropout2d(p=p),
            nn.Hardswish(),
            nn.Flatten(start_dim=1, end_dim=2),
            Permute(2,1),
            #nn.Linear(in_features=1088, out_features=dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(dim, 1088),
            Permute(2,1),
            nn.Unflatten(1, (32, 34)),
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
        return latent, recon