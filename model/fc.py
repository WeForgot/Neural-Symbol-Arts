import torch
import torch.nn as nn

from einops.layers.torch import Rearrange

from axial_positional_embedding import AxialPositionalEmbedding


class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding = 0, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

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

class FCModel(nn.Module):
    def __init__(self, patch_size, channels=3):
        super(FCModel, self).__init__()
        self.block1 = nn.Sequential(
            DepthWiseConv2d(dim_in = channels, dim_out = 8, kernel_size = 5, padding='same', bias = False),
            ChannelSELayer(num_channels = 8, reduction_ratio = 2),
            nn.BatchNorm2d(num_features = 8),
            nn.Hardswish(),
            nn.Dropout2d(p = 0.2)
        )

        self.block2 = nn.Sequential(
            DepthWiseConv2d(dim_in = 8, dim_out = 16, kernel_size = 5, padding='same', bias = False),
            ChannelSELayer(num_channels = 16, reduction_ratio = 2),
            nn.BatchNorm2d(num_features = 16),
            nn.Hardswish(),
            nn.Dropout2d(p = 0.2)
        )

        self.block3 = nn.Sequential(
            DepthWiseConv2d(dim_in = 16, dim_out = 32, kernel_size = 5, padding='same', bias = False),
            ChannelSELayer(num_channels = 32, reduction_ratio = 2),
            nn.BatchNorm2d(num_features = 32),
            nn.Hardswish(),
            nn.Dropout2d(p = 0.2)
        )

        self.to_patch = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)
        self.axial1 = AxialPositionalEmbedding(dim = 512, axial_shape = (32,32))
        self.axial2 = AxialPositionalEmbedding(dim = 512, axial_shape = (32,32))
        self.axial3 = AxialPositionalEmbedding(dim = 512, axial_shape = (32,32))
        self.axial4 = AxialPositionalEmbedding(dim = 512, axial_shape = (32,32))

        self.bn1 = nn.BatchNorm1d(num_features = 36)
        self.bn2 = nn.BatchNorm1d(num_features = 36)
        self.bn3 = nn.BatchNorm1d(num_features = 36)

    
    def forward(self, w):
        x = self.block1(w)
        y = self.block2(x)
        z = self.block3(y)
        x = self.to_patch(x)
        y = self.to_patch(y)
        z = self.to_patch(z)
        x = self.axial1(x)
        y = self.axial2(y)
        z = self.axial3(z)
        x = self.bn1(x)
        y = self.bn2(y)
        z = self.bn3(z)
        out = torch.cat([x,y,z], dim=-1)
        out = self.axial4(out)
        return out