import torch
import torch.nn as nn

from einops.layers.torch import Rearrange

from axial_attention import AxialAttention
from axial_positional_embedding import AxialPositionalEmbedding
from linear_attention_transformer import LinearAttentionTransformer


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
    def __init__(self, dim, patch_size, channels=3):
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
        self.axial1 = AxialPositionalEmbedding(dim = dim, axial_shape = (8,8))
        self.axial2 = AxialPositionalEmbedding(dim = dim, axial_shape = (8,8))
        self.axial3 = AxialPositionalEmbedding(dim = dim, axial_shape = (8,8))
        self.axial4 = AxialPositionalEmbedding(dim = dim, axial_shape = (8,8))

        self.a_axial1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model = dim, nhead = 8, activation='gelu'
            ), num_layers=1
        )
        self.a_axial2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model = dim, nhead = 8, activation='gelu'
            ), num_layers=1
        )
        self.a_axial3 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model = dim, nhead = 8, activation='gelu'
            ), num_layers=1
        )
        self.a_axial4 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model = dim, nhead = 8, activation='gelu'
            ), num_layers=3
        )


        self.bn1 = nn.BatchNorm1d(num_features = 36)
        self.bn2 = nn.BatchNorm1d(num_features = 36)
        self.bn3 = nn.BatchNorm1d(num_features = 36)

        self.embedding = nn.Linear(in_features = dim, out_features = dim, bias = False)

    
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
        x = self.a_axial1(x)
        y = self.a_axial2(y)
        z = self.a_axial3(z)
        x = self.bn1(x)
        y = self.bn2(y)
        z = self.bn3(z)
        out = torch.cat([x,y,z], dim=-1)
        out = self.axial4(out)
        out = self.a_axial4(out)
        return out

class ConvBNActivation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=False):
        super(ConvBNActivation, self).__init__()
        if out_channels == None:
            out_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, bias=bias)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.act1 = nn.Hardswish()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, kernel_size=3, squeeze_ratio=4, bias=False):
        super(SqueezeExcitation, self).__init__()
        out_channels = int(in_channels/squeeze_ratio)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias, padding='same')
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=kernel_size, bias=bias, padding='same')
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x

class SimpleConv(nn.Module):
    def __init__(self, dim, channels=3):
        super(SimpleConv, self).__init__()
        self.layers = nn.Sequential(
            ConvBNActivation(in_channels=channels, out_channels=8, kernel_size=5, stride=3),
            SqueezeExcitation(in_channels=8),
            ConvBNActivation(in_channels=8, out_channels=16, kernel_size=3, stride=3),
            SqueezeExcitation(in_channels=16),
            ConvBNActivation(in_channels=16, out_channels=32, kernel_size=3, stride=3),
            SqueezeExcitation(in_channels=32),
            ConvBNActivation(in_channels=32, out_channels=dim, kernel_size=3, stride=3),
            SqueezeExcitation(in_channels=dim),
            Rearrange('b c h w -> b (h w) c')
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x