from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from x_transformers import ViTransformerWrapper, ContinuousTransformerWrapper, Encoder, Decoder

from einops import rearrange
class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding = 'same', stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

def conv2d(in_channels, out_channels, kernel_size, stride, padding='same', p=0.0):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels,out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(num_features=out_channels),
        nn.ReLU(),
        nn.Dropout2d(p=p)
    )

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, downsample=False, bias=False):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, bias=bias, padding='same')
        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.act1 = nn.Hardswish()
        self.bn2 = nn.BatchNorm2d(num_features=in_channels)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, bias=bias) if downsample else nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same', bias=bias)
        
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act1(y)
        x = self.bn2(x + y)
        return self.conv2(x)

class InverseResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsample=False, bias=False):
        super(InverseResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, bias=bias, padding='same')
        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.act1 = nn.Hardswish()
        self.bn2 = nn.BatchNorm2d(num_features=in_channels)
        self.conv2 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, bias=bias) if upsample else nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding='same', bias=bias)
        
    
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.act1(y)
        x = self.bn2(x + y)
        return self.conv2(x)

class EmbeddingConvNet(nn.Module):
    def __init__(self, channels=3, classes=5):
        super(EmbeddingConvNet, self).__init__()
        self.encoder = nn.Sequential(
            conv2d(channels, 8, 9, 1),
            conv2d(8, 8, 7, 1),
            conv2d(8, 16, 7, 1),
            conv2d(16, 16, 5, 1),
            conv2d(16, 32, 5, 1),
            conv2d(32, 32, 5, 1),
            conv2d(32, 64, 3, 1),
            conv2d(64, 64, 3, 1),
            DepthWiseConv2d(dim_in = 64, dim_out = classes, kernel_size = 1, bias = False)
        )

        self.decoder = nn.Sequential(
            conv2d(classes, 64, 1, 1),
            conv2d(64, 64, 3, 1),
            conv2d(64, 32, 3, 1),
            conv2d(32, 32, 5, 1),
            conv2d(32, 16, 5, 1),
            conv2d(16, 16, 5, 1),
            conv2d(16, 8, 7, 1),
            conv2d(8, 8, 7, 1),
            DepthWiseConv2d(dim_in = 8, dim_out = channels, kernel_size = 9, bias = False)
        )


    
    def forward(self, x):
        y = self.encoder(x)
        z = self.decoder(y)
        return y, z

class EmbeddingResnet(nn.Module):
    def __init__(self, channels=3, classes=5):
        super(EmbeddingResnet, self).__init__()
        self.encoder = nn.Sequential(OrderedDict([
            ('res1', ResBlock(channels, 8, 7)),
            ('dep1', DepthWiseConv2d(8, 8, 7, bias=False)),
            ('res2', ResBlock(8, 16, 5)),
            ('dep2', DepthWiseConv2d(16, 16, 5, bias=False)),
            ('res3', ResBlock(16, 32, 3)),
            ('dep3', DepthWiseConv2d(32, 32, kernel_size=3, bias=False)),
            ('res4', ResBlock(32, 64, 3)),
            ('dep4', DepthWiseConv2d(64, classes, 1, bias=False))
        ]))

        self.decoder = nn.Sequential(OrderedDict([
            ('dep5', DepthWiseConv2d(classes, 64, 1, bias=False)),
            ('res5', ResBlock(64, 32, 3)),
            ('dep6', DepthWiseConv2d(32, 32, 3, bias=False)),
            ('res6', ResBlock(32, 16, 3)),
            ('dep7', DepthWiseConv2d(16, 16, 5, bias=False)),
            ('res7', ResBlock(16, 8, 5)),
            ('dep8', DepthWiseConv2d(8, 8, 7, bias=False)),
            ('res8', ResBlock(8, channels, 7))
        ]))
    
    def forward(self, x):
        y = self.encoder(x)
        z = self.decoder(y)
        return y, z