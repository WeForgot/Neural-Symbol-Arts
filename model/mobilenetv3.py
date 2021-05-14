import torch
import torch.nn as nn
from torchvision.models.mobilenetv3 import mobilenet_v3_small

class MyMobileNetV3(nn.Module):
    def __init__(self, dim):
        super(MyMobileNetV3, self).__init__()
        self.net = mobilenet_v3_small()
        self.net.avgpool = nn.AdaptiveAvgPool2d((100, dim))
    
    def forward(self, x):
        x = self.net.features(x)
        x = torch.flatten(x, 2, 3)
        x = self.net.avgpool(x)
        return x