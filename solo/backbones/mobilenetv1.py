import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SparsConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size, stride = 1, padding=0, dilation=1, groups: int = 1, bias: bool = True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.prune_flag = False
        self.register_buffer('mask', torch.ones_like(self.weight).cuda())
        
    def _switch(self, n):
        self.mask = self.mask

    def forward(self, input: Tensor) -> Tensor:
        if self.prune_flag:
            weight = self.weight.mul(self.mask)
        else:
            weight = self.weight
        
        out = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out
    
    def extra_repr(self):
        return super(SparsConv2d, self).extra_repr() + ", prune={}".format(self.prune_flag)

class Net(nn.Module):
    def __init__(self, alpha:float=1.0):
        super(Net, self).__init__()
        self.alpha = alpha
        self.last_channel = int(1024*alpha)

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                SparsConv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                SparsConv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                SparsConv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, int(32*alpha), 2), 
            conv_dw(int(32*alpha),  int(64*alpha), 1),
            conv_dw(int(64*alpha), int(128*alpha), 2),
            conv_dw(int(128*alpha), int(128*alpha), 1),
            conv_dw(int(128*alpha), int(256*alpha), 2),
            conv_dw(int(256*alpha), int(256*alpha), 1),
            conv_dw(int(256*alpha), int(512*alpha), 2),
            conv_dw(int(512*alpha), int(512*alpha), 1),
            conv_dw(int(512*alpha), int(512*alpha), 1),
            conv_dw(int(512*alpha), int(512*alpha), 1),
            conv_dw(int(512*alpha), int(512*alpha), 1),
            conv_dw(int(512*alpha), int(512*alpha), 1),
            conv_dw(int(512*alpha), int(1024*alpha), 2),
            conv_dw(int(1024*alpha), int(1024*alpha), 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(int(1024*alpha), 100)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, int(1024*self.alpha))
        x = self.fc(x)
        return x

def mobilenet_v1(method, *args, **kwargs):
    model = Net(alpha=1)
    return model

def mobilenetv1_2x(method, *args, **kwargs):
    model = Net(alpha=2.0)
    return model

def mobilenetv1_4x(method, *args, **kwargs):
    model = Net(alpha=4.0)
    return model