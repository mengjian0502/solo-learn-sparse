"""
Sparse convolution
"""

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
        # self.mask = self.__getattr__(f'mask{n}')
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

class SparsLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, nspars:int=4, uspars:bool=False):
        super().__init__(in_features, out_features, bias)
        self.uspars = uspars
        self.nspars = nspars

        if self.uspars:
            for n in range(nspars):
                self.register_buffer(f'mask{n}', torch.ones_like(self.weight))
        else:
            self.register_buffer('mask', torch.ones_like(self.weight))

        # initialize weight mask
        self._switch(0)

    def _switch(self, n):
        self.mask = self.__getattr__(f'mask{n}')
    
    def forward(self, input: Tensor) -> Tensor:
        out = F.linear(input, self.weight.mul(self.mask), self.bias)
        return out

    def extra_repr(self):
        return super(SparsLinear, self).extra_repr() + ", nspars={}, uspars={}".format(self.nspars, self.uspars)

class SparseBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features: int, eps: float = 0.00001, momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True, device=None, dtype=None):
        super().__init__(num_features, eps, momentum, affine, track_running_stats, device, dtype)
        self.prune_flag = False
        self.register_buffer("mask", torch.ones_like(self.weight))
    
    def forward(self, input: Tensor) -> Tensor:       
        return super().forward(input)


