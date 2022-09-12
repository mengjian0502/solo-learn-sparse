"""
N:M Pruner for CL method with shared encoder (e.g., SimCLR) and momentum gradient
"""

import torch
import torch.nn as nn
from solo.backbones import SparsConv2d, SparsLinear
from collections import OrderedDict
from typing import List
from .duo_nm_sparse import DuoNMMask

class DuoNMMMask(DuoNMMask):
    def __init__(self, model: nn.Module, optimizer=None, prune_rate=0.3, prune_rate_decay=None, args=None, train_steps=None, slist: List = None, Itertrain=False, N: int = 2, M: int = 4, momentum=0.99):
        super().__init__(model, optimizer, prune_rate, prune_rate_decay, args, train_steps, slist, Itertrain, N, M)

        self.momentum = momentum
        self.init_momentum_buffer()
    
    def name(self):
        return "Duo momentum N2M mask"
    
    def init_momentum_buffer(self):
        self.ema_grad = OrderedDict()

        for on, om in self.model.named_modules():
            if isinstance(om, SparsConv2d):
                self.ema_grad[on] = torch.zeros_like(om.weight.data)
    
    def get_gradient_for_weights(self, weight, name):
        g = super().get_gradient_for_weights(weight, name)
        running_g = self.ema_grad[name]
        running_g = running_g.mul(self.momentum) + (1-self.momentum) * g
        self.ema_grad[name] = running_g
        return running_g