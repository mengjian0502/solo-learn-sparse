"""
Pruning with momentum score
"""

import copy
import torch
import torch.nn as nn
from solo.backbones import SparsConv2d, SparsLinear
from typing import Dict, List
from collections import OrderedDict
from .sparse import Mask

class MMask(Mask):
    def __init__(self, model: nn.Module, optimizer=None, prune_rate=0.3, prune_rate_decay=None, args=None, train_steps=None, slist: List = None, Itertrain=False, momentum=0.99):
        super().__init__(model, optimizer, prune_rate, prune_rate_decay, args, train_steps, slist, Itertrain)
        # momentum
        self.momentum = momentum

    def name(self):
        return "Momentum mask"

    def init_momentum_buffer(self):
        self.online_ema = OrderedDict()
        for (on, om) in self.model.named_modules():
            if isinstance(om, SparsConv2d):
                self.online_ema[on] = om.weight.data.abs()
    
    def ema(self):
        for (on, om) in self.model.named_modules():
            online_ema = self.online_ema[on]

            # ema
            online_ema = online_ema.mul(self.momentum) + (1 - self.momentum) * om.weight.data.abs()

            # update buffer
            self.online_ema[on] = online_ema

    