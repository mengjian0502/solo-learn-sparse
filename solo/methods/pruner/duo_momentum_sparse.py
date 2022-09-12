"""
Pruner for CL method with shared encoder (e.g., SimCLR) with momentum-based score
"""

import math
import copy
import torch
import torch.nn as nn
from solo.backbones import SparsConv2d, SparsLinear
from typing import Dict, List
from collections import OrderedDict
from .duo_sparse import DuoMask

class DuoMMask(DuoMask):
    def __init__(self, model: nn.Module, optimizer=None, prune_rate=0.3, prune_rate_decay=None, args=None, train_steps=None, slist: List = None, Itertrain=False, momentum:float=0.99):
        super().__init__(model, optimizer, prune_rate, prune_rate_decay, args, train_steps, slist, Itertrain)
        self.momentum = momentum

        # initialize momentum score
        self.init_momentum_buffer()

    def name(self):
        return "Duo momentum mask"

    def init_momentum_buffer(self):
        self.online_ema = OrderedDict()
        self.ema_grad = OrderedDict()

        for on, om in self.model.named_modules():
            if isinstance(om, SparsConv2d):
                self.online_ema[on] = om.weight.data.abs()
                self.ema_grad[on] = torch.zeros_like(om.weight.data)

    def ema(self):
        for on, om in self.model.named_modules():
            if isinstance(om, SparsConv2d):
                online_ema = self.online_ema[on]

                # ema
                online_ema = online_ema.mul(self.momentum) + (1 - self.momentum) * om.weight.data.abs()

                # update buffer
                self.online_ema[on] = online_ema

    def collect_score(self):
        scores = []
        for n, m in self.model.named_modules():
            if isinstance(m, (SparsConv2d)):
                online_ema = self.online_ema[n]
                scores.append(online_ema)
        scores = torch.cat([torch.flatten(x) for x in scores])
        return scores

    def get_gradient_for_weights(self, weight, name):
        g = super().get_gradient_for_weights(weight, name)
        running_g = self.ema_grad[name]
        running_g = running_g.mul(self.momentum) + (1-self.momentum) * g
        self.ema_grad[name] = running_g
        return running_g

    def step(self, n, bidx):
        self.ema()
        self.prune_rate_decay.step()
        self.prune_rate = self.prune_rate_decay.get_dr()
        self.steps[n] += 1
        
        if self.iter:
            prune_step = self.steps[n] * len(self.slist) if bidx < len(self.slist) else (self.steps[n]-1) * len(self.slist)
        else:
            prune_step = self.steps[n]
        # print("prune step={}, n={}, {}".format(prune_step, n, prune_step % self.prune_every_k_steps))
    
        if prune_step >= (self.init_prune_epoch * self.train_steps*self.args.multiplier) and prune_step % self.prune_every_k_steps == 0:
            if self.args.final_density < 1.0:
                self.pruning(prune_step, bidx)
                self.prune_and_regrow(bidx)
                
