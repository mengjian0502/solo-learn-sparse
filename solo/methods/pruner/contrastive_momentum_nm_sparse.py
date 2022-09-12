"""
Contrastive pruning with momentum score and N:M sparsity
"""

import torch
import torch.nn as nn
from solo.backbones import SparsConv2d, SparsLinear
from typing import List
from collections import OrderedDict
from .contrastive_nm_sparse import ContrastiveNM

class ContrastiveMNM(ContrastiveNM):
    def __init__(self, online_model: nn.Module, offline_model: nn.Module, optimizer=None, prune_rate=0.3, prune_rate_decay=None, args=None, train_steps=None, slist: List = None, Itertrain=False, N: int = 2, M: int = 4, momentum:float=0.99):
        super(ContrastiveMNM, self).__init__(online_model, offline_model, optimizer, prune_rate, prune_rate_decay, args, train_steps, slist, Itertrain, N, M)

        self.momentum = momentum

        # initialize momentum score
        self.init_momentum_buffer()

    def name(self):
        return "Contrastive momentum NM Mask: M={}, N={}".format(self.M, self.N)
    
    def init_momentum_buffer(self):
        self.online_ema = OrderedDict()
        self.offline_ema = OrderedDict()
        for (on, om), (mn, mm) in zip(self.model.named_modules(), self.offline_model.named_modules()):
            if isinstance(om, (SparsConv2d, SparsLinear)):
                self.online_ema[on] = om.weight.data.abs()
                self.offline_ema[mn] = mm.weight.data.abs()

    def ema(self):
        for (on, om), (mn, mm) in zip(self.model.named_modules(), self.offline_model.named_modules()):
            if isinstance(om, (SparsConv2d, SparsLinear)):
                online_ema = self.online_ema[on]
                offline_ema = self.offline_ema[mn]
                
                # ema
                online_ema = online_ema.mul(self.momentum) + (1 - self.momentum) * om.weight.data.abs()
                offline_ema = offline_ema.mul(self.momentum) + (1 - self.momentum) * mm.weight.data.abs()

                # update buffer
                self.online_ema[on] = online_ema
                self.offline_ema[on] = offline_ema
    
    def update_mask(self, prob):
        for (on, om), (mn, mm) in zip(self.model.named_modules(), self.offline_model.named_modules()):
            if isinstance(om, SparsConv2d):
                ow = self.online_ema[on]

                # get mask
                os = self.mp_grp_mask(ow, prob)
                # reshape
                os = self.mask2weight(os, ow)
                
                # update online mask
                self.masks[on] = os

                if self.args.sparse_enck:
                    mw = self.offline_ema[mn]
                    
                    # get mask
                    ms = self.mp_grp_mask(mw, prob - self.args.density_gap)
                    # reshape
                    ms = self.mask2weight(ms, mw)
                    # update offline mask
                    self.offline_masks[mn] = ms
                else:
                    self.offline_masks[mn] = torch.ones_like(os)
    
    def step(self, n, bidx):
        self.ema()
        self.prune_rate_decay.step()
        self.prune_rate = self.prune_rate_decay.get_dr()
        self.steps[n] += 1
        
        if self.iter:
            prune_step = self.steps[n] * len(self.slist) if bidx < len(self.slist) else (self.steps[n]-1) * len(self.slist)
        else:
            prune_step = self.steps[n]
    
        if prune_step >= (self.init_prune_epoch * self.train_steps*self.args.multiplier) and prune_step % self.prune_every_k_steps == 0:
            if self.args.final_density < 1.0:
                self.pruning(prune_step, bidx)
            
            # overlap
            self.online_overlap = self.overlap(self.masks, self.online_buffer)
            
            # fetch the new mask
            self._latch_mask()