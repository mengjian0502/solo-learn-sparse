"""
Contrastive pruning with momentum-based scores
"""

import torch
import torch.nn as nn
from solo.backbones import SparsConv2d, SparsLinear
from typing import Dict, List
from collections import OrderedDict
from .contrastive_sparse import ContrastiveMask


class ContrastiveMMask(ContrastiveMask):
    def __init__(self, online_model: nn.Module, offline_model: nn.Module, optimizer=None, prune_rate=0.3, prune_rate_decay=None, args=None, train_steps=None, slist: List = None, Itertrain=False, momentum:float=0.9):
        super().__init__(online_model, offline_model, optimizer, prune_rate, prune_rate_decay, args, train_steps, slist, Itertrain)

        self.offline_model = offline_model
        self.momentum = momentum

        # initialize momentum score
        self.init_momentum_buffer()
    
    def name(self):
        return "Contrastive Momentum Mask. Momentum={:.3f}".format(self.momentum)

    def init_momentum_buffer(self):
        self.online_ema = OrderedDict()
        self.offline_ema = OrderedDict()
        for (on, om), (mn, mm) in zip(self.model.named_modules(), self.offline_model.named_modules()):
            if isinstance(om, (SparsConv2d, SparsLinear)):
                self.online_ema[on] = om.weight.data.abs().cuda()
                self.offline_ema[mn] = mm.weight.data.abs().cuda()

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
        
    def collect_score(self):
        online_weight_abs = []
        offline_weight_abs = []
        for (on, om), (mn, mm) in zip(self.model.named_modules(), self.offline_model.named_modules()):
            if isinstance(om, (SparsConv2d, SparsLinear)):
                online_ema = self.online_ema[on]
                offline_ema = self.offline_ema[mn]
                
                online_weight_abs.append(online_ema)
                offline_weight_abs.append(offline_ema)
        
        online_mp_scores = torch.cat([torch.flatten(x) for x in online_weight_abs])
        offline_mp_scores = torch.cat([torch.flatten(x) for x in offline_weight_abs])
        return online_mp_scores, offline_mp_scores

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
                self.prune_and_regrow(bidx)

            # # compute the overlap of consecutive epochs
            # self.online_overlap = self.overlap(self.masks, self.online_buffer)
            # self.offline_overlap = self.overlap(self.offline_masks, self.offline_buffer)
            # self.regrow = self.regrow_overlap()
        
            # fetch the new mask
            self._latch_mask()