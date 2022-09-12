"""
Contrastive pruning with double regrow
"""

import copy
import torch
import torch.nn as nn
from solo.backbones import SparsConv2d, SparsLinear
from typing import Dict, List
from .contrastive_sparse import ContrastiveMask
from .utils import AverageMeter
from collections import OrderedDict

class ContrastiveRegMask(ContrastiveMask):
    def __init__(self, online_model: nn.Module, offline_model: nn.Module, optimizer=None, prune_rate=0.3, prune_rate_decay=None, args=None, train_steps=None, slist: List = None, Itertrain=False, g_momentum=0.99):
        super().__init__(online_model, offline_model, optimizer, prune_rate, prune_rate_decay, args, train_steps, slist, Itertrain)
        # offline model 
        self.offline_model = offline_model

        # offline masks
        self.offline_masks = {}
        self.offline_pruning_count = {}

        # magnitude pruning mask
        self.mp_mask = {}

        # current pruning rate
        self.curr_prune_rate = self.args.init_density
        self.init_prune_epoch = self.args.init_prune_epoch

        # overlap val
        self.online_overlap = 0.0
        self.offline_overlap = 0.0

        # ema gradient
        self.g_momentum = g_momentum
        self.init_momentum_buffer()

    def init_momentum_buffer(self):
        """
        Memory buffer for instant gradient
        """
        state_copy = self.model.state_dict()
        self.mom_buffer = OrderedDict()
        
        # initialize the buffer as zero
        for n, v in state_copy.items():
            if 'mask' in n:
                name = n.replace('.mask', '')
                self.mom_buffer[name] = torch.zeros_like(v)
        print("[Debug] Momentum Buffer initialized! Moving grad = {}".format(self.args.crm))

    def name(self):
        return "Contrastive Regrow Mask"
    
    def offline_layer_stats(self):
        self.offline_nonzeros = {}
        self.offline_zeros = {}
        for name, mask in self.offline_masks.items():
            self.offline_nonzeros[name] = mask.sum().item()
            self.offline_zeros[name] = mask.numel() - self.offline_nonzeros[name]
    
    def get_offline_grad(self, weight, name):
        # gradient of online model
        g = self.buffer[name]
        
        # ema grad
        ema_g = self.mom_buffer[name]
        ema_g = ema_g.mul(self.g_momentum) + (1 - self.g_momentum) * g

        self.mom_buffer[name] = ema_g
        return ema_g

    def offline_regrow(self, new_mask, total_regrowth, weight, name):
        grad = self.get_offline_grad(weight, name)

        # only grow the weights within the current sparsity range
        grad = grad * (new_mask==0).float()
        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0
        return new_mask

    def prune_and_regrow(self, step, apply_mask=True):
        # prune and regrow of online model
        super().prune_and_regrow(step, apply_mask)

        # offline model stats
        self.offline_layer_stats()

        # prune and regrow for offline model
        for n, m in self.offline_model.named_modules():
            if isinstance(m, (SparsConv2d, SparsLinear)):
                weight = m.weight

                # update mask for pruning
                new_mask = self.magnitude_death(weight, n)
                self.offline_pruning_count[n] = int(self.offline_nonzeros[n] - new_mask.sum().item())

                # regrow
                new_mask = self.offline_regrow(new_mask, self.offline_pruning_count[n], weight, n)

                # record mask
                self.offline_masks[n] = new_mask

                # apply mask
                if apply_mask:
                    m.mask = new_mask.clone()
        
        offline_spars = self._sparsity(self.offline_model)
        print("[Offline model] Sparsity after regrow at step [{}] = {:3f}".format(step, offline_spars*100))
    