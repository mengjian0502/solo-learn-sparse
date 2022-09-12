"""
Contrastive mask
"""

import copy
import torch
import torch.nn as nn
from solo.backbones import SparsConv2d, SparsLinear
from typing import Dict, List
from .sparse import Mask


class ContrastiveMask(Mask):
    def __init__(self, online_model: nn.Module, offline_model:nn.Module, optimizer=None, prune_rate=0.3, prune_rate_decay=None, args=None, train_steps=None, slist: List = None, Itertrain=False):
        super().__init__(online_model, optimizer, prune_rate, prune_rate_decay, args, train_steps, slist, Itertrain)
        # offline model 
        self.offline_model = offline_model

        # offline masks
        self.offline_masks = {}

        # magnitude pruning mask
        self.mp_mask = {}

        # current pruning rate
        self.curr_prune_rate = 1 - self.args.init_density
        self.init_prune_epoch = self.args.init_prune_epoch

        # overlap val
        self.online_overlap = 0.0
        self.offline_overlap = 0.0

    def name(self):
        return "Contrastive Mask"

    def _sync_mask(self):
        for k, v in self.masks.items():
            self.offline_masks[k] = copy.deepcopy(v)
    
    def _latch_mask(self):
        self.online_buffer = copy.deepcopy(self.masks)
        self.offline_buffer = copy.deepcopy(self.offline_masks)

    def apply_mask(self):
        for (on, om), (mn, mm) in zip(self.model.named_modules(), self.offline_model.named_modules()):
            if hasattr(om, "mask"):
                om.mask.data = self.masks[on]
                mm.mask.data = self.offline_masks[mn]


    def reg_masks(self, train):
        for (on, om), (mn, mm) in zip(self.model.named_modules(), self.offline_model.named_modules()):
            if hasattr(om, "mask"):
                self.masks[on] = om.mask
                self.offline_masks[mn] = mm.mask
        if train:
            if self.args.init_density < 1.0:
                self.init(self.args.init_density, self.masks)
                self.init(self.args.init_density+self.args.density_gap, self.offline_masks)
                self.erk = True
            
            self.apply_mask()
            self._latch_mask()

        # sparsity
        online_spars = self._sparsity(self.model)
        offline_spars = self._sparsity(self.offline_model)

        print("[Online model] Sparsity after pruning at step [0] = {:3f}".format(online_spars*100))
        print("[Offline model] Sparsity after pruning at step [0] = {:3f}".format(offline_spars*100))

    def collect_score(self):
        online_weight_abs = []
        offline_weight_abs = []
        for (on, om), (mn, mm) in zip(self.model.named_modules(), self.offline_model.named_modules()):
            if isinstance(om, (SparsConv2d, SparsLinear)):
                online_weight_abs.append(om.weight.data.abs())
                offline_weight_abs.append(mm.weight.data.abs())
        
        online_mp_scores = torch.cat([torch.flatten(x) for x in online_weight_abs])
        offline_mp_scores = torch.cat([torch.flatten(x) for x in offline_weight_abs])
        return online_mp_scores, offline_mp_scores

    def get_threshold(self, mp_scores, curr_prune_rate:float):
        num_params_to_keep = int(len(mp_scores) * (1 - curr_prune_rate))
        topkscore, _ = torch.topk(mp_scores, num_params_to_keep, sorted=True)
        threshold = topkscore[-1]
        return threshold

    def update_mask(self, online_thre, offline_thre):
        for (on, om), (mn, mm) in zip(self.model.named_modules(), self.offline_model.named_modules()):
            if isinstance(om, (SparsConv2d, SparsLinear)):
                self.masks[on] = om.weight.abs().gt(online_thre).float()
                self.offline_masks[mn] = mm.weight.abs().gt(offline_thre).float()
    
    def _sparsity(self, model:nn.Module):
        total_params = 0
        spars_params = 0
        for name, module in model.named_modules():
            if isinstance(module, (SparsConv2d, SparsLinear)):
                mask = module.mask
                total_params += mask.numel()
                spars_params += mask[mask.eq(0)].numel()
        spars = spars_params / total_params
        return spars

    def mp(self):
        """
        magnitude pruning
        """
        # magnitude score
        online_mp_scores, offline_mp_scores = self.collect_score()
        
        if self.args.init_density == 1.0:
            online_thre = online_mp_scores.max().item()
        else:
            online_thre = self.get_threshold(online_mp_scores, self.curr_prune_rate)
            
        offline_thre = self.get_threshold(offline_mp_scores, self.curr_prune_rate-self.args.density_gap)

        # update and apply the masks
        self.update_mask(online_thre, offline_thre)
    

    def pruning(self, step, bidx, apply_mask=True):
        """
        Step 1: Scheduled pruning
        """
        curr_prune_iter = int(step / self.prune_every_k_steps)
        final_iter = int((self.args.final_prune_epoch * self.train_steps*self.args.multiplier) / self.prune_every_k_steps)
        ini_iter = int((self.init_prune_epoch * self.train_steps*self.args.multiplier) / self.prune_every_k_steps)
        total_prune_iter = final_iter - ini_iter

        # message
        print(f'Pruning Progress is {curr_prune_iter - ini_iter} / {total_prune_iter}')
        if curr_prune_iter >= ini_iter and curr_prune_iter <= final_iter:
            # update sparsity schedule
            ramping_decay = (1 - ((curr_prune_iter - ini_iter) / total_prune_iter)) ** 3
            self.curr_prune_rate = (1 - self.args.init_density) + (self.args.init_density - self.args.final_density) * (1 - ramping_decay)

            # get magnitude socre & update mask
            self.mp()

            # record the magnitude-based pruning results
            self.mp_mask = copy.deepcopy(self.masks)

            if apply_mask:
                self.apply_mask()
        
        # sparsity
        online_spars = self._sparsity(self.model)
        offline_spars = self._sparsity(self.offline_model)

        print("[Online model] Sparsity after pruning at step [{}] = {:3f}".format(bidx, online_spars*100))
        print("[Offline model] Sparsity after pruning at step [{}] = {:3f}".format(bidx, offline_spars*100))


    def collect_masks(self, step):
        # get score & update mask
        self.mp()

        # # gradient score
        # self.prune_and_regrow(step, False)

    def contrastive_overlap(self):
        return self.overlap(self.masks, self.offline_masks)

    def step(self, n, bidx):
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

            # compute the overlap of consecutive epochs
            self.online_overlap = self.overlap(self.masks, self.online_buffer)
            self.offline_overlap = self.overlap(self.offline_masks, self.offline_buffer)
            self.regrow = self.regrow_overlap()
        
            # fetch the new mask
            self._latch_mask()