"""
Pruner for CL method with shared encoder (e.g., SimCLR)
"""
import math
import copy
import torch
import torch.nn as nn
from solo.backbones import SparsConv2d, SparsLinear
from typing import Dict, List
from .sparse import Mask

class DuoMask(Mask):
    def __init__(self, model: nn.Module, optimizer=None, prune_rate=0.3, prune_rate_decay=None, args=None, train_steps=None, slist: List = None, Itertrain=False):
        super().__init__(model, optimizer, prune_rate, prune_rate_decay, args, train_steps, slist, Itertrain)
        self.contrastive_masks = {}
        self.contrastive_pruning_count = {}
        
        # temporal masks
        self.tmasks = {}

    def name(self):
        return "Duo mask"

    def get_sparsity(self):
        online_spars = self._sparsity(self.masks)    
        contrastive_spars = self._sparsity(self.contrastive_masks)
        return online_spars, contrastive_spars

    def _sparsity(self, mask:Dict):
        total_params = 0
        spars_params = 0
        for name, mask in mask.items():
            total_params += mask.numel()
            spars_params += mask[mask.eq(0)].numel()
        spars = spars_params / total_params
        return spars

    def _contrastive_layer_stats(self):
        self.cname2nonzeros = {}
        self.cname2zeros = {}

        for name, mask in self.contrastive_masks.items():
            self.cname2nonzeros[name] = mask.sum().item()
            self.cname2zeros[name] = mask.numel() - self.cname2nonzeros[name]
    
    def reg_masks(self, train):
        for on, om in self.model.named_modules():
            if hasattr(om, "mask"):
                self.masks[on] = om.mask.data
                self.contrastive_masks[on] = torch.ones_like(om.mask.data)
        
        if train:
            self.init(self.args.init_density, self.masks)
            self.init(self.args.init_density+self.args.density_gap, self.contrastive_masks)

            # apply online mask only
            self.apply_mask()
        
        # initialize temporal overalp mask based on the final sparsity
        fthre = self.mp_threshold()
        for n, m in self.model.named_modules():
            if isinstance(m, (SparsConv2d, SparsLinear)):
                self.tmasks[n] = m.weight.abs().gt(fthre).float()

        online_spars, contrastive_spars = self.get_sparsity()    
        print("Sparsity after pruning at step [0] = {:3f}".format(online_spars*100))
        print("Contrastive Sparsity after pruning at step [0] = {:3f}".format(contrastive_spars*100))
        

    def get_threshold(self, mp_scores, curr_prune_rate:float):
        num_params_to_keep = int(len(mp_scores) * (1 - curr_prune_rate))
        topkscore, _ = torch.topk(mp_scores, num_params_to_keep, sorted=True)
        threshold = topkscore[-1]
        return threshold

    def update_mask(self, threshold, cthreshold):
        for n, m in self.model.named_modules():
            if isinstance(m, (SparsConv2d, SparsLinear)):
                self.masks[n] = m.weight.abs().gt(threshold).float()
                self.contrastive_masks[n] = m.weight.abs().gt(cthreshold).float()
    
    def mp(self):
        mp_scores = self.collect_score()
        threshold = self.get_threshold(mp_scores, self.curr_prune_rate)
        cthreshold = self.get_threshold(mp_scores, self.curr_prune_rate-self.args.density_gap)

        # update masks
        self.update_mask(threshold, cthreshold)
    
    def magnitude_death(self, weight, name, name2nonzeros:Dict, name2zeros:Dict):
        """
        Step 2-1: Remove the most non-significant weights inside remaining weights
        """
        num_remove = math.ceil(self.prune_rate*name2nonzeros[name])
        if num_remove == 0.0: return weight.data != 0.0
        num_zeros = name2zeros[name]
        k = math.ceil(num_zeros + num_remove)
        x, idx = torch.sort(torch.abs(weight.data.view(-1)))
        threshold = x[k-1].item()
        return (torch.abs(weight.data) > threshold)

    def prune_and_regrow(self, step, apply_mask=True):
        
        # layer statistics
        self._layer_stats()
        self._contrastive_layer_stats()

        # record the magnitude pruning results
        self.mp_masks = copy.deepcopy(self.masks)

        # prune
        for n, m in self.model.named_modules():
            if isinstance(m, (SparsConv2d, SparsLinear)):
                weight = m.weight
                
                # update mask for pruning
                new_mask = self.magnitude_death(weight, n, self.name2nonzeros, self.name2zeros)
                new_cmask = self.magnitude_death(weight, n, self.cname2nonzeros, self.cname2zeros)

                # amount of regrow
                self.pruning_count[n] = int(self.name2nonzeros[n] - new_mask.sum().item())
                self.contrastive_pruning_count[n] = int(self.cname2nonzeros[n] - new_cmask.sum().item())

                # regrow
                new_mask = self.gradient_growth(new_mask, self.pruning_count[n], weight, n)
                new_cmask = self.gradient_growth(new_cmask, self.contrastive_pruning_count[n], weight, n)

                # record mask
                self.masks[n] = new_mask
                self.contrastive_masks[n] = new_cmask

                # apply mask
                if apply_mask:
                    m.mask = new_mask.clone()
        
        online_spars, contrastive_spars = self.get_sparsity()
        print("[Online mask] Sparsity after regrow at step [{}] = {:3f}; apply mask = {}".format(step, online_spars*100, apply_mask))
        print("[Contrastive mask] Sparsity after regrow at step [{}] = {:3f}; apply mask = {}".format(step, contrastive_spars*100, apply_mask))
    
    def mp_scores(self):
        weight_abs = []
        for n, m in self.model.named_modules():
            if isinstance(m, (SparsConv2d, SparsLinear)):
                weight_abs.append(m.weight.data.abs())
        mp_scores = torch.cat([torch.flatten(x) for x in weight_abs])
        return mp_scores
    
    def mp_threshold(self):
        mp_scores = self.mp_scores()
        threshold = self.get_threshold(mp_scores, 1-self.args.final_density)
        return threshold

    def temporal_ovlp(self):
        fthre = self.mp_threshold()
        overlap = 0
        total = 0
        
        for n, m in self.model.named_modules():
            if isinstance(m, (SparsConv2d, SparsLinear)):
                mask = m.weight.abs().gt(fthre).float()
                tm = self.tmasks[n] 

                # compute overlap
                xor = torch.bitwise_xor(mask.data.int(), tm.data.int())
                ovlp = xor[xor.eq(0.)].numel()

                overlap += ovlp
                total += mask.numel()

                # update masks
                self.tmasks[n] = mask

        return overlap / total