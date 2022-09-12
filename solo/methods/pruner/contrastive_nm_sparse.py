"""
Contrastive pruning with N:M sparsity
"""

import math
import numpy as np
import torch
import torch.nn as nn
from solo.backbones import SparsConv2d, SparsLinear
from typing import Dict, List
from torch import Tensor
from .contrastive_sparse import ContrastiveMask

class ContrastiveNM(ContrastiveMask):
    def __init__(self, online_model: nn.Module, offline_model: nn.Module, optimizer=None, prune_rate=0.3, prune_rate_decay=None, args=None, train_steps=None, slist: List = None, Itertrain=False, N:int=2, M:int=4):
        super(ContrastiveNM, self).__init__(online_model, offline_model, optimizer, prune_rate, prune_rate_decay, args, train_steps, slist, Itertrain)
        # offline model
        self.offline_model = offline_model

        # structure
        self.M = M
        self.N = N
        assert N < M, "N must be less than M"
        self.sratio = 1 - N / M

        # offline masks
        self.offline_masks = {}

        # magnitude pruning mask
        self.mp_masks = {}
        self.offline_mp_masks = {}

        # current pruning rate
        self.curr_prune_rate = 1 - self.args.init_density
        self.init_prune_epoch = self.args.init_prune_epoch

        # overlap val
        self.online_overlap = 0.0
        self.offline_overlap = 0.0

        # final probability
        self.final_prob = 0.01

    def name(self):
        return "Contrastive NM Mask: M={}, N={}".format(self.M, self.N)

    def reg_masks(self, train):
        for (on, om), (mn, mm) in zip(self.model.named_modules(), self.offline_model.named_modules()):
            if hasattr(om, "mask"):
                self.masks[on] = om.mask
                
                if self.args.sparse_enck:
                    self.offline_masks[mn] = mm.mask
                else:
                    self.offline_masks[mn] = torch.ones_like(mm.weight)
        if train:
            if self.args.init_density < 1.0:
                self.masks, self.name2nzgrp, self.name2zerogrp = self.init(self.model, self.args.init_density, self.masks)
                if self.args.sparse_enck:
                    self.offline_masks, self.offline_name2nzgrp, self.offline_name2zerogrp = self.init(self.offline_model, self.args.init_density+self.args.density_gap, self.offline_masks)
                self.erk = True
        
            self.apply_mask()
            self._latch_mask()

        # sparsity
        online_spars = self._sparsity(self.model)
        if self.args.sparse_enck:
            offline_spars = self._sparsity(self.offline_model)
        else:
            offline_spars = 0.0

        print("[Online model] Sparsity after pruning at step [0] = {:3f}".format(online_spars*100))
        print("[Offline model] Sparsity after pruning at step [0] = {:3f}".format(offline_spars*100))

    def structured_layer_stats(self, masks:Dict):
        """
        Get stats of the model
        """
        name2nzgrp = {}
        name2zerogrp = {}

        for name, mask in masks.items():
            group = self._get_groups(mask)
            
            if len(mask.size()) == 4:
                m = mask.permute(0,2,3,1).reshape(group, int(self.M))
            elif len(mask.size()) == 2:
                m = mask.reshape(group, int(self.M))
            
            gsum = m.sum(dim=1)
            nzgrp = gsum[gsum.eq(self.M)].numel()

            # sparse and non sparse groups
            name2nzgrp[name] = nzgrp
            name2zerogrp[name] = group - nzgrp
        return name2nzgrp, name2zerogrp

    def init(self, model:nn.Module, density:float, mask_dict:Dict, erk_power_scale:float=1.0):
        print('initialize by ERK')
        self.total_params, _ = self._param_stats()

        # initialize group stats
        name2nzgrp, name2zerogrp = self.structured_layer_stats(mask_dict)

        is_epsilon_valid = False
        dense_layers = set()
        while not is_epsilon_valid:
            divisor = 0
            rhs = 0
            raw_probabilities = {}
            for name, mask in mask_dict.items():
                n_param = mask.numel()
                n_zeros = n_param * (1 - density)
                n_ones = n_param * density

                if name in dense_layers:
                    # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                    rhs -= n_zeros

                else:
                    # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                    # equation above.
                    rhs += n_ones
                    # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                    raw_probabilities[name] = (np.sum(list(mask.size())) / mask.numel()) ** erk_power_scale
                    # Note that raw_probabilities[mask] * n_param gives the individual
                    # elements of the divisor.
                    divisor += raw_probabilities[name] * n_param
            # By multipliying individual probabilites with epsilon, we should get the
            # number of parameters per layer correctly.
            epsilon = rhs / divisor
            # If epsilon * raw_probabilities[mask.name] > 1. We set the sparsities of that
            # mask to 0., so they become part of dense_layers sets.
            max_prob = np.max(list(raw_probabilities.values()))
            max_prob_one = max_prob * epsilon
            if max_prob_one > 1:
                is_epsilon_valid = False
                for mask_name, mask_raw_prob in raw_probabilities.items():
                    if mask_raw_prob == max_prob:
                        print(f"Sparsity of var:{mask_name} had to be set to 0.")
                        dense_layers.add(mask_name)
            else:
                is_epsilon_valid = True

        density_dict = {}
        total_nonzero = 0.0

        # With the valid epsilon, we can set sparsities of the remaning layers.
        for name, m in model.named_modules():
            if isinstance(m, SparsConv2d):
                mask = mask_dict[name]
                n_param = mask.numel()
                if name in dense_layers:
                    density_dict[name] = 1.0
                else:
                    probability_one = epsilon * raw_probabilities[name]
                    density_dict[name] = probability_one
                
                # initialize n2m mask
                smask = self.random_grp_mask(m.weight.data, 1.0 - density_dict[name])
                smask = smask.reshape(m.weight.data.permute(0,2,3,1).shape)
                smask = smask.permute(0,3,1,2)

                # update mask
                mask_dict[name] = smask

                # sparsity
                spars = smask[smask.eq(0.)].numel() / smask.numel()
                print(
                    f"layer: {name}, shape: {mask.shape}, probablistic density: {density_dict[name]}, sparsity={spars}"
                )

                total_nonzero += smask[smask.eq(1.)].numel()

        print(f"Overall sparsity {1 - total_nonzero / self.total_params}")
        return mask_dict, name2nzgrp, name2zerogrp

    def _get_groups(self, tensor:Tensor):
        length = tensor.numel()
        group = int(length/self.M)
        return group

    def get_grp_threshold(self, wgrp, prob):
        gsum = wgrp.sum(dim=1)
        num_grps_to_keep = int(len(gsum) * (1 - prob))
        try:
            topkscore, _ = torch.topk(gsum, num_grps_to_keep, sorted=True)
        except:
            import pdb;pdb.set_trace()
        threshold = topkscore[-1]
        return gsum, threshold

    def n2m(self, wgrp):
        index = torch.argsort(wgrp, dim=1)[:, :int(self.M-self.N)]
        w_b = torch.ones(wgrp.shape, device=wgrp.device)
        w_b = w_b.scatter_(dim=1, index=index, value=0)
        return w_b
    
    def random_grp_mask(self, weight, prob):
        group = self._get_groups(weight)
        wgrp = torch.randn(weight.shape).abs().permute(0,2,3,1).reshape(group, int(self.M)).cuda()

        # N2M masks
        w_b = self.n2m(wgrp)

        # probability sample 
        if prob < 1.0:
            rnumel = math.ceil((1-prob)*group)
            ridx = torch.randperm(rnumel)
            w_b[ridx] = 1.
        
        return w_b

    def mp_grp_mask(self, weight, prob):
        group = self._get_groups(weight)
        wgrp = weight.detach().abs().permute(0,2,3,1).reshape(group, int(self.M))
        
        # N2M masks
        w_b = self.n2m(wgrp)
        
        # Update n2m mask with mp scores
        if prob < 1.0:
            gsum, threshold = self.get_grp_threshold(wgrp, prob)
            w_b[gsum.gt(threshold)] = 1.0
    
        return w_b

    def mask2weight(self, mask, weight):
        mask = mask.reshape(weight.permute(0,2,3,1).shape)
        mask = mask.permute(0,3,1,2)
        return mask

    def update_mask(self, prob):
        for (on, om), (mn, mm) in zip(self.model.named_modules(), self.offline_model.named_modules()):
            if isinstance(om, SparsConv2d):
                ow = om.weight.clone()

                # get mask
                os = self.mp_grp_mask(ow, prob)
                # reshape
                os = self.mask2weight(os, ow)
                
                # update online mask
                self.masks[on] = os

                if self.args.sparse_enck:
                    mw = mm.weight.clone()
                    
                    # get mask
                    ms = self.mp_grp_mask(mw, prob - self.args.density_gap)
                    # reshape
                    ms = self.mask2weight(ms, mw)
                    # update offline mask
                    self.offline_masks[mn] = ms
                else:
                    self.offline_masks[mn] = torch.ones_like(os)

    def pruning(self, step, bidx, apply_mask=True):
        curr_prune_iter = int(step / self.prune_every_k_steps)
        final_iter = int((self.args.final_prune_epoch * self.train_steps*self.args.multiplier) / self.prune_every_k_steps)
        ini_iter = int((self.args.init_prune_epoch * self.train_steps*self.args.multiplier) / self.prune_every_k_steps)
        total_prune_iter = final_iter - ini_iter

        # message
        print(f'Pruning Progress is {curr_prune_iter - ini_iter} / {total_prune_iter}')

        # prune
        if curr_prune_iter >= ini_iter and curr_prune_iter <= final_iter:
            # update sparsity schedule
            ramping_decay = (1 - ((curr_prune_iter - ini_iter) / total_prune_iter)) ** 3
            
            # current probability
            self.curr_prob = (1 - self.args.init_density) + (self.args.init_density - self.final_prob) * (1 - ramping_decay)
            
            # update mask
            self.update_mask(self.curr_prob)
            if apply_mask:
                self.apply_mask()
        
        # sparsity stats
        total_params, spars_params = self._param_stats()
        sparsity = spars_params / total_params
        print("Sparsity after pruning at step [{}] = {:.3f} with prob={:.3f}".format(bidx, sparsity*100, self.curr_prob))
    
    def update_mp_mask(self, online_thre, offline_thre):
        for (on, om), (mn, mm) in zip(self.model.named_modules(), self.offline_model.named_modules()):
            if isinstance(om, (SparsConv2d, SparsLinear)):
                self.mp_masks[on] = om.weight.abs().gt(online_thre).float()
                self.offline_mp_masks[mn] = mm.weight.abs().gt(offline_thre).float()

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
        self.update_mp_mask(online_thre, offline_thre)
    
    def contrastive_overlap(self):
        return self.overlap(self.mp_masks, self.offline_mp_masks)

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
            
            # overlap
            self.online_overlap = self.overlap(self.masks, self.online_buffer)
            
            # fetch the new mask
            self._latch_mask()