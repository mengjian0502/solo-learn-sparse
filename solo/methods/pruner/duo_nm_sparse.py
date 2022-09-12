"""
N:M Pruner for CL method with shared encoder (e.g., SimCLR)
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from solo.backbones import SparsConv2d, SparsLinear
from typing import Dict, List
from .duo_sparse import DuoMask

class DuoNMMask(DuoMask):
    def __init__(self, model: nn.Module, optimizer=None, prune_rate=0.3, prune_rate_decay=None, args=None, train_steps=None, slist: List = None, Itertrain=False, N:int=2, M:int=4):
        super().__init__(model, optimizer, prune_rate, prune_rate_decay, args, train_steps, slist, Itertrain)
        self.M = M
        self.N = N

        assert N < M, "N must be less than M"

        self.sratio = 1 - N / M
        
        self.curr_prune_rate = 1 - self.args.init_density
        self.init_prune_epoch = self.args.init_prune_epoch

        # final probability
        self.final_prob = 0.01

    def name(self):
        return "Duo mask with N:M sparsity"

    def reg_masks(self, train):
        for (on, om) in self.model.named_modules():
            if hasattr(om, "mask"):
                self.masks[on] = om.mask
                self.contrastive_masks[on] = torch.ones_like(om.mask.data)
        
        if train:
            if self.args.init_density < 1.0:
                self.masks, self.name2nzgrp, self.name2zerogrp = self.init(self.model, self.args.init_density, self.masks)
                if self.args.sparse_enck:
                    self.contrastive_masks, self.cname2nonzeros, self.cname2zeros = self.init(self.model, self.args.init_density+self.args.density_gap, self.contrastive_masks)
                self.erk = True
            
            # apply online mask only
            self.apply_mask()
        
        online_spars, contrastive_spars = self.get_sparsity()    
        print("Sparsity after pruning at step [0] = {:3f}".format(online_spars*100))
        print("Contrastive Sparsity after pruning at step [0] = {:3f}".format(contrastive_spars*100))


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
        for on, om in self.model.named_modules():
            if isinstance(om, SparsConv2d):
                ow = om.weight.clone()

                # get mask
                os = self.mp_grp_mask(ow, prob)
                # reshape
                os = self.mask2weight(os, ow)
                
                # update online mask
                self.masks[on] = os

                if self.args.sparse_enck:                    
                    # get mask
                    ms = self.mp_grp_mask(ow, prob - self.args.density_gap)
                    # reshape
                    ms = self.mask2weight(ms, ow)
                    # update offline mask
                    self.contrastive_masks[on] = ms
                else:
                    self.contrastive_masks[on] = torch.ones_like(os)

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


    def group_death(self, weight, name, module, name2nzgrp:Dict, name2zerogrp:Dict):
        group = self._get_groups(weight)

        # number of groups we want to remove (temporarily)
        num_remove = int(self.prune_rate*name2nzgrp[name]) 
        
        if num_remove == 0.0: 
            w_b = weight.data.ne(0).float()
        else:
            # number of pruned groups
            num_zeros = name2zerogrp[name]
            
            # total number of sparse groups
            k = int(num_zeros + num_remove)

            if isinstance(module, SparsConv2d):
                weight_temp = weight.detach().abs().permute(0,2,3,1).reshape(group, int(self.M))

            index = torch.argsort(weight_temp, dim=1)[:, :int(self.M-self.N)]

            w_b = torch.ones(weight_temp.shape, device=weight_temp.device)
            w_b = w_b.scatter_(dim=1, index=index, value=0) # fill all the groups with N:M sparsity

            # prune more unimportant groups
            wgsum = torch.sum(weight_temp.abs(), dim=1)
            y, idx = torch.sort(torch.abs(wgsum).flatten())
            w_b[idx[:(wgsum.size(0)-k)]] = 1.
            
            # reshape
            if isinstance(module, SparsConv2d):
                w_b = w_b.reshape(weight.permute(0,2,3,1).shape)
                w_b = w_b.permute(0,3,1,2)
        return w_b, num_remove

    
    def prune_and_regrow(self, step, apply_mask=True):
        self.name2nzgrp, self.name2zerogrp = self.structured_layer_stats(self.masks)
        self.cname2nonzeros, self.cname2zeros = self.structured_layer_stats(self.contrastive_masks)

        for n, m in self.model.named_modules():
            if isinstance(m, (SparsConv2d)):
                weight = m.weight
                
                # update the mask for pruning
                new_mask, num_remove = self.group_death(weight, n, m, self.name2nzgrp, self.name2zerogrp)
                new_cmask, cnum_remove = self.group_death(weight, n, m, self.cname2nonzeros, self.cname2zeros)

                self.pruning_count[n] = num_remove
                self.contrastive_pruning_count[n] = cnum_remove
                
                new_mask, _ = self.grp_grad_growth(new_mask, self.pruning_count[n], weight, n)
                new_cmask, _ = self.grp_grad_growth(new_cmask, self.contrastive_pruning_count[n], weight, n)

                # record mask
                self.masks[n] = new_mask
                self.contrastive_masks[n] = new_cmask

                # apply mask
                if apply_mask:
                    m.mask = new_mask.clone()
        
        online_spars, contrastive_spars = self.get_sparsity()
        print("[Online mask] Sparsity after regrow at step [{}] = {:3f}; apply mask = {}".format(step, online_spars*100, apply_mask))
        print("[Contrastive mask] Sparsity after regrow at step [{}] = {:3f}; apply mask = {}".format(step, contrastive_spars*100, apply_mask))
    

    def grp_grad_growth(self, new_mask, total_regrowth, weight, name):
        grad = self.get_gradient_for_weights(weight, name)
        group = self._get_groups(grad)
        
        # gradient group
        if len(new_mask.size())==4:
            gradgrp = grad.abs().permute(0,2,3,1).reshape(group, int(self.M))
            m = new_mask.permute(0,2,3,1).reshape(group, int(self.M))
        elif len(new_mask.size())==2:
            gradgrp = grad.abs().reshape(group, int(self.M))
            m = new_mask.reshape(group, int(self.M))

        # only grow the weights within the current sparsity
        msum = torch.sum(m, dim=1)
        sidx = msum.eq(self.N).float()

        gradgrp = gradgrp*sidx[:, None]
        gsum = torch.sum(gradgrp, dim=1)
        y, idx = torch.sort(gsum.flatten(), descending=True)            
        
        # regrow
        m[idx[:total_regrowth]] = 1.0
        msum = torch.sum(m, dim=1)
        # print(msum.unique())
        
        # reshape
        if len(new_mask.size())==4:
            rgmask = m.reshape(new_mask.permute(0,2,3,1).shape)
            rgmask = rgmask.permute(0,3,1,2)
        elif len(new_mask.size())==2:
            rgmask = m.reshape(new_mask.shape)

        return rgmask, msum[msum.eq(self.N)].numel()