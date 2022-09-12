"""
Masking
"""
import math
import numpy as np
import pandas as pd
import copy
import torch
import torch.optim as optim
import torch.nn as nn
from solo.backbones import SparsConv2d, SparsLinear
from typing import List, Dict
from collections import OrderedDict

class CosineDecay(object):
    def __init__(self, prune_rate, T_max, eta_min=0.005, last_epoch=-1):
        self.sgd = optim.SGD(torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))]), lr=prune_rate)
        self.cosine_stepper = torch.optim.lr_scheduler.CosineAnnealingLR(self.sgd, T_max, eta_min, last_epoch)

    def step(self):
        self.cosine_stepper.step()

    def get_dr(self):
        return self.sgd.param_groups[0]['lr']


class Mask(object):
    def __init__(self, model:nn.Module, optimizer=None, prune_rate=0.3, prune_rate_decay=None, args=None, train_steps=None, slist:List=None, Itertrain=False):
        self.args = args
        self.train_steps = train_steps
        self.prune_rate_decay = prune_rate_decay
        self.prune_rate = prune_rate   # prune rate during the "prune and regrow" step 
        self.optimizer = optimizer
        self.model = model

        # global mask
        self.masks = {}

        # prunnig
        self.total_params = 0 
        self.prune_every_k_steps = self.args.update_frequency
        self.pruning_count = {}

        self.steps = [0 for i in range(len(slist))]

        # switchable pruning
        self.slist = slist
        self.final_density = self.slist[0]

        # iterative pruning
        self.iter = Itertrain

        # regrow amount
        self.regrow = 0.0
        self.mp_masks = {}

        # initial pruning epochs
        self.init_prune_epoch = self.args.init_prune_epoch
        self.init_buffer()
    
    def init_buffer(self):
        """
        Memory buffer for mask
        """
        state_copy = self.model.state_dict()
        self.buffer = OrderedDict()

        # initialize the buffer as zero
        for n, v in state_copy.items():
            if 'mask' in n:
                name = n.replace('.mask', '')
                self.buffer[name] = torch.zeros_like(v)
        print("[Debug] Buffer initialized! Moving grad = {}".format(self.args.crm))

    def switch(self, n):
        self.final_density = self.slist[n]
        
        # restate the masks for the subnet
        for name, module in self.model.named_modules():
            if isinstance(module, (SparsConv2d, SparsLinear)):
                self.masks[name] = module.mask

    def _param_stats(self):
        total_params = 0
        spars_params = 0
        for name, module in self.model.named_modules():
            if isinstance(module, (SparsConv2d, SparsLinear)):
                mask = module.mask
                total_params += mask.numel()
                spars_params += mask[mask.eq(0)].numel()
        return total_params, spars_params

    def _layer_stats(self):
        self.name2nonzeros = {}
        self.name2zeros = {}

        for name, mask in self.masks.items():
            self.name2nonzeros[name] = mask.sum().item()
            self.name2zeros[name] = mask.numel() - self.name2nonzeros[name]
        
    def reg_masks(self, train):
        for name, module in self.model.named_modules():
            if isinstance(module, (SparsConv2d, SparsLinear)):
                self.masks[name] = module.mask
        
        if train:
            # initial sparsity
            self.init(self.args.init_density, self.masks)
        
            # apply mask
            self.apply_mask()

    def init(self, density:float, mask_dict:Dict, erk_power_scale:float=1.0):
        print('initialize by ERK')
        self.total_params, _ = self._param_stats()

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
        for name, mask in mask_dict.items():
            n_param = mask.numel()
            if name in dense_layers:
                density_dict[name] = 1.0
            else:
                probability_one = epsilon * raw_probabilities[name]
                density_dict[name] = probability_one
            print(
                f"layer: {name}, shape: {mask.shape}, density: {density_dict[name]}"
            )
            mask_dict[name][:] = (torch.rand(mask.shape) < density_dict[name]).float().data.cuda()

            total_nonzero += density_dict[name] * mask.numel()
        print(f"Overall sparsity {1 - total_nonzero / self.total_params}")


    def apply_mask(self):
        for n, m in self.model.named_modules():
            if isinstance(m, (SparsConv2d, SparsLinear)):
                m.mask.data = self.masks[n].clone()

    def collect_score(self):
        weight_abs = []
        for n, m in self.model.named_modules():
            if isinstance(m, (SparsConv2d, SparsLinear)):
                weight_abs.append(m.weight.data.abs())
        mp_scores = torch.cat([torch.flatten(x) for x in weight_abs])
        return mp_scores

    def update_mask(self, threshold):
        for n, m in self.model.named_modules():
            if isinstance(m, (SparsConv2d, SparsLinear)):
                self.masks[n] = m.weight.abs().gt(threshold).float()

    def mp(self):
        # magnitude score
        mp_scores = self.collect_score()
        num_params_to_keep = int(len(mp_scores) * (1 - self.curr_prune_rate))
        topkscore, _ = torch.topk(mp_scores, num_params_to_keep, sorted=True)
        threshold = topkscore[-1]

        # update and apply the masks
        self.update_mask(threshold)

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
            self.curr_prune_rate = (1 - self.args.init_density) + (self.args.init_density - self.final_density) * (1 - ramping_decay)

            # get magnitude socre & update mask
            self.mp()

            if apply_mask:
                self.apply_mask()
        
        # sparsity
        total_params, spars_params = self._param_stats()
        sparsity = spars_params / total_params
        print("Sparsity after pruning at step [{}] = {:3f}".format(bidx, sparsity*100))

    def prune_and_regrow(self, step, apply_mask=True):
        """
        Step 2: Layer-wise pruning followed by re-growing
        """

        # layer statistics
        self._layer_stats()

        # record the magnitude pruning results
        self.mp_masks = copy.deepcopy(self.masks)

        # prune
        for n, m in self.model.named_modules():
            if isinstance(m, (SparsConv2d, SparsLinear)):
                weight = m.weight
                
                # update mask for pruning
                new_mask = self.magnitude_death(weight, n)
                self.pruning_count[n] = int(self.name2nonzeros[n] - new_mask.sum().item())

                # regrow
                new_mask = self.gradient_growth(new_mask, self.pruning_count[n], weight, n)
                
                # record mask
                self.masks[n] = new_mask

                # apply mask
                if apply_mask:
                    m.mask = new_mask.clone()

        # sparsity
        total_params, spars_params = self._param_stats()
        sparsity = spars_params / total_params
        print("[Online model] Sparsity after regrow at step [{}] = {:3f}; apply mask = {}".format(step, sparsity*100, apply_mask))

    def magnitude_death(self, weight, name):
        """
        Step 2-1: Remove the most non-significant weights inside remaining weights
        """
        num_remove = math.ceil(self.prune_rate*self.name2nonzeros[name])
        if num_remove == 0.0: return weight.data != 0.0
        num_zeros = self.name2zeros[name]
        k = math.ceil(num_zeros + num_remove)
        x, idx = torch.sort(torch.abs(weight.data.view(-1)))
        threshold = x[k-1].item()
        # if num_zeros > 0:
        #     import pdb;pdb.set_trace()
        return (torch.abs(weight.data) > threshold)

    
    def gradient_growth(self, new_mask, total_regrowth, weight, name):
        """
        Step 2-2: Regrow the weights with the most significant gradient
        """
        grad = self.get_gradient_for_weights(weight, name)
        
        # update the buffer 
        self.buffer[name] = grad.data
        
        # only grow the weights within the current sparsity range
        grad = grad*(new_mask==0).float()
        y, idx = torch.sort(torch.abs(grad).flatten(), descending=True)
        new_mask.data.view(-1)[idx[:total_regrowth]] = 1.0

        return new_mask

    def get_gradient_for_weights(self, weight, name):
        grad = weight.grad.clone()
        return grad

    def regrow_overlap(self):
        return self.overlap(self.mp_masks, self.masks)

    def step(self, n, bidx):
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
            # self.regrow = self.regrow_overlap()

        
    def nlprune(self, tspars:float):
        r"""Direct pruning without learning (after training)
        """
        mp_scores = self.collect_score()
        num_params_to_keep = int(len(mp_scores) * (1 - tspars))
        
        topkscore, _ = torch.topk(mp_scores, num_params_to_keep, sorted=True)
        threshold = topkscore[-1]

        # update and apply the masks
        self.update_mask(threshold)
        self.apply_mask()

    def overlap(self, mask1:Dict, mask2:Dict):
        """
        XOR element-wise overlap between masks
        """
        overlap = 0
        total = 0
        for (on, om), (mn, mm) in zip(mask1.items(), mask2.items()):
            num_all = om.numel()                
            
            # compute overlap
            xor = torch.bitwise_xor(om.data.int(), mm.data.int())
            ovlp = xor[xor.eq(0.)].numel()

            overlap += ovlp
            total += num_all
        
        return overlap / total
    

class MaskStat:
    def __init__(self, model:nn.Module, slist:List, logger):
        self.model = model
        self.slist = slist
        self.logger = logger
    
    def sparsity(self):
        sparsity_all = OrderedDict()
        cnt = 0
        total_param = 0
        total_nz = 0
        for n, m in self.model.named_modules():
            if isinstance(m, (SparsConv2d, SparsLinear)):
                spars_list = []
                mask = m.__getattr__(f"mask")
                nz = mask.sum().item()
                
                total_param += mask.numel()
                total_nz += nz

                spars = 1 - nz/mask.numel()
                spars_list.append(spars)
                name = n +'.sparsity'
                sparsity_all[name] = np.array(spars_list)
                cnt += 1
        self.sdf = pd.DataFrame.from_dict(sparsity_all)
        self.sdf = self.sdf.T
        self.sdf.to_csv("./layerwise_sparsity.csv")
        overall_sparsity = 1 - total_nz / total_param
        print(overall_sparsity)
    
    def overlap(self, save_path):
        overlap_all = OrderedDict()
        cnt = 0
        for n, m in self.model.named_modules():
            if isinstance(m, (SparsConv2d, SparsLinear)):
                mask_ref = m.__getattr__(f"mask{0}")
                num_all = mask_ref.numel()
                overlap = []
                for i in range(1, len(self.slist)):
                    mask = m.__getattr__(f"mask{i}")
                    xor = torch.bitwise_xor(mask.int(), mask_ref.int())
                    s = xor[xor.eq(0.)].numel() / num_all
                    overlap.append(s)
                name = n +'.overlap'
                overlap_all[name] = np.array(overlap)
                cnt += 1
        self.odf = pd.DataFrame.from_dict(overlap_all)
        self.odf = self.odf.T
        # self.odf.to_csv("./layerwise_overlap_Iter.csv")
        self.odf.to_csv(save_path)