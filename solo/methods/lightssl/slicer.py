"""
Channel Slicer
"""

import torch
import torch.nn as nn
from torch import Tensor

class Slicer(object):
    def __init__(self, model:nn.Module, train_steps:float, interval:int, scale:float=0.5):
        self.model = model
        self.train_steps = train_steps
        self.prune_flag = False

        # masks
        self.masks = {}
        self.bn_masks = []

        # ratio
        self.alpha = scale

        # steps
        self.steps = 0

        # initialization
        self.reg_masks()
        self.prune()
        self.apply_masks()
        self.prune_every_k_steps = interval

        # sparsity
        s, nz = self.get_sparsity()
        print("\n#########################")
        print("\nSparsity after intitialization: {:.2f}, Param: {:.2f}M".format(s, nz/1e+6))
        print("Interval: {}; Slice ratio: {}\n".format(self.prune_every_k_steps, self.alpha))
        print("#########################\n")

    def reg_masks(self):
        for n, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d) and hasattr(m, "mask"):
                weight = m.weight.data
                mask = torch.ones_like(weight)
                self.masks[n] = mask
    
    def filter_prune(self, weight:Tensor):
        co = weight.size(0)
        filters = weight.contiguous().view(co, -1)

        # filter wise score
        fscores = filters.abs().sum(dim=1)
        fkeep = int(self.alpha * co)

        # scores
        topkscore, _ = torch.topk(fscores, fkeep, sorted=True)
        fthre = topkscore[-1]
        keep = fscores.ge(fthre)

        # mask
        mask = torch.ones_like(weight)
        mask = mask.mul(keep[:,None,None,None])

        return mask

    def channel_prune(self, weight:Tensor, mask:Tensor):
        """
        Channel wise pruning on top of the pruned filters
        """
        cin = weight.size(1)
        channels = weight.permute(1,0,2,3).contiguous().view(cin, -1)
        
        # channel wise score
        cscores = channels.abs().sum(dim=1)
        num_grps_to_keep = int(self.alpha * cin)

        topkscore, _ = torch.topk(cscores, num_grps_to_keep, sorted=True)
        threshold = topkscore[-1]

        ckeep = cscores.ge(threshold)
        mask = mask.mul(ckeep[None, :, None, None])

        return mask

    def reduce(self, mask:Tensor):
        assert len(mask.size()) == 4
        return mask.mean([1,2,3])

    def prune(self):
        nbn = 0
        for n, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d) and hasattr(m, "mask"):
                weight = m.weight.data
                mask_ = self.filter_prune(weight)
                if weight.size(1) > 3:
                    mask_ = self.channel_prune(weight, mask_)
                
                # update mask
                self.masks[n] = mask_
                
                # bn mask
                bnm = self.reduce(mask_)
                self.bn_masks.append(bnm)
            elif isinstance(m, nn.BatchNorm2d) and hasattr(m, "prune_flag"):
                m.mask.data = self.bn_masks[nbn].cuda()
                nbn += 1
                
        
    def apply_masks(self):
        for n, m in self.model.named_modules():
            if isinstance(m, nn.Conv2d) and hasattr(m, "mask"):
                mask = self.masks[n]
                m.mask = mask

    def step(self):
        self.steps += 1

        if self.steps >= 1 * (self.train_steps) and self.steps % self.prune_every_k_steps == 0:
            print("Update mask @ Step {}".format(self.steps))
            self.prune()
            self.apply_masks()

    def remove_mask(self):
        for m in self.model.modules():
            if hasattr(m, "prune_flag"):
                m.prune_flag = False

    def activate_mask(self):
        for m in self.model.modules():
            if hasattr(m, "prune_flag"):
                m.prune_flag = True

    def get_sparsity(self):
        total = 0
        nz = 0
        nparams = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) and hasattr(m, "mask"):
                total += m.mask.numel()
                nz += m.mask.sum().item()
                nparams += nz
        return 1 - nz/total, nz
    
    