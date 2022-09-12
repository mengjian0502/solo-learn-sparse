"""
"""

import torch
import torch.nn as nn
from solo.backbones import SparsConv2d, SparsLinear
from typing import List
from .contrastive_sparse import ContrastiveMask

def _normalize_scores(scores):
    """
    Normalizing scheme for LAMP.
    """
    # sort scores in an ascending order
    sorted_scores,sorted_idx = scores.view(-1).sort(descending=False)
    # compute cumulative sum
    scores_cumsum_temp = sorted_scores.cumsum(dim=0)
    scores_cumsum = torch.zeros(scores_cumsum_temp.shape,device=scores.device)
    scores_cumsum[1:] = scores_cumsum_temp[:len(scores_cumsum_temp)-1]
    # normalize by cumulative sum
    sorted_scores /= (scores.sum() - scores_cumsum)
    # tidy up and output
    new_scores = torch.zeros(scores_cumsum.shape,device=scores.device)
    new_scores[sorted_idx] = sorted_scores
#     print(scores.shape)
    return new_scores.view(scores.shape)

class ContrastiveLAMP(ContrastiveMask):
    def __init__(self, online_model: nn.Module, offline_model: nn.Module, optimizer=None, prune_rate=0.3, prune_rate_decay=None, args=None, train_steps=None, slist: List = None, Itertrain=False):
        super().__init__(online_model, offline_model, optimizer, prune_rate, prune_rate_decay, args, train_steps, slist, Itertrain)

    def name(self):
        return "Contrastive LAMP"

    def collect_score(self):
        online_weights = []
        offline_weights = []
        for (on, om), (mn, mm) in zip(self.model.named_modules(), self.offline_model.named_modules()):
            if isinstance(om, (SparsConv2d, SparsLinear)):
                online_weights.append(om.weight.data)
                offline_weights.append(mm.weight.data)
        
        # scores
        online_scores = [_normalize_scores(w**2).view(-1) for w in online_weights]
        offline_scores = [_normalize_scores(w**2).view(-1) for w in offline_weights]
        return online_scores, offline_scores

    def update_mask(self, online_thre, offline_thre):
        for (on, om), (mn, mm) in zip(self.model.named_modules(), self.offline_model.named_modules()):
            if isinstance(om, (SparsConv2d, SparsLinear)):
                os = _normalize_scores(om.weight.data**2)
                ms = _normalize_scores(mm.weight.data**2)

                # mask
                self.masks[on] = os.gt(online_thre).float()
                self.offline_masks[mn] = ms.gt(offline_thre).float()

    def get_threshold(self, mp_scores, curr_prune_rate:float):
        mp_scores = torch.cat(mp_scores, dim=0)
        return super().get_threshold(mp_scores, curr_prune_rate)
    
    def mp(self):
        # scores
        online_scores, offline_scores = self.collect_score()
        
        # get threshold
        online_thre = self.get_threshold(online_scores, self.curr_prune_rate)
        offline_thre = self.get_threshold(offline_scores, self.curr_prune_rate-self.args.density_gap)

        # update mask
        self.update_mask(online_thre, offline_thre)

    

        

    
    
