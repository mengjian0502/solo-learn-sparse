"""
Contrastive structured pruning
"""

import math
import numpy as np
import torch
import torch.nn as nn
from solo.backbones import SparsConv2d, SparsLinear
from typing import Dict, List
from torch import Tensor
from .contrastive_sparse import ContrastiveMask

class ContrastiveGrp(ContrastiveMask):
    def __init__(self, online_model: nn.Module, offline_model: nn.Module, optimizer=None, prune_rate=0.3, prune_rate_decay=None, args=None, train_steps=None, slist: List = None, Itertrain=False):
        super(ContrastiveMask, self).__init__(online_model, offline_model, optimizer, prune_rate, prune_rate_decay, args, train_steps, slist, Itertrain)

