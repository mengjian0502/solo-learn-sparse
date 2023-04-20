"""
Asymmetrical Contrastive Learning based on Barlow Twins
"""

import argparse
import torch
import torch.nn as nn
from typing import Any, Dict, List, Sequence, Tuple

from solo.losses.barlow import barlow_loss_func, distill_loss_func
from solo.methods.base import BaseMethod, BaseMomentumMethod
from solo.methods.lightssl import Slicer
from solo.utils.momentum import initialize_momentum_params, ParamCopier

class BarlowACL(BaseMomentumMethod):
    def __init__(self, args, proj_output_dim: int, proj_hidden_dim: int, lamb: float, scale_loss: float, **kwargs):
        super().__init__(**kwargs)    

        self.lamb = lamb
        self.scale_loss = scale_loss

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        self.momentum_projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        initialize_momentum_params(self.projector, self.momentum_projector)

        # No momentume update, directly copy
        self.momentum_updater = ParamCopier()

        # cross distillation
        self.alpha = self.extra_args['alpha']
        self.llamb = self.extra_args['loglamb']
        
    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(BarlowACL, BarlowACL).add_model_specific_args(parent_parser)
        parser = parent_parser.add_argument_group("barlow_twins")

        # projector
        parser.add_argument("--proj_output_dim", type=int, default=2048)
        parser.add_argument("--proj_hidden_dim", type=int, default=2048)

    
        # parameters
        parser.add_argument("--lamb", type=float, default=0.0051)
        parser.add_argument("--scale_loss", type=float, default=0.024)
        return parent_parser

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params
    
    def forward(self, X):
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        return out
    
    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Adds (projector, momentum_projector) to the parent's momentum pairs.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs.
        """

        extra_momentum_pairs = [(self.projector, self.momentum_projector)]
        return super().momentum_pairs + extra_momentum_pairs
    
    @torch.no_grad()
    def momentum_forward(self, X: torch.Tensor) -> Dict:
        """Performs the forward pass of the momentum backbone and projector.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of
                the parent and the momentum projected features.
        """

        out = super().momentum_forward(X)
        z = self.momentum_projector(out["feats"])
        out.update({"z": z})
        return out
    
    def training_step(self, batch: List[Any], batch_idx: int) -> Dict[str, Any]:
        outs = super().training_step(batch, batch_idx)

        _, X, targets = batch
        X = [X] if isinstance(X, torch.Tensor) else X

        # remove small crops
        X = X[: self.num_large_crops]

        momentum_outs = [self._shared_step_momentum(x, targets) for x in X]
        momentum_outs = {
            "momentum_" + k: [out[k] for out in momentum_outs] for k in momentum_outs[0].keys()
        }

        if self.momentum_classifier is not None:
            # momentum loss and stats
            momentum_outs["momentum_loss"] = (
                sum(momentum_outs["momentum_loss"]) / self.num_large_crops
            )
            momentum_outs["momentum_acc1"] = (
                sum(momentum_outs["momentum_acc1"]) / self.num_large_crops
            )
            momentum_outs["momentum_acc5"] = (
                sum(momentum_outs["momentum_acc5"]) / self.num_large_crops
            )

            metrics = {
                "train_momentum_class_loss": momentum_outs["momentum_loss"],
                "train_momentum_acc1": momentum_outs["momentum_acc1"],
                "train_momentum_acc5": momentum_outs["momentum_acc5"],
            }
            self.log_dict(metrics, on_epoch=True, sync_dist=True)

            # adds the momentum classifier loss together with the general loss
            outs["loss"] += momentum_outs["momentum_loss"]

        outs.update(momentum_outs)
        

        class_loss = outs["loss"]
        q1, q2 = outs["z"]
        k1, k2 = outs["momentum_z"]

        # compute the barlow loss
        loss12 = barlow_loss_func(q1, k2, lamb=self.lamb, scale_loss=self.scale_loss)
        loss21 = barlow_loss_func(q2, k1, lamb=self.lamb, scale_loss=self.scale_loss)
        barlow_loss = (loss12 + loss21) / 2


        # cross distillation
        if self.extra_args['distype'] == "copy":
            ds12 = barlow_loss_func(q1, k1, lamb=self.lamb, scale_loss=self.scale_loss)
            ds21 = barlow_loss_func(q2, k2, lamb=self.lamb, scale_loss=self.scale_loss)
            ds_loss = (ds12 + ds21) / 2
            loss = self.alpha * barlow_loss + (1-self.alpha) * ds_loss
        elif self.extra_args['distype'] == "log":
            ds12 = distill_loss_func(t=k1, s=q1, scale_loss=self.scale_loss)
            ds21 = distill_loss_func(t=k2, s=q2, scale_loss=self.scale_loss)
            ds_loss = (ds12 + ds21) / 2
            loss = barlow_loss + ds_loss.mul(self.llamb)        

        return loss + class_loss


    def on_train_batch_end(self, outputs: Dict[str, Any], batch: Sequence[Any], batch_idx: int):

        # update momentum backbone and projector
        momentum_pairs = self.momentum_pairs
        for mp in momentum_pairs:
            self.momentum_updater.update(*mp)
            