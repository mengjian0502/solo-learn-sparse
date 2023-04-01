# Copyright 2022 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import argparse
from typing import Any, List, Sequence

import torch
import torch.nn as nn
from solo.losses.barlow import barlow_loss_func, distill_loss_func
from solo.methods.base import BaseMethod
from solo.methods.lightssl import Slicer


class BarlowTwins(BaseMethod):
    def __init__(
        self, args, proj_hidden_dim: int, proj_output_dim: int, lamb: float, scale_loss: float, **kwargs
    ):
        """Implements Barlow Twins (https://arxiv.org/abs/2103.03230)

        Args:
            proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
            proj_output_dim (int): number of dimensions of projected features.
            lamb (float): off-diagonal scaling factor for the cross-covariance matrix.
            scale_loss (float): scaling factor of the loss.
        """

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

        # slicer
        self.slicer = Slicer(model=self.backbone, train_steps=args.train_steps, interval=args.train_steps, scale=0.5)
        self.alpha = 0.9

    @staticmethod
    def add_model_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parent_parser = super(BarlowTwins, BarlowTwins).add_model_specific_args(parent_parser)
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

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for Barlow Twins reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of Barlow loss and classification loss.
        """

        # out = super().training_step(batch, batch_idx)

        _, X, targets = batch

        X = [X] if isinstance(X, torch.Tensor) else X

        # check that we received the desired number of crops
        assert len(X) == self.num_crops

        # outs = [self.base_training_step(x, targets) for x in X[: self.num_large_crops]]
        
        outs = []
        for i, x in enumerate(X[: self.num_large_crops]):
            # forward pass
            out = self.base_training_step(x, targets)
            if i == 0:
                self.slicer.activate_mask()
            else:
                self.slicer.remove_mask()
            outs.append(out)

        # mirrored outputs
        routs = []
        with torch.no_grad():
            for i, x in enumerate(X[::-1]):
                out = self.base_training_step(x, targets)
                if i == 0:
                    self.slicer.activate_mask()
                else:
                    self.slicer.remove_mask()
                routs.append(out)
        
        outs = {k: [out[k] for out in outs] for k in outs[0].keys()}
        routs = {k: [rout[k] for rout in routs] for k in routs[0].keys()}

        if self.multicrop:
            multicrop_outs = [self.multicrop_forward(x) for x in X[self.num_large_crops :]]
            for k in multicrop_outs[0].keys():
                outs[k] = outs.get(k, []) + [out[k] for out in multicrop_outs]

        # loss and stats
        outs["loss"] = sum(outs["loss"]) / self.num_large_crops
        outs["acc1"] = sum(outs["acc1"]) / self.num_large_crops
        outs["acc5"] = sum(outs["acc5"]) / self.num_large_crops

        metrics = {
            "train_class_loss": outs["loss"],
            "train_acc1": outs["acc1"],
            "train_acc5": outs["acc5"],
        }

        self.log_dict(metrics, on_epoch=True, sync_dist=True)

        if self.knn_eval:
            targets = targets.repeat(self.num_large_crops)
            mask = targets != -1
            self.knn(
                train_features=torch.cat(outs["feats"][: self.num_large_crops])[mask].detach(),
                train_targets=targets[mask],
            )


        class_loss = outs["loss"]
        z1, z2 = outs["z"]  # z1 is the output of the dense (teacher) from img1, z2 is the output of sparse (student) from img2
        zd, zk = routs["z"] # zd is the output of dense (teacher) from img2, zk is the output of sparse (student) from img1

        # ------- barlow twins loss -------
        loss12 = barlow_loss_func(z1, z2, lamb=self.lamb, scale_loss=self.scale_loss)
        loss21 = barlow_loss_func(zd, zk, lamb=self.lamb, scale_loss=self.scale_loss)
        barlow_loss = (loss12 + loss21) / 2

        # # ------- symmetric log distillation loss -------
        # ds12 = distill_loss_func(t=z1, s=zk, scale_loss=self.scale_loss)
        # ds21 = distill_loss_func(t=zd, s=z2, scale_loss=self.scale_loss)
        # ds_loss = (ds12 + ds21) / 2

        # ------- symmetric BT distillation loss -------
        ds12 = barlow_loss_func(z1, zk, lamb=self.lamb, scale_loss=self.scale_loss)
        ds21 = barlow_loss_func(zd, z2, lamb=self.lamb, scale_loss=self.scale_loss)
        ds_loss = (ds12 + ds21) / 2

        loss = self.alpha * barlow_loss + (1-self.alpha) * ds_loss

        self.log("train_barlow_loss", barlow_loss, on_epoch=True, sync_dist=True)
        self.log("train_distill_loss", ds_loss, on_epoch=True, sync_dist=True)
        
        self.prune_step()
        return loss + class_loss

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int, dataloader_idx: int = None):
        self.slicer.activate_mask()
        return super().validation_step(batch, batch_idx, dataloader_idx)

    def prune_step(self):
        self.slicer.step()
        s, _ = self.slicer.get_sparsity()
        metrics = {"sparsity": s}

        self.log_dict(metrics, on_epoch=True, sync_dist=True)