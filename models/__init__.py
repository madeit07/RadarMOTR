# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from argparse import Namespace
import torch

from .optimizer import build_adamW
from .radarmotr import build as build_radarmotr


def build_model(args: Namespace):
    return build_radarmotr(args)

def build_optimizer(model: torch.nn.Module, args: Namespace) -> torch.optim.Optimizer:
    return build_adamW(model, args)
