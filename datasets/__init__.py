# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

from argparse import Namespace
from .radartrack import build as build_e2e_radartrack

def build_dataset(dataset: str, split: str, args: Namespace):
    if dataset == 'rdtrack' or dataset == 'ratrack':
        return build_e2e_radartrack(split, args)

    raise ValueError(f'Dataset {dataset} is not supported')
