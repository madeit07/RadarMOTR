# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


"""
Backbone modules.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models._api import get_model

from util.misc import NestedTensor

from .circular_resnet import circular_resnet18
from ..position_encoding import build_position_encoding


BACKBONES = {
    'circular_resnet18': circular_resnet18
}

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, backbone_name: str, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        if not train_backbone:
            for name, parameter in backbone.named_parameters():
                if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                    parameter.requires_grad_(False)

        if backbone_name in ('resnet18', 'circular_resnet18', 'resnet34'):
            self.num_channels = [128, 256, 512]
        elif backbone_name in ('resnet50', 'resnet101', 'resnet152'):
            self.num_channels = [512, 1024, 2048]
        else:
            raise NotImplementedError(f'Backbone {backbone_name} not supported.')

        self.strides = [8, 16, 32]

        if return_interm_layers:
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
        else:
            return_layers = {'layer4': "0"}
            self.strides = self.strides[-1]
            self.num_channels = self.num_channels[-1]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 weights=None):
        norm_layer = FrozenBatchNorm2d

        backbone_cfg = {
            'replace_stride_with_dilation': [False, False, dilation],
            'weights': weights,
            'norm_layer': norm_layer
        }

        if name in BACKBONES:
            backbone = BACKBONES[name](**backbone_cfg)
        else:
            backbone = get_model(name, **backbone_cfg)

        super().__init__(backbone, name, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: list[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    if args.use_circular_padding:
        assert args.backbone == 'circular_resnet18', 'Circular padding is only supported for backbone \"circular_resnet18\"'

    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation, args.backbone_weights)
    model = Joiner(backbone, position_embedding)
    return model
