# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

import math
from argparse import Namespace

import torch
from torch import nn

from models.structures import Instances


class QueryInteractionBase(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()
        self.args = args

    def _build_layers(self, dim_in: int, hidden_dim: int, dim_out: int, args: Namespace):
        raise NotImplementedError()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _select_active_tracks(self, data: dict) -> Instances:
        raise NotImplementedError()

    def _update_track_embedding(self, track_instances):
        raise NotImplementedError()


class QueryInteractionModule(QueryInteractionBase):
    def __init__(self, dim_in: int, hidden_dim: int, dim_out: int, args: Namespace):
        super().__init__(args)
        raise NotImplementedError()


class QueryInteractionModuleV2(QueryInteractionBase):
    def __init__(self, dim_in: int, hidden_dim: int, dim_out: int, args: Namespace):
        super().__init__(args)
        self.update_query_pos = args.qim_update_query_pos
        self.score_threshold = args.qim_score_threshold
        self.iou_threshold = args.qim_iou_threshold

        self._build_layers(dim_in, hidden_dim, dim_out, args)
        self._reset_parameters()

    def _build_layers(self, dim_in: int, hidden_dim: int, dim_out: int, args: Namespace):
        dropout = args.qim_dropout

        self.self_attn = nn.MultiheadAttention(dim_in, 8, dropout)
        self.linear1 = nn.Linear(dim_in, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim_in)

        if self.update_query_pos:
            self.linear_pos1 = nn.Linear(dim_in, hidden_dim)
            self.linear_pos2 = nn.Linear(hidden_dim, dim_in)
            self.dropout_pos1 = nn.Dropout(dropout)
            self.dropout_pos2 = nn.Dropout(dropout)
            self.norm_pos = nn.LayerNorm(dim_in)

        self.linear_feat1 = nn.Linear(dim_in, hidden_dim)
        self.linear_feat2 = nn.Linear(hidden_dim, dim_in)
        self.dropout_feat1 = nn.Dropout(dropout)
        self.dropout_feat2 = nn.Dropout(dropout)
        self.norm_feat = nn.LayerNorm(dim_in)

        self.norm1 = nn.LayerNorm(dim_in)
        self.norm2 = nn.LayerNorm(dim_in)
        if self.update_query_pos:
            self.norm3 = nn.LayerNorm(dim_in)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        if self.update_query_pos:
            self.dropout3 = nn.Dropout(dropout)
            self.dropout4 = nn.Dropout(dropout)

        self.activation = nn.ReLU(True)

    def _select_active_tracks(self, data: dict[str]) -> Instances:
        track_instances: Instances = data['track_instances']
        if self.training:
            active_idxes = (track_instances.obj_idxes >= 0) | (track_instances.scores > self.score_threshold)
            active_track_instances = track_instances[active_idxes]
            active_track_instances.obj_idxes[active_track_instances.iou <= self.iou_threshold] = -1
        else:
            active_track_instances = track_instances[track_instances.obj_idxes >= 0]

        return active_track_instances

    def _update_track_embedding(self, track_instances: Instances) -> Instances:
        is_pos = track_instances.scores > self.score_threshold
        track_instances.ref_pts[is_pos] = track_instances.pred_boxes.detach().clone()[is_pos]

        out_embed = track_instances.output_embedding
        query_feat = track_instances.query_pos
        query_pos = pos2posemb(track_instances.ref_pts)
        q = k = query_pos + out_embed

        tgt = out_embed
        tgt2 = self.self_attn(q[:, None], k[:, None], value=tgt[:, None])[0][:, 0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        if self.update_query_pos:
            query_pos2 = self.linear_pos2(self.dropout_pos1(self.activation(self.linear_pos1(tgt))))
            query_pos = query_pos + self.dropout_pos2(query_pos2)
            query_pos = self.norm_pos(query_pos)
            track_instances.query_pos = query_pos

        query_feat2 = self.linear_feat2(self.dropout_feat1(self.activation(self.linear_feat1(tgt))))
        query_feat = query_feat + self.dropout_feat2(query_feat2)
        query_feat = self.norm_feat(query_feat)
        track_instances.query_pos[is_pos] = query_feat[is_pos]

        return track_instances

    def forward(self, data: dict[str]) -> Instances:
        active_track_instances = self._select_active_tracks(data)
        active_track_instances = self._update_track_embedding(active_track_instances)
        return active_track_instances


def pos2posemb(pos, num_pos_feats=64, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    posemb = pos[..., None] / dim_t
    posemb = torch.stack((posemb[..., 0::2].sin(), posemb[..., 1::2].cos()), dim=-1).flatten(-3)
    return posemb


def build(qim_module: str, dim_in: int, hidden_dim: int, dim_out: int, args: Namespace) -> QueryInteractionBase:
    available_modules = {
        'QIM': QueryInteractionModule,
        'QIMv2': QueryInteractionModuleV2,
    }
    assert qim_module in available_modules, f'Invalid query interaction layer: {qim_module}'

    return available_modules[qim_module](dim_in, hidden_dim, dim_out, args)
