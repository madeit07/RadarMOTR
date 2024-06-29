# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
import math
import time
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from models.structures import Instances
from util import checkpoint
from util.misc import (MetricLogger, NestedTensor, SmoothedValue,
                       inverse_sigmoid, nested_tensor_from_tensors)

from .backbones.backbone import build_backbone
from .clip_matcher import ClipMatcher
from .deformable_detr import MLP
from .deformable_transformer_plus import (DeformableTransformer, build_deforamble_transformer,
                                          pos2posemb)
from .matcher import build_matcher
from .postprocess import TrackerPostProcess
from .qim import QueryInteractionBase
from .qim import build as build_qim
from .tracker import RuntimeTrackerBase
from .tracker import build as build_track_base


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class RadarMOTR(nn.Module):
    def __init__(self, backbone, transformer: DeformableTransformer, num_classes: int, num_queries: int, num_feature_levels: int,
                 criterion: ClipMatcher, qim: QueryInteractionBase, post_process: TrackerPostProcess, track_base: RuntimeTrackerBase,
                 aux_loss=True, with_box_refine=False, two_stage=False, memory_bank=None, use_grad_checkpointing=False, query_denoise: float = 0,
                 use_circular_padding=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()

        self.num_queries = num_queries
        self.qim = qim
        self.transformer = transformer
        self.num_classes = num_classes
        self.num_feature_levels = num_feature_levels
        self.use_grad_checkpointing = use_grad_checkpointing
        self.query_denoise = query_denoise
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage
        self.post_process = post_process
        self.track_base = track_base
        self.criterion = criterion
        self.memory_bank = memory_bank

        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.position = nn.Embedding(num_queries, 4)
        self.proposal_embed = nn.Embedding(1, hidden_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        if query_denoise:
            self.refine_embed = nn.Embedding(1, hidden_dim)

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []

            padding_mode = 'zeros'
            if use_circular_padding:
                padding_mode = 'circular'

            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1, padding_mode=padding_mode),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1, padding_mode=padding_mode),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1, padding_mode=padding_mode),
                    nn.GroupNorm(32, hidden_dim),
                )])

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        nn.init.uniform_(self.position.weight.data, 0, 1)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers
        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # HACK: hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # HACK: hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

        self.mem_bank_len = 0 if memory_bank is None else memory_bank.max_his_length

        self.track_keys = list(self._generate_empty_tracks()._fields.keys())

        # Internal metric logger
        # Used to profile performance of model internally
        self.add_profiler(MetricLogger(delimiter="  "))

    def add_profiler(self, logger: MetricLogger):
        logger.add_meter('time_backbone', SmoothedValue(fmt='{avg:.2f}'))
        logger.add_meter('time_transformer', SmoothedValue(fmt='{avg:.2f}'))
        logger.add_meter('time_qim', SmoothedValue(fmt='{avg:.2f}'))
        logger.add_meter('time_criterion', SmoothedValue(fmt='{avg:.2f}'))

        self.profiler = logger

    def _generate_empty_tracks(self, proposals: torch.Tensor = None):
        track_instances = Instances((1, 1))
        num_queries, d_model = self.query_embed.weight.shape  # (300, 512)
        device = self.query_embed.weight.device
        if proposals is None:
            track_instances.ref_pts = self.position.weight
            track_instances.query_pos = self.query_embed.weight
        else:
            track_instances.ref_pts = torch.cat([self.position.weight, proposals[:, :4]])
            track_instances.query_pos = torch.cat([self.query_embed.weight, pos2posemb(proposals[:, 4:], d_model) + self.proposal_embed.weight])

        track_instances.output_embedding = torch.zeros((len(track_instances), d_model), device=device)
        track_instances.obj_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.matched_gt_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.disappear_time = torch.zeros((len(track_instances), ), dtype=torch.long, device=device)
        track_instances.iou = torch.ones((len(track_instances),), dtype=torch.float, device=device)
        track_instances.scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.pred_boxes = torch.zeros((len(track_instances), 4), dtype=torch.float, device=device)
        track_instances.pred_logits = torch.zeros((len(track_instances), self.num_classes), dtype=torch.float, device=device)

        mem_bank_len = self.mem_bank_len
        track_instances.mem_bank = torch.zeros((len(track_instances), mem_bank_len, d_model), dtype=torch.float32, device=device)
        track_instances.mem_padding_mask = torch.ones((len(track_instances), mem_bank_len), dtype=torch.bool, device=device)
        track_instances.save_model_interval = torch.zeros((len(track_instances), ), dtype=torch.float32, device=device)

        return track_instances.to(device)

    def reset(self):
        self.track_base.reset()

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b, }
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    def _forward_backbone(self, samples: NestedTensor):
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        return srcs, masks, pos

    def _forward_single_image(self, samples: NestedTensor,
                              track_instances: Instances,
                              gtboxes: torch.Tensor = None):
        start_time = time.time()
        srcs, masks, pos = self._forward_backbone(samples)
        self.profiler.update(time_backbone=(time.time() - start_time))

        if gtboxes is not None:
            n_dt = len(track_instances)
            ps_tgt = self.refine_embed.weight.expand(gtboxes.size(0), -1)
            query_embed = torch.cat([track_instances.query_pos, ps_tgt])
            ref_pts = torch.cat([track_instances.ref_pts, gtboxes])
            attn_mask = torch.zeros((len(ref_pts), len(ref_pts)), dtype=bool, device=ref_pts.device)
            attn_mask[:n_dt, n_dt:] = True
        else:
            query_embed = track_instances.query_pos
            ref_pts = track_instances.ref_pts
            attn_mask = None

        start_time = time.time()
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact = \
            self.transformer(srcs,
                             masks,
                             pos,
                             query_embed,
                             ref_pts=ref_pts,
                             mem_bank=track_instances.mem_bank,
                             mem_bank_pad_mask=track_instances.mem_padding_mask,
                             attn_mask=attn_mask)
        self.profiler.update(time_transformer=(time.time() - start_time))

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        out['hs'] = hs[-1]

        return out

    def _post_process_single_image(self, pred: dict[str], track_instances: Instances,
                                   is_last: bool, gt_instances: Instances = None, *,
                                   is_prediction: bool = False):
        if self.query_denoise > 0:
            n_ins = len(track_instances)
            ps_logits = pred['pred_logits'][:, n_ins:]
            ps_boxes = pred['pred_boxes'][:, n_ins:]
            pred['hs'] = pred['hs'][:, :n_ins]
            pred['pred_logits'] = pred['pred_logits'][:, :n_ins]
            pred['pred_boxes'] = pred['pred_boxes'][:, :n_ins]
            pred['ps_outputs'] = [{'pred_logits': ps_logits, 'pred_boxes': ps_boxes}]

            if 'aux_outputs' in pred:
                for aux_outputs in pred['aux_outputs']:
                    pred['ps_outputs'].append({
                        'pred_logits': aux_outputs['pred_logits'][:, n_ins:],
                        'pred_boxes': aux_outputs['pred_boxes'][:, n_ins:],
                    })
                    aux_outputs['pred_logits'] = aux_outputs['pred_logits'][:, :n_ins]
                    aux_outputs['pred_boxes'] = aux_outputs['pred_boxes'][:, :n_ins]

        with torch.no_grad():
            if is_prediction:
                track_scores = pred['pred_logits'][0, :, 0].sigmoid()
            else:
                track_scores = pred['pred_logits'][0, :].sigmoid().max(dim=-1).values

        track_instances.scores = track_scores
        track_instances.pred_logits = pred['pred_logits'][0]
        track_instances.pred_boxes = pred['pred_boxes'][0]
        track_instances.output_embedding = pred['hs'][0]

        if is_prediction:
            # each track will be assigned an unique global id by the track base.
            self.track_base.update(track_instances)
        else:
            # Calculate loss
            assert gt_instances is not None
            # the track id will be assigned by the mather.
            pred['track_instances'] = track_instances
            start_time = time.time()
            track_instances = self.criterion.match(pred, gt_instances)
            self.profiler.update(time_criterion=(time.time() - start_time))

        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)

        if not is_last:
            qim_in = {
                'track_instances': track_instances
            }
            start_time = time.time()
            out_track_instances = self.qim(qim_in)
            self.profiler.update(time_qim=(time.time() - start_time))
            pred['track_instances'] = out_track_instances
        else:
            pred['track_instances'] = None

        return pred

    @torch.no_grad()
    def predict(self, img: torch.Tensor | NestedTensor,
                ori_img_size: tuple[int, int],
                track_instances: Instances = None,
                proposals: torch.Tensor = None) -> dict[str]:
        """Predicts the tracks for the current frame.

        Args:
            img (torch.Tensor | NestedTensor): Current frame image.
            ori_img_size (tuple[int, int]): Original size of the current frame image used to scale the output bounding boxes.
            track_instances (Instances, optional): Tracks from the previous frame. Defaults to None.
            proposals (torch.Tensor, optional): Detections for the current frame of an external detector. Defaults to None.

        The input boxes must be in format `cxcywh`.

        Returns:
            dict[str]: Dictionary containing the predicted tracks (`track_instances`) and optionally reference points.
                       The returned boxes are in format `xywh`.
        """
        if not isinstance(img, NestedTensor):
            img = nested_tensor_from_tensors([img])

        if track_instances is None:
            track_instances = self._generate_empty_tracks(proposals)
        else:
            track_instances = Instances.cat([
                self._generate_empty_tracks(proposals),
                track_instances])

        res = self._forward_single_image(img, track_instances)
        res = self._post_process_single_image(res, track_instances, is_last=False, is_prediction=True)

        track_instances = res['track_instances']
        track_instances = self.post_process(track_instances, ori_img_size)
        ret = {'track_instances': track_instances}

        if 'ref_pts' in res:
            ref_pts = res['ref_pts']
            img_h, img_w = ori_img_size
            scale_fct = torch.as_tensor([img_w, img_h], device=ref_pts.device)
            ref_pts = ref_pts * scale_fct[None]
            ret['ref_pts'] = ref_pts

        return ret

    def forward(self, data: dict[str, list]):
        self.criterion.init_clip()

        frames: list[torch.Tensor] = data['imgs']
        outputs = {
            'pred_logits': [],
            'pred_boxes': [],
        }
        track_instances: Optional[Instances] = None

        for frame_index, (frame, gt_instances, proposals) in enumerate(zip(frames, data['gt_instances'], data['proposals'])):
            frame.requires_grad = False
            is_last = frame_index == (len(frames) - 1)

            if self.query_denoise > 0:
                l_1 = l_2 = self.query_denoise
                gtboxes = gt_instances.boxes.clone()
                _rs = torch.rand_like(gtboxes) * 2 - 1
                gtboxes[..., :2] += gtboxes[..., 2:] * _rs[..., :2] * l_1
                gtboxes[..., 2:] *= 1 + l_2 * _rs[..., 2:]
            else:
                gtboxes = None

            if track_instances is None:
                track_instances = self._generate_empty_tracks(proposals)
            else:
                track_instances = Instances.cat([
                    self._generate_empty_tracks(proposals),
                    track_instances])

            if self.use_grad_checkpointing and frame_index < (len(frames) - 1):
                frame_res = self._forward_with_grad_cp(frame, track_instances, gtboxes)
            else:
                frame = nested_tensor_from_tensors([frame])
                frame_res = self._forward_single_image(frame, track_instances, gtboxes)

            frame_res = self._post_process_single_image(frame_res, track_instances, is_last, gt_instances)

            track_instances = frame_res['track_instances']
            outputs['pred_logits'].append(frame_res['pred_logits'])
            outputs['pred_boxes'].append(frame_res['pred_boxes'])

        outputs['track_instances'] = track_instances
        outputs['losses_dict'] = self.criterion.get_clip_loss()

        return outputs, self.profiler.get_global_avg_metrics()

    def _forward_with_grad_cp(self, frame: torch.Tensor, track_instances: Instances, gtboxes: torch.Tensor):
        """This uses gradient checkpointing.
        """
        def fn(frame, gtboxes, *args):
            frame = nested_tensor_from_tensors([frame])
            tmp = Instances((1, 1), **dict(zip(self.track_keys, args)))
            frame_res = self._forward_single_image(frame, tmp, gtboxes)

            results = [
                frame_res['pred_logits'],
                frame_res['pred_boxes'],
                frame_res['hs']
            ]

            if 'aux_outputs' in frame_res:
                results.extend(aux['pred_logits'] for aux in frame_res['aux_outputs'])
                results.extend(aux['pred_boxes'] for aux in frame_res['aux_outputs'])

            return tuple(results)

        args = [frame, gtboxes] + [track_instances.get(k) for k in self.track_keys]
        params = tuple(p for p in self.parameters() if p.requires_grad)
        tmp = checkpoint.CheckpointFunction.apply(fn, len(args), *args, *params)

        res = {
            'pred_logits': tmp[0],
            'pred_boxes': tmp[1],
            'hs': tmp[2],
        }

        # Aux outputs
        if self.aux_loss and len(tmp) > 3:
            n_decoders = self.transformer.n_decoders
            res['aux_outputs'] = [{
                'pred_logits': tmp[3 + i],
                'pred_boxes': tmp[3 + (n_decoders - 1) + i],
            } for i in range(n_decoders - 1)]

        return res


def build(args):
    num_classes = len(args.classes)
    assert num_classes > 0, 'At least one class must exist.'

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)

    qim = build_qim(args.query_interaction_layer,
                    dim_in=transformer.d_model,
                    hidden_dim=args.dim_feedforward,
                    dim_out=transformer.d_model * 2,
                    args=args)

    track_base = build_track_base(args)

    img_matcher = build_matcher(args)

    loss_weights = {
        'loss_ce': args.cls_loss_coef,
        'loss_bbox': args.bbox_loss_coef,
        'loss_giou': args.giou_loss_coef,
    }

    criterion = ClipMatcher(num_classes,
                            matcher=img_matcher,
                            loss_weights=loss_weights,
                            losses=['labels', 'boxes'],
                            focal_alpha=args.focal_alpha,
                            focal_gamma=args.focal_gamma)

    model = RadarMOTR(
        backbone,
        transformer,
        qim=qim,
        track_base=track_base,
        post_process=TrackerPostProcess(),
        num_feature_levels=args.num_feature_levels,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        criterion=criterion,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        use_grad_checkpointing=args.use_grad_checkpointing,
        query_denoise=args.query_denoise,
    )

    return model, criterion
