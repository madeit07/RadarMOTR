# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch
import torch.nn.functional as F
from torchvision.ops.boxes import box_convert

from models.deformable_detr import SetCriterion, sigmoid_focal_loss
from models.matcher import HungarianMatcher
from models.structures import Boxes, Instances, matched_boxlist_iou
from util.box_ops import box_giou
from util.misc import accuracy, get_world_size, is_dist_avail_and_initialized


class ClipMatcher(SetCriterion):
    def __init__(self, num_classes: int, matcher: HungarianMatcher, loss_weights: dict[str, float],
                 losses: list[str], focal_alpha: float = 0.25, focal_gamma: float = 2):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            loss_weights: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__(num_classes, matcher, loss_weights, losses, focal_alpha, focal_gamma)

        self._num_samples = 0
        self._current_frame_idx = 0
        self._losses_dict: dict[str, torch.Tensor] = {}

    def init_clip(self):
        """Initialize the matcher for a single clip.
        """
        self._num_samples = 0
        self.sample_device = None
        self._current_frame_idx = 0
        self._losses_dict = {}

    def get_clip_loss(self):
        return self._losses_dict

    def _step(self):
        self._current_frame_idx += 1

    def get_num_boxes(self, num_samples: int):
        num_boxes = torch.as_tensor(num_samples, dtype=torch.float, device=self.sample_device)

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)

        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        return num_boxes

    def get_loss(self, loss: str, outputs: dict[str],
                 gt_instances: list[Instances],
                 indices: list[tuple],
                 num_boxes: int, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, gt_instances, indices, num_boxes, **kwargs)

    def loss_boxes(self, outputs: dict[str], gt_instances: list[Instances], indices: list[tuple], num_boxes: int):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        # We ignore the regression loss of the track-disappear slots.
        #TODO: Make this filter process more elegant.
        filtered_idx = []
        for src_per_img, tgt_per_img in indices:
            keep = tgt_per_img != -1
            filtered_idx.append((src_per_img[keep], tgt_per_img[keep]))
        indices = filtered_idx
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([gt_per_img.boxes[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0)

        # for pad target, don't calculate regression loss, judged by whether obj_id=-1
        target_obj_ids = torch.cat([gt_per_img.obj_ids[i] for gt_per_img, (_, i) in zip(gt_instances, indices)], dim=0) # size(16)
        mask = (target_obj_ids != -1)

        loss_bbox = F.l1_loss(src_boxes[mask], target_boxes[mask], reduction='none')
        loss_giou = 1 - torch.diag(box_giou(
            box_convert(src_boxes[mask], 'cxcywh', 'xyxy'),
            box_convert(target_boxes[mask], 'cxcywh', 'xyxy')))

        losses = {
            'loss_bbox': loss_bbox.sum() / num_boxes,
            'loss_giou': loss_giou.sum() / num_boxes,
        }

        return losses

    def loss_labels(self, outputs: dict[str], gt_instances: list[Instances], indices: list[tuple], num_boxes: int, log=False):
        """Classification loss (Focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # The matched gt for disappear track query is set -1.
        labels = []
        for gt_per_img, (_, J) in zip(gt_instances, indices):
            labels_per_img = torch.ones_like(J)
            # set labels of track-appear slots to 0.
            if len(gt_per_img) > 0:
                labels_per_img[J != -1] = gt_per_img.labels[J[J != -1]]
            labels.append(labels_per_img)
        target_classes_o = torch.cat(labels)
        target_classes[idx] = target_classes_o

        gt_labels_target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[:, :, :-1]  # no loss for the last (background) class
        gt_labels_target = gt_labels_target.to(src_logits)
        loss_ce = sigmoid_focal_loss(src_logits.flatten(1),
                                     gt_labels_target.flatten(1),
                                     alpha=self.focal_alpha,
                                     gamma=self.focal_gamma,
                                     num_boxes=num_boxes,
                                     mean_in_dim1=False)

        losses = {
            'loss_ce': loss_ce.sum()
        }

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    def _match_decoder_layer(self, unmatched_outputs: dict[str],
                             untracked_gt_instances: Instances,
                             unmatched_track_indices: torch.Tensor,
                             untracked_tgt_indices: torch.Tensor):
        """Match for a single decoder layer.
        """
        new_track_indices = self.matcher(unmatched_outputs, [untracked_gt_instances])  # list[tuple(src_idx, tgt_idx)]

        src_idx = new_track_indices[0][0]
        tgt_idx = new_track_indices[0][1]
        # concat src and tgt.
        new_matched_indices = torch.stack([unmatched_track_indices[src_idx], untracked_tgt_indices[tgt_idx]], dim=1)
        return new_matched_indices

    def match(self, outputs: dict[str], gt_instances: Instances) -> Instances:
        """Performs a matching for a single frame.

        Args:
            outputs (dict[str]): Predictions.
            gt_instances (Instances): The ground truth tracks to compare to.

        Returns:
            Instances: Predicated tracks with matches included.
        """
        track_instances: Instances = outputs['track_instances']
        pred_logits = track_instances.pred_logits
        pred_boxes = track_instances.pred_boxes
        obj_idxes = gt_instances.obj_ids

        device = pred_logits.device
        self.sample_device = device

        predictions = {
            'pred_logits': pred_logits.unsqueeze(0),
            'pred_boxes': pred_boxes.unsqueeze(0),
        }

        # step1. inherit and update the previous tracks.
        num_disappear_track = 0
        track_instances.matched_gt_idxes[:] = -1
        i, j = torch.where(track_instances.obj_idxes[:, None] == obj_idxes)
        track_instances.matched_gt_idxes[i] = j

        full_track_indices = torch.arange(len(track_instances), dtype=torch.long, device=device)
        matched_track_indices = (track_instances.obj_idxes >= 0)  # occu
        prev_matched_indices = torch.stack([
            full_track_indices[matched_track_indices],
            track_instances.matched_gt_idxes[matched_track_indices]], dim=1)

        # step2. select the unmatched slots.
        # note that the FP tracks whose obj_idxes are -2 will not be selected here.
        unmatched_track_indices = full_track_indices[track_instances.obj_idxes == -1]

        # step3. select the untracked gt instances (new tracks).
        tgt_indices = track_instances.matched_gt_idxes
        tgt_indices = tgt_indices[tgt_indices != -1]

        tgt_state = torch.zeros(len(gt_instances), device=device)
        tgt_state[tgt_indices] = 1
        untracked_tgt_indices = torch.arange(len(gt_instances), device=device)[tgt_state == 0]
        # untracked_tgt_indexes = select_unmatched_indexes(tgt_indexes, len(gt_instances))
        untracked_gt_instances = gt_instances[untracked_tgt_indices]

        # step4. do matching between the unmatched slots and GTs.
        unmatched_outputs = {
            'pred_logits': track_instances.pred_logits[unmatched_track_indices].unsqueeze(0),
            'pred_boxes': track_instances.pred_boxes[unmatched_track_indices].unsqueeze(0),
        }
        new_matched_indices = self._match_decoder_layer(unmatched_outputs,
                                                        untracked_gt_instances,
                                                        unmatched_track_indices,
                                                        untracked_tgt_indices)

        # step5. update obj_idxes according to the new matching result.
        track_instances.obj_idxes[new_matched_indices[:, 0]] = gt_instances.obj_ids[new_matched_indices[:, 1]].long()
        track_instances.matched_gt_idxes[new_matched_indices[:, 0]] = new_matched_indices[:, 1]

        # step6. calculate iou.
        active_idxes = (track_instances.obj_idxes >= 0) & (track_instances.matched_gt_idxes >= 0)
        active_track_boxes = track_instances.pred_boxes[active_idxes]
        if len(active_track_boxes) > 0:
            gt_boxes = gt_instances.boxes[track_instances.matched_gt_idxes[active_idxes]]
            active_track_boxes = box_convert(active_track_boxes, 'cxcywh', 'xyxy')
            gt_boxes = box_convert(gt_boxes, 'cxcywh', 'xyxy')
            track_instances.iou[active_idxes] = matched_boxlist_iou(Boxes(active_track_boxes), Boxes(gt_boxes))

        # step7. merge the unmatched pairs and the matched pairs.
        matched_indices = torch.cat([new_matched_indices, prev_matched_indices], dim=0)

        # step8. calculate losses.
        self._num_samples += len(gt_instances) + num_disappear_track
        fidx = self._current_frame_idx
        for loss in self.losses:
            new_track_loss = self.get_loss(loss,
                                           predictions,
                                           gt_instances=[gt_instances],
                                           indices=[(matched_indices[:, 0], matched_indices[:, 1])],
                                           num_boxes=1)
            self._losses_dict.update({f'frame_{fidx}_{key}': value for key, value in new_track_loss.items()})

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                unmatched_outputs_layer = {
                    'pred_logits': aux_outputs['pred_logits'][0, unmatched_track_indices].unsqueeze(0),
                    'pred_boxes': aux_outputs['pred_boxes'][0, unmatched_track_indices].unsqueeze(0),
                }
                new_matched_indices_layer = self._match_decoder_layer(unmatched_outputs_layer,
                                                                      untracked_gt_instances,
                                                                      unmatched_track_indices,
                                                                      untracked_tgt_indices)
                matched_indices_layer = torch.cat([new_matched_indices_layer, prev_matched_indices], dim=0)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = self.get_loss(loss,
                                           aux_outputs,
                                           gt_instances=[gt_instances],
                                           indices=[(matched_indices_layer[:, 0], matched_indices_layer[:, 1])],
                                           num_boxes=1)
                    self._losses_dict.update({f'frame_{fidx}_aux{i}_{key}': value for key, value in l_dict.items()})

        if 'ps_outputs' in outputs:
            ar = torch.arange(len(gt_instances), device=device)
            indices = [(ar, ar)]
            for i, ps_outputs in enumerate(outputs['ps_outputs']):
                l_dict = self.get_loss('boxes',
                                       ps_outputs,
                                       gt_instances=[gt_instances],
                                       indices=indices,
                                       num_boxes=1)
                self._losses_dict.update({f'frame_{fidx}_ps{i}_{key}': value for key, value in l_dict.items()})

        self._step()

        return track_instances

    def forward(self, outputs: dict[str]) -> dict[str, torch.Tensor]:
        num_samples = self.get_num_boxes(self._num_samples)
        # losses of each frame are calculated during the model's forwarding and are outputted by the model as outputs['losses_dict'].
        losses = outputs.pop('losses_dict')
        for loss_name in losses.keys():
            losses[loss_name] /= num_samples

        return losses

    def scale_loss(self, loss: dict[str, torch.Tensor]):
        """Scale the losses according to configured criterion loss weights.

        Args:
            loss (dict[str, torch.Tensor]): Total loss including for each frame.

        Returns:
            dict[str, torch.Tensor]: Scaled losses.
        """
        scaled_loss: dict[str, torch.Tensor] = {}
        for key, value in loss.items():
            for loss_name, weight in self._loss_weights.items():
                if loss_name in key:
                    scaled_loss[key] = value * weight
                    break

        return scaled_loss

    def sum_frame_loss(self, loss: dict[str, torch.Tensor], include_aux: bool = True, include_ps: bool = True):
        """Sums the loss calculated for each frame per category (bbox, ce, giou).

        Args:
            loss (dict[str, torch.Tensor]): Total loss including for each frame.
            include_aux (bool, optional): If aux loss should be included in the sum. Defaults to True.
            include_ps (bool, optional): If ps loss should be included in the sum. Defaults to True.

        Returns:
            dict[str, torch.Tensor]: Summed up losses.
        """
        sum_loss: dict[str, torch.Tensor] = {}

        for key, value in loss.items():
            if 'frame' not in key or 'loss' not in key:
                continue
            if not include_aux and 'aux' in key:
                continue
            if not include_ps and 'ps' in key:
                continue

            loss_name = '_'.join(key.rsplit('_', 2)[1:]) # Gets loss_giou from e.g. frame_0_aux0_loss_giou
            sum_loss.setdefault(loss_name, 0)
            sum_loss[loss_name] += value

        return sum_loss
