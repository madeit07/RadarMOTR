import torch
from torch import nn
from torchvision.ops.boxes import box_convert

from models.structures import Instances


class TrackerPostProcess(nn.Module):
    """This module converts the model's output into the format expected by TrackEval"""
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, track_instances: Instances, target_size: tuple[int, int]) -> Instances:
        out_logits = track_instances.pred_logits
        out_bbox = track_instances.pred_boxes

        # prob = out_logits.sigmoid()
        scores = out_logits[..., 0].sigmoid()
        # scores, labels = prob.max(-1)

        # Convert from relative [0, 1] to absolute [0, image width/height] coordinates
        img_h, img_w = target_size
        scale = torch.as_tensor([img_w, img_h, img_w, img_h], device=out_bbox.device)
        boxes = out_bbox * scale[None, :]

        # Convert to [x, y, w, h] format
        boxes = box_convert(boxes, 'cxcywh', 'xywh')

        track_instances.boxes = boxes
        track_instances.scores = scores
        track_instances.labels = torch.full_like(scores, 0)
        # track_instances.remove('pred_logits')
        # track_instances.remove('pred_boxes')
        return track_instances
