from argparse import Namespace

import torch
from torchvision.ops.boxes import box_convert

from trackers.tracker_base import Tracker
from trackers.util.sort import Sort


class SortTracker(Tracker):
    def __init__(self, args: Namespace):
        super().__init__()

        self.max_age: int = args.max_age
        self.min_hits: int = args.min_hits
        self.iou_threshold: float = args.iou_threshold
        self.range_resolution: float = args.range_resolution
        self.velocity_resolution: float = args.velocity_resolution
        self.fps: int = args.fps

        self.model = Sort(self.max_age, self.min_hits, self.iou_threshold, self.fps)

    def track_frame(self, frame: int, data: tuple[torch.Tensor, dict[str, torch.Tensor]]):
        _, targets = data

        img_h, img_w = targets['orig_size'].unbind()
        max_v = img_w * self.velocity_resolution
        max_r = img_h * self.range_resolution
        scale = torch.tensor([max_v, max_r, max_v, max_r])

        detections = targets['proposals'][:, :4]

        # xc,yc,w,h -> v,r,w,h (center)
        detections *= scale
        detections[:, 0] = detections[:, 0] - max_v / 2
        detections[:, 1] = detections[:, 1] * -1 + max_r

        tracks = self.model.update(detections.numpy())

        bbox_tracks = torch.from_numpy(tracks[:, :4])

        # v,r,w,h (center) -> xc,yc,w,h (center)
        bbox_tracks[:, 0] = bbox_tracks[:, 0] + max_v / 2
        bbox_tracks[:, 1] = (bbox_tracks[:, 1] - max_r) * -1
        bbox_tracks /= torch.tensor([self.velocity_resolution, self.range_resolution, self.velocity_resolution, self.range_resolution])

        # xc,yc,w,h (center) -> x,y,w,h (top left)
        bbox_xywh = box_convert(bbox_tracks, 'cxcywh', 'xywh')

        identities = torch.tensor(tracks[:, 4], dtype=torch.int64)

        return bbox_xywh, identities

    def reset(self):
        self.model.reset()
