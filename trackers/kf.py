from argparse import Namespace

import torch
import numpy as np

from scipy.ndimage import label
from skimage.measure import regionprops

from trackers.tracker_base import Tracker
from trackers.util.kf import RDSort


def _clip(x, x_min, x_max):
    return max(x_min, min(x, x_max))

class KFTracker(Tracker):
    def __init__(self, args: Namespace):
        super().__init__()

        self.max_age: int = args.max_age
        self.min_hits: int = args.min_hits
        self.dist_threshold: float = args.dist_threshold
        self.range_resolution: float = args.range_resolution
        self.velocity_resolution: float = args.velocity_resolution
        self.fps: int = args.fps

        self.model = RDSort(self.max_age, self.min_hits, self.dist_threshold, self.fps)

    def track_frame(self, frame: int, data: tuple[torch.Tensor, dict[str, torch.Tensor]]):
        image, targets = data

        img_h, img_w = targets['orig_size'].unbind()
        img_h, img_w = img_h.item(), img_w.item()
        max_v = img_w * self.velocity_resolution
        max_r = img_h * self.range_resolution

        detections = targets['proposals'][:, :4].numpy()

        # xc,yc,w,h -> x,y,w,h
        detections[:, 0] -= (detections[:, 2] / 2)
        detections[:, 1] -= (detections[:, 3] / 2)

        centroids = []
        for detection in detections:
            x_min, x_max, y_min, y_max = int(detection[0]), int(detection[0] + detection[2]), int(detection[1]), int(detection[1] + detection[3])
            x_min = _clip(x_min, 0, img_w - 1)
            x_max = _clip(x_max, 0, img_w - 1)
            y_min = _clip(y_min, 0, img_h - 1)
            y_max = _clip(y_max, 0, img_h - 1)

            partitioned_rd = image[0, y_min:y_max+1, x_min:x_max+1].numpy()
            threshold_probability = np.max(partitioned_rd) * 0.5

            bw = np.array(partitioned_rd)
            bw[bw < threshold_probability] = 0

            img_cc, _ = label(bw)
            cc = regionprops(img_cc)

            i = np.argmax([c.area for c in cc])
            c = np.array(cc[i].centroid)
            c[0] += y_min
            c[1] += x_min
            centroids.append(c)

        # r,v
        centroids = np.stack(centroids)

        centroids *= np.array([self.range_resolution, self.velocity_resolution])
        centroids[:, 0] = centroids[:, 0] * -1 + max_r
        centroids[:, 1] = centroids[:, 1] - max_v / 2

        order = np.arange(len(centroids)).reshape(-1, 1)
        dets = np.concatenate((centroids, order), axis=1)

        tracks = self.model.update(dets)

        indices = tracks[:, 3].astype(np.int32)
        tracks_boxes = detections[indices]
        ids = tracks[:, 2].astype(np.int32)

        identities = torch.tensor(ids, dtype=torch.int64)

        return torch.from_numpy(tracks_boxes), identities

    def reset(self):
        self.model.reset()
