from argparse import Namespace
from copy import deepcopy

import torch

from trackers.tracker_base import Tracker
from models.radarmotr import RadarMOTR
from models.structures import Instances
from datasets.data_prefetcher import data_dict_to_device
from models import build_model
from util.tool import load_model


class RadarMOTRTracker(Tracker):
    def __init__(self, args: Namespace):
        super().__init__()

        self.device = torch.device(args.device)
        self.score_threshold = args.score_threshold
        self.area_threshold = args.area_threshold

        self.model = self.build_model(self.device, args)

        self.tracks: Instances = None

    def build_model(self, device, args: Namespace) -> RadarMOTR:
        model, _ = build_model(args)
        model = load_model(model, args.model_path)
        model.eval()

        model.to(device, non_blocking=True)

        return model

    @staticmethod
    def filter_by_score(dt_instances: Instances, score_threshold: float) -> Instances:
        keep = dt_instances.scores > score_threshold
        keep &= dt_instances.obj_idxes >= 0
        return dt_instances[keep]

    @staticmethod
    def filter_by_area(dt_instances: Instances, area_threshold: float) -> Instances:
        areas = dt_instances.boxes[:, 2] * dt_instances.boxes[:, 3]
        keep = areas > area_threshold
        return dt_instances[keep]

    def track_frame(self, frame: int, data: tuple[torch.Tensor, dict[str, torch.Tensor]]):
        data = data_dict_to_device(data, self.device)
        image, targets = data
        proposals = targets['proposals']
        image_size = tuple(targets['orig_size'].tolist())

        if self.tracks is not None:
            self.tracks.remove('boxes')
            self.tracks.remove('labels')

        res = self.model.predict(image, image_size, self.tracks, proposals)
        self.tracks = res['track_instances']

        dt_instances = deepcopy(self.tracks)

        dt_instances = self.filter_by_score(dt_instances, self.score_threshold)
        dt_instances = self.filter_by_area(dt_instances, self.area_threshold)

        bbox_xywh = dt_instances.boxes
        identities = dt_instances.obj_idxes

        return bbox_xywh, identities

    def reset(self):
        self.model.reset()
        self.tracks = None

