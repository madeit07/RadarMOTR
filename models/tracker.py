from argparse import Namespace
import torch
from models.structures import Instances

class RuntimeTrackerBase:
    def __init__(self, score_threshold: float = 0.6, filter_score_threshold: float = 0.5, miss_tolerance: int = 10):
        self.score_threshold = score_threshold
        self.filter_score_threshold = filter_score_threshold
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def reset(self):
        self.max_obj_id = 0

    def update(self, track_instances: Instances):
        device = track_instances.obj_idxes.device

        track_instances.disappear_time[track_instances.scores > self.score_threshold] = 0
        new_obj = (track_instances.obj_idxes == -1) & (track_instances.scores > self.score_threshold)
        disappeared_obj = (track_instances.obj_idxes >= 0) & (track_instances.scores < self.filter_score_threshold)
        num_new_objs = new_obj.sum().item()

        track_instances.obj_idxes[new_obj] = self.max_obj_id + torch.arange(num_new_objs, device=device)
        self.max_obj_id += num_new_objs

        track_instances.disappear_time[disappeared_obj] += 1
        to_del = disappeared_obj & (track_instances.disappear_time >= self.miss_tolerance)
        track_instances.obj_idxes[to_del] = -1


def build(args: Namespace) -> RuntimeTrackerBase:
    return RuntimeTrackerBase(args.score_threshold, args.filter_score_threshold, args.miss_tolerance)
