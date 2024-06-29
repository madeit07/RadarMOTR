from __future__ import annotations

import configparser
import csv
import os
import random
from argparse import Namespace

import numpy as np
import pandas as pd
import cv2
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

import datasets.transforms as T
import datasets.radar_transforms as RDT
from models.structures.instances import Instances

DATASETS = ['rdtrack', 'ratrack']

# Calculated mean and std of train and val dataset with `tools/image_mean_std.py`
STANDARDIZATION = {
    'rdtrack': {
        'mean': [0.2445, 0.2445, 0.2445],
        'std': [0.0394, 0.0394, 0.0394],
    },
    'ratrack': {
        'mean': [0.3084, 0.3084, 0.3084],
        'std': [0.0746, 0.0746, 0.0746],
    },
}


def get_image(path: str) -> torch.Tensor:
    """Returns the image given by path as a tensor and the height and width of the image.

    Args:
        path (str): Path to the image to convert to a tensor.

    Returns:
        tuple[Tensor, tuple[int, int]]: Image tensor in format CHW.
    """
    cv_image = cv2.imread(path)
    h, w, c = cv_image.shape
    assert w > 0 and h > 0, f'Invalid image {path} with w={w} h={h}'

    image = TF.to_tensor(cv_image)

    return image


def get_sequence_paths(data_path: str, dataset: str, split: str) -> list[tuple[str, str]]:
    split_name = f'{dataset}-{split}'
    seqmap = os.path.join(data_path, 'seqmaps', f'{split_name}.txt')
    split_dir = os.path.join(data_path, split_name)

    sequences: list[tuple[str, str]] = []
    with open(seqmap, 'r', newline='') as f:
        seq_reader = csv.reader(f)
        sequences = [(seq[0], os.path.join(split_dir, seq[0])) for seq in seq_reader if seq[0] != 'name']

    return sequences


def parse_sequence_info(sequence_dir: str):
    seqinfo_path = os.path.join(sequence_dir, 'seqinfo.ini')
    if not os.path.exists(seqinfo_path):
        raise FileNotFoundError(f'Sequence info ini in {sequence_dir} is missing.')

    seqinfo = configparser.ConfigParser()
    seqinfo.read(seqinfo_path)

    return {
        'id': seqinfo.get('Sequence', 'name', fallback=None),
        'name': seqinfo.get('Sequence', 'desc', fallback=None),
        'measurement': seqinfo.get('Sequence', 'meas', fallback=None),
        'radar': int(seqinfo.get('Sequence', 'radar', fallback='-1')),
        'len': int(seqinfo.get('Sequence', 'seqLength', fallback='-1')),
        'fps': int(seqinfo.get('Sequence', 'frameRate', fallback='-1')),
        'image_dir': seqinfo.get('Sequence', 'imDir', fallback='img1'),
        'image_ext': seqinfo.get('Sequence', 'imExt', fallback='.png'),
        'width': int(seqinfo.get('Image', 'width', fallback='0')),
        'height': int(seqinfo.get('Image', 'height', fallback='0')),
        'min': int(seqinfo.get('Image', 'min', fallback='-1')),
        'max': int(seqinfo.get('Image', 'max', fallback='-1')),
        'minQ': int(seqinfo.get('Image', 'minQ', fallback='-1')),
    }


def _targets_to_instances(targets: dict, img_shape) -> Instances:
    gt_instances = Instances(tuple(img_shape))

    gt_instances.boxes = targets['boxes']
    gt_instances.labels = targets['labels']
    gt_instances.obj_ids = targets['obj_ids']

    return gt_instances


class RadarTrack(Dataset):
    MOT_DET_COLUMNS = ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'ignored1', 'ignored2']
    MOT_DET_DTYPES = {
        'frame': np.int64,
        'id': np.int64,
        'bb_left': np.float32,
        'bb_top': np.float32,
        'bb_width': np.float32,
        'bb_height': np.float32,
        'conf': np.float32,
    }

    MOT_GT_COLUMNS =  ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height', 'consider_entry', 'class', 'visibility']
    MOT_GT_DTYPES = {
        'frame': np.int64,
        'id': np.int64,
        'bb_left': np.float32,
        'bb_top': np.float32,
        'bb_width': np.float32,
        'bb_height': np.float32,
        'consider_entry': np.int32,
        'class': np.int32,
        'visibility': np.float32,
    }

    FRAME_FILENAME_FORMAT = '{frame:06d}.png'

    def __init__(self, data_path: str, dataset: str = 'rdtrack', split: str = 'train', transforms: T.MotCompose = None, *,
                 sample_mode: str = 'fixed_interval', sample_size: int = 1, sample_stride: int = 1):
        if not os.path.exists(data_path):
            raise ValueError(f'Data path {data_path} does not exist.')

        self.dataset = dataset
        self.data_path = data_path

        assert sample_size > 0, 'Sample size must be greater 0.'
        assert sample_stride > 0, 'Sample stride must be greater 0.'

        self.sample_mode = sample_mode
        self.sample_size = sample_size
        self.sample_stride = sample_stride

        self.transforms = transforms

        self._sequences = self._parse_sequences(data_path, dataset, split)
        self._frame_indices = [(seq_id, t) for seq_id, seq in self._sequences.items() for t in range(len(seq))]

    @property
    def sequences(self):
        return self._sequences

    @property
    def frame_indices(self):
        return self._frame_indices

    def __len__(self) -> int:
        return len(self._frame_indices)

    def get_continuous_frames(self, seq_id: str, indices: list[int]):
        return zip(*[self.get_frame_data(seq_id, t) for t in indices])

    def get_frame_data(self, seq_id: str, t: int):
        sequence = self.sequences[seq_id]
        return sequence[t]

    def random_sample_indices(self, seq_id: str, t: int):
        stride = random.randint(1, self.sample_stride)
        t_max = len(self.sequences[seq_id]) - 1

        return [min(t + stride * i, t_max) for i in range(self.sample_size)]

    def fixed_sample_indices(self, seq_id: str, t: int):
        t_max = len(self.sequences[seq_id]) - 1

        return [min(t + self.sample_stride * i, t_max) for i in range(self.sample_size)]

    def get_indices(self, mode: str, seq_id: str, t: int):
        modes = {
            'random_interval': self.random_sample_indices,
            'fixed_interval': self.fixed_sample_indices,
        }
        assert mode in modes, f'Unsupported sample mode {mode}'
        return modes[mode](seq_id, t)

    def __getitem__(self, index: int) -> tuple[dict[str, list], dict]:
        seq_id, t = self.frame_indices[index]
        indices = self.get_indices(self.sample_mode, seq_id, t)
        images, targets = self.get_continuous_frames(seq_id, indices)

        # Here the bounding box format is xywh
        # But when MotTranslateBoxes transform is specified they will be converted to cxcywh

        if self.transforms:
            images, targets = self.transforms(images, targets)

        gt_instances, proposals = [], []
        for img_i, targets_i in zip(images, targets):
            gt_instances_i = _targets_to_instances(targets_i, img_i.shape[1:3])

            gt_instances.append(gt_instances_i)
            proposals.append(targets_i['proposals'])

        return {
            'imgs': images,
            'proposals': proposals,
            'gt_instances': gt_instances,
        }

    def _parse_sequences(self, data_path: str, dataset: str, split: str) -> dict[str, RadarSequence]:
        sequence_paths = get_sequence_paths(data_path, dataset, split)
        return {seq_id: RadarSequence(seq_path) for seq_id, seq_path in sequence_paths}

class RadarSequence(Dataset):
    """Represents one sequence recorded on a multi radar system containing the tracks per each frame.
    """
    def __init__(self, sequence_dir: str):
        self.sequence_dir = sequence_dir

        self.dtype_box = torch.float32
        self.dtype_id = torch.int64

        self._info = parse_sequence_info(sequence_dir)
        self._frames = self.parse_sequence(sequence_dir)

    @property
    def id(self) -> str:
        return self._info['id']

    @property
    def name(self) -> str:
        return self._info['name']

    @property
    def measurement(self) -> str:
        return self._info['measurement']

    @property
    def radar(self) -> int:
        return self._info['radar']

    def __getitem__(self, t: int):
        try:
            frame = self._frames[t]
        except IndexError:
            print(self, t)
            raise

        image = get_image(frame['path'])
        image_size = image.shape[1:] # Skip channels

        id_offset = len(self) * 10000  # 10000 unique ids is enough for a video.

        targets: dict[str, torch.Tensor] = {}

        targets['frame_idx'] = frame['index']
        targets['size'] = torch.as_tensor(image_size)
        targets['orig_size'] = torch.as_tensor(image_size)

        targets['proposals'] = frame['detections']
        targets['boxes'] = frame['boxes']
        targets['obj_ids'] = frame['ids'] + id_offset
        targets['labels'] = torch.zeros(frame['ids'].shape, dtype=frame['ids'].dtype)

        return image, targets

    def __len__(self):
        return len(self._frames)

    def __str__(self) -> str:
        return f'SEQ: {self.id} [{self.measurement}/{self.name}/Radar={self.radar}] (N_FRAMES: {len(self)})'

    def __repr__(self) -> str:
        return self.__str__()

    def parse_detections(self, file: str):
        df = pd.read_csv(file, names=RadarTrack.MOT_DET_COLUMNS, dtype=RadarTrack.MOT_DET_DTYPES)

        select_bb = ['bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf']
        return {int(k): torch.tensor(v.values, dtype=self.dtype_box) for k, v in df.groupby('frame')[select_bb]}

    def parse_tracks(self, file: str):
        df = pd.read_csv(file, names=RadarTrack.MOT_GT_COLUMNS, dtype=RadarTrack.MOT_GT_DTYPES)
        df = df[df['consider_entry'] > 0]

        select_bb = ['bb_left', 'bb_top', 'bb_width', 'bb_height']
        frame_grp = df.groupby('frame')
        return {
            int(k): {
                'ids': torch.tensor(v_id.values, dtype=self.dtype_id),
                'boxes': torch.tensor(v_bb.values, dtype=self.dtype_box),
            } for (k, v_bb), (_, v_id) in zip(frame_grp[select_bb], frame_grp['id'])
        }

    def parse_sequence(self, sequence_dir: str) -> list[dict[str]]:
        det_file = os.path.join(sequence_dir, 'det', 'det.txt')
        gt_file = os.path.join(sequence_dir, 'gt', 'gt.txt')
        img_dir = os.path.join(sequence_dir, 'img1')

        detections = self.parse_detections(det_file)
        tracks = self.parse_tracks(gt_file)

        def no_boxes(include_conf: bool = False):
            dim = 5 if include_conf else 4
            return torch.as_tensor([], dtype=self.dtype_box).reshape(-1, dim)

        def no_ids():
            return torch.as_tensor([], dtype=self.dtype_id)

        return [{
                    'index': torch.as_tensor(frame_idx),
                    'path': os.path.join(img_dir, RadarTrack.FRAME_FILENAME_FORMAT.format(frame=frame_idx)),
                    'detections': detections.get(frame_idx, no_boxes(include_conf=True)),
                    'ids': tracks.get(frame_idx, {}).get('ids', no_ids()),
                    'boxes': tracks.get(frame_idx, {}).get('boxes', no_boxes()),
                } for frame_idx in range(1, self._info['len'] + 1)]


class RadarTrackSequences(RadarTrack):
    def __init__(self, data_path: str, dataset: str = 'rdtrack', split: str = 'val', transforms: T.MotCompose = None):
        super().__init__(data_path, dataset, split, transforms)

        self._sequence_order = list(self.sequences.items())

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[dict[str, list], dict]:
        seq_id, seq = self._sequence_order[index]
        indices = torch.arange(len(seq), dtype=torch.int64)
        images, targets = self.get_continuous_frames(seq_id, indices)

        if self.transforms:
            images, targets = self.transforms(images, targets)

        gt_instances, proposals = [], []
        for img_i, targets_i in zip(images, targets):
            gt_instances_i = _targets_to_instances(targets_i, img_i.shape[1:3])

            gt_instances.append(gt_instances_i)
            proposals.append(targets_i['proposals'])

        return {
            'imgs': images,
            'proposals': proposals,
            'gt_instances': gt_instances,
        }


def build_transforms(dataset: str, split: str, args: Namespace) -> T.MotCompose:
    transformations = [
        # T.MotToTensor(), # We already convert the image to a tensor
        T.MotTranslateBoxes(),
    ]

    if not args.augment_data:
        return T.MotCompose(transformations)

    if split == 'train':
        transformations.extend([
            RDT.MotRandomReverseAndHFlip(p=args.aug_random_reverse_and_hflip_prob),
            RDT.MotRandomNoise(p=args.aug_random_noise_prob, mean=0.25, std=0.01),
        ])

    # Always normalize in the last step.
    assert dataset in STANDARDIZATION, f'Unsupported dataset {dataset}'
    transformations.append(T.MotNormalize(mean=STANDARDIZATION[dataset]['mean'],
                                          std=STANDARDIZATION[dataset]['std']))

    return T.MotCompose(transformations)


def build(split: str, args: Namespace):
    transforms = build_transforms(args.dataset, split, args)

    if split == 'train':
        return RadarTrack(args.data_path, args.dataset, split, transforms,
                          sample_mode=args.sample_mode,
                          sample_size=args.sample_size,
                          sample_stride=args.sample_stride)

    if split == 'val':
        return RadarTrackSequences(args.data_path, args.dataset, split, transforms)

    raise NotImplementedError(f'Unknown dataset split "{split}"')
