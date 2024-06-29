from argparse import Namespace
from logging import Logger
import glob
import os

import sacred
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import util.misc as utils
from eval_trackeval import run_trackeval
from tools.eval_to_markdown import to_markdown
from datasets.radartrack import RadarSequence, build_transforms, get_image, get_sequence_paths, DATASETS
from experiment import new_experiment
from trackers.tracker_base import Tracker
from trackers.radarmotr import RadarMOTRTracker
from trackers.sort import SortTracker
from trackers.kf import KFTracker

# Sacred config
ex = new_experiment('Tracker-Eval',
                    training=False,
                    save_gpu_info=False,
                    save_cpu_info=False,
                    save_git_info=False)
ex.add_config(os.path.join('configs', 'eval.yaml'))
ex.add_named_config('radarmotr', os.path.join('configs', 'trackers', 'radarmotr.yaml'))
ex.add_named_config('resnet18', os.path.join('configs', 'resnet18.yaml'))
ex.add_named_config('debug', os.path.join('configs', 'eval_debug.yaml'))
ex.add_named_config('sort', os.path.join('configs', 'trackers', 'sort.yaml'))
ex.add_named_config('kf', os.path.join('configs', 'trackers', 'kf.yaml'))


class RadarDataset(RadarSequence):
    """Represents one sequence recorded on a multi radar system containing the tracks per each frame.
    """
    def __init__(self, sequence_dir: str, transforms):
        super().__init__(sequence_dir)
        self.transforms = transforms

    def __getitem__(self, t: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        frame = self._frames[t]

        image = get_image(frame['path'])
        image_size = image.shape[1:] # Skip channels

        targets: dict[str, torch.Tensor] = {}
        targets['size'] = torch.as_tensor(image_size)
        targets['orig_size'] = torch.as_tensor(image_size)

        # BB format: cxcywh
        targets['proposals'] = frame['detections']

        if self.transforms:
            images, targets = self.transforms([image], [targets])

        return images[0], targets[0]

    @staticmethod
    def collate(batch):
        return tuple(batch[0])


def build_tracker(tracker: str, **kwargs) -> Tracker:
    if tracker == 'RadarMOTR':
        model = RadarMOTRTracker(**kwargs)
    elif tracker == 'SORT':
        model = SortTracker(**kwargs)
    elif tracker == 'KF':
        model = KFTracker(**kwargs)
    elif tracker is None:
        raise ValueError('Please select the tracker to evaluate by passing: "RadarMOTR" or "SORT".')
    else:
        raise NotImplementedError(f'Tracker "{tracker}" not supported.')

    return model


@ex.capture
def eval(args: Namespace, _log: Logger):
    if args.dataset not in DATASETS:
        raise NotImplementedError(f'Dataset {args.dataset} not supported!')

    if not args.output_dir:
        raise FileNotFoundError('No output directory specified!')

    tracker = build_tracker(args.tracker, args=args)
    _log.info(f'Using tracker: {args.tracker}')

    _log.info(f'Evaluating on dataset {args.dataset} in \"{args.data_path}\".')

    sequences = get_sequence_paths(args.data_path, args.dataset, args.split)

    output_dir = os.path.join(args.output_dir, f'{args.dataset}-{args.split}', args.tracker_dirname)
    os.makedirs(output_dir, exist_ok=True)
    _log.info(f'Output directory: \"{output_dir}\"')

    # Remove all previous track results
    for file in glob.iglob(f'{args.dataset}{"[0-9]" * 4}.txt', root_dir=output_dir):
        os.unlink(os.path.join(output_dir, file))

    _log.info(f'Starting tracking.')

    transforms = build_transforms(args.dataset, args.split, args)
    for _, sequence in tqdm(sequences, desc='Evaluation', unit='Sequence'):
        dataset = RadarDataset(sequence, transforms)
        loader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.workers,
                            pin_memory=True,
                            collate_fn=dataset.collate)

        tracker.reset()
        tracker.track(loader, dataset.id, output_dir)

    _log.info(f'Starting evaluation.')

    run_trackeval(output_dir, args)

    print()
    to_markdown(output_dir)
    print()


@ex.main
def main(_config, _run):
    sacred.commands.print_config(_run)
    args = utils.nested_dict_to_namespace(_config)
    eval(args)

if __name__ == '__main__':
    ex.run_commandline()
