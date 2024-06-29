import contextlib
import glob
import math
import os
import re
import shutil
import subprocess
import sys
from argparse import ArgumentParser, ArgumentTypeError, Namespace

import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import torch
import joblib
import tqdm
import cv2
from torch.utils.data import DataLoader

# Used to find the datasets.radartrack module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.radartrack import RadarSequence, RadarTrack, get_sequence_paths

torch.multiprocessing.set_sharing_strategy('file_system')


class Dataset(RadarSequence):
    """Represents one sequence recorded on a multi radar system containing the tracks per each frame.
    """
    def __init__(self, sequence_dir: str, tracks_dir: str):
        super().__init__(sequence_dir)

        self._tracks: dict[int, dict[str, torch.Tensor]] = {}
        if tracks_dir:
            track_file = os.path.join(tracks_dir, f'{self.id}.txt')
            if os.path.exists(track_file):
                self._tracks = self.parse_tracks(track_file)

    @property
    def min_rd_value(self) -> int:
        return self._info['min']

    @property
    def max_rd_value(self) -> int:
        return self._info['max']

    @property
    def min_rd_005_quantile(self) -> int:
        return self._info['minQ']

    @property
    def fps(self) -> int:
        return self._info['fps']

    def __getitem__(self, t: int) -> tuple[torch.Tensor, dict[str]]:
        frame = self._frames[t]

        frame_idx = frame['index'].item()
        tracks = self._tracks.get(frame_idx, {})
        frame['track_ids'] = tracks.get('ids', torch.as_tensor([], dtype=self.dtype_id))
        frame['track_boxes'] = tracks.get('boxes', torch.as_tensor([], dtype=self.dtype_box).reshape(-1, 4))

        # Format HWC
        image = cv2.imread(frame['path'])
        image = image[..., 0]

        return image, frame

    @staticmethod
    def collate(batch):
        return tuple(batch[0])


# https://stackoverflow.com/a/58936697
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def rup(x):
    return int(math.ceil(x / 10.0)) * 10

def rdown(x):
    return int(math.floor(x / 10.0)) * 10

def get_rectangles(x: int, y: int, w: int, h: int, xmax: int, **kwargs):
    rectangles: list[patches.Rectangle] = []

    if x < 0:
        rectangles.append(patches.Rectangle((0, y), w + x, h, **kwargs))
        rectangles.append(patches.Rectangle((xmax + x, y), -x, h, **kwargs))
    elif (x + w) >= xmax:
        rectangles.append(patches.Rectangle((x, y), xmax - x, h, **kwargs))
        rectangles.append(patches.Rectangle((0, y), (x + w) - xmax, h, **kwargs))
    else:
        rectangles.append(patches.Rectangle((x, y), w, h, **kwargs))

    return rectangles

def plot(image: np.ndarray, data: dict[str, torch.Tensor], dataset: Dataset, color_cycle: list, output_dir: str, args: Namespace):
    frame = data['index'].item()
    detections = data['detections']
    gt_ids = data['ids']
    gt_boxes = data['boxes']
    track_ids = data['track_ids']
    track_boxes = data['track_boxes']
    xmax = image.shape[1]

    fig = plt.figure()
    fig.set_size_inches(args.fig_width, args.fig_height)
    fig.set_dpi(args.fig_dpi)
    ax = fig.add_subplot()

    # Set font size of title and axis labels
    if args.font_size > 0:
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(args.font_size)

    # Crop range doppler matrix
    if args.max_height:
        image = image[:args.max_height, ...]

    loc = plticker.MultipleLocator(base=10.0)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)

    if dataset.measurement:
        ax.set_title(f'Frame {frame:06d}\n\n{dataset.measurement}/{dataset.id}\n[Radar={dataset.radar}]\n({dataset.name})', pad=20)
    else:
        ax.set_title(f'Frame {frame:06d}\n\n{dataset.id}\n[Radar={dataset.radar}]\n({dataset.name})', pad=20)

    vmin = args.min_rd or rdown(dataset.min_rd_005_quantile)
    vmax = args.max_rd or rup(dataset.max_rd_value)

    img = ax.imshow(image,
                    cmap=args.colormap,
                    vmin=vmin,
                    vmax=vmax)

    ax.set_ylabel('Range')
    ax.set_xlabel('Doppler-Velocity')

    if args.hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    elif args.alt_axis:
        ax.set_xticks(np.linspace(0, image.shape[1], num=3))
        ax.set_xticklabels(['-', 0, '+'])

        ax.set_ylabel('Range', labelpad=4)
        ax.set_yticks([0])
        ax.set_yticklabels([0])
        ax.annotate('', xy=(-0.072, 0), xycoords='axes fraction', xytext=(-0.072, 0.99),
                    arrowprops=dict(arrowstyle='->', color='k'))
    else:
        ticks = np.linspace(0, image.shape[1], num=5)
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks - ticks[len(ticks) // 2])

    if args.add_colorbar:
        fig.colorbar(img, ax=ax)

    if args.plot_det:
        for box in detections:
            x, y, w, h, _ = box.unbind()

            rects = get_rectangles(x, y, w, h, xmax,
                                   linewidth=1,
                                   linestyle='dashed',
                                   edgecolor='w',
                                   facecolor='none')
            for r in rects:
                ax.add_patch(r)

    if args.plot_gt:
        for id, box in zip(gt_ids, gt_boxes):
            id = id.item()
            x, y, w, h = box.unbind()
            color = color_cycle[(id - 1) % len(color_cycle)]

            rects = get_rectangles(x, y, w, h, xmax,
                                   linewidth=2,
                                   edgecolor=color,
                                   facecolor='none')
            for r in rects:
                ax.add_patch(r)

            text = ax.text(x, y,
                           str(id),
                           verticalalignment='bottom',
                           color=color,
                           fontsize=12,
                           weight='bold')

            text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                                   path_effects.Normal()])

    for id, box in zip(track_ids, track_boxes):
        id = id.item()
        x, y, w, h = box.unbind()
        color = 'r'

        rects = get_rectangles(x, y, w, h, xmax,
                               linewidth=1,
                               edgecolor=color,
                               facecolor='none')
        for r in rects:
            ax.add_patch(r)

        text = ax.text(x + w, y,
                       str(id),
                       va='top',
                       ha='right',
                       color=color,
                       fontsize=12,
                       weight='bold')

        text.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                               path_effects.Normal()])

    out_file = os.path.join(output_dir, RadarTrack.FRAME_FILENAME_FORMAT.format(frame=frame))
    fig.savefig(out_file, dpi=300)

    plt.close()


def get_color_cycle():
    # color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # return color_cycle[:2] + color_cycle[4:] # Skip green and red (that will be prediction color)

    return [
        'tab:orange',
        'tab:pink',
        'tab:olive',
        'darkviolet',
        'skyblue',
        'springgreen',
        'lightcoral',
        'wheat',
        'fuchsia',
        'tab:cyan',
        'tab:purple',
    ]


def visualize(dataset: Dataset, output_dir: str, color_cycle: list[str], args: Namespace):
    loader = DataLoader(dataset,
                        batch_size=1,
                        shuffle=False,
                        num_workers=args.workers,
                        collate_fn=dataset.collate)

    # If the frame index does not exist in the sequence the progress bar will show incomplete state
    if len(args.frames) == 0:
        total = len(loader)
    else:
        total = len(args.frames)

    with tqdm_joblib(tqdm.tqdm(desc='Plotting frames', unit=' Frames', total=total, leave=False)) as pbar:
        with joblib.Parallel(n_jobs=args.workers) as parallel:
            parallel(joblib.delayed(plot)(image, data, dataset, color_cycle, output_dir, args) for image, data in loader
                     if len(args.frames) == 0 or data['index'].item() in args.frames) # When user passes specific frames to plot, only plot these

    if args.create_video:
        fps = args.fps or dataset.fps

        subprocess.run(['ffmpeg',
                        '-r', str(fps),
                        '-i', os.path.join(output_dir, '%06d.png'),
                        '-vcodec', 'mpeg4',
                        '-b:v', '250k',
                        '-y', os.path.join(output_dir, f'{dataset.id}.mp4')])

        if args.remove_frames:
            for frame in glob.glob(os.path.join(output_dir, '*.png')):
                os.unlink(frame)


def clear_directory(folder: str):
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=False)
        except Exception as e:
            print(f'Failed to delete {path}. Reason: {e}')


def parse_frames_nums(argument: str):
    match = re.match(r'(\d+)(?:-(\d+))?$', argument)
    if not match:
        raise ArgumentTypeError(f'{argument} is not a range of number. Expected forms like \"0-5\" or \"2\".')

    start = int(match.group(1), 10)
    if start < 0:
        raise ArgumentTypeError(f'First number must be 0 or positive.')

    end = int(match.group(2) or str(start), 10)
    return set(range(start, end + 1))


def output_base_dir(split: str, args: Namespace) -> str:
    return os.path.join(args.output_dir, f'{args.dataset}-{split}', args.output_name)


def main(args: Namespace):
    color_cycle = get_color_cycle()

    # Convert sequence numbers to full sequence name
    args.sequences = [f'{args.dataset}{num:04d}' for num in args.sequences]

    sequences: list[tuple[str, str, str]] = []
    for split in args.splits:
        seq_paths = get_sequence_paths(args.data_path, args.dataset, split)
        seq_paths = map(lambda s: (split, *s), seq_paths)

        if args.sequences:
            sequences.extend(filter(lambda s: s[1] in args.sequences, seq_paths))
        else:
            sequences.extend(seq_paths)

    if not sequences:
        print('Nothing to visualize. Maybe forgot to specify correct dataset split?')
        return

    pbar = tqdm.tqdm(sequences, desc='Visualizing', unit='Sequence')
    for split, id, sequence in pbar:
        dataset = Dataset(sequence, args.tracks_path)

        pbar.set_description(f'Visualizing {id}')
        pbar.set_postfix({'frames': len(dataset)})

        output_dir = os.path.join(output_base_dir(split, args), id)
        os.makedirs(output_dir, exist_ok=True)

        if args.clear:
            clear_directory(output_dir)

        visualize(dataset, output_dir, color_cycle, args=args)

if __name__ == '__main__':
    parser = ArgumentParser(description=('Visualizes Range-Doppler maps by plotting heatmaps and converting the individual frames to a video. '
                                         'Additionally to the raw Range-Doppler data, CFAR detections, '
                                         'ground truth bounding boxes and bounding box predictions can be inserted.'))

    # Input
    inputgrp = parser.add_argument_group('Input')
    inputgrp.add_argument('--data', '-d', dest='data_path', type=str, required=True, help='Path to the directory containing the splits of the radar sequences.')
    inputgrp.add_argument('--dataset', '-ds', dest='dataset', type=str, default=None, help='Name of the dataset. Default is directory name of data path.')
    inputgrp.add_argument('--splits', dest='splits', nargs='+', type=str, default=['val'], help='Dataset splits to consider. (Default: %(default)s)')
    inputgrp.add_argument('--sequences', '-s', dest='sequences', nargs='+', type=int, default=[], help='Specific sequences in the split to visualize. (Default is that all sequences in the data directory of given splits will be visualized.)')
    inputgrp.add_argument('--frames', '-f', dest='frame_sets', nargs='+', type=parse_frames_nums, default=[], metavar='RANGE', help='Only visualize given frame ranges (e.g. 0-100) of given sequences. (Default is that all frames will be visualized.)')

    # Track data
    trackgrp = parser.add_argument_group('Track', description='Track data settings to include in Range-Doppler maps.')
    trackgrp.add_argument('--tracks', '-t', dest='tracks_path', type=str, help='Path to the directory containing the tracker evaluation results. This will plot the track predictions.')
    trackgrp.add_argument('--no-detections', '-nd', dest='plot_det', action='store_false', help='Do not show CFAR detection bounding boxes in the plot.')
    trackgrp.add_argument('--no-gt', '-ng', dest='plot_gt', action='store_false', help='Do not show the ground truth bounding boxes and id in the plot.')

    # Export options
    exportgrp = parser.add_argument_group('Export', description='Export settings for frames and video.')
    exportgrp.add_argument('--output', '-o', dest='output_dir', type=str, required=True, help='Path to the directory where the results should be saved.')
    exportgrp.add_argument('--name', '-n', dest='output_name', type=str, default='', help='Name of the directory where the results should be saved.')
    exportgrp.add_argument('--clear', '-c', dest='clear', action='store_true', help='Clears the output directory before writing to it.')
    exportgrp.add_argument('--video', '-vid', dest='create_video', action='store_true', help='Create a video from created plots. Does nothing if --frames is set.')
    exportgrp.add_argument('--remove-frames', '-rmf', dest='remove_frames', action='store_true', help='Deletes the frames after converting to video. Does nothing if --video flag was not set.')
    exportgrp.add_argument('--fps', '-r', dest='fps', type=int, default=None, help='Framerate of the resulting video. (Default is framerate specified by sequence.)')

    # Style options
    stylegrp = parser.add_argument_group('Style', description='Style settings for the Range-Doppler maps.')
    stylegrp.add_argument('--colormap', dest='colormap', type=str, default='viridis', help='Colormap of the plotted Range-Doppler maps. (Default: %(default)s)')
    stylegrp.add_argument('--min-rd', dest='min_rd', type=int, default=None, help='Overwrite minimum value of Range-Doppler map range used for the colormap.')
    stylegrp.add_argument('--max-rd', dest='max_rd', type=int, default=None, help='Overwrite maximum value of Range-Doppler map range used for the colormap.')
    stylegrp.add_argument('--font-size', dest='font_size', type=int, default=-1, help='Font size of text.')
    stylegrp.add_argument('--hide-colorbar', dest='add_colorbar', action='store_false', help='Add color bar to the left side of the plot.')
    stylegrp.add_argument('--max-height', dest='max_height', type=int, default=None, help='Maximum height (range) of the Range-Doppler map. This is used to make the maps smaller in size.')
    stylegrp.add_argument('--fig-width', dest='fig_width', type=int, default=5, help='Width in inches of the figure. (Default: %(default)s)')
    stylegrp.add_argument('--fig-height', dest='fig_height', type=int, default=8, help='Height in inches of the figure. (Default: %(default)s)')
    stylegrp.add_argument('--fig-dpi', dest='fig_dpi', type=int, default=300, help='DPI of the figure. (Default: %(default)s)')
    axgrp = stylegrp.add_mutually_exclusive_group()
    axgrp.add_argument('--hide-ticks', dest='hide_ticks', action='store_true', help='Do not show the ticks and number at the axis of the plot.')
    axgrp.add_argument('--alt-axis', dest='alt_axis', action='store_true', help='Plot an alternative representation of the range and velocity axis.')

    # Misc
    miscgrp = parser.add_argument_group('Misc', description='General settings.')
    miscgrp.add_argument('--workers', dest='workers', type=int, default=4, help='Number of workers for parallel data fetching and processing. (Default: 4)')
    miscgrp.add_argument('--version', '-v', action='version', version='1.0.0')

    args = parser.parse_args()

    args.frames = set()
    args.frames.update(*args.frame_sets)

    if args.frames:
        if args.create_video:
            print('Creating video with frames out of order is unsupported.')
        args.create_video = False

    if not args.create_video:
        if args.remove_frames:
            print('Removing frames without creating a video is unsupported.')
        args.remove_frames = False

    if args.dataset is None:
        args.dataset = os.path.basename(args.data_path).lower()

    main(args)

# Example 1: Visualize every sequence in the data folder belonging to train or val dataset
# python visualize.py --clear -d ../data/dataset/RDTrack -o ../data/vid --splits val train --video -rmf

# Example 2: Visualize sequence with number 12 and 22 with the matplotlib jet colormap
# python visualize.py --clear -d ../data/dataset/RDTrack -o ../data/vid -c jet -s 12 22 --video -rmf

# Example 3: Visualize every sequence in the data folder belonging to val dataset and add evaluated tracks
# python visualize.py --clear -d ../data/dataset/RDTrack -o ../data/vid -n run4 -t ../data/trackers/rdtrack-val/run4 -vid

# Example 4: Visualize GT for frames 12 and 13 from sequence 12 without creating a video and alternative styling
# python visualize.py --clear -d ../data/dataset/RDTrack -o ../data/vid -n run16_e49_ms50_v2 -s 12 -f 12-13 --alt-axis --hide-colorbar --font-size 12 --min-rd 80 --max-rd 115 --max-height 192

# Example 5: Visualize every sequence in the data folder belonging to val dataset and add evaluated tracks
# python visualize.py --clear -d ../data/dataset/RDTrack -o ../data/vid -n run21_e49_ms20 -t ../data/trackers/rdtrack-val/run21_e49_ms20 -vid --fps 10
