import os
import sys
from argparse import ArgumentParser, Namespace

# Used to find the datasets.radartrack module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms as TT
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datasets.radartrack import RadarTrack, get_image, get_sequence_paths, parse_sequence_info


class RadarDataset(Dataset):
    def __init__(self, data_dir: str, dataset: str, splits: list[str], transform=None):
        super().__init__()

        self.data_dir = data_dir
        self.dataset = dataset
        self.splits = splits
        self.transform = transform

        self.images: list[str] = []
        for split in splits:
            for _, seq_dir in get_sequence_paths(data_dir, dataset, split):
                seq_info = parse_sequence_info(seq_dir)
                self.images.extend(self.get_image_paths(seq_dir, seq_info['image_dir'], seq_info['len']))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        path = self.images[index]
        image = get_image(path)

        if self.transform:
            image = self.transform(image)

        return image

    def get_image_paths(self, seq_dir: str, img_dir: str, seq_len: int):
        img_dir = os.path.join(seq_dir, img_dir)
        return [os.path.join(img_dir, RadarTrack.FRAME_FILENAME_FORMAT.format(frame=frame_idx)) for frame_idx in range(1, seq_len + 1)]


def main(args: Namespace):
    transform = None
    if args.mean and args.std:
        print(f'Normalizing images with mean={args.mean} and std={args.std}')

        transform = TT.Compose([
            TT.Normalize(mean=args.mean, std=args.std)
        ])

    dataset = RadarDataset(args.data_dir, args.dataset, args.splits, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    min = torch.tensor(1)
    max = torch.tensor(0)

    mean = torch.zeros(3)
    with tqdm(loader, unit=' Images', desc='Calculating mean', total=len(loader.dataset)) as pbar:
        for image in loader:
            image = image.view(image.shape[0], image.shape[1], -1) # type: torch.Tensor
            mean += image.mean(2).sum(0)

            image_min = image.min()
            if image_min < min:
                min = image_min

            image_max = image.max()
            if image_max > max:
                max = image_max

            # Update batch
            pbar.update(image.shape[0])

    mean /= len(loader.dataset)

    var = torch.zeros(3)
    pixel_count: int = 0
    with tqdm(loader, unit=' Images', desc='Calculating std dev', total=len(loader.dataset)) as pbar:
        for image in loader:
            image = image.view(image.shape[0], image.shape[1], -1) # type: torch.Tensor
            var += ((image - mean.unsqueeze(1)) ** 2).sum([0, 2])

            pixel_count += image.nelement()

            # Update batch
            pbar.update(image.shape[0])

    std = torch.sqrt(var / pixel_count)

    print('Min', min * 255)
    print('Max', max * 255)
    print('Mean', mean)
    print('Std', std)

if __name__ == '__main__':
    parser = ArgumentParser(description='Radar dataset image mean and std dev calculation script')
    parser.add_argument('--version', '-v', action='version', version='1.0.0')
    parser.add_argument('--data', '-d', dest='data_dir', type=str, required=True, help='Path to the directory containing the splits of the radar sequences.')
    parser.add_argument('--name', '-n', dest='dataset', type=str, default='RDTrack', help='Dataset name.')
    parser.add_argument('--splits', dest='splits', nargs='+', type=str, default=['train', 'val'], help='Dataset splits to consider.')
    parser.add_argument('--workers', dest='workers', type=int, default=4, help='Number of workers the dataloader can use to load images.')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=32, help='Number images in one batch to process per step.')
    parser.add_argument('--std', dest='std', nargs=3, type=float, default=[], help='Calculated std dev to check.')
    parser.add_argument('--mean', dest='mean', nargs=3, type=float, default=[], help='Calculated mean to check.')

    args = parser.parse_args()

    main(args)
