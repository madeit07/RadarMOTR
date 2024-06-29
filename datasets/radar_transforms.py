import random
import torch
import torchvision.transforms.functional as F


class MotRandomNoise():
    """Randomly adds a gaussian noise over the image.
    """
    def __init__(self, p: float = 0.5, mean: float = 0, std: float = 0.1):
        assert 0 <= p <= 1, 'Probability not between [0;1]'
        self.p = p
        self.mean = mean
        self.std = std

    def __call__(self, images: list[torch.Tensor], targets: dict[str, torch.Tensor]):
        if random.random() >= self.p:
            return images, targets

        noisy_images: list[torch.Tensor] = []
        for image in images:
            gauss = torch.normal(self.mean, self.std, size=image.shape)
            noisy_images.append(image + gauss)

        return noisy_images, targets


class MotRandomReverseAndHFlip():
    """
    Reverses the sample sequences and flips images and boxes horizontally.
    This wil modify the sequence such that the objects appear
    to traverse in the opposite direction of the radar.
    """
    def __init__(self, p: float = 0.5):
        assert 0 <= p <= 1, 'Probability not between [0;1]'
        self.p = p

    def __call__(self, images: list[torch.Tensor], targets: dict[str, torch.Tensor]):
        if random.random() >= self.p:
            return images, targets

        ret_images: list[torch.Tensor] = []
        ret_targets: list[torch.Tensor] = []
        for image, target in reversed(list(zip(images, targets))):
            img, tgt = self.hflip(image, target)
            ret_images.append(img)
            ret_targets.append(tgt)

        return ret_images, ret_targets

    def hflip(self, image: torch.Tensor, target: dict[str, torch.Tensor]):
        flipped_image = F.hflip(image)

        c, h, w = image.shape

        target = target.copy()
        if 'boxes' in target:
            boxes = target['boxes'].clone()
            boxes[:, [0]] = boxes[:, [0]] * -1 + (w - 1)
            target['boxes'] = boxes

        if 'proposals' in target:
            proposals = target['proposals'].clone()
            proposals[:, [0]] = proposals[:, [0]] * -1 + (w - 1)
            target['proposals'] = proposals

        return flipped_image, target
