import os
from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class Tracker(ABC):
    save_format = '{frame},{id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1\n'

    @abstractmethod
    def track_frame(self, frame: int, data: tuple[torch.Tensor, dict[str]]) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

    def track(self, loader: DataLoader, seq_id: str, output_dir: str):
        lines: list[str] = []
        n_tracks: int = 0
        seen_id_map: dict[int, int] = {}

        for i, data in enumerate(tqdm(loader, desc=seq_id, unit=' Frames')):
            frame_idx = i + 1

            bbox_xywh, identities = self.track_frame(frame_idx, data)

            for xywh, track_id in zip(bbox_xywh, identities):
                track_id = track_id.item()
                if track_id < 0 or track_id is None:
                    continue

                # Ids may be all over the place, this will replace them with a continuous one (1, 2, ...)
                continuous_id = seen_id_map.get(track_id, -1)
                if continuous_id < 0:
                    n_tracks += 1
                    continuous_id = n_tracks
                    seen_id_map[track_id] = continuous_id

                x, y, w, h = xywh.unbind()
                lines.append(self.save_format.format(frame=frame_idx, id=continuous_id, x=x, y=y, w=w, h=h))

        # Save results
        with open(os.path.join(output_dir, f'{seq_id}.txt'), 'w') as f:
            f.writelines(lines)

    def reset(self):
        pass
