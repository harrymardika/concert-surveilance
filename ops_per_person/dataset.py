
import torch.utils.data as data
from PIL import Image
import os
import numpy as np
import random

class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file, num_segments=12, transform=None, train=True):
        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.transform = transform
        self.train = train
        self._parse_list()

    def _parse_list(self):
        self.video_list = []
        for line in open(self.list_file):
            parts = line.strip().rsplit(" ", 2)  # split from right!
            self.video_list.append(parts)

    def _get_indices(self, n_frames):
        if n_frames >= self.num_segments:
            if self.train:
                # jittered sampling
                segment_size = n_frames / self.num_segments
                offsets = []
                for i in range(self.num_segments):
                    start = int(segment_size * i)
                    end = int(segment_size * (i + 1))
                    offsets.append(random.randint(start, max(start, end - 1)))
                return offsets
            else:
                # deterministic (first & last included)
                return np.linspace(0, n_frames - 1, self.num_segments).astype(int)
        else:
            return [i % n_frames for i in range(self.num_segments)]

    def __getitem__(self, index):
        directory, n_frames, label = self.video_list[index]
        n_frames = int(n_frames)

        offsets = self._get_indices(n_frames)

        images = []
        for offset in offsets:
            frame_idx = offset + 1  # adjust if your dataset is 0-indexed
            p = os.path.join(self.root_path, directory, f"img_{frame_idx:05d}.jpg")

            if not os.path.exists(p):
                # fallback to last frame
                p = os.path.join(self.root_path, directory, f"img_{n_frames:05d}.jpg")

            images.append(Image.open(p).convert('RGB'))

        if self.transform:
            images = self.transform(images)

        return images, int(label)

    def __len__(self):
        return len(self.video_list)
