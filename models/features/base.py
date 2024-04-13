from abc import ABC, abstractmethod
import os
import numpy as np
import hashlib


class Feature(ABC):
    name: str
    path: str

    @abstractmethod
    def get_feature_by_postition(self, start: int, end: int):
        pass

    def to_dict(self):
        return dict(
            name=self.name,
            path=self.path,
            min=self.min,
            max=self.max,
            mean=self.mean,
            std=self.std,
            memmap_shape=self.memmap_shape
        )

    def get_feature_by_index(self, id: int):
        return self.memmap[id, :]

    def create_memmap(self, window_size: int, max_windows: int):
        if os.path.exists(self.path) and os.path.isfile(self.path):
            raise FileExistsError(f'File {self.path} already exist')
        else:
            self.memmap = np.memmap(self.path, mode='w+', dtype=np.float64, shape=(max_windows, window_size))
            self.memmap_shape = (max_windows, window_size)

    def load_memmap(self):
        self.memmap = np.memmap(self.path, shape=self.memmap_shape, mode='r+')

    def save_memmap(self):
        self.memmap.flush()

    def save_position(
        self,
        start: int,
        end: int,
        idx: int
    ):
        pos_arr = self.get_feature_by_postition(start, end)
        self.memmap[idx, :] = pos_arr
        return hashlib.sha256(pos_arr).hexdigest()

    def generate_meta(self, length):
        feature_slice = self.memmap[:length, :]
        self.min = feature_slice.min()
        self.max = feature_slice.max()
        self.mean = feature_slice.mean()
        self.std = feature_slice.std()
