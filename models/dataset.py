import torch
import cooler
from typing import List
from .features.base import Feature
from .features.norms import NormTypes


def fft2d(x):
    square = max(x.shape)
    x = x.reshape(square, square)
    x = torch.fft.fft2(x)
    x = torch.fft.fftshift(x)
    x = x.reshape(1, square, square)
    return x


def ifft2d(x):
    square = max(x.shape)
    x = x.reshape(square, square)
    x = torch.fft.ifftshift(x)
    x = torch.fft.ifft2(x)
    x = x.reshape(1, square, square)
    return x


def fft1d(x):
    line = max(x.shape)
    x = x.reshape(line)
    x = torch.fft.fft(x)
    x = torch.fft.fftshift(x)
    x = x.reshape(line, 1)
    return x


def ifft1d(x):
    line = max(x.shape)
    x = x.reshape(line)
    x = torch.fft.ifftshift(x)
    x = torch.fft.ifft(x)
    x = x.reshape(line, 1)
    return x


class HiCMapDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cooler_entity: cooler.Cooler,
        datastorage: str,
        window_size: int,
        split_fnc: callable,
        features_list: List[Feature],
        norm_type: NormTypes,
        is_fourier: bool
    ):
        self.features = features_list
        self.norm_type = norm_type
        self.is_fourier = is_fourier
        self.clr = cooler_entity
        self.datastorage = datastorage
        self.bin = cooler_entity.binsize
        self._slice_size = window_size
        self.new_index = {}
        internal_idx = 0
        for el in range(len(self.datastorage) - 1):
            if split_fnc(self.datastorage[el]):
                self.new_index[internal_idx] = el
                internal_idx += 1
        self.length = len(self.new_index)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        from_storage = self.datastorage[self.new_index[idx]]
        item = from_storage['map']
        item = torch.from_numpy(item).reshape(1, self._slice_size, self._slice_size).float()
        features = [from_storage[el.name] for el in self.features]
        features = [torch.from_numpy(obj).reshape((1, self._slice_size)).float() for obj in features]
        if self.is_fourier:
            features = [fft1d(obj) for obj in features]
            item = fft2d(item)
        return item, *features

    def get_coordinates(self, idx):
        from_datastorage = self.datastorage[self.new_index[idx]]
        start_coord, end_coord = from_datastorage['start_position'], from_datastorage['end_position']
        return start_coord, end_coord

    def get_description(self, idx):
        start_coord, end_coord = self.get_coordinates(idx)
        bins = self.clr.bins()
        start_desc = bins[start_coord][['chrom', 'start']].to_dict(orient='records')[0]
        end_desc = bins[end_coord][['chrom', 'end']].to_dict(orient='records')[0]
        return start_desc, end_desc
