import numpy as np
import hashlib
import cooler
from typing import List
import os
import pickle
import json
from .features.base import Feature
from .features.norms import NormTypes
from .utils import hic_transform


class DiscRow():
    idx: int
    start_position: int
    end_position: int

    def __init__(self, idx, start_position, end_position, *features):
        self.idx = int(idx)
        self.start_position = start_position
        self.end_position = end_position
        self.map = None
        for el in features:
            self.__setattr__(el.name, None)

    def set_feature(self, name, val):
        self.__setattr__(name, val)

    def set_map(self, map):
        self.map = hashlib.sha256(map).hexdigest()

    def to_dict(self):
        return self.__dict__.copy()

    @classmethod
    def from_json(self, json_name):
        with open(json_name, 'r') as inf:
            files = json.load(inf)
        blank = dict()
        for key, val in files.items():
            blank[int(key)] = object.__new__(DiscRow)
            blank[int(key)].__dict__.update(val)
        return blank

    def get_row(
        self,
        idx: int,
        map_array: np.memmap,
        norm_type: NormTypes,
        *features
    ):
        out = dict()
        if self.idx != idx:
            raise ValueError('Wrong idx')
        out['idx'] = idx
        map = np.ascontiguousarray(map_array[idx, :, :])
        if self.map != hashlib.sha256(map).hexdigest():
            raise ValueError('Something wrong with map array')
        out['map'] = map
        out['start_position'] = self.start_position
        out['end_position'] = self.end_position

        for feature in features:
            feature_name = feature.name
            if feature_name not in self.__dict__:
                raise ValueError(f'No such feature {feature_name}')
            feature_val = np.ascontiguousarray(feature.get_feature_by_index(idx))
            if hashlib.sha256(feature_val).hexdigest() != self.__getattribute__(feature_name):
                raise ValueError(f'Wrong feature {feature_name}')
            feature_val = feature.norm(feature_val, norm_type)
            out[feature_name] = feature_val
        return out


class DiscStorage():

    def __init__(
        self,
        storage_path: str,
        cooler_name: str,
        force_rewrite: bool = False
    ):

        self.storage_path = storage_path
        self.cooler_name = cooler_name
        self.storage_path = self.storage_path.rstrip('/')
        if not os.path.exists(self.storage_path):
            raise FileNotFoundError('Storage directory did not exists')

        if not os.path.exists(f'{self.storage_path}/tmp'):
            os.mkdir(f'{self.storage_path}/tmp')

        if not os.path.exists(f'{self.storage_path}/{self.cooler_name}'):
            raise FileNotFoundError('Cooler file did not exists')

        if force_rewrite:
            if os.path.exists(f'{self.storage_path}/meta.json'):
                os.remove(f'{self.storage_path}/meta.json')
            if os.path.exists(f'{self.storage_path}/.maps.npy'):
                os.remove(f'{self.storage_path}/.maps.npy')
            if os.path.exists(f'{self.storage_path}/.features.pkl'):
                os.remove(f'{self.storage_path}/.features.pkl')

        if os.path.exists(self.storage_path+'/meta.json'):
            with open(self.storage_path+'/meta.json', 'r') as inf:
                self._meta = json.load(inf)
        else:
            self._meta = dict()

    def generate_dataset(
        self,
        resolution: int,  # Разрешение HiC-карты
        window_size: int,  # Размер окна (в бинах)
        features: List[Feature],
        force_rewrite: bool = False
    ):
        self.features = features
        self._meta['resolution'] = resolution
        self._meta['cooler'] = self.cooler_name
        self._meta['window_size'] = window_size
        self._meta['maps'] = '.maps.npy'

        self.clr = cooler.Cooler(f'{self.storage_path}/{self.cooler_name}::resolutions/{resolution}')

        input_shape = self.clr.shape[0]
        raw_length = int(input_shape // window_size * 2)

        self.map_array = np.memmap(f'{self.storage_path}/{self._meta["maps"]}', mode='w+', shape=(raw_length, window_size, window_size), dtype=np.float64)
        for ft in self.features:
            ft.create_memmap(window_size, raw_length, force_rewrite)

        length = 0
        self._index = dict()
        print(f'Start processing {self.cooler_name}, determine {raw_length} windows')

        for idx in range(2, raw_length-3):  # пропускаем первую и последнюю карты

            left_border = window_size * idx // 2
            right_border = left_border + window_size

            item = self.clr.matrix(balance=True)[left_border - window_size:right_border + window_size, left_border - window_size:right_border + window_size]

            verdict, item = hic_transform(item, window_size)

            if verdict:
                item = np.ascontiguousarray(item)
                row = DiscRow(length, left_border, right_border, *features)
                self.map_array[length, :, :] = item
                row.set_map(item)

                for feature in self.features:
                    val = feature.save_position(left_border, right_border, length)
                    row.set_feature(feature.name, val)
                self._index[length] = row
                length += 1

            if length % 100 == 0:
                print(f"Loaded {length} maps")
        self._meta['length'] = length
        self.map_array.flush()
        self._meta['memmap_shape'] = self.map_array.shape
        for feature in self.features:
            feature.save_memmap()
            feature.generate_meta(self._meta['length'])
        self._meta['features'] = [
            ft.to_dict() for ft in self.features
        ]

        with open(f'{self.storage_path}/index.json', 'w') as outf:
            json.dump({key: obj.to_dict() for key, obj in self._index.items()}, outf)

        with open(f'{self.storage_path}/meta.json', 'w') as outf:
            json.dump(self._meta, outf)

        with open(f'{self.storage_path}/features.pkl', 'wb') as outf:
            pickle.dump(self.features, outf)

    def load_index(self):
        self.clr = cooler.Cooler(f'{self.storage_path}/{self.cooler_name}::resolutions/{self._meta["resolution"]}')
        self.map_array = np.memmap(f"{self.storage_path}/.maps.npy", mode='r', shape=tuple(self._meta['memmap_shape']), dtype=np.float64)
        with open(f'{self.storage_path}/features.pkl', 'rb') as inf:
            self.features = pickle.load(inf)
        self._index = DiscRow.from_json(f'{self.storage_path}/index.json')

    def __len__(self):
        return self._meta['length']

    def __getitem__(self, idx):
        norm_type = NormTypes.NONE
        if isinstance(idx, tuple):
            idx, norm_type = idx
        return self._index[idx].get_row(idx, self.map_array, norm_type, *self.features)

    def fast_get(self, idx):
        return self._index[idx].to_dict()
