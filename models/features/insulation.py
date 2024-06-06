from .base import Feature
from cooler import Cooler
from cooltools import insulation
import os
import pandas as pd
import numpy as np
from .utils import nan_interpolator


class InsulationFeature(Feature):
    def __init__(
        self,
        base_path: str
    ):
        self.name = 'insulation'
        self.path = base_path + '/.insulatuion.npy'
        self.base_path = base_path

    def load(
        self,
        resolution: int,
        cooler_entity: Cooler,
        insulation_window: int,
        transform: bool = False
    ):
        self.insulation_window = insulation_window
        self.transform = transform
        if not os.path.isfile(f'{self.base_path}/tmp/insulation_track_{resolution//1000}kb.csv'):
            windows = [3*resolution, 5*resolution, 10*resolution, 25*resolution]
            self.insulation_table = insulation(cooler_entity, windows, verbose=True)
            self.insulation_table.to_csv(f'{self.base_path}/tmp/insulation_track_{resolution//1000}kb.csv', index=False, sep='\t')
        else:
            self.insulation_table = pd.read_csv(f'{self.base_path}/tmp/insulation_track_{resolution//1000}kb.csv', sep='\t')

    def get_feature_by_postition(
        self,
        start: int,
        end: int
    ):
        local_track = self.insulation_table.iloc[start:end, :]
        local_track = local_track[self.insulation_window]
        if self.transform:
            local_track = local_track.apply(lambda x: np.float64(int(x)))
        local_track = local_track.to_numpy()
        local_track = nan_interpolator(local_track)
        return local_track

    def to_dict(self):
        return dict(
            super().to_dict(),
            insulation_window=self.insulation_window
        )
