from .base import Feature
import pandas as pd
from ....utils.common import nan_interpolator


class StripesFeature(Feature):
    def __init__(
        self,
        base_path: str
    ):
        self.name = 'stripes'
        self.path = base_path + '/.stripes.npy'
        self.base_path = base_path

    def load(
        self,
        stripes_file: str
    ):
        self.stripes_file = f'{self.base_path}/{stripes_file}'
        self.stripes_table = pd.read_csv(self.stripes_file, sep='\t', names=['chrom', 'start', 'end', 'cross_score'])
        self.stripes_table['cross_score'] = nan_interpolator(self.stripes_table['cross_score'].to_numpy())

    def get_feature_by_postition(
        self,
        start: int,
        end: int
    ):
        return self.stripes_table[start:end]['cross_score'].to_numpy()

    def to_dict(self):
        return dict(
            name=self.name,
            path=self.path,
            stripes_file=self.stripes_file
        )
