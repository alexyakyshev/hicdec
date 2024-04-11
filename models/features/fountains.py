from .base import Feature
import pandas as pd
from .utils import nan_interpolator
from cooler import Cooler


class FountainsFeature(Feature):
    def __init__(
        self,
        base_path: str
    ):
        self.name = 'fountains'
        self.path = base_path + '/.fountains.npy'
        self.base_path = base_path

    def load(
        self,
        fountains_file: str,
        cooler_entity: Cooler
    ):
        self.fountains_file = f'{self.base_path}/{fountains_file}'
        fountains = pd.read_csv(self.fountains_file, sep='\t', index_col=0)

        def fountains_interpolate(fountains_df, map_df):
            fountains_join = fountains_df[['chrom', 'start', 'end', 'FS']].copy()
            fountains_join['key'] = fountains_join['chrom'].astype('str') + ':' + fountains_join['start'].astype('str')
            fountains_join = fountains_join[['key', 'FS']]
            map_join = map_df[['chrom', 'start', 'end']].copy()
            map_join['key'] = map_join['chrom'].astype('str') + ':' + map_join['start'].astype('str')
            joined = map_join.join(fountains_join.set_index('key'), on='key')
            joined['FS'] = nan_interpolator(joined['FS'].to_numpy())
            return joined

        self.fountains_table = fountains_interpolate(fountains, cooler_entity.bins()[:])

    def get_feature_by_postition(
        self,
        start: int,
        end: int
    ):
        return self.fountains_table[start:end]['FS'].to_numpy()

    def to_dict(self):
        return dict(
            super().to_dict(),
            fountains_file=self.fountains_file
        )
