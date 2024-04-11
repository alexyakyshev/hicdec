from .base import Feature
from cooler import Cooler
import cooltools
import bioframe
import os
import pandas as pd
from .utils import nan_interpolator


class CompartmentFeature(Feature):
    def __init__(
        self,
        base_path: str
    ):
        self.name = 'compartment'
        self.path = base_path + '/.compartment.npy'
        self.base_path = base_path

    def load(
        self,
        cooler_name: str,
        cooler_obj: str,
        compartment_res: int,
        genome_name: str,
    ):
        self.compartment_res = compartment_res
        clr_compartments = Cooler(f'{self.base_path}/{cooler_name}::resolutions/{compartment_res}')
        bins = clr_compartments.bins()[:]
        if not os.path.isfile(f'{self.base_path}/{genome_name}_gc_cov_{compartment_res//1000}kb.tsv'):
            genome = bioframe.load_fasta(f'{self.base_path}/{genome_name}')
            gc_cov = bioframe.frac_gc(bins[['chrom', 'start', 'end']], genome)
            gc_cov.to_csv(f'{self.base_path}/{genome_name}_gc_cov_{compartment_res//1000}kb.tsv', index=False, sep='\t')
        else:
            gc_cov = pd.read_csv(f'{self.base_path}/{genome_name}_gc_cov_{compartment_res//1000}kb.tsv', sep='\t')

        if not os.path.isfile(f'{self.base_path}/tmp/{genome_name}_eigvec_{compartment_res//1000}kb.tsv')\
           or not os.path.isfile(f'{self.base_path}/tmp/{genome_name}_eigval_{compartment_res//1000}kb.tsv'):
            view_df = pd.DataFrame({'chrom': clr_compartments.chromnames,
                                    'start': 0,
                                    'end': clr_compartments.chromsizes.values,
                                    'name': clr_compartments.chromnames}
                                   )
            cis_eigs = cooltools.eigs_cis(
                clr_compartments,
                gc_cov,
                view_df=view_df,
                n_eigs=3,
                )
            cis_eigs[0].to_csv(f'{self.base_path}/tmp/{genome_name}_eigval_{compartment_res//1000}kb.tsv', index=False, sep='\t')
            cis_eigs[1].to_csv(f'{self.base_path}/tmp/{genome_name}_eigvec_{compartment_res//1000}kb.tsv', index=False, sep='\t')
        else:
            cis_eigs = (
                pd.read_csv(f'{self.base_path}/tmp/{genome_name}_eigval_{compartment_res//1000}kb.tsv', sep='\t'),
                pd.read_csv(f'{self.base_path}/tmp/{genome_name}_eigvec_{compartment_res//1000}kb.tsv', sep='\t')
            )

        def compartment_interpolate(compartment_df, map_df):
            compartment_join = compartment_df[['chrom', 'start', 'end', 'E1', 'E2', 'E3']].copy()
            compartment_join['key'] = compartment_join['chrom'].astype('str') + ':' + compartment_join['start'].astype('str')
            compartment_join = compartment_join[['key', 'E1', 'E2', 'E3']]
            map_join = map_df[['chrom', 'start', 'end']].copy()
            map_join['key'] = map_join['chrom'].astype('str') + ':' + map_join['start'].astype('str')
            joined = map_join.join(compartment_join.set_index('key'), on='key')
            joined['E1'] = nan_interpolator(joined['E1'].to_numpy())
            joined['E2'] = nan_interpolator(joined['E2'].to_numpy())
            joined['E3'] = nan_interpolator(joined['E3'].to_numpy())
            return joined
        self.compartment_table = compartment_interpolate(cis_eigs[1], cooler_obj.bins()[:])

    def get_feature_by_postition(
        self,
        start: int,
        end: int
    ):
        return self.compartment_table[start:end]['E1'].to_numpy()

    def to_dict(self):
        return dict(
            super().to_dict(),
            compartment_resolution=self.compartment_res
        )

    def get_feature_by_index(self, id: int):
        return self.memmap[id, :]
