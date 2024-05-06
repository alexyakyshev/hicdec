import numpy as np
from cooltools.lib.numutils import observed_over_expected, interp_nan


def hic_transform(
    hic_map: np.array,
    framesize: int
):
    submap = hic_map[framesize:2*framesize, framesize:2*framesize]
    not_na_columns_mark = np.logical_not(np.isnan(submap)).sum(axis=0) != 0
    if not_na_columns_mark.sum() / framesize < 0.9:
        return False, hic_map[framesize:2*framesize, framesize:2*framesize]

    item = hic_map
    not_na_mask = np.logical_not(np.all(np.isnan(item), axis=0))
    item, _, _, _ = observed_over_expected(item, mask=not_na_mask)
    item = interp_nan(item)
    item[item == 0.0] = np.quantile(a=item[item != 0], q=0.01)
    item = np.log2(item)
    return True, item[framesize:2*framesize, framesize:2*framesize]
