import numpy as np


def nan_interpolator(obj: np.array):
    nans, x = np.isnan(obj), lambda z: z.nonzero()[0]
    obj[nans] = np.interp(x(nans), x(~nans), obj[~nans])
    return obj
