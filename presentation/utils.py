import numpy as np


def block(global_flux, lo, hi):
    copy = global_flux.copy()
    median = np.median(global_flux)
    indices = np.arange(len(global_flux))
    mask = np.logical_and(indices >= lo, indices <= hi)
    copy[mask] = median
    return copy


