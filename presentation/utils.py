import numpy as np
from clean_utils.normalization import norm_features
from preprocess.get_more_feathres import get_more_features
import pandas as pd
from config import *
from preprocess.kepler_io import *


__global_flux, __local_flux, __norm_feature = None, None, None


def block(global_flux, lo, hi):
    copy = global_flux.copy()
    median = np.median(global_flux)
    indices = np.arange(len(global_flux))
    mask = np.logical_and(indices >= lo, indices <= hi)
    copy[mask] = median
    return copy


def auto_block(model, kepid, block_size=50):
    if __norm_feature is None:
        global __feature
        # generate test_feature
        all_features = get_more_features(dr24=True)
        __norm_features = norm_features(all_features.values)
        df = pd.read_csv(csv_folder, csv_name_drop_unk)
        idx = df[df['kepid'] == int(kepid)].index[0]
        test_feature = __norm_features[idx]

    if __global_flux is None:
        time, flux = get_time_flux_by_ID(keipd)


    begin = 0
    end = len(global_flux)
    lo = begin
    hi = lo + block_size

    while hi <= end:
        flux = block(global_flux, lo, hi)
        pc_prob = model.predict([global_flux, local_flux, test_feature])
