from config import *
import os
import pandas as pd
import numpy as np
from clean_utils.normalization import norm_kepid
import warnings

# warnings.filterwarnings('error')


def get_more_features(columns=None, kepid=None):
    if columns is None:
        columns = ['tce_period', 'tce_impact',
                   'tce_duration', 'tce_depth',
                   'tce_ror', 'tce_num_transits',
                   'tce_model_snr', 'tce_model_chisq', 'tce_robstat',
                   'tce_prad', 'tce_sradius']

    fname = os.path.join(csv_folder, csv_name_drop_unk)
    df24 = pd.read_csv(fname, comment='#')
    df24['norm_kepid'] = df24['kepid'].apply(norm_kepid)
    if kepid is not None:
        df24 = df24[df24['norm_kepid'] == norm_kepid(kepid)]

    df24['int_label'] = df24['av_training_set'].apply(
        lambda x: 1 if x == 'PC' else 0)
    df24.sort_values(by=['int_label', 'norm_kepid', 'tce_plnt_num'],
                     ascending=[False, True, True],
                     inplace=True)
    return df24[columns]


def write_more_features(columns=None):
    """
    :param columns: what features to write in dr24 file
        default: ['tce_impact', 'tce_duration', 'tce_depth']
    :return: None

    (feature_file, label_file) = (f1.txt, l1.txt)
    """
    feature_file = os.path.join(train_root_dir, 'features', 'f1.txt')
    label_file = os.path.join(train_root_dir, 'features', 'l1.txt')

    # register the two files to the GlobalVars

    if not os.path.exists(os.path.dirname(feature_file)):
        os.makedirs(os.path.dirname(feature_file))

    features = get_more_features(columns)
    labels = get_more_features('av_training_set')
    labels = list(map(lambda label: 1 if label == 'PC' else 0, labels))
    labels = np.array(labels).astype(np.int)
    # print(features.values)
    values = _normalize(features.values)
    np.savetxt(feature_file, values, fmt="%.6f")
    np.savetxt(label_file, labels)


def _normalize(values):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    try:
        values = (values - mean) / std
    except Warning as e:
        print(e, std)
    return values


if __name__ == '__main__':
    # print(get_more_features().head())
    write_more_features()
